"""Where do the guest registers live between context accesses?

`probe/guestops.py` established that the guest register file is at
`entry_rbp+0x90` and that handlers name their reads and writes to it. It also
measured the problem this probe exists for: **only 8.5 % and 9.3 % of handlers
touch it.** The guest code obviously does more arithmetic than that, so the values
must spend most of their time somewhere else, and the obvious candidate is the host
register file -- which is exactly where §5 of `E-GUEST-STATE-IS-SPLIT.md` found six
guest registers sitting at dispatch boundaries.

That earlier measurement could not finish the job, and it said so: it seeded the
machine registers and asked which ones still *mention* a guest symbol, which
conflates "still holds guest `rcx`" with "holds guest `rcx` plus seven". This probe
asks a question the context model makes answerable.

**Seed the guest context, not the host registers.** Once the context is located, it
can be overwritten with fresh symbols `G_<reg>` -- one per architectural slot -- and
the walk resumed. From then on:

* a host register that comes to hold **exactly** `G_rax` is *caching* guest `rax`,
  and the assignment is read off rather than guessed;
* a host register that comes to hold an **expression over** `G_*` is doing the
  guest's arithmetic, and the expression is the guest operation;
* and a context slot that changes is the context being written back.

Falsifiers, stated before the run:

1. **the cache assignment must be a partial injection** -- one guest register per
   host register, no host register carrying two at once -- and it must be **stable**
   within a function, or it is not an assignment;
2. **composite expressions must appear.** If every `G_*`-carrying host register
   holds a bare symbol and none ever holds an operator, then the arithmetic is not
   reachable from the context and this method has failed. Reported as a failure, not
   as a null result;
3. **the seeding must survive.** If the VM re-initialises the context from the entry
   save block immediately after seeding, every `G_*` disappears and the run says
   nothing; the probe counts how many boundaries still carry a `G_*` so that case is
   visible rather than silent.

Usage:
  python -m probe.guestcache <module> <root_va> [--handlers N] [--steps N] [--show N]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core.expr import W64, Arena, render  # noqa: E402
from core.guestctx import ARCH, FLAGS, GuestContext  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0
GUEST_CTX_FROM_ENTRY_RBP = 0x90
SEED_SITE = -1
MREGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
         "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    handlers, show, steps = 300, 10, 200_000
    for x in sys.argv[1:]:
        if x.startswith("--handlers="):
            handlers = int(x.split("=", 1)[1], 0)
        if x.startswith("--show="):
            show = int(x.split("=", 1)[1], 0)
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, STACK_BASE)
    L = Lifter(img, m, a, vm.va, vm.end)

    gname: dict[int, str] = {}

    def g_syms(v: int) -> list[str]:
        return sorted({gname[n] for n in a.leaves(v) if n in gname})

    base: list[int] = []
    ctx: list[GuestContext] = []
    seeded = [False]
    ctx_written: set[int] = set()
    seed_at: list[int] = []

    snapshots: list[tuple[int, dict[str, int]]] = []
    ctx_rd: collections.Counter = collections.Counter()
    ctx_wr: collections.Counter = collections.Counter()

    orig_load = Machine.load
    orig_store = Machine.store

    def load(self, addr, nbytes, site, image_u8):
        v = orig_load(self, addr, nbytes, site, image_u8)
        if site != SEED_SITE and ctx:
            c = self.a.as_const(addr)
            r = None if c is None else ctx[0].reg_at(c)
            if r is not None:
                ctx_rd[r] += 1
        return v

    def store(self, addr, value, nbytes, site):
        c = self.a.as_const(addr)
        if c is not None and base and ctx and not seeded[0]:
            off = c - base[0]
            r = ctx[0].reg_at(c)
            if r is not None:
                ctx_written.add(off)
        if site != SEED_SITE and ctx:
            r = None if c is None else ctx[0].reg_at(c)
            if r is not None:
                ctx_wr[r] += 1
        return orig_store(self, addr, value, nbytes, site)

    Machine.load = load
    Machine.store = store

    def do_seed(site: int) -> None:
        seeded[0] = True
        seed_at.append(site)
        for i, reg in enumerate(ARCH):
            if reg == "rsp":
                # **`rsp` is deliberately not seeded**, and the first run of this
                # probe is why: with the guest `rsp` symbolic the walk stopped at
                # 7104 instructions with `no_concrete_rsp`, against 28151 with it
                # left alone. The engine needs a concrete machine stack pointer,
                # and the guest `rsp` slot is what the VM derives its own from --
                # so seeding it does not measure the VM, it stops it. The same
                # lesson was learned once already in `probe/guest_effect.py` and
                # recorded there; it applies here for the same reason.
                continue
            s = a.opaque(f"G_{reg}", W64)
            gname[s] = f"G_{reg}"
            orig_store(m, a.const(ctx[0].base + 8 * i, W64), s, 8, SEED_SITE)
        s = a.opaque(f"G_{FLAGS}", W64)
        gname[s] = f"G_{FLAGS}"
        orig_store(m, a.const(ctx[0].addr_of(FLAGS), W64), s, 8, SEED_SITE)

    ip, stop, at, seen, live = root, None, 0, 0, 0
    # A dispatch-boundary snapshot only sees what *survived* to the boundary, and
    # that is why the first run of this probe found bare symbols and almost no
    # arithmetic: an intermediate result lives in a scratch register for a few
    # instructions and is written back to the context before the next dispatch.
    # So the host register file is differenced at **every instruction** instead,
    # and every write whose new value mentions a `G_*` is logged. Cost is fifteen
    # comparisons per instruction and it makes the log complete rather than
    # sampled.
    cur = {r: m.reg(r, 64) for r in MREGS}
    arith: list[tuple[int, int, str, int, int]] = []
    handler = [0]
    try:
        while at < steps and len(snapshots) < handlers:
            kind, payload = L.step(ip)
            at += 1
            prev, cur = cur, {r: m.reg(r, 64) for r in MREGS}
            if seeded[0]:
                for r in MREGS:
                    if cur[r] != prev[r] and g_syms(cur[r]):
                        arith.append((handler[0], ip, r, prev[r], cur[r]))
            if not base:
                b = a.as_const(m.reg("rbp", 64))
                if b is not None:
                    base.append(b)
                    ctx.append(GuestContext(b + GUEST_CTX_FROM_ENTRY_RBP))
            if kind != "next":
                stop = payload
                break
            ip = payload
            if len(L.dispatches) <= seen:
                continue
            seen = len(L.dispatches)
            dest = L.dispatches[-1][1]
            handler[0] = dest
            if ctx and not seeded[0]:
                # Seed once the VM has actually filled the context, so that the
                # seeding is not immediately overwritten by initialisation.
                if len(ctx_written) >= 14:
                    do_seed(dest)
                    cur = {r: m.reg(r, 64) for r in MREGS}
                continue
            if not ctx:
                continue
            snap = {r: m.reg(r, 64) for r in MREGS}
            if any(g_syms(v) for v in snap.values()):
                live += 1
            snapshots.append((dest, snap))
    finally:
        Machine.load = orig_load
        Machine.store = orig_store

    print(f"root {root:#x}   walked {at} instructions   "
          f"stop {stop.kind if stop else 'budget'}")
    print(f"entry frame rbp0: {base[0]:#x}" if base else "entry frame NOT found")
    if ctx:
        print(f"guest context base: {ctx[0].base:#x}   "
              f"slots written before seeding: {len(ctx_written)}")
    print(f"seeded at handler {seed_at[0]:#x}" if seed_at else "NOT seeded")
    print(f"boundaries after seeding: {len(snapshots)}   "
          f"of which carry a G_* somewhere in the host registers: {live}")
    print()

    if not snapshots:
        print("nothing to report")
        return 2

    # ------------------------------------------------------- the cache, read off
    bare: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    composite: list[tuple[int, str, str]] = []
    for dest, snap in snapshots:
        for r, v in snap.items():
            syms = g_syms(v)
            if not syms:
                continue
            if len(syms) == 1 and a.op(v) == "opaque":
                bare[r][syms[0]] += 1
            elif a.as_const(v) is None:
                composite.append((dest, r, render(a, v, limit=22)))

    print("=== host registers carrying a guest symbol (the cache) ===")
    print(f"{'host':<6} {'boundaries':>11}  guest registers seen there")
    print("-" * 70)
    for r in MREGS:
        if r not in bare:
            continue
        tot = sum(bare[r].values())
        print(f"{r:<6} {tot:>11}  {dict(bare[r].most_common())}")
    print()

    # Falsifier 1, stated as a number.
    multi = {r: c for r, c in bare.items() if len(c) > 1}
    print(f"FALSIFIER 1 -- host registers carrying more than one guest register: "
          f"{len(multi)} of {len(bare)}")
    for r, c in sorted(multi.items()):
        print(f"    {r}: {dict(c)}")
    print(f"              host registers carrying a guest register at all: "
          f"{len(bare)}")
    print(f"              guest registers cache-resident, by host register: "
          f"{sorted({k for c in bare.values() for k in c})}")
    print()

    print("=== guest arithmetic: host registers holding an expression over G_* ===")
    print(f"FALSIFIER 2 -- composite expressions observed at boundaries: "
          f"{len(composite)}")
    seen_shape: collections.Counter = collections.Counter()
    for dest, r, txt in composite:
        seen_shape[(dest, r, txt)] += 1
    for (dest, r, txt), n in seen_shape.most_common(show):
        print(f"  x{n:<3} handler {dest:#x}: {r} = {txt[:130]}")
    print()

    # ------------------------------------------------- every instruction, not just
    # the boundaries. This is where the guest computation actually is.
    print("=== every host-register write that mentions a guest symbol ===")
    print(f"such writes: {len(arith)}")
    kinds: collections.Counter = collections.Counter()
    ops: collections.Counter = collections.Counter()
    for dest, site, r, before, after in arith:
        syms = g_syms(after)
        is_bare = len(syms) == 1 and a.op(after) == "opaque"
        kinds["copy of " + syms[0] if is_bare else "computed"] += 1
        if not is_bare:
            ops[(dest, r, render(a, after, limit=14))] += 1
    print(f"  copies of a single guest symbol: "
          f"{sum(n for k, n in kinds.items() if k.startswith('copy'))}")
    print(f"  computed from guest symbols:      "
          f"{sum(n for k, n in kinds.items() if k == 'computed')}")
    print()
    print(f"  distinct computed expressions, by handler and destination register: "
          f"{len(ops)}")
    for (dest, r, txt), n in ops.most_common(show):
        print(f"    x{n:<4} handler {dest:#x}  {r} = {txt[:136]}")
    print()

    print("=== context traffic after seeding ===")
    print(f"context reads:  {sum(ctx_rd.values())}  {dict(ctx_rd.most_common())}")
    print(f"context writes: {sum(ctx_wr.values())}  {dict(ctx_wr.most_common())}")
    print()

    # Falsifier 3, as a number.
    frac = live / max(len(snapshots), 1)
    print(f"FALSIFIER 3 -- fraction of boundaries still carrying a G_* after "
          f"seeding: {frac:.3f}")
    print("    a VM that immediately re-initialises the context from the save "
          "block would give 0.0 and the run would say nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
