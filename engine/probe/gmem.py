"""Where does a handler keep guest state, and what does it compute with it?

Companion to `probe/gops.py`. `gops` asked whether a machine register *changed*
between two dispatch boundaries; it answered, and the answer is that most of what
moves is the VM's own dispatch scratch (`rax`, `rcx`, `rdx`, `r8`-`r11` are
clobbered on nearly every transition). The guest effect is therefore not in the
register file at a boundary -- so it has to be in **memory**, and this probe is
about memory.

The measurement is a census of every memory operation that touches a guest
symbol:

* symbols `m_<reg>` are seeded into the machine register file before the entry
  sequence, so `m_rcx` *is* the guest `rcx` at entry;
* once the prologue has written all sixteen context slots, the context image is
  overwritten with a second family `i_<reg>`. A load returning `i_rcx` therefore
  proves the context is a **live read source**, which the previous round could not
  tell apart from register residency, because both came back as the same symbol.

**The decisive classification is by address.** A structure that holds guest state
assigns each address one role: over a run, a given slot sees many *reads* but the
values stored there belong to one guest register. A scratch cell sees many
different guest registers written to the same address. So for every address that
guest values reach, the probe counts the distinct symbols that pass through it,
and the shape of that distribution separates the live context from the dispatch's
scratch area and the emulated stack.

Falsifiers, stated before the run:

1. **self-contamination.** The probe's own snapshot reads must not appear in the
   census. They are issued with `site = -1` and filtered on it. The previous run
   counted 4128 of its own reads as VM behaviour -- 96 % of its "image reads" --
   and every conclusion drawn from that number would have been about the probe.
2. a *context slot* must receive **one** guest register, not many; a cell that
   receives many is scratch by definition and is reported as such;
3. the census is printed with the raw address as well as the offset from the
   entry frame, so an offset that is only meaningful relative to some other base
   is visible rather than assumed.

Usage:
  python -m probe.gmem <module> <root_va> [--handlers N] [--steps N] [--show N]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core.expr import W64, Arena, render  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0
SNAPSHOT_SITE = -1

GUEST_SLOTS = {
    0x00: "rflags", 0x08: "rax", 0x10: "rbp", 0x18: "rbx", 0x20: "rcx",
    0x28: "rdi", 0x30: "rdx", 0x38: "rsi", 0x48: "r8", 0x50: "r9",
    0x58: "r10", 0x60: "r11", 0x68: "r12", 0x70: "r13", 0x78: "r14",
    0x80: "r15",
}
MREGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
         "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
SLOT_OF = {v: k for k, v in GUEST_SLOTS.items()}


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    handlers, show, steps = 300, 8, 120_000
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

    sym_node: dict[str, int] = {}
    for name in MREGS:
        s = a.opaque(f"m_{name}", W64)
        sym_node[f"m_{name}"] = s
        m.set_reg(name, s, 64)
    guest_nodes: dict[int, str] = {v: k for k, v in sym_node.items()}

    def names_of(v: int) -> list[str]:
        return sorted({guest_nodes[n] for n in a.leaves(v) if n in guest_nodes})

    # ---------------------------------------------------------- instrumentation
    base: list[int] = []
    seeded = [False]
    written: set[int] = {0x10}  # see `gops`: slot +0x10 predates the frame base
    seed_at: list[int] = []

    # addr -> Counter(symbol) for reads and writes, and the sites responsible.
    rd_addr: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    wr_addr: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    rd_site: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    wr_site: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    # Values written that are *computed from* guest symbols rather than being
    # one: those are the guest arithmetic, and they are the point.
    composites: list[tuple[int, int, str, list[str]]] = []
    reg_moves: list[tuple[int, int, str, str, str]] = []

    orig_load = Machine.load
    orig_store = Machine.store
    phase = ["prologue"]

    def load(self, addr, nbytes, site, image_u8):
        v = orig_load(self, addr, nbytes, site, image_u8)
        if site == SNAPSHOT_SITE:
            return v
        nm = names_of(v)
        if nm:
            c = self.a.as_const(addr)
            if c is not None and c not in self.mem_owned:
                key = c
                rd_addr[key].update(nm)
                rd_site[key][site] += 1
        return v

    def store(self, addr, value, nbytes, site):
        c = self.a.as_const(addr)
        if c is not None and base and not seeded[0] and c - base[0] in GUEST_SLOTS:
            written.add(c - base[0])
        nm = names_of(value)
        if nm:
            n = self.a.node(value)
            # A bare symbol is a *move*; anything with an operator over symbols is
            # arithmetic. The distinction is the whole question.
            op = self.a.op(value)
            if c is not None:
                wr_addr[c].update(nm)
                wr_site[c][site] += 1
            if op not in ("opaque", "load"):
                composites.append((site, c if c is not None else -1,
                                   render(self.a, value, limit=26), nm))
        return orig_store(self, addr, value, nbytes, site)

    Machine.load = load
    Machine.store = store
    m.mem_owned = set()  # cells the probe itself wrote (the re-seed)

    def boundary(read_ctx: bool):
        snap = {name: m.reg(name, 64) for name in MREGS}
        img_snap: dict[int, int] = {}
        if read_ctx and base:
            for off in GUEST_SLOTS:
                img_snap[off] = m.load(a.const(base[0] + off, W64), 8,
                                       SNAPSHOT_SITE, img.image_u8)
        return snap, img_snap

    bounds: list[tuple[int, int, dict[str, int]]] = []
    ip, stop, at, seen = root, None, 0, 0
    try:
        while at < steps and len(bounds) < handlers + 1:
            kind, payload = L.step(ip)
            at += 1
            if not base:
                b = a.as_const(m.reg("rbp", 64))
                if b is not None:
                    base.append(b)
            if kind != "next":
                stop = payload
                break
            ip = payload
            if len(L.dispatches) <= seen:
                continue
            seen = len(L.dispatches)
            dest = L.dispatches[-1][1]
            if not seeded[0] and len(written) >= len(GUEST_SLOTS):
                seeded[0] = True
                seed_at.append(dest)
                phase[0] = "steady"
                for off, gname in GUEST_SLOTS.items():
                    s = a.opaque(f"i_{gname}", W64)
                    guest_nodes[s] = f"i_{gname}"
                    addr = base[0] + off
                    m.mem_owned.add(addr)
                    orig_store(m, a.const(addr, W64), s, 8, SNAPSHOT_SITE)
            snap, _ = boundary(seeded[0])
            bounds.append((dest, L.dispatches[-1][0], snap))
    finally:
        Machine.load = orig_load
        Machine.store = orig_store

    print(f"root {root:#x}   walked {at} instructions   "
          f"stop {stop.kind if stop else 'budget'}")
    print(f"entry frame (first concrete rbp): {base[0]:#x}" if base else
          "entry frame NOT found")
    print(f"context re-seeded at handler {seed_at[0]:#x}" if seed_at else
          "context NOT re-seeded")
    print(f"boundaries: {len(bounds)}   "
          f"snapshot reads excluded from the census by site={SNAPSHOT_SITE}")
    print()

    # ------------------------------------------------- the second family test
    print("=== addresses that guest values reach ===")
    print(f"{'reads':>6} {'writes':>6}  {'address':>18} {'rel rbp0':>10}  "
          f"distinct guest symbols")
    print("-" * 78)
    alladdr = set(rd_addr) | set(wr_addr)
    for c in sorted(alladdr):
        r, w = sum(rd_addr[c].values()), sum(wr_addr[c].values())
        syms = sorted(set(rd_addr[c]) | set(wr_addr[c]))
        rel = f"{c - base[0]:+#x}" if base else "?"
        print(f"{r:>6} {w:>6}  {c:>#18x} {rel:>10}  "
              f"{len(syms)}: {', '.join(syms[:6])}"
              + (" ..." if len(syms) > 6 else ""))
    print()

    # Falsifier 2, as a stated count rather than a claim.
    ctx_like = [c for c in alladdr
                if len(set(rd_addr[c]) | set(wr_addr[c])) == 1]
    scratch = [c for c in alladdr
               if len(set(rd_addr[c]) | set(wr_addr[c])) > 1]
    print(f"cells holding exactly one guest symbol (context-shaped): {len(ctx_like)}")
    print(f"cells holding more than one (scratch-shaped):            {len(scratch)}")
    print()

    print("=== the context-shaped cells, by slot role ===")
    # If these are a context, then the address minus some base is a constant per
    # guest register. Solve for the base that makes that true, instead of guessing.
    for gname in [GUEST_SLOTS[o] for o in sorted(GUEST_SLOTS)]:
        hits = [c for c in ctx_like
                if set(rd_addr[c]) | set(wr_addr[c]) == {gname}
                or set(rd_addr[c]) | set(wr_addr[c]) == {f"m_{gname}"}
                or set(rd_addr[c]) | set(wr_addr[c]) == {f"i_{gname}"}]
        if not hits:
            continue
        offs = sorted({c - base[0] for c in hits}) if base else []
        print(f"  {gname:<7} {len(hits):>3} cells   rel rbp0: "
              f"{', '.join(f'{o:+#x}' for o in offs[:8])}")
    print()

    print("=== stores of a value COMPUTED from a guest register ===")
    print("(a bare symbol is a move; an operator over symbols is the guest's own "
          "arithmetic)")
    print(f"such stores: {len(composites)}")
    for site, c, txt, nm in composites[:show]:
        rel = f"rbp0{c - base[0]:+#x} (@{c:#x})" if c >= 0 and base else "?"
        print(f"  at {site:#x} -> {rel}")
        print(f"      {txt}")
    print()

    print("=== the same question for registers ===")
    moved = collections.Counter()
    for i in range(len(bounds) - 1):
        dest = bounds[i][0]
        before, after = bounds[i][2], bounds[i + 1][2]
        for r in MREGS:
            if before[r] == after[r]:
                continue
            moved[r] += 1
            nb, na = names_of(before[r]), names_of(after[r])
            op = a.op(after[r])
            if (nb or na) and op not in ("opaque", "load") and a.as_const(after[r]) is None:
                reg_moves.append((dest, L.dispatches[-1][0], r,
                                  render(a, before[r], limit=16),
                                  render(a, after[r], limit=16)))
    print(f"registers moved, by transition: {dict(moved.most_common())}")
    print(f"moves landing on a value computed from a guest register: {len(reg_moves)}")
    for dest, _s, r, b, af in reg_moves[:show]:
        print(f"  handler {dest:#x}: {r} = {af[:120]}")
        print(f"       was {b[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
