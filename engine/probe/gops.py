"""What guest operation does each handler perform?

The engine recovers *transitions* -- "control went from handler A to handler B".
A devirtualizer owes the analyst the guest program: "this handler adds the guest's
`rbx` to its `rax`". This probe takes the first step of that, and it is a
correction of the previous one.

`probe/guest_effect.py` asked whether a machine register *mentions* a guest
symbol. It never asked whether the value **changed**, so it could not see an
operation: `rsi` holding `g_rcx` and `rsi` holding `g_rcx + 1` were both counted
as "still mentions a guest symbol" and reported identically. That is the third
time this workspace has mistaken a *mention* for a *measurement*.

So here the question is stated as a difference. At every dispatch boundary the
whole machine register file is snapshotted by **node identity**, and consecutive
snapshots are compared. Whatever moved between two boundaries moved because one
handler ran, and the before/after pair is the handler's effect.

Two symbol families, because one cannot answer the question:

* `m_<reg>` -- the machine register file seeded before the entry sequence, so
  `m_rcx` *is* the guest `rcx` at entry;
* `i_<reg>` -- the guest context image, overwritten with fresh symbols once the
  prologue has finished writing all sixteen slots.

A register still holding `m_rcx` is carrying a guest value across dispatches. A
register that comes to hold `i_rax` was **loaded from the context image**, which
is the question the previous probe could not ask. A store whose value mentions
either is the VM moving guest state.

Falsifiers, stated before the run:

1. a register whose new value mentions the vip, a `.tvm0` address, the frame base
   or the stack base is VM bookkeeping and is reported as such, not counted as a
   guest effect;
2. the **same handler entered twice must produce the same effect shape** -- if it
   does not, the "effect" is not a function of the handler and nothing here is a
   semantics;
3. an untouched register must hold the **identical node**, not an equal
   expression, or a handler that copied a register onto itself would look like an
   operation.

`rsp` is deliberately not seeded: the VM emulates the guest stack inside its own
frame, so a symbolic `rsp` puts every guest push at an unknown address and
recovers nothing.

Usage:
  python -m probe.gops <module> <root_va> [--handlers N] [--steps N] [--show N]
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

# Guest context slots at the entry frame. Measured in
# `evidence/E-CONTEXT-MAPPING.md`; not guessed from disassembly.
GUEST_SLOTS = {
    0x00: "rflags", 0x08: "rax", 0x10: "rbp", 0x18: "rbx", 0x20: "rcx",
    0x28: "rdi", 0x30: "rdx", 0x38: "rsi", 0x48: "r8", 0x50: "r9",
    0x58: "r10", 0x60: "r11", 0x68: "r12", 0x70: "r13", 0x78: "r14",
    0x80: "r15",
}
MREGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
         "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    handlers, show, steps = 300, 10, 120_000
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

    # `m_*`: the machine register file at entry. Because `root` is the entry of a
    # virtualized function, the machine registers *are* the guest registers there,
    # so `m_rcx` is the guest `rcx` and no mapping has to be assumed.
    sym_node: dict[str, int] = {}
    for name in MREGS:
        s = a.opaque(f"m_{name}", W64)
        sym_node[f"m_{name}"] = s
        m.set_reg(name, s, 64)
    guest_nodes: set[int] = set(sym_node.values())

    def names_of(v: int) -> list[str]:
        """The guest symbols an expression actually depends on."""
        return [a.node(n)[1] for n in a.leaves(v) if n in guest_nodes]

    def is_guest(v: int) -> bool:
        return any(n in guest_nodes for n in a.leaves(v))

    # ---------------------------------------------------------- instrumentation
    base: list[int] = []
    seeded = [False]
    written: set[int] = {0x10}  # slot +0x10 predates the frame base; see below
    seed_site: list[int] = []
    img_loaded: list[tuple[int, str]] = []
    guest_stores: list[tuple[int, int, int, int, str]] = []
    cur_site = [0]

    orig_load = Machine.load
    orig_store = Machine.store

    def load(self, addr, nbytes, site, image_u8):
        v = orig_load(self, addr, nbytes, site, image_u8)
        nm = names_of(v)
        if nm:
            # A load that produced a guest symbol: either the context image, or a
            # spill slot. Which one is decided by *which* symbol came back.
            img_loaded.append((site, ",".join(sorted(set(nm)))))
        return v

    def store(self, addr, value, nbytes, site):
        c = self.a.as_const(addr)
        if c is not None and base and not seeded[0]:
            off = c - base[0]
            if off in GUEST_SLOTS:
                written.add(off)
        nm = names_of(value)
        if nm:
            rbp = self.a.as_const(self.reg("rbp", 64))
            rbp0 = base[0] if base else 0
            guest_stores.append((site, c if c is not None else -1,
                                 (c - rbp0) if (c is not None and base) else 0,
                                 rbp if rbp is not None else -1,
                                 ",".join(sorted(set(nm)))))
        return orig_store(self, addr, value, nbytes, site)

    Machine.load = load
    Machine.store = store

    # A boundary is a dispatch this engine resolved. `step` has just executed the
    # dispatch `jmp`, so the state here is the *next* handler's entry state.
    bounds: list[tuple[int, int, dict[str, int], dict[int, int]]] = []
    ip, stop, at, seen = root, None, 0, 0
    try:
        while at < steps and len(bounds) < handlers + 1:
            kind, payload = L.step(ip)
            at += 1
            cur_site[0] = ip
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
            snap = {name: m.reg(name, 64) for name in MREGS}
            imgsnap: dict[int, int] = {}
            if seeded[0]:
                for off in GUEST_SLOTS:
                    imgsnap[off] = m.load(a.const(base[0] + off, W64), 8, 0,
                                          img.image_u8)
            elif len(written) >= len(GUEST_SLOTS):
                # Every slot has been written, so the image is complete and this
                # is the first dispatch after that. Replace it with fresh symbols
                # so an image *read* is distinguishable from register residency.
                #
                # Slot +0x10 (guest rbp) is saved by `mov qword ptr [rsp+0x10], rbp`
                # -- the instruction *before* `mov rbp, rsp` creates the base -- so
                # it can never be attributed to an offset by watching stores and a
                # completeness test that waits for it never fires. It is seeded
                # anyway, below; only the *test* excludes it.
                seeded[0] = True
                seed_site.append(dest)
                for off, gname in GUEST_SLOTS.items():
                    s = a.opaque(f"i_{gname}", W64)
                    sym_node[f"i_{gname}"] = s
                    guest_nodes.add(s)
                    orig_store(m, a.const(base[0] + off, W64), s, 8, 0)
            bounds.append((dest, L.dispatches[-1][0], snap, imgsnap))
    finally:
        Machine.load = orig_load
        Machine.store = orig_store

    print(f"root {root:#x}   walked {at} instructions   "
          f"stop {stop.kind if stop else 'budget'}")
    print(f"entry frame (first concrete rbp): {base[0]:#x}" if base
          else "entry frame NOT found")
    print(f"context image re-seeded with fresh symbols at handler "
          f"{seed_site[0]:#x}" if seed_site else
          "context image NOT re-seeded (prologue never wrote all slots)")
    print(f"dispatch boundaries snapshotted: {len(bounds)}")
    print()

    if len(bounds) < 2:
        print("fewer than two boundaries: nothing to difference")
        return 2

    # ------------------------------------------------------- the difference
    # A transition's effect is the set of machine registers whose *node identity*
    # changed between two consecutive boundaries.
    leaks = ("0x1401", "0x1400", ":")  # .tvm0 addresses and loaded nodes
    per_handler: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    per_handler_regs: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    moved: collections.Counter = collections.Counter()
    transitions: list[tuple[int, int, list[tuple[str, int, int]]]] = []
    silent = 0
    guest_moves = 0
    bookkeeping = 0

    for i in range(len(bounds) - 1):
        dest = bounds[i][0]          # the handler that ran
        site = bounds[i][1]
        before, after = bounds[i][2], bounds[i + 1][2]
        changed = [(r, before[r], after[r]) for r in MREGS
                   if before[r] != after[r]]
        if not changed:
            silent += 1
        transitions.append((dest, site, changed))
        for r, b, af in changed:
            moved[r] += 1
            if is_guest(b) or is_guest(af):
                guest_moves += 1
            txt = render(a, af, limit=14)
            if any(t in txt for t in leaks):
                bookkeeping += 1
            per_handler_regs[dest][r] += 1
        shape = "; ".join(f"{r}:{'G' if (is_guest(b) or is_guest(af)) else '-'}"
                          f"{'L' if any(t in render(a, af, limit=14) for t in leaks) else ''}"
                          for r, b, af in sorted(changed))
        per_handler[dest][shape] += 1

    print("=== how many machine registers each handler moves ===")
    print(f"transitions: {len(transitions)}")
    print(f"transitions where no machine register changed: {silent}")
    print(f"register-moves whose value mentions a guest symbol on either side: "
          f"{guest_moves}")
    print(f"register-moves landing on VM bookkeeping (a `.tvm0` address or a load): "
          f"{bookkeeping}")
    print(f"registers moved, by transition count: {dict(moved.most_common())}")
    print()

    print("=== does the same handler always do the same thing? ===")
    # Falsifier 2. A handler is a *function*, so its effect shape must be
    # single-valued. A handler that shows two shapes is either vip-dependent --
    # which contradicts block identity being (handler, vip) and would be a
    # finding in itself -- or the shape is measuring something else.
    multi = {d: c for d, c in per_handler.items() if len(c) > 1}
    print(f"handlers entered more than once: "
          f"{sum(1 for d, c in per_handler.items() if sum(c.values()) > 1)}")
    print(f"handlers with more than one effect shape: {len(multi)} of "
          f"{len(per_handler)}")
    for d, c in list(sorted(multi.items()))[:6]:
        print(f"  {d:#x}: {dict(c)}")
    print()

    print("=== per-handler effects (guest-relevant ones only) ===")
    rows = []
    for dest, site, changed in transitions:
        g = [(r, b, af) for r, b, af in changed
             if is_guest(b) or is_guest(af)]
        if not g:
            continue
        rows.append((dest, site, g))
    print(f"transitions with at least one guest-relevant register move: {len(rows)}")
    for dest, site, g in rows[:show]:
        print(f"--- handler {dest:#x} (dispatched at {site:#x}) ---")
        for r, b, af in g:
            print(f"    {r:<4} = {render(a, af, limit=18)[:150]}")
            if render(a, b, limit=18) != render(a, af, limit=18):
                print(f"         was {render(a, b, limit=18)[:140]}")
    print()

    print("=== image reads and guest stores ===")
    reads = collections.Counter(img_loaded)
    print(f"loads that returned a guest symbol: {len(img_loaded)}")
    for (site, nm), n in reads.most_common(12):
        print(f"  {n:>5}x at {site:#x}  -> {nm}")
    print()
    stores = collections.Counter(guest_stores)
    print(f"stores whose value mentions a guest symbol: {len(guest_stores)}")
    for (site, addr, off, rbp, nm), n in stores.most_common(12):
        rel = f"rbp0{off:+#x}" if addr >= 0 and base else "?"
        print(f"  {n:>5}x at {site:#x}  -> {rel}   value={nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
