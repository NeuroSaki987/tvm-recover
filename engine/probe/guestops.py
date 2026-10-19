"""Recovered guest operations: what a handler does to the guest register file.

This is the step the engine has been built towards. Everything before it recovers
*transitions* -- "control went from handler A to handler B". What an analyst needs
is the program: "this handler adds the guest's `rbx` to its `rax`". With the guest
register image located (`core/guestctx.py`), a load or store there is a guest
register read or write, and the engine now names them as it interprets.

The attribution is exact rather than statistical. `Lifter.guest_reads` and
`.guest_writes` are append-only logs; a mark is taken at each dispatch boundary,
and the slice since the previous mark is precisely the effect of the handler
dispatched at the previous boundary. Nothing is inferred from timing or position.

The falsifier is structural and it is the reason this is worth printing at all:
**a handler that does not depend on the guest must not appear here**. The VM's
dispatch arithmetic, its handler-table walk and its vip update all touch memory,
and if any of it reached the guest context the log would be noise. So the probe
reports the ratio, and reports separately how many handlers touch the context at
all -- a small number with a large per-handler effect is a semantics; a large
number with tiny effects is a census of the wrong thing.

Usage:
  python -m probe.guestops <module> <root_va> [--handlers N] [--steps N] [--show N]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core.expr import W64, Arena, render  # noqa: E402
from core.guestctx import GuestContext  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0
GUEST_CTX_FROM_ENTRY_RBP = 0x90


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    handlers, show, steps = 300, 14, 120_000
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

    entry_rbp: list[int] = []
    effects: list[tuple[int, list, list]] = []
    ip, stop, at, seen = root, None, 0, 0
    marks = (0, 0)
    prev: list[int | None] = [None]

    while at < steps and len(effects) < handlers:
        kind, payload = L.step(ip)
        at += 1
        if not entry_rbp:
            b = a.as_const(m.reg("rbp", 64))
            if b is not None:
                entry_rbp.append(b)
                # The image is `entry_rbp + 0x90`; the offset is measured on this
                # build in `probe/guestmap.py` and cross-checked against the
                # reference's `vmctx` output.
                L.enable_guest_census(
                    GuestContext(b + GUEST_CTX_FROM_ENTRY_RBP))
        if kind != "next":
            stop = payload
            break
        ip = payload
        if len(L.dispatches) <= seen:
            continue
        seen = len(L.dispatches)
        dest = L.dispatches[-1][1]
        reads, writes = L.guest_effect(*marks)
        if prev[0] is not None and (reads or writes):
            effects.append((prev[0], reads, writes))
        marks = (len(L.guest_reads), len(L.guest_writes))
        prev[0] = dest

    print(f"root {root:#x}   walked {at} instructions   "
          f"stop {stop.kind if stop else 'budget'}")
    print(f"entry frame rbp0: {entry_rbp[0]:#x}" if entry_rbp else
          "entry frame NOT found")
    if entry_rbp:
        ctx = GuestContext(entry_rbp[0] + GUEST_CTX_FROM_ENTRY_RBP)
        lo, hi = ctx.span
        print(f"guest register image: {lo:#x} .. {hi:#x}   "
              f"({ctx.offset('rax'):#x} rax .. {ctx.offset('rflags'):#x} rflags)")
    print(f"handler transitions: {seen}")
    print()
    print(f"guest register reads recorded:  {len(L.guest_reads)}")
    print(f"guest register writes recorded: {len(L.guest_writes)}")
    print(f"handlers that touched the guest at all: {len(effects)} of {seen}")
    print()

    if not entry_rbp:
        return 2

    by_reg_read = collections.Counter(r for _s, r in L.guest_reads)
    by_reg_write = collections.Counter(r for _s, r, _v in L.guest_writes)
    print("registers read,  by count: "
          f"{dict(by_reg_read.most_common())}")
    print("registers written, by count: "
          f"{dict(by_reg_write.most_common())}")
    print()
    # The host cache, derived from the symbolic state rather than from a second
    # seeded walk. **This is the materialisation layout, not the handler-time
    # cache, and the two disagree** -- `rsi <- guest rsi` here against
    # `rsi <- guest rcx` 107/107 in `probe/guestcache.py`. See `core/guestctx.py`.
    cache = L.guest_cache
    print("which host register each slot is MATERIALISED into (not the "
          "handler-time cache):")
    if cache.assignment:
        for host, guest in sorted(cache.assignment.items()):
            print(f"    {host:<5} <- guest {guest:<7} "
                  f"({cache.counts(host).get(guest, 0)} times, "
                  f"unambiguous)")
    else:
        print("    none: no host register ever held a bare context load")
    print("host registers that stage several guest registers (not a cache):")
    for host, guests in sorted(cache.staging.items()):
        print(f"    {host:<5} carries {len(guests)}: {', '.join(guests[:8])}")
    print()

    # A handler's effect shape, so the same handler can be checked for consistency.
    shape: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for dest, reads, writes in effects:
        s = "rd[" + ",".join(sorted(r for _a, r in reads)) + "] " \
            "wr[" + ",".join(sorted(r for _a, r, _v in writes)) + "]"
        shape[dest][s] += 1
    multi = {d: c for d, c in shape.items() if len(c) > 1}
    print(f"distinct handlers with a guest effect: {len(shape)}")
    print(f"handlers whose effect shape is not single-valued: {len(multi)}"
          "   (a handler is a function of the guest state, so a second shape is"
          " either vip-dependence or a measurement error)")
    for d, c in list(sorted(multi.items()))[:5]:
        print(f"    {d:#x}: {dict(c)}")
    print()

    print("=== recovered guest effects ===")
    for dest, reads, writes in effects[:show]:
        print(f"--- handler {dest:#x} ---")
        if reads:
            print("    reads : " + ", ".join(f"{r}@{addr:#x}"
                                              for addr, r in reads[:8]))
        for addr, r, v in writes[:8]:
            print(f"    write : {r:<7} = {render(a, v, limit=20)[:120]}")

    # The falsifier, stated as a number: how much of what the handlers do to this
    # block is *not* guest state. If nearly every handler touches it, the block is
    # scratch and the naming is meaningless.
    frac = len(effects) / max(seen, 1)
    print()
    print(f"FALSIFIER -- fraction of handlers touching the guest register file: "
          f"{frac:.3f}")
    print("    a VM that used this block as scratch would touch it on nearly "
          "every handler (1.0); a VM that keeps guest state there touches it "
          "when the guest code touches a register, which is not every handler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
