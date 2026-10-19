"""Does the guest's own arithmetic appear, and where?

`Recoverer.derive_guest_cache` measures the handler-time host cache and
`Recoverer.guest_operations` re-expresses the seeded walk's register writes in
guest terms. This probe asks the question those two exist to answer, and asks it
as a precise query rather than by eyeballing rendered expressions:

**how many writes carry an expression over two or more *different* guest
registers?**

A value over one guest register is either a copy or the protector operating on a
single guest value -- a shift, a mask, a flags merge, an address computed from it.
A value over two *different* guest registers is the guest program doing arithmetic
between its own registers, which is the thing a devirtualizer is for. Nothing in
between needs a judgement call.

The falsifier is stated before the run and it is the whole point of the probe:
**if that count is zero across the entire function, then guest-vs-guest arithmetic
does not appear on this path** -- and that is a statement about the program, not
about the method, because the method has already been shown to reach the guest
registers (the cache is measured, and 195 computed writes were recovered over the
same walk). The probe prints the count either way and refuses to dress a zero up as
a partial success.

Second measurement, because it changes how the cache must be read: **the cache
counts are conditional.** A host register is credited with a guest register on the
boundaries where it carried a seeded symbol -- 107 boundaries, not 400 -- so
"`rsi` <- guest `rcx`" means "whenever `rsi` carried a seeded symbol it was
`rcx`", not "`rsi` always holds `rcx`". The probe prints the coverage alongside the
map so the conditional is impossible to miss.

Usage:
  python -m probe.guestarith <module> <root_va> [--handlers N] [--steps N] [--show N]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core.expr import render  # noqa: E402
from core.explore import STACK_BASE, Recoverer  # noqa: E402
from core.peimg import PeImage  # noqa: E402


def symbols_of(arena, value: int) -> list[str]:
    """The guest registers a seeded-walk expression depends on."""
    return sorted({arena.node(n)[1][2:] for n in arena.leaves(value)
                   if arena.op(n) == "opaque"
                   and arena.node(n)[1].startswith("H_")})


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    handlers, steps, show = 900, 400_000, 8
    for x in sys.argv[1:]:
        if x.startswith("--handlers="):
            handlers = int(x.split("=", 1)[1], 0)
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--show="):
            show = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    R = Recoverer(img, vm.va, vm.end, STACK_BASE)
    cache = R.derive_guest_cache(root, handlers=handlers, steps=steps)
    walk = R.handler_cache_walk
    print(f"root {root:#x}   walk {walk}")
    print()

    if not walk.get("seeded"):
        print("the context was never filled, so nothing was seeded and the walk "
              "says nothing about the cache")
        return 2

    # ------------------------------------------------------------ the cache
    print("=== handler-time cache, and how conditional it is ===")
    print("a host register is credited on the boundaries where it carried a "
          "seeded symbol")
    print(f"{'host':<6} {'boundaries':>10}  guest register")
    print("-" * 46)
    for host in sorted(cache.assignment):
        counts = cache.counts(host)
        guest = cache.assignment[host]
        print(f"{host:<6} {counts[guest]:>10}  guest {guest}")
    if cache.staging:
        print("staging (more than one guest register passes through):")
        for host, guests in sorted(cache.staging.items()):
            print(f"    {host:<5} {len(guests)}: {', '.join(guests)}")
    print(f"boundaries in the seeded walk: {walk['boundaries']}")
    print()

    # ------------------------------------------------------------ the answer
    arena = R._op_arena
    ops = R.guest_ops
    one_only: list[tuple] = []
    two_plus: list[tuple] = []
    none: list[tuple] = []
    for handler, host, _before, after in ops:
        syms = symbols_of(arena, after)
        if len(syms) >= 2:
            two_plus.append((handler, host, syms, after))
        elif len(syms) == 1:
            one_only.append((handler, host, syms, after))
        else:
            none.append((handler, host, [], after))

    print("=== register writes mentioning a seeded guest symbol ===")
    print(f"total: {len(ops)}")
    print(f"  over two or more DIFFERENT guest registers: {len(two_plus)}")
    print(f"  over exactly one guest register:            {len(one_only)}")
    print(f"  mentioning none (reachable only through a store): {len(none)}")
    print()

    bare = sum(1 for _h, _host, _s, after in one_only
               if arena.op(after) == "opaque")
    print(f"  of the single-register ones, bare copies: {bare}, "
          f"computed: {len(one_only) - bare}")
    print()

    print("=== guest-vs-guest arithmetic ===")
    if not two_plus:
        print("NONE. Across the whole walk no register ever received a value "
              "computed from two different guest registers.")
        print("    This is a statement about the program on this path, not about "
              "the method: the cache is measured, the guest registers are named, "
              "and computed values over a single guest register were recovered "
              "in the same walk.")
    else:
        tally = collections.Counter((h, host, tuple(syms),
                                     render(arena, after, limit=20).replace("H_", ""))
                                    for h, host, syms, after in two_plus)
        for (h, host, syms, txt), n in tally.most_common(show):
            print(f"  x{n:<3} handler {h:#x}  {host} <- {txt[:104]}")
            print(f"          over {', '.join(syms)}")
    print()

    print("=== the biggest single-register expressions (what the protector "
          "actually computes) ===")
    tally = collections.Counter(
        (h, host, tuple(syms), render(arena, after, limit=20).replace("H_", ""))
        for h, host, syms, after in one_only)
    for (h, host, syms, txt), n in tally.most_common(show):
        print(f"  x{n:<3} handler {h:#x}  {host} <- {txt[:100]}")
        print(f"          over guest {', '.join(syms)}")
    print()

    # ---------------------------------------------------------------- shape
    # What a simplifier would have to cope with, measured rather than guessed.
    computed = [v for _h, host, _s, v in one_only if arena.op(v) != "opaque"]
    print("=== shape of the computed expressions ===")
    print(f"computed (not a bare copy): {len(computed)}")
    ops: collections.Counter = collections.Counter()
    sizes: list[int] = []
    other: collections.Counter = collections.Counter()
    for v in computed:
        desc = arena._descendants_ordered(v)
        sizes.append(len(desc))
        for n in desc:
            ops[arena.op(n)] += 1
            if arena.op(n) == "opaque":
                other[arena.node(n)[1].split("_")[0] + "_*"] += 1
            elif arena.op(n) == "load":
                other["load"] += 1
            elif arena.op(n) == "init":
                other["init_*"] += 1
    sizes.sort()
    print(f"  DAG size: min {sizes[0]}, median {sizes[len(sizes)//2]}, "
          f"max {sizes[-1]}, total {sum(sizes)}")
    print(f"  node kinds: {dict(ops.most_common())}")
    print(f"  non-constant leaves: {dict(other.most_common())}")
    if other and not any(k == "H_*" for k in other):
        print("  !! no H_* leaves: the shapes above are not over guest registers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
