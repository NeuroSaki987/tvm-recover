"""The substitution fast path must agree with the algorithm it replaced.

`Arena.substitute` no longer rebuilds the part of the DAG that no pin can reach;
it reuses a cached normal form instead. The claim is that this is the *same*
computation with the repetition removed, and a claim like that has to be
falsified rather than asserted: the original algorithm is kept here, verbatim,
and both are run over a corpus of DAGs and pin sets.

The generator is deliberately adversarial rather than random. It builds the
shapes the folder has rules for -- nested masks, `sel` trees, packed/unpacked
flags, sliceable shifts, self-cancelling arithmetic -- because a disagreement can
only come from a node the folder rewrites, and a uniform random DAG contains
almost none of those.

Usage: python -m tests.test_subst
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

from core.expr import W8, W16, W32, W64, Arena, _children  # noqa: E402


def substitute_naive(a: Arena, r: int, pinmap: dict[int, int]) -> int:
    """The algorithm `Arena.substitute` used before the fast path was added."""
    memo: dict[int, int] = {}
    stack: list[tuple[int, bool]] = [(r, False)]
    while stack:
        cur, ready = stack.pop()
        if cur in memo:
            continue
        pin = pinmap.get(cur)
        if pin is not None:
            memo[cur] = pin
            continue
        n = a.nodes[cur]
        t = n[0]
        if t == "c":
            memo[cur] = cur
            continue
        if t in ("init", "param", "opaque"):
            memo[cur] = cur
            continue
        kids = _children(n)
        if not ready and any(k not in memo for k in kids):
            stack.append((cur, True))
            for k in reversed(kids):
                if k not in memo:
                    stack.append((k, False))
            continue
        memo[cur] = a._rebuild(n, [memo[k] for k in kids])
    return memo[r]


def build_corpus(seed: int = 20261016) -> list[tuple[Arena, list[int]]]:
    """(arena, interesting roots) for a set of adversarial shapes."""
    rnd = random.Random(seed)
    cases: list[tuple[Arena, list[int]]] = []

    for trial in range(60):
        a = Arena()
        leaves = [a.opaque(f"x{i}", W32) for i in range(3)]
        leaves += [a.opaque(f"b{i}", W8) for i in range(3)]
        pool: list[int] = list(leaves)
        roots: list[int] = []

        for _ in range(26):
            op = rnd.choice(["and", "or", "xor", "add", "sub", "shl", "shr", "sel",
                             "trunc", "zext", "eq", "note", "pack"])
            x = rnd.choice(pool)
            if op == "sel":
                c = rnd.choice(leaves[3:6])
                v = a.select(c, x, rnd.choice(pool), a.width(x))
            elif op == "trunc":
                w = rnd.choice([W8, W16, W32])
                v = a.trunc(x, w) if a.width(x) > w else a.zext(x, W64)
            elif op == "zext":
                v = a.zext(x, W64) if a.width(x) < W64 else x
            elif op == "note":
                v = a.un("not", x, a.width(x))
            elif op == "pack":
                v = a.bin("or",
                          a.bin("shl", a.bin("and", rnd.choice(leaves[3:6]),
                                             a.const(1, W8), W8),
                                a.const(rnd.randrange(0, 8), W8), W8),
                          a.bin("shl", a.bin("and", rnd.choice(leaves[3:6]),
                                             a.const(1, W8), W8),
                                a.const(rnd.randrange(0, 8), W8), W8),
                          W8)
            elif op in ("shl", "shr"):
                v = a.bin(op, x, a.const(rnd.randrange(0, 9), a.width(x)), a.width(x))
            elif op in ("eq",):
                v = a.bin("eq", x, rnd.choice(pool), a.width(x))
            else:
                v = a.bin(op, x, rnd.choice(pool), a.width(x)) \
                    if a.width(x) == a.width(rnd.choice(pool)) else x
            pool.append(v)
            if len(pool) > 40:
                pool.pop(0)
            roots.append(v)

        # Self-cancelling arithmetic, which is the shape the protector is built
        # from and the one the interner is supposed to collapse.
        base = a.opaque("z", W32)
        k = a.const(rnd.randrange(1, 1 << 20), W32)
        roots.append(a.bin("sub", a.bin("add", base, k, W32), a.bin("add", base, k, W32), W32))
        roots.append(a.bin("xor", a.bin("or", base, k, W32), a.bin("or", base, k, W32), W32))
        roots.append(a.bin("and", a.un("not", a.bin("or", base, k, W32), W32),
                           a.bin("and", a.un("not", base, W32),
                                 a.un("not", k, W32), W32), W32))
        cases.append((a, roots))
    return cases


def main() -> int:
    bad = 0
    checks = 0
    for arena, roots in build_corpus():
        all_leaves = arena.leaves(roots[-1]) if roots else []
        for root in roots:
            leaves = arena.leaves(root)
            pinsets: list[dict[int, int]] = [{}]
            for lf in leaves[:4]:
                w = arena.width(lf)
                pinsets.append({lf: arena.const(0, w)})
                pinsets.append({lf: arena.const(1, w)})
                pinsets.append({lf: arena.const((1 << w) - 1, w)})
            if len(leaves) >= 2:
                pinsets.append({leaves[0]: arena.const(0, arena.width(leaves[0])),
                                leaves[1]: arena.const(1, arena.width(leaves[1]))})
            for pins in pinsets:
                got = arena.substitute(root, pins)
                want = substitute_naive(arena, root, pins)
                checks += 1
                if got != want:
                    bad += 1
                    if bad <= 6:
                        from core.expr import render
                        print("DISAGREEMENT")
                        print(f"  root  {render(arena, root, limit=10)[:90]}")
                        print(f"  pins  { {k: v for k, v in pins.items()} }")
                        print(f"  fast  {render(arena, got, limit=10)[:90]}")
                        print(f"  naive {render(arena, want, limit=10)[:90]}")
    print(f"substitution equivalence checks: {checks}   disagreements: {bad}")
    if bad:
        print("SUBSTITUTION FAST PATH DOES NOT REPRODUCE THE ORIGINAL")
        return 1
    print("ALL SUBSTITUTION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
