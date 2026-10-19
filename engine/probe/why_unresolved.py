"""Why a dispatch stayed unresolved: candidates, and what pinning each one gives.

`resolve_dispatch` reports "unresolved" as a single word, which is useless when
the address is a large expression. This walks the same three devices the resolver
uses and prints what each one actually produced, so the answer to "why" is a list
of attempts rather than a guess.

Usage: python -m probe.why_unresolved <module> <root_va> [--steps N]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena, boolean_subexpressions, render  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402
from core.explore import Recoverer, hypothesis_candidates  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    steps = 400_000
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    arena = Arena()
    mach = Machine(arena, STACK_BASE)
    L = Lifter(img, mach, arena, vm.va, vm.end)

    ip, stop = root, None
    for _ in range(steps):
        kind, payload = L.step(ip)
        if kind != "next":
            stop = payload
            break
        ip = payload

    if stop is None or getattr(stop, "dest", None) is None:
        print(f"stopped without a symbolic destination: {stop}")
        return 2
    a = L.a
    dest = stop.dest
    print(f"stop            {stop.kind} at {stop.site:#x}")
    print(f"destination     {render(a, dest, limit=40)[:160]}")
    print()

    R = Recoverer(img, vm.va, vm.end, STACK_BASE)
    R.img = img
    executable = R._executable

    leaves = R.symbolic_leaves(a, dest)
    print(f"symbolic leaves ({len(leaves)}):")
    for lf in leaves:
        print(f"  {lf}: {render(a, lf, limit=20)[:120]}")
    # `symbolic_leaves` deliberately excludes entry registers, because a dispatch
    # on a function argument is not an index to enumerate. But "no leaves" and
    # "no *enumerable* leaves" are different statements, and conflating them made
    # the enumeration look like it had failed for a reason it had not.
    kinds: dict[str, int] = {}
    seen_any: set[int] = set()
    stack = [dest]
    inits = []
    while stack:
        cur = stack.pop()
        if cur in seen_any:
            continue
        seen_any.add(cur)
        n = a.nodes[cur]
        t = n[0]
        kinds[t] = kinds.get(t, 0) + 1
        if t == "init":
            inits.append(cur)
        if t == "bin":
            stack.append(n[2])
            stack.append(n[3])
        elif t == "un":
            stack.append(n[2])
        elif t in ("trunc", "zext", "sext", "load"):
            stack.append(n[1])
        elif t == "sel":
            stack.extend((n[1], n[2], n[3]))
    print(f"node kinds in the destination: {kinds}")
    print(f"entry-register leaves: {len(inits)}")
    for lf in inits[:8]:
        print(f"  {lf}: {render(a, lf, limit=8)[:60]}")
    print()

    a_dest = dest
    cands = boolean_subexpressions(a, a_dest)
    print(f"provably-0/1 subexpressions: {len(cands)}")
    for c in cands[:12]:
        got = {}
        for val in (0, 1):
            f = a.as_const(a.substitute(dest, {c: a.const(val, a.width(c))}))
            got[val] = f
        conc = {k: v for k, v in got.items() if v is not None}
        ex = [f"{v:#x}" for v in conc.values() if executable(v)]
        print(f"  {render(a, c, limit=20)[:70]}")
        print(f"      folded: {({k: (hex(v) if v is not None else None) for k, v in got.items()})}"
              f"   executable: {ex}")
    print()

    hyp = hypothesis_candidates(a, dest)
    print(f"bit hypotheses: {len(hyp)}")
    for leaf, m, bit, rewritten in hyp[:6]:
        sub = {leaf: rewritten}
        got = {}
        for val in (0, 1):
            t = dict(sub)
            t[bit] = a.const(val, a.width(bit))
            got[val] = a.as_const(a.substitute(dest, t))
        ex = [f"{v:#x}" for v in got.values() if v is not None and executable(v)]
        print(f"  mask {m:#x} of {render(a, leaf, limit=12)[:40]}  ->  executable {ex}")
    print()

    print("index enumeration:")
    enum = R.enumerate_dispatch(L, dest, {})
    print(f"  {enum if enum is None else (len(enum[0]), 'targets')}")
    print()
    print("enumeration, stage by stage:")
    leaves = [lf for lf in R.symbolic_leaves(a, dest) if a.width(lf) >= 8]
    print(f"  leaves with width >= 8: {len(leaves)}")
    first = leaves[0]
    others = [lf for lf in leaves if lf != first][:3]
    print(f"  first={render(a, first, limit=8)[:50]}  others={len(others)}")
    for label, value_of_others in (("others=0", 0), ("others=1", 1)):
        folded_ok = 0
        targets = set()
        for v in range(256):
            trial = {first: a.const(v, a.width(first))}
            for lf in others:
                trial[lf] = a.const(value_of_others, a.width(lf))
            f = a.as_const(a.substitute(dest, trial))
            if f is None:
                continue
            folded_ok += 1
            if executable(f):
                targets.add(f)
        print(f"  {label}: folded {folded_ok}/256, executable targets {len(targets)}"
              f"  {sorted(hex(t) for t in targets)[:6]}")
    print()
    print("  variation of the other leaves alone (first=0):")
    for lf in others:
        hits = []
        for v in (0, 1, 2, 3, 0xFF, 0x100, 0x7FFFFFFF):
            if v >= (1 << a.width(lf)):
                continue
            trial = {first: a.const(0, a.width(first)), lf: a.const(v, a.width(lf))}
            f = a.as_const(a.substitute(dest, trial))
            hits.append((v, f"{f:#x}" if f is not None else None))
        print(f"    {render(a, lf, limit=8)[:40]}: {hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
