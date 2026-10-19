"""Every `call` whose target did not fold: what is that target made of?

`0x14015b1ba` leaves two `unresolved call` blocks — `call qword ptr [0x14011c548]`
at `0x14015cb9b` and `call qword ptr [0x14011c538]` at `0x14015d582`. Both slots are
`00 00 .. 00` on disk inside writable `.data`, and there is no delay-import
directory (`probe.delayload`), so the value really is chosen at run time and
`PeImage.foldable` is right to refuse it. The reference closes this root with 0
unresolved blocks, so it must be modelling these calls somehow.

**The claim this probe tests, stated before running.** *H:* in this module, every
`call` whose target does not fold goes through a pointer the program writes at run
time, so its destination is a read at a **constant** address that the image declines
to fold — never a VM-internal computed destination.

**The falsifier.** For every root, classify each `dispatch_call` destination. If any
one of them is *not* a read at a constant, non-foldable address — i.e. it is a read
at a symbolic address, a bare entry register, or an arbitrary computed expression —
then *H* is false, and modelling symbolic call targets as opaque returns would be a
guess rather than a reading of the image. In that case the change must be restricted
to the decidable class and the rest left unresolved.

The population is every `dispatch_call` stop reached by full CFG recovery over the
module's 14 roots, so the counts below are per-root and the total is named with it.

Usage: python -m probe.calltarget_census <module> [--steps=N] [--blocks=N] [--root=0xADDR]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core.explore import STACK_BASE, Recoverer  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.peimg import PeImage  # noqa: E402
from core.devirt import module_roots  # noqa: E402

OBSERVED: list[tuple[int, int, str, str]] = []   # (root, site, class, detail)


def classify(img: PeImage, arena, dest: int) -> tuple[str, str]:
    """(class, detail) for one symbolic call destination."""
    n = arena.node(dest)
    kind = n[0]
    if kind == "load":
        addr = arena.as_const(n[1])
        width = n[2]
        if addr is None:
            return "load@symbolic-address", f"width {width}"
        sec = img.section_for_va(addr)
        name = sec.name if sec else "unmapped"
        foldable = img.foldable(addr, width // 8)
        if foldable:
            # Foldable and yet symbolic: the folder saw a stored value that is not a
            # constant, i.e. something on this path wrote the slot.
            return "load@constant-foldable-but-symbolic", (
                f"[{addr:#x}] {name} width {width}")
        bs = [img.image_u8(addr + i) for i in range(width // 8)]
        text = ("zero-on-disk" if all(b == 0 for b in bs if b is not None)
                else "nonzero-on-disk" if all(b is not None for b in bs)
                else "partly-unbacked")
        return "load@constant-nonfoldable", (
            f"[{addr:#x}] {name} width {width} {text} "
            f"writable={sec.writable if sec else '?'}")
    if kind in ("init", "param", "opaque"):
        return f"{kind}-value", arena.op(dest) if kind != "init" else n[1]
    return "computed", kind


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    steps, blocks, only = 400_000, 512, []
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        elif x.startswith("--blocks="):
            blocks = int(x.split("=", 1)[1], 0)
        elif x.startswith("--root="):
            only.append(int(x.split("=", 1)[1], 0))

    img = PeImage(args[0])
    vm = img.section_by_name(".tvm0")
    roots = only or module_roots(img)

    real_step = Lifter.step
    current_root = [0]

    def counting_step(self, ip):
        kind, payload = real_step(self, ip)
        if kind == "stop" and getattr(payload, "kind", None) == "dispatch_call":
            cls, detail = classify(img, self.a, payload.dest)
            OBSERVED.append((current_root[0], payload.site, cls, detail))
        return kind, payload

    Lifter.step = counting_step
    per_root = {}
    # The rule under test is `Lifter._runtime_bound_slot`; once it is in place the
    # class it covers no longer reaches a `dispatch_call` stop at all, so a probe
    # that only watched stops would report an empty population and look like the
    # question had gone away. Both channels are therefore counted: what still
    # *stops*, and what the rule *absorbed*.
    absorbed: dict[tuple[int, int], int] = {}
    stops_seen: dict[int, int] = {}
    try:
        for root in sorted(roots):
            current_root[0] = root
            before = len(OBSERVED)
            R = Recoverer(img, vm.va, vm.end, STACK_BASE, step_budget=steps,
                          block_budget=blocks)
            cfg = R.recover(root)
            per_root[root] = (len(cfg.blocks), cfg.unresolved, cfg.steps,
                              len(OBSERVED) - before)
            for slot, n in R.lifter.runtime_bound_calls.items():
                absorbed[(root, slot)] = absorbed.get((root, slot), 0) + n
            stops_seen[root] = sum(1 for r, _s, _c, _d in OBSERVED if r == root)
    finally:
        Lifter.step = real_step

    print(f"module {args[0]}")
    print(f"roots  {len(roots)}   steps/block {steps}   blocks {blocks}")
    print()
    head = (f"{'root':>14} {'blocks':>7} {'unres':>6} {'instr':>9} "
            f"{'dispatch_call stops':>20}")
    print(head)
    print("-" * len(head))
    tot_blocks = tot_unres = tot_instr = tot_calls = 0
    for root in sorted(per_root):
        b, u, s, c = per_root[root]
        tot_blocks += b
        tot_unres += u
        tot_instr += s
        tot_calls += c
        print(f"{root:>14} {b:>7} {u:>6} {s:>9} {c:>20}")
    print("-" * len(head))
    print(f"{'TOTAL':>14} {tot_blocks:>7} {tot_unres:>6} {tot_instr:>9} {tot_calls:>20}")
    print()

    # A stop is re-observed on every attempt, so the population that matters is the
    # distinct (root, site) pairs.
    distinct = {(r, s, c, d) for r, s, c, d in OBSERVED}
    print(f"{len(OBSERVED)} observation(s) at {len({(r, s) for r, s, _, _ in distinct})} "
          f"distinct (root, site) pair(s)")
    print()
    by_class = collections.Counter(c for _r, _s, c, _d in distinct)
    print("by class, over distinct (root, site) pairs:")
    width = max(len(k) for k in by_class) if by_class else 10
    for cls, n in by_class.most_common():
        print(f"  {cls:<{width}}  {n:>4}")
    print()
    print("detail:")
    for (root, site, cls, detail) in sorted(distinct):
        print(f"  {root:#x} @ {site:#x}  {cls:<34} {detail}")
    print()
    print(f"absorbed by `Lifter._runtime_bound_slot` into an opaque call: "
          f"{len(absorbed)} distinct (root, slot) pair(s)")
    for (root, slot), n in sorted(absorbed.items()):
        sec = img.section_for_va(slot)
        print(f"  {root:#x} slot {slot:#x}  {sec.name if sec else '?'}  "
              f"{n} call(s)")
    print()
    if not distinct and not absorbed:
        print("FALSIFIER NOT EXERCISED: no symbolic call target was reached at all,")
        print("so this probe says nothing about the claim.")
        return 0
    if not distinct:
        print(f"Every symbolic call target in the population ({len(absorbed)} distinct "
              f"(root, slot) pair(s)) is a read at a constant address the image "
              f"declines to fold, and none is left as a `dispatch_call` stop.")
        return 0
    nondecidable = [d for d in distinct
                    if d[2] != "load@constant-nonfoldable"]
    if nondecidable:
        print(f"H IS FALSIFIED: {len(nondecidable)} of {len(distinct)} distinct "
              f"symbolic call target(s) are NOT a read at a constant non-foldable "
              f"address.")
        for root, site, cls, detail in sorted(nondecidable):
            print(f"  {root:#x} @ {site:#x}  {cls}  {detail}")
        return 0
    print(f"H SURVIVES: all {len(distinct)} distinct remaining symbolic call target(s) "
          f"are reads at a constant address the image declines to fold, leaving "
          f"{len(absorbed)} absorbed slot(s) outside this list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
