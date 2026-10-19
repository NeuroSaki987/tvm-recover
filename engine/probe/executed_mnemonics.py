"""Which mnemonics does the VM actually *execute*, and how often?

`probe.mnemonic_census` sweeps `.tvm0` linearly, which is not the same population as
code that runs: the section holds handler code *and* the bytecode and tables the
handlers read as data, so a linear sweep can decode thousands of instructions the VM
never executes and can split a real instruction at a data boundary.

The question this probe answers is asked because of `_rotate`: its body ignores the
`op_name` it is constructed with, so `_rotate("ror")` and `_rotate("rol")` build the
same **left** rotation. A linear census reported 884 `ror` and no `rol`, which would
make that a large wrong-value bug; an executed census says whether it is a live bug or
an artefact of sweeping data.

The population here is every instruction the lifter actually steps, over all 14 roots of
the module, during full CFG recovery. `_ROTATE_NAMES` and the shift names are called out
so the specific family this round touches is named rather than left in a table.

Usage: python -m probe.executed_mnemonics <module> [--steps=N] [--blocks=N] [--root=0xADDR]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core.devirt import module_roots  # noqa: E402
from core.explore import STACK_BASE, Recoverer  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.peimg import PeImage  # noqa: E402

WATCH = ("rol", "ror", "shl", "sal", "shr", "sar", "inc", "dec", "neg",
         "mul", "imul", "jo", "jno", "seto", "lahf")
EXECUTED: collections.Counter = collections.Counter()
SITES: dict[str, set] = {}


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

    def counting_step(self, ip):
        size, insn, _raw = self.decode(ip)
        if insn is not None:
            EXECUTED[insn.mnemonic] += 1
            SITES.setdefault(insn.mnemonic, set()).add(ip)
        return real_step(self, ip)

    Lifter.step = counting_step
    per_root = {}
    try:
        for root in sorted(roots):
            R = Recoverer(img, vm.va, vm.end, STACK_BASE, step_budget=steps,
                          block_budget=blocks)
            cfg = R.recover(root)
            per_root[root] = (len(cfg.blocks), cfg.unresolved, cfg.steps)
    finally:
        Lifter.step = real_step

    total = sum(EXECUTED.values())
    print(f"module {args[0]}")
    print(f"roots {len(roots)}   executed instruction steps {total}  "
          f"({sum(s for _b, _u, s in per_root.values())} counted by the recoverer)")
    print()
    print(f"{'mnemonic':<10} {'executions':>11} {'distinct sites':>15}  note")
    print("-" * 64)
    for mn, n in EXECUTED.most_common():
        note = "<< watched" if mn in WATCH else ""
        print(f"{mn:<10} {n:>11} {len(SITES[mn]):>15}  {note}")
    print()
    print("watched family (executed, not swept):")
    for mn in WATCH:
        n = EXECUTED.get(mn, 0)
        print(f"  {mn:<6} {n:>8} execution(s) at {len(SITES.get(mn, ()))} distinct site(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
