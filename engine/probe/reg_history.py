"""Every write to one register along the path, with instruction indices.

`state_at` says *that* a register differs at some address; this says *which
instruction put the value there*, and at what point in the walk. That is what
turns "the engines disagree about r9" into "they disagree about the instruction
at index N", which is a `--break-at` away from a direct comparison.

Usage: python -m probe.reg_history <module> <root_va> <reg> [--steps N] [--tail K] [--from N]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena, render  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    steps, tail, start = 400_000, 40, 0
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--tail="):
            tail = int(x.split("=", 1)[1], 0)
        if x.startswith("--from="):
            start = int(x.split("=", 1)[1], 0)
    path, root, reg = args[0], int(args[1], 0), args[2]

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, STACK_BASE)
    L = Lifter(img, m, a, vm.va, vm.end)

    ip = root
    prev = m.reg(reg, 64)
    history: list[tuple[int, int]] = []  # (instruction index, site)
    stop = None
    for i in range(steps):
        size, insn, _ = L.decode(ip)
        text = f"{insn.mnemonic} {insn.op_str}" if insn is not None else "<undecodable>"
        kind, payload = L.step(ip)
        cur = m.reg(reg, 64)
        if cur != prev:
            history.append((i, ip))
            if i >= start:
                before = a.as_const(prev)
                after = a.as_const(cur)
                bs = f"{before:#x}" if before is not None else render(a, prev)[:56]
                as_ = f"{after:#x}" if after is not None else render(a, cur)[:56]
                print(f"  [{i:>6}] {ip:#x}  {text:<36} {reg}: {bs}  ->  {as_}")
            prev = cur
        if kind != "next":
            stop = payload
            break
        ip = payload

    print()
    print(f"stop {stop}")
    print(f"writes to {reg}: {len(history)}")
    print(f"last {min(tail, len(history))} write sites:")
    for i, site in history[-tail:]:
        print(f"  [{i:>6}] {site:#x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
