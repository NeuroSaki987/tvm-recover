"""The seven modelled flags at one address, as expressions.

Ordinary register dumps cannot show this, and the flags are where the VM's
dispatch index comes from -- `sete`, `cmovcc`, `pushfq` and the packed-RFLAGS
word all read them. When a register's expression is a flag that looks like the
wrong flag, this is the only way to tell which.

Usage: python -m probe.flags_at <module> <root_va> <at_va> [--steps N]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena, render  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import FLAG_BITS, FLAGS, Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    steps = 400_000
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
    path, root, at = args[0], int(args[1], 0), int(args[2], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, STACK_BASE)
    L = Lifter(img, m, a, vm.va, vm.end)

    ip, executed = root, 0
    for _ in range(steps):
        if ip == at:
            break
        kind, payload = L.step(ip)
        executed += 1
        if kind != "next":
            print(f"stopped early at {ip:#x}: {payload}")
            break
        ip = payload

    print(f"--- flags at {at:#x} (after {executed} instructions) ---")
    for f in FLAGS:
        v = m.flag(f)
        c = a.as_const(v)
        text = f"{c:#x}" if c is not None else render(a, v, limit=40)
        print(f"  {f:<3} bit {FLAG_BITS[f]:<3}  {text[:130]}")
    print()
    print(f"packed word: {render(a, m.pack_flags(), limit=40)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
