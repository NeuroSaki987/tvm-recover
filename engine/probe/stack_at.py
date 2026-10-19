"""What is around the stack address a stop depends on?

When a dispatch depends on `load64[A]` and nothing wrote `A`, three things are
possible and they need different answers: the engine's stack pointer has drifted
from the VM's, the store exists but was dropped, or the slot is genuinely
unwritten. This walks to a stop and prints the pointer state and the written map
around `A`, so the answer is visible rather than inferred.

Usage: python -m probe.stack_at <module> <root_va> <abs_addr> [--steps N] [--span N]
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
    steps, span = 400_000, 0x120
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--span="):
            span = int(x.split("=", 1)[1], 0)
    path, root, addr = args[0], int(args[1], 0), int(args[2], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, STACK_BASE)
    L = Lifter(img, m, a, vm.va, vm.end)

    ip, stop = root, None
    for i in range(steps):
        kind, payload = L.step(ip)
        if kind != "next":
            stop, at = payload, i
            break
        ip = payload

    print(f"stop            {stop.kind} at {stop.site:#x}")
    print(f"leaf address    {addr:#x}   STACK_BASE {STACK_BASE:#x}   "
          f"delta {addr - STACK_BASE:+#x}")
    rsp = a.as_const(m.reg("rsp", 64))
    rbp = a.as_const(m.reg("rbp", 64))
    for name in ("rsp", "rbp", "rcx", "rax", "rdi", "rsi", "rdx", "r8", "r9"):
        v = m.reg(name, 64)
        c = a.as_const(v)
        print(f"  {name:<4} {'%#x' % c if c is not None else render(a, v, 20)[:70]}")
    print()
    if rbp is not None:
        print(f"  rbp + 0xb28 = {rbp + 0xB28:#x}   (= STACK_BASE "
              f"{'yes' if rbp + 0xB28 == STACK_BASE else 'NO'})")
    print()
    lo, hi = addr - span, addr + span
    written = sorted(x for x in m.mem if lo <= x < hi)
    print(f"memory entries in [{lo:#x}, {hi:#x}): {len(written)}")
    if written:
        run_lo = written[0]
        prev = None
        for x in written:
            if prev is None or x != prev + 1:
                if prev is not None:
                    print(f"    ..{prev:#x}", end="")
                    print()
                print(f"    {x:#x}", end="")
            prev = x
        if prev is not None:
            print(f"    ..{prev:#x}")
    else:
        print("    NONE")
    print()
    top = sorted(x for x in m.mem if x >= STACK_BASE - 0x400)
    print(f"memory entries at or above STACK_BASE-0x400: {len(top)}")
    print(f"  highest: {hex(top[-1]) if top else '-'}   "
          f"lowest: {hex(top[0]) if top else '-'}")
    print(f"total concrete stores performed: {len(m.mem)}")
    print(f"stores dropped (symbolic address): {len(m.sym_stores)}")
    print()
    # An instruction the lifter boxed may well be the write that is missing: it
    # records the operands it could not model instead of performing them.
    print(f"boxed (unmodelled) instructions: {len(L.boxed)}")
    seen: dict[str, int] = {}
    for b in L.boxed:
        key = f"{b.text.split()[0] if b.text else '?'}"
        seen[key] = seen.get(key, 0) + 1
    for k, n in sorted(seen.items(), key=lambda kv: -kv[1])[:16]:
        print(f"  {n:>4}  {k}")
    print()
    print("boxed instructions with a memory operand (the candidates):")
    n = 0
    for b in L.boxed:
        if "[" in b.text:
            n += 1
            if n <= 20:
                print(f"  {b.site:#x}  {b.text}")
    print(f"  ... {n} in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
