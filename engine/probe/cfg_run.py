"""Recover the CFG of one virtualized function and report it the way `cfg` does.

The point of this probe is a like-for-like comparison with the reference's `cfg`
command on the same root: blocks recovered, terminators by kind, unresolved
blocks, and interpreted instructions. Those four numbers are what "semantic
recovery" means operationally, and they are the numbers the archived structural
engine could not produce at all.

Usage: python -m probe.cfg_run <module> <root_va> [--blocks N] [--steps N] [--list]
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from core.explore import STACK_BASE, Cfg, Recoverer  # noqa: E402
from core.peimg import PeImage  # noqa: E402


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    blocks, steps, listing = 512, 200_000, "--list" in sys.argv
    for x in sys.argv[1:]:
        if x.startswith("--blocks="):
            blocks = int(x.split("=", 1)[1], 0)
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    if vm is None:
        print("no .tvm0")
        return 2

    R = Recoverer(img, vm.va, vm.end, STACK_BASE, step_budget=steps, block_budget=blocks)
    t0 = time.perf_counter()
    cfg = R.recover(root)
    dt = time.perf_counter() - t0

    print(f"module            {path}")
    print(f"root              {root:#x}")
    print(f"vip slot          {('rbp+%#x' % R.vip_slot) if R.vip_slot is not None else 'not found'}"
          f"   (from {R.vip_samples} slots holding .tvm0 pointers)")
    print(f"unwind slot       {('0x%x' % R.unwind_slot) if R.unwind_slot is not None else 'none'}")
    print()
    print(f"recovered         {len(cfg.blocks)} blocks in {dt * 1000:.0f} ms "
          f"({cfg.steps} VM instructions interpreted)")
    print(f"forks             {cfg.forks}")
    print(f"unresolved        {cfg.unresolved}")
    print(f"terminators       {', '.join(f'{v} {k}' for k, v in cfg.summary().items())}")
    print(f"distinct handlers {len({b.key.handler for b in cfg.blocks.values()})}")
    print(f"distinct vips     {len({b.key.vip for b in cfg.blocks.values()})}")

    if cfg.unresolved:
        print()
        print("unresolved blocks:")
        for k in cfg.order:
            b = cfg.blocks[k]
            if b.kind == "unresolved":
                print(f"  {k}  {b.detail}")
                if b.retry:
                    print(f"      precise pass: {b.retry}")
                if b.predicate:
                    print(f"      predicate: {b.predicate}")

    if listing:
        print()
        for k in cfg.order:
            b = cfg.blocks[k]
            print(f"  {k}  cost {b.cost:<6} events {b.events:<5} {b.kind:<10} "
                  f"-> {', '.join(str(s) for s in b.succs) or '-'}")
            if b.predicate:
                print(f"      [{b.predicate}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
