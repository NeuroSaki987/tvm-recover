"""Recover the CFG of every virtualized function in a module and summarise.

The per-root numbers are the deliverable's acceptance measurement: blocks
recovered, terminator mix, unresolved blocks, interpreted instructions. Reported
per root rather than aggregated, because ACE's modules disagree with each other
about which recovery rules suffice and an average would hide that.

Usage: python -m probe.cfg_corpus <module> [--blocks N] [--steps N] [--limit N]
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from core.explore import STACK_BASE, Recoverer  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.peimg import PeImage  # noqa: E402
from core.entries import vm_entries  # noqa: E402


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    blocks, steps, limit, listing = 512, 200_000, 0, "--list" in sys.argv
    if "--nosum" in sys.argv:
        # A/B control: the same recovery with native-call summaries off.
        Lifter.summaries_default = False
    if "--sum" in sys.argv:
        # The other half of the A/B: summaries are off by default, so the "on"
        # arm has to ask for them.
        Lifter.summaries_default = True
    for x in sys.argv[1:]:
        if x.startswith("--blocks="):
            blocks = int(x.split("=", 1)[1], 0)
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--limit="):
            limit = int(x.split("=", 1)[1], 0)
    path = args[0]
    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    roots = sorted({t for _, t in vm_entries(img)})
    if limit:
        roots = roots[:limit]

    print(f"module {path.split(chr(92))[-1]}  .tvm0 {vm.va:#x}..{vm.end:#x}  roots {len(roots)}")
    hdr = (f"{'root':>14} {'blocks':>7} {'unres':>6} {'forks':>6} {'instr':>9} "
           f"{'handlers':>9} {'ms':>7}  terminator mix")
    print(hdr)
    print("-" * len(hdr))

    tot_blocks = tot_unres = tot_instr = 0
    tot_sum_ok = tot_sum_steps = 0
    tot_rejects: dict[str, int] = {}
    closed = 0
    for root in roots:
        R = Recoverer(img, vm.va, vm.end, STACK_BASE, step_budget=steps, block_budget=blocks)
        t0 = time.perf_counter()
        cfg = R.recover(root)
        dt = (time.perf_counter() - t0) * 1000
        mix = ", ".join(f"{k} {v}" for k, v in cfg.summary().items())
        handlers = len({b.key.handler for b in cfg.blocks.values()})
        print(f"{root:>#14x} {len(cfg.blocks):>7} {cfg.unresolved:>6} {cfg.forks:>6} "
              f"{cfg.steps:>9} {handlers:>9} {dt:>7.0f}  {mix}")
        if cfg.unresolved == 0 and len(cfg.blocks) >= 2:
            closed += 1
        tot_blocks += len(cfg.blocks)
        tot_unres += cfg.unresolved
        tot_instr += cfg.steps
        tot_sum_ok += cfg.summary_ok
        tot_sum_steps += cfg.summary_steps
        for why, n in cfg.summary_rejects.items():
            tot_rejects[why] = tot_rejects.get(why, 0) + n
        if listing and cfg.unresolved:
            for k in cfg.order:
                b = cfg.blocks[k]
                if b.kind == "unresolved":
                    print(f"      {k}  {b.detail}")
                    if b.retry:
                        print(f"          precise pass: {b.retry}")
                    if b.predicate:
                        print(f"          [{b.predicate}]")

    print()
    print(f"TOTAL blocks {tot_blocks}  unresolved {tot_unres}  "
          f"instructions {tot_instr}")
    print(f"      roots that closed completely with more than one block: {closed}/{len(roots)}")
    print(f"      native-call summaries: {tot_sum_ok} callees interpreted to a value "
          f"in {tot_sum_steps} steps")
    if tot_rejects:
        for why, n in sorted(tot_rejects.items(), key=lambda kv: -kv[1]):
            print(f"        abandoned: {n:>5}  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
