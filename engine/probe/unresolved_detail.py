"""Per-block unresolved detail from the production pipeline, for one root.

`probe.cfg_run` builds a bare `Recoverer`, and the production path
(`core.devirt.devirtualize`) builds one with the guest census, the measured handler
cache and the load-folding hook. Those two configurations do not always agree, and
after the runtime-bound-call change `0x14015b1ba` is an instance: the bare recoverer
stops on the VM's guest-exit re-entry with a symbolic `rsp`, while the production
pipeline closes the root with 0 unresolved blocks. A disagreement like that has to be
*reported with both numbers*, not resolved by quoting one of them.

This probe prints, for one root: the unresolved blocks with their site and detail,
the terminator census, and the counters that say which rules fired.

Usage: python -m probe.unresolved_detail <module> <root_va> [--census] [--steps=N] [--blocks=N]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.devirt import devirtualize  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.peimg import PeImage  # noqa: E402

BUILT: list = []


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    steps, blocks, handlers = 400_000, 512, 900
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        elif x.startswith("--blocks="):
            blocks = int(x.split("=", 1)[1], 0)
        elif x.startswith("--handlers="):
            handlers = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    # `devirtualize` keeps its `Recoverer` local, so the lifter counters have to be
    # collected from the constructor rather than read off the result.
    real_init = Lifter.__init__

    def recording_init(self, *a, **kw):
        real_init(self, *a, **kw)
        BUILT.append(self)

    Lifter.__init__ = recording_init
    try:
        fn = devirtualize(img, vm, root, handlers=handlers, steps=steps, blocks=blocks)
    finally:
        Lifter.__init__ = real_init

    print(f"root              {root:#x}")
    print(f"blocks            {fn.blocks}")
    print(f"unresolved        {fn.unresolved}")
    print(f"instructions      {fn.instructions}")
    print(f"truncated         {fn.truncated}")
    print(f"recovery          {fn.recovery}")
    kinds: dict[str, int] = {}
    for b in fn.cfg.blocks.values():
        kinds[b.kind] = kinds.get(b.kind, 0) + 1
    print(f"terminators       {dict(sorted(kinds.items(), key=lambda kv: -kv[1]))}")
    print()
    absorbed: dict[int, int] = {}
    for L in BUILT:
        for slot, n in L.runtime_bound_calls.items():
            absorbed[slot] = absorbed.get(slot, 0) + n
    print(f"{len(BUILT)} lifter(s) built (cache walk + recovery)")
    print("calls absorbed by `Lifter._runtime_bound_slot`: "
          + (", ".join(f"{s:#x} x{n}" for s, n in sorted(absorbed.items()))
             or "NONE"))
    print()
    if not fn.unresolved:
        print("no unresolved block: nothing to explain on this root in this configuration")
    for key in fn.cfg.order:
        b = fn.cfg.blocks[key]
        if b.kind != "unresolved":
            continue
        print(f"  {key}")
        print(f"      detail    {b.detail}")
        print(f"      site      {b.site:#x}")
        if b.predicate:
            print(f"      predicate {b.predicate}")
        if b.retry:
            print(f"      retry     {b.retry}")
        print(f"      cost      {b.cost}   successors {[str(s) for s in b.succs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
