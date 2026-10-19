"""Who wrote the byte the engine trusted?

Used when a precise pass produces a provably impossible result. The engine
believes a scratch byte because something stored it; this prints every store that
touched the window around it, with the site and whether the stored value was a
constant or an expression. A constant that should have been an expression is a
lifter bug and shows up here immediately.

Usage: python -m probe.who_wrote <module> <root_va> <offset_from_rbp> [--steps N]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena, render  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    steps = 400_000
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
    path, root, off = args[0], int(args[1], 0), int(args[2], 0)
    absolute = None
    for x in sys.argv[1:]:
        if x.startswith("--abs="):
            absolute = int(x.split("=", 1)[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, 0x7FFF_FFFE_FDA0)
    L = Lifter(img, m, a, vm.va, vm.end)

    log: list[tuple[int, int, int, str, str]] = []
    orig = Machine.store

    def store(self, addr, value, nbytes, site):
        orig(self, addr, value, nbytes, site)
        c = self.a.as_const(addr)
        # The probe address is only known once the run has stopped, so record
        # every store outside the image and filter afterwards. Buffering the
        # whole log is what makes the answer trustworthy: a filtered log that
        # silently records nothing looks identical to "nothing was written".
        if c is not None and c >= 0x1000_0000_0000:
            isc = self.a.as_const(value)
            log.append((site, c, nbytes,
                        f"{isc:#x}" if isc is not None else "EXPR",
                        render(self.a, value)[:70]))

    Machine.store = store
    ip, prev_rbp = root, None
    stop = None
    for _ in range(steps):
        rbp = a.as_const(m.reg("rbp", 64))
        if rbp is not None:
            prev_rbp = rbp
        k, p = L.step(ip)
        if k != "next":
            stop = p
            break
        ip = p

    rbp = a.as_const(m.reg("rbp", 64)) or prev_rbp
    probe = absolute if absolute is not None else rbp + off
    print(f"stop            {stop}")
    print(f"rbp             {rbp:#x}")
    print(f"probe address   {probe:#x}")
    print(f"memory image    {[m.mem.get(probe + j) for j in range(8)]}")
    print(f"stores outside the image during the whole run: {len(log)}")
    hits = [e for e in log if probe - 8 < e[1] < probe + 16]
    print(f"stores touching the surrounding 24 bytes: {len(hits)}")
    for e in hits:
        print(f"   site {e[0]:#x}  addr {e[1]:#x}  {e[2]}B  value {e[3]}  {e[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
