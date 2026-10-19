"""Disassemble a native routine, to see what a summary would have to interpret.

Usage: python -m probe.dis <module> <va> [--n N]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from capstone import CS_ARCH_X86, CS_MODE_64, Cs  # noqa: E402

from core.peimg import PeImage  # noqa: E402


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    n = 60
    for x in sys.argv[1:]:
        if x.startswith("--n="):
            n = int(x.split("=", 1)[1], 0)
    img = PeImage(args[0])
    va = int(args[1], 0)
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = False
    body = img.image_bytes(va, n * 8)
    if body is None:
        print("not file-backed")
        return 2
    for i, insn in enumerate(md.disasm(body, va)):
        if i >= n:
            break
        print(f"{insn.address:#012x}  {insn.bytes.hex():<20} {insn.mnemonic:<8} {insn.op_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
