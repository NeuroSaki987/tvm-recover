"""Which of the calls the lifter refused to model are VM trampolines?

A `call` to a `.text` address is not necessarily a call to a native routine. The
protector emits one-instruction trampolines -- `jmp rel32` into `.tvm0` -- and a
call through one is a call *into the virtual machine*. If that is what these
callees are, then "the return value is unknowable" is the wrong description
entirely, and the summary interpreter is being pointed at VM code.

Usage: python -m probe.thunk_census <module> [va ...]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.peimg import PeImage  # noqa: E402
from core.entries import vm_entries  # noqa: E402

CALLEES = (0x140011BA0, 0x14000D4E0, 0x14000D6E0, 0x140003290, 0x1400C03E0,
           0x1400DED98, 0x1400C05DC, 0x1400C05EC)


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    img = PeImage(args[0])
    vm = img.section_by_name(".tvm0")
    thunks = {s: t for s, t in vm_entries(img)}
    print(f"trampolines into .tvm0 found in the image: {len(thunks)}")
    for s in sorted(thunks):
        print(f"  thunk {s:#x} -> .tvm0 {thunks[s]:#x}")
    print()
    extra = [int(x, 0) for x in args[1:]]
    for callee in list(CALLEES) + extra:
        if callee in thunks:
            print(f"  callee {callee:#x}: THUNK into .tvm0 at {thunks[callee]:#x}")
            continue
        body = img.image_bytes(callee, 8) or b""
        kind = "?"
        if body[:2] == b"\xff\x25":
            # jmp qword ptr [rip+disp32] -- a jump through a pointer, which in
            # this image is how an import is reached from a direct call.
            disp = int.from_bytes(body[2:6], "little", signed=True)
            slot = callee + 6 + disp
            kind = (f"jmp-through-pointer to {slot:#x}"
                    f" = {img.imports.get(slot, 'not an IAT slot')}")
        elif body[:1] == b"\xe9":
            kind = "jmp rel32 (check the target)"
        else:
            kind = "real function"
        print(f"  callee {callee:#x}: {kind}   bytes {body.hex()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
