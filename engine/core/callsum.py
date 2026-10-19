"""What a native callee writes, as an object and a length.

`Lifter._guest_call` used to leave memory untouched, which made a read after a call return the
pre-call value. Replacing that with "forget everything from the pointer argument up to the top of the
stack window" was not a fix: `probe/callcost.py` shows the two intra-image callees on
`0x1401687d5` write **below** their pointer argument -- 773 observed stores between
`0x7ffffffef2d8` and `0x7ffffffef358`, every one of them under the argument -- while the region the
rule deleted (`0x7ffffffefd50` upward) is exactly the VM's own bookkeeping that the recovery needs.
The guess was over-approximate and under-approximate at the same time, and both halves cost.

So a callee's writes are described rather than bounded by a direction:

* **A named import**, from its signature. `_Out_` parameters are written, `_In_` ones are read, and
  the object's size is the structure's. This is a claim about the Win32 API, so it lives in one table
  with the reason for each entry.
* **An intra-image callee**, from what the summarizer *observed* it write, including a failed
  attempt: `Summary.stores` records every store the partial interpretation performed. Unexplored
  paths may write more, so the observed region is extended to the pointer argument -- and no further,
  because every observed store on this corpus is below it.
* **Anything else**, one page below the pointer: the stack direction, bounded, and stated as the
  fallback rather than dressed up as knowledge.

Every entry is falsifiable: the observed region comes from a measurement, and the import table says
which parameter and how many bytes.
"""

from __future__ import annotations

# How far below its pointer argument an unknown callee may write. A stack-shaped work area is the
# observed pattern; one page is the bound, and it is deliberately on the side the evidence supports.
UNKNOWN_WORK_BELOW = 0x1000

# `symbol -> (argument register written through, byte length)`, or None for a callee that writes no
# memory. Sizes are the Win32 structures the parameter points at:
#   Absolute SECURITY_DESCRIPTOR on x64: 40 bytes (four aligned pointers).
#   The 20-byte self-relative layout is not what these two APIs accept.
#   https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-security_descriptor
#   CRITICAL_SECTION     40 bytes on x64
#   lpEventAttributes    `_In_opt_` SECURITY_ATTRIBUTES: read, never written
#   GetTickCount         no parameters at all
#   GetPrivateProfileIntW reads two strings and a file name; writes nothing
IMPORT_WRITES: dict[str, tuple[str, int] | None] = {
    "ADVAPI32.dll!InitializeSecurityDescriptor": ("rcx", 0x28),
    "ADVAPI32.dll!SetSecurityDescriptorDacl": ("rcx", 0x28),
    "KERNEL32.dll!EnterCriticalSection": ("rcx", 0x28),
    "KERNEL32.dll!CreateEventW": None,
    "KERNEL32.dll!GetTickCount": None,
    "KERNEL32.dll!GetPrivateProfileIntW": None,
}

ARG_ORDER = ("rcx", "rdx", "r8", "r9")


def import_region(name: str, args) -> list[tuple[int, int]] | None:
    """`[(lo, hi)]` for the memory a named import writes, or None if the callee is not described.

    None and `[]` are different answers and the difference is load-bearing: `[]` means the signature
    says the callee writes nothing, None means this table has never seen it and the caller must fall
    back to the conservative rule.
    """
    if name not in IMPORT_WRITES:
        return None
    spec = IMPORT_WRITES[name]
    if spec is None:
        return []
    reg, length = spec
    if reg not in ARG_ORDER:
        return None
    value = args.get(reg)
    if value is None:
        return None
    return [(value, value + length)]
