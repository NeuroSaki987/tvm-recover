"""Who calls this function? Two things make the obvious answer wrong.

The engine's own `PeImage.pdata` reads the right table -- the EXCEPTION data
directory, not the `.pdata` *section*, which on this corpus holds only 41-76 % of
the real entries because the protector relocates the directory into `.tvm0`. But
it returns `(begin, end)` pairs and stops there, and that is enough to lose a
call graph in two separate ways.

**1. PGO splits one function into chunks, and callers call the primary.**
Measured on `ACE-CORE202706.sys` (`8F35B7677E79`): 2654 RUNTIME_FUNCTION
records, **2131 of them (80 %) carry `UNW_FLAG_CHAININFO`** pointing at a parent.
A call site is attributed to the chunk it falls inside, so looking up callers of
a cold chunk's own start finds nothing and the function looks uncalled. The fix
is to follow the chain to the primary before searching.

**2. A tail call leaves no `call` site.**
An obfuscated driver reaches many helpers with `jmp rel32` only. A graph built
from `call` alone reports those helpers as uncalled. Both mnemonics are edges.

Measured effect of the two fixes on one root of that module, tracing up from the
physical-memory reader at `0x5CE0`:

    before:  0x5CE0 <- 0x5B4A <- (nothing)          chain dies at level 1
    after :  0x5CE0 <- 0x5B4A <- 0x5A40 <- 0x1220E3 <- 0xB9C0
                    <- 0x1198F <- 0x1167D <- 0x11470  chain dies at level 6

and what it dies on is worth reading: `0x11470` is a **primary** record (no
CHAININFO) in `.text` with zero `call`, zero `jmp` and no function pointer in
`.reloc` -- it is entered through a computed address, which is where static
caller tracing ends and something else has to take over.

Usage: python -m probe.caller_trace <module> <va> [--depth=N] [--max-sites=N]

`--depth` (default 4) is how many levels up to walk. `--max-sites` (default 6)
caps how many call sites are printed per function; the total is always printed.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from capstone import CS_ARCH_X86, CS_MODE_64, Cs  # noqa: E402
from capstone.x86 import X86_OP_IMM  # noqa: E402

from core.peimg import DIR_EXCEPTION, PeImage  # noqa: E402

# UNWIND_INFO.Flags occupies the top five bits of the first byte.
UNW_FLAG_CHAININFO = 0x4
# UNWIND_CODE is two bytes; the chained RUNTIME_FUNCTION follows the array,
# aligned to four bytes.
UNWIND_CODE_SIZE = 2


def _runtime_functions(img: PeImage) -> list[tuple[int, int, int]]:
    """(begin, end, unwind) for every record, as absolute VAs."""
    out: list[tuple[int, int, int]] = []
    if not img.is64 or len(img.directories) <= DIR_EXCEPTION:
        return out
    rva, size = img.directories[DIR_EXCEPTION]
    if not rva or not size:
        return out
    base = img.image_base
    raw = img.image_bytes(base + rva, size) or b""
    for i in range(len(raw) // 12):
        o = i * 12
        b = int.from_bytes(raw[o:o + 4], "little")
        e = int.from_bytes(raw[o + 4:o + 8], "little")
        u = int.from_bytes(raw[o + 8:o + 12], "little")
        if b == 0 and e == 0:
            continue
        out.append((base + b, base + e, u))
    out.sort()
    return out


def _chained_primary(img: PeImage, unwind_rva: int, depth: int = 0) -> int | None:
    """The primary record's begin VA that `unwind_rva` chains to, if any.

    `depth` bounds the walk: a malformed or adversarial chain must not spin.
    """
    if depth > 8 or not unwind_rva:
        return None
    head = img.image_bytes(img.image_base + unwind_rva, 4)
    if not head or len(head) < 4:
        return None
    flags = head[0] >> 3
    if not (flags & UNW_FLAG_CHAININFO):
        return None
    count = head[2]
    pos = unwind_rva + 4 + count * UNWIND_CODE_SIZE
    pos = (pos + 3) & ~3
    rec = img.image_bytes(img.image_base + pos, 12)
    if not rec or len(rec) < 12:
        return None
    begin = int.from_bytes(rec[0:4], "little")
    chained_unwind = int.from_bytes(rec[8:12], "little")
    if begin == 0:
        return None
    deeper = _chained_primary(img, chained_unwind, depth + 1)
    return deeper if deeper is not None else img.image_base + begin


class Functions:
    """RUNTIME_FUNCTION lookup, with CHAININFO folded in."""

    def __init__(self, img: PeImage):
        self.records = _runtime_functions(img)
        self._begins = [r[0] for r in self.records]
        self.primary_of: dict[int, int] = {}
        for begin, _end, unwind in self.records:
            p = _chained_primary(img, unwind)
            if p is not None and p != begin:
                self.primary_of[begin] = p

    def containing(self, va: int) -> tuple[int, int] | None:
        lo, hi = 0, len(self._begins) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            begin, end, _u = self.records[mid]
            if va < begin:
                hi = mid - 1
            elif va >= end:
                lo = mid + 1
            else:
                return begin, end
        return None

    def primary(self, va: int) -> int:
        """The primary start for `va`, following CHAININFO when the chunk has one."""
        c = self.containing(va)
        if c is None:
            return va
        return self.primary_of.get(c[0], c[0])


def _call_edges(img: PeImage) -> dict[int, list[int]]:
    """target VA -> call-site VAs, from `call rel32` **and** `jmp rel32`.

    Resynchronises byte by byte: a section that starts mid-instruction (the
    protector's own thunks do) otherwise stops the decoder at offset zero and
    the whole section yields nothing.
    """
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    edges: dict[int, list[int]] = {}
    for sec in img.sections:
        if not sec.executable:
            continue
        code = img.image_bytes(sec.va, max(sec.vsize, sec.rsize))
        if not code:
            continue
        o = 0
        n = len(code)
        while o < n:
            ins = None
            for cand in md.disasm(code[o:o + 15], sec.va + o, count=1):
                ins = cand
            if ins is None:
                o += 1
                continue
            if ins.mnemonic in ("call", "jmp") and ins.operands:
                op = ins.operands[0]
                if op.type == X86_OP_IMM:
                    edges.setdefault(op.imm, []).append(ins.address)
            o += ins.size
    return edges


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    opts = {x.split("=")[0]: x.split("=", 1)[1]
            for x in sys.argv[1:] if x.startswith("--") and "=" in x}
    if len(args) < 2:
        print(__doc__)
        return 2
    depth = int(opts.get("--depth", "4"), 0)
    max_sites = int(opts.get("--max-sites", "6"), 0)

    img = PeImage(args[0])
    root = int(args[1], 0)
    # Accept an RVA as well as a VA: the corpus's reports quote RVAs, the engine
    # works in VAs, and silently reading a header because the caller passed
    # `0x5ce0` instead of `0x140005ce0` is the kind of quiet wrong answer this
    # workspace keeps paying for.
    if root < img.image_base:
        print(f"  (interpreting {root:#x} as an RVA; VA {img.image_base + root:#x})")
        root += img.image_base
    funcs = Functions(img)
    edges = _call_edges(img)

    print(f"{img.path}")
    print(f"  RUNTIME_FUNCTION records {len(funcs.records)}, "
          f"of which {len(funcs.primary_of)} chain to a primary "
          f"({100 * len(funcs.primary_of) / max(len(funcs.records), 1):.0f} %)")
    c = funcs.containing(root)
    print(f"  start {root:#x} is in "
          f"{('the record %#x..%#x' % c) if c else 'no record'}")
    if c and funcs.primary_of.get(c[0]) is not None:
        print(f"  that chunk chains to the primary {funcs.primary_of[c[0]]:#x}")
    print()

    seen: set[tuple[int, int]] = set()
    frontier: list[tuple[int, int]] = [(root, 0)]
    dead_ends: list[int] = []
    while frontier:
        target, level = frontier.pop(0)
        if (target, level) in seen or level > depth:
            continue
        seen.add((target, level))
        here = funcs.containing(target)
        # The edge search must use the primary: a caller calls the primary, not
        # a PGO cold chunk's own start.
        start = funcs.primary(target)
        sites = edges.get(start, [])
        pad = "  " * (level + 1)
        label = f"{start:#x}"
        if here is not None:
            label += f" (record {here[0]:#x}..{here[1]:#x})"
        if start != (here[0] if here else target):
            label += " [chaininfo -> primary]"
        if not sites:
            print(f"{pad}{label} <- no call and no jmp in any executable section")
            dead_ends.append(start)
            continue
        print(f"{pad}{label} <- {len(sites)} edge(s)")
        for site in sorted(set(sites))[:max_sites]:
            sc = funcs.containing(site)
            where = f"inside {sc[0]:#x}" if sc else "outside any record"
            print(f"{pad}    {site:#x}  ({where})")
            if level < depth and sc is not None:
                frontier.append((sc[0], level + 1))

    if dead_ends:
        print()
        print("Where the trace stops (statically unreachable by call or jmp):")
        for va in dead_ends:
            body = img.image_bytes(va, 8) or b""
            print(f"  {va:#x}  first bytes {body.hex()}")
        print("  A record here is entered through a computed address -- a table, a")
        print("  callback registration, or the VM dispatcher -- and static caller")
        print("  tracing has nothing left to follow. That is a boundary, not a bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
