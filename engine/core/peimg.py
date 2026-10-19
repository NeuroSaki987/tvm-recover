"""PE image access for TVM semantic recovery.

Deliberately independent of `pefile`: the recovery engine needs only four things
from a PE -- the section table (name, VA span, raw span, characteristics), the
import directory (slot VA -> module!symbol), the image base and the raw bytes by
VA -- and it needs them for both PE32+ and PE32, because a third of the ACE
`.tvm0` corpus ships as 32-bit images.

Nothing here knows about the VM. Layout questions belong to `vmctx.py`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# Section characteristics
IMAGE_SCN_CNT_CODE = 0x00000020
IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_READ = 0x40000000
IMAGE_SCN_MEM_WRITE = 0x80000000

# Data directories
DIR_EXCEPTION = 3
DIR_IMPORT = 1
DIR_IAT = 12


@dataclass(frozen=True)
class Section:
    name: str
    va: int  # absolute VA of the section start
    vsize: int  # VirtualSize
    raw: int  # PointerToRawData (file offset)
    rsize: int  # SizeOfRawData
    chars: int
    index: int

    @property
    def end(self) -> int:
        return self.va + max(self.vsize, self.rsize)

    @property
    def executable(self) -> bool:
        return bool(self.chars & IMAGE_SCN_MEM_EXECUTE)

    @property
    def writable(self) -> bool:
        return bool(self.chars & IMAGE_SCN_MEM_WRITE)

    @property
    def contains_code(self) -> bool:
        return bool(self.chars & (IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE))


class PeImage:
    """A PE parsed down to what a VM evaluator actually reads."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            self.data = fh.read()
        self._parse()

    # ------------------------------------------------------------------ parse

    def _u16(self, off: int) -> int:
        return struct.unpack_from("<H", self.data, off)[0]

    def _u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.data, off)[0]

    def _u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.data, off)[0]

    def _parse(self) -> None:
        d = self.data
        if d[:2] != b"MZ":
            raise ValueError(f"{self.path}: not a PE (no MZ)")
        e_lfanew = self._u32(0x3C)
        if d[e_lfanew : e_lfanew + 4] != b"PE\0\0":
            raise ValueError(f"{self.path}: not a PE (no PE signature)")
        coff = e_lfanew + 4
        self.machine = self._u16(coff)
        nsec = self._u16(coff + 2)
        size_opt = self._u16(coff + 16)
        opt = coff + 20

        magic = self._u16(opt)
        if magic == 0x20B:
            self.is64 = True
        elif magic == 0x10B:
            self.is64 = False
        else:
            raise ValueError(f"{self.path}: unknown optional header magic {magic:#x}")
        # Both optional-header layouts put these at the same offsets: they precede
        # the field whose width differs (ImageBase at 0x18/0x1C... see below).
        self.image_base = self._u64(opt + 0x18) if self.is64 else self._u32(opt + 0x1C)
        self.entry_point = self.image_base + self._u32(opt + 0x10)
        hdr_end = 0x70 if self.is64 else 0x60
        self.size_of_image = self._u32(opt + 0x38)
        self.section_alignment = self._u32(opt + 0x20)
        self.file_alignment = self._u32(opt + 0x24)
        n_dirs = self._u32(opt + (0x6C if self.is64 else 0x5C))
        dirs_off = opt + (0x70 if self.is64 else 0x60)
        self.directories: list[tuple[int, int]] = []
        for i in range(min(n_dirs, 16)):
            off = dirs_off + i * 8
            if off + 8 > opt + size_opt:
                break
            self.directories.append((self._u32(off), self._u32(off + 4)))
        # `hdr_end` already points past the data directories.
        del hdr_end

        sec_off = opt + size_opt
        self.sections: list[Section] = []
        for i in range(nsec):
            o = sec_off + i * 40
            raw_name = d[o : o + 8]
            name = raw_name.split(b"\0", 1)[0].decode("latin-1")
            vsize = self._u32(o + 8)
            va = self._u32(o + 12)
            rsize = self._u32(o + 16)
            roff = self._u32(o + 20)
            chars = self._u32(o + 36)
            self.sections.append(
                Section(
                    name=name,
                    va=self.image_base + va,
                    vsize=vsize,
                    raw=roff,
                    rsize=rsize,
                    chars=chars,
                    index=i,
                )
            )
        self._by_name = {s.name: s for s in self.sections}

        # Sorted section starts, for the covering-section binary search.
        self._starts = sorted((s.va, s) for s in self.sections)

        self._imports: dict[int, tuple[str, str]] | None = None
        self._pdata: list[tuple[int, int]] | None = None
        self._loader_bound: list[tuple[int, int]] | None = None

    # -------------------------------------------------------------- sections

    def section_by_name(self, name: str) -> Section | None:
        return self._by_name.get(name)

    def section_for_va(self, va: int) -> Section | None:
        """Covering section for `va`, or None when `va` is outside the image."""
        if not self._starts:
            return None
        lo, hi = 0, len(self._starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._starts[mid][0] <= va:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return None
        sec = self._starts[lo - 1][1]
        if va >= sec.end:
            return None
        return sec

    def in_section(self, name: str, va: int) -> bool:
        s = self._by_name.get(name)
        return s is not None and s.va <= va < s.end

    # ------------------------------------------------------------- raw bytes

    def _file_offset(self, va: int) -> int | None:
        sec = self.section_for_va(va)
        if sec is None:
            return None
        delta = va - sec.va
        if sec.rsize == 0:
            return None
        if delta >= sec.rsize:
            # VirtualSize beyond the raw data: zero-filled, not present in file.
            return None
        return sec.raw + delta

    def read_va(self, va: int, size: int) -> bytes | None:
        """Raw bytes at `va`, or None if any part is not file-backed."""
        off = self._file_offset(va)
        if off is None or off + size > len(self.data):
            return None
        return self.data[off : off + size]

    def image_u8(self, va: int) -> int | None:
        """One byte of the loaded image, as the loader would map it.

        Raw bytes where the file backs them, zero for the zero-filled tail of a
        section. This is what a `mov cx,[imm]` inside the VM actually reads, so
        it must not return None merely because the byte is past SizeOfRawData.
        """
        sec = self.section_for_va(va)
        if sec is None:
            return None
        delta = va - sec.va
        if delta < sec.rsize:
            off = sec.raw + delta
            if off < len(self.data):
                return self.data[off]
            return None
        return 0

    def image_load(self, va: int, size: int) -> int | None:
        """Little-endian integer built from `size` image bytes at `va`."""
        acc = 0
        for i in range(size):
            b = self.image_u8(va + i)
            if b is None:
                return None
            acc |= b << (8 * i)
        return acc

    def image_bytes(self, va: int, size: int) -> bytes | None:
        out = bytearray()
        for i in range(size):
            b = self.image_u8(va + i)
            if b is None:
                return None
            out.append(b)
        return bytes(out)

    # --------------------------------------------------------------- imports

    @property
    def imports(self) -> dict[int, tuple[str, str]]:
        """IAT slot VA -> (module, symbol). Empty on images without imports."""
        if self._imports is not None:
            return self._imports
        out: dict[int, tuple[str, str]] = {}
        if len(self.directories) <= DIR_IMPORT:
            self._imports = out
            return out
        rva, size = self.directories[DIR_IMPORT]
        if rva == 0 or size == 0:
            self._imports = out
            return out
        base = self.image_base
        off = self._file_offset(base + rva)
        if off is None:
            self._imports = out
            return out
        d = self.data
        i = 0
        ptr_size = 8 if self.is64 else 4
        ord_flag = 1 << (63 if self.is64 else 31)
        while i < 4096:
            ent = off + i * 20
            if ent + 20 > len(d):
                break
            oft = self._u32(ent)
            name_rva = self._u32(ent + 12)
            first = self._u32(ent + 16)
            if oft == 0 and name_rva == 0 and first == 0:
                break
            mod = self._read_cstr(self._file_offset(base + name_rva)) if name_rva else None
            thunk_rva = oft or first
            t_off = self._file_offset(base + thunk_rva)
            if t_off is not None:
                k = 0
                while k < 65536:
                    to = t_off + k * ptr_size
                    if to + ptr_size > len(d):
                        break
                    val = self._u64(to) if self.is64 else self._u32(to)
                    if val == 0:
                        break
                    if val & ord_flag:
                        sym = f"ordinal_{val & 0xFFFF}"
                    else:
                        nm_off = self._file_offset(base + (val & 0x7FFFFFFF))
                        sym = self._read_cstr(nm_off + 2) if nm_off is not None else None
                    slot = base + first + k * ptr_size
                    out[slot] = (mod or "?", sym or "?")
                    k += 1
            i += 1
        self._imports = out
        return out

    def _read_cstr(self, off: int | None) -> str | None:
        if off is None or off >= len(self.data):
            return None
        end = self.data.find(b"\0", off)
        if end < 0:
            return None
        return self.data[off:end].decode("latin-1")

    # ------------------------------------------------------- .pdata / unwind

    @property
    def pdata(self) -> list[tuple[int, int]]:
        """(begin_va, end_va) for every RUNTIME_FUNCTION, sorted by begin.

        Exposed because the VM section's bytecode is *not* code and the only
        trustworthy statement about where a real function starts in the host
        image comes from here. Empty for PE32 images, whose unwind format is
        different and which this engine does not need.
        """
        if self._pdata is not None:
            return self._pdata
        out: list[tuple[int, int]] = []
        if self.is64 and len(self.directories) > DIR_EXCEPTION:
            rva, size = self.directories[DIR_EXCEPTION]
            if rva and size:
                off = self._file_offset(self.image_base + rva)
                if off is not None:
                    base = self.image_base
                    for i in range(size // 12):
                        o = off + i * 12
                        if o + 12 > len(self.data):
                            break
                        b = self._u32(o)
                        e = self._u32(o + 4)
                        if b == 0 and e == 0:
                            continue
                        out.append((base + b, base + e))
        out.sort()
        self._pdata = out
        return out

    def pdata_body_all(self, va: int, byte: int) -> bool:
        """True when the .pdata record covering `va` is non-empty and all `byte`."""
        table = self.pdata
        if not table:
            return False
        lo, hi = 0, len(table)
        while lo < hi:
            mid = (lo + hi) // 2
            if table[mid][0] <= va:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return False
        begin, end = table[lo - 1]
        if not (begin <= va < end):
            return False
        body = self.read_va(begin, end - begin)
        if not body:
            return False
        return all(x == byte for x in body)

    # ---------------------------------------------------------- loader effects

    def is_bss(self, va: int) -> bool:
        """True when `va` is in a section's zero-initialized tail.

        Past `SizeOfRawData` the loader maps zeros, so the region *is* writable
        but its value at load time is still known. Refusing it along with the
        rest of a writable section would throw away a fact.
        """
        sec = self.section_for_va(va)
        if sec is None:
            return False
        return (va - sec.va) >= sec.rsize

    @property
    def loader_bound(self) -> list[tuple[int, int]]:
        """Ranges the loader rewrites: the IAT directory and every IAT slot.

        Kept separate from writability because an IAT slot need not live in a
        writable section, and it is overwritten regardless.
        """
        if self._loader_bound is not None:
            return self._loader_bound
        out: list[tuple[int, int]] = []
        if len(self.directories) > DIR_IAT:
            rva, size = self.directories[DIR_IAT]
            if rva and size:
                lo = self.image_base + rva
                out.append((lo, lo + size))
        for slot in self.imports:
            out.append((slot, slot + 8))
        out.sort()
        merged: list[tuple[int, int]] = []
        for lo, hi in out:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        self._loader_bound = merged
        return merged

    def is_loader_bound(self, va: int, size: int) -> bool:
        end = va + max(size, 1)
        return any(va < hi and lo < end for lo, hi in self.loader_bound)

    def foldable(self, va: int, size: int) -> bool:
        """Whether a read here may be folded to the value in the *file*.

        No, for anything the loader or the runtime writes before the virtualized
        code runs: a writable global may hold a placeholder on disk, and an IAT
        slot always does. Folding such a read produces a value that is correct
        for the file and wrong for the process -- and because the result is a
        plausible number, nothing downstream can tell.

        An address outside the image is not the image's business: the engine's
        own memory model covers it.
        """
        sec = self.section_for_va(va)
        if sec is None:
            return True
        if self.is_loader_bound(va, size):
            return False
        return (not sec.writable) or self.is_bss(va)

    def summary(self) -> str:
        kind = "PE32+" if self.is64 else "PE32"
        return (
            f"{self.path.split(chr(92))[-1]}: {kind} machine={self.machine:#x} "
            f"base={self.image_base:#x} entry={self.entry_point:#x} "
            f"sections={len(self.sections)} imports={len(self.imports)}"
        )


def image_base_of(path: str) -> int:
    return PeImage(path).image_base
