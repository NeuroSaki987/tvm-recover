"""Read and *edit* a PE image: append sections, move the entry point, rebuild the relocation table.

`engine/core/peimg.py` is the read side the recovery engine uses, and it deliberately stops at what
the VM evaluator needs. `write-devirt` needs the other half -- the fields a loader acts on -- and it
needs them written back without disturbing anything it did not change. So this module parses the same
headers into named fields and offers exactly four mutations:

    append_section(name, data, characteristics)   -> the RVA the section was placed at
    set_entry(rva)                                -> AddressOfEntryPoint
    set_dir(index, rva, size)                     -> one data directory
    set_bytes(rva, data)                          -> bytes at an RVA, in any section

and one derived one, `rebuild_relocations`, which is the only mutation that can be got wrong in a way
the loader will not notice until the image happens to be loaded at a different base.

Three properties are worth stating because they are what the tests check:

**Everything is recomputed from the file, never from a remembered constant.** Section RVAs come from
the section table, raw offsets from the current end of the file, and the checksum is recomputed over
the finished bytes -- an image whose stored checksum does not match its contents is the shape a
tooling bug leaves behind.

**A section header that does not fit is refused, not shifted.** Adding a section needs 40 bytes of
header space after the existing table, *below* `SizeOfHeaders` (the first section's raw data starts
there). When there is not enough, the honest answer is "this image needs its raw data moved and this
tool will not do that silently", because moving raw data means rewriting every section's
PointerToRawData and every file offset that refers to them (the debug directory's stamps, the
certificate table) -- each of which is a chance to produce an image that loads on one build of
Windows and not another.

**The certificate table is a file offset, not an RVA, and it lives at the end of the file.** Appending
raw data after it would leave the file's tail pointing into the middle of the image, so the default is
to strip it and say so: the signature cannot survive a modified image in any case, and a stale
signature is worse than none, because tooling reports "signed by X" for bytes X never signed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SCN_CNT_CODE = 0x00000020
IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
IMAGE_SCN_MEM_DISCARDABLE = 0x02000000
IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_READ = 0x40000000
IMAGE_SCN_MEM_WRITE = 0x80000000

DIR_EXPORT, DIR_IMPORT, DIR_RESOURCE, DIR_EXCEPTION, DIR_SECURITY = 0, 1, 2, 3, 4
DIR_BASERELOC, DIR_DEBUG, DIR_TLS, DIR_LOAD_CONFIG = 5, 6, 9, 10
DIR_IAT = 12

#: `IMAGE_DEBUG_TYPE_CODEVIEW`. The entry of this kind is the one that names a PDB, so it is the one
#: `peinject.stamp_codeview` repoints at the record inside `.neko`.
DEBUG_TYPE_CODEVIEW = 2

#: Base relocation types this module writes. `DIR64` is the x64 absolute fixup; `ABSOLUTE` is the
#: padding type the loader skips, and it is what a block's odd trailing entry uses.
REL_ABSOLUTE = 0
REL_DIR64 = 10


class PeError(ValueError):
    """The image is not one this module will edit, and the reason says which check refused it."""


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a if a else n


def pe_checksum(data: bytes, field_off: int) -> int:
    """The `CheckSum` field's algorithm, over `data` with the field itself treated as zero.

    A 16-bit one's-complement sum of the file, folded, plus the file length. Verified against the
    unmodified host image (its stored value is reproduced), which is the only way to know that an
    implementation of this is right -- a wrong one still produces a plausible-looking number.
    """
    total = 0
    n = len(data) & ~1
    i = 0
    while i < n:
        if field_off <= i < field_off + 4:
            word = 0
        else:
            word = data[i] | (data[i + 1] << 8)
        total += word
        total = (total & 0xFFFF) + (total >> 16)
        i += 2
    if len(data) & 1:
        total += data[-1]
        total = (total & 0xFFFF) + (total >> 16)
    total = (total & 0xFFFF) + (total >> 16)
    return (total + len(data)) & 0xFFFFFFFF


@dataclass
class Section:
    index: int
    name: str
    rva: int
    vsize: int
    raw: int
    rsize: int
    chars: int
    header_off: int

    @property
    def end(self) -> int:
        """First RVA past the section, using the same rule the loader uses to size it."""
        return self.rva + max(self.vsize, 1)

    @property
    def image_span(self) -> int:
        return max(self.vsize, self.rsize)

    @property
    def executable(self) -> bool:
        return bool(self.chars & IMAGE_SCN_MEM_EXECUTE)

    @property
    def writable(self) -> bool:
        return bool(self.chars & IMAGE_SCN_MEM_WRITE)

    def describe(self) -> str:
        return (f'{self.name:<8} rva {self.rva:#08x} vsize {self.vsize:#08x} '
                f'raw {self.raw:#08x} rsize {self.rsize:#08x} chars {self.chars:#010x}')


class Pe:
    """A PE image parsed down to the fields an editor or a loader acts on."""

    def __init__(self, data: bytes, label: str = '<memory>'):
        self.data = bytearray(data)
        self.label = label
        self._parse()

    @classmethod
    def from_path(cls, path: str | Path) -> 'Pe':
        p = Path(path)
        return cls(p.read_bytes(), str(p))

    # ------------------------------------------------------------------ parse
    def _u16(self, off: int) -> int:
        return struct.unpack_from('<H', self.data, off)[0]

    def _u32(self, off: int) -> int:
        return struct.unpack_from('<I', self.data, off)[0]

    def _u64(self, off: int) -> int:
        return struct.unpack_from('<Q', self.data, off)[0]

    def _parse(self) -> None:
        d = self.data
        if len(d) < 0x40 or d[:2] != b'MZ':
            raise PeError(f'{self.label}: not a PE image (no MZ at 0)')
        self.e_lfanew = self._u32(0x3C)
        if d[self.e_lfanew:self.e_lfanew + 4] != b'PE\0\0':
            raise PeError(f'{self.label}: no PE signature at e_lfanew={self.e_lfanew:#x}')
        self.coff = self.e_lfanew + 4
        self.machine = self._u16(self.coff)
        self.nsec = self._u16(self.coff + 2)
        self.timestamp = self._u32(self.coff + 4)
        self.size_opt = self._u16(self.coff + 16)
        self.file_chars = self._u16(self.coff + 18)
        self.opt = self.coff + 20
        self.magic = self._u16(self.opt)
        if self.magic == 0x20B:
            self.is64 = True
        elif self.magic == 0x10B:
            self.is64 = False
        else:
            raise PeError(f'{self.label}: optional header magic {self.magic:#x} is neither PE32+ '
                          'nor PE32')
        self.entry_rva = self._u32(self.opt + 0x10)
        self.base_of_code = self._u32(self.opt + 0x14)
        self.image_base = self._u64(self.opt + 0x18) if self.is64 else self._u32(self.opt + 0x1C)
        self.section_alignment = self._u32(self.opt + 0x20)
        self.file_alignment = self._u32(self.opt + 0x24)
        self.size_of_code = self._u32(self.opt + 0x04)
        self.size_of_init = self._u32(self.opt + 0x08)
        self.size_of_image = self._u32(self.opt + 0x38)
        self.size_of_headers = self._u32(self.opt + 0x3C)
        self.checksum_off = self.opt + 0x40
        self.checksum = self._u32(self.checksum_off)
        self.subsystem = self._u16(self.opt + 0x5C)
        self.dll_chars = self._u16(self.opt + 0x5E)
        self.nrva_off = self.opt + (0x6C if self.is64 else 0x5C)
        self.dirs_off = self.opt + (0x70 if self.is64 else 0x60)
        n_rva = self._u32(self.nrva_off)
        self.n_dirs = min(n_rva, 16)
        self.dirs: list[tuple[int, int]] = []
        for i in range(self.n_dirs):
            off = self.dirs_off + i * 8
            if off + 8 > self.opt + self.size_opt:
                self.dirs.append((0, 0))
                continue
            self.dirs.append((self._u32(off), self._u32(off + 4)))

        self.sections: list[Section] = []
        sec_tab = self.opt + self.size_opt
        for i in range(self.nsec):
            o = sec_tab + i * 40
            if o + 40 > len(d):
                raise PeError(f'{self.label}: section table runs past end of file')
            name = d[o:o + 8].split(b'\0', 1)[0].decode('latin-1')
            self.sections.append(Section(
                index=i, name=name, rva=self._u32(o + 12), vsize=self._u32(o + 8),
                raw=self._u32(o + 20), rsize=self._u32(o + 16), chars=self._u32(o + 36),
                header_off=o))
        self.sec_table_off = sec_tab

    # ---------------------------------------------------------------- queries
    def section(self, name: str) -> Section | None:
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def sections_named(self, name: str) -> list[Section]:
        return [s for s in self.sections if s.name == name]

    def section_at(self, rva: int) -> Section | None:
        for s in self.sections:
            if s.rva <= rva < s.end:
                return s
        return None

    def rva_to_offset(self, rva: int) -> int | None:
        s = self.section_at(rva)
        if s is None:
            return None
        delta = rva - s.rva
        if delta >= s.rsize:
            return None
        off = s.raw + delta
        return off if off < len(self.data) else None

    def image_bytes(self, rva: int, size: int) -> bytes:
        """The bytes the loader maps at `rva`, zero where the file does not back them."""
        s = self.section_at(rva)
        if s is None:
            raise PeError(f'{self.label}: rva {rva:#x} is outside every section')
        delta = rva - s.rva
        if delta + size > s.image_span:
            raise PeError(f'{self.label}: rva {rva:#x}+{size} runs past section {s.name}')
        out = bytearray()
        for i in range(size):
            off = s.raw + delta + i if (delta + i) < s.rsize else None
            if off is not None and off < len(self.data):
                out.append(self.data[off])
            else:
                out.append(0)
        return bytes(out)

    def end_of_image(self) -> int:
        return max((s.end for s in self.sections), default=self.size_of_headers)

    def end_of_file(self) -> int:
        return max((s.raw + s.rsize for s in self.sections), default=self.size_of_headers)

    @property
    def has_relocations(self) -> bool:
        return (len(self.dirs) > DIR_BASERELOC and self.dirs[DIR_BASERELOC][0] != 0
                and self.dirs[DIR_BASERELOC][1] != 0)

    def summary(self) -> str:
        kind = 'PE32+' if self.is64 else 'PE32'
        return (f'{self.label}: {kind} machine={self.machine:#x} base={self.image_base:#x} '
                f'entry={self.entry_rva:#x} sections={len(self.sections)} '
                f'image={self.size_of_image:#x} headers={self.size_of_headers:#x} '
                f'relocs={"yes" if self.has_relocations else "no"}')


def read_relocation_blocks(pe: Pe) -> list[bytes]:
    """The existing base relocation blocks, verbatim, in the order the loader reads them."""
    if not pe.has_relocations:
        return []
    rva, size = pe.dirs[DIR_BASERELOC]
    blob = pe.image_bytes(rva, size)
    out: list[bytes] = []
    i = 0
    while i + 8 <= len(blob):
        page, blk = struct.unpack_from('<II', blob, i)
        if blk < 8 or i + blk > len(blob):
            raise PeError(f'malformed relocation block at +{i:#x}: size {blk}')
        if page == 0 and blk == 0:
            break
        out.append(blob[i:i + blk])
        i += blk
    return out


@dataclass
class Added:
    """One queued section, before it is written."""
    name: str
    rva: int
    data: bytearray
    chars: int
    raw: int = 0


@dataclass
class Patch:
    """A queued edit of an existing image. `finish()` produces the new file bytes."""

    pe: Pe
    added: list[Added] = field(default_factory=list)
    entry_rva: int | None = None
    stripped_certificate: tuple[int, int] | None = None
    notes: list[str] = field(default_factory=list)
    #: `(pad, sections)` when `grow_headers` moved the raw data, else None.
    header_growth: tuple[int, int] | None = None

    # ------------------------------------------------------------------ add
    def append_section(self, name: str, data: bytes, chars: int) -> int:
        """Queue a section at the end of the image and return the RVA it will occupy.

        Placement accounts for sections queued earlier in the same patch, not only for the sections
        already in the file: without that a second append would be given the same RVA as the first,
        and both would be written.
        """
        if len(name) > 8:
            raise PeError(f'section name {name!r} is longer than the 8-byte field')
        pe = self.pe
        span = align_up(max(len(data), 1), pe.section_alignment)
        last_rva = pe.end_of_image()
        for q in self.added:
            last_rva = max(last_rva, q.rva + align_up(max(len(q.data), 1), pe.section_alignment))
        rva = align_up(last_rva, pe.section_alignment)
        for other in list(pe.sections) + [Section(index=-1, name=q.name, rva=q.rva, vsize=len(q.data),
                                                  raw=0, rsize=0, chars=q.chars, header_off=0)
                                          for q in self.added]:
            if other.rva < rva + span and rva < other.rva + max(other.vsize, other.rsize, 1):
                raise PeError(f'new section {name!r} at {rva:#x} would overlap {other.name} at '
                              f'{other.rva:#x}')
        self.added.append(Added(name=name, rva=rva, data=bytearray(data), chars=chars))
        return rva

    def rva_of_new(self, name: str) -> int:
        for a in self.added:
            if a.name == name:
                return a.rva
        raise KeyError(name)

    # --------------------------------------------------------------- mutate
    def set_entry(self, rva: int) -> None:
        self.entry_rva = rva

    def set_bytes(self, rva: int, data: bytes) -> None:
        """Write bytes into a queued section (or an existing one) at `rva`."""
        for a in self.added:
            if a.rva <= rva and rva + len(data) <= a.rva + len(a.data):
                off = rva - a.rva
                a.data[off:off + len(data)] = data
                return
        s = self.pe.section_at(rva)
        if s is None:
            raise PeError(f'set_bytes: rva {rva:#x} is outside every section')
        off = self.pe.rva_to_offset(rva)
        if off is None:
            raise PeError(f'set_bytes: rva {rva:#x} is not file-backed (past SizeOfRawData)')
        self.pe.data[off:off + len(data)] = data
        self.notes.append(f'patched {len(data)} byte(s) at rva {rva:#x} in {s.name}')

    def strip_certificate(self) -> None:
        """Drop the Authenticode blob and zero the security directory entry."""
        pe = self.pe
        if len(pe.dirs) <= DIR_SECURITY:
            return
        off, size = pe.dirs[DIR_SECURITY]
        if not off or not size:
            return
        if off + size > len(pe.data):
            raise PeError(f'certificate table at file offset {off:#x}+{size:#x} runs past EOF')
        self.stripped_certificate = (off, size)
        del pe.data[off:off + size]
        pe.dirs[DIR_SECURITY] = (0, 0)
        self.notes.append(f'stripped the {size} byte certificate table at file offset {off:#x} '
                          '(a modified image cannot keep a valid signature)')

    def set_dir(self, index: int, rva: int, size: int) -> None:
        while len(self.pe.dirs) <= index:
            self.pe.dirs.append((0, 0))
            self.pe.n_dirs = len(self.pe.dirs)
        self.pe.dirs[index] = (rva, size)

    # ------------------------------------------------------------- header space
    def header_room(self) -> int:
        """Bytes left between the end of the section table and `SizeOfHeaders`."""
        pe = self.pe
        return pe.size_of_headers - (pe.sec_table_off + 40 * (pe.nsec + len(self.added)))

    def grow_headers(self, sections: int = 1) -> int:
        """Insert `FileAlignment`-aligned padding so `sections` more headers fit. Returns the pad.

        **A kernel image is tight on header space.** `ACE-CORE.sys3` has 11 section headers ending at
        `0x3E8` with `SizeOfHeaders = 0x400`: 24 bytes, and one more header needs 40. The only way to
        add a section without reusing someone else's header is to raise `SizeOfHeaders` and move every
        section's raw data down, which is a *bounded, enumerable* edit rather than a rewrite. What
        moves, exhaustively:

          * every section's `PointerToRawData` (+pad),
          * `SizeOfHeaders` (+pad, staying FileAlignment-aligned),
          * the certificate table's entry, which is a **file offset**, not an RVA (+pad),
          * every debug-directory entry's `PointerToRawData`, also a file offset (+pad).

        RVAs, virtual sizes and every other field are untouched, so the image maps exactly as before
        from a file that is `pad` bytes longer. `peinject.verify()` re-parses the result and checks
        all four lists, so "raw data moved" is a measurement there rather than a claim here. Called
        explicitly and reported by the caller: a silent shift is what this module refuses to do.
        """
        pe = self.pe
        have = pe.size_of_headers - (pe.sec_table_off + 40 * (pe.nsec + len(self.added)))
        need = 40 * sections - have
        if need <= 0:
            return 0
        pad = align_up(need, pe.file_alignment)
        at = pe.size_of_headers
        if min(s.raw for s in pe.sections) < at:
            raise PeError(f'the first section\'s raw data starts at '
                          f'{min(s.raw for s in pe.sections):#x}, below SizeOfHeaders {at:#x}: '
                          'growing the headers would overwrite it')
        d = pe.data
        d[at:at] = bytes(pad)
        for s in pe.sections:
            if s.raw >= at:
                s.raw += pad
                struct.pack_into('<I', d, s.header_off + 20, s.raw)
        pe.size_of_headers += pad
        struct.pack_into('<I', d, pe.opt + 0x3C, pe.size_of_headers)
        # The certificate table is a file offset.
        if len(pe.dirs) > DIR_SECURITY:
            off, size = pe.dirs[DIR_SECURITY]
            if off >= at and size:
                pe.dirs[DIR_SECURITY] = (off + pad, size)
                self.notes.append(f'certificate table moved with the raw data: file offset '
                                  f'{off:#x} -> {off + pad:#x}')
        # Debug directory entries carry a file offset too.
        if len(pe.dirs) > DIR_DEBUG and pe.dirs[DIR_DEBUG][0] and pe.dirs[DIR_DEBUG][1]:
            rva, size = pe.dirs[DIR_DEBUG]
            for i in range(size // 28):
                entry = rva + i * 28
                off = pe.rva_to_offset(entry)
                if off is None:
                    continue
                ptr = struct.unpack_from('<I', d, off + 24)[0]
                if ptr >= at:
                    struct.pack_into('<I', d, off + 24, ptr + pad)
        self.header_growth = (pad, sections)
        self.notes.append(f'grew the header block by {pad} byte(s) at file offset {at:#x} so '
                          f'{sections} more section header(s) fit; every PointerToRawData, the '
                          'certificate file offset and every debug entry pointer moved with it')
        return pad

    # ------------------------------------------------------ base relocations
    def relocation_blocks(self) -> list[bytes]:
        """The existing base relocation blocks, verbatim, in the order the loader reads them."""
        return read_relocation_blocks(self.pe)

    def add_relocations(self, sites: list[int], blocks: list[bytes] | None = None) -> bytes:
        """DIR64 fixups for `sites` (RVAs), appended to the existing blocks as one blob.

        Blocks are keyed by 4 KiB page and the loader requires them sorted and non-overlapping, so
        the sites are grouped by page and the pages merged with anything already present.
        """
        base_blocks = self.relocation_blocks() if blocks is None else list(blocks)
        by_page: dict[int, list[int]] = {}
        for raw in base_blocks:
            page = struct.unpack_from('<I', raw, 0)[0]
            body = raw[8:]
            by_page.setdefault(page, [])
            for k in range(0, len(body) - 1, 2):
                ent = struct.unpack_from('<H', body, k)[0]
                by_page[page].append(ent)
        for rva in sites:
            page = rva & ~0xFFF
            by_page.setdefault(page, []).append(rva & 0xFFF)
        out = bytearray()
        for page in sorted(by_page):
            ents = sorted(set(by_page[page]))
            if not ents:
                continue
            while len(ents) % 2:
                ents.append(0)  # type ABSOLUTE pads an odd count
            body = bytearray()
            for e in ents:
                body += struct.pack('<H', (REL_ABSOLUTE << 12) if e == 0 else
                                    ((REL_DIR64 << 12) | (e & 0xFFF)))
            out += struct.pack('<II', page, 8 + len(body)) + body
        return bytes(out)

    # ----------------------------------------------------------------- write
    def finish(self) -> bytes:
        """Apply the queued edits and return the new file bytes.

        Steps 4-6 (sizes, entry point, directories, checksum) run even when no section was added:
        an edit that only strips a certificate or moves the entry point still has to be written, and
        a `finish()` that silently returned the original bytes would make those edits look applied.
        """
        pe = self.pe
        d = pe.data
        added = self.added
        body = bytearray(d)

        if added:
            # 1. header space for the new section headers
            need = 40 * len(added)
            have = pe.size_of_headers - (pe.sec_table_off + 40 * pe.nsec)
            if need > have:
                raise PeError(
                    f'only {have} byte(s) of header space after {pe.nsec} section header(s), '
                    f'{need} needed for {len(added)} new section(s): this image needs its raw data '
                    'moved and this tool will not do that silently (see the module docstring)')

            # 2. raw data, aligned to FileAlignment, after everything already in the file
            raw = align_up(max(pe.end_of_file(), len(d)), pe.file_alignment)
            for a in added:
                a.raw = raw
                span = align_up(len(a.data), pe.file_alignment)
                if len(body) < raw:
                    body += bytes(raw - len(body))
                body += bytes(a.data) + bytes(span - len(a.data))
                raw += span

            # 3. section headers
            struct.pack_into('<H', body, pe.coff + 2, pe.nsec + len(added))
            for i, a in enumerate(added):
                o = pe.sec_table_off + (pe.nsec + i) * 40
                if o + 40 > len(body):
                    raise PeError('section header lands past end of file')
                name = a.name.encode('latin-1')[:8].ljust(8, b'\0')
                struct.pack_into('<8sIIIIIIHHI', body, o, name, len(a.data), a.rva,
                                 align_up(len(a.data), pe.file_alignment), a.raw, 0, 0, 0, 0,
                                 a.chars)

        # 4. sizes
        size_of_image = pe.size_of_image
        for a in added:
            size_of_image = max(size_of_image,
                                align_up(a.rva + len(a.data), pe.section_alignment))
        struct.pack_into('<I', body, pe.opt + 0x38, size_of_image)
        code_add = sum(align_up(len(a.data), pe.file_alignment) for a in added
                       if a.chars & IMAGE_SCN_CNT_CODE)
        init_add = sum(align_up(len(a.data), pe.file_alignment) for a in added
                       if a.chars & IMAGE_SCN_CNT_INITIALIZED_DATA)
        if code_add:
            struct.pack_into('<I', body, pe.opt + 0x04, pe.size_of_code + code_add)
        if init_add:
            struct.pack_into('<I', body, pe.opt + 0x08, pe.size_of_init + init_add)
        if not pe.base_of_code:
            struct.pack_into('<I', body, pe.opt + 0x14, added[0].rva)

        # 5. entry point and directories
        entry = self.entry_rva if self.entry_rva is not None else pe.entry_rva
        struct.pack_into('<I', body, pe.opt + 0x10, entry)
        n_dirs = max(pe.n_dirs, len(pe.dirs))
        struct.pack_into('<I', body, pe.nrva_off, n_dirs)
        for i in range(n_dirs):
            rva, size = pe.dirs[i] if i < len(pe.dirs) else (0, 0)
            off = pe.dirs_off + i * 8
            if off + 8 > pe.opt + pe.size_opt:
                raise PeError('data directory runs past the optional header')
            struct.pack_into('<II', body, off, rva, size)

        # 6. checksum, over the bytes as they now stand
        struct.pack_into('<I', body, pe.checksum_off, 0)
        struct.pack_into('<I', body, pe.checksum_off,
                         pe_checksum(bytes(body), pe.checksum_off))
        return bytes(body)


def write_file(path: str | Path, data: bytes) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p
