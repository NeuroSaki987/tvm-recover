"""Read, inspect and *rebase* a PDB: the file that carries the recovered names into a decompiler.

**Why this module exists (measured, not assumed).** The PDB link.exe writes describes the *payload
DLL*: its section table says `.text` starts at RVA `0x1000`. The shipped code does not live there -- it
lives in `.neko`, appended to the host at whatever RVA the host's section table allowed (measured:
`0x204000`). A debugger-facing loader resolves a CodeView symbol as
`section_table[segment - 1].VirtualAddress + offset`, so every name in that PDB resolves `0x203000`
bytes below the code it names, i.e. inside the host's *original* `.text`. The names load and point
somewhere else. That is the whole of the "the PDB does not work" symptom, and it is a layout fact, not
a tool setting: the fix is to move the PDB's own section table to where the code went.

Two consequences follow and are handled here rather than hoped away:

* **A PDB is a container of arbitrary byte streams** (MSF 7.00): a header, a stream directory keyed by
  page numbers, then streams. Nothing is at a fixed offset, so the section-header substream is located
  by *matching it against a section table this code already knows* -- the one in the image the PDB
  describes. There is no way to find it by position alone and no way to be wrong quietly: the matcher
  requires name, virtual size *and* RVA to agree for every section, in order.
* **Rebasing is a 4-byte edit per section** (`IMAGE_SECTION_HEADER.VirtualAddress`), which is why the
  verification below re-parses the file the writer produced instead of trusting the edit.

What this module deliberately does not do: rewrite symbol records. They are segment:offset pairs, so a
uniform move of the section table moves all of them at once -- that is the point of rebasing rather
than renaming. The one place the *segment index* would have to change is a PDB that describes several
images at once, which this one does not.

Run:  python llvm/pdbedit.py info   FILE.pdb
      python llvm/pdbedit.py sections FILE.pdb [--image=payload.dll]
      python llvm/pdbedit.py symbols FILE.pdb --image=payload.dll [--limit=20]
      python llvm/pdbedit.py rebase FILE.pdb --out=NEW.pdb --image=payload.dll --delta=0x203000
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

MAGIC = b'Microsoft C/C++ MSF 7.00\r\n\x1aDS\x00\x00\x00'
PAGE = 4096

#: Stream indices the format fixes. Everything else is allocated by the writer.
STREAM_INFO = 1
STREAM_DBI = 3

#: `DBIStream::Header` is 64 bytes and the substream sizes below it are in this order, so the section
#: header substream is *not* addressable by arithmetic. It is found by matching (see `locate`).
#: 20 fields: the trailing `Padding` is part of the header and forgetting it puts the first module
#: record 4 bytes early, which parses as a module whose symbol stream is garbage.
DBI_HEADER = struct.Struct('<iIIHHHHHHiiiiiiIIHHI')

#: `IMAGE_SECTION_HEADER`, the only part of it that matters here.
SECTION_HEADER = struct.Struct('<8sIIIIIIHHI')
SECTION_HEADER_SIZE = SECTION_HEADER.size          # 40

#: CodeView symbol records we care about: the ones that carry a segment:offset.
S_PUB32 = 0x110E
S_LPROC32 = 0x110F
S_GPROC32 = 0x1110
S_LPROC32_ID = 0x1146
S_GPROC32_ID = 0x1147
PROC_KINDS = (S_LPROC32, S_GPROC32, S_LPROC32_ID, S_GPROC32_ID)

#: A *reference* to a procedure or data symbol: `(sumName, ibSym, imod, name)`. The global stream uses
#: these to point at the real record, which lives in the module stream `imod` at byte `ibSym`.
S_PROCREF = 0x1125
S_DATAREF = 0x1126
S_LPROCREF = 0x1127

#: A function-scoped variable (CodeView 2013+). It carries a *name*; the value's location is in the
#: `S_DEFRANGE*` records that follow it. Measured on this build: `{flags u32, 2 zero bytes, name}`,
#: padded to a 4-byte boundary, which is why the name is read from offset 6 rather than 4.
S_LOCAL = 0x113E
S_DEFRANGE_REGISTER = 0x1141
S_DEFRANGE_REGISTER_REL = 0x1142
S_OBJNAME = 0x1101
S_COMPILE3 = 0x113C
S_BUILDINFO = 0x114C
S_FRAMEPROC = 0x1012

#: Every module's symbol stream begins with this, before the first record.
CV_SIGNATURE_C13 = 4


class PdbError(RuntimeError):
    """The file is not an MSF 7.00 container, or it does not describe the image it was asked about."""


class SectionHeader:
    """One `IMAGE_SECTION_HEADER` as the PDB stores it: names and virtual layout, no characteristics."""

    __slots__ = ('name', 'vsize', 'rva', 'rsize', 'raw', 'chars')

    def __init__(self, blob: bytes):
        (name, vsize, rva, rsize, raw, _rel, _ln, _nrel, _nln, chars) = \
            SECTION_HEADER.unpack_from(blob, 0)
        self.name = name.rstrip(b'\0').decode('ascii', 'replace')
        self.vsize = vsize
        self.rva = rva
        self.rsize = rsize
        self.raw = raw
        self.chars = chars

    def as_dict(self) -> dict:
        return {'name': self.name, 'rva': self.rva, 'vsize': self.vsize, 'rsize': self.rsize}

    def __repr__(self) -> str:
        return f'<{self.name} rva {self.rva:#x} vsize {self.vsize:#x}>'


class Pdb:
    """An MSF 7.00 file: page-addressed streams, read and written through the page map."""

    def __init__(self, data: bytes, label: str = '<pdb>'):
        if data[:32] != MAGIC:
            raise PdbError(f'{label}: not an MSF 7.00 container (magic {data[:32]!r})')
        (self.page_size, self.free_map, self.n_pages, self.dir_bytes,
         _unknown, block_map) = struct.unpack_from('<IIIIII', data, 32)
        if self.page_size != PAGE:
            raise PdbError(f'{label}: page size {self.page_size} is not {PAGE}')
        self.label = label
        self.data = bytearray(data)
        block = self.page_size * block_map
        want = (self.dir_bytes + self.page_size - 1) // self.page_size
        dir_pages = struct.unpack_from(f'<{want}I', self.data, block)
        directory = b''.join(bytes(self.data[p * self.page_size:(p + 1) * self.page_size])
                            for p in dir_pages)[:self.dir_bytes]
        self.n_streams = struct.unpack_from('<I', directory, 0)[0]
        sizes = struct.unpack_from(f'<{self.n_streams}I', directory, 4)
        pos = 4 + 4 * self.n_streams
        self.pages: list[list[int]] = []
        self.sizes = list(sizes)
        for size in sizes:
            if size in (0, 0xFFFFFFFF):
                self.pages.append([])
                continue
            npages = (size + self.page_size - 1) // self.page_size
            self.pages.append(list(struct.unpack_from(f'<{npages}I', directory, pos)))
            pos += 4 * npages

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_path(cls, path: str | Path) -> 'Pdb':
        p = Path(path)
        return cls(p.read_bytes(), str(p))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes(self.data))
        return p

    # ------------------------------------------------------------------ streams
    def stream(self, index: int) -> bytes:
        size = self.sizes[index]
        if size in (0, 0xFFFFFFFF):
            return b''
        return b''.join(bytes(self.data[p * self.page_size:(p + 1) * self.page_size])
                        for p in self.pages[index])[:size]

    def _file_offset(self, index: int, off: int) -> int:
        """Where byte `off` of stream `index` lives in the file. Streams are page-chained."""
        page_size = self.page_size
        page = self.pages[index][off // page_size]
        return page * page_size + off % page_size

    def read(self, index: int, off: int, size: int) -> bytes:
        return self.stream(index)[off:off + size]

    def write(self, index: int, off: int, payload: bytes) -> None:
        """Patch bytes inside a stream, page by page -- the stream's pages need not be contiguous."""
        for i, byte in enumerate(payload):
            self.data[self._file_offset(index, off + i)] = byte

    # ------------------------------------------------------------------ the section table
    def locate_text(self, rva: int, limit: int = 32) -> tuple[int, int, int]:
        """Find the section table of a *rebased* PDB: the `.text` record whose RVA is `rva`.

        `locate` matches the whole table against an image this code already has, which is the right
        check before a rebase and impossible after one: the table no longer describes that image. For
        verifying the shipped pair there is only one number to go on -- where `.text` claims to be --
        so that record is the anchor and the records after it are taken while they still look like
        section headers.
        """
        key = struct.pack('<8s', b'.text\0\0\0')
        for index in range(self.n_streams):
            blob = self.stream(index)
            if not blob:
                continue
            at = blob.find(key)
            while at >= 0:
                if at + SECTION_HEADER_SIZE <= len(blob):
                    head = SectionHeader(blob[at:])
                    if head.rva == rva:
                        count = 1
                        while count < limit and at + (count + 1) * SECTION_HEADER_SIZE <= len(blob):
                            nxt = SectionHeader(blob[at + count * SECTION_HEADER_SIZE:])
                            if not _plausible(nxt):
                                break
                            count += 1
                        return index, at, count
                at = blob.find(key, at + 1)
        raise PdbError(f'{self.label}: no section table whose .text is at {rva:#x}; this PDB does '
                       'not describe an image with the payload in that section')

    def locate(self, expect: list[dict]) -> tuple[int, int]:
        """Find the section-header substream: the (stream, byte offset) whose records are `expect`.

        The substream has no length in front of it, so position is useless and matching is everything.
        The key is the first record's name + virtual size + RVA -- three independently computed numbers,
        which a section table cannot coincidentally match in a 1.3 MB file.
        """
        if not expect:
            raise PdbError(f'{self.label}: no section table to search for')
        first = expect[0]
        key = struct.pack('<8sII', first['name'].encode('ascii')[:8].ljust(8, b'\0'),
                          first['vsize'], first['rva'])
        for index in range(self.n_streams):
            blob = self.stream(index)
            if not blob:
                continue
            at = blob.find(key)
            while at >= 0:
                if at + SECTION_HEADER_SIZE * len(expect) <= len(blob):
                    got = [SectionHeader(blob[at + i * SECTION_HEADER_SIZE:]) for i in range(len(expect))]
                    if all(g.name == e['name'] and g.vsize == e['vsize'] and g.rva == e['rva']
                           for g, e in zip(got, expect) if e):
                        return index, at
                at = blob.find(key, at + 1)
        raise PdbError(f'{self.label}: no section table matching the image was found; the PDB does '
                       f'not describe this image (first record sought: {first["name"]!r} '
                       f'vsize {first["vsize"]:#x} rva {first["rva"]:#x})')

    def section_headers(self, expect: list[dict]) -> list[SectionHeader]:
        index, off = self.locate(expect)
        return self.headers_at(index, off, len(expect))

    def headers_at(self, index: int, off: int, count: int) -> list[SectionHeader]:
        blob = self.stream(index)
        return [SectionHeader(blob[off + i * SECTION_HEADER_SIZE:])
                for i in range(count)]

    def rebase(self, expect: list[dict], delta: int) -> dict:
        """Move every section by `delta` and re-read the file to prove the move landed.

        The read-back uses the position `locate` returned, not `locate` again: after the move the
        table no longer matches the image it was found by, and re-running the matcher would fail on
        the very file the move just produced.
        """
        index, off = self.locate(expect)
        before = self.headers_at(index, off, len(expect))
        if delta:
            for i, want in enumerate(expect):
                if not want:
                    continue
                self.write(index, off + i * SECTION_HEADER_SIZE + 12,
                           struct.pack('<I', (want['rva'] + delta) & 0xFFFFFFFF))
        after = self.headers_at(index, off, len(expect))
        moved = [{'name': b.name, 'from': b.rva, 'to': a.rva, 'delta': a.rva - b.rva}
                 for b, a in zip(before, after)]
        wrong = [m for m in moved if m['delta'] != delta]
        return {'stream': index, 'offset': off, 'delta': delta, 'sections': len(expect),
                'moved': moved, 'ok': not wrong,
                'problems': [f'{m["name"]}: {m["from"]:#x} -> {m["to"]:#x} (delta {m["delta"]:#x} '
                             f'!= {delta:#x})' for m in wrong]}

    # ------------------------------------------------------------------ identity
    def info(self) -> dict:
        """The PDB Info stream: the GUID and age a CodeView record in an image must agree with."""
        blob = self.stream(STREAM_INFO)
        if len(blob) < 28:
            raise PdbError(f'{self.label}: the PDB info stream is {len(blob)} bytes, expected >= 28')
        version, signature, age = struct.unpack_from('<III', blob, 0)
        return {'version': version, 'signature': signature, 'age': age,
                'guid': blob[12:28].hex(), 'guid_bytes': blob[12:28]}

    def dbi(self) -> dict:
        blob = self.stream(STREAM_DBI)
        if len(blob) < DBI_HEADER.size:
            raise PdbError(f'{self.label}: the DBI stream is {len(blob)} bytes, expected >= 64')
        (signature, version_header, age, glob, build, pub, dll_ver, symrec, rbld, modinfo,
         seccontrib, secmap, srcinfo, tsmap, _mfc, dbghdr, ec, flags, machine,
         _padding) = DBI_HEADER.unpack_from(blob, 0)
        return {'signature': signature, 'version_header': version_header, 'age': age,
                'global_stream': glob, 'public_stream': pub, 'symrec_stream': symrec,
                'build': build, 'machine': machine, 'flags': flags,
                'modinfo_size': modinfo, 'seccontrib_size': seccontrib, 'secmap_size': secmap,
                'srcinfo_size': srcinfo, 'tsmap_size': tsmap, 'dbghdr_size': dbghdr,
                'ec_size': ec}

    # ------------------------------------------------------------------ symbols
    def modinfo(self) -> list[dict]:
        """The DBI stream's module list: which stream holds each object's symbols, and how big.

        A module's data is three length-prefixed substreams -- CodeView symbols, C11 line info, C13
        line info -- so `sym_stream`/`sym_size` are the only numbers needed to walk the records that
        name things inside the code.
        """
        blob = self.stream(STREAM_DBI)
        dbi = self.dbi()
        # The DBI header is fixed at 64 bytes: `VersionHeader` is a version *signature*
        # (`0x01310977`), not a size, so it cannot be used as an offset.
        off = DBI_HEADER.size
        end = off + dbi['modinfo_size']
        if end > len(blob):
            raise PdbError(f'{self.label}: the module info substream runs past the DBI stream')
        # Fixed-size module records (64 bytes each), then a 0xFFFFFFFF terminator, then one string
        # buffer holding the object name and object path of every module in record order.
        fixed = 64
        mods: list[dict] = []
        while off + fixed <= end:
            word = struct.unpack_from('<I', blob, off)[0]
            if word == 0xFFFFFFFF:
                off += 4
                break
            (_unused, _sc, _flags, sym_stream, sym_size, c11, c13,
             _files) = struct.unpack_from('<I28sHHIIIH', blob, off)
            if sym_stream >= self.n_streams or sym_size > self.sizes[sym_stream]:
                break
            mods.append({'index': len(mods) + 1, 'sym_stream': sym_stream, 'sym_size': sym_size,
                         'c11_size': c11, 'c13_size': c13})
            off += fixed
        strings = [s.decode('ascii', 'replace') for s in blob[off:end].split(b'\0') if s]
        for i, mod in enumerate(mods):
            mod['name'] = strings[2 * i] if 2 * i < len(strings) else ''
            mod['object'] = strings[2 * i + 1] if 2 * i + 1 < len(strings) else ''
        return mods

    def symbols(self) -> list[dict]:
        """Every CodeView symbol record, from the shared record stream and from each module stream.

        Records are `uint16 length, uint16 kind, body[length - 2]` -- the length counts the kind and
        the body but not itself, which is the one detail that makes a naive walk stop after one
        record. **A module stream starts with the 4-byte C13 signature** (`CV_SIGNATURE_C13`) before
        its first record; reading from offset 0 instead parses that number as a record length and
        loses everything after it. The three shapes that carry an address are decoded; everything
        else is reported by kind, because "how many of each" is the question a PDB is usually asked.
        """
        out: list[dict] = []
        dbi = self.dbi()
        for rec in _records(self.stream(dbi['symrec_stream'])):
            out.append(_decode(rec, 'records'))
        for mod in self.modinfo():
            blob = self.stream(mod['sym_stream'])[:mod['sym_size']]
            if struct.unpack_from('<I', blob, 0)[0] != CV_SIGNATURE_C13:
                raise PdbError(f'{self.label}: module {mod["index"]} does not start with the C13 '
                               f'signature; its symbol stream cannot be walked safely')
            for rec in _records(blob[4:]):
                entry = _decode(rec, mod['name'] or f'module {mod["index"]}')
                entry['module'] = mod['index']
                out.append(entry)
        return out

    def procs(self) -> list[dict]:
        """The function symbols: `(name, segment, offset)`. This is what names a decompiled function."""
        return [s for s in self.symbols() if 'segment' in s and s.get('kind_name', '')
                in ('S_GPROC32', 'S_LPROC32', 'S_GPROC32_ID', 'S_LPROC32_ID', 'S_PUB32')]

    def locals(self) -> list[str]:
        """The names of the function-scoped variables the PDB carries, in file order.

        This is the list a decompiler turns into `v429` and `t_rdx` instead of `a1` and `v1`. It is the
        one part of a PDB that cannot be recovered from the image at all, so it is measured rather
        than assumed.
        """
        names = [s['name'] for s in self.symbols() if s['kind_name'] == 'S_LOCAL' and s.get('name')]
        return names


def _records(blob: bytes) -> list[dict]:
    out: list[dict] = []
    off = 0
    while off + 4 <= len(blob):
        length, kind = struct.unpack_from('<HH', blob, off)
        if length < 2 or off + 2 + length > len(blob):
            out.append({'kind': kind, 'kind_name': f'0x{kind:04x}(truncated)', 'at': off})
            break
        out.append({'kind': kind, 'at': off, 'body': blob[off + 4:off + 2 + length],
                    'size': length})
        off += 2 + length
    return out


def _decode(rec: dict, where: str) -> dict:
    kind = rec['kind']
    body = rec.get('body', b'')
    entry = {'kind': kind, 'kind_name': _kind_name(kind), 'at': rec['at'], 'where': where}
    if kind == S_PUB32 and len(body) >= 10:
        flags, offset, seg = struct.unpack_from('<IIH', body, 0)
        entry.update(flags=flags, offset=offset, segment=seg, name=_cvstring(body, 10))
    elif kind in PROC_KINDS and len(body) >= 35:
        (_parent, _end, _next, _len, _dbg0, _dbg1, _type) = struct.unpack_from('<IIIIIiI', body, 0)
        offset, seg = struct.unpack_from('<IH', body, 28)
        entry.update(offset=offset, segment=seg, name=_cvstring(body, 35))
    elif kind == S_PROCREF and len(body) >= 10:
        _sum, ib_sym, imod = struct.unpack_from('<IIH', body, 0)
        entry.update(ib_sym=ib_sym, imod=imod, name=_cvstring(body, 10))
    elif kind == S_LOCAL and len(body) >= 6:
        # Measured: `flags u32`, then two zero bytes, then the name, padded to 4.
        name = _printable(body, 6) or _printable(body, 4)
        if name:
            entry.update(name=name, flags=struct.unpack_from('<I', body, 0)[0])
    else:
        text = _printable(body, 0)
        if text:
            entry['name'] = text
    return entry


def _printable(body: bytes, at: int) -> str:
    """The longest printable run in `body[at:]`, for record kinds this module does not decode."""
    best = ''
    run = []
    for byte in body[at:]:
        if 32 <= byte < 127:
            run.append(chr(byte))
        else:
            if len(run) > len(best):
                best = ''.join(run)
            run = []
    if len(run) > len(best):
        best = ''.join(run)
    return best


def _cvstring(body: bytes, at: int) -> str:
    raw = body[at:]
    end = raw.find(b'\0')
    return (raw if end < 0 else raw[:end]).decode('ascii', 'replace')


def _plausible(header: 'SectionHeader') -> bool:
    """Whether a 40-byte window still looks like an `IMAGE_SECTION_HEADER`.

    Used to know where a section table *ends* without being told how many sections there are: a name
    of printable characters and a non-zero virtual size is what every real record has and what
    whatever follows the table does not.
    """
    if header.vsize == 0 or header.rsize == 0 or header.rva == 0:
        return False
    name = header.name
    return bool(name) and all(32 <= ord(ch) < 127 for ch in name)


_KINDS = {
    S_PUB32: 'S_PUB32', S_LPROC32: 'S_LPROC32', S_GPROC32: 'S_GPROC32',
    S_LPROC32_ID: 'S_LPROC32_ID', S_GPROC32_ID: 'S_GPROC32_ID',
    S_PROCREF: 'S_PROCREF', S_DATAREF: 'S_DATAREF', S_LPROCREF: 'S_LPROCREF',
    S_LOCAL: 'S_LOCAL', S_OBJNAME: 'S_OBJNAME', S_COMPILE3: 'S_COMPILE3',
    S_BUILDINFO: 'S_BUILDINFO', S_FRAMEPROC: 'S_FRAMEPROC',
    S_DEFRANGE_REGISTER: 'S_DEFRANGE_REGISTER',
    S_DEFRANGE_REGISTER_REL: 'S_DEFRANGE_REGISTER_REL',
    0x1107: 'S_CONSTANT', 0x1108: 'S_UDT', 0x110B: 'S_LDATA32', 0x110C: 'S_GDATA32',
    0x1111: 'S_REGREL32', 0x1112: 'S_LTHREAD32', 0x1113: 'S_GTHREAD32',
    0x0006: 'S_REGISTER', 0x114F: 'S_PROC_ID_END',
}


def _kind_name(kind: int) -> str:
    return _KINDS.get(kind, f'0x{kind:04x}')


# ------------------------------------------------------------------------------ image side

def image_sections(pe) -> list[dict]:
    """A `peedit.Pe`'s section table in the shape `locate`/`rebase` compare against."""
    return [{'name': s.name, 'vsize': s.vsize, 'rva': s.rva} for s in pe.sections]


def check(host_path: str | Path, pdb_path: str | Path,
          map_path: str | Path | None = None) -> dict:
    """Is this PDB usable on this image? Every link in the chain, measured.

    Five questions, in the order a loader asks them:

    1. does the image carry a CodeView record at all (no record: nothing looks for a PDB),
    2. do that record's GUID and age equal the PDB's (a mismatch is refused by the loader),
    3. does the PDB's section table put `.text` where the payload actually is,
    4. does every function symbol resolve to an address inside `.neko` -- and, given the linker map,
       to the address the linker gave it,
    5. how many recovered names the PDB carries.

    A pass here is not a pass in a decompiler: it says the four numbers a decompiler reads are
    consistent, and that the names it reads resolve onto the instructions they describe.
    """
    import peedit
    host = peedit.Pe.from_path(host_path)
    db = Pdb.from_path(pdb_path)
    problems: list[str] = []
    neko = host.section('.neko')
    if neko is None:
        raise PdbError(f'{host.label}: no .neko section; this is not an injected image')
    cv = codeview(host)
    info = db.info()
    if cv is None:
        problems.append('the image carries no CodeView record, so no loader would look for this PDB')
    else:
        if cv['guid_bytes'] != info['guid_bytes']:
            problems.append(f'guid mismatch: image RSDS {cv["guid"]} vs PDB {info["guid"]}')
        if cv['age'] != info['age']:
            problems.append(f'age mismatch: image RSDS {cv["age"]} vs PDB {info["age"]}')
        if cv['record_rva'] < neko.rva or cv['record_rva'] >= neko.end:
            problems.append(f'the RSDS record at {cv["record_rva"]:#x} is not inside .neko '
                            f'[{neko.rva:#x}, {neko.end:#x})')
    index, off, count = db.locate_text(neko.rva)
    headers = db.headers_at(index, off, count)
    rows = []
    symbols = db.procs()
    inside = 0
    for s in symbols:
        seg = s['segment']
        if not (1 <= seg <= len(headers)):
            problems.append(f'{s.get("name")}: segment {seg} is outside the {len(headers)}-section '
                            'table')
            continue
        rva = headers[seg - 1].rva + s['offset']
        ok = neko.rva <= rva < neko.end
        inside += int(ok)
        rows.append({'name': s['name'], 'kind': s['kind_name'], 'segment': seg,
                     'offset': s['offset'], 'rva': rva, 'in_neko': ok})
    if symbols and inside != len(symbols):
        problems.append(f'{len(symbols) - inside} of {len(symbols)} function symbol(s) resolve '
                        f'outside .neko [{neko.rva:#x}, {neko.end:#x})')
    matched = None
    if map_path is not None:
        # The map file lists `0001:<offset>` -- an offset inside section 1, which is exactly what a
        # CodeView symbol carries. Comparing *offsets* makes this check independent of the rebase, so
        # it still means something after the section table has been moved.
        want: dict[str, int] = {}
        for line in Path(map_path).read_text(encoding='utf-8', errors='replace').splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith('0001:'):
                try:
                    want[parts[1]] = int(parts[0][5:], 16)
                except ValueError:
                    continue
        matched = 0
        wrong = []
        for row in rows:
            if row['name'] not in want or row['segment'] != 1:
                continue
            if row['offset'] == want[row['name']]:
                matched += 1
            else:
                wrong.append(f'{row["name"]}: pdb offset {row["offset"]:#x} != map '
                             f'{want[row["name"]]:#x}')
        if wrong:
            problems.append(f'{len(wrong)} symbol(s) disagree with the linker map: {wrong[:3]}')
    names = db.locals()
    return {'ok': not problems, 'problems': problems,
            'neko': {'rva': neko.rva, 'vsize': neko.vsize, 'end': neko.end},
            'codeview': cv, 'pdb_info': {'guid': info['guid'], 'age': info['age']},
            'pdb_sections': [h.as_dict() for h in headers],
            'symbols': len(symbols), 'symbols_inside_neko': inside,
            'symbols_matching_map': matched,
            'locals': len(names), 'locals_distinct': len(set(names)),
            'first_locals': names[:8], 'rows': rows[:8]}


def codeview(pe) -> dict | None:
    """The CodeView record the image's debug directory points at, if there is one."""
    import peedit
    if len(pe.dirs) <= peedit.DIR_DEBUG or pe.dirs[peedit.DIR_DEBUG] == (0, 0):
        return None
    rva, size = pe.dirs[peedit.DIR_DEBUG]
    for i in range(size // 28):
        blob = pe.image_bytes(rva + i * 28, 28)
        (_chars, _ts, _maj, _min, typ, dsize, daddr, _ptr) = struct.unpack('<IIHHIIII', blob)
        if typ != 2 or not dsize:
            continue
        body = pe.image_bytes(daddr, dsize)
        if body[:4] != b'RSDS' or len(body) < 24:
            return None
        guid = body[4:20]
        age = struct.unpack_from('<I', body, 20)[0]
        return {'entry_rva': rva + i * 28, 'entry_index': i, 'record_rva': daddr,
                'dir_rva': rva, 'dir_size': size,
                'size': dsize, 'guid': guid.hex(), 'guid_bytes': guid, 'age': age,
                'path': body[24:].split(b'\0', 1)[0].decode('ascii', 'replace')}
    return None


# ------------------------------------------------------------------------------ CLI

def main(argv: list[str]) -> int:
    import peedit
    path: Path | None = None
    image: Path | None = None
    out: Path | None = None
    map_path: Path | None = None
    delta: int | None = None
    limit = 20
    what = 'info'
    for a in argv:
        if a.startswith('--image='):
            image = Path(a.split('=', 1)[1])
        elif a.startswith('--map='):
            map_path = Path(a.split('=', 1)[1])
        elif a.startswith('--out='):
            out = Path(a.split('=', 1)[1])
        elif a.startswith('--delta='):
            delta = int(a.split('=', 1)[1], 0)
        elif a.startswith('--limit='):
            limit = int(a.split('=', 1)[1], 0)
        elif a in ('info', 'sections', 'symbols', 'procs', 'check', 'rebase'):
            what = a
        elif a in ('-h', '--help'):
            print(__doc__)
            return 2
        elif not a.startswith('--'):
            path = Path(a)
    if path is None:
        print(__doc__)
        return 2
    pdb = Pdb.from_path(path)
    if what == 'info':
        info = pdb.info()
        dbi = pdb.dbi()
        print(f'{path}: {len(pdb.data)} B, {pdb.n_pages} pages, {pdb.n_streams} streams')
        print(f'  info: version {info["version"]} signature {info["signature"]:#x} '
              f'age {info["age"]} guid {info["guid"]}')
        print(f'  dbi : age {dbi["age"]} machine {dbi["machine"]:#x} symrec stream '
              f'{dbi["symrec_stream"]} global {dbi["global_stream"]} public {dbi["public_stream"]}')
        return 0
    if what == 'check':
        report = check(image, path, map_path)
        print(f'{image.name} + {path.name}: {"OK" if report["ok"] else "PROBLEMS"}')
        print(f'  .neko rva {report["neko"]["rva"]:#x}..{report["neko"]["end"]:#x}')
        cv = report['codeview'] or {}
        print(f'  image RSDS guid {cv.get("guid")} age {cv.get("age")} path {cv.get("path")!r} '
              f'at {cv.get("record_rva", 0):#x}')
        print(f'  pdb        guid {report["pdb_info"]["guid"]} age {report["pdb_info"]["age"]}')
        print(f'  pdb .text  {report["pdb_sections"][0]["rva"]:#x} '
              f'(vsize {report["pdb_sections"][0]["vsize"]})')
        print(f'  symbols {report["symbols"]}, inside .neko {report["symbols_inside_neko"]}'
              + (f', matching the map {report["symbols_matching_map"]}' if report[
                  'symbols_matching_map'] is not None else ''))
        print(f'  locals {report["locals"]} ({report["locals_distinct"]} distinct), e.g. '
              f'{report["first_locals"]}')
        for row in report['rows'][:5]:
            print(f'    {row["kind"]:<12} seg {row["segment"]} off {row["offset"]:#08x} -> '
                  f'{row["rva"]:#x}  {row["name"]}')
        for problem in report['problems']:
            print(f'  PROBLEM {problem}')
        return 0 if report['ok'] else 1
    if image is None:
        print('this subcommand needs --image=<the PE the PDB describes>')
        return 2
    pe = peedit.Pe.from_path(image)
    expect = image_sections(pe)
    if what == 'sections':
        for header in pdb.section_headers(expect):
            print(f'  {header.name:<9} rva {header.rva:#08x} vsize {header.vsize:<8} '
                  f'rsize {header.rsize:<8}')
        return 0
    if what == 'symbols':
        syms = pdb.symbols()
        counts: dict[str, int] = {}
        for s in syms:
            counts[s['kind_name']] = counts.get(s['kind_name'], 0) + 1
        print(f'{len(syms)} symbol record(s): '
              + ', '.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
        for mod in pdb.modinfo():
            print(f'  module {mod["index"]}: sym stream {mod["sym_stream"]} '
                  f'{mod["sym_size"]} B  {mod["name"]}')
        shown = 0
        for s in syms:
            if 'name' in s and shown < limit:
                print(f'  {s["kind_name"]:<14} seg {s.get("segment", 0):>3} off '
                      f'{s.get("offset", 0):#08x}  {s["name"]}')
                shown += 1
        return 0
    if what == 'procs':
        headers = pdb.section_headers(expect)
        rows = pdb.procs()
        print(f'{len(rows)} function symbol(s) for {image.name}:')
        for s in rows[:limit]:
            seg = s['segment']
            base = headers[seg - 1].rva if 1 <= seg <= len(headers) else None
            rva = f'{base + s["offset"]:#x}' if base is not None else 'seg?'
            print(f'  {s["kind_name"]:<12} seg {seg:>2} off {s["offset"]:#08x} -> rva {rva:<10} '
                  f'{s["name"]}')
        return 0
    if what == 'rebase':
        if delta is None or out is None:
            print('rebase needs --delta=<n> and --out=<file>')
            return 2
        report = pdb.rebase(expect, delta)
        pdb.save(out)
        print(f'{report["sections"]} section(s) moved by {delta:#x} in stream {report["stream"]} '
              f'at byte {report["offset"]}: {"OK" if report["ok"] else "PROBLEMS"}')
        for m in report['moved']:
            print(f'  {m["name"]:<9} {m["from"]:#08x} -> {m["to"]:#08x}')
        for problem in report['problems']:
            print(f'  PROBLEM {problem}')
        print(f' wrote {out}')
        return 0 if report['ok'] else 1
    print(__doc__)
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
