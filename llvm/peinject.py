"""Inject the write-devirt payload into a real PE: append `.neko`, redirect the entry point, handle
relocations.

The four operations, in the order they happen:

**1. Append `.neko`** -- read+execute (`0x60000020`), name padded to the 8-byte field
(`b'.neko\\0\\0\\0'`), holding the payload's machine code and its read-only data, followed by the
self-describing header `payload.py` built. Nothing in the host's existing sections is modified, so an
analyst can diff the two files and see exactly one new section plus the two header fields below.

**2. Redirect `AddressOfEntryPoint`** to `neko_rva + entry_offset`. The payload's first instruction is
the entry stub, so the loader now runs the devirtualized code before the original code -- and the
stub's last instruction is a tail `jmp` to the original entry point, with the loader's own stack
pointer and arguments intact (measured in `llvm/miniload.py`, not asserted here).

**3. Patch two 8-byte RVAs inside `.neko`** (`tvm_entry_rva`, `tvm_oep_rva` in the payload's read-only
data). They are what the stub uses to compute the image base without a relocation:
`base = &tvm_payload_entry - tvm_entry_rva`, then `target = base + tvm_oep_rva`.

**4. Relocations.** This payload needs *none*, and that is a measured property, not an assumption:
`payload.build()` refuses an image with a base relocation directory, and this module refuses to inject
a payload whose manifest reports one unless the caller supplies the fixup sites. The writer exists
(`peedit.Patch.add_relocations`, exercised by `llvm/tests/test_write_devirt.py` against a synthetic
payload that has a DIR64 fixup) because the alternative -- assuming nothing will ever need one -- is
how a tool acquires a class of image it silently breaks. When it is used, the fixups are merged with
the host's existing relocation blocks into one new read-only section and the directory entry is
repointed at it, because the directory must describe one contiguous range and the host's blocks are
nowhere near the new section.

Two host properties are handled explicitly rather than silently:

* **The certificate table is a file offset at the end of the file.** Appending raw data there would
  leave the tail pointing into the image, so by default the signature is stripped (and reported). A
  modified image cannot keep a valid signature anyway; a stale one is worse than none.
* **Header space.** A new section header needs 40 bytes below `SizeOfHeaders`. When the image has
  none, `peedit` refuses with the numbers instead of moving raw data.

Also measured, and worth knowing: five of the twenty-four shipped ACE modules have a `CheckSum` field
that does not match their contents (this host among them), so "recompute the checksum" is not a
restoration of anything the vendor shipped. The default recomputes it anyway -- a newly written file
should describe itself -- and `keep_checksum=True` preserves the original bytes.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def _boot_env() -> None:
    """Load `main/.env` if present. `main/` is not on sys.path for a llvm/ script, so walk up."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'bootenv.py').is_file():
            text = str(parent)
            if text not in sys.path:
                sys.path.insert(0, text)
            try:
                from bootenv import boot
                boot(here)
            except Exception:
                pass
            return


_boot_env()
sys.path.insert(0, str(HERE))

import payload as payload_mod  # noqa: E402
import pdbedit  # noqa: E402
import peedit  # noqa: E402

MACHINE_AMD64 = 0x8664
DEBUG_TYPE_CODEVIEW = peedit.DEBUG_TYPE_CODEVIEW
#: IMAGE_DEBUG_TYPE_EX_DLLCHARACTERISTICS and its CET bit.
DEBUG_TYPE_EX_DLLCHARACTERISTICS = 20
CET_COMPAT_BIT = 0x1
#: The kernel shape's state block: writable, not executable, zero-initialised.
STATE_SECTION_NAME = '.neko.b'
STATE_SECTION_CHARS = 0xC0000040


class InjectRefusal(RuntimeError):
    """The image cannot be patched, and the reason names the check that refused it."""


def inject(host: bytes | Path, payload_bin: bytes, meta: dict, *,
           strip_certificate: bool = True, keep_checksum: bool = False,
           reloc_sites: list[int] | None = None, label: str = '<host>',
           pdb: Path | None = None, pdb_out: Path | None = None) -> tuple[bytes, dict]:
    """Return `(patched_file_bytes, report)`.

    `reloc_sites` are RVAs **inside `.neko`** that hold an absolute address and therefore need a
    DIR64 fixup; the RVA of the section is added to each. An empty list (the default, and what this
    payload reports) means the host's relocation table is not touched at all.

    `pdb` is the PDB `payload.build()` produced and `pdb_out` where the rebased copy goes (default: a
    sibling of the output named as the image's RSDS record names it). Two edits make it usable by a
    decompiler: the CodeView record in `.neko` becomes the one the image's debug directory points at,
    and the PDB's section table is moved to the host's `.neko` RVA -- see `stamp_codeview` and
    `write_rebased_pdb`.
    """
    pe = host if isinstance(host, peedit.Pe) else peedit.Pe(bytes(host) if isinstance(host, bytes)
                                                           else Path(host).read_bytes(), label)
    if pe.machine != MACHINE_AMD64:
        raise InjectRefusal(f'{pe.label}: machine {pe.machine:#x} is not AMD64; the payload is '
                            'x86-64 code and the recovery only produced AMD64 listings')

    section = payload_bin
    header = payload_mod.read_header(section)
    if pe.sections_named(payload_mod.SECTION_NAME):
        raise InjectRefusal(f'{pe.label}: a section named {payload_mod.SECTION_NAME!r} already '
                            'exists; refusing to create a second one that a name search would have '
                            'to disambiguate')

    # Facts about the *input*, captured before any edit: the certificate strip and the section append
    # both change the file, and a report that quoted post-edit numbers as the host's would be wrong in
    # the one place it is most likely to be believed.
    host_bytes = len(pe.data)
    original_checksum = pe.checksum
    host_checksum_valid = peedit.pe_checksum(bytes(pe.data), pe.checksum_off) == original_checksum

    links = meta.get('links', {})
    if links.get('relocations'):
        raise InjectRefusal(f"{pe.label}: the payload image reports {links['relocations']} base "
                            'relocation(s); pass their sites explicitly (reloc_sites) so the host '
                            'relocation table can be rebuilt')

    patch = peedit.Patch(pe)
    cert_note = None
    if strip_certificate:
        before = pe.dirs[peedit.DIR_SECURITY] if len(pe.dirs) > peedit.DIR_SECURITY else (0, 0)
        patch.strip_certificate()
        if patch.stripped_certificate:
            cert_note = {'file_offset': before[0], 'size': before[1]}

    # **Every section this function will append has to be counted, not just the named ones.**
    # `merge_unwind` adds one of its own (`.neko.e`), and it was missing from this sum: on
    # `ACE-CORE.sys` the host had exactly 80 bytes of header space, the estimate asked for 80
    # (`1 + state`), nothing was grown, and the injection then failed on the *third* section with
    # "only 80 byte(s) of header space ... 120 needed". `ACE-CORE.sys3` had 24 bytes and grew for
    # two, so a corpus that started with that image never saw it.
    neko_rva_room = (1 + (1 if reloc_sites else 0) + (1 if meta.get('unwind') else 0)
                     + (1 if meta.get('state_section') else 0))
    growth = 0
    if patch.header_room() < 40 * neko_rva_room:
        # **Kernel images are tight on header space** (`ACE-CORE.sys3`: 24 bytes, one header needs 40).
        # Growing it moves every section's raw data down one FileAlignment unit, which is enumerated and
        # re-verified by `verify()` -- and reported here, because an image whose raw data moved must not
        # look like one that did not.
        growth = patch.grow_headers(sections=neko_rva_room)

    neko_rva = patch.append_section(payload_mod.SECTION_NAME, section, payload_mod.SECTION_CHARS)
    unwind = None
    if meta.get('unwind'):
        # **The life-or-death step for a kernel image.** `.pdata` travels inside `.neko` (see
        # `payload._not_copied`) but the host's exception directory still describes only the host's own
        # code, so a fault *inside* our frames leaves the unwinder with no frame to walk -- which in a
        # driver is a bugcheck, not an exception. The merged table has to be one contiguous, sorted
        # array, and the host's has no room after it, so it is rebuilt in a section of its own and the
        # directory repointed -- the same pattern the relocation merge uses.
        unwind = merge_unwind(patch, section, meta, neko_rva)
    state_section = None
    if meta.get('state_section'):
        # **A driver thread has 12 KiB of kernel stack**, so the kernel build takes the 82 KiB
        # `tvm_state_t` off the stack and addresses it through a third patched RVA -- the same
        # zero-relocation trick as the entry point. Leaving that RVA at its built value (0) would point
        # the payload's state at the image base, i.e. write into the PE headers on the first statement.
        size = int(meta['state_section'])
        if size <= 0 or size % 16:
            raise InjectRefusal(f'the payload reports a {size}-byte state section; expected a positive '
                               'multiple of 16')
        state_rva = patch.append_section(STATE_SECTION_NAME, bytes(size), STATE_SECTION_CHARS)
        patch.set_bytes(neko_rva + header['api']['tvm_state_rva'], struct.pack('<Q', state_rva))
        state_section = {'name': STATE_SECTION_NAME, 'rva': state_rva, 'bytes': size,
                         'chars': STATE_SECTION_CHARS}
        patch.notes.append(f'state block: {size} byte(s) in {STATE_SECTION_NAME} at rva '
                           f'{state_rva:#x} (writable, not executable); tvm_state_rva patched')
    entry_rva = neko_rva + header['entry_offset']
    original_ep = pe.entry_rva
    patch.set_entry(entry_rva)

    api = header['api']
    patch.set_bytes(neko_rva + api['tvm_entry_rva'], struct.pack('<Q', entry_rva))
    patch.set_bytes(neko_rva + api['tvm_oep_rva'], struct.pack('<Q', original_ep))

    reloc_section = None
    n_relocs = 0
    if reloc_sites:
        sites = [neko_rva + s for s in reloc_sites]
        blocks = patch.add_relocations(sites)
        reloc_section = patch.append_section('.neko.r', blocks, 0x42000040)
        patch.set_dir(peedit.DIR_BASERELOC, reloc_section, len(blocks))
        n_relocs = len(sites)

    out = patch.finish()
    debug = None
    if meta.get('codeview'):
        out, debug = stamp_codeview(bytes(out), neko_rva, meta)
    if pdb is not None:
        debug = dict(debug or {})
        debug['pdb'] = write_rebased_pdb(pdb, pdb_out, meta, neko_rva)
    if keep_checksum:
        out = bytearray(out)
        struct.pack_into('<I', out, pe.checksum_off, pe.checksum)
        out = bytes(out)

    check = verify(pe, out, neko_rva, entry_rva, debug)
    report = {
        'host': pe.label, 'host_bytes': host_bytes, 'out_bytes': len(out),
        'host_entry_rva': original_ep, 'neko_rva': neko_rva, 'neko_chars': payload_mod.SECTION_CHARS,
        'neko_name': list(payload_mod.SECTION_NAME_PADDED), 'entry_rva': entry_rva,
        'entry_offset': header['entry_offset'],
        'payload_bytes': len(section), 'image_span': header['image_span'],
        'nfunc': header['nfunc'], 'state_bytes': header['state_bytes'], 'fuel': header['fuel'],
        'n_relocs': n_relocs, 'reloc_section_rva': reloc_section, 'unwind': unwind,
        'state_section': state_section,
        'certificate': cert_note, 'checksum': {'stored_new': struct.unpack_from(
            '<I', out, pe.checksum_off)[0], 'stored_old': original_checksum},
        'host_checksum_was_valid': host_checksum_valid,
        'header_growth': growth,
        'notes': list(patch.notes), 'verify': check, 'debug': debug,
    }
    return out, report


def cet_status(pe: peedit.Pe) -> dict:
    """Does this image opt into CET (shadow stack / IBT), and is its entry instrumented?

    Two independent markers, both read rather than assumed:

    * `IMAGE_DEBUG_TYPE_EX_DLLCHARACTERISTICS` (20), whose data word carries
      `EX_CET_COMPAT = 0x1` -- the image-level opt-in a linker writes for `/CETCOMPAT`;
    * `endbr64` (F3 0F 1E FA) as the entry point's first instruction, which is what the loader's
      indirect call to an instrumented entry must land on.

    Report both: an image can carry the flag without instrumentation (and then it is the *loader's*
    job to check), and it can be instrumented without the flag (harmless when CET is off).
    """
    entry_endbr = False
    try:
        entry_endbr = pe.image_bytes(pe.entry_rva, 4) == b'\xf3\x0f\x1e\xfa'
    except peedit.PeError:
        pass
    flag = 0
    seen = []
    if len(pe.dirs) > peedit.DIR_DEBUG and pe.dirs[peedit.DIR_DEBUG][0] and pe.dirs[peedit.DIR_DEBUG][1]:
        rva, size = pe.dirs[peedit.DIR_DEBUG]
        for i in range(size // 28):
            blob = pe.image_bytes(rva + i * 28, 28)
            (_chars, _ts, _maj, _min, typ, dsize, daddr, _ptr) = struct.unpack('<IIHHIIII', blob)
            seen.append(typ)
            if typ == DEBUG_TYPE_EX_DLLCHARACTERISTICS and dsize:
                flag = struct.unpack_from('<I', pe.image_bytes(daddr, 4), 0)[0]
    return {'cet_compat_flag': flag,
            'cet_compat': bool(flag & CET_COMPAT_BIT),
            'entry_is_endbr64': entry_endbr,
            'debug_types': seen,
            'enabled': bool(flag & CET_COMPAT_BIT) or entry_endbr}


def merge_unwind(patch: peedit.Patch, section: bytes, meta: dict, neko_rva: int) -> dict:
    """Merge the payload's `.pdata` records into a new exception directory. The kernel-critical step.

    A `RUNTIME_FUNCTION` is three RVAs -- `(begin, end, unwind_info)` -- and every one of them moves
    with `.neko`, so each record is rebased by the same delta and then appended to the host's table.
    The result must be **one contiguous array sorted by `begin`**, which is what the unwinder binary-
    searches; the payload's code sits above every host section, so "host records, then payload records"
    is already sorted. It is asserted rather than assumed, and `verify()` re-reads the finished image
    to check it again.

    Why this cannot be skipped: `RtlVirtualUnwind` runs behind every exception, every page fault and
    every bugcheck report. A frame with no record does not "raise" -- the unwinder runs off the end of
    the table, and a driver dies on the spot (`KERNEL_SECURITY_CHECK_FAILURE`, or an unwind to garbage
    and a page fault inside the handler). In user mode the same hole costs a bad stack trace; in kernel
    mode it costs the machine.
    """
    pe = patch.pe
    first = meta['image_first_rva']
    delta = neko_rva - first
    host_rva, host_size = (pe.dirs[peedit.DIR_EXCEPTION]
                           if len(pe.dirs) > peedit.DIR_EXCEPTION else (0, 0))
    host = pe.image_bytes(host_rva, host_size) if host_size else b''
    if len(host) % 12:
        raise InjectRefusal(f'the host exception directory is {host_size} bytes, not a multiple of '
                            'the 12-byte RUNTIME_FUNCTION')
    unw = meta['unwind']
    off = unw['rva_in_section']
    raw = section[off:off + 12 * unw['entries']]
    if len(raw) != 12 * unw['entries']:
        raise InjectRefusal(f'the payload manifest says {unw["entries"]} unwind record(s) at '
                            f'{off:#x}, but the section holds {len(raw) // 12}')

    def entries(blob: bytes, shift: int) -> list[tuple[int, int, int]]:
        out = []
        for i in range(len(blob) // 12):
            begin, end, info = struct.unpack_from('<III', blob, i * 12)
            out.append((begin + shift, end + shift, info + shift))
        return out

    host_rows = entries(host, 0)
    payload_rows = entries(raw, delta)
    merged = host_rows + payload_rows
    begins = [r[0] for r in merged]
    if begins != sorted(begins):
        # The payload's code must sit above every host function for the concatenation to be sorted.
        raise InjectRefusal('the merged exception table would not be sorted by begin RVA: the payload '
                            f'section is at {neko_rva:#x} and the highest host begin is '
                            f'{max((r[0] for r in host_rows), default=0):#x}')
    blob = b''.join(struct.pack('<III', b, e, u) for b, e, u in merged)
    rva = patch.append_section('.neko.e', blob, 0x40000040)   # initialized data, read-only
    patch.set_dir(peedit.DIR_EXCEPTION, rva, len(blob))
    patch.notes.append(f'exception directory rebuilt: {len(host_rows)} host record(s) + '
                       f'{len(payload_rows)} payload record(s) = {len(merged)} at rva {rva:#x}')
    return {'rva': rva, 'bytes': len(blob), 'host_entries': len(host_rows),
            'payload_entries': len(payload_rows), 'entries': len(merged), 'delta': delta,
            'host_dir': [host_rva, host_size], 'sorted': True,
            'payload_range': [payload_rows[0][0], payload_rows[-1][1]] if payload_rows else None}


def stamp_codeview(out: bytes, neko_rva: int, meta: dict) -> tuple[bytes, dict]:
    """Point the image's debug directory at the CodeView record that is now inside `.neko`.

    **Without this the PDB is unreachable.** A debugger-facing loader finds a PDB by reading the
    image's `IMAGE_DEBUG_DIRECTORY`, taking the CodeView (`RSDS`) record it points at, and matching
    that record's GUID+age against the PDB's own. The host's directory still describes the *vendor's*
    PDB, so a perfect PDB for this code is one nothing asks for. The record itself is already in
    `.neko` -- link.exe put it in `.rdata` and `.rdata` is copied -- so the only edit is the entry.

    The host's other debug entries (VC feature, PGO) are left alone: only the CodeView entry is
    repointed. When the host has no CodeView entry, the payload's own directory is used instead and
    the report says so, because that replaces the host's entries rather than editing one.
    """
    cv = meta['codeview']
    pe = peedit.Pe(out, 'patched')
    data = bytearray(out)
    rec_rva = neko_rva + cv['rva_in_section']
    rec_off = pe.rva_to_offset(rec_rva)
    if rec_off is None:
        raise InjectRefusal(f'the CodeView record at {rec_rva:#x} is not backed by file bytes')
    body = bytes(data[rec_off:rec_off + cv['size']])
    if body[:4] != b'RSDS':
        raise InjectRefusal(f'the record at {rec_rva:#x} is {body[:4]!r}, not an RSDS CodeView '
                            'record; the payload manifest and the section disagree')
    guid = body[4:20].hex()
    if guid != cv['guid']:
        raise InjectRefusal(f'the section carries a CodeView record with guid {guid} but the '
                            f'manifest recorded {cv["guid"]}; refusing to point a debugger at a '
                            'record that was not the one the PDB was built against')

    # `IMAGE_DEBUG_DIRECTORY` is `{chars, timestamp, major, minor, type, size, address, pointer}`:
    # `SizeOfData` is at +16 and NOT at +12, which is where `Type` lives. Writing the size over the
    # type is invisible until a reader finds no CodeView entry at all -- the record is there, nothing
    # points at it, and the PDB is once again one nothing asks for.
    DIR_SIZE_OFF, DIR_ADDR_OFF, DIR_PTR_OFF = 16, 20, 24
    rva, size = pe.dirs[peedit.DIR_DEBUG] if len(pe.dirs) > peedit.DIR_DEBUG else (0, 0)
    note = {'record_rva': rec_rva, 'size': cv['size'], 'guid': guid, 'age': cv['age'],
            'source': None, 'entry_index': None, 'entry_rva': None, 'changed_section': None,
            'replaced_directory': False}
    for i in range(size // 28):
        entry_rva = rva + i * 28
        kind = struct.unpack_from('<I', pe.image_bytes(entry_rva, 28), 12)[0]
        if kind != DEBUG_TYPE_CODEVIEW:
            continue
        off = pe.rva_to_offset(entry_rva)
        if off is None:
            raise InjectRefusal(f'the debug directory entry at {entry_rva:#x} is not backed by file')
        struct.pack_into('<III', data, off + DIR_SIZE_OFF, cv['size'], rec_rva, rec_off)
        note.update(source='host', entry_index=i, entry_rva=entry_rva,
                    changed_section=pe.section_at(entry_rva).name)
        break
    else:
        # No CodeView entry: repoint the whole directory at the payload's own, relocated. The payload
        # has exactly one entry (link.exe writes it), and its address fields are image coordinates, so
        # they are rewritten to section coordinates here.
        note['source'] = 'payload'
        note['replaced_directory'] = True
    if note['source'] == 'payload':
        dir_rva = neko_rva + cv['dir_rva_in_section']
        dir_off = pe.rva_to_offset(dir_rva)
        if dir_off is None:
            raise InjectRefusal('the payload carries no debug directory to relocate')
        struct.pack_into('<III', data, dir_off + DIR_SIZE_OFF, cv['size'], rec_rva, rec_off)
        struct.pack_into('<II', data, pe.dirs_off + peedit.DIR_DEBUG * 8, dir_rva, 28)
        note.update(entry_index=0, entry_rva=dir_rva)
    struct.pack_into('<I', data, pe.checksum_off, peedit.pe_checksum(bytes(data), pe.checksum_off))
    note['ok'] = True
    return bytes(data), note


def write_rebased_pdb(pdb: Path, out: Path | None, meta: dict, neko_rva: int) -> dict:
    """Copy the PDB and move its section table to where the code actually is.

    A CodeView symbol is a `segment:offset` pair, resolved by adding the *section table's* virtual
    address. The PDB's table describes the payload image (`.text` at `0x1000`); the shipped code is in
    `.neko` at whatever RVA the host allowed. So the same file with a different section table names
    the right instructions -- no symbol record has to be rewritten, which is why this is a 4-byte edit
    per section and a read-back rather than a rewrite.
    """
    expect = [{'name': s['name'], 'vsize': s['vsize'], 'rva': s['rva']}
              for s in meta['dll']['sections']]
    delta = neko_rva - meta['image_first_rva']
    target = out if out is not None else pdb
    if Path(target).resolve() == Path(pdb).resolve():
        raise InjectRefusal(
            f'the rebased PDB would overwrite the one it is made from ({pdb}). A PDB can only be '
            'rebased once: the section table it matches is the payload image\'s, and after the move '
            'it no longer matches, so a second pass refuses. Point pdb_out somewhere else -- the '
            'shipped copy belongs beside the image, the build\'s own copy belongs in the build '
            'directory')
    db = pdbedit.Pdb.from_path(pdb)
    report = db.rebase(expect, delta)
    db.save(target)
    again = pdbedit.Pdb.from_path(target)
    first = again.headers_at(report['stream'], report['offset'], len(expect))[0]
    report.update({'written': str(target), 'delta': delta,
                   'image_first_rva': meta['image_first_rva'], 'neko_rva': neko_rva,
                   'readback_text_rva': first.rva,
                   'readback_ok': first.rva == meta['dll']['sections'][0]['rva'] + delta})
    if not report['readback_ok']:
        report['ok'] = False
        report['problems'].append(f'read back {first.rva:#x} for .text, expected '
                                 f'{meta["dll"]["sections"][0]["rva"] + delta:#x}')
    return report


def verify(host: peedit.Pe, out: bytes, neko_rva: int, entry_rva: int,
           debug: dict | None = None) -> dict:
    """Re-parse the written file and check every field the injection changed."""
    pe = peedit.Pe(out, 'patched')
    problems: list[str] = []
    if pe.machine != host.machine:
        problems.append(f'machine changed {host.machine:#x} -> {pe.machine:#x}')
    extra = [s.name for s in pe.sections if s.name in ('.neko.r', '.neko.e', '.neko.b')
             and host.section(s.name) is None]
    expected_sections = host.nsec + 1 + len(extra)
    if pe.nsec != expected_sections:
        problems.append(f'section count {pe.nsec} != {expected_sections}')
    if pe.entry_rva != entry_rva:
        problems.append(f'entry rva {pe.entry_rva:#x} != {entry_rva:#x}')
    neko = pe.section(payload_mod.SECTION_NAME)
    if neko is None:
        problems.append(f'no {payload_mod.SECTION_NAME} section in the output')
    else:
        if neko.rva != neko_rva:
            problems.append(f'.neko rva {neko.rva:#x} != {neko_rva:#x}')
        if neko.chars != payload_mod.SECTION_CHARS:
            problems.append(f'.neko characteristics {neko.chars:#010x} != '
                            f'{payload_mod.SECTION_CHARS:#010x}')
        if neko.writable:
            problems.append('.neko is writable; it must be R+X only')
    if not (pe.size_of_image >= max(s.rva + max(s.vsize, s.rsize) for s in pe.sections)):
        problems.append('SizeOfImage does not cover the last section')
    if pe.size_of_image % pe.section_alignment:
        problems.append('SizeOfImage is not SectionAlignment aligned')
    # Every original section must be byte-identical **except** the debug directory entry this
    # injection deliberately repoints: exactly twelve bytes of `{size, address, pointer}` when the
    # host had a CodeView entry (or the directory's own `(rva, size)` pair when it did not).
    allowed: list[tuple[str, int, int]] = []
    if debug and debug.get('entry_rva'):
        sec = host.section_at(debug['entry_rva'])
        if sec is not None:
            start = debug['entry_rva'] - sec.rva
            allowed.append((sec.name, start + (4 if debug.get('replaced_directory') else 16),
                            start + 28))
    changed: list[str] = []
    for s in host.sections:
        a = host.data[s.raw:s.raw + s.rsize]
        b = pe.data[s.raw:s.raw + s.rsize]
        if a == b:
            continue
        diffs = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
        windows = [(lo, hi) for (name, lo, hi) in allowed if name == s.name]
        if diffs and all(any(lo <= d < hi for (lo, hi) in windows) for d in diffs):
            changed.append(f'{s.name}: {len(diffs)} byte(s) in the debug directory entry')
            continue
        problems.append(f'original section {s.name} changed at '
                        f'{", ".join(f"{d:#x}" for d in diffs[:4])}')
    # the checksum must describe the file as written
    if peedit.pe_checksum(out, pe.checksum_off) != pe.checksum:
        problems.append('the recomputed checksum does not match the file')
    return {'ok': not problems, 'problems': problems,
            'sections': len(pe.sections), 'size_of_image': pe.size_of_image,
            'entry': pe.entry_rva, 'checksum': pe.checksum, 'intended_changes': changed}


def main(argv: list[str]) -> int:
    host: Path | None = None
    payload_dir: Path | None = None
    out: Path | None = None
    pdb_src: Path | None = None
    pdb_dst: Path | None = None
    keep_cert = False
    keep_sum = False
    for a in argv:
        if a.startswith('--host='):
            host = Path(a.split('=', 1)[1])
        elif a.startswith('--payload='):
            payload_dir = Path(a.split('=', 1)[1])
        elif a.startswith('--out='):
            out = Path(a.split('=', 1)[1])
        elif a.startswith('--pdb-out='):
            pdb_dst = Path(a.split('=', 1)[1])
        elif a.startswith('--pdb='):
            pdb_src = Path(a.split('=', 1)[1])
        elif a == '--keep-certificate':
            keep_cert = True
        elif a == '--keep-checksum':
            keep_sum = True
        elif a in ('-h', '--help'):
            print(__doc__)
            return 2
    if host is None or payload_dir is None:
        print(__doc__)
        return 2
    meta = json.loads((payload_dir / 'payload.json').read_text(encoding='ascii'))
    data = (payload_dir / 'payload.bin').read_bytes()
    target = out or (payload_dir / f'{host.stem}.patched{host.suffix}')
    if pdb_src is not None:
        cv = meta.get('codeview') or {}
        # The name the image's RSDS record carries is the name a loader looks for, so the rebased copy
        # lands beside the image under exactly that name.
        pdb_dst = pdb_dst or (target.parent / (cv.get('path') or 'payload.pdb'))
    try:
        patched, report = inject(host, data, meta, strip_certificate=not keep_cert,
                                 keep_checksum=keep_sum, label=str(host),
                                 pdb=pdb_src, pdb_out=pdb_dst)
    except (InjectRefusal, peedit.PeError, pdbedit.PdbError) as exc:
        print(f'REFUSED: {exc}')
        return 1
    peedit.write_file(target, patched)
    (Path(str(target) + '.json')).write_text(json.dumps(report, indent=2), encoding='ascii')
    print(f"{report['host_bytes']} -> {report['out_bytes']} bytes   "
          f".neko at rva {report['neko_rva']:#x} ({report['payload_bytes']} B, chars "
          f"{report['neko_chars']:#010x})")
    print(f"  entry {report['host_entry_rva']:#x} -> {report['entry_rva']:#x}   "
          f"functions {report['nfunc']}   relocs written {report['n_relocs']}")
    v = report['verify']
    print(f"  verify: {'OK' if v['ok'] else 'PROBLEMS'}  {v.get('problems') or ''}"
          + (f"  {v['intended_changes']}" if v.get('intended_changes') else ''))
    dbg = report.get('debug') or {}
    if dbg.get('source'):
        print(f"  codeview: {dbg['source']} entry #{dbg['entry_index']} -> record at "
              f"{dbg['record_rva']:#x} ({dbg['size']} B, guid {dbg['guid']} age {dbg['age']})")
    pdb_report = dbg.get('pdb')
    if pdb_report:
        print(f"  pdb: {pdb_report['written']} -- {pdb_report['sections']} section(s) moved by "
              f"{pdb_report['delta']:#x}, .text read back at {pdb_report['readback_text_rva']:#x}"
              f" ({'OK' if pdb_report['ok'] else 'PROBLEMS'})")
        for problem in pdb_report.get('problems') or []:
            print(f'    PROBLEM {problem}')
    print(f'  wrote {target}')
    return 0 if v['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
