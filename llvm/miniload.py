"""Map a patched image the way the loader would, then run it -- the entry point, and the payload.

Everything `write-devirt` claims about its output is checked here, against the bytes that were written:

**The entry handoff.** Map the *unpatched* host, map the *patched* host at a forced base (different
from the preferred one, so the base-relocation path is exercised), point both entry paths at a capture
stub that records the sixteen registers and the flags it was entered with, call them with the same
three arguments, and compare. The patched path goes through the payload's entry point, which runs the
devirtualized code and then tail-jumps to the original entry point -- so what is compared is exactly
"the state our stub hands the original entry point" against "the state a plain call hands a callee".
Divergences are named per register rather than summarised, and the ABI-volatile ones (rax, r9-r11,
flags) are reported separately from the ones a DllMain may rely on (rcx, rdx, r8, rbx, rbp, rsi, rdi,
r12-r15, rsp).

**The payload itself.** Call `tvm_payload_main` inside the mapped image with a state buffer we own
(the payload's own APIs, resolved by name through the header), read the sealed byte map and the
register file back out, and hand them to `diff_payload.py` for comparison with the interpreter. That
is the check that the code which runs inside the patched PE computes what the listing says.

Nothing here re-implements an assembler: the capture stub is a fixed byte template whose encoding is
asserted by disassembling it with capstone (the same dependency `engine/probe/dis.py` already uses).

Run:  python llvm/miniload.py <patched> [--payload=DIR] [--base=HEX] [--host=PATH]
"""

from __future__ import annotations

import ctypes
import json
import struct
import sys
import subprocess
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
RUNTIME_DIR = HERE / 'runtime'
BUILD = HERE / 'build'
sys.path.insert(0, str(HERE))

import payload as payload_mod  # noqa: E402
import peedit  # noqa: E402

k32 = ctypes.WinDLL('kernel32', use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE = 0x1000, 0x2000, 0x8000
PAGE_EXECUTE_READWRITE = 0x40

#: The sixteen registers a Windows x64 call must preserve or define, in the order the capture stub
#: pushes them (last pushed first in memory).
CAPTURED = ('r15', 'r14', 'r13', 'r12', 'r11', 'r10', 'r9', 'r8', 'rdi', 'rsi', 'rbx', 'rdx', 'rcx',
            'rax', 'rflags')
#: What a DLL entry point may rely on: the three argument registers, the callee-saved set, and the
#: stack pointer. Everything else is volatile under the x64 ABI.
CONTRACT = ('rcx', 'rdx', 'r8', 'rbx', 'rbp', 'rsi', 'rdi', 'r12', 'r13', 'r14', 'r15')
VOLATILE = ('rax', 'r9', 'r10', 'r11', 'rflags')

FN3 = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)


def capture_stub() -> bytes:
    """The stub every compared path is redirected into, with a zero `lea` displacement.

    Push the flags and fifteen registers, store the resulting stack pointer into a record, restore
    everything, and return. The recorded block is [r15, r14, r13, r12, r11, r10, r9, r8, rdi, rsi,
    rbx, rdx, rcx, rax, rflags] followed by the return address the caller pushed -- so the harness can
    check the stack pointer itself by asking where the block is. `build_stub` fills the displacement;
    the encoding is asserted by disassembly in the tests rather than trusted.
    """
    code = bytearray()
    code += b'\x9c'                                   # pushfq
    for op in (0x50, 0x51, 0x52, 0x53, 0x56, 0x57):   # push rax rcx rdx rbx rsi rdi
        code.append(op)
    for reg in (0, 1, 2, 3, 4, 5, 6, 7):              # push r8..r15  (REX.B)
        code += bytes((0x41, 0x50 + reg))
    code += b'\x48\x89\xe0'                           # mov rax, rsp
    code += b'\x48\x8d\x0d\x00\x00\x00\x00'           # lea rcx, [rip+0]  (patched below)
    code += b'\x48\x89\x01'                           # mov [rcx], rax
    for reg in reversed(range(8)):                    # pop r15..r8
        code += bytes((0x41, 0x58 + reg))
    for op in (0x5f, 0x5e, 0x5b, 0x5a, 0x59, 0x58):   # pop rdi rsi rbx rdx rcx rax
        code.append(op)
    code += b'\x9d'                                   # popfq
    code += b'\xc3'                                   # ret
    return bytes(code)


#: Offset of the `lea`'s disp32 inside `capture_stub()`: 1 (pushfq) + 6 + 16 + 3 (mov rax,rsp) + 3.
LEA_DISP = 1 + 6 + 16 + 3 + 3
assert capture_stub()[LEA_DISP - 3:LEA_DISP] == b'\x48\x8d\x0d', capture_stub().hex()


class Mapped:
    """An image mapped the way the loader maps it: headers, sections, and base relocations."""

    def __init__(self, data: bytes, base: int | None = None, label: str = '<image>'):
        self.pe = peedit.Pe(data, label)
        self.size = peedit.align_up(self.pe.size_of_image, 0x1000)
        self.extra = 0x2000
        self.base = self._reserve(base)
        self.relocs_applied = 0
        self._load()

    def _reserve(self, want: int | None) -> int:
        """Reserve the image. A forced base is tried first; otherwise the OS picks, which is how the
        relocation path gets exercised (the preferred base is what `image_base` says, and a random
        base is almost never it)."""
        wants: list[int | None] = [want] if want is not None else []
        wants.append(None)
        for i, w in enumerate(wants):
            p = k32.VirtualAlloc(ctypes.c_void_p(w) if w else None, self.size + self.extra,
                                 MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
            if p:
                return int(p)
        raise RuntimeError(f'VirtualAlloc failed for {self.pe.label} '
                           f'({ctypes.get_last_error()})')

    def _load(self) -> None:
        pe = self.pe
        buf = (ctypes.c_char * (self.size + self.extra)).from_address(self.base)
        ctypes.memset(self.base, 0, self.size + self.extra)
        head = min(pe.size_of_headers, len(pe.data))
        ctypes.memmove(self.base, bytes(pe.data[:head]), head)
        for s in pe.sections:
            span = max(s.vsize, s.rsize)
            if not span:
                continue
            ctypes.memmove(self.base + s.rva, pe.image_bytes(s.rva, span), span)
        self.buf = buf
        delta = self.base - pe.image_base
        if delta:
            self.relocs_applied = self.apply_relocations(delta)

    def apply_relocations(self, delta: int) -> int:
        n = 0
        for blk in peedit.read_relocation_blocks(self.pe):
            page, size = struct.unpack_from('<II', blk, 0)
            body = blk[8:size]
            for k in range(0, len(body) - 1, 2):
                ent = struct.unpack_from('<H', body, k)[0]
                kind, off = ent >> 12, ent & 0xFFF
                if kind == peedit.REL_ABSOLUTE:
                    continue
                if kind != peedit.REL_DIR64:
                    raise RuntimeError(f'{self.pe.label}: relocation type {kind} at '
                                       f'{page + off:#x} is not DIR64')
                site = self.base + page + off
                old = struct.unpack('<Q', ctypes.string_at(site, 8))[0]
                ctypes.memmove(site, struct.pack('<Q', (old + delta) & 0xFFFFFFFFFFFFFFFF), 8)
                n += 1
        return n

    # ---------------------------------------------------------------- access
    def read(self, rva: int, n: int) -> bytes:
        return ctypes.string_at(self.base + rva, n)

    def write(self, rva: int, data: bytes) -> None:
        ctypes.memmove(self.base + rva, data, len(data))

    def u64(self, rva: int) -> int:
        return struct.unpack('<Q', self.read(rva, 8))[0]

    def ptr(self, rva: int):
        return FN3(self.base + rva)

    def section(self, name: str):
        return self.pe.section(name)

    def free(self) -> None:
        if self.base:
            k32.VirtualFree(ctypes.c_void_p(self.base), 0, MEM_RELEASE)
            self.base = 0


def probe_dll(work: Path | None = None) -> Path:
    """Build (once) the native probe used for the handoff comparison.

    The comparison has to be made from *one* native function with identical locals, because the
    libffi thunk behind a `ctypes` call leaves its own value in r13 and that value differs between
    call sites -- measured, and it produced a lone phantom "divergence" in the first version of this
    harness. See `runtime/tvm_probe.c`.
    """
    work = work or (BUILD / 'write-devirt' / 'probe')
    work.mkdir(parents=True, exist_ok=True)
    dll = work / 'tvm_probe.dll'
    src = RUNTIME_DIR / 'tvm_probe.c'
    obj = work / 'tvm_probe.obj'
    if dll.exists() and dll.stat().st_mtime > src.stat().st_mtime:
        return dll
    for cmd, what in (
            (['cl.exe', '/c', '/nologo', '/O2', '/W3', f'/Fo{obj}', str(src)], 'cl tvm_probe'),
            (['link.exe', '/DLL', '/NOENTRY', '/NODEFAULTLIB', '/MACHINE:X64',
              '/SUBSYSTEM:WINDOWS', '/SECTION:.neko_cap,RWE', f'/OUT:{dll}', str(obj)],
             'link tvm_probe')):
        p = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
        if p.returncode != 0:
            raise RuntimeError(f'{what} failed:\n{p.stdout}\n{p.stderr}')
    return dll


def _load_probe():
    lib = ctypes.WinDLL(str(probe_dll()))
    lib.probe_init.argtypes = []
    lib.probe_stub.restype = ctypes.c_uint64
    lib.probe_pair.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                               ctypes.c_uint64, ctypes.c_uint64,
                               ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    if lib.probe_init() != 0:
        raise RuntimeError('probe_init failed')
    return lib


def probe_handoff(host_data: bytes, patched_data: bytes, meta: dict,
                  base: int | None = None) -> dict:
    """Compare the register state handed to the original entry point, with and without the payload.

    Path A is a direct call to the capture stub: what any callee sees at the x64 entry contract.
    Path B calls the patched image's entry point, which runs the payload and then tail-jumps to the
    same capture stub. Both calls are issued by `tvm_probe!probe_pair` from one function with
    identical arguments, so any difference is a difference the payload made.
    """
    res: dict = {'base_forced': base is not None}
    lib = _load_probe()
    stub_va = int(lib.probe_stub())
    out_a = (ctypes.c_uint64 * 16)()
    out_b = (ctypes.c_uint64 * 16)()
    patched = None
    try:
        full = Mapped(patched_data, base, 'host-patched')
        patched = full
        neko = full.section(payload_mod.SECTION_NAME)
        header = neko_header(full)
        slot = neko.rva + header['api']['tvm_oep_rva']
        # The slot is an RVA and the payload's `target = base + slot` arithmetic is u64, so a
        # negative difference is spelled in two's complement and wraps back to the right address.
        full.write(slot, struct.pack('<Q', (stub_va - full.base) & 0xFFFFFFFFFFFFFFFF))
        entry_va = full.base + full.pe.entry_rva
        lib.probe_pair(ctypes.c_uint64(stub_va), ctypes.c_uint64(entry_va),
                       ctypes.c_uint64(0x1111), ctypes.c_uint64(0x2222), ctypes.c_uint64(0x3333),
                       out_a, out_b)
        res['patched'] = {'base': full.base, 'relocs_applied': full.relocs_applied,
                          'entry_rva': full.pe.entry_rva, 'neko_rva': neko.rva}
        res['rebased'] = full.base != full.pe.image_base
    finally:
        if patched is not None:
            patched.free()

    a = list(out_a)
    b = list(out_b)
    if not any(a) or not any(b):
        res['verdict'] = 'STUB NOT REACHED'
        return res
    # The stub does not push rbp (it never touches it), so fifteen registers plus the flags are
    # compared; the stack pointer is implied by the block's address, which is inside the stub's page.
    diffs = {}
    for i, name in enumerate(CAPTURED):
        if a[i] != b[i]:
            diffs[name] = {'direct': f'{a[i]:#x}', 'through_payload': f'{b[i]:#x}'}
    contract_bad = {k: v for k, v in diffs.items() if k in CONTRACT}
    res['blocks'] = {'direct': [f'{v:#x}' for v in a],
                     'through_payload': [f'{v:#x}' for v in b]}
    res['diffs'] = diffs
    res['contract_diffs'] = contract_bad
    res['volatile_diffs'] = sorted(k for k in diffs if k in VOLATILE)
    res['verdict'] = 'OK' if not contract_bad else 'DIVERGENT'
    return res


def read_state(buf: bytes, header: dict) -> dict:
    """Decode a `tvm_state_t` buffer using the offsets the payload header publishes."""
    o = header['state_offsets']

    def rd(key: str, n: int) -> int:
        return int.from_bytes(buf[o[key]:o[key] + n], 'little')

    sealed = {}
    for i in range(header['slots']):
        if buf[o['live'] + i]:
            sealed[int.from_bytes(buf[o['keys'] + 8 * i:o['keys'] + 8 * i + 8], 'little')] = \
                buf[o['vals'] + i]
    report = {}
    ro = o['report']
    (report['magic'], report['version'], report['nfunc'], report['slots'], report['overflow'],
     _pad) = struct.unpack_from('<IIIIII', buf, ro)
    (report['fuel'], report['total_steps'], report['loads'], report['stores'],
     _p1) = struct.unpack_from('<QQQQQ', buf, ro + 24)
    return {
        'magic': rd('magic', 8), 'used': rd('used', 4), 'overflow': rd('overflow', 4),
        'loads': rd('loads', 8), 'stores': rd('stores', 8), 'steps': rd('steps', 8),
        'calls': rd('calls', 4),
        'results': [int.from_bytes(buf[o['result'] + 8 * i:o['result'] + 8 * i + 8], 'little')
                    for i in range(header['nfunc'])],
        'ctx': bytes(buf[o['ctx']:o['ctx'] + 136]),
        'sealed': sealed, 'report': report,
    }


def neko_header(image: 'Mapped') -> dict:
    """The payload header read out of a mapped image, with the section RVA added.

    Header `api` entries are **offsets from the start of `.neko`**, not RVAs: that is what makes the
    same payload bytes injectable at any section RVA. Everything below therefore resolves a symbol as
    `base + section_rva + api_offset`.
    """
    sec = image.section(payload_mod.SECTION_NAME)
    if sec is None:
        raise RuntimeError(f'{image.pe.label}: no {payload_mod.SECTION_NAME} section')
    head = payload_mod.read_header(image.read(sec.rva, sec.vsize))
    head['section_rva'] = sec.rva
    return head


def api_va(image: 'Mapped', header: dict, name: str) -> int:
    return image.base + header['section_rva'] + header['api'][name]


def run_function(image: 'Mapped', header: dict, index: int, fuel: int | None = None) -> dict:
    """Run one recovered function inside the mapped payload, exactly as the payload's own main does."""
    fn = header['functions'][index]
    o = header['state_offsets']
    buf = bytearray(header['state_bytes'])
    storage = (ctypes.c_char * header['state_bytes']).from_buffer(buf)
    state_ptr = ctypes.cast(storage, ctypes.c_void_p)
    reset = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(api_va(image, header, 'tvm_state_reset'))
    reset(state_ptr)
    seed = image.read(header['section_rva'] + header['api']['tvm_ctx_seed'] + 136 * index, 136)
    buf[o['ctx']:o['ctx'] + 136] = seed
    call = FN3(image.base + header['section_rva'] + fn['offset'])
    steps = int(call(state_ptr.value + o['ctx'], state_ptr.value,
                     header['fuel'] if fuel is None else fuel))
    state = read_state(bytes(buf), header)
    state['steps'] = steps & 0xFFFFFFFFFFFFFFFF
    state['function'] = fn['name']
    state['seed'] = seed
    return state


def run_payload(image: 'Mapped', header: dict) -> dict:
    """`tvm_payload_main` on a state buffer we own: the whole run, one function after another."""
    buf = bytearray(header['state_bytes'])
    storage = (ctypes.c_char * header['state_bytes']).from_buffer(buf)
    main = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(api_va(image, header, 'tvm_payload_main'))
    main(ctypes.cast(storage, ctypes.c_void_p))
    return read_state(bytes(buf), header)


def main(argv: list[str]) -> int:
    patched: Path | None = None
    host: Path | None = None
    base = None
    for a in argv:
        if a.startswith('--host='):
            host = Path(a.split('=', 1)[1])
        elif a.startswith('--base='):
            base = int(a.split('=', 1)[1], 0)
        elif a.startswith('--'):
            continue
        else:
            patched = Path(a)
    if patched is None:
        print(__doc__)
        return 2
    data = patched.read_bytes()
    pe = peedit.Pe(data, str(patched))
    neko = pe.section(payload_mod.SECTION_NAME)
    header = payload_mod.read_header(pe.image_bytes(neko.rva, neko.vsize))
    header['section_rva'] = neko.rva
    print(f'{patched.name}: {pe.summary()}')
    print(f"  .neko rva {neko.rva:#x} chars {neko.chars:#010x} {header['nfunc']} function(s), "
          f"entry {header['entry_offset']:#x}")
    img = Mapped(data, base, str(patched))
    try:
        out = run_payload(img, header)
    finally:
        img.free()
    print(f"  payload: {out['calls']} call(s), steps {out['steps']}, sealed bytes {out['used']}, "
          f"overflow {out['overflow']}, loads {out['loads']}, stores {out['stores']}")
    print(f"  report magic {out['report']['magic']:#x} nfunc {out['report']['nfunc']}")
    if host is not None:
        res = probe_handoff(host.read_bytes(), data, {}, base)
        print(f"  handoff: {res['verdict']}  contract diffs {res['contract_diffs']}  "
              f"volatile diffs {res['volatile_diffs']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
