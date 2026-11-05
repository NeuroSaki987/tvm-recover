import struct
import sys
import os
import shutil
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guestir as G
import lower_llvm as L
import whole as W


def listing():
    return G.parse_listing('; devirtualized 0x1000\n; native bits: 32\n'
        '0x1000@ctx0x0: [entry] ; #0\n'
        '    v1:32 = trunc32(in.rax)\n'
        '    v2:32 = v1:32 + 0x1\n'
        '    guest_rax_1 = zext64(v2:32)   ; home eax\n'
        '    -> return\n')


def test_i386_listing_preserves_target_through_both_emitters():
    fn = listing()
    assert fn.header['native_bits'] == 32
    import llvmlite.binding as llvm
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    for emit in (L.emit_function, W.emit_whole):
        text, manifest = emit(fn)
        assert 'target triple = "i386-pc-windows-msvc"' in text
        module = llvm.parse_assembly(text)
        module.verify()
        target = llvm.Target.from_triple(module.triple).create_target_machine(cpu='i686', codemodel='default')
        module.data_layout = str(target.target_data)
        obj = target.emit_object(module)
        assert struct.unpack_from('<H', obj)[0] == 0x14c


def test_unknown_native_width_is_rejected():
    fn = listing()
    fn.header['native_bits'] = 16
    import pytest
    with pytest.raises(L.HazardError, match='native width'):
        L.emit_function(fn)


def test_toolchain_infers_i386_object_from_module(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'toolchain_spike'))
    import toolchain
    text, _ = L.emit_function(listing())
    obj = tmp_path / 'native32.obj'
    toolchain.emit_object(text, str(obj))
    assert struct.unpack_from('<H', obj.read_bytes())[0] == 0x14c


def test_toolchain_selects_an_i386_default_cpu():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'toolchain_spike'))
    import toolchain
    text, _ = L.emit_function(listing())

    facts = toolchain.target_facts(text)

    assert facts['triple'] == 'i386-pc-windows-msvc'
    assert facts['cpu'] == 'i686'
    assert 'p:32:' in facts['data_layout']


def test_native_projection_cannot_be_mistaken_for_recovered_guest_code():
    fn = listing()
    fn.header['state_projection'] = 'native_i386'
    import pytest
    for emit in (L.emit_function, W.emit_whole):
        with pytest.raises(L.HazardError, match='no proved Guest backing layout'):
            emit(fn)


def test_payload_builds_a_real_i386_section_and_rejects_a_width_mismatch(tmp_path):
    import payload
    import peedit
    import peinject
    import pytest
    source = tmp_path / 'native32.txt'
    source.write_text('; devirtualized 0x1000\n; native bits: 32\n'
                      '0x1000@ctx0x0: [entry] ; #0\n'
                      '    v1:32 = trunc32(in.rax)\n'
                      '    v2:32 = v1:32 + 0x1\n'
                      '    guest_rax_1 = zext64(v2:32)   ; home eax\n'
                      '    -> return\n', encoding='utf-8')
    meta = payload.build([source], tmp_path / 'payload32', split=0)
    assert meta['native_bits'] == 32
    assert peedit.Pe.from_path(tmp_path / 'payload32' / 'build' / 'payload.dll').machine == 0x14c
    assert meta['links']['relocation_type'] == peedit.REL_HIGHLOW
    assert meta['links']['relocations'] == len(meta['links']['relocation_sites'])

    # The injector validates the width from the payload header rather than
    # accepting any PE and discovering the mismatch after the entry is patched.
    host = tmp_path / 'fake64.exe'
    host.write_bytes(b'MZ')
    with pytest.raises(Exception):
        peinject.inject(host, (tmp_path / 'payload32' / 'payload.bin').read_bytes(), meta)


def test_i386_payload_injects_into_a_real_pe32_host_and_returns_to_oep(tmp_path):
    import payload
    import peedit
    import peinject
    if os.name != 'nt' or not shutil.which('cl.exe') or not shutil.which('link.exe'):
        import pytest
        pytest.skip('MSVC Windows tools required')
    cl32 = Path(shutil.which('cl.exe')).parent.parent / 'x86' / 'cl.exe'
    link32 = Path(shutil.which('link.exe')).parent.parent / 'x86' / 'link.exe'
    sdk = Path(os.environ.get('WindowsSdkDir', r'C:/Program Files (x86)/Windows Kits/10'))
    libraries = sorted((sdk / 'Lib').glob('*/um/x86/kernel32.lib'))
    if not cl32.exists() or not link32.exists() or not libraries:
        import pytest
        pytest.skip('MSVC x86 compiler and x86 Windows SDK required')

    source = tmp_path / '0x1000.txt'
    source.write_text(
        '; devirtualized 0x1000\n; native bits: 32\n'
        '0x1000@ctx0x0: [entry] ; #0\n'
        '    v1:32 = trunc32(in.rax)\n'
        '    v2:32 = v1:32 + 0x1\n'
        '    guest_rax_1 = zext64(v2:32)   ; home eax\n'
        '    -> return\n', encoding='utf-8')
    meta = payload.build([source], tmp_path / 'payload', split=0)

    host_src = tmp_path / 'host32.c'
    host_src.write_text(
        'typedef unsigned long DWORD;\n'
        '__declspec(dllimport) void __stdcall ExitProcess(DWORD);\n'
        'volatile DWORD marker;\n'
        'void entry(void) { marker = 7; ExitProcess(marker == 7 ? 0 : 1); }\n'
        'void second(void) { marker = 9; }\n', encoding='ascii')
    host_obj = tmp_path / 'host32.obj'
    host_exe = tmp_path / 'host32.exe'
    for cmd, what in (
            ([str(cl32), '/c', '/nologo', '/O2', '/GS-', '/Zl',
              f'/Fo{host_obj}', str(host_src)], 'cl32'),
            ([str(link32), '/NODEFAULTLIB', '/MACHINE:X86', '/SUBSYSTEM:WINDOWS',
              '/FIXED:NO', '/ALIGN:0x1000', '/FILEALIGN:0x1000', '/ENTRY:entry',
              f'/OUT:{host_exe}', str(host_obj), str(libraries[-1])], 'link32')):
        result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
        assert result.returncode == 0, f'{what}: {result.stdout}\n{result.stderr}'

    patched, report = peinject.inject(
        host_exe, (tmp_path / 'payload' / 'payload.bin').read_bytes(), meta,
        label='synthetic-pe32')
    assert report['verify']['ok'], report['verify']['problems']
    assert report['native_bits'] == 32
    assert report['reloc_type'] == 'HIGHLOW'
    # Do not put words such as "patch", "setup", or "update" in the name:
    # Windows installer detection can promote the process to require
    # elevation even though this synthetic image has no manifest.
    target = tmp_path / 'host32.neko.exe'
    target.write_bytes(patched)
    result = subprocess.run(
        [str(Path(os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe'))),
         '/c', str(target)], capture_output=True, text=True, errors='replace', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

    # Compile the kernel-style shape so tvm_entry32.c actually addresses the
    # external state block. A 32-bit RVA is exactly one pointer wide and must
    # not overwrite the following generated data with a stray upper dword.
    state_meta = payload.build(
        [source], tmp_path / 'payload_section', split=0, state='section')
    state_offset = state_meta['api']['tvm_state_rva']
    payload = bytearray(
        (tmp_path / 'payload_section' / 'payload.bin').read_bytes())
    # Make the linker's four-byte alignment padding observable so the test
    # distinguishes a 4-byte patch from an accidental 8-byte patch.
    payload[state_offset + 4:state_offset + 8] = b'\xdd\xcc\xbb\xaa'
    payload_bin = bytes(payload)
    original_tail = payload_bin[state_offset + 4:state_offset + 8]
    patched_state, state_report = peinject.inject(
        host_exe, payload_bin, state_meta, label='synthetic-pe32-state')
    assert state_report['verify']['ok'], state_report['verify']['problems']
    state_pe = peedit.Pe(patched_state)
    expected_state_rva = state_report['state_section']['rva']
    assert state_pe.image_bytes(
        state_report['neko_rva'] + state_offset, 4
    ) == struct.pack('<I', expected_state_rva)
    assert state_pe.image_bytes(
        state_report['neko_rva'] + state_offset + 4, 4
    ) == original_tail
    state_target = tmp_path / 'host32.neko-state.exe'
    state_target.write_bytes(patched_state)
    result = subprocess.run(
        [str(Path(os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe'))),
         '/c', str(state_target)],
        capture_output=True, text=True, errors='replace', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(('expression', 'expected'), [
    ('v1:64 / v2:64', '0x7fffffffffffffffULL'),
    ('v1:64 % v2:64', '0x1ULL'),
    ('v1:64 * v2:64', '0xfffffffffffffffeULL'),
    ('v1:64 << v2:64', '0xfffffffffffffffcULL'),
    ('v1:64 >> v2:64', '0x3fffffffffffffffULL'),
], ids=['div64', 'rem64', 'mul64', 'shl64', 'shr64'])
def test_i386_payload_runtime_executes_i64_helpers(tmp_path, expression, expected):
    import payload
    if os.name != 'nt' or not shutil.which('cl.exe') or not shutil.which('link.exe'):
        pytest.skip('MSVC Windows tools required')
    cl32 = Path(shutil.which('cl.exe')).parent.parent / 'x86' / 'cl.exe'
    link32 = Path(shutil.which('link.exe')).parent.parent / 'x86' / 'link.exe'
    sdk = Path(os.environ.get('WindowsSdkDir', r'C:/Program Files (x86)/Windows Kits/10'))
    libraries = sorted((sdk / 'Lib').glob('*/um/x86/kernel32.lib'))
    if not cl32.exists() or not link32.exists() or not libraries:
        pytest.skip('MSVC x86 compiler and x86 Windows SDK required')

    source = tmp_path / '0x1000.txt'
    source.write_text(
        '; devirtualized 0x1000\n; native bits: 32\n'
        '0x1000@ctx0x0: [entry] ; #0\n'
        '    v1:64 = in.rax\n'
        '    v2:64 = in.rcx\n'
        f'    v3:64 = {expression}\n'
        '    guest_rax_1 = v3:64   ; home eax\n'
        '    -> return\n', encoding='utf-8')
    out = tmp_path / 'payload'
    payload.build([source], out, split=0)

    harness = tmp_path / 'runtime32.c'
    harness.write_text(f'''
typedef unsigned long long u64;
extern long long tvm_1000(void*, void*, long long);
int mainCRTStartup(void) {{
    u64 ctx[17];
    unsigned i;
    for (i = 0; i != 17; ++i) ctx[i] = 0;
    ctx[0] = 0xffffffffffffffffULL;
    ctx[2] = 2;
    if (tvm_1000(ctx, 0, 2000) != 1) return 1;
    if (ctx[0] != {expected}) return 2;
    return 0;
}}
''', encoding='ascii')
    obj = tmp_path / 'runtime32.obj'
    exe = tmp_path / 'runtime32.exe'
    build = out / 'build'
    objects = [
        'payload.obj', 'tvm_runtime.obj', 'tvm_sealed.obj', 'tvm_gen.obj',
        'tvm_entry32.obj', 'tvm_chkstk_x86.obj', 'tvm_entry32_c.obj',
        'tvm_i386_helpers.obj',
    ]
    compile_result = subprocess.run(
        [str(cl32), '/nologo', '/c', '/O2', '/GS-', '/Zl',
         f'/Fo{obj}', str(harness)],
        capture_output=True, text=True, errors='replace')
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    link_result = subprocess.run(
        [str(link32), '/nologo', '/MACHINE:X86', '/NODEFAULTLIB',
         '/SUBSYSTEM:CONSOLE', '/ENTRY:mainCRTStartup', f'/OUT:{exe}',
         str(obj), *(str(build / name) for name in objects), str(libraries[-1])],
        capture_output=True, text=True, errors='replace')
    assert link_result.returncode == 0, link_result.stdout + link_result.stderr
    run_result = subprocess.run([str(exe)], capture_output=True,
                                text=True, errors='replace', timeout=30)
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr


def test_i386_msvc_integer_helpers_clean_stack_parameters(tmp_path):
    if os.name != 'nt' or not shutil.which('cl.exe') or not shutil.which('link.exe'):
        pytest.skip('MSVC Windows tools required')
    bins = Path(shutil.which('cl.exe')).parent.parent / 'x86'
    cl32, link32, ml32 = bins / 'cl.exe', bins / 'link.exe', bins / 'ml.exe'
    if not cl32.exists() or not link32.exists() or not ml32.exists():
        pytest.skip('MSVC x86 toolchain required')

    ops = tmp_path / 'ops.c'
    ops.write_text('''
typedef unsigned long long u64;
u64 mul64(u64 a, u64 b) { return a * b; }
u64 shl64(u64 a, unsigned n) { return a << n; }
u64 shr64(u64 a, unsigned n) { return a >> n; }
u64 rem64(u64 a, u64 b) { return a % b; }
u64 div64(u64 a, u64 b) { return a / b; }
''', encoding='ascii')
    harness = tmp_path / 'harness.c'
    harness.write_text('''
typedef unsigned long long u64;
extern u64 mul64(u64, u64);
extern u64 shl64(u64, unsigned);
extern u64 shr64(u64, unsigned);
extern u64 rem64(u64, u64);
extern u64 div64(u64, u64);
int mainCRTStartup(void) {
    u64 a = 0xffffffffffffffffULL;
    if (mul64(a, 2) != 0xfffffffffffffffeULL) return 1;
    if (shl64(a, 1) != 0xfffffffffffffffeULL) return 2;
    if (shr64(a, 1) != 0x7fffffffffffffffULL) return 3;
    if (rem64(a, 2) != 1) return 4;
    if (div64(a, 2) != 0x7fffffffffffffffULL) return 5;
    return 0;
}
''', encoding='ascii')
    ops_obj, harness_obj = tmp_path / 'ops.obj', tmp_path / 'harness.obj'
    helper_obj, exe = tmp_path / 'helpers.obj', tmp_path / 'helpers.exe'
    commands = [
        [str(cl32), '/nologo', '/c', '/O2', '/GS-', '/Zl',
         f'/Fo{ops_obj}', str(ops)],
        [str(cl32), '/nologo', '/c', '/O2', '/GS-', '/Zl',
         f'/Fo{harness_obj}', str(harness)],
        [str(ml32), '/c', '/nologo', '/coff', f'/Fo{helper_obj}',
         str(Path(__file__).resolve().parents[1] / 'runtime' / 'tvm_i386_helpers.asm')],
        [str(link32), '/nologo', '/MACHINE:X86', '/NODEFAULTLIB',
         '/SUBSYSTEM:CONSOLE', '/ENTRY:mainCRTStartup', f'/OUT:{exe}',
         str(ops_obj), str(harness_obj), str(helper_obj)],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True,
                            errors='replace', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_i386_chkstk_probes_and_allocates_from_native_stack_pointer(tmp_path):
    if os.name != 'nt' or not shutil.which('cl.exe') or not shutil.which('link.exe'):
        import pytest
        pytest.skip('MSVC Windows tools required')
    bins = Path(shutil.which('cl.exe')).parent.parent / 'x86'
    cl32, link32, ml32 = bins / 'cl.exe', bins / 'link.exe', bins / 'ml.exe'
    if not cl32.exists() or not link32.exists() or not ml32.exists():
        import pytest
        pytest.skip('MSVC x86 toolchain required')

    harness_c = tmp_path / 'chkstk_harness.c'
    harness_c.write_text('''
extern int __cdecl run_probe(void*);
static unsigned char probe_memory[0x4000];
int mainCRTStartup(void) {
    void* top = probe_memory + 0x3000;
    return run_probe(top) == 1 ? 0 : 1;
}
''', encoding='ascii')
    harness_asm = tmp_path / 'chkstk_harness.asm'
    harness_asm.write_text('''
.386
.model flat
_TEXT SEGMENT DWORD PUBLIC 'CODE' USE32
EXTERN __chkstk:PROC
PUBLIC _run_probe
_run_probe PROC
    push    ebx
    push    esi
    mov     ebx, esp
    mov     esi, DWORD PTR [esp+12]
    ; The old implementation reads [esp+8] after __chkstk's push.
    ; Make that unrelated slot an unmapped address.
    mov     DWORD PTR [esi], 40000000h
    mov     eax, 2000h
    mov     esp, esi
    call    __chkstk
    mov     edx, esi
    sub     edx, 2000h
    cmp     esp, edx
    jne     probe_bad_esp
    mov     eax, 1
    mov     esp, ebx
    pop     esi
    pop     ebx
    ret
probe_bad_esp:
    mov     esp, ebx
    pop     esi
    pop     ebx
    xor     eax, eax
    ret
_run_probe ENDP
_TEXT ENDS
END
''', encoding='ascii')
    helper_asm = Path(__file__).resolve().parents[1] / 'runtime' / 'tvm_chkstk32.asm'
    harness_obj = tmp_path / 'chkstk_harness.obj'
    assembly_obj = tmp_path / 'chkstk_assembly.obj'
    helper_obj = tmp_path / 'chkstk_helper.obj'
    exe = tmp_path / 'chkstk_harness.exe'
    commands = [
        [str(cl32), '/nologo', '/c', '/O2', '/GS-', '/Zl',
         f'/Fo{harness_obj}', str(harness_c)],
        [str(ml32), '/c', '/nologo', '/coff', f'/Fo{assembly_obj}',
         str(harness_asm)],
        [str(ml32), '/c', '/nologo', '/coff', f'/Fo{helper_obj}',
         str(helper_asm)],
        [str(link32), '/nologo', '/MACHINE:X86', '/NODEFAULTLIB',
         '/SUBSYSTEM:CONSOLE', '/ENTRY:mainCRTStartup', f'/OUT:{exe}',
         str(harness_obj), str(assembly_obj), str(helper_obj)],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors='replace')
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True,
                            errors='replace', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_i386_emitted_function_runs_in_a_32_bit_synthetic_harness(tmp_path):
    import os
    import shutil
    import subprocess
    import pytest
    if os.name != 'nt' or not shutil.which('cl.exe') or not shutil.which('link.exe'):
        pytest.skip('MSVC Windows tools required')
    cl32 = Path(shutil.which('cl.exe')).parent.parent / 'x86' / 'cl.exe'
    sdk = Path(os.environ.get('WindowsSdkDir', r'C:/Program Files (x86)/Windows Kits/10'))
    libraries = sorted((sdk / 'Lib').glob('*/um/x86/kernel32.lib'))
    if not cl32.exists() or not libraries:
        pytest.skip('MSVC x86 compiler and x86 Windows SDK required')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'toolchain_spike'))
    import toolchain
    text, _ = W.emit_whole(listing())
    ir_obj = tmp_path / 'transfer.obj'
    toolchain.emit_object(text, str(ir_obj))
    # The executable contains only a synthetic increment transfer and this
    # harness. No instructions from a protected sample are executed.
    source = tmp_path / 'harness.c'
    source.write_text('''
typedef unsigned long long u64;
__declspec(dllimport) void __stdcall ExitProcess(unsigned int);
extern u64 tvm_1000(u64*, u64);
void mainCRTStartup(void) {
    u64 context[17];
    volatile u64* clear = context;
    unsigned i;
    for (i = 0; i != 17; ++i) clear[i] = 0;
    context[0] = 0xffffffffULL;
    if (tvm_1000(context, 4) != 1 || context[0] != 0) ExitProcess(1);
    context[0] = 5;
    if (tvm_1000(context, 4) != 1 || context[0] != 6) ExitProcess(2);
    ExitProcess(0);
}
''', encoding='ascii')
    native_obj = tmp_path / 'harness.obj'
    exe = tmp_path / 'harness.exe'
    commands = [
        [str(cl32), '/nologo', '/c', '/O2', '/GS-', '/Zl', f'/Fo{native_obj}', str(source)],
        [shutil.which('link.exe'), '/nologo', '/MACHINE:X86', '/NODEFAULTLIB',
         '/SUBSYSTEM:CONSOLE', '/ENTRY:mainCRTStartup', f'/OUT:{exe}',
         str(native_obj), str(ir_obj), str(libraries[-1])],
        [str(exe)],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
