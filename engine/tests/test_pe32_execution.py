from types import SimpleNamespace

from core.expr import Arena, render
from core.mach import Machine
from core.lift64 import Lifter


def setup(code, stack=0x10000):
    image = SimpleNamespace(is64=False, imports={}, image_u8=lambda addr: None,
        image_bytes=lambda addr, size: code[addr-0x1000:addr-0x1000+size],
        foldable=lambda addr, size: False,
        section_for_va=lambda addr: SimpleNamespace(executable=True, writable=True, end=0x2000),
        is_loader_bound=lambda addr, size: False)
    arena = Arena()
    machine = Machine(arena, stack)
    lifter = Lifter(image, machine, arena, 0x1000, 0x2000)
    return arena, machine, lifter


def test_pe32_decodes_inc_instead_of_rex_prefix():
    a, m, l = setup(bytes.fromhex('40 90'))
    m.set_reg('eax', a.const(7, 32), 32)
    assert l.step(0x1000) == ('next', 0x1001)
    assert a.as_const(m.reg('eax', 32)) == 8


def test_pe32_push_pop_and_flags_use_four_bytes():
    a, m, l = setup(bytes.fromhex('50 5b 9c 9d'))
    m.set_reg('eax', a.const(0x12345678, 32), 32)
    l.step(0x1000)
    assert m.concrete_rsp() == 0xfffc
    l.step(0x1001)
    assert a.as_const(m.reg('ebx', 32)) == 0x12345678
    l.step(0x1002)
    assert m.concrete_rsp() == 0xfffc
    l.step(0x1003)
    assert m.concrete_rsp() == 0x10000


def test_pe32_effective_address_wraps_at_32_bits():
    a, m, l = setup(bytes.fromhex('8d 43 08'))
    m.set_reg('ebx', a.const(0xfffffffc, 32), 32)
    l.step(0x1000)
    assert a.as_const(m.reg('eax', 32)) == 4
    _, insn, _ = l.decode(0x1000)
    assert a.as_const(l.mem_address(insn)) == 4


def test_diagnostic_render_is_bounded():
    a = Arena()
    ref = a.opaque('x', 64)
    for _ in range(16):
        ref = a.bin('mul', ref, ref, 64)
    assert len(render(a, ref)) <= 4096


def test_pe32_pushad_popad_preserve_original_esp_and_skip_saved_esp():
    a, m, l = setup(bytes.fromhex('60 61'))
    for i, r in enumerate(('eax', 'ecx', 'edx', 'ebx', 'ebp', 'esi', 'edi')):
        m.set_reg(r, a.const(i + 10, 32), 32)
    l.step(0x1000)
    assert m.concrete_rsp() == 0xffe0
    assert a.as_const(m.load(a.const(0xffec, 32), 4, 0, lambda _: None)) == 0x10000
    l.step(0x1001)
    assert m.concrete_rsp() == 0x10000
    for i, r in enumerate(('eax', 'ecx', 'edx', 'ebx', 'ebp', 'esi', 'edi')):
        assert a.as_const(m.reg(r, 32)) == i + 10


def test_pe32_call_ret_immediate_balance_stack():
    a, m, l = setup(bytes.fromhex('e8 01 00 00 00 90 c2 08 00'))
    assert l.step(0x1000) == ('next', 0x1006)
    assert m.concrete_rsp() == 0xfffc
    assert l.step(0x1006) == ('next', 0x1005)
    assert m.concrete_rsp() == 0x10008


def test_pe32_return_address_thunk_rewrites_ret_on_stack():
    # call next+1; the target rewrites the pushed return address (0x1005)
    # and returns to 0x1010.
    a, m, l = setup(bytes.fromhex(
        'e8 01 00 00 00 90 83 04 24 0b c3 90 90 90 90 90 90'))
    l.summary_enabled = False
    assert l.step(0x1000) == ('next', 0x1010)
    assert m.concrete_rsp() == 0x10000
    event = l.events[-1]
    assert event['kind'] == 'native_return_thunk'
    assert event['shape'] == 'add [esp], imm; ret'
    assert event['continuation'] == 0x1010
    assert event['stack_cleanup'] == 0


def test_pe32_return_address_thunk_preserves_add_flags():
    # The rewritten return address is still produced by an architectural ADD.
    # Its arithmetic flags are visible to the continuation at 0x1010.
    a, m, l = setup(bytes.fromhex(
        'e8 01 00 00 00 90 83 04 24 0b c3 90 90 90 90 90 90'))
    l.summary_enabled = False

    assert l.step(0x1000) == ('next', 0x1010)
    assert {
        flag: a.as_const(m.flag(flag))
        for flag in ('cf', 'pf', 'af', 'zf', 'sf', 'of')
    } == {
        'cf': 0,
        'pf': 0,
        'af': 1,
        'zf': 0,
        'sf': 0,
        'of': 0,
    }


def test_pe32_cache_probe_can_disable_return_address_thunks():
    a, m, l = setup(bytes.fromhex(
        'e8 01 00 00 00 90 83 04 24 0b c3 90 90 90 90 90 90'))
    l.vm_hi = 0x1002
    l.summary_enabled = False
    l.return_address_thunks = False
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'native_call_abi'
    assert l.events[-1]['kind'] == 'call'


def test_pe32_return_address_thunk_pops_into_register_then_jumps():
    # call next+1; pop ECX, add 10 through LEA, jump ECX.
    a, m, l = setup(bytes.fromhex(
        'e8 01 00 00 00 90 59 8d 89 0a 00 00 00 ff e1 90'))
    l.summary_enabled = False
    assert l.step(0x1000) == ('next', 0x100f)
    assert m.concrete_rsp() == 0x10000
    assert a.as_const(m.reg('ecx', 32)) == 0x100f
    event = l.events[-1]
    assert event['shape'] == 'pop r32; lea r32,[r32+imm]; jmp r32'
    assert event['continuation'] == 0x100f


def test_pe32_near_return_address_thunk_shape_is_not_guessed():
    # This reaches nearby code, but it is not the call-next+1 thunk shape.
    # The ordinary callee ret can still supply a separate static contract.
    a, m, l = setup(bytes.fromhex('e8 02 00 00 00 90 90 c3'))
    l.vm_hi = 0x1002
    l.summary_enabled = False
    kind, stop = l.step(0x1000)
    assert kind == 'next' and stop == 0x1005
    event = l.events[-1]
    assert event['kind'] == 'call'
    assert event['native_return_contract']['instructions_visited'] == 1
    assert event['native_return_contract']['return_sites'] == [{'site': 0x1007}]


def test_pe32_loop_decrements_ecx_without_changing_flags():
    a, m, l = setup(bytes.fromhex('e2 fe'))
    m.set_reg('ecx', a.const(2, 32), 32)
    before = dict(m.flags)
    assert l.step(0x1000) == ('next', 0x1000)
    assert a.as_const(m.reg('ecx', 32)) == 1
    assert m.flags == before
    assert l.step(0x1000) == ('next', 0x1002)


def test_fs_address_is_not_flat_absolute_memory():
    a, m, l = setup(bytes.fromhex('64 a1 18 00 00 00'))
    _, insn, _ = l.decode(0x1000)
    addr = l.mem_address(insn)
    assert a.as_const(addr) is None
    assert 'fs_base' in render(a, addr)


def test_lodsd_loads_and_advances_esi_by_four():
    a, m, l = setup(bytes.fromhex('ad'))
    m.set_reg('esi', a.const(0x8000, 32), 32)
    m.store(a.const(0x8000, 32), a.const(0x12345678, 32), 4, 0)
    l.step(0x1000)
    assert a.as_const(m.reg('eax', 32)) == 0x12345678
    assert a.as_const(m.reg('esi', 32)) == 0x8004


def test_rep_movsd_copies_and_decrements_ecx():
    a, m, l = setup(bytes.fromhex('f3 a5'))
    for reg, val in [('esi', 0x8000), ('edi', 0x9000), ('ecx', 2)]:
        m.set_reg(reg, a.const(val, 32), 32)
    m.store(a.const(0x8000, 32), a.const(0x123456789abcdef0, 64), 8, 0)
    l.step(0x1000)
    assert a.as_const(m.load(a.const(0x9000, 32), 8, 0, lambda _: None)) == 0x123456789abcdef0
    assert a.as_const(m.reg('ecx', 32)) == 0


def test_pe32_cfg_serializes_native_state_without_x64_layout():
    from core.explore import Recoverer
    from core.gueststate import build
    a, m, l = setup(bytes.fromhex('40 50 c3'))
    r = Recoverer(l.img, 0x1000, 0x2000, step_budget=20, guest_census=True)
    cfg = r.recover(0x1000)
    program = build(cfg, r.arena)
    assert program.blocks
    assert program.verify() == []
    assert not program.validation['roundtrip_errors']
    assert r.guest_ctx.native_registers
    assert r.stack_base <= 0xffffffff


def test_unbacked_native_code_is_not_decoded_as_zero_fill():
    a, m, l = setup(bytes.fromhex('00 00'))
    l.img._file_offset = lambda addr: None
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'unreadable'
    assert l.steps == 0


def test_win32_unknown_call_does_not_assume_win64_arguments_or_cleanup():
    a, m, l = setup(bytes.fromhex('e8 fb 1f 00 00'))
    l.summary_enabled = False
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'native_call_abi'
    assert l.events[-1]['abi'] == 'win32'
    assert l.events[-1]['stack_cleanup'] is None


def test_win32_stdcall_decoration_establishes_cleanup_and_stack_arguments():
    a, m, l = setup(bytes.fromhex('ff 15 00 80 00 00'))
    l.img.imports[0x8000] = ('test.dll', '_Example@8')
    m.store(a.const(0x10000, 32), a.const(17, 32), 4, 0)
    m.store(a.const(0x10004, 32), a.const(29, 32), 4, 0)
    assert l.step(0x1000) == ('next', 0x1006)
    assert m.concrete_rsp() == 0x10008
    assert [a.as_const(v) for _, v in l.events[-1]['args']] == [17, 29]
    assert a.width(l.events[-1]['ret']) == 32


def test_win32_documented_stdcall_imports_establish_stack_cleanup():
    cases = (
        ('HeapAlloc', 12, [0x1234, 0, 0x20]),
        ('MultiByteToWideChar', 24, [65001, 0, 0x8000, -1, 0x9000, 16]),
    )
    for symbol, cleanup, values in cases:
        a, m, l = setup(bytes.fromhex('ff 15 00 80 00 00'))
        l.img.imports[0x8000] = ('KERNEL32.dll', symbol)
        for index, value in enumerate(values):
            m.store(a.const(0x10000 + 4 * index, 32), a.const(value & 0xffffffff, 32), 4, 0)
        assert l.step(0x1000) == ('next', 0x1006)
        event = l.events[-1]
        assert event['stack_cleanup'] == cleanup
        assert event['cleanup_mode'] == 'callee'
        assert event['abi_contract']['convention'] == 'stdcall'
        assert event['abi_contract']['stack_bytes'] == cleanup
        assert [a.as_const(value) for _, value in event['args']] == [
            value & 0xffffffff for value in values]
        assert m.concrete_rsp() == 0x10000 + cleanup


def test_win32_caller_cleanup_is_inferred_but_not_applied_twice():
    a, m, l = setup(bytes.fromhex('e8 fb 1f 00 00 83 c4 08'))
    l.summary_enabled = False
    m.store(a.const(0x10000, 32), a.const(17, 32), 4, 0)
    m.store(a.const(0x10004, 32), a.const(29, 32), 4, 0)
    assert l.step(0x1000) == ('next', 0x1005)
    event = l.events[-1]
    assert event['stack_cleanup'] == 8
    assert event['cleanup_mode'] == 'caller'
    assert event['cleanup_evidence'][0]['instruction'] == 'add'
    assert m.concrete_rsp() == 0x10000
    assert [a.as_const(v) for _, v in event['args']] == [17, 29]
    assert l.step(0x1005) == ('next', 0x1008)
    assert m.concrete_rsp() == 0x10008


def test_win32_caller_cleanup_rejects_a_stack_frame_restore():
    # `mov esp, ebp; add esp, 8` is a frame restore followed by an unrelated
    # adjustment, not evidence that the call used caller cleanup.
    a, m, l = setup(bytes.fromhex('e8 fb 1f 00 00 8b e5 83 c4 08'))
    l.summary_enabled = False

    kind, stop = l.step(0x1000)

    assert kind == 'stop' and stop.kind == 'native_call_abi'
    assert m.concrete_rsp() == 0x10000


def test_win32_direct_native_cfg_proves_one_return_cleanup():
    # entry is outside the synthetic VM. Both branches of the native
    # callee reach the same ret 8.
    a, m, l = setup(bytes.fromhex(
        'e8 0b 00 00 00'
        '90 90 90 90 90 90 90 90 90 90 90'
        '75 09 c2 08 00 90 90 90 90 90 90'
        'c2 08 00'))
    l.vm_hi = 0x1002
    l.summary_enabled = False
    m.store(a.const(0x10000, 32), a.const(1, 32), 4, 0)
    m.store(a.const(0x10004, 32), a.const(2, 32), 4, 0)
    assert l.step(0x1000) == ('next', 0x1005)
    event = l.events[-1]
    assert event['stack_cleanup'] == 8
    assert event['cleanup_mode'] == 'callee'
    assert event['argument_contract'] == 'bounded direct native return contract'
    assert len(event['native_return_contract']['return_sites']) == 2
    assert [a.as_const(value) for _, value in event['args']] == [1, 2]
    assert m.concrete_rsp() == 0x10008


def test_win32_direct_native_cfg_accepts_ret_prefixes():
    # Capstone names the semantically equivalent F3 C3 and F2 C3 encodings
    # `repz ret` and `bnd ret`; both are ordinary near returns for this proof.
    for prefixed_ret in ('f3 c3', 'f2 c3'):
        a, m, l = setup(bytes.fromhex(
            'e8 0b 00 00 00'
            '90 90 90 90 90 90 90 90 90 90 90'
            + prefixed_ret + ' 90 90'))
        l.vm_hi = 0x1002
        l.summary_enabled = False

        assert l.step(0x1000) == ('next', 0x1005)
        event = l.events[-1]
        assert event['stack_cleanup'] == 0
        assert event['cleanup_mode'] == 'callee'
        assert event['argument_contract'] == (
            'bounded direct native return contract')
        assert event['native_return_contract']['return_sites'] == [{
            'site': 0x1010,
        }]


def test_win32_direct_native_cfg_proves_nested_call_cleanup():
    # Main returns with `ret 8`; its direct leaf call contributes `ret 0`.
    # The caller then removes the pushed argument with a literal stack adjust.
    a, m, l = setup(bytes.fromhex(
        'e8 0b 00 00 00'
        '90 90 90 90 90 90 90 90 90 90 90'
        '55 8b ec 6a 00 e8 08 00 00 00'
        '83 c4 04 5d c2 08 00 90 c3 cc cc'))
    l.vm_hi = 0x1002
    l.summary_enabled = False

    assert l.step(0x1000) == ('next', 0x1005)
    event = l.events[-1]
    assert event['stack_cleanup'] == 8
    assert event['cleanup_mode'] == 'callee'
    assert event['argument_contract'] == 'bounded direct native return contract'
    assert m.concrete_rsp() == 0x10008


def test_win32_direct_native_cfg_accounts_for_documented_import_cleanup():
    # The outer native helper calls the no-argument GetLastError import before
    # returning. The documented import convention is part of the static stack
    # proof; an unknown indirect callee remains a refusal.
    a, m, l = setup(bytes.fromhex(
        'e8 0b 00 00 00'
        '90 90 90 90 90 90 90 90 90 90 90'
        'ff 15 00 80 00 00 c3'))
    l.img.imports[0x8000] = ('KERNEL32.dll', 'GetLastError')
    l.vm_hi = 0x1002
    l.summary_enabled = False

    assert l.step(0x1000) == ('next', 0x1005)
    event = l.events[-1]
    assert event['stack_cleanup'] == 0
    assert event['cleanup_mode'] == 'callee'
    assert event['argument_contract'] == 'bounded direct native return contract'


def test_win32_direct_native_cfg_rejects_unknown_import_cleanup():
    a, m, l = setup(bytes.fromhex(
        'e8 0b 00 00 00'
        '90 90 90 90 90 90 90 90 90 90 90'
        'ff 15 00 80 00 00 c3'))
    l.img.imports[0x8000] = ('example.dll', 'UnknownExport')
    l.vm_hi = 0x1002
    l.summary_enabled = False

    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'native_call_abi'


def test_win32_direct_native_cfg_accounts_for_observed_import_contracts():
    contracts = (
        ('IsDebuggerPresent', 0),
        ('SetUnhandledExceptionFilter', 4),
        ('UnhandledExceptionFilter', 4),
        ('GetCurrentProcess', 0),
        ('TerminateProcess', 8),
    )
    for symbol, cleanup in contracts:
        a, m, l = setup(bytes.fromhex(
            'e8 0b 00 00 00'
            '90 90 90 90 90 90 90 90 90 90 90')
            + b'\x6a\x00' * (cleanup // 4)
            + bytes.fromhex('ff 15 00 80 00 00 c3'))
        l.img.imports[0x8000] = ('KERNEL32.dll', symbol)
        l.vm_hi = 0x1002
        l.summary_enabled = False

        assert l.step(0x1000) == ('next', 0x1005)
        event = l.events[-1]
        assert event['stack_cleanup'] == 0
        assert event['cleanup_mode'] == 'callee'
        assert event['native_return_contract'] == {
            'stack_cleanup': 0,
            'instructions_visited': cleanup // 4 + 2,
            'return_sites': [{'site': 0x1010 + (cleanup // 4) * 2 + 6}],
            'method': (
                'bounded direct native CFG; every path preserves the return '
                'slot, balances its stack, and terminates with one ret '
                'immediate'),
        }


def test_win32_direct_native_cfg_tracks_frame_pointer_epilogue():
    # Standard MSVC frame:
    #   push ebp; mov ebp,esp; ...; mov esp,ebp; pop ebp; ret
    # The frame is balanced and never overwrites the caller's return slot.
    a, m, l = setup(bytes.fromhex(
        'e8 0b 00 00 00'
        '90 90 90 90 90 90 90 90 90 90 90'
        '55 8b ec 6a ff 68 11 11 11 11'
        '64 a1 00 00 00 00 50 8b e5 5d c3'))
    l.vm_hi = 0x1002
    l.summary_enabled = False
    assert l.step(0x1000) == ('next', 0x1005)
    event = l.events[-1]
    assert event['stack_cleanup'] == 0
    assert event['cleanup_mode'] == 'callee'
    assert event['native_return_contract']['return_sites'] == [{'site': 0x1024}]
    assert m.concrete_rsp() == 0x10000


def test_win32_direct_native_cfg_tracks_leave_epilogue():
    # Standard MSVC frame:
    #   push ebp; mov ebp,esp; ...; leave; ret
    # LEAVE restores ESP from EBP and pops the saved frame pointer.
    a, m, l = setup(bytes.fromhex(
        'e8 0b 00 00 00'
        '90 90 90 90 90 90 90 90 90 90 90'
        '55 8b ec 6a ff c9 c3'))
    l.vm_hi = 0x1002
    l.summary_enabled = False

    assert l.step(0x1000) == ('next', 0x1005)
    event = l.events[-1]
    assert event['stack_cleanup'] == 0
    assert event['cleanup_mode'] == 'callee'
    assert event['native_return_contract']['return_sites'] == [{
        'site': 0x1016,
    }]
    assert m.concrete_rsp() == 0x10000


def test_win32_direct_native_cfg_rejects_stack_pointer_exchange():
    # `xchg esi, esp` changes the stack pointer without a literal adjustment.
    a, m, l = setup(bytes.fromhex(
        'e8 01 00 00 00 90'
        '87 e6 c3'))
    l.vm_hi = 0x1002
    l.summary_enabled = False
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'native_call_abi'


def test_win32_direct_native_cfg_rejects_frame_pointer_exchange():
    # After `mov ebp,esp`, exchanging EBP through the second operand invalidates
    # the later frame restore even though the instruction's first operand is ESI.
    a, m, l = setup(bytes.fromhex(
        'e8 01 00 00 00 90'
        '55 8b ec 87 ee 8b e5 5d c3'))
    l.vm_hi = 0x1002
    l.summary_enabled = False
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'native_call_abi'


def test_win32_direct_native_cfg_rejects_conflicting_returns():
    # jne selects ret 0 or ret 8. The condition and target cleanup are
    # correlated, but no value proves the correlation, so no contract.
    a, m, l = setup(bytes.fromhex(
        'e8 0b 00 00 00'
        '90 90 90 90 90 90 90 90 90 90 90'
        '75 09 c3 90 90 90 90 90 90 90 90'
        'c2 08 00'))
    l.vm_hi = 0x1002
    l.summary_enabled = False
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'native_call_abi'
    assert stop.info['event']['stack_cleanup'] is None


def test_pe32_cfg_identity_tracks_vm_pointer_registers():
    from core.explore import Recoverer
    a, m, l = setup(bytes.fromhex('90'))
    r = Recoverer(l.img, 0x1000, 0x2000, step_budget=20)
    m.set_reg('esi', a.const(0x1100, 32), 32)
    first = r.identity(l)
    m.set_reg('esi', a.const(0x1200, 32), 32)
    assert r.identity(l) != first


def test_win32_full_memory_havoc_does_not_enumerate_four_gigabytes(monkeypatch):
    import core.gueststate as gueststate
    from tests.test_joint_memory import JointMemoryTests
    from core.xstate import REGS
    helper = JointMemoryTests()
    p = helper.program(stores=[(0, None, 1 << 32)])
    normal_range = range

    def bounded_range(*args):
        result = normal_range(*args)
        assert len(result) <= 1024, 'havoc enumerated address space instead of tracked memory'
        return result

    monkeypatch.setattr(gueststate, 'range', bounded_range, raising=False)
    out = p.blocks['A']['memory_out']
    values = p._evaluate('A', dict.fromkeys(REGS, 0), {0x1234: 7, 1 << 40: 9}, [out])
    assert values[out] == {1 << 40: 9}


def test_fastcall_total_bytes_does_not_prove_parameter_types():
    a, m, l = setup(bytes.fromhex('ff 15 00 80 00 00'))
    l.img.imports[0x8000] = ('test.dll', '@Example@16')
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'native_call_abi'


def test_pe32_bugcheck_has_typed_terminal_with_stack_arguments():
    a, m, l = setup(bytes.fromhex('ff 15 00 80 00 00'))
    l.img.imports[0x8000] = ('ntoskrnl.exe', 'KeBugCheck')
    m.store(a.const(0x10000, 32), a.const(0xdead, 32), 4, 0)
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'noreturn'
    assert a.as_const(stop.info['event']['args'][0][1]) == 0xdead


def test_pe32_unknown_tail_import_keeps_abi_refusal():
    a, m, l = setup(bytes.fromhex('ff 25 00 80 00 00'))
    l.img.imports[0x8000] = ('test.dll', 'Unknown')
    kind, stop = l.step(0x1000)
    assert kind == 'stop' and stop.kind == 'native_call_abi'
