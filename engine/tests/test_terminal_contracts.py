"""Non-returning Windows contracts must never acquire a synthetic continuation."""

import unittest
from unittest.mock import patch

from tests.test_state_repair import lifter


class TerminalContractTests(unittest.TestCase):
    def test_bugcheckex_preserves_the_fifth_stack_argument(self):
        a, m, l = lifter()
        l.imports[0x3000] = ('ntoskrnl.exe', 'KeBugCheckEx')
        address = a.bin('add', m.reg('rsp', 64), a.const(0x20, 64), 64)
        value = a.const(0x12345678, 64)
        m.store(address, value, 8, 0)
        _kind, stop = l._guest_call(None, 0x1100, 0x1106, 0x3000)
        self.assertEqual({'4': value}, stop.info['event']['stack_args'])

    def test_documented_bugcheck_calls_record_arguments_and_stop(self):
        for module, symbol in [('ntoskrnl.exe', 'KeBugCheck'),
                               ('NTOSKRNL.EXE', 'KeBugCheckEx')]:
            with self.subTest(module=module, symbol=symbol):
                a, m, l = lifter()
                l.imports[0x3000] = (module, symbol)
                code = a.const(0x139, 64)
                m.set_reg('rcx', code, 64)
                before_rax = m.reg('rax', 64)
                before_version = m.mem_version
                with patch.object(m, 'forget_call_writes',
                                  side_effect=AssertionError('unreachable post-call effects')):
                    kind, stop = l._guest_call(None, 0x1100, 0x1106, 0x3000)
                self.assertEqual(kind, 'stop')
                self.assertEqual(stop.kind, 'noreturn')
                self.assertEqual(stop.site, 0x1100)
                event = l.events[-1]
                self.assertTrue(event['noreturn'])
                self.assertEqual(event['import_slot'], 0x3000)
                self.assertEqual(dict(event['args'])['rcx'], code)
                self.assertEqual(stop.info['event'], event)
                self.assertEqual(l.guest_trace[-1][1], 'call')
                self.assertTrue(l.guest_trace[-1][2]['noreturn'])
                self.assertEqual(m.reg('rax', 64), before_rax)
                self.assertEqual(m.mem_version, before_version)

    def test_identically_named_unknown_or_other_module_calls_still_return(self):
        for module, symbol in [('other.sys', 'KeBugCheck'),
                               ('ntoskrnl.exe', 'KeBugCheckCallback'),
                               ('ntoskrnl.exe', 'kebugcheck'),
                               ('ntoskrnl.exe', 'KeDelayExecutionThread')]:
            with self.subTest(module=module, symbol=symbol):
                _a, _m, l = lifter()
                l.imports[0x3000] = (module, symbol)
                self.assertEqual(l._guest_call(None, 0x1100, 0x1106, 0x3000),
                                 ('next', 0x1106))
                self.assertFalse(l.events[-1].get('noreturn', False))

    def test_unknown_computed_call_has_no_terminal_contract(self):
        a, _m, l = lifter()
        self.assertEqual(l._guest_call(None, 0x1100, 0x1106, None,
                                      target_expr=a.opaque('callee', 64)),
                         ('next', 0x1106))

    def instruction(self, raw):
        a, m, l = lifter()
        insn = next(l._md.disasm(bytes.fromhex(raw), 0x1100))
        l.decode = lambda _ip: (insn.size, insn, insn.bytes)
        return a, m, l

    def test_int29_is_fastfail_with_unsigned_32bit_code(self):
        a, m, l = self.instruction('cd29')
        m.set_reg('rcx', a.const(0x1234567800000003, 64), 64)
        kind, stop = l.step(0x1100)
        self.assertEqual(kind, 'stop')
        self.assertEqual(stop.kind, 'fastfail')
        self.assertEqual(stop.site, 0x1100)
        self.assertEqual(a.width(stop.info['code']), 32)
        self.assertEqual(a.as_const(stop.info['code']), 3)

    def test_fastfail_preserves_symbolic_code(self):
        a, m, l = self.instruction('cd29')
        m.set_reg('rcx', a.init_reg('rcx'), 64)
        _kind, stop = l.step(0x1100)
        self.assertEqual(stop.kind, 'fastfail')
        self.assertEqual(a.width(stop.info['code']), 32)
        self.assertIsNone(a.as_const(stop.info['code']))

    def test_other_interrupts_and_breakpoints_remain_unresolved(self):
        for raw in ('cd2a', 'cd03', 'cc'):
            with self.subTest(raw=raw):
                _a, _m, l = self.instruction(raw)
                l.guest_exit_trap = False
                kind, stop = l.step(0x1100)
                self.assertEqual(kind, 'stop')
                self.assertEqual(stop.kind, 'unsupported')


if __name__ == '__main__':
    unittest.main(verbosity=2)
