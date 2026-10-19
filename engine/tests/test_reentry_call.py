"""Unknown callees require a complete guest-call boundary and explicit ABI effects."""
import unittest

from core.expr import Arena
from core.ir import call_target
from core.lift64 import Lifter
from core.mach import Machine
from tests.test_int3_exit import _Image, BASE, LEA_RSP, MOV_SLOT_RBP, MOV_RBP_RSP

RESTORE = bytes.fromhex('488b2424')
CALL = bytes.fromhex('ff5008')
PROLOGUE = LEA_RSP + MOV_SLOT_RBP + MOV_RBP_RSP


def setup(prefix=RESTORE, call=CALL, prologue=PROLOGUE):
    a = Arena(); m = Machine(a, 0x10000)
    m.store(a.const(0x10000, 64), a.const(0x20000, 64), 8, 0)
    l = Lifter(_Image(prefix + call + prologue), m, a, BASE, BASE + 0x100)
    if prefix == RESTORE:
        kind, target = l.step(BASE)
        assert kind == 'next' and target == BASE + 4
    return a, m, l


class ReentryCallTests(unittest.TestCase):
    def test_computed_call_returns_with_abi_clobbers_and_symbolic_target_provenance(self):
        a, m, l = setup()
        nonvolatile = {r: m.reg(r, 64) for r in ('rbx', 'rbp', 'rsi', 'rdi', 'r12', 'r13', 'r14', 'r15')}
        volatile = {r: m.reg(r, 64) for r in ('rax', 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11')}
        kind, target = l.step(BASE + 4)
        self.assertEqual(('next', BASE + 7), (kind, target))
        self.assertEqual(0x20000, m.concrete_rsp())
        self.assertEqual(nonvolatile, {r: m.reg(r, 64) for r in nonvolatile})
        self.assertTrue(all(volatile[r] != m.reg(r, 64) for r in volatile))
        event = next(e for e in l.events if e['kind'] == 'call')
        self.assertIsNone(event['target'])
        self.assertEqual('load', a.node(event['target_expr'])[0])
        self.assertEqual(f'indirect@{BASE + 4:#x}', call_target(l, event))
        self.assertIn('continuation_contract', event)

    def test_missing_or_malformed_boundary_stays_unresolved(self):
        negatives = [
            (bytes.fromhex('90909090'), CALL, PROLOGUE),
            (RESTORE, CALL, bytes.fromhex('90') + PROLOGUE),
            (RESTORE, CALL, bytes.fromhex('488da424700a0000') + MOV_SLOT_RBP + MOV_RBP_RSP),
            (RESTORE, CALL, LEA_RSP + MOV_SLOT_RBP),
            (RESTORE, bytes.fromhex('ff5044'), PROLOGUE),
            (RESTORE, bytes.fromhex('ff544808'), PROLOGUE),
            (RESTORE, bytes.fromhex('64ff5008'), PROLOGUE),
        ]
        for prefix, call, prologue in negatives:
            with self.subTest(prefix=prefix.hex(), call=call.hex(), prologue=prologue.hex()):
                _a, _m, l = setup(prefix, call, prologue)
                kind, stop = l.step(BASE + len(prefix))
                self.assertEqual('stop', kind)
                self.assertEqual('dispatch_call', stop.kind)
                self.assertFalse(any(e['kind'] == 'call' for e in l.events))

    def test_rule_can_be_disabled(self):
        _a, _m, l = setup()
        l.computed_reentry_calls = False
        kind, stop = l.step(BASE + 4)
        self.assertEqual('stop', kind)
        self.assertEqual('dispatch_call', stop.kind)

    def test_direct_entry_and_intervening_instruction_do_not_supply_restore(self):
        for execute_other in (False, True):
            a = Arena(); m = Machine(a, 0x10000)
            l = Lifter(_Image(RESTORE + CALL + PROLOGUE), m, a, BASE, BASE + 0x100)
            if execute_other:
                l.step(BASE)
                l.step(BASE + 7)  # jump away and back without a new restore
            kind, stop = l.step(BASE + 4)
            self.assertEqual('stop', kind)
            self.assertEqual('dispatch_call', stop.kind)

    def test_checkpoint_restores_predecessor_evidence(self):
        from core.state import LifterCheckpoint
        _a, _m, l = setup()
        checkpoint = LifterCheckpoint(l)
        l.step(BASE + 7)
        checkpoint.restore(l)
        self.assertEqual(('next', BASE + 7), l.step(BASE + 4))

    def test_volatile_simd_clobbered_and_nonvolatile_simd_preserved(self):
        _a, m, l = setup()
        before = {name: l.xmm_halves(name) for name in ('xmm0', 'xmm5', 'xmm6', 'xmm15')}
        l.step(BASE + 4)
        for name in ('xmm0', 'xmm5'):
            self.assertTrue(all(x != y for x, y in zip(before[name], l.xmm_halves(name))))
        for name in ('xmm6', 'xmm15'):
            self.assertEqual(before[name], l.xmm_halves(name))

    def test_machine_snapshot_rewinds_or_clears_predecessor_evidence(self):
        from core.state import MachineSnapshot
        for with_lifter in (False, True):
            a = Arena(); m = Machine(a, 0x10000)
            l = Lifter(_Image(RESTORE + CALL + PROLOGUE), m, a, BASE, BASE + 0x100)
            snap = MachineSnapshot(m, lifter=l if with_lifter else None)
            l.step(BASE)
            snap.restore(m, l)
            kind, stop = l.step(BASE + 4)
            self.assertEqual('stop', kind)
            self.assertEqual('dispatch_call', stop.kind)
