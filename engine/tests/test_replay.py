"""A guided re-lift must consume resolved transfers as well as forced ones."""
import unittest

from core.lift64 import Stop
from tests.test_state_repair import lifter


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.a, self.m, self.l = lifter()

    def run_replay(self, guide, step, limit=100):
        from core.replay import replay
        self.l.step = step
        return replay(self.l, guide[0].site, guide, limit=limit)

    def test_resolved_transfers_advance_the_guide(self):
        from core.replay import TraceStep
        dest = self.a.opaque('dispatch', 64)
        guide = [TraceStep(0x1000, 0x1100, ((0x1000, 0x1100),)),
                 TraceStep(0x1100, 0x1200, ((0x1100, 0x1200),))]
        def step(ip):
            if ip == 0x1000:
                self.l.dispatches.append((ip, 0x1100))
                return 'next', 0x1100
            return 'stop', Stop('dispatch', ip, dest=dest)
        result = self.run_replay(guide, step)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(2, result.transfers)
        self.assertEqual([(0x1000, 0x1100), (0x1100, 0x1200)], self.l.dispatches)
        self.assertEqual(0x1200, self.a.as_const(dict(result.guards)[dest]))

    def test_native_branch_is_guided_with_an_explicit_predicate(self):
        from core.replay import TraceStep
        pred = self.a.bin('eq', self.a.init_reg('rcx'), self.a.const(0, 64), 64)
        guide = [TraceStep(0x1000, 0x1020, ())]
        result = self.run_replay(guide, lambda ip: ('stop', Stop('native_branch', ip, pred=pred,
                                                              taken=0x1020, not_taken=0x1010)))
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(1, result.forced_branches)
        self.assertEqual(1, self.a.as_const(dict(result.guards)[pred]))

    def test_wrong_folded_target_is_rejected_and_attempt_is_rolled_back(self):
        from core.replay import TraceStep
        before = dict(self.m.regs)
        def step(ip):
            self.m.set_reg('rax', self.a.const(99, 64), 64)
            self.l.dispatches.append((ip, 0x1300))
            self.l.guest_trace.append((1, 'home', ('rax', 'rax', self.a.const(99, 64))))
            return 'next', 0x1300
        result = self.run_replay([TraceStep(0x1000, 0x1100, ((0x1000, 0x1100),))], step)
        self.assertFalse(result.ok)
        self.assertEqual('transfer_mismatch', result.reason)
        self.assertEqual(before, self.m.regs)
        self.assertEqual([], self.l.dispatches)
        self.assertEqual([], self.l.guest_trace)

    def test_constant_dispatch_cannot_be_forced_to_a_different_target(self):
        from core.replay import TraceStep
        guide = [TraceStep(0x1000, 0x1100, ((0x1000, 0x1100),))]
        result = self.run_replay(guide, lambda ip: ('stop', Stop('dispatch', ip,
                                              dest=self.a.const(0x1200, 64))))
        self.assertFalse(result.ok)
        self.assertEqual('contradictory_guard', result.reason)

    def test_native_instruction_drift_is_rejected(self):
        from core.replay import TraceStep
        guide = [TraceStep(0x1000, 0x1001, ()), TraceStep(0x1001, 0x1100, ())]
        result = self.run_replay(guide, lambda ip: ('next', 0x1002))
        self.assertFalse(result.ok)
        self.assertEqual('next_ip_mismatch', result.reason)

    def test_budget_is_a_hard_bound(self):
        from core.replay import TraceStep
        guide = [TraceStep(0x1000, 0x1001, ()), TraceStep(0x1001, 0x1002, ())]
        result = self.run_replay(guide, lambda ip: ('next', ip + 1), limit=1)
        self.assertFalse(result.ok)
        self.assertEqual('step_budget', result.reason)
        self.assertEqual(1, result.steps)

    def test_guide_collection_counts_resolved_transfers_and_slices_instructions(self):
        from core.segment import collect
        def step(ip):
            if ip in (0x1000, 0x1101):
                target = 0x1100 if ip == 0x1000 else 0x1200
                self.l.dispatches.append((ip, target))
                return 'next', target
            return 'next', ip + 1
        self.l.step = step
        result = collect(self.l, 0x1000, [1, 2], limit=5)
        self.assertTrue(result['ok'])
        self.assertEqual({1: 0, 2: 2}, result['offsets'])
        self.assertEqual([0x1100, 0x1101], [s.site for s in result['records']])

    def test_guide_collection_cannot_walk_forever(self):
        from core.segment import collect
        self.l.step = lambda ip: ('next', ip)
        result = collect(self.l, 0x1000, [1, 2], limit=3)
        self.assertFalse(result['ok'])
        self.assertEqual('guide_step_budget', result['reason'])
        self.assertEqual(3, result['steps'])

    def test_exception_also_rolls_back_native_state(self):
        from core.replay import TraceStep
        before = dict(self.m.regs)
        def step(ip):
            self.m.set_reg('rax', self.a.const(99, 64), 64)
            raise RuntimeError('decode failure')
        with self.assertRaises(RuntimeError):
            self.run_replay([TraceStep(0x1000, 0x1001)], step)
        self.assertEqual(before, self.m.regs)

    def test_unresolved_call_is_never_treated_as_a_jump(self):
        from core.replay import replay, TraceStep
        from core.lift64 import Lifter
        from tests.test_state_repair import stub_image
        image = stub_image(image_bytes=lambda ip, size: bytes.fromhex('ff d1') if ip == 0x1000 else b'')
        l = Lifter(image, self.m, self.a, 0x1000, 0x2000)
        before = dict(self.m.regs)
        for transfers in ((), ((0x1000, 0x1100),)):
            result = replay(l, 0x1000, [TraceStep(0x1000, 0x1100, transfers)])
            self.assertFalse(result.ok)
            self.assertEqual('unsupported_indirect_call', result.reason)
            self.assertEqual(before, self.m.regs)

    def test_modular_counter_relation_is_derived_from_symbolic_expression(self):
        from core.segment import derive_update
        x = self.a.opaque('counter', 64)
        after = self.a.zext(self.a.bin('add', self.a.trunc(x, 32), self.a.const(1, 32), 32), 64)
        relation = derive_update(self.a, x, after, self.a.const(2, 64), self.a.const(3, 64))
        self.assertEqual(32, relation['bits'])
        self.assertEqual(1, relation['delta'])
        self.assertEqual('structural_identity', relation['proof'])

    def test_observed_delta_is_not_accepted_when_symbolic_function_differs(self):
        from core.segment import derive_update
        x, y = self.a.opaque('x', 64), self.a.opaque('other_register', 64)
        output = self.a.bin('add', y, self.a.const(1, 64), 64)
        self.assertIsNone(derive_update(self.a, x, output, self.a.const(2, 64), self.a.const(3, 64)))


if __name__ == '__main__':
    unittest.main()
