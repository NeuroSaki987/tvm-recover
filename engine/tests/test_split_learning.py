"""Successful immutable address proofs guide later queries without guessing data."""
import unittest
from unittest.mock import patch
from core.expr import Arena
from core.memory import finite_addresses


class SplitLearningTests(unittest.TestCase):
    def test_reuses_a_successful_guard_before_enumerating_candidates(self):
        a = Arena()
        flag = a.bin('and', a.init_reg('rcx'), a.const(1, 64), 64)
        first = a.select(flag, a.const(0x8000, 64), a.const(0x9000, 64), 64)
        second = a.select(flag, a.const(0xA000, 64), a.const(0xB000, 64), 64)
        policy = dict(max_nodes=None, max_rebuild_work=None)
        proof = finite_addresses(a, first, **policy)
        self.assertIsNotNone(proof)
        with patch('core.expr.boolean_subexpressions', side_effect=AssertionError('repeat enumeration')):
            learned = finite_addresses(a, second, **policy)
        self.assertEqual({0xA000, 0xB000}, set(learned[1:]))
        self.assertEqual(proof[0], learned[0])

    def test_normalization_proves_constant_address_without_any_boolean(self):
        a = Arena(); x = a.init_reg('rcx')
        zero = a._add(('bin', 'xor', x, x, 64))
        addr = a._add(('bin', 'add', zero, a.const(0x8000, 64), 64))
        proof = finite_addresses(a, addr, max_nodes=None, max_rebuild_work=None)
        self.assertIsNotNone(proof)
        self.assertEqual((0x8000, 0x8000), proof[1:])

    def test_failed_hint_never_reuses_its_previous_addresses(self):
        a = Arena(); flag = a.bin('and', a.init_reg('rcx'), a.const(1, 64), 64)
        first = a.select(flag, a.const(0x8000, 64), a.const(0x9000, 64), 64)
        finite_addresses(a, first, max_nodes=None, max_rebuild_work=None)
        unknown = a.bin('add', a.init_reg('rdx'), a.init_reg('rdx'), 64)
        addr = a.select(flag, unknown, a.const(0xA000, 64), 64)
        self.assertIsNone(finite_addresses(a, addr, max_nodes=None, max_rebuild_work=None))

    def test_bounded_queries_keep_their_existing_candidate_budget(self):
        a = Arena(); flag = a.bin('and', a.init_reg('rcx'), a.const(1, 64), 64)
        addr = a.select(flag, a.const(0x8000, 64), a.const(0x9000, 64), 64)
        with patch.object(a, 'substitute', side_effect=AssertionError('zero budget bypass')):
            self.assertIsNone(finite_addresses(a, addr, max_rebuild_work=0))
