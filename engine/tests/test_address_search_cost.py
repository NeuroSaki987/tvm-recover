"""Avoid searching a two-point range when an independent full word survives."""
import unittest
from unittest.mock import patch

from core.expr import Arena
from core.memory import finite_addresses


class AddressSearchCostTests(unittest.TestCase):
    def setUp(self):
        self.a = Arena()

    def test_free_base_plus_complex_offset_never_substitutes(self):
        a = self.a
        base = a.init_reg('rcx')
        cond = a.bin('and', a.init_reg('rflags'), a.const(1, 64), 64)
        offset = a.select(cond, a.const(8, 64), a.const(16, 64), 64)
        address = a.bin('add', base, offset, 64)
        with patch.object(a, 'substitute', side_effect=AssertionError('unnecessary full-DAG rebuild')):
            self.assertIsNone(finite_addresses(a, address))

    def test_nested_reversible_word_operations_are_rejected_without_search(self):
        a = self.a
        cond = a.bin('and', a.init_reg('rflags'), a.const(1, 64), 64)
        value = a.select(cond, a.const(3, 64), a.const(9, 64), 64)
        for operator in ('add', 'sub', 'xor'):
            inner = a.bin(operator, a.init_reg('rcx'), value, 64)
            address = a.bin('xor', a.un('neg', inner, 64), a.const(0xDEADBEEF, 64), 64)
            with patch.object(a, 'substitute', side_effect=AssertionError('unnecessary rebuild')):
                self.assertIsNone(finite_addresses(a, address))

    def test_shared_base_can_cancel_and_must_still_split(self):
        a = self.a
        base = a.init_reg('rcx')
        cond = a.bin('and', a.init_reg('rflags'), a.const(1, 64), 64)
        offset = a.select(cond, a.const(0x8000, 64), a.const(0x9000, 64), 64)
        # Keep a valid unfurled shape so the cancellation is discovered by
        # substitution rather than by the initial algebraic builder.
        cancelled = a._add(('bin', 'sub', base, base, 64))
        address = a._add(('bin', 'add', cancelled, offset, 64))
        result = finite_addresses(a, address)
        self.assertIsNotNone(result)
        self.assertEqual({0x8000, 0x9000}, set(result[1:]))

    def test_masked_word_is_not_treated_as_unconstrained(self):
        a = self.a
        cond = a.bin('and', a.init_reg('rcx'), a.const(1, 64), 64)
        address = a.bin('add', a.const(0x8000, 64), cond, 64)
        result = finite_addresses(a, address)
        self.assertIsNotNone(result)
        self.assertEqual({0x8000, 0x8001}, set(result[1:]))

    def test_cache_is_arena_local(self):
        for _ in range(2):
            a = Arena()
            cond = a.bin('and', a.init_reg('rcx'), a.const(1, 64), 64)
            address = a.select(cond, a.const(0x8000, 64), a.const(0x9000, 64), 64)
            self.assertIsNotNone(finite_addresses(a, address))

    def test_node_limit_precedes_free_word_dependency_analysis(self):
        a = self.a
        value = a.init_reg('rcx')
        for i in range(150):
            value = a.bin('add', a.opaque(f'word{i}', 64), value, 64)
        with patch('core.memory._free_word', side_effect=AssertionError('node budget bypass')):
            self.assertIsNone(finite_addresses(a, value, max_nodes=1, max_rebuild_work=0))

    def test_free_word_dependencies_are_bounded_and_not_retained(self):
        a = self.a
        value = a.init_reg('rcx')
        for i in range(150):
            value = a.bin('add', a.opaque(f'word{i}', 64), value, 64)
        finite_addresses(a, value, max_rebuild_work=0)
        self.assertFalse(getattr(a, '_word_dependencies', {}))

    def test_optional_split_has_a_work_budget_and_reports_incomplete_search(self):
        a = self.a
        value = a.init_reg('rcx')
        # A byte data dependency and many boolean masks; this is not certified
        # by the free-word proof and must still have bounded search cost.
        value = a.trunc(value, 8)
        for i in range(24):
            cond = a.bin('and', a.init_reg('rdx'), a.const(1 << (i % 8), 64), 64)
            term = a.bin('eq', cond, a.const(0, 64), 64)
            value = a.bin('add', value, a.trunc(term, 8), 8)
        address = a.zext(value, 64)
        with patch.object(a, 'substitute', wraps=a.substitute) as calls:
            self.assertIsNone(finite_addresses(a, address, max_rebuild_work=1))
        self.assertEqual(0, calls.call_count)
        self.assertEqual(1, a.address_search_stats['work_budget'])

    def test_larger_budget_can_retry_a_previous_budget_miss(self):
        a = self.a
        cond = a.bin('and', a.init_reg('rcx'), a.const(1, 64), 64)
        address = a.select(cond, a.const(0x8000, 64), a.const(0x9000, 64), 64)
        self.assertIsNone(finite_addresses(a, address, max_rebuild_work=0))
        result = finite_addresses(a, address, max_rebuild_work=1000)
        self.assertIsNotNone(result)
        self.assertEqual({0x8000, 0x9000}, set(result[1:]))

    def test_failed_load_address_probe_does_not_read_a_partial_arm(self):
        from tests.test_state_repair import lifter
        a, _m, l = lifter()
        flag = a.bin('and', a.init_reg('rflags'), a.const(1, 64), 64)
        unknown = a.init_reg('rcx')
        twice = a.bin('add', unknown, unknown, 64)
        addr = a.select(flag, twice, a.const(0x8000, 64), 64)
        with patch.object(l, '_read_at', return_value=a.const(7, 64)) as read:
            self.assertIsNone(l._load_via_predicate_split(addr, 8, 123))
        self.assertEqual(0, read.call_count)

    def test_load_and_store_queries_share_only_address_proofs(self):
        from tests.test_state_repair import lifter
        a, m, l = lifter()
        flag = a.bin('and', a.init_reg('rflags'), a.const(1, 64), 64)
        addr = a.select(flag, a.const(0x8000, 64), a.const(0x9000, 64), 64)
        m.store(a.const(0x8000, 64), a.const(7, 64), 8, 1)
        m.store(a.const(0x9000, 64), a.const(9, 64), 8, 2)
        first = l._load_via_predicate_split(addr, 8, 3)
        m.store(a.const(0x8000, 64), a.const(42, 64), 8, 4)
        second = l._load_via_predicate_split(addr, 8, 5)
        condition = a.node(first)[1]
        self.assertEqual(7, a.as_const(a.substitute(first, {condition: a.const(1, 64)})))
        self.assertEqual(42, a.as_const(a.substitute(second, {condition: a.const(1, 64)})))
        self.assertGreater(a.address_search_stats['cache_hits'], 0)

    def test_large_provable_load_address_retains_precision(self):
        from tests.test_state_repair import lifter
        a, _m, l = lifter()
        flag = a.bin('and', a.init_reg('rflags'), a.const(1, 64), 64)
        addr = a.select(flag, a.const(0x8000, 64), a.const(0x9000, 64), 64)
        # A large valid DAG whose single boolean pin still resolves both arms.
        # Loading must preserve that proof even beyond the optional store cap.
        for _ in range(5000):
            addr = a._add(('bin', 'xor', addr, a.const(0, 64), 64))
        with patch.object(l, '_read_at', side_effect=lambda addr, _n, _s: a.const(addr, 64)):
            result = l._load_via_predicate_split(addr, 8, 1)
        self.assertIsNotNone(result)
        condition = a.node(result)[1]
        self.assertEqual(0x8000, a.as_const(a.substitute(result, {condition: a.const(1, 64)})))
        self.assertEqual(0x9000, a.as_const(a.substitute(result, {condition: a.const(0, 64)})))
