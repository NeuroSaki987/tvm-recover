import unittest
from unittest.mock import patch
from core.expr import Arena


class IncrementalNormalizeTests(unittest.TestCase):
    def test_new_parent_reuses_already_normalized_ancestors(self):
        a = Arena(); value = a.init_reg('rcx')
        for _ in range(10000):
            value = a._add(('un', 'not', value, 64))
        a.normalize(value)
        parent = a._add(('bin', 'add', value, a.const(7, 64), 64))
        with patch.object(a, '_rebuild', wraps=a._rebuild) as rebuild:
            a.normalize(parent)
        self.assertLess(rebuild.call_count, 5)

    def test_changed_load_hook_invalidates_normalization_cache(self):
        a = Arena(); addr = a.const(0x1000, 64)
        load = a._add(('load', addr, 64, 0))
        self.assertEqual(load, a.normalize(load))
        a.versioned_load_folder = lambda _addr, width, _version: a.const(7, width)
        self.assertEqual(7, a.as_const(a.normalize(load)))

    def test_memory_versions_remain_distinct(self):
        a = Arena(); addr = a.const(0x1000, 64)
        old = a._add(('load', addr, 64, 0)); new = a._add(('load', addr, 64, 1))
        a.versioned_load_folder = lambda _addr, width, version: a.const(9 if version == 0 else 42, width)
        self.assertEqual(9, a.as_const(a.normalize(old)))
        self.assertEqual(42, a.as_const(a.normalize(new)))
        self.assertEqual(9, a.as_const(a.normalize(old)))
