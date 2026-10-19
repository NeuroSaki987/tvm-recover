"""A proved terminal boundary must stay distinct from a normal function return."""
import unittest
from types import SimpleNamespace
from core.explore import Recoverer, Block, BlockKey, Cfg
from core.lift64 import Stop
from core.gueststate import build
from core.xstate import REGS
from tests.test_state_repair import lifter


class TerminalFlowTests(unittest.TestCase):
    def test_noreturn_and_fastfail_end_paths_without_inventing_return(self):
        for kind in ('noreturn', 'fastfail'):
            a, _m, l = lifter(); key = BlockKey(0x1000, 0)
            block = Block(key); cfg = Cfg(entry=key)
            stop = Stop(kind, 0x1100, info={'code': a.const(3, 32)})
            r = Recoverer(SimpleNamespace(), 0x1000, 0x2000)
            self.assertEqual([], r._terminate(l, stop, key, block, {}, cfg))
            self.assertEqual(kind, block.kind)
            self.assertEqual(0, cfg.unresolved)
            self.assertEqual(0x1100, block.terminal['site'])

    def test_terminal_code_and_call_arguments_are_serialized_dependencies(self):
        a, _m, _l = lifter(); key = BlockKey(0x1000, 0)
        values = {r: a.init_reg(r) for r in REGS}
        b = Block(key, kind='fastfail', guest_entry=values, guest_exit=values)
        b.terminal = {'kind': 'fastfail', 'site': 0x1100, 'code': a.trunc(a.init_reg('rcx'), 32)}
        p = build(Cfg(entry=key, blocks={key: b}), a)
        terminal = p.blocks[str(key)]['terminal']
        self.assertEqual('fastfail', terminal['kind'])
        self.assertEqual(32, p.nodes[terminal['code']]['width'])
        self.assertEqual([], p.verify())
        p.blocks[str(key)]['terminal']['code'] = len(p.nodes) + 100
        self.assertTrue(p.verify())

    def test_terminal_descriptor_cannot_be_removed_or_widened(self):
        a, _m, _l = lifter(); key = BlockKey(0x1000, 0)
        values = {r: a.init_reg(r) for r in REGS}
        b = Block(key, kind='fastfail', guest_entry=values, guest_exit=values)
        b.terminal = {'kind': 'fastfail', 'site': 0x1100, 'code': a.const(3, 32)}
        p = build(Cfg(entry=key, blocks={key: b}), a)
        del p.blocks[str(key)]['terminal']
        self.assertTrue(p.verify())

    def test_terminal_only_history_is_not_hidden_by_closure_or_names(self):
        from core.stateclosure import analyze_memory
        from core.symbols import catalog
        a, _m, _l = lifter(); key = BlockKey(0x1000, 0)
        values = {r: a.init_reg(r) for r in REGS}
        b = Block(key, kind='fastfail', guest_entry=values, guest_exit=values)
        b.terminal = {'kind': 'fastfail', 'site': 0x1100,
                      'code': a.load(a.const(0x8000, 64), 32, 7)}
        p = build(Cfg(entry=key, blocks={key: b}), a)
        self.assertFalse(analyze_memory(p, str(key))['memory_closed'])
        self.assertTrue(any(row['kind'] == 'history' for row in catalog(p, key)))
