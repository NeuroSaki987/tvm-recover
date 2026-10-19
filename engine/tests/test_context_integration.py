"""Caller state must initialize every recovery phase, never just its probe."""
import unittest
from types import SimpleNamespace
from core.explore import Recoverer
from core.expr import Arena
from core.mach import Machine
from tests.test_state_repair import stub_image


class ContextIntegrationTests(unittest.TestCase):
    def test_context_is_replayed_for_both_sampling_and_execution(self):
        class Context:
            target_root = 0x1000
            calls = 0
            def replay(self, _img, _lo, _hi, stack):
                self.calls += 1
                a = Arena(); m = Machine(a, stack)
                m.set_reg('r11', a.const(0x123456, 64), 64)
                return a, m, {'scope': 'caller-specialized', 'target_root': self.target_root}
        context = Context()
        img = stub_image(image_bytes=lambda _va, _n: b'\xc3')
        r = Recoverer(img, 0x1000, 0x2000, entry_context=context)
        a, m, _l = r.prepare(0x1000)
        self.assertEqual(2, context.calls)
        self.assertEqual(0x123456, a.as_const(m.reg('r11', 64)))
        self.assertEqual('caller-specialized', r.entry_contract['scope'])

    def test_context_for_another_root_is_rejected(self):
        context = SimpleNamespace(target_root=0x1100)
        r = Recoverer(stub_image(), 0x1000, 0x2000, entry_context=context)
        with self.assertRaises(ValueError):
            r.prepare(0x1000)
