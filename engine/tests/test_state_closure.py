"""Guarded register closure and explicitly scoped local effect contracts."""
import json
import unittest
from importlib.util import find_spec
from types import SimpleNamespace

from core.expr import Arena
from core.gueststate import Program, build
from core.xstate import REGS
from tests.test_guest_state import block


class StateClosureTests(unittest.TestCase):
    def setUp(self):
        self.a = Arena()
        self.ins = {r: self.a.init_reg(r) for r in REGS}

    def program(self, out=None, key='A', guards=(), effects=None):
        b = block(key, self.ins, out or self.ins)
        b.path_constraints = guards
        b.local_effects = effects or {}
        return build(SimpleNamespace(entry=key, blocks={key: b}), self.a)

    def test_greatest_fixed_point_excludes_transitive_carry(self):
        out = dict(self.ins, rax=self.a.opaque('incoming', 64),
                   rbx=self.ins['rax'], rcx=self.ins['rbx'],
                   r10=self.a.bin('add', self.ins['r10'], self.a.const(1, 64), 64))
        p = self.program(out)
        closure = p.closure('A')
        self.assertEqual(set(REGS) - {'rax', 'rbx', 'rcx'}, set(closure['registers']))
        self.assertEqual(['rbx', 'rcx'], closure['transitive_exclusions'])
        self.assertTrue(closure['conditional'])

    def test_guard_dependency_can_prevent_repeatable_subsystem(self):
        p = self.program(dict(self.ins, rax=self.a.opaque('incoming', 64)),
                         guards=[(self.ins['rax'], self.a.const(1, 64))])
        self.assertEqual([], p.closure('A')['registers'])
        self.assertEqual(['rax'], p.closure('A')['unavailable_guard_registers'])

    def test_guards_are_checked_before_unrequested_outputs(self):
        p = self.program(dict(self.ins, rax=self.a.opaque('unavailable', 64)),
                         guards=[(self.ins['r10'], self.a.const(7, 64))])
        self.assertEqual({'rbx': 9}, p.run_projection('A', {'rbx': 9, 'r10': 7}, {}, ['rbx']))
        with self.assertRaisesRegex(ValueError, 'guard'):
            p.run_projection('A', {'rbx': 9, 'r10': 8}, {}, ['rax'])

    def test_projection_iterates_closed_subsystem_with_partial_state(self):
        p = self.program(dict(self.ins, rax=self.a.opaque('unavailable', 64),
                             r10=self.a.bin('add', self.ins['r10'], self.a.const(1, 64), 64)))
        regs = p.closure('A')['registers']
        state = dict.fromkeys(regs, 0)
        for _ in range(100):
            state = p.run_projection('A', state, {}, regs)
        self.assertEqual(100, state['r10'])
        self.assertNotIn('rax', state)

    def test_local_effect_requires_witness_and_preserves_correlation(self):
        effect = self.a.opaque('mul_undefined_pf#7', 8)
        incoming = self.a.opaque('mul_undefined_pf#6', 8)
        p = self.program(dict(self.ins, rax=self.a.zext(effect, 64),
                             rbx=self.a.zext(effect, 64), rcx=self.a.zext(incoming, 64)),
                         effects={effect: dict(kind='undefined_flag', flag='pf', site=4096, step=7)})
        b = p.blocks['A']
        self.assertEqual(1, len(b['local_effects']))
        self.assertEqual(1, len(b['residual_inputs']))
        self.assertNotIn('rax', p.closure('A')['registers'])
        self.assertEqual([], p.validation['roundtrip_errors'])
        with self.assertRaises(KeyError):
            p.run_projection('A', {}, {}, ['rax', 'rbx'])
        for witness in (0, 1, 0):
            got = p.run_projection('A', {}, {}, ['rax', 'rbx'],
                                   {('effect', 'A', effect): witness})
            self.assertEqual(dict(rax=witness, rbx=witness), got)
        restored = Program.from_dict(json.loads(json.dumps(p.to_dict())))
        self.assertEqual(p.closure('A'), restored.closure('A'))

    def test_signature_ignores_block_label_but_includes_guards_and_constants(self):
        p = self.program()
        q = self.program(key='B')
        self.assertEqual(p.closure('A')['signature'], q.closure('B')['signature'])
        r = self.program(guards=[(self.ins['r10'], self.a.const(7, 64))])
        self.assertNotEqual(p.closure('A')['signature'], r.closure('A')['signature'])
        s = self.program(dict(self.ins, r10=self.a.const(8, 64)))
        self.assertNotEqual(p.closure('A')['signature'], s.closure('A')['signature'])

    def test_memory_dependent_register_is_not_register_only_closed(self):
        loaded = self.a.load(self.ins['rdi'], 64, 0)
        p = self.program(dict(self.ins, rax=loaded))
        self.assertNotIn('rax', p.closure('A')['registers'])
        self.assertTrue(p.closure('A')['outputs']['rax']['memory'])

    def test_effect_scope_is_verified(self):
        effect = self.a.opaque('flag', 8)
        p = self.program(dict(self.ins, rax=self.a.zext(effect, 64)),
                         effects={effect: dict(kind='undefined_flag', flag='af', site=4096, step=1)})
        p.nodes[p.blocks['A']['local_effects'][0]]['block'] = 'foreign'
        self.assertTrue(p.verify())


class EffectOriginTests(unittest.TestCase):
    def test_mul_origin_and_checkpoint_rollback(self):
        from core.lift64 import Lifter
        from core.mach import Machine
        from core.state import LifterCheckpoint
        from tests.test_state_repair import stub_image
        a = Arena()
        m = Machine(a, 0x10000)
        l = Lifter(stub_image(image_bytes=lambda addr, size: bytes.fromhex('48 f7 e1')),
                   m, a, 0x1000, 0x2000)
        checkpoint = LifterCheckpoint(l)
        l.step(0x1000)
        first = list(l.local_effects)
        self.assertEqual(4, len(first))
        self.assertEqual({'sf', 'zf', 'af', 'pf'}, {origin['flag'] for _, origin in first})
        self.assertTrue(all(origin['site'] == 0x1000 for _, origin in first))
        checkpoint.restore(l)
        self.assertEqual([], l.local_effects)
        l.step(0x1000)
        self.assertTrue({r for r, _ in first}.isdisjoint(r for r, _ in l.local_effects))


class GuardDomainTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('z3'), 'optional z3-solver is not installed')
    def test_proves_modular_output_bound_without_sampling_away_wraparound(self):
        from core.guarddomain import derive_bound
        a = Arena()
        x = a.init_reg('r10')
        after = a.zext(a.bin('add', a.trunc(x, 32), a.const(1, 32), 32), 64)
        condition = a.bin('ult', after, a.const(183, 64), 64)
        result = derive_bound(a, x, after, [(condition, a.const(1, 8))], 2, 1)
        self.assertEqual(182, result['maximum_output'])
        self.assertEqual('complete_bitvector_equivalence', result['proof'])

    @unittest.skipUnless(find_spec('z3'), 'optional z3-solver is not installed')
    def test_disconnected_domain_is_rejected_by_proof(self):
        from core.guarddomain import derive_bound
        a = Arena()
        x = a.init_reg('r10')
        after = a.bin('add', x, a.const(1, 64), 64)
        p = a.bin('ult', after, a.const(183, 64), 64)
        hidden = a.bin('eq', x, a.const(1 << 63, 64), 64)
        condition = a.bin('or', p, hidden, 8)
        self.assertIsNone(derive_bound(a, x, after, [(condition, a.const(1, 8))], 2, 1))

    def test_unknown_dependency_cannot_be_treated_as_true(self):
        from core.guarddomain import derive_bound
        a = Arena()
        x = a.init_reg('r10')
        after = a.bin('add', x, a.const(1, 64), 64)
        self.assertIsNone(derive_bound(a, x, after,
                                     [(a.opaque('other', 8), a.const(1, 8))], 2, 1))

    def test_segment_boundary_addresses_use_consistent_representation(self):
        from core.segment import closed_subsystem
        row = dict(entry_ip='0x1000', next_ip=0x1000,
                   closure=dict(registers=['r10'], signature='same'))
        self.assertTrue(closed_subsystem([row, row])['same_control_boundary'])
        changed = dict(row, next_ip=0x2000)
        self.assertFalse(closed_subsystem([row, changed])['same_control_boundary'])


if __name__ == '__main__':
    unittest.main()
