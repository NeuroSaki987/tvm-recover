"""Memory is a carried state component, with lazy byte-accurate write epochs."""
import json
import random
import unittest
from types import SimpleNamespace

from core.expr import Arena
from core.gueststate import Program, build
from core.xstate import REGS
from tests.test_guest_state import block


class JointMemoryTests(unittest.TestCase):
    def setUp(self):
        self.a = Arena()
        self.ins = {r: self.a.init_reg(r) for r in REGS}

    def program(self, stores=(), load=None, guards=(), outputs=None):
        p = Program(entry='A')
        params = {r: p.add('param', 64, block='A', reg=r) for r in REGS}
        mem = p.add('memparam', 0, block='A')
        start = mem
        for addr, value, size in stores:
            address = params[addr] if isinstance(addr, str) else p.add('c', 64, value=addr)
            if value is None:
                mem = p.add('havoc', 0, (mem, address), bytes=size, native_version=mem + 1)
            else:
                val = p.add('opaque', 64, name=value, native_ref=500) if isinstance(value, str) else p.add('c', 64, value=value)
                mem = p.add('store', 0, (mem, address, val), bytes=size, native_version=mem + 1)
        out = dict(params)
        if load:
            addr, size = load
            address = params[addr] if isinstance(addr, str) else p.add('c', 64, value=addr)
            value = p.add('load', size * 8, (address, mem), native_version=mem)
            out['rax'] = p.add('zext', 64, (value,)) if size < 8 else value
        if outputs:
            out.update(outputs(p, params, mem))
        p.blocks['A'] = dict(params=params, memory_in=start, memory_out=mem,
                             outputs=out, guards=[[params[r], p.add('c', 64, value=v)] for r, v in guards],
                             return_outputs={}, residual_inputs=[], local_effects=[])
        self.assertEqual([], p.verify())
        return p

    def run_joint(self, p, memory=None, state=None, externals=None):
        return p.run_state('A', state or dict.fromkeys(REGS, 0), memory or {}, externals)

    def test_irrelevant_write_value_is_not_needed_by_a_load(self):
        p = self.program([(0x9000, 'unavailable', 8)], load=(0x8000, 1))
        state, mem = self.run_joint(p, {0x8000: 17})
        self.assertEqual(17, state['rax'])
        self.assertEqual(17, mem[0x8000])
        with self.assertRaises(KeyError):
            _ = mem[0x9000]

    def test_latest_write_overrides_missing_earlier_value(self):
        p = self.program([(0x8000, 'unavailable', 8), (0x8000, 0x1234, 8)], load=(0x8000, 8))
        state, mem = self.run_joint(p)
        self.assertEqual(0x1234, state['rax'])
        self.assertEqual(0x34, mem[0x8000])
        self.assertEqual(0x12, mem[0x8001])

    def test_overlap_wrap_and_havoc(self):
        mask = (1 << 64) - 1
        p = self.program([(mask, 0xAABB, 2), (0, 0xCC, 1)], load=(mask, 2))
        state, mem = self.run_joint(p)
        self.assertEqual(0xCCBB, state['rax'])
        self.assertEqual({mask: 0xBB, 0: 0xCC}, mem.materialize())
        forgotten = self.program([(0x8000, 7, 8), (0x8000, None, 1)], load=(0x8000, 1))
        with self.assertRaises(KeyError):
            self.run_joint(forgotten, {0x8000: 99})
        wiped = self.program([(0, None, 0)], load=(0x8000, 1))
        with self.assertRaises(KeyError):
            self.run_joint(wiped, {0x8000: 99})

    def test_symbolic_alias_decision_requires_address_but_not_disjoint_value(self):
        p = self.program([('rdi', 'missing', 8)], load=(0x8000, 1))
        state = dict.fromkeys(REGS, 0)
        state['rdi'] = 0x9000
        self.assertEqual(11, self.run_joint(p, {0x8000: 11}, state)[0]['rax'])
        state['rdi'] = 0x8000
        with self.assertRaises(KeyError):
            self.run_joint(p, {0x8000: 11}, state)

    def test_memory_feedback_across_segments_preserves_epochs_and_forks(self):
        def counter(p, params, mem):
            address = p.add('c', 64, value=0x9000)
            old = p.add('load', 64, (address, mem), native_version=0)
            return {'rax': old}
        # Read old memory in A; B writes next value. A must see the preceding B
        # state without reseeding its entry memory or losing the parent epoch.
        reader = self.program(outputs=counter)
        writer = self.program([(0x9000, 42, 8)])
        seed = {0x9000 + i: (7 >> (8 * i)) & 255 for i in range(8)}
        first, initial = self.run_joint(reader, seed)
        _, modified = self.run_joint(writer, initial)
        self.assertEqual(7, first['rax'])
        self.assertEqual(42, self.run_joint(reader, modified)[0]['rax'])
        self.assertEqual(7, self.run_joint(reader, initial)[0]['rax'])
        self.assertEqual(7, seed[0x9000])

    def test_register_memory_consistency_and_guards(self):
        from core.guestctx import GuestContext
        p = self.program(load=(0x8000, 1), guards=[('r10', 1)])
        p.assumptions['context_base'] = 0x8000
        state = dict.fromkeys(REGS, 0)
        state.update(rax=12, r10=1)
        result, mem = self.run_joint(p, {}, state)
        self.assertEqual(12, result['rax'])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.run_joint(p, {GuestContext(0x8000).addr_of('rax'): 99}, state)
        state['r10'] = 0
        with self.assertRaisesRegex(ValueError, 'guard'):
            self.run_joint(p, {}, state)

    def test_joint_closure_admits_entry_memory_but_preserves_external_obligations(self):
        p = self.program(load=(0x8000, 1))
        c = p.memory_closure('A')
        self.assertTrue(c['memory_closed'])
        self.assertEqual(list(REGS), c['registers'])
        self.assertTrue(c['closed'])
        q = self.program([(0x9000, 'missing', 8)], load=(0x8000, 1))
        c = q.memory_closure('A')
        self.assertTrue(c['memory_closed'])
        self.assertFalse(c['closed'])
        self.assertEqual(1, len(c['native_inputs']))
        self.assertEqual(1, c['outputs']['rax']['loads'])
        restored = Program.from_dict(json.loads(json.dumps(p.to_dict())))
        self.assertEqual(p.memory_closure('A'), restored.memory_closure('A'))

    def test_historical_memory_is_not_relabelled_current(self):
        p = self.program()
        h = p.add('history', 0, version=99)
        address = p.add('c', 64, value=0x8000)
        load = p.add('load', 64, (address, h), native_version=99)
        p.blocks['A']['outputs']['rax'] = load
        self.assertFalse(p.memory_closure('A')['memory_closed'])
        with self.assertRaises(KeyError):
            self.run_joint(p, {0x8000: 7})

    def test_random_differential_against_eager_bytes(self):
        rng = random.Random(0xB17E)
        for _ in range(40):
            stores = [(0x8000 + rng.randrange(8), rng.getrandbits(64), rng.choice([1, 2, 4, 8]))
                      for _ in range(20)]
            p = self.program(stores, load=(0x8000, 8))
            seed = {0x8000 + i: rng.randrange(256) for i in range(16)}
            expected = dict(seed)
            for addr, value, size in stores:
                for i in range(size):
                    expected[addr + i] = (value >> (i * 8)) & 255
            out, memory = self.run_joint(p, seed)
            self.assertEqual(expected, memory.materialize())
            self.assertEqual(sum(expected[0x8000 + i] << (i * 8) for i in range(8)), out['rax'])

    def test_measured_seed_is_explicit_initialization_and_never_reapplied(self):
        p = self.program(load=(0x8000, 1))
        p.blocks['A']['entry_memory_seed'] = {'32768': 17}
        with self.assertRaises(KeyError):
            self.run_joint(p)
        memory = p.initial_memory('A')
        self.assertEqual(17, self.run_joint(p, memory)[0]['rax'])
        memory[0x8000] = 99
        self.assertEqual(17, p.initial_memory('A')[0x8000])
        writer = self.program([(0x8000, 42, 1)])
        _, updated = self.run_joint(writer, memory)
        self.assertEqual(42, self.run_joint(p, updated)[0]['rax'])

    def test_inputs_and_witnesses_are_frozen_in_the_carried_memory(self):
        p = self.program([(0x9000, 'value', 8)])
        external = {('opaque', 500): 12}
        _, memory = self.run_joint(p, externals=external)
        external[('opaque', 500)] = 99
        self.assertEqual(12, memory[0x9000])

    def test_global_havoc_kills_old_unknown_addresses_for_materialization(self):
        p = self.program()
        b = p.blocks['A']
        address = p.add('opaque', 64, native_ref=501, name='missing_address')
        value = p.add('c', 64, value=17)
        store = p.add('store', 0, (b['memory_in'], address, value), bytes=8, native_version=1)
        b['memory_out'] = p.add('havoc', 0, (store, address), bytes=0, native_version=2)
        _, memory = self.run_joint(p, {0x8000: 99})
        self.assertIsNone(memory.get(0x8000))
        self.assertEqual({}, memory.materialize())

    def test_finite_image_backing_is_preserved_and_written_bytes_override_it(self):
        from core.statememory import MemoryInput
        seed = {0x8001: 99}
        image = bytearray([10, 11, 12, 13])
        source = MemoryInput(seed, [(0x8000, image)])
        with self.assertRaises(AttributeError):
            source.overrides = {0x8000: 99}
        with self.assertRaises(AttributeError):
            source.regions = ()
        seed[0x8001] = 4
        image[0] = 8
        p = self.program([(0x8002, 42, 1)], load=(0x8000, 4))
        state, memory = self.run_joint(p, source)
        self.assertEqual(10 | (99 << 8) | (42 << 16) | (13 << 24), state['rax'])
        self.assertEqual({0x8001: 99, 0x8002: 42}, memory.materialize([0x8001, 0x8002]))
        self.assertEqual({0x8000: 10, 0x8001: 99, 0x8002: 42, 0x8003: 13}, memory.materialize())
        self.assertIsNone(memory.get(0x7FFF))

    def test_joint_output_must_agree_with_its_physical_backing_bytes(self):
        p = self.program()
        p.assumptions['context_base'] = 0x8000
        p.blocks['A']['outputs']['rax'] = p.add('c', 64, value=9)
        with self.assertRaisesRegex(ValueError, 'output.*disagree'):
            self.run_joint(p)

    def test_disjoint_write_scan_does_not_cache_a_byte_at_every_epoch(self):
        p = self.program([(0x9000 + i * 8, i, 8) for i in range(1000)], load=(0x8000, 1))
        _, memory = self.run_joint(p, {0x8000: 23})
        byte_tasks = [task for task in memory.owner.cache if task[0] == 'byte']
        self.assertLess(len(byte_tasks), 5)

    def test_guarded_edges_carry_selected_values_and_memory(self):
        from core.mach import Machine
        machine = Machine(self.a, 0x10000)
        pointer = self.a.const(0x9000, 64)
        machine.store(pointer, self.a.const(42, 64), 8, 1)
        first = block('A', self.ins, self.ins, 0, machine.mem_version, succs=['B'])
        predicate = self.a.bin('eq', self.ins['r10'], self.a.const(1, 64), 64)
        first.guest_edges = {'B': [
            dict(values=dict(self.ins, rbx=self.a.const(7, 64)), pins={predicate: self.a.const(0, 8)}),
            dict(values=dict(self.ins, rbx=self.a.const(9, 64)), pins={predicate: self.a.const(1, 8)})]}
        second = block('B', self.ins, dict(self.ins, rax=self.a.load(pointer, 64, machine.mem_version)),
                       machine.mem_version, machine.mem_version)
        p = build(SimpleNamespace(entry='A', blocks={'A': first, 'B': second}), self.a)
        for selector, expected in [(0, 7), (1, 9)]:
            state = dict.fromkeys(REGS, 0)
            state['r10'] = selector
            state, memory = p.run_state_edge('A', 'B', state, {})
            state, memory = p.run_state('B', state, memory)
            self.assertEqual(42, state['rax'])
            self.assertEqual(expected, state['rbx'])
        with self.assertRaises(ValueError):
            p.run_state_edge('A', 'missing', state, memory)

    def test_edge_only_history_still_prevents_memory_closure(self):
        p = self.program()
        b = p.blocks['A']
        history = p.add('history', 0, version=99)
        address = p.add('c', 64, value=0x8000)
        loaded = p.add('load', 64, (address, history), native_version=99)
        p.edges = [dict(source='A', target='A', args=dict(b['outputs'], rax=loaded),
                        memory=b['memory_out'], guards=[], recovered=True)]
        self.assertFalse(p.memory_closure('A')['memory_closed'])
