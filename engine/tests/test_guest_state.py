"""Executable Guest state transfers, tested independently of the protected image."""
import unittest
from types import SimpleNamespace

from core.expr import Arena
from core.mach import Machine
from core.guestctx import GuestContext
from core import xstate


def block(key, entry, exit_, start=0, end=0, succs=(), trace=()):
    return SimpleNamespace(key=key, guest_entry=entry, guest_exit=exit_,
                           memory_entry=start, memory_exit=end, succs=list(succs),
                           kind='branch' if succs else 'return', guest_trace=list(trace))


class GuestStateTests(unittest.TestCase):
    def setUp(self):
        self.a = Arena()
        self.inputs = {r: self.a.init_reg(r) for r in xstate.REGS}

    def compile(self, *blocks):
        from core.gueststate import build
        return build(SimpleNamespace(entry=blocks[0].key,
                                     blocks={b.key: b for b in blocks}), self.a)

    def test_copy_arithmetic_is_not_a_guest_commit(self):
        b = SimpleNamespace(guest_trace=[(1, 'carry', ('rax', 'r11',
                              self.inputs['rax'], self.a.const(99, 64)))])
        got = xstate.exit_state(self.a, b, {r: ('in', r) for r in xstate.REGS})
        self.assertEqual(('in', 'rax'), got['rax'])

    def test_capture_is_passive_and_covers_flags(self):
        from core.gueststate import capture
        m = Machine(self.a, 0x10000)
        ctx = GuestContext(0x8000)
        m.store(self.a.const(ctx.addr_of('rflags'), 64), self.a.const(0x246, 64), 8, 1)
        before = (dict(m.mem), m.mem_version, list(m.sym_loads), list(m.unmapped_loads))
        state = capture(m, ctx, lambda _: None)
        self.assertEqual(set(xstate.REGS), set(state))
        self.assertEqual(0x246, self.a.as_const(state['rflags']))
        self.assertEqual(before, (dict(m.mem), m.mem_version, m.sym_loads, m.unmapped_loads))

    def test_observation_does_not_allocate_native_dag_nodes(self):
        from core.gueststate import observe
        m = Machine(self.a, 0x10000)
        count = len(self.a.nodes)
        observation = observe(m, GuestContext(0x8000), lambda _: None)
        self.assertEqual(count, len(self.a.nodes))
        m.store(self.a.const(0x8000, 64), self.a.const(42, 64), 8, 1)
        # Observation belongs to the old state, even if materialised much later.
        self.assertIsNone(self.a.as_const(observation.materialize()['rax']))

    def test_loop_transfers_are_parametric_and_do_not_widen(self):
        out = dict(self.inputs, rax=self.a.bin('add', self.inputs['rax'], self.a.const(1, 64), 64))
        p = self.compile(block('loop', self.inputs, out, succs=['loop']))
        state = dict.fromkeys(xstate.REGS, 0)
        for i in range(1000):
            state, memory = p.run_block('loop', state, {})
        self.assertEqual(1000, state['rax'])
        self.assertEqual([], p.verify())
        self.assertEqual(17, len(p.blocks['loop']['outputs']))
        self.assertLess(len(p.nodes), 60)

    def test_diamond_passes_predecessor_values(self):
        left = dict(self.inputs, rax=self.a.const(7, 64))
        right = dict(self.inputs, rax=self.a.const(11, 64))
        join_in = dict(self.inputs, rax=self.a.opaque('arrival_rax', 64))
        join_out = dict(join_in, rax=self.a.bin('add', join_in['rax'], self.a.const(2, 64), 64))
        p = self.compile(block('A', self.inputs, self.inputs, succs=['L', 'R']),
                         block('L', self.inputs, left, succs=['J']),
                         block('R', self.inputs, right, succs=['J']),
                         block('J', join_in, join_out))
        seed = dict.fromkeys(xstate.REGS, 0)
        for side, result in [('L', 9), ('R', 13)]:
            s, mem = p.run_block(side, seed, {})
            s, _ = p.run_block('J', s, mem)
            self.assertEqual(result, s['rax'])
        self.assertEqual({'L', 'R'}, {e['source'] for e in p.edges if e['target'] == 'J'})

    def test_constant_and_alias_entry_require_guards(self):
        entry = dict(self.inputs, rax=self.a.const(7, 64), rbx=self.inputs['rcx'])
        out = dict(entry, rdx=self.a.const(7, 64))
        p = self.compile(block('A', entry, out))
        seed = dict.fromkeys(xstate.REGS, 0)
        with self.assertRaises(ValueError):
            p.run_block('A', seed, {})
        seed['rax'] = 7
        seed['rbx'], seed['rcx'] = 5, 5
        s, _ = p.run_block('A', seed, {})
        self.assertEqual(7, s['rdx'])
        seed['rcx'] = 6
        with self.assertRaises(ValueError):
            p.run_block('A', seed, {})

    def test_partial_flags_and_memory_match_native_execution(self):
        from core.gueststate import capture
        ctx = GuestContext(0x8000)
        m = Machine(self.a, 0x10000)
        for reg, value in self.inputs.items():
            m.store(self.a.const(ctx.addr_of(reg), 64), value, 8, 1)
        entry = capture(m, ctx, lambda _: None)
        start = m.mem_version
        m.store(self.a.const(ctx.addr_of('rflags') + 1, 64), self.a.const(0xA5, 8), 1, 2)
        m.store(self.a.const(0x9000, 64), self.a.const(0xBEEF, 16), 2, 3)
        out = capture(m, ctx, lambda _: None)
        p = self.compile(block('A', entry, out, start, m.mem_version))
        seed = dict.fromkeys(xstate.REGS, 0)
        seed['rflags'] = 0xFEDCBA9876543210
        s, mem = p.run_block('A', seed, {})
        self.assertEqual(0xFEDCBA987654A510, s['rflags'])
        self.assertEqual(0xEF, mem[0x9000])
        self.assertEqual(0xBE, mem[0x9001])

    def test_memory_loads_observe_order_and_havoc(self):
        m = Machine(self.a, 0x10000)
        ptr = self.inputs['rcx']
        before = m.load(ptr, 8, 1, lambda _: None)
        m.store(ptr, self.a.const(42, 64), 8, 2)
        after = m.load(ptr, 8, 3, lambda _: None)
        out = dict(self.inputs, rax=before, rbx=after)
        p = self.compile(block('A', self.inputs, out, 0, m.mem_version))
        seed = dict.fromkeys(xstate.REGS, 0)
        seed['rcx'] = 0x9000
        memory = {0x9000 + i: (9 >> (8 * i)) & 255 for i in range(8)}
        s, mem = p.run_block('A', seed, memory)
        self.assertEqual(9, s['rax'])
        self.assertEqual(42, s['rbx'])
        self.assertEqual(42, mem[0x9000])
        self.assertEqual(9, memory[0x9000])

    def test_havoc_never_resurrects_old_bytes(self):
        m = Machine(self.a, 0x10000)
        m.store(self.a.const(0x9000, 64), self.a.const(8, 64), 8, 1)
        start = m.mem_version
        m.forget(self.a.const(0x9000, 64), 8)
        loaded = m.load(self.a.const(0x9000, 64), 8, 2, lambda _: None)
        p = self.compile(block('A', self.inputs, dict(self.inputs, rax=loaded), start, m.mem_version))
        with self.assertRaises(KeyError):
            p.run_block('A', dict.fromkeys(xstate.REGS, 0), {0x9000 + i: 0 for i in range(8)})

    def test_backing_state_and_memory_share_one_entry_contract(self):
        ctx = GuestContext(0x8000)
        p = self.compile(block('A', self.inputs, self.inputs))
        p.assumptions['context_base'] = ctx.base
        state = dict.fromkeys(xstate.REGS, 0)
        state['rax'] = 0x1122334455667788
        _, mem = p.run_block('A', state, {})
        self.assertEqual(0x88, mem[ctx.addr_of('rax')])
        with self.assertRaises(ValueError):
            p.run_block('A', state, {ctx.addr_of('rax'): 0x77})

    def test_serialized_transfers_execute_without_native_arena(self):
        import json
        from core.gueststate import Program
        p = self.compile(block('A', self.inputs, self.inputs))
        restored = Program.from_dict(json.loads(json.dumps(p.to_dict())))
        state = {r: 0x123456 + i for i, r in enumerate(xstate.REGS)}
        self.assertEqual((state, {}), restored.run_block('A', state, {}))

    def test_return_registers_are_separate_from_backing_slots(self):
        b = block('A', self.inputs, self.inputs)
        b.native_return = dict(self.inputs, rax=self.a.const(42, 64))
        p = self.compile(b)
        seed = dict.fromkeys(xstate.REGS, 1)
        self.assertEqual(1, p.run_block('A', seed, {})[0]['rax'])
        self.assertEqual(42, p.run_return('A', seed, {})[0]['rax'])

    def test_edge_conditions_are_enforced(self):
        cond = self.a.bin('eq', self.inputs['rcx'], self.a.const(0, 64), 64)
        b = block('A', self.inputs, self.inputs, succs=['B'])
        b.guest_edges = {'B': {'values': self.inputs, 'pins': {cond: self.a.const(1, 64)}}}
        p = self.compile(b, block('B', self.inputs, self.inputs))
        seed = dict.fromkeys(xstate.REGS, 0)
        self.assertEqual(seed, p.run_edge('A', 'B', seed, {})[0])
        seed['rcx'] = 3
        with self.assertRaises(ValueError):
            p.run_edge('A', 'B', seed, {})

    def test_inactive_select_arm_does_not_require_unknown_memory(self):
        cond = self.a.bin('eq', self.inputs['rcx'], self.a.const(0, 64), 64)
        dead = self.a.load(self.a.const(0xDEAD, 64), 64, 0)
        selected = self.a.select(cond, self.a.const(42, 64), dead, 64)
        p = self.compile(block('A', self.inputs, dict(self.inputs, rax=selected)))
        state = dict.fromkeys(xstate.REGS, 0)
        self.assertEqual(42, p.run_block('A', state, {})[0]['rax'])
        state['rcx'] = 1
        with self.assertRaises(KeyError):
            p.run_block('A', state, {})

    def test_edge_only_native_dependencies_are_reported(self):
        b = block('A', self.inputs, self.inputs, succs=['B'])
        flag = self.a.opaque('native_edge_flag', 8)
        b.guest_edges = {'B': {'values': self.inputs, 'pins': {flag: self.a.const(1, 8)}}}
        p = self.compile(b, block('B', self.inputs, self.inputs))
        self.assertTrue(p.blocks['A']['residual_inputs'])

    def test_segment_path_constraints_apply_to_block_execution(self):
        b = block('segment', self.inputs, self.inputs)
        pred = self.a.bin('eq', self.inputs['rcx'], self.a.const(0, 64), 64)
        b.path_constraints = [(pred, self.a.const(1, 64))]
        p = self.compile(b)
        seed = dict.fromkeys(xstate.REGS, 0)
        self.assertEqual(seed, p.run_block('segment', seed, {})[0])
        seed['rcx'] = 9
        with self.assertRaises(ValueError):
            p.run_block('segment', seed, {})

    def test_same_target_preserves_both_conditioned_edge_variants(self):
        cond = self.a.bin('eq', self.inputs['rcx'], self.a.const(0, 64), 64)
        b = block('A', self.inputs, self.inputs, succs=['B', 'B'])
        b.guest_edges = {'B': [
            {'values': dict(self.inputs, rax=self.a.const(7, 64)), 'pins': {cond: self.a.const(1, 64)}},
            {'values': dict(self.inputs, rax=self.a.const(11, 64)), 'pins': {cond: self.a.const(0, 64)}}]}
        p = self.compile(b, block('B', self.inputs, self.inputs))
        seed = dict.fromkeys(xstate.REGS, 0)
        self.assertEqual(7, p.run_edge('A', 'B', seed, {})[0]['rax'])
        seed['rcx'] = 8
        self.assertEqual(11, p.run_edge('A', 'B', seed, {})[0]['rax'])

    def test_exported_native_branch_guard_survives_legacy_mode(self):
        from core.explore import Recoverer, Block, BlockKey, Cfg
        from core.lift64 import Stop
        from tests.test_state_repair import lifter
        a, m, l = lifter()
        self.a = a
        self.inputs = {r: a.init_reg(r) for r in xstate.REGS}
        recovery = Recoverer(SimpleNamespace(), 0x1000, 0x2000, path_sensitive=False)
        recovery._executable = lambda address: address in (0x1100, 0x1200)
        recovery.identity = lambda _lifter: 0
        key = BlockKey(0x1000, 0)
        source = Block(key, guest_entry=self.inputs, guest_exit=self.inputs)
        predicate = a.bin('eq', self.inputs['rcx'], a.const(0, 64), 64)
        stop = Stop('native_branch', 0x1000, pred=predicate, taken=0x1100, not_taken=0x1200)
        successors = recovery._terminate(l, stop, key, source, {}, Cfg())
        for target, snapshot, pins in successors:
            source.succs.append(target)
            source.guest_edges[target] = {'values': self.inputs,
                                         'pins': recovery.guest_edge_pins(l, stop, target, pins)}
        targets = [block(target, self.inputs, self.inputs) for target, _, _ in successors]
        p = self.compile(source, *targets)
        seed = dict.fromkeys(xstate.REGS, 0)
        self.assertEqual(seed, p.run_edge(key, targets[0].key, seed, {})[0])
        with self.assertRaises(ValueError):
            p.run_edge(key, targets[1].key, seed, {})

    def test_random_native_differential_all_slots_and_overlapping_stores(self):
        import random
        from core.gueststate import capture
        ctx = GuestContext(0x8000)
        rng = random.Random(0x4755455354)
        # The same native recipe runs once symbolically for compilation and
        # separately with concrete byte storage as the execution oracle.
        recipe = [(rng.choice(xstate.REGS), rng.randrange(8), 1, rng.getrandbits(8)) for _ in range(30)]
        recipe += [('rflags', 4, 4, 0x12345678), ('rax', 2, 2, 0xABCD)]

        def execute(machine, seeds):
            for r, v in seeds.items():
                machine.store(self.a.const(ctx.addr_of(r), 64), v, 8, 1)
            incoming = capture(machine, ctx, lambda _: None)
            version = machine.mem_version
            for r, off, size, value in recipe:
                machine.store(self.a.const(ctx.addr_of(r) + off, 64),
                              self.a.const(value, size * 8), size, 2)
            return incoming, version, capture(machine, ctx, lambda _: None)

        m = Machine(self.a, 0x10000)
        entry, start, output = execute(m, self.inputs)
        p = self.compile(block('A', entry, output, start, m.mem_version))
        self.assertEqual([], p.validation['roundtrip_errors'])
        for _ in range(100):
            seeds = {r: rng.getrandbits(64) for r in xstate.REGS}
            native = Machine(self.a, 0x10000)
            _, _, actual = execute(native, {r: self.a.const(v, 64) for r, v in seeds.items()})
            predicted, _ = p.run_block('A', seeds, {})
            self.assertEqual({r: self.a.as_const(v) for r, v in actual.items()}, predicted)


if __name__ == '__main__':
    unittest.main()
