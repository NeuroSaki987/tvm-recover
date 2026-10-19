"""Control-flow recovery must execute each successor under its own condition."""
import unittest
from importlib.util import find_spec
from types import SimpleNamespace

from core.expr import Arena
from core.explore import Recoverer, Cfg, Block, BlockKey
from core.lift64 import Stop
from tests.test_state_repair import lifter, no_image, stub_image


class PathRecoveryTests(unittest.TestCase):
    def test_historical_normalization_cannot_leak_a_sibling_value(self):
        a, m, l = lifter()
        p = a.bin('eq', a.init_reg('rcx'), a.const(0, 64), 64)
        address = a.const(0x2000, 64)
        m.store(address, a.select(p, a.const(11, 64), a.const(22, 64), 64), 8, 0)
        r = Recoverer(stub_image(foldable=lambda p, n: True), 0x1000, 0x3000)
        r._install_load_folder(a, l)
        historical = a.load(address, 64, m.mem_version)
        from core.state import MachineSnapshot
        left, right = MachineSnapshot(m, lifter=l), MachineSnapshot(m, lifter=l)
        left.restrict(a, {p: a.const(1, 64)})
        right.restrict(a, {p: a.const(0, 64)})
        for snapshot, expected in [(left, 11), (right, 22)]:
            snapshot.restore(m, l)
            self.assertEqual(expected, a.as_const(m.load(address, 8, 0, no_image)))
            # This node predates the split. Its global normal form cannot be
            # a branch-local constant, even when its epoch equals current memory.
            self.assertIsNone(a.as_const(a.substitute(historical, {})))

    def branch(self, native):
        a, m, l = lifter()
        p = a.bin('eq', a.init_reg('rcx'), a.const(0, 64), 64)
        m.set_reg('rsp', a.select(p, a.const(0x9000, 64), a.const(0xa000, 64), 64), 64)
        m.store(a.const(0x8000, 64), m.reg('rsp', 64), 8, 1)
        r = Recoverer(SimpleNamespace(), 0x1000, 0x2000)
        r._executable = lambda t: t in (0x1100, 0x1200)
        r.identity = lambda L: L.a.as_const(L.m.reg('rsp', 64)) or 0
        key = BlockKey(0x1000, 0)
        if native:
            stop = Stop('native_branch', 0x1000, pred=p, taken=0x1100, not_taken=0x1200)
        else:
            dest = a.select(p, a.const(0x1100, 64), a.const(0x1200, 64), 64)
            stop = Stop('dispatch', 0x1000, dest=dest)
        result = r._terminate(l, stop, key, Block(key), {}, Cfg())
        by_target = {}
        for target, snapshot, pins in result:
            snapshot.restore(m, l)
            by_target[target.handler] = (a.as_const(m.reg('rsp', 64)),
                                       a.as_const(m.load(a.const(0x8000, 64), 8, 2, no_image)),
                                       pins, target.vip)
        self.assertEqual(0x9000, by_target[0x1100][0])
        self.assertEqual(0xa000, by_target[0x1200][0])
        self.assertEqual(0x9000, by_target[0x1100][1])
        self.assertEqual(0xa000, by_target[0x1200][1])
        self.assertEqual(0x9000, by_target[0x1100][3])
        self.assertEqual(0xa000, by_target[0x1200][3])

    def test_native_branch_conditions_machine_and_frame(self):
        self.branch(True)

    def test_dispatch_conditions_each_successor_separately(self):
        self.branch(False)

    @unittest.skipUnless(find_spec('z3'), 'optional z3-solver is not installed')
    def test_multiway_proof_conditions_each_target(self):
        a, m, l = lifter()
        idx = a.bin('and', a.init_reg('rcx'), a.const(3, 64), 64)
        address = a.bin('add', a.const(0x2000, 64),
                        a.bin('shl', idx, a.const(3, 64), 64), 64)
        a.versioned_load_folder = lambda p, w, v: (a.const(0x1100 + (p - 0x2000) * 2, w)
                                                   if 0x2000 <= p < 0x2020 else None)
        dest = a.load(address, 64, 4)
        m.set_reg('rax', dest, 64)
        r = Recoverer(SimpleNamespace(), 0x1000, 0x3000)
        r._executable = lambda t: t in (0x1100, 0x1110, 0x1120, 0x1130)
        r.identity = lambda L: 0
        key = BlockKey(0x1000, 0)
        result = r._terminate(l, Stop('dispatch', 0x1000, dest=dest), key, Block(key), {}, Cfg())
        self.assertEqual(4, len(result))
        for target, snapshot, pins in result:
            self.assertEqual(target.handler, a.as_const(snapshot.regs['rax']))
            self.assertEqual(target.handler, a.as_const(pins[dest]))
        self.assertEqual(1, r.proof_stats['complete'])

    @unittest.skipUnless(find_spec('z3'), 'optional z3-solver is not installed')
    def test_impossible_path_is_pruned_not_reported_as_a_hole(self):
        a, m, l = lifter()
        x = a.init_reg('rcx')
        p = a.bin('eq', x, a.const(0, 64), 64)
        q = a.bin('eq', x, a.const(1, 64), 64)
        pins = {p: a.const(1, 64), q: a.const(1, 64)}
        r = Recoverer(SimpleNamespace(), 0x1000, 0x3000)
        r._executable = lambda t: True
        key = BlockKey(0x1000, 0)
        block = Block(key)
        cfg = Cfg()
        result = r._terminate(l, Stop('dispatch', 0x1000, dest=a.init_reg('rdx')),
                              key, block, pins, cfg)
        self.assertEqual([], result)
        self.assertEqual('infeasible', block.kind)
        self.assertEqual(0, cfg.unresolved)


if __name__ == '__main__':
    unittest.main()
