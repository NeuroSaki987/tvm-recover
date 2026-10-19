"""State correctness regressions, independent of the protected binary."""
import unittest
from types import SimpleNamespace

from core.expr import Arena
from core.mach import Machine
from core.explore import MachineSnapshot
from core.lift64 import Lifter
from core.guestctx import GuestContext, GuestCache


def no_image(_addr):
    return None


def stub_image(**kw):
    """An image double that satisfies the interface `Lifter` actually requires.

    It used to be `SimpleNamespace(imports={})`, which was enough only because `summaries_default`
    was `False` and `_guest_call` therefore never reached `Lifter.executable` -> `section_for_va`.
    Turning the summaries on made two tests raise `AttributeError` inside the callee they were not
    testing at all. A double that models less than the interface it stands in for makes the tests
    pass for the wrong reason, so the missing methods are here rather than left out.
    """
    base = {
        "imports": {},
        "section_for_va": lambda va: None,
        "image_u8": lambda addr: None,
        "foldable": lambda addr, size: False,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def lifter():
    a = Arena()
    m = Machine(a, 0x10000)
    l = Lifter(stub_image(), m, a, 0x1000, 0x2000)
    cache = GuestCache()
    cache.note('rax', 'rax')
    cache.boundaries = 1
    l.enable_guest_census(GuestContext(0x8000), cache)
    return a, m, l


class MemoryRepairTests(unittest.TestCase):
    def setUp(self):
        self.a = Arena()
        self.m = Machine(self.a, 0x10000)
        self.p = self.a.opaque('p', 64)

    def c(self, value):
        return self.a.const(value, 64)

    def test_snapshot_restores_memory_version_and_windows(self):
        snap = MachineSnapshot(self.m)
        old = self.m.mem_version
        self.m.store(self.p, self.c(5), 8, 1)
        self.m.sym_store_windows.append((123, 8, 1))
        snap.restore(self.m)
        self.assertEqual(old, self.m.mem_version)
        self.assertEqual([], self.m.sym_store_windows)

    def test_branch_writes_allocate_distinct_tokens(self):
        snap = MachineSnapshot(self.m)
        self.m.store(self.c(0x9000), self.c(1), 8, 1)
        left = self.m.load(self.p, 8, 2, no_image)
        snap.restore(self.m)
        self.m.store(self.c(0x9000), self.c(2), 8, 3)
        right = self.m.load(self.p, 8, 4, no_image)
        self.assertNotEqual(left, right)

    def test_unknown_write_invalidates_concrete_value(self):
        self.m.store(self.c(0x9000), self.c(1), 8, 1)
        self.m.store(self.p, self.c(5), 8, 2)
        got = self.m.load(self.c(0x9000), 8, 3, no_image)
        self.assertIsNone(self.a.as_const(got))

    def test_unknown_write_does_not_resurrect_image_bytes(self):
        self.m.store(self.p, self.c(5), 8, 1)
        got = self.m.load(self.c(0x9000), 8, 2, lambda _: 0x11)
        self.assertIsNone(self.a.as_const(got))

    def test_readonly_image_bytes_are_stable_on_nonfaulting_paths(self):
        self.m.readonly = lambda addr, size: 0x9000 <= addr and addr + size <= 0xa000
        self.m.store(self.p, self.c(5), 8, 1)
        got = self.m.load(self.c(0x9000), 8, 2, lambda _: 0x11)
        self.assertEqual(0x1111111111111111, self.a.as_const(got))

    def test_exact_symbolic_store_forwards(self):
        self.m.store(self.p, self.c(5), 8, 1)
        self.assertEqual(5, self.a.as_const(self.m.load(self.p, 8, 2, no_image)))

    def test_bounded_disjoint_store_preserves_concrete_bytes(self):
        self.m.store(self.c(0x9000), self.c(7), 8, 1)
        idx = self.a.zext(self.a.opaque('idx', 8), 64)
        addr = self.a.bin('add', self.c(0xA000), idx, 64)
        self.m.store(addr, self.c(3), 8, 2)
        self.assertEqual(7, self.a.as_const(self.m.load(self.c(0x9000), 8, 3, no_image)))

    def test_historical_load_keeps_its_version_after_substitution(self):
        q = self.a.opaque('q', 64)
        v = self.a.load(q, 64, 17)
        rebuilt = self.a.substitute(v, {q: self.c(0x9000)})
        self.assertEqual(17, self.a.node(rebuilt)[3])

    def test_scratch_precision_loss_cannot_refold_after_window_closes(self):
        from core.explore import Recoverer
        a, m, l = lifter()
        img = SimpleNamespace(foldable=lambda p, n: True, image_u8=lambda p: 0x11)
        recoverer = Recoverer.__new__(Recoverer)
        recoverer.img = img
        recoverer._install_load_folder(a, l)
        m.symbolic_window = (0x9000, 0x9100)
        v = m.load(a.const(0x9000, 64), 4, 1, img.image_u8)
        self.assertEqual(v, m.load(a.const(0x9000, 64), 4, 2, img.image_u8))
        m.symbolic_window = None
        self.assertIsNone(a.as_const(a.substitute(v, {})))

    def test_restored_branches_do_not_reuse_epoch_numbers(self):
        snap = MachineSnapshot(self.m)
        self.m.store(self.c(0x9000), self.c(1), 8, 1)
        left_epoch = self.m.mem_version
        snap.restore(self.m)
        self.m.store(self.c(0x9000), self.c(2), 8, 2)
        self.assertNotEqual(left_epoch, self.m.mem_version)

    def test_disjoint_symbolic_store_preserves_forwarding(self):
        q = self.a.bin('add', self.p, self.c(8), 64)
        self.m.store(self.p, self.c(7), 8, 1)
        self.m.store(q, self.c(9), 8, 2)
        self.assertEqual(7, self.a.as_const(self.m.load(self.p, 8, 3, no_image)))

    def test_overlapping_symbolic_store_invalidates_forwarding(self):
        q = self.a.bin('add', self.p, self.c(4), 64)
        self.m.store(self.p, self.c(7), 8, 1)
        self.m.store(q, self.c(9), 8, 2)
        self.assertIsNone(self.a.as_const(self.m.load(self.p, 8, 3, no_image)))

    def test_forget_never_resurrects_image(self):
        p = self.c(0x9000)
        self.m.forget(p, 8)
        self.assertIsNone(self.a.as_const(self.m.load(p, 8, 1, lambda _: 0x11)))

    def test_two_target_store_preserves_condition_and_unrelated_memory(self):
        cond = self.a.bin('eq', self.p, self.c(0), 64)
        self.m.store(self.c(0x9000), self.c(7), 8, 1)
        self.m.store(self.c(0x8000), self.c(11), 8, 1)
        target = self.a.select(cond, self.c(0x9000), self.c(0xa000), 64)
        self.m.store(target, self.c(99), 8, 2)
        loaded = self.m.load(self.c(0x9000), 8, 3, no_image)
        self.assertEqual(99, self.a.as_const(self.a.substitute(loaded, {cond: self.c(1)})))
        self.assertEqual(7, self.a.as_const(self.a.substitute(loaded, {cond: self.c(0)})))
        self.assertEqual(11, self.a.as_const(self.m.load(self.c(0x8000), 8, 3, no_image)))

    def test_historical_load_cannot_read_later_memory(self):
        from core.explore import Recoverer
        a, m, l = lifter()
        img = SimpleNamespace(foldable=lambda p, n: True, image_u8=lambda p: 0x11)
        recoverer = Recoverer.__new__(Recoverer)
        recoverer.img = img
        recoverer._install_load_folder(a, l)
        p = a.opaque('address')
        old = m.load(p, 8, 1, img.image_u8)
        m.store(a.const(0x9000, 64), a.const(7, 64), 8, 2)
        rebuilt = a.substitute(old, {p: a.const(0x9000, 64)})
        # Old state sees the original image, not the later overlay's 7.
        self.assertEqual(0x1111111111111111, a.as_const(rebuilt))


class CensusRepairTests(unittest.TestCase):
    def test_checkpoint_restores_census_and_machine(self):
        from core.state import LifterCheckpoint
        a, m, l = lifter()
        checkpoint = LifterCheckpoint(l)
        m.store(a.const(0x8000, 64), a.opaque('new_value'), 8, 1)
        m.load(a.const(0x8000, 64), 8, 2, no_image)
        checkpoint.restore(l)
        self.assertEqual([], l.guest_trace)
        self.assertEqual({}, l.guest_identity)
        self.assertEqual({}, m.mem)

    def test_failed_native_summary_rolls_back_census(self):
        from core.x86sum import summarize
        from core.lift64 import Stop
        a, m, l = lifter()
        epoch = m.mem_version
        def step(ip):
            m.store(a.const(0x8000, 64), a.opaque('trial'), 8, ip)
            m.load(a.const(0x8000, 64), 8, ip, no_image)
            return 'stop', Stop('unreadable', ip)
        l.step = step
        result = summarize(l, 0x1100, 0x1200, 10)
        self.assertFalse(result.ok)
        self.assertEqual([], l.guest_trace)
        self.assertEqual({}, l.guest_identity)
        self.assertEqual({}, m.mem)
        self.assertEqual(epoch, m.mem_version)

    def test_narrow_load_does_not_claim_full_register_identity(self):
        a, m, l = lifter()
        m.store(a.const(0x8000, 64), a.opaque('whole'), 8, 1)
        value = m.load(a.const(0x8000, 64), 4, 2, no_image)
        self.assertNotIn(a.zext(value, 64), l.guest_identity)

    def test_offset_partial_store_is_recorded(self):
        a, m, l = lifter()
        m.store(a.const(0x8002, 64), a.const(0xabcd, 16), 2, 1)
        parts = [payload for _, kind, payload in l.guest_trace if kind == 'part']
        self.assertEqual(1, len(parts))
        self.assertEqual(('rax', 2, 2), parts[0][1:4])

    def test_native_call_enters_ordered_trace(self):
        _, _, l = lifter()
        l._guest_call(None, 0x1100, 0x1105, 0x4000)
        self.assertEqual(1, sum(kind == 'call' for _, kind, _ in l.guest_trace))

    def test_native_calls_clobber_distinct_flag_values(self):
        _, m, l = lifter()
        l._guest_call(None, 0x1100, 0x1105, 0x4000)
        first = dict(m.flags)
        l._guest_call(None, 0x1200, 0x1205, 0x4000)
        for flag in ('cf', 'pf', 'af', 'zf', 'sf', 'of'):
            self.assertNotEqual(first[flag], m.flags[flag], flag)

    def test_runtime_store_wins_over_nonfoldable_image(self):
        a, m, l = lifter()
        l.img = SimpleNamespace(foldable=lambda p, size: False, image_u8=lambda p: 0)
        m.store(a.const(0x9000, 64), a.const(7, 64), 8, 1)
        self.assertEqual(7, a.as_const(l._read_at(0x9000, 8, 2)))
        self.assertIsNone(a.as_const(l._read_at(0xa000, 8, 3)))

    def test_exact_budget_return_is_not_truncated(self):
        from core.explore import Recoverer
        from core.lift64 import Stop
        a, m, l = lifter()
        r = Recoverer(SimpleNamespace(), 0x1000, 0x2000, step_budget=1)
        r.prepare = lambda root: (a, m, l)
        r.identity = lambda lifter: 0
        l.step = lambda ip: ('stop', Stop('return', ip))
        r.recover(0x1100)
        self.assertFalse(r.truncated)

    def test_cache_frequency_does_not_create_identity_at_arbitrary_entry(self):
        from core.explore import Recoverer
        from core.lift64 import Stop
        a, m, l = lifter()
        r = Recoverer(SimpleNamespace(), 0x1000, 0x2000, step_budget=5)
        r.prepare = lambda root: (a, m, l)
        r.identity = lambda lifter: 0
        l.step = lambda ip: ('stop', Stop('return', ip))
        r.recover(0x1100)
        self.assertEqual({}, l.guest_identity)

    def test_rejected_retry_does_not_leak_trial_events(self):
        from core.explore import Recoverer
        from core.lift64 import Stop
        a, m, l = lifter()
        r = Recoverer(SimpleNamespace(), 0x1000, 0x2000, step_budget=5)
        r.prepare = lambda root: (a, m, l)
        r.identity = lambda lifter: 0
        r._scratch_window = lambda lifter: (0x9000, 0x9100)
        r._brief = lambda lifter, stop: 'bad target'
        r._executable = lambda target: False
        def step(ip):
            trial = m.symbolic_window is not None
            l.guest_trace.append((l.steps, 'carry', ('rax', 'rax', a.const(0, 64), a.const(2 if trial else 1, 64))))
            return 'stop', Stop('unreadable' if trial else 'bad_target', ip)
        l.step = step
        cfg = r.recover(0x1100)
        events = next(iter(cfg.blocks.values())).guest_trace
        self.assertEqual(1, len(events))
        self.assertEqual(1, a.as_const(events[0][2][-1]))


if __name__ == '__main__':
    unittest.main()
