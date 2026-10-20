import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core import ctxlayout


class LayoutVoteTests(unittest.TestCase):
    def test_stack_slot_anchors_window_instead_of_earliest_overlap(self):
        sample = {0x100 + 8 * i: 0 for i in range(17)}
        sample[0x120] = 0x800000
        sample[0x150] = 0x140001000
        self.assertEqual((0x100, 2), ctxlayout.vote(
            sample, 0x140000000, 0x140010000, 0x800000))

    def test_multiple_stack_anchored_windows_are_ambiguous(self):
        sample = {0x120: 0x800000, 0x150: 0x140001000,
                  0x220: 0x800000, 0x250: 0x140001008}
        self.assertEqual((0, 0), ctxlayout.vote(
            sample, 0x140000000, 0x140010000, 0x800000))

    def test_tied_vote_totals_are_not_proof_of_one_base(self):
        votes = [(0x100, 2)] * 4 + [(0x200, 2)] * 4
        self.assertEqual((None, 4), ctxlayout.locate_by_vote(votes, 0, 1))

    def test_unique_repeated_vote_is_accepted(self):
        self.assertEqual((0x100, 4), ctxlayout.locate_by_vote(
            [(0x100, 2)] * 4, 0, 1))

    def test_probe_measurement_starts_with_fresh_stack(self):
        path = Path(__file__).parents[1] / 'probe' / 'ctx_map.py'
        spec = importlib.util.spec_from_file_location('ctx_map_test', path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        starts = []
        class FakeLifter:
            def __init__(self, img, m, a, lo, hi):
                self.m, self.a = m, a
            def step(self, ip):
                starts.append(self.a.as_const(self.m.reg('rsp', 64)))
                self.m.set_reg('rsp', self.a.const(starts[-1] - 0x100, 64), 64)
                return 'stop', SimpleNamespace(kind='test', site=ip)
        img = SimpleNamespace(section_by_name=lambda name: SimpleNamespace(va=0x1000, end=0x2000))
        with patch.object(probe, 'PeImage', return_value=img), \
             patch.object(probe, 'Lifter', FakeLifter), \
             patch.object(probe.ctxlayout, 'for_image', return_value=None), \
             patch.object(probe.sys, 'argv', ['ctx_map.py', 'stub', '0x1000', '--steps=1']), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, probe.main())
        self.assertEqual([probe.STACK_BASE, probe.STACK_BASE], starts)
