"""Keep the legacy slot/binding counterexample reproducible, and pin the corrected default.

Current execution uses explicit captured values through valueflow. The legacy
readings remain opt-in diagnostics; in_sync is not a correctness certificate.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLVM = HERE.parent
sys.path.insert(0, str(LLVM))
# The counterexample needs the *engine's* lowerer, which is the authority on what a name means.
sys.path.insert(0, str(LLVM.parent / 'engine'))

import guestir as G  # noqa: E402
import interp as I  # noqa: E402
import probe_register_file_reading as R  # noqa: E402

#: One block that writes the rcx slot twice and reads the *first* name after the second write.
#: `guest_rcx_2 = ~guest_rcx_1` is the `home`, and `v3 = guest_rcx_1 | 0x10` is the stale reference:
#: slot-now reads `~A`, name-bound reads `A`.
LISTING = """\
; devirtualized 0x1000
; recovery {'path_sensitive': True, 'proof_dispatch': True, 'conditioned_edges': 0, 'proofs': {}, 'loop_cutpoints': 0}
; blocks 1   unresolved 0   instructions 40
; lines 6   state 3   computation kept 4   suppressed 0
0x1000@ctx0x0000000000000000:   [return]   ; #0
    guest_rcx_1 = 0x1010   [ctx+0x10 = rcx]   ; write to ctx @0x1004
    v2:64 = ~guest_rcx_1
    guest_rcx_2 = v2:64   ; home rcx
    v3:64 = guest_rcx_1 | 0x10
    guest_rdx_1 = v3:64   [ctx+0x18 = rdx]   ; write to ctx @0x1008
"""


def parse() -> G.Function:
    return G.parse_listing(LISTING, Path('synthetic.txt'))


class RegisterReadingTest(unittest.TestCase):
    def test_listing_parses(self):
        fn = parse()
        self.assertEqual(len(fn.blocks), 1)
        kinds = [st.kind for st in fn.blocks[0].stmts]
        self.assertIn('home', kinds)

    def test_the_two_readings_disagree(self):
        fn = parse()
        block = fn.blocks[0]
        out = {}
        for reading in ('slot', 'binding'):
            ctx = I.Ctx({})
            I.run_block(block, ctx, I.Mem(regions=[(0, 1 << 64)]), I.Inputs(strict=False),
                        reading=reading)
            out[reading] = ctx.read(G.CTX_LAYOUT['rdx'], 64)
        # guest_rcx_1 = 0x1010, ~A = 0xffffffffffffefef
        self.assertEqual(out['binding'], 0x1010 | 0x10)
        self.assertEqual(out['slot'], 0xFFFFFFFFFFFFEFEF | 0x10)
        self.assertNotEqual(out['slot'], out['binding'])

    def test_default_reading_preserves_the_binding(self):
        """The executable default preserves captured values; slot is an audit-only legacy mode."""
        fn = parse()
        block = fn.blocks[0]
        got = []
        for kwargs in ({}, {'reading': 'binding'}):
            ctx = I.Ctx({})
            I.run_block(block, ctx, I.Mem(regions=[(0, 1 << 64)]), I.Inputs(strict=False),
                        **kwargs)
            got.append(ctx.read(G.CTX_LAYOUT['rdx'], 64))
        self.assertEqual(got[0], got[1])
        self.assertEqual(got[0], 0x1010)

    def test_stale_reference_is_in_sync(self):
        """The slot's last writer is the register's last definition, so slot-now == register-file."""
        rec = R.classify(parse())
        self.assertEqual(len(rec['stale']), 1)
        row = rec['stale'][0]
        self.assertEqual(row['reg'], 'rcx')
        self.assertEqual(row['reading_name'], 'guest_rcx_1')
        self.assertEqual(row['reading_slot'], 'guest_rcx_2')
        self.assertEqual(row['reading_register'], 'guest_rcx_2')
        self.assertTrue(row['in_sync'])
        self.assertFalse(row['register_ahead'])
        self.assertFalse(row['slot_ahead'])

    def test_ctx_load_is_a_read_and_the_count_is_robust(self):
        """`stale_reg_refs` counts `ctx-load` as a write; with only real writes the count is the same."""
        rec = R.classify(parse())
        strict = R.classify(parse(), R.TRUE_WRITE_KINDS)
        self.assertEqual(len(rec['stale']), len(strict['stale']))


class LegacyReadingRegressionTest(unittest.TestCase):
    """Retain the original counterexample for the explicit legacy audit modes.

    The production default is tested against the original Arena trace separately
    in test_valueflow; agreement between two legacy slot readers proves nothing
    about the corrected executable value flow.
    """

    def _lower(self):
        from types import SimpleNamespace
        from core.expr import Arena
        from core.lower import Lowering
        a = Arena()
        x = a.init_reg('rcx')
        y = a.un('not', x, 64)
        # Three immutable value nodes: the third stores the ORIGINAL x, not the live rcx.
        trace = [(1, 'store', (0x100, 'rcx', x)),
                 (2, 'store', (0x101, 'rcx', y)),
                 (3, 'store', (0x102, 'rbx', x))]
        lines = Lowering(a, {}).events(SimpleNamespace(guest_trace=trace))
        block = G.Block('synthetic', 0, 0, 'return', 0,
                        stmts=[G._parse_stmt(line) for line in lines])
        return lines, G.Function(root=0, blocks=[block]), block

    def test_the_lowerer_emits_the_captured_value(self):
        lines, _fn, _b = self._lower()
        self.assertIn('guest_rcx_1 = in.rcx', '\n'.join(lines))
        self.assertIn('guest_rcx_2 = v1:64', '\n'.join(lines))
        # The last statement uses the EARLIER name even though the slot now holds ~X.
        self.assertIn('guest_rbx_1 = guest_rcx_1', '\n'.join(lines))

    def test_in_sync_is_true_and_still_wrong(self):
        _lines, fn, _b = self._lower()
        rec = R.classify(fn)
        self.assertEqual(len(rec['stale']), 1)
        row = rec['stale'][0]
        self.assertTrue(row['in_sync'],
                        'the two current-state labels agree -- which is why in_sync cannot settle this')
        self.assertEqual(row['reading_name'], 'guest_rcx_1')
        self.assertEqual(row['reading_slot'], 'guest_rcx_2')

    def test_counterexample_is_a_real_divergence(self):
        """The emitted trace asks for X; the shipped reading produces ~X."""
        _lines, _fn, block = self._lower()
        got = {}
        for reading in ('slot', 'binding'):
            ctx = I.Ctx({'rcx': 0x1234})
            I.run_block(block, ctx, I.Mem(regions=[(0, 1 << 64)]), I.Inputs(strict=False),
                        reading=reading)
            got[reading] = ctx.read(G.CTX_LAYOUT['rbx'], 64)
        self.assertEqual(got['binding'], 0x1234, 'the captured value, which the trace asks for')
        self.assertEqual(got['slot'], (~0x1234) & ((1 << 64) - 1),
                         'the slot, which the shipped transcription produces')
        self.assertNotEqual(got['slot'], 0x1234,
                            'DEFECT: the shipped reading does not compute the emitted trace here')

    def test_differential_cannot_see_it(self):
        """Both sides of the differential parse the same listing, so they agree on the wrong answer."""
        _lines, _fn, block = self._lower()
        outs = {}
        for reading in ('slot', 'slot'):
            ctx = I.Ctx({'rcx': 0x1234})
            I.run_block(block, ctx, I.Mem(regions=[(0, 1 << 64)]), I.Inputs(strict=False),
                        reading=reading)
            outs.setdefault(reading, ctx.read(G.CTX_LAYOUT['rbx'], 64))
        self.assertEqual(outs['slot'], (~0x1234) & ((1 << 64) - 1))


if __name__ == '__main__':
    unittest.main()
