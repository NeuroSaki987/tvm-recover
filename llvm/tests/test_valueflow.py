"""Captured values survive slot writes, CFG edges and compilation cuts."""
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guestir as G
import interp as I


def fixture(body, tail=''):
    return G.parse_listing('; devirtualized 0x100\n'
        '0x100@ctx0x0: [return]   ; #0\n' + body + '    -> return\n' + tail)


class ValueFlowTest(unittest.TestCase):
    def test_actual_engine_counterexample(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'engine'))
        from core.expr import Arena
        from core.lower import Lowering
        from types import SimpleNamespace
        a = Arena()
        x = a.init_reg('rcx')
        trace = [(1, 'store', (0x100, 'rcx', x)),
                 (2, 'store', (0x101, 'rcx', a.un('not', x, 64))),
                 (3, 'store', (0x102, 'rbx', x))]
        block = G.Block('synthetic', 0, 0, 'return', 0, stmts=[
            G._parse_stmt(line) for line in Lowering(a, {}).events(SimpleNamespace(guest_trace=trace))])
        ctx = I.Ctx({'rcx': 0x1234})
        rr = I.run_block(block, ctx, I.Mem(), I.Inputs())
        self.assertEqual(rr.errors, [])
        self.assertEqual(ctx.slots()['rbx'], 0x1234)
        self.assertEqual(ctx.slots()['rcx'], (~0x1234) & ((1 << 64) - 1))

    def test_default_preserves_capture(self):
        fn = fixture('    guest_rcx_1 = in.rcx   ; write to ctx @0x101\n'
                     '    guest_rcx_2 = 0xff   ; write to ctx @0x102\n'
                     '    guest_rbx_1 = guest_rcx_1   ; write to ctx @0x103\n')
        ctx = I.Ctx({'rcx': 0x1234})
        rr = I.run_block(fn.blocks[0], ctx, I.Mem(), I.Inputs())
        self.assertEqual(rr.errors, [])
        self.assertEqual(ctx.slots()['rbx'], 0x1234)

    def test_explicit_read_captures_at_read_point(self):
        fn = fixture('    guest_rcx_1 = 0x12   ; write to ctx @0x101\n'
                     '    guest_rcx_1   ; read from ctx @0x102 [ctx+0x10 = rcx]\n'
                     '    guest_rcx_2 = 0x34   ; write to ctx @0x103\n'
                     '    guest_rbx_1 = guest_rcx_1   ; write to ctx @0x104\n')
        ctx = I.Ctx()
        rr = I.run_block(fn.blocks[0], ctx, I.Mem(), I.Inputs())
        self.assertEqual(rr.errors, [])
        self.assertEqual(ctx.slots()['rbx'], 0x12)

    def test_cross_block_capture_precedes_local_overwrite(self):
        fn = G.parse_listing('''; devirtualized 0x100
0x100@ctx0x0: [goto]   ; #0
    guest_rcx_1 = 0x12   ; write to ctx @0x101
    -> 0x200@ctx0x0
0x200@ctx0x0: [return]   ; #1
    guest_rcx_1 = 0x34   ; write to ctx @0x201
    guest_rbx_1 = #0:guest_rcx_1   ; write to ctx @0x202
    -> return
''')
        ctx = I.Ctx()
        rr = I.run_function(fn, ctx, I.Mem(), I.Inputs(), fuel=4)
        self.assertEqual(rr.errors, [])
        self.assertEqual(ctx.slots()['rbx'], 0x12)

    def test_compiled_capture_whole_and_chunk_plan(self):
        import whole as W
        import diff_whole
        fn = fixture('    guest_rcx_1 = in.rcx   ; write to ctx @0x101\n'
                     '    guest_rcx_2 = 0xff   ; write to ctx @0x102\n'
                     '    guest_rbx_1 = guest_rcx_1   ; write to ctx @0x103\n')
        for opt in (0, 2):
            report = diff_whole.compare_function(fn, fuel=4, opt=opt)
            self.assertTrue(report['ok'], report)
        pieces = W.chunk_plan(fn, 1)
        self.assertTrue(any('capture_b0_rcx_1' in p['live_out'] for p in pieces))

    def test_join_does_not_guess_the_first_predecessor(self):
        from valueflow import prepare, ValueFlowError
        fn = G.parse_listing('''; devirtualized 0x100
0x100@ctx0x0: [goto]   ; #0
    guest_rcx_1 = 0x12   ; write to ctx @0x101
    -> 0x200@ctx0x0, 0x300@ctx0x0
0x200@ctx0x0: [goto]   ; #1
    guest_rcx_1 = 0x34   ; write to ctx @0x201
    -> 0x300@ctx0x0
0x300@ctx0x0: [return]   ; #2
    guest_rbx_1 = #0:guest_rcx_1   ; write to ctx @0x301
    -> return
''')
        with self.assertRaisesRegex(ValueFlowError, 'continuity is not proved'):
            prepare(fn)
        # An explicit read is valid at the join whichever predecessor executes.
        b = fn.blocks[-1]
        b.stmts.insert(0, G._parse_stmt(
            '#0:guest_rcx_1   ; read from ctx @0x300 [ctx+0x10 = rcx]'))
        prepare(fn)

    def test_loop_carried_value_cannot_skip_the_backedge(self):
        from valueflow import prepare, ValueFlowError
        fn = G.parse_listing('''; devirtualized 0x100
0x100@ctx0x0: [goto]   ; #0
    guest_rcx_1 = 0x12   ; write to ctx @0x101
    -> 0x200@ctx0x0
0x200@ctx0x0: [goto]   ; #1
    guest_rbx_1 = #0:guest_rcx_1   ; write to ctx @0x201
    guest_rcx_1 = 0x34   ; write to ctx @0x202
    -> 0x200@ctx0x0
''')
        with self.assertRaises(ValueFlowError):
            prepare(fn)

    def test_owner_survives_infeasible_block_renumbering(self):
        fn = G.parse_listing('''; devirtualized 0x100
0x100@ctx0x0: [goto]   ; #0
    -> 0x300@ctx0x0
0x200@ctx0x0: [return]   ; #1
    -> <infeasible: false path>
0x300@ctx0x0: [goto]   ; #2
    guest_rcx_1 = 0x12   ; write to ctx @0x301
    -> 0x400@ctx0x0
0x400@ctx0x0: [return]   ; #3
    guest_rcx_1 = 0x34   ; write to ctx @0x401
    guest_rbx_1 = #2:guest_rcx_1   ; write to ctx @0x402
    -> return
''')
        self.assertEqual(fn.blocks[-1].index, 2)
        self.assertEqual(fn.blocks[-1].source_index, 3)
        ctx = I.Ctx()
        rr = I.run_function(fn, ctx, I.Mem(), I.Inputs(), fuel=4)
        self.assertEqual(rr.errors, [])
        self.assertEqual(ctx.slots()['rbx'], 0x12)

    def test_partial_write_defines_a_captured_full_value(self):
        fn = fixture('    guest_rax_1 = insert_bits(in.rax, 8, 8, v1:8)   ; PARTIAL write to ctx @0x101\n'
                     '    guest_rax_2 = 0x99   ; write to ctx @0x102\n'
                     '    guest_rbx_1 = guest_rax_1   ; write to ctx @0x103\n')
        fn.blocks[0].stmts.insert(0, G.Stmt('def', 'v1', 8, ('const', 0xaa, 8)))
        ctx = I.Ctx({'rax': 0x11223344})
        rr = I.run_block(fn.blocks[0], ctx, I.Mem(), I.Inputs())
        self.assertEqual(rr.errors, [])
        self.assertEqual(ctx.slots()['rbx'], 0x1122aa44)

    def test_native_chunked_and_whole_have_the_trace_value(self):
        import payload as P
        import miniload as M
        listing = '''; devirtualized 0x100
0x100@ctx0x0: [goto]   ; #0
    guest_rcx_1 = 0x1234   ; write to ctx @0x101
    -> 0x200@ctx0x0
0x200@ctx0x0: [return]   ; #1
    guest_rcx_1 = 0xff   ; write to ctx @0x201
    guest_rbx_1 = #0:guest_rcx_1   ; write to ctx @0x202
    -> return
'''
        with tempfile.TemporaryDirectory(prefix='valueflow-') as td:
            root = Path(td)
            src = root / '0x100.txt'
            src.write_text(listing)
            for split_mode, opt in ((False, 0), (False, 2), (True, 0), (True, 2)):
                out = root / f'{split_mode}-{opt}'
                with patch.object(P, 'MAX_WHOLE_STATEMENTS', 0 if split_mode else 7000):
                    meta = P.build([src], out, opt=opt, fuel=4, split=1)
                header = P.read_header((out / 'payload.bin').read_bytes())
                header['section_rva'] = meta['image_first_rva']
                image = M.Mapped((out / 'build' / 'payload.dll').read_bytes(), None, 'synthetic')
                try:
                    native = M.run_function(image, header, 0)
                    ctx = native['ctx']
                    self.assertEqual(int.from_bytes(ctx[8:16], 'little'), 0x1234)
                    self.assertEqual(int.from_bytes(ctx[16:24], 'little'), 0xff)
                    self.assertEqual(native['steps'], 2)
                finally:
                    image.free()


if __name__ == '__main__':
    unittest.main()
