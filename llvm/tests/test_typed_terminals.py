"""Analysis terminals keep a distinct ABI status and never execute a host trap."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'toolchain_spike'))

import guestir as G
import interp as I
import lower_llvm as L
import whole as W


def fixture(kind='fastfail'):
    return ('; devirtualized 0x100\n'
            '0x100@ctx0x200: [branch] ; #0\n'
            '    v1:8 = zext8(flag_cf)\n'
            '    ; branch condition: v1:8   ; successor when 0: 0x110@ctx0x200'
            '   ; successor when 1: 0x120@ctx0x200\n'
            '    -> 0x110@ctx0x200, 0x120@ctx0x200\n'
            '0x110@ctx0x200: [return] ; #1\n'
            '    v2:64 = 0x11\n'
            '    -> return\n'
            '0x120@ctx0x200: [terminal] ; #2\n'
            '    call ntoskrnl.exe!KeBugCheck @0x125\n'
            '    v3:32 = 0x3\n'
            f'    -> <{kind}: code=v3:32>\n')


class TypedTerminalTests(unittest.TestCase):
    def test_status_decoder_rejects_unassigned_tag(self):
        from execution_status import decode
        self.assertEqual(decode(2)['status'], 'normal')
        self.assertEqual(decode((1 << 63) | 2)['status'], 'terminal')
        self.assertEqual(decode(-3)['steps'], 2)
        self.assertEqual(decode(-3)['status'], 'exhausted')
        with self.assertRaises(ValueError):
            decode(1 << 62)

    def test_engine_fastfail_code_spelling_is_preserved_without_claiming_value(self):
        fn = G.parse_listing(fixture().replace('code=v3:32>', 'code=v3:32 @0x129>'))
        _, man = W.emit_whole(fn)
        self.assertEqual(man['terminals'][0]['detail'], 'code=v3:32 @0x129')

    def test_parser_preserves_kind_detail_and_call_provenance(self):
        for kind in ('noreturn', 'fastfail'):
            with self.subTest(kind=kind):
                fn = G.parse_listing(fixture(kind))
                b = fn.blocks[-1]
                self.assertEqual(b.term, kind)
                self.assertEqual(b.terminal_detail, 'code=v3:32')
                self.assertEqual(b.calls, [('ntoskrnl.exe!KeBugCheck', 0x125)])

    def test_interpreter_distinguishes_normal_terminal_and_fuel(self):
        fn = G.parse_listing(fixture())
        for bit, fuel, expected in ((0, 5, 2), (1, 5, (1 << 63) | 2), (1, 1, (1 << 64) - 2)):
            with self.subTest(bit=bit, fuel=fuel):
                rr = I.run_function(fn, I.Ctx(), I.Mem(),
                                    I.Inputs(undef={'flag_cf': bit}, strict=False), fuel=fuel)
                self.assertEqual(rr.encoded_status, expected)
                self.assertEqual(rr.finished, bit == 0)
                self.assertEqual(rr.exhausted, fuel == 1)
                self.assertEqual(bool(rr.terminal), bit == 1 and fuel > 1)
                if rr.terminal:
                    self.assertEqual(rr.terminal['kind'], 'fastfail')
                    self.assertEqual(rr.terminal['detail'], 'code=v3:32')

    def test_terminal_per_block_emission_refuses_ambiguous_void_return(self):
        fn = G.parse_listing(fixture())
        with self.assertRaisesRegex(L.HazardError, 'whole-function'):
            L.emit_function(fn, {2})

    def test_unknown_terminal_stays_refused(self):
        fn = G.parse_listing(fixture('unknown'))
        with self.assertRaises(W.WholeRefusal):
            W.emit_whole(fn)

    def test_compiled_whole_function_distinguishes_both_branch_arms_and_timeout(self):
        import diff_whole
        for kind in ('noreturn', 'fastfail'):
            fn = G.parse_listing(fixture(kind))
            text, man = W.emit_whole(fn)
            self.assertEqual(man['terminals'][0]['kind'], kind)
            self.assertEqual(man['terminals'][0]['calls'][0]['site'], 0x125)
            self.assertNotIn('llvm.trap', text)
            self.assertEqual(L.hazard_lint(text), [])
            for opt in (0, 2):
                report = diff_whole.compare_function(fn, fuel=5, opt=opt)
                self.assertTrue(report['ok'], report)
                self.assertEqual({c['status native'] for c in report['cases']},
                                 {'normal', 'terminal'})
            report = diff_whole.compare_function(fn, fuel=1, opt=0)
            self.assertTrue(report['ok'], report)
            self.assertEqual({c['status native'] for c in report['cases']}, {'exhausted'})

    def test_generated_chunked_payload_preserves_terminal_status_and_step_totals(self):
        import payload as P
        import miniload as M
        import execution_status as S
        with tempfile.TemporaryDirectory(prefix='typed-terminals-') as td:
            root = Path(td)
            listing = root / '0x100.txt'
            listing.write_text(fixture(), encoding='utf-8')
            out = root / 'payload'
            with patch.object(P, 'MAX_WHOLE_STATEMENTS', 0), patch.object(P, 'WITNESS_VALUE', 1):
                meta = P.build([listing], out, opt=0, fuel=5, split=1)
            self.assertEqual(meta['chunked_functions'], 1)
            header = P.read_header((out / 'payload.bin').read_bytes())
            header['section_rva'] = meta['image_first_rva']
            image = M.Mapped((out / 'build' / 'payload.dll').read_bytes(), None, 'synthetic payload')
            try:
                native = M.run_function(image, header, 0)
                decoded = S.decode(native['steps'])
                self.assertEqual(decoded['status'], 'terminal')
                self.assertEqual(decoded['steps'], 2)
                result = M.run_payload(image, header)
                self.assertEqual(result['steps'], 2)
                self.assertEqual(result['results'], [(1 << 63) | 2])
                exhausted = M.run_function(image, header, 0, fuel=1)
                self.assertEqual(S.decode(exhausted['steps'])['status'], 'exhausted')
                self.assertEqual(S.decode(exhausted['steps'])['steps'], 1)
            finally:
                image.free()


if __name__ == '__main__':
    unittest.main(verbosity=2)
