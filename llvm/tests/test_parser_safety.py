"""Small counterexamples for call parsing and incomplete-listing emission."""

import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import guestir as G
import lower_llvm as L
import whole as W


def listing(body):
    return ('; devirtualized 0x100\n'
            '0x100@ctx0x200: [entry] ; #0\n'
            + body + '    -> return\n')


class TestCallArguments(unittest.TestCase):
    def test_nested_cast_and_parentheses(self):
        e = G.parse_expr('call 0x123(rcx=trunc32(in.rcx), rdx=(in.rdx + 0x1))')
        self.assertEqual(e[2], (
            ('rcx', ('cast', 'trunc', ('entry', 'rcx'), None, 32)),
            ('rdx', ('bin', 'add', ('entry', 'rdx'), ('const', 1, None), None))))

    def test_nested_commas_and_site(self):
        e = G.parse_expr('call 0x123(rcx=insert_bits(in.rcx, 0x1, 0x8, 0x8), '
                         'rdx=call 0x456(rcx=in.rax, rdx=in.rdx)) @0x789')
        self.assertEqual([name for name, _ in e[2]], ['rcx', 'rdx'])
        self.assertEqual(e[2][0][1][0], 'insert')
        self.assertEqual(e[2][1][1][0], 'call')
        self.assertEqual(len(e[2][1][1][2]), 2)
        self.assertEqual(e[3], 0x789)

    def test_reject_malformed_argument(self):
        for text in ('call 0x123(rcx=in.rcx junk)',
                     'call 0x123(rcx=in.rcx rdx=in.rdx)',
                     'call 0x123(rcx=trunc32(in.rcx)',
                     'call 0x123(rcx=in.rcx,)'):
            with self.subTest(text=text), self.assertRaises(G.ParseError):
                G.parse_expr(text)

    def test_empty_and_simple_calls_preserved(self):
        self.assertEqual(G.parse_expr('call 0x123()'), ('call', '0x123', (), 0))
        self.assertEqual(G.parse_expr('call 0x123(rcx=in.rcx)'),
                         ('call', '0x123', (('rcx', ('entry', 'rcx')),), 0))


class TestEmissionSafety(unittest.TestCase):
    def test_entry_context_is_preserved_and_refused_by_every_emission_path(self):
        contract = {'scope': 'caller-specialized', 'caller_entry': '0x400',
                    'call_site': '0x450', 'target_root': '0x100',
                    'precondition': {'rcx': '0x600'}}
        text = '; entry context: ' + json.dumps(contract) + '\n' + listing('    v1:64 = 0x1\n')
        fn = G.parse_listing(text, Path('specialized.txt'))
        self.assertEqual(fn.header['entry_context'], contract)
        for emit in (L.emit_function, W.emit_whole, lambda fn: W.emit_chunked(fn, 1),
                     lambda fn: L.Emitter(fn.blocks[0], fn).emit()):
            with self.subTest(emit=emit.__name__):
                with self.assertRaisesRegex(L.HazardError, 'guarded call-site bridge') as cm:
                    emit(fn)
                self.assertIn('specialized.txt:1', str(cm.exception))
        import payload as P
        with tempfile.TemporaryDirectory(prefix='entry-context-') as td:
            specialized = Path(td) / '0x100.txt'
            ordinary = Path(td) / '0x200.txt'
            specialized.write_text(text, encoding='utf-8')
            ordinary.write_text(listing('    v1:64 = 0x1\n').replace('0x100', '0x200'),
                                encoding='utf-8')
            module = P.emit_module([specialized, ordinary])
            self.assertEqual([m['root'] for m in module['manifests']], [0x200])
            self.assertEqual(module['refusals'][0]['root'], '0x100')
            self.assertIn('guarded call-site bridge', module['refusals'][0]['reason'])

    def test_malformed_or_ambiguous_entry_context_is_rejected(self):
        for payload in ('{bad json}', '[]', 'null', '1'):
            with self.subTest(payload=payload), self.assertRaisesRegex(G.ParseError,
                                                                       'entry context'):
                G.parse_listing('; entry context: ' + payload + '\n' + listing(''))
        with self.assertRaisesRegex(G.ParseError, 'duplicate entry context'):
            G.parse_listing('; entry context: {}\n; entry context: {}\n' + listing(''))

    def test_entry_context_cannot_hide_behind_indentation_or_unknown_scope(self):
        for prefix in ('', '    ', '\t'):
            fn = G.parse_listing(prefix + '; entry context: {"scope":"other"}\n' + listing(''))
            with self.subTest(prefix=repr(prefix)), self.assertRaises(L.HazardError):
                W.emit_whole(fn)

    def test_unknown_body_line_cannot_hide_behind_indentation(self):
        for prefix in ('', '\t'):
            with self.subTest(prefix=repr(prefix)):
                fn = G.parse_listing(listing(prefix + 'mystery [0x40000], 0x2\n'),
                                     Path('broken-indent.txt'))
                self.assertEqual(fn.unparsed, ['mystery [0x40000], 0x2'])
                for emit in (L.emit_function, W.emit_whole):
                    with self.assertRaises(L.HazardError) as cm:
                        emit(fn)
                    self.assertIn('broken-indent.txt:3', str(cm.exception))
                    self.assertIn('mystery [0x40000], 0x2', str(cm.exception))

    def test_body_indentation_does_not_drop_valid_effects_or_reject_notes(self):
        text = ('preamble note\n; devirtualized 0x100\n\n'
                '0x100@ctx0x200: [entry] ; #0\n'
                '; body comment\n\t; tab comment\n'
                'store64 [0x30000], 0x1\n\tstore64 [0x40000], 0x2\n'
                '\t-> return\n')
        fn = G.parse_listing(text)
        self.assertEqual(fn.unparsed, [])
        self.assertEqual([st.kind for st in fn.blocks[0].stmts], ['store', 'store'])
        for emit in (L.emit_function, W.emit_whole):
            text, _ = emit(fn)
            self.assertEqual(L.hazard_lint(text), [])

    def test_both_emitters_refuse_unparsed_effect_with_location(self):
        fn = G.parse_listing(listing('    mystery [0x40000], 0x2\n'), Path('broken.txt'))
        self.assertEqual(fn.unparsed, ['mystery [0x40000], 0x2'])
        for emit in (L.emit_function, W.emit_whole):
            with self.subTest(emit=emit.__name__):
                with self.assertRaises(L.HazardError) as cm:
                    emit(fn)
                self.assertIn('broken.txt:3', str(cm.exception))
                self.assertIn('mystery [0x40000], 0x2', str(cm.exception))

    def test_truncated_function_can_still_emit_settled_block(self):
        fn = G.parse_listing('; TRUNCATED\n' + listing('    v1:64 = 0x1\n'))
        self.assertTrue(fn.header['truncated'])
        text, manifests = L.emit_function(fn, {0})
        self.assertEqual(manifests[0]['block'], 0)
        self.assertEqual(L.hazard_lint(text), [])

    def test_constant_store_destination_in_manifest(self):
        fn = G.parse_listing(listing('    store64 [0x30000], 0x1\n'))
        _, manifests = L.emit_function(fn)
        _, manifest = W.emit_whole(fn)
        self.assertEqual(manifests[0]['guest_addrs'], [0x30000])
        self.assertEqual(manifest['guest_addrs'], [0x30000])

    def test_store_collects_nested_load_but_not_literal_value(self):
        st = G._parse_stmt('store64 [0x30000], load64 [0x50000]')
        self.assertEqual(set(L._abs_addresses(st)), {0x30000, 0x50000})
        st = G._parse_stmt('store64 [in.rcx], 0x60000')
        self.assertEqual(L._abs_addresses(st), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
