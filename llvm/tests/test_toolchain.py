"""Exercise the toolchain API used by check_ir and differential execution."""
import ctypes
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'toolchain_spike'))
import toolchain

IR = 'target triple = "x86_64-pc-windows-msvc"\ndefine i32 @answer() { ret i32 42 }\n'


class ToolchainTests(unittest.TestCase):
    def test_assemble_verifies_and_reports_typed_errors(self):
        self.assertIn('answer', str(toolchain.assemble(IR)))
        for text in ('not LLVM IR', 'define i32 @broken() { ret i64 1 }'):
            with self.subTest(text=text), self.assertRaises(toolchain.VerifyError):
                toolchain.assemble(text)

    def test_invalid_optimization_levels_are_typed_errors(self):
        for level in (-1, 4, 'bogus', None, 1.5):
            with self.subTest(level=level), self.assertRaises(toolchain.ToolchainError):
                toolchain.emit_assembly(IR, opt=level)

    def test_target_facts_reports_real_layout(self):
        facts = toolchain.target_facts(IR)
        self.assertEqual(facts['triple'], 'x86_64-pc-windows-msvc')
        self.assertTrue(facts['data_layout'].startswith('e-'), facts)
        self.assertIn('p', facts['data_layout'])

    def test_missing_configured_linker_is_a_typed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'TVM_MSVC_BIN': tmp}):
                with self.assertRaisesRegex(toolchain.ToolchainError, 'could not run link.exe'):
                    toolchain.link_dll([], str(Path(tmp) / 'missing.dll'))

    @unittest.skipUnless(os.name == 'nt' and (shutil.which('link.exe') or os.environ.get('TVM_MSVC_BIN')), 'requires MSVC on Windows')
    def test_emit_coff_link_and_execute_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / 'answer.obj'
            dll = Path(tmp) / 'answer.dll'
            self.assertEqual(toolchain.emit_object(IR, str(obj)), str(obj))
            self.assertEqual(obj.read_bytes()[:2], b'\x64\x86')
            toolchain.link_dll([str(obj)], str(dll), extra_args=('/EXPORT:answer',))
            lib = ctypes.CDLL(str(dll))
            try:
                lib.answer.restype = ctypes.c_int
                self.assertEqual(lib.answer(), 42)
            finally:
                import _ctypes
                _ctypes.FreeLibrary(lib._handle)


if __name__ == '__main__':
    unittest.main()
