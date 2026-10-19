"""Root discovery must not interpret an instruction's interior byte as a JMP."""
import struct
import unittest
from types import SimpleNamespace

from core.entries import analyze_entries, raw_candidates, vm_entries, vm_roots


class Image:
    def __init__(self, code, vm=0x3000, pdata=(), entry=0):
        self.image_base = 0
        self.is64 = True
        self.entry_point = entry
        self.pdata = list(pdata)
        self.directories = []
        self.sections = [
            SimpleNamespace(name='.text', va=0x1000, end=0x1200, rsize=0x200,
                            executable=True),
            SimpleNamespace(name='.tvm0', va=vm, end=vm + 0x100, rsize=0x100,
                            executable=True),
        ]
        self.blobs = {0x1000: bytes(code).ljust(0x200, b'\xcc'), vm: b'\xc3' * 0x100}

    def read_va(self, va, size):
        for base, blob in self.blobs.items():
            if base <= va < base + len(blob):
                return blob[va - base:va - base + size]
        return None

    def section_by_name(self, name):
        return next((s for s in self.sections if s.name == name), None)

    def section_for_va(self, va):
        return next((s for s in self.sections if s.va <= va < s.end), None)


def jump(site, target):
    return b'\xe9' + struct.pack('<i', target - site - 5)


class EntryBoundaryTests(unittest.TestCase):
    def test_modrm_e9_inside_anchored_sub_is_not_an_entry(self):
        # Actual CORE3 idiom: E9 is the ModRM of SUB RCX,2, not an opcode.
        code = bytes.fromhex('4883e90241ba05000080c3')
        vm = 0x1002 + 5 + int.from_bytes(code[3:7], 'little', signed=True)
        img = Image(code, vm, pdata=[(0x1000, 0x100b)])
        self.assertEqual(vm_entries(img), [])
        self.assertEqual(raw_candidates(img), [(0x1002, vm)])
        self.assertEqual(vm_entries(img, legacy=True), [(0x1002, vm)])
        row, = analyze_entries(img).candidates
        self.assertEqual(row.status, 'rejected')
        self.assertEqual(row.covering_instruction, (0x1000, 4, 'sub rcx, 2'))

    def test_anchored_direct_jump_is_preserved(self):
        img = Image(jump(0x1000, 0x3000), pdata=[(0x1000, 0x1005)])
        self.assertEqual(vm_entries(img), [(0x1000, 0x3000)])

    def test_branch_target_can_intentionally_overlap_an_instruction(self):
        code = bytearray(b'\xcc' * 0x110)
        code[:10] = b'\x48\xb8' + jump(0x1002, 0x3000) + b'\x90\x90\x90'
        code[10] = 0xc3
        code[0x100:0x105] = jump(0x1100, 0x1002)
        img = Image(code, pdata=[(0x1000, 0x100b), (0x1100, 0x1105)])
        self.assertEqual(vm_entries(img), [(0x1002, 0x3000)])

    def test_unproven_boundary_is_preserved_without_guessed_rejection(self):
        img = Image(jump(0x1000, 0x3000))
        self.assertEqual(vm_entries(img), [(0x1000, 0x3000)])
        self.assertEqual(vm_entries(img, strict=True), [])
        self.assertEqual(vm_roots(img), [0x3000])
        self.assertEqual(analyze_entries(img).candidates[0].status, 'unproven')

    def test_pe_entry_point_provides_a_boundary_anchor(self):
        img = Image(jump(0x1000, 0x3000), entry=0x1000)
        self.assertEqual(vm_entries(img), [(0x1000, 0x3000)])

    def test_incomplete_traversal_does_not_reject_a_potential_overlap(self):
        code = bytes.fromhex('4883e90241ba05000080c3')
        vm = 0x1002 + 5 + int.from_bytes(code[3:7], 'little', signed=True)
        img = Image(code, vm, pdata=[(0x1000, 0x100b)])
        result = analyze_entries(img, instruction_limit=1)
        self.assertFalse(result.complete)
        self.assertEqual(result.candidates[0].status, 'unproven')

    def test_export_without_unwind_record_provides_an_anchor(self):
        img = Image(jump(0x1000, 0x3000))
        blob = bytearray(img.blobs[0x1000])
        struct.pack_into('<I', blob, 0x180 + 20, 1)  # NumberOfFunctions
        struct.pack_into('<I', blob, 0x180 + 28, 0x11c0)  # AddressOfFunctions
        struct.pack_into('<I', blob, 0x1c0, 0x1000)
        img.blobs[0x1000] = bytes(blob)
        img.directories = [(0x1180, 0x40)]
        self.assertEqual(vm_entries(img, strict=True), [(0x1000, 0x3000)])
        self.assertIn('export-index:0', analyze_entries(img).candidates[0].provenance)


if __name__ == '__main__':
    unittest.main()
