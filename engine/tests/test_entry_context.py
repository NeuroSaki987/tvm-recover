"""Caller-specialized entry states retain real stack writes and call semantics."""
import os
import unittest
from pathlib import Path

from core.entrycontext import EntryContext, EntryContextError, EntryContextFactory, EntryContextSpec
from tests.test_entry_boundaries import Image, jump


class ReplayImage(Image):
    imports = {}

    def image_bytes(self, va, size):
        return self.read_va(va, size)

    def image_u8(self, va):
        data = self.read_va(va, 1)
        return data[0] if data else None

    def is_loader_bound(self, va, size):
        return False

    def foldable(self, va, size):
        return self.section_for_va(va) is not None


def fixture(prefix=None):
    prefix = bytes.fromhex('4883ec38 48c744242034120000') if prefix is None else prefix
    site = 0x1000 + len(prefix)
    code = bytearray(b'\xcc' * 0x200)
    code[:len(prefix)] = prefix
    code[len(prefix):len(prefix) + 5] = b'\xe8' + jump(site, 0x1100)[1:]
    code[0x100:0x105] = jump(0x1100, 0x3000)
    img = ReplayImage(code, pdata=[(0x1000, site + 5)])
    for section in img.sections:
        section.writable = False
    img.blobs[0x3000] = bytes.fromhex('4c8b442428 c3').ljust(0x100, b'\xcc')
    return img, site


class EntryContextTests(unittest.TestCase):
    def test_caller_stack_argument_and_return_address_are_preserved(self):
        img, site = fixture()
        prepared = EntryContextFactory(EntryContextSpec(0x3000, 0x1000, site))(img, 0x3000, 0x3100, 0x8000)
        a, m, L = prepared.arena, prepared.machine, prepared.lifter
        self.assertEqual(prepared.entry, 0x3000)
        self.assertEqual(a.as_const(m.reg('rsp', 64)), 0x8000 - 0x38 - 8)
        self.assertEqual(a.as_const(m.load(m.reg('rsp', 64), 8, 0, img.image_u8)), site + 5)
        self.assertEqual(L.step(prepared.entry)[0], 'next')
        self.assertEqual(a.as_const(m.reg('r8', 64)), 0x1234)
        self.assertGreater(m.mem_version, 0)
        self.assertEqual(prepared.contract['scope'], 'caller-specialized')
        self.assertEqual(prepared.contract['target_chain'], [0x1100, 0x3000])
        self.assertEqual(a.node(m.reg('rcx', 64))[0], 'init')

    def test_factory_replays_fresh_state_for_each_sampling_or_recovery(self):
        img, site = fixture()
        factory = EntryContextFactory(EntryContextSpec(0x3000, 0x1000, site))
        first = factory(img, 0x3000, 0x3100, 0x8000)
        second = factory(img, 0x3000, 0x3100, 0x8000)
        self.assertIsNot(first.arena, second.arena)
        self.assertIsNot(first.machine, second.machine)
        self.assertEqual(first.contract, second.contract)

    def test_unknown_branch_is_not_forced_toward_the_call(self):
        img, site = fixture(bytes.fromhex('85c9 7402 9090'))
        with self.assertRaises(EntryContextError) as caught:
            EntryContextFactory(EntryContextSpec(0x3000, 0x1000, site))(img, 0x3000, 0x3100, 0x8000)
        self.assertEqual(caught.exception.reason, 'prefix-stop')
        self.assertIn('native_branch', str(caught.exception))

    def test_call_to_another_root_is_refused(self):
        img, site = fixture()
        with self.assertRaises(EntryContextError) as caught:
            EntryContextFactory(EntryContextSpec(0x3010, 0x1000, site))(img, 0x3000, 0x3100, 0x8000)
        self.assertEqual(caught.exception.reason, 'target-mismatch')

    def test_opaque_prefix_call_is_not_a_source_of_known_context(self):
        prefix = b'\xe8' + jump(0x1000, 0x9000)[1:]
        img, site = fixture(prefix)
        with self.assertRaises(EntryContextError) as caught:
            EntryContextFactory(EntryContextSpec(0x3000, 0x1000, site))(img, 0x3000, 0x3100, 0x8000)
        self.assertEqual(caught.exception.reason, 'opaque-prefix-call')

    def test_budget_and_unvalidated_entry_remain_explicit_refusals(self):
        img, site = fixture()
        with self.assertRaises(EntryContextError) as caught:
            EntryContextFactory(EntryContextSpec(0x3000, 0x1000, site, max_steps=0))(img, 0x3000, 0x3100, 0x8000)
        self.assertEqual(caught.exception.reason, 'prefix-budget')
        img.pdata = []
        with self.assertRaises(EntryContextError) as caught:
            EntryContextFactory(EntryContextSpec(0x3000, 0x1000, site))(img, 0x3000, 0x3100, 0x8000)
        self.assertEqual(caught.exception.reason, 'unvalidated-caller-entry')

    def test_real_core3_caller_preserves_symbolic_fifth_argument(self):
        # The real sample has no fixed location; point TVM_SAMPLES at the corpus root (the
        # directory holding `07-samples/`) to run this against a real CORE3 image.
        root = os.environ.get('TVM_SAMPLES', '')
        sample = Path(root) / '07-samples' / 'extended' / 'core3' / 'ACE-CORE.sys3' if root else None
        if sample is None or not sample.exists():
            self.skipTest('set TVM_SAMPLES to the sample corpus root to run this')
        from core.peimg import PeImage
        img = PeImage(str(sample))
        vm = img.section_by_name('.tvm0')
        context = EntryContext(0x14004cff8, 0x14004d020, 0x145bea7b1)
        a, m, contract = context.replay(img, vm.va, vm.end, 0x7ffffffefda0)
        rsp = a.as_const(m.reg('rsp', 64))
        value = m.load(a.const(rsp + 0x28, 64), 8, 0, img.image_u8)
        node = a.node(value)
        self.assertEqual(node[0], 'load')
        self.assertEqual(a.as_const(node[1]), 0x1400beae0)
        self.assertIsNone(a.as_const(value))
        self.assertEqual(contract['prefix_steps'], 7)
        self.assertEqual(contract['target_chain'], [0x14004d9b8, 0x145bea7b1])
        self.assertEqual(contract['scope'], 'caller-specialized')


if __name__ == '__main__':
    unittest.main()
