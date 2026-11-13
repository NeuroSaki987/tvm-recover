"""A call graph needs two things the RUNTIME_FUNCTION table does not say.

Both are measured on `ACE-CORE202706.sys` (`8F35B7677E79`), where 2131 of 2654
records (80 %) carry `UNW_FLAG_CHAININFO`:

1. **The caller calls the primary, not the chunk.** PGO splits one function into
   several records; attributing a call site to the record it falls inside makes a
   cold chunk's own start look uncalled. `Functions.primary` folds the chain.

2. **A tail call leaves no `call` site.** `jmp rel32` is an edge too, and an
   obfuscated driver reaches many helpers only that way.

These tests pin the two rules on a synthetic image, in both directions: a chunk
with CHAININFO resolves to its parent, and a chunk *without* one keeps its own
start (so the fold cannot be faked by folding everything).
"""
import unittest

from probe.caller_trace import Functions, _call_edges, _chained_primary

BASE = 0x140000000
DIR_EXCEPTION = 3


def _rf(begin: int, end: int, unwind: int) -> bytes:
    return begin.to_bytes(4, "little") + end.to_bytes(4, "little") + unwind.to_bytes(4, "little")


class StubImage:
    """Just the surfaces `caller_trace` reads: bytes by VA, base, sections, directories."""

    def __init__(self, size: int = 0x4000):
        self.image_base = BASE
        self.is64 = True
        self._mem = bytearray(size)
        self.directories = [None] * 16
        self.sections: list = []

    def put(self, rva: int, data: bytes) -> None:
        self._mem[rva:rva + len(data)] = data

    def image_bytes(self, va: int, n: int):
        rva = va - self.image_base
        if rva < 0 or rva + n > len(self._mem):
            return None
        return bytes(self._mem[rva:rva + n])


class StubSection:
    def __init__(self, va: int, size: int, executable: bool = True):
        self.va = va
        self.vsize = size
        self.rsize = size
        self.executable = executable


def build_image() -> StubImage:
    """Three records: 0x1000 primary, 0x1100 a chunk chaining to it, 0x1200 primary."""
    img = StubImage()
    img.directories[DIR_EXCEPTION] = (0x2000, 3 * 12)
    img.put(0x2000, _rf(0x1000, 0x1080, 0x3000)
            + _rf(0x1100, 0x1180, 0x3010)
            + _rf(0x1200, 0x1280, 0x3020))
    # 0x3000: Flags = 0 -> a primary, nothing to follow.
    img.put(0x3000, bytes([0x01, 0x00, 0x00, 0x00]))
    # 0x3010: Flags = UNW_FLAG_CHAININFO (0x4 << 3), CountOfCodes = 0, so the
    # chained RUNTIME_FUNCTION sits at 0x3010 + 4, already 4-byte aligned.
    img.put(0x3010, bytes([0x01 | (0x4 << 3), 0x00, 0x00, 0x00]))
    img.put(0x3014, _rf(0x1000, 0x1080, 0x3000))
    # 0x3020: another primary.
    img.put(0x3020, bytes([0x01, 0x00, 0x00, 0x00]))
    # Code: call at 0x1000 -> 0x1200 ; jmp at 0x1005 -> 0x1100
    img.put(0x1000, bytes.fromhex("e8fb010000"))       # call rel32 -> 0x1200
    img.put(0x1005, bytes.fromhex("e9f6000000"))       # jmp  rel32 -> 0x1100
    img.sections = [StubSection(BASE + 0x1000, 0x400)]
    return img


class ChainInfoTests(unittest.TestCase):
    def test_chunk_resolves_to_its_primary(self):
        img = build_image()
        funcs = Functions(img)
        self.assertEqual(len(funcs.records), 3)
        self.assertEqual(funcs.primary(BASE + 0x1100), BASE + 0x1000)

    def test_primary_keeps_its_own_start(self):
        """The fold must not swallow records that have no CHAININFO."""
        img = build_image()
        funcs = Functions(img)
        self.assertEqual(funcs.primary(BASE + 0x1000), BASE + 0x1000)
        self.assertEqual(funcs.primary(BASE + 0x1200), BASE + 0x1200)

    def test_containing_finds_the_record(self):
        img = build_image()
        funcs = Functions(img)
        self.assertEqual(funcs.containing(BASE + 0x1040), (BASE + 0x1000, BASE + 0x1080))
        self.assertIsNone(funcs.containing(BASE + 0x2000))

    def test_chained_primary_is_none_without_the_flag(self):
        img = build_image()
        self.assertIsNone(_chained_primary(img, 0x3000))
        self.assertEqual(_chained_primary(img, 0x3010), BASE + 0x1000)


class CallEdgeTests(unittest.TestCase):
    def test_jmp_is_an_edge(self):
        """A tail call leaves no `call` site; dropping `jmp` loses the helper."""
        img = build_image()
        edges = _call_edges(img)
        self.assertIn(BASE + 0x1100, edges)
        self.assertEqual(edges[BASE + 0x1100], [BASE + 0x1005])

    def test_call_is_an_edge(self):
        img = build_image()
        edges = _call_edges(img)
        self.assertIn(BASE + 0x1200, edges)
        self.assertEqual(edges[BASE + 0x1200], [BASE + 0x1000])

    def test_non_executable_sections_are_skipped(self):
        img = build_image()
        img.sections = [StubSection(BASE + 0x1000, 0x400, executable=False)]
        self.assertEqual(_call_edges(img), {})


if __name__ == "__main__":
    unittest.main()
