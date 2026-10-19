"""Mnemonic census over real `.tvm0` code.

The engine only needs semantics for what the VM actually executes. This measures
that set instead of guessing it: decode linearly from each VM entry trampoline
target and count mnemonics, then report the modelled/boxed split the lifter
would produce.

Usage: python -m probe.mnemonic_census <module> [<module> ...]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from capstone import CS_ARCH_X86, CS_MODE_64, Cs  # noqa: E402

from core.peimg import PeImage  # noqa: E402


def entries(img: PeImage, vm_section: str = ".tvm0") -> list[tuple[int, int]]:
    vm = img.section_by_name(vm_section)
    if vm is None:
        return []
    out: list[tuple[int, int]] = []
    for sec in img.sections:
        if sec.name == vm_section or not sec.executable:
            continue
        body = img.read_va(sec.va, sec.rsize)
        if not body:
            continue
        i = 0
        n = len(body)
        while i + 5 <= n:
            if body[i] != 0xE9:
                i += 1
                continue
            rel = int.from_bytes(body[i + 1 : i + 5], "little", signed=True)
            tgt = sec.va + i + 5 + rel
            if vm.va <= tgt < vm.end:
                pad = 0
                while i + 5 + pad < n and body[i + 5 + pad] == 0xCC:
                    pad += 1
                out.append((sec.va + i, tgt))
                i += 5 + max(pad, 1)
            else:
                i += 1
    return out


def census(path: str, max_bytes: int = 0x200000) -> None:
    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    if vm is None:
        print(f"{path}: no .tvm0")
        return
    ents = entries(img)
    roots = sorted({t for _, t in ents})
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = False

    counts: collections.Counter[str] = collections.Counter()
    decoded = 0
    for root in roots:
        body = img.image_bytes(root, min(max_bytes, vm.end - root))
        if not body:
            continue
        for insn in md.disasm(body, root):
            counts[insn.mnemonic] += 1
            decoded += 1
        if decoded > 4_000_000:
            break

    print(f"=== {path.split(chr(92))[-1]} : .tvm0 {vm.end - vm.va:#x} bytes, "
          f"{len(ents)} trampolines, {len(roots)} distinct roots, {decoded} decoded ===")
    for mn, c in counts.most_common(60):
        print(f"  {mn:<12} {c:>9}")
    print(f"  distinct mnemonics: {len(counts)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        census(p)
