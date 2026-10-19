"""Emit every self-contained block and check it against the LLVM parser/verifier, one at a time.

A failure here is a transcription defect, not an LLVM defect: the emitted text is produced by
`lower_llvm` alone, so a parse error means the emitter wrote a malformed instruction. Reporting it
per block, with the offending line, is what makes it fixable rather than a mystery about line 218
of a 20 000-line module.

Run:  python llvm/check_ir.py [--root=HEX] [--block=N] [--all]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'toolchain_spike'))

import guestir as G  # noqa: E402
import lower_llvm as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

def _boot_env() -> None:
    """Load `main/.env` if present. `main/` is not on sys.path for a llvm/ script, so walk up."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'bootenv.py').is_file():
            text = str(parent)
            if text not in sys.path:
                sys.path.insert(0, text)
            try:
                from bootenv import boot
                boot(here)
            except Exception:
                pass
            return


_boot_env()
LISTINGS = ROOT / 'out' / 'memloop-listings'


def main(argv: list[str]) -> int:
    only_root = only_block = None
    everything = '--all' in argv
    listing_dir = LISTINGS
    for a in argv:
        if a.startswith('--root='):
            only_root = a.split('=', 1)[1].lower().replace('0x', '')
        elif a.startswith('--block='):
            only_block = int(a.split('=', 1)[1], 0)
        elif a.startswith('--listings='):
            listing_dir = Path(a.split('=', 1)[1])
    try:
        import toolchain
    except ImportError as exc:
        print(f'llvmlite/toolchain unavailable: {exc}')
        return 2
    ok = bad = skipped = 0
    paths = sorted(listing_dir.glob('0x*.txt'))
    if not paths:
        # An empty corpus must not read as a pass. "0 blocks verified, 0 failed" is a green tick over
        # nothing, which is the failure mode this project keeps having to undo.
        print(f'no listings under {listing_dir}\n'
              f'  generate them first:  cd engine && python -m core.devirt <module> --out ../out/listings\n'
              f'  or point at an existing corpus with --listings=DIR')
        return 2
    for p in paths:
        if only_root and only_root not in p.stem.lower():
            continue
        fn = G.parse_listing(p.read_text(encoding='utf-8', errors='replace'), p)
        for b in fn.blocks:
            if only_block is not None and b.index != only_block:
                continue
            if not everything and b.term != 'return' and not (b.term == 'goto'
                                                              and len(b.succs) == 1):
                skipped += 1
                continue
            try:
                text, manifests = L.emit_function(fn, {b.index})
            except L.HazardError as exc:
                print(f'{p.stem} b{b.index}: EMIT REFUSED: {exc}')
                bad += 1
                continue
            problems = L.hazard_lint(text)
            if problems:
                print(f'{p.stem} b{b.index}: HAZARD {problems[:3]}')
                bad += 1
                continue
            try:
                toolchain.assemble(text)
            except toolchain.VerifyError as exc:
                print(f'{p.stem} b{b.index}: {exc}')
                lines = text.splitlines()
                import re as _re
                m = _re.search(r'<string>:(\d+):', str(exc))
                if m:
                    n = int(m.group(1))
                    for i in range(max(1, n - 6), min(len(lines), n + 3) + 1):
                        print(f'    {i:5} {"->" if i == n else "  "} {lines[i - 1][:140]}')
                bad += 1
                continue
            ok += 1
    print()
    print(f'{ok} block(s) assemble and verify, {bad} failed, {skipped} not self-terminating')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
