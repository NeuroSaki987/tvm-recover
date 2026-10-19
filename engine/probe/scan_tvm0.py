"""Which binaries in a directory tree carry a `.tvm0` section, and how big is it?

The project's conclusions are all measured on one module, and one module cannot falsify a
rule — it can only fail to exercise it. The honest generalisation test is to run the same
checks on *different* ACE builds, so the first question is which of them contain virtualized
code at all.

`PeImage` decides that; nothing here interprets anything. A candidate is reported when an
executable section's name is `.tvm0` (whose name the protector may pad or vary slightly), and
the root count comes from `core.devirt.module_roots`, i.e. the same trampoline discovery the
product uses.

Usage: python -m probe.scan_tvm0 <directory> [--all] [--limit N]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

from core.peimg import PeImage  # noqa: E402

CANDIDATE_SUFFIXES = {".exe", ".dll", ".sys3", ".sys2", ".sys", ".sysa", ".sysa2"}


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    show_all = "--all" in sys.argv
    limit = 0
    for x in sys.argv[1:]:
        if x.startswith("--limit="):
            limit = int(x.split("=", 1)[1], 0)
    root = Path(args[0])
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix.lower() in CANDIDATE_SUFFIXES)
    print(f"{len(files)} PE-ish file(s) under {root}")
    print()

    hits, skipped = [], 0
    for path in files:
        try:
            img = PeImage(str(path))
        except Exception as exc:                     # not a PE, or unreadable
            skipped += 1
            if show_all:
                print(f"  --   {path.name:<32} not a PE ({type(exc).__name__})")
            continue
        vm = img.section_by_name(".tvm0")
        if vm is None:
            if show_all:
                names = ",".join(s.name for s in img.sections)
                print(f"  --   {path.name:<32} no .tvm0   sections {names[:70]}")
            continue
        try:
            from core.devirt import module_roots
            roots = sorted(set(module_roots(img)))
        except Exception as exc:
            roots = []
            print(f"  ??   {path.name:<32} .tvm0 at {vm.va:#x} but roots failed: {exc}")
        hits.append((path, vm, roots))

    print()
    print(f"{'file':<52} {'.tvm0 va':>12} {'size':>9} {'roots':>6}  sha256")
    print("-" * 114)
    for path, vm, roots in hits:
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        rel = str(path.relative_to(root))
        print(f"{rel:<52} {vm.va:>#12x} {vm.end - vm.va:>9,} {len(roots):>6}  {digest}")
        for r in roots[:limit or len(roots)]:
            print(f"{'':<52}   root {r:#x}")
    print()
    print(f"{len(hits)} file(s) with a .tvm0, {skipped} skipped as not-a-PE, "
          f"{len(files) - skipped - len(hits)} PE(s) without one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
