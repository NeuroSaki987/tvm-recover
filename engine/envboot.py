"""`main/.env` bootstrap for tools that run outside `main/`.

Three lines at the top of a tool — before anything reads `os.environ` — is all it takes:

    import envboot
    envboot.boot(__file__)

`envboot` lives in `engine/` so a probe can `import envboot` with no path juggling: `engine/` is
already the working directory of every engine command, and it is already on `sys.path` when
`python -m probe.x` runs. It in turn locates `main/bootenv.py`, which loads `main/.env`.

Both halves are no-ops when no `.env` exists and never raise, because a fresh clone with no local
configuration is the normal case rather than an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["boot"]

_main_dir: Path | None = None


def main_dir() -> Path | None:
    """`main/`, found by walking up from this file until a directory holds `envfile.py`."""
    global _main_dir
    if _main_dir is None:
        for parent in Path(__file__).resolve().parents:
            if (parent / "envfile.py").is_file():
                _main_dir = parent
                break
        else:
            _main_dir = Path()
    return _main_dir if _main_dir != Path() else None


def boot(caller: str | Path | None = None) -> dict[str, str]:
    """Load `main/.env` into `os.environ`. Returns what was applied (empty when there is no file)."""
    root = main_dir()
    if root is None:
        return {}
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    try:
        import envfile
        return envfile.load()
    except Exception:
        return {}
