"""Find and load `main/.env` from a tool that is not running inside `main/`.

`engine/probe/*.py` runs with `engine/` as the working directory, so `main/envfile.py` is not
importable through `sys.path` and requiring `PYTHONPATH` for a convenience feature is the wrong
trade. This walks up from the calling file until it finds `envfile.py`, imports it from there, and
loads the `.env` beside it.

Safe to call more than once and safe to call when there is no `.env` or no `envfile.py` -- a fresh
clone with no configuration is the normal case, not an error.

Usage, at the very top of a tool, before anything reads `os.environ`:

    from bootenv import boot
    boot(__file__)
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["boot"]


def boot(caller: str | Path, *, levels: int = 5) -> dict[str, str]:
    """Load `main/.env` into `os.environ` for the tool at `caller`. Returns what was applied."""
    here = Path(caller).resolve()
    for parent in list(here.parents)[:levels]:
        candidate = parent / "envfile.py"
        if candidate.is_file():
            text = str(parent)
            if text not in sys.path:
                sys.path.insert(0, text)
            try:
                import envfile
                return envfile.load()
            except Exception:
                return {}
    return {}
