"""Load `main/.env` into `os.environ` for the tools that read configuration from the environment.

The tools take paths as arguments or from the environment; no path is baked into the source. `.env`
is the convenient middle ground for a local checkout: one file, one place to set `TVM_MODULE` /
`TVM_SAMPLES` / `TVM_HOST` / `GHIDRA_HOME` instead of exporting them in every shell.

Two ways in, both idempotent and both a no-op when there is no `.env`:

    # from a tool that already knows where the repository root is
    from envfile import load
    load()

    # from a tool that does not, and would rather not depend on sys.path
    import bootenv          # same directory as this file, copy or import through it
    bootenv.load()

`environment()` returns the parsed mapping without touching `os.environ`, for a caller that wants to
decide. An already-set environment variable always wins, so `TVM_MODULE=x python -m core.devirt`
overrides the file for one command without editing it.

Format: `KEY=VALUE` per line. Blank lines and `#` comments are ignored; a leading `export ` is
allowed; matching single or double quotes around the value are stripped; a leading `~` expands to the
user's home. **`$VAR` is never expanded** -- a path containing `$` stays literal, which is what a
Windows path needs.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["environment", "find", "load", "parse"]

#: The file this module reads, next to the repository root (`main/`).
path = Path(__file__).resolve().parent / ".env"


def parse(text: str) -> dict[str, str]:
    """Parse `.env` text into a mapping. Pure, so it can be tested without the filesystem.

    Tolerates a UTF-8 BOM and CRLF line endings: a `.env` written by a Windows editor almost always
    has both, and a BOM that reaches the first key name (`\\ufeffTVM_STEPS`) would silently make the
    first setting invisible.
    """
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if not key:
            continue
        # An inline `#` comment only counts when it is unquoted and preceded by whitespace, so a
        # path containing `#` survives. Quotes are stripped from both ends of the value.
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            for i, ch in enumerate(value):
                if ch == "#" and i and value[i - 1].isspace():
                    value = value[:i].rstrip()
                    break
        if not value:
            out[key] = ""
            continue
        if value.startswith("~"):
            value = os.path.expanduser(value)
        out[key] = value
    return out


def find(start: Path | str | None = None, *, levels: int = 4) -> Path | None:
    """Walk up from `start` (default: this file) looking for a `.env` next to `envfile.py`.

    `engine/probe/loopcorpus.py` runs with `engine/` as the working directory, so it cannot import
    `main/envfile.py` through `sys.path`; it finds it by walking up instead.
    """
    here = Path(start) if start is not None else Path(__file__).resolve().parent
    here = here.resolve()
    for _ in range(levels + 1):
        candidate = here / ".env"
        if candidate.is_file():
            return candidate
        if here.parent == here:
            break
        here = here.parent
    return None


def environment(dotenv: Path | None = None) -> dict[str, str]:
    """The configured variables, without applying them."""
    target = Path(dotenv) if dotenv is not None else path
    try:
        return parse(target.read_text(encoding="utf-8"))
    except OSError:
        return {}


def load(dotenv: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Apply the file's variables to `os.environ` and return what was applied.

    `override=False` (the default) leaves an existing environment variable alone. An empty value is
    skipped, so blanking a key is how you disable it.
    """
    applied: dict[str, str] = {}
    for key, value in environment(dotenv).items():
        if not value:
            continue
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
