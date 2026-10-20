"""LLVM IR text -> COFF object, for the payload link.

`payload.py` calls exactly one function here:

    toolchain.emit_object(ir_text, obj_path, opt=0)

This module also supplies assemble/VerifyError for check_ir and link_dll for the
differential runner. It deliberately does **not** shell out to `clang.exe` -- the
link step already uses `link.exe`/`cl.exe` for the C runtime pieces, and adding a
second assembler to the chain would mean two independent tools had to agree about
the object format before anything could be linked.

## LLVM 22 and `initialize()`

`llvmlite` 0.50 wraps **LLVM 22.1**, where `llvmlite.binding.initialize()` is
deprecated and *raises* rather than warning:

    RuntimeError: llvmlite.binding.initialize() is deprecated and will be removed.
    LLVM initialization is now handled automatically.

So nothing here initializes LLVM. Calling the old sequence is the first thing a
reader would try and the first thing that would fail, which is why it is written
down instead of left to be rediscovered.

## Why the triple comes from the module

`lower_llvm.py` emits `target triple = "x86_64-pc-windows-msvc"` itself, and the IR
also carries a `"CodeView"` module flag and `cf-protection-branch`, which only mean
anything to that target. So the triple is taken from the module when it has one and
only falls back to the constant when it does not -- emitting this IR for the host
triple (`x86_64-pc-windows-msvc` here, but not on every machine this repo is read
on) would silently change codegen rather than fail.

## Relocation model

`reloc='static'` and `codemodel='small'`, matching what the payload link uses
(`/DLL /NOENTRY /FIXED:NO` into a single `.neko` section). The payload builder
asserts afterwards that the linked image has **no** base relocations
(`payload.build` refuses one), so this is the setting that makes that assertion
hold rather than an arbitrary choice: RIP-relative references and `rel32` calls need
no fixups at all in this model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import llvmlite.binding as llvm

#: The triple `lower_llvm.py` emits. Only used when the module does not declare one.
DEFAULT_TRIPLE = "x86_64-pc-windows-msvc"

#: Accepted `opt` values, mapped to llvmlite's level argument.
_OPT_LEVELS = {0: 0, 1: 1, 2: 2, 3: 3}


class ToolchainError(RuntimeError):
    """Raised when the IR cannot be turned into an object.

    Deliberately not `BuildRefusal`: that type lives in `payload.py` and means "the
    design declined this function". This one means "the tool could not build it",
    and a caller needs to be able to tell those apart when reading a transcript.
    """


class VerifyError(ToolchainError):
    """IR could not be parsed or verified."""


#: Registered once, lazily. LLVM 22 still requires the *target* to be registered
#: explicitly even though it no longer wants `initialize()`, so the two calls that
#: survive are made here rather than at import time (importing a module must not be
#: able to fail because an optional backend is missing).
_targets_ready = False


def _ensure_targets() -> None:
    global _targets_ready
    if _targets_ready:
        return
    # `initialize_native_target` registers the X86 backend this IR needs;
    # `initialize_native_asmprinter` is what `emit_assembly` uses. `initialize()` is
    # deliberately NOT called -- on LLVM 22 it raises instead of warning.
    for call in ("initialize_native_target", "initialize_native_asmprinter"):
        fn = getattr(llvm, call, None)
        if fn is not None:
            try:
                fn()
            except RuntimeError:                          # already initialised
                pass
    _targets_ready = True


def _target_machine(module, opt: int = 0, cpu: str = "", features: str = ""):
    """A target machine for the module's own triple."""
    _ensure_targets()
    triple = (getattr(module, "triple", "") or "").strip() or DEFAULT_TRIPLE
    try:
        target = llvm.Target.from_triple(triple)
    except Exception as exc:                                  # pragma: no cover
        raise ToolchainError(f"no LLVM target for triple {triple!r}: {exc}") from exc
    level = _OPT_LEVELS.get(opt) if isinstance(opt, int) else None
    if level is None:
        raise ToolchainError(f"opt level {opt!r} is not one of 0/1/2/3")
    try:
        return target.create_target_machine(
            cpu=cpu or "x86-64",
            features=features,
            opt=level,
            reloc="static",
            codemodel="small",
        )
    except TypeError:                                         # older llvmlite keyword
        return target.create_target_machine(
            cpu=cpu or "x86-64",
            features=features,
            opt=level,
            reloc="static",
            codemodel="small",
        )


def parse(ir_text: str, verify: bool = True):
    """Parse IR text into a module, verifying it unless asked not to.

    Verification is on by default because the cheapest possible failure here is a
    malformed module, and an unverified module reachable through the optimiser
    produces a *segfault in the LLVM library* rather than a Python exception --
    which is a crash with no traceback and no file name.
    """
    try:
        # A UTF-8 BOM is invisible in every editor and fatal to the LLVM parser
        # (`expected top-level entity` at 1:1). `payload.py` writes without one, but
        # any IR that has been through PowerShell's `Out-File -Encoding utf8` has one,
        # and the failure mode is a parse error pointing at a character nobody can see.
        module = llvm.parse_assembly(ir_text.lstrip("\ufeff"))
    except Exception as exc:
        raise VerifyError(f"parse_assembly failed: {exc}") from exc
    if verify:
        try:
            module.verify()
        except Exception as exc:
            raise VerifyError(f"module verification failed: {exc}") from exc
    return module


def assemble(ir_text: str):
    """Parse and verify IR, as expected by check_ir."""
    return parse(ir_text)


def link_dll(objects, out: str, extra_args=()) -> str:
    """Link x64 COFF objects into a DLL without an entry point or CRT."""
    msvc_bin = os.environ.get("TVM_MSVC_BIN")
    linker = os.path.join(msvc_bin, "link.exe") if msvc_bin else shutil.which("link.exe")
    if not linker:
        raise ToolchainError("link.exe not found; set TVM_MSVC_BIN or add MSVC to PATH")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    command = [linker, "/NOLOGO", "/DLL", "/NOENTRY", "/NODEFAULTLIB",
               "/MACHINE:X64", f"/OUT:{out}", *map(os.fspath, objects), *extra_args]
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    except OSError as exc:
        raise ToolchainError(f"could not run link.exe: {exc}") from exc
    if result.returncode:
        raise ToolchainError(f"link.exe failed ({result.returncode}):\n{result.stdout}{result.stderr}")
    return out


def emit_object(ir_text: str, obj_path: str, opt: int = 0,
                cpu: str = "", features: str = "") -> str:
    """Compile IR text to a COFF object at `obj_path`. Returns the path.

    This is the signature `payload.py` calls. `opt` is the codegen level 0..3 that
    `--opt` advertises; the build itself stays level 0 for readability, and this
    function is what actually makes a different level take effect.
    """
    module = parse(ir_text)
    tm = _target_machine(module, opt=opt, cpu=cpu, features=features)
    try:
        blob = tm.emit_object(module)
    except Exception as exc:
        raise ToolchainError(f"emit_object failed: {exc}") from exc
    if not blob:
        raise ToolchainError("emit_object produced zero bytes")
    parent = os.path.dirname(os.path.abspath(obj_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(obj_path, "wb") as fh:
        fh.write(blob)
    return obj_path


def emit_assembly(ir_text: str, opt: int = 0, cpu: str = "",
                  features: str = "") -> str:
    """The assembly text for the same input -- for reading, not shipping."""
    module = parse(ir_text)
    tm = _target_machine(module, opt=opt, cpu=cpu, features=features)
    return tm.emit_assembly(module)


def target_facts(ir_text: str) -> dict:
    """Report the triple, default CPU, and LLVM's complete data layout string."""
    module = parse(ir_text, verify=False)
    triple = (getattr(module, "triple", "") or "").strip() or DEFAULT_TRIPLE
    tm = _target_machine(module, opt=0)
    return {"triple": triple, "cpu": "x86-64", "data_layout": str(tm.target_data)}


def main(argv: list[str]) -> int:
    """`python toolchain.py <in.ll> <out.obj> [--opt=N]` -- debugging aid only."""
    args = [a for a in argv[1:] if not a.startswith("--")]
    opt = 0
    for a in argv[1:]:
        if a.startswith("--opt="):
            opt = int(a.split("=", 1)[1], 0)
    if len(args) < 2:
        print(__doc__)
        return 2
    src, dst = args[0], args[1]
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    emit_object(text, dst, opt=opt)
    print(f"  wrote          {dst} ({os.path.getsize(dst):,} bytes)")
    try:
        for k, v in target_facts(text).items():
            print(f"  {k:14} {v}")
    except Exception as exc:                                  # diagnostics only
        print(f"  (target facts unavailable: {exc})")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main(sys.argv))
