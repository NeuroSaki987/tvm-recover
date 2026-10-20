"""Build the **write-devirt payload**: the recovered functions as position-independent machine code.

This module turns the verified LLVM IR into one blob of bytes that `llvm/peinject.py` appends to a real
PE as a read+execute section named `.neko`. Four things make the blob what it needs to be.

**1. Sealed memory, not the analysis addresses.** The recovered functions address guest memory at
absolute addresses from the analysis run (`0x7ffffffefda0`, and computed pointers above the user-space
limit). Executing that inside another process would fault, or corrupt whatever is mapped there.
`mem_mode='sealed'` therefore routes every access through `tvm_ld<bits>`/`tvm_st<bits>` and a private
byte map (`runtime/tvm_sealed.c`) whose semantics are `interp.Mem` with `default=seed_free` exactly,
so the injected code can afterwards be compared byte for byte against the interpreter. The address
stays an immediate operand, so the code still says which address the recovery named.

**2. All mutable state is the caller's.** `tvm_state_t` lives in the caller's stack frame, which is
what lets `.neko` be R+X (`0x60000020`) with no writable global anywhere in it. The builder *checks*
that instead of assuming it: a linked image with a writable section, an import, or a base relocation is
refused, because each would contradict the claim the section characteristics make.

**3. No relocations -- measured, not hoped.** References inside the payload are RIP-relative, calls are
`rel32`, and the original entry point's address is computed from two RVAs the injector patches into
read-only data (`base = &entry - entry_rva`). So the blob is correct wherever the host is loaded,
including in a host with no `.reloc` directory at all. `build()` refuses a payload whose linked image
contains a single DIR64 fixup; `peedit.Patch.add_relocations` can still write fixups, which the tests
exercise on a payload that does have one, so the machinery is tested rather than merely present.

**4. The refusals are carried, not dropped.** A root whose blocks the listing does not settle cannot be
composed (`whole.WholeRefusal`), so it is left out and named in the manifest with its reason. The
payload header records how many functions it holds; the evidence document lists what was left out.
A function silently missing from a "devirtualized" binary would be a lie by omission.

Run:  python llvm/payload.py [--listings=DIR] [--out=DIR] [--fuel=N] [--opt=0|1|2|3] [--mem=sealed]
                             [--split=N] [--pdb=FILE]
      (default `--opt=0`, chosen for readability -- see `build()` for the measured trade-off; `--split`
      cuts only the functions that do not fit a decompiler, and `--pdb` writes the listing's names into
      a separate PDB from the *same* link, so the file describes the shipped section bytes)
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME = HERE / 'runtime'
BUILD = HERE / 'build'

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'toolchain_spike'))

import guestir as G  # noqa: E402
import pdbedit  # noqa: E402
import peedit  # noqa: E402
import toolchain  # noqa: E402
import whole as W  # noqa: E402
from lower_llvm import HazardError, _type_hint, _undef_names, hazard_lint  # noqa: E402

DEFAULT_LISTINGS = ROOT / 'out' / 'hard-cases-listings-final'


def _env(name: str, fallback: str = '') -> str:
    """A configuration value from `main/.env`, then the environment, then `fallback`."""
    return _BOOT.get(name) or os.environ.get(name) or fallback


def _booted() -> dict:
    """Load `main/.env` once, at import, so every module-level default below sees it.

    `main/bootenv.py` is one directory up and `main/` is not on `sys.path` for a `llvm/` script, so
    locate it by walking up rather than requiring PYTHONPATH for a convenience feature.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'bootenv.py').is_file():
            text = str(parent)
            if text not in sys.path:
                sys.path.insert(0, text)
            try:
                from bootenv import boot
                return boot(here)
            except Exception:
                return {}
    return {}


_BOOT = _booted()


def _env_int(name: str, fallback: int) -> int:
    raw = _env(name)
    try:
        return int(raw, 0) if raw else fallback
    except ValueError:
        return fallback


def _tool(name: str) -> str:
    """`link.exe` / `cl.exe`, preferring `TVM_MSVC_BIN` from `main/.env` over whatever is on PATH.

    The MSVC toolset directory is the one setting that most often differs between machines and is
    the most annoying to export in every shell, so it is configurable rather than assumed to be on
    PATH. A name with no configured directory passes through unchanged and is resolved by PATH.
    """
    bin_dir = _env('TVM_MSVC_BIN')
    if bin_dir:
        candidate = Path(bin_dir) / name
        if candidate.is_file():
            return str(candidate)
    return name


#: Name and characteristics of the appended code section. `CNT_CODE` $|$ `MEM_EXECUTE` $|$
#: `MEM_READ`, i.e. exactly "code, executable, readable" and not writable: the payload holds no
#: mutable byte, so a writable section would be a lie about the artifact.
SECTION_NAME = '.neko'
SECTION_CHARS = 0x60000020
SECTION_NAME_PADDED = SECTION_NAME.encode('ascii').ljust(8, b'\0')

#: Sealed byte-map capacity and its log2. One slot per guest *byte* written.
SLOTS = 8192
SLOT_BITS = 13

#: Fuel per recovered function -- the same budget the closure measurement used (`diff_whole.py`), so
#: the agreement recorded there carries over instead of being re-tuned until the payload fits.
FUEL = 2000

#: Statements per emitted machine function. **This is the answer to "IDA refuses the function as too
#: big".** The largest recovered block in this corpus is 36 073 statements, and no decompiler takes a
#: function that size; cutting at 1000 statements gives pieces of roughly 1 500 instructions (a few
#: KiB of code), 278 pieces over the whole corpus, and the cost is one call per piece plus a 136-byte
#: spill (at most 17 values cross any cut). `0` disables the cut and reproduces the one-function shape.
DEFAULT_SPLIT = 1000

#: A recovered function with no more statements than this is emitted as **one machine function**, with
#: its real branches and labels; above it, `--split` cuts it into pieces. 7000 statements is ~7 700
#: instructions and ~23 KiB at `-O0` (measured: 1.10 instructions per statement), which is under both
#: lines a decompiler draws -- 5000 instructions where Hex-Rays starts to struggle, and IDA's 64 KiB
#: auto-analysis limit -- with room to spare. In this corpus 9 of the 13 functions come in under it and
#: the remaining 4 (14 888 to 43 636 statements) are the ones that get chunked.
MAX_WHOLE_STATEMENTS = 7000

#: **How many recovered functions the payload can hold.** The state keeps one result per function, and
#: the report is a fixed struct, so the cap is a layout constant -- `runtime/tvm_sealed.h`'s
#: `TVM_NFUNC_MAX` must equal it and `tvm_state_layout_ok()` checks the consequence at runtime.
#: It was 64, which is what a 14-function user-mode module needs; a kernel driver has **144** virtualized
#: roots (`ACE-CORE.sys3`), and the build refused the corpus at 68 functions rather than truncating.
#: 256 costs 1 536 bytes of state and keeps every root of every driver in this distribution.
#: **512 is the DeltaForce base image's requirement:** 351 well-formed VM roots, of which 265 compose,
#: so 256 rejected the corpus with `265 functions exceeds TVM_NFUNC_MAX`. The cost is linear in this
#: constant only -- `result[]` -- so 512 adds 2 048 bytes to a state block already dominated by the
#: 8 192-slot byte map, and `STATE_BYTES` is derived from it rather than written out.
NFUNC_MAX = 512

#: The state layout, mirrored from `runtime/tvm_sealed.h`. `tvm_sealed.c` static-asserts every one of
#: these with `offsetof`, so a compiler that pads differently fails the build rather than letting the
#: harness read a wrong offset. **Derived from `NFUNC_MAX`, not written out**: the numbers used to be
#: literals, and raising the function cap then meant recomputing four offsets by hand.
_RESULT = 56
_CTX_BYTES = 136                      # 17 x u64, the register file
_REPORT_BYTES = 64
OFF = {
    'magic': 0, 'slots': 8, 'used': 12, 'overflow': 16, 'loads': 24, 'stores': 32,
    'steps': 40, 'calls': 48, 'result': _RESULT,
    'ctx': _RESULT + 8 * NFUNC_MAX,
    'report': _RESULT + 8 * NFUNC_MAX + _CTX_BYTES,
    'keys': _RESULT + 8 * NFUNC_MAX + _CTX_BYTES + _REPORT_BYTES,
}
OFF['vals'] = OFF['keys'] + 8 * SLOTS
OFF['live'] = OFF['keys'] + 9 * SLOTS
STATE_BYTES = OFF['live'] + SLOTS
assert STATE_BYTES % 16 == 0, STATE_BYTES

#: Symbols published in the header **in this order**; the header stores only their offsets, so the
#: order is part of the format and `build()` asserts the count has not changed silently.
API_SYMBOLS = (
    'tvm_payload_entry', 'tvm_payload_main', 'tvm_state_reset', 'tvm_report_fill',
    'tvm_state_bytes', 'tvm_state_slot_bits', 'tvm_seed_byte', 'tvm_state_layout_ok',
    'tvm_ld8', 'tvm_ld16', 'tvm_ld32', 'tvm_ld64', 'tvm_st8', 'tvm_st16', 'tvm_st32', 'tvm_st64',
    'tvm_undef_flag', 'tvm_call_reg', 'tvm_call_flags', 'tvm_call_ret', 'tvm_native_call',
    'tvm_udiv', 'tvm_urem', 'tvm_entry_rva', 'tvm_oep_rva', 'tvm_state_rva', 'tvm_witness',
    'tvm_witness_count',
    'tvm_ctx_seed',
)
NAPI = len(API_SYMBOLS)

HEADER_MAGIC = b'NKPH'
TRAILER_MAGIC = b'NKPT'
#: **Bumped when the field layout changed.** Version 1 packed the codegen level into the ninth `I`
#: field, which landed at offset 32 -- on top of the `fuel` quadword. Reading it back returned the fuel
#: budget (2000) instead of the level, which is exactly the kind of plausible-looking number this
#: project's tests are supposed to catch, and `test_header_round_trip` now does.
HEADER_VERSION = 4
#: Q-group and table offsets, chosen so that no field overlaps another.
HEADER_I_FIELDS = 10          # magic + version, header_bytes, nfunc, slots, slot_bits, n_witness,
                              # witness_value, opt, mem_mode, reserved
HEADER_Q_BASE = 48            # after 44 bytes of fixed fields, 8-byte aligned
MEM_MODE_CODE = {'sealed': 0, 'direct': 1}
MEM_MODE_NAME = {v: k for k, v in MEM_MODE_CODE.items()}
#: 32-bit tag 'NEKO' as a little-endian u32, so a reader can confirm it is looking at our state.
STATE_MAGIC = 0x4F4B454E
REPORT_MAGIC = 0x31504B4E

#: The witness assignment the payload runs with. The per-block differential could enumerate or sample
#: witnesses because it drove a harness; the injected payload has no harness, so the choice is made
#: once, here, and written into the header.
WITNESS_VALUE = 0

#: Fixed part of the section header: the scalar fields, then the state-offset table, then the API
#: symbol table. Function records and the free-text note follow.
OFF_BASE = HEADER_Q_BASE + 4 * 8
API_BASE = OFF_BASE + 8 * len(OFF)
HEADER_FIXED = API_BASE + 8 * NAPI
FUNC_RECORD = 64   # offset, root, blocks, statements, pieces, name[24]


class BuildRefusal(RuntimeError):
    """The payload cannot be built, and the reason names the measurement that refused it."""


def _ctx_seed() -> dict[str, int]:
    """The register file every payload run starts from.

    Taken from `diff_whole._ctx_seed` -- the initial state the closure measurement used -- *by
    calling it*, not by re-deriving it. If the two ever diverged, the payload's comparison against
    the interpreter would compare two readings of a different experiment and still look exact.
    """
    import diff_whole
    return diff_whole._ctx_seed()


# -------------------------------------------------------------------------------- IR assembly

def emit_module(paths: list[Path], fuel: int = FUEL,
                stack_probe: str = 'chkstk', dbg: bool = False,
                split: int = 0, cf_protection: bool = False) -> dict:
    """Every composable function of the given listings as one LLVM module.

    `stack_probe` decides whether the emitted functions are allowed to call `__chkstk`.

    **The payload says `chkstk`, unlike every other artifact in this project.** `whole.py`'s default
    attribute (`"stack-probe-size"="4294967295"`) tells LLVM that no frame this emitter can produce
    needs a stack probe, which is true for a per-block or per-function *test* DLL -- and the module is
    linked `/NODEFAULTLIB`, so a probe call would be an unresolved symbol. The payload links
    `runtime/tvm_chkstk.asm`, so it has a `__chkstk`, and it therefore does not need to make that
    claim at all. It matters at `-O0`: the frame then holds a stack slot per SSA value, so a
    fifty-thousand-statement block can exceed a page, and "no probe needed" would be a false statement
    that lets the first store step over the guard page. Frame size per level is printed by
    `payload.build` (`frame_bytes`) and recomputable by rebuilding at each `--opt` level.

    `dbg` adds the debug records that carry the listing's own variable names (see
    `attach_debug_metadata`); it is what `--pdb` turns on, and it changes nothing in the machine code.

    `split` cuts every recovered function into machine functions of at most that many statements (see
    `whole.emit_chunked`). It is the answer to "IDA refuses the function as too big": the largest
    recovered block in this corpus is **36 073 statements**, which no decompiler will take, and with
    `split` the same block becomes a chain of small functions that a dispatcher (generated into
    `tvm_gen.c`) walks. `0` keeps the one-function-per-recovered-function shape.
    """
    attrs = '' if stack_probe == 'chkstk' else ' "stack-probe-size"="4294967295"'
    parsed = [(p, G.parse_listing(p.read_text(encoding='utf-8', errors='replace'), p))
              for p in paths]
    table: list[str] = []
    for _p, fn in parsed:
        for b in fn.blocks:
            for st in b.stmts:
                for name in _undef_names(st):
                    if name not in table:
                        table.append(name)
    table.sort()
    bodies: list[str] = []
    manifests: list[dict] = []
    refusals: list[dict] = []
    uses: set[str] = set()
    for p, fn in parsed:
        try:
            total = sum(len(b.stmts) for b in fn.blocks)
            # **A function that fits is emitted whole; only the ones that do not are cut.**
            #
            # `--split` alone applied to every function turned 149 recovered blocks into 149 machine
            # functions plus 68 extra cuts -- 217 functions, each one a block of a control-flow graph
            # with its live values handed over through a spill array, which is faithful and tiring to
            # read. A function of a few thousand statements compiles to ~1.1 instructions per statement
            # (measured, `llvm/scratch_census.py`), so it stays well inside both limits a decompiler
            # has: under 5000 instructions and far under IDA's 64 KiB auto-analysis size. Those are
            # emitted as one function with real `br`/labels, and only the four functions that exceed
            # `MAX_WHOLE_STATEMENTS` are chunked.
            if split and total > MAX_WHOLE_STATEMENTS:
                lines, man, u = W.emit_chunked(fn, split, table, 'sealed', attrs, dbg=dbg)
            else:
                lines, man, u = W.whole_function_text(fn, table, 'sealed', attrs, dbg=dbg)
        except (W.WholeRefusal, HazardError) as exc:
            # **A refusal is per function, never per build.** `WholeRefusal` was the only exception
            # caught here, and `HazardError` -- which the emitter raises for a value it cannot
            # transcribe -- escaped and aborted the whole payload. Measured on `ACE-CORE.sys3`: one
            # unrecognised value spelling (`box7625#edx`, an unmodelled instruction's result) killed a
            # build of 15 completed functions. A function that cannot be composed is a named hole;
            # forty that can be are not the thing to throw away.
            refusals.append({'listing': p.name, 'root': f'{fn.root:#x}', 'reason': str(exc)})
            continue
        bodies.append('\n'.join(lines))
        manifests.append(man)
        uses |= u
    if not bodies:
        raise BuildRefusal('no function could be composed; every listing refused')
    if len(bodies) > NFUNC_MAX:
        raise BuildRefusal(f'{len(bodies)} functions exceeds TVM_NFUNC_MAX ({NFUNC_MAX}); raise '
                           '`payload.NFUNC_MAX` and `TVM_NFUNC_MAX` in runtime/tvm_sealed.h together')
    head = [
        '; write-devirt payload: every composable recovered function of one module, as one module.',
        ';',
        '; HAZARD POSTURE: as elsewhere -- no nsw/nuw/exact/inbounds/noalias/nonnull/',
        '; dereferenceable/noundef, no undef, no poison, `align 1` on every access, shifts masked,',
        '; division via a call. Guest memory goes through the sealed accessors, so no instruction',
        '; here dereferences an address the analysis invented.',
        '',
        'target triple = "x86_64-pc-windows-msvc"',
        '',
    ]
    for i, name in enumerate(table):
        head.append(f'; witness {i} = {name}')
    text = '\n'.join(head + _type_hint(uses) + [''] + bodies) + '\n'
    if dbg:
        text = attach_debug_metadata(text, manifests, cf_protection=cf_protection)
    elif cf_protection:
        # **`endbr64` at every function entry, from a module flag.** CET/IBT on a kernel thread makes
        # every indirect branch target have to start with `endbr64`, and the loader's call into our
        # entry point is an indirect branch. `clang -fcf-protection` sets exactly these flags; measured
        # on llvmlite/LLVM 22.1, `cf-protection-branch` turns the first four bytes of a function into
        # `f3 0f 1e fa` and costs 4 bytes per function (`llvm/cf_protection_probe.py`).
        text += ('\n!llvm.module.flags = !{!900, !901}\n'
                 '!900 = !{i32 4, !"cf-protection-branch", i32 1}\n'
                 '!901 = !{i32 4, !"cf-protection-return", i32 1}\n')
    return {'text': text, 'table': table, 'manifests': manifests, 'refusals': refusals,
            'uses': sorted(uses), 'fuel': fuel}


#: How each width is described to CodeView. Unsigned, because the listing's widths are bit-vectors and
#: calling them `signed` would be a claim the recovery does not make.
DBG_TYPES = ((1, 'bool', 8), (8, 'u8', 8), (16, 'u16', 16), (32, 'u32', 32), (64, 'u64', 64))


def attach_debug_metadata(text: str, manifests: list[dict], cf_protection: bool = False) -> str:
    """Append the debug information that carries the listing's names, and resolve the placeholders.

    The output is only ever consumed by `--pdb`, through a *second* link whose only product is the
    PDB, so nothing here reaches the image. Four measured facts decide the shape:

    1. **llvmlite's codegen needs the `!"CodeView"` module flag.** Without it `CodeViewDebug` emits an
       `S_COMPILE3` record and then nothing: the object's `.debug$S` stays at 64 bytes and no local
       variable reaches it, whatever shape the `DISubprogram`/`DILocalVariable` metadata takes. With the flag the same module produces
       476 bytes carrying the names. Clang's `-gcodeview` adds exactly this flag.
    2. **Each named value needs its own `llvm.dbg.value`**: a `DILocalVariable` with no `dbg.value` is
       dropped (measured by naming two values and declaring one).
    3. **A real `!DILocation` on ordinary instructions is required too** -- see `whole.py`.
    4. Node ids cannot be known while bodies are emitted, so placeholders are substituted here, and a
       placeholder left behind is deliberately invalid LLVM syntax so the parser fails loudly.
    """
    # The CET flag's node id is allocated from the same counter as everything else. A *fixed* id
    # collides with a generated `DILocalVariable` as soon as the corpus has enough named values --
    # measured on the driver build: `error: Metadata id is already used` at `!701` with 68 functions.
    lines: list[str] = ['', '; ---- debug information: the listing\'s own names ----',
                        '!llvm.dbg.cu = !{!0}',
                        ('!llvm.module.flags = !{!1, !2, !3, __CFID__}' if cf_protection
                         else '!llvm.module.flags = !{!1, !2, !3}'),
                        '!1 = !{i32 2, !"Dwarf Version", i32 4}',
                        '!2 = !{i32 2, !"Debug Info Version", i32 3}',
                        # Without this flag LLVM emits the compile unit and then drops every function
                        # and local record. It is the whole difference between a PDB with names and a
                        # PDB with none.
                        '!3 = !{i32 1, !"CodeView", i32 1}',
                        '!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !4, '
                        'producer: "write-devirt (llvm/payload.py)", isOptimized: false, '
                        'runtimeVersion: 0, emissionKind: FullDebug)',
                        '!4 = !DIFile(filename: "payload.ll", directory: "tvm-devirt")']
    type_id: dict[int, str] = {}
    nxt = 5
    for width, name, bits in DBG_TYPES:
        type_id[width] = f'!{nxt}'
        lines.append(f'!{nxt} = !DIBasicType(name: "{name}", size: {bits}, '
                     'encoding: DW_ATE_unsigned)')
        nxt += 1
    if cf_protection:
        # `endbr64` at every entry: the node id comes from the same counter as the type nodes, so it
        # cannot collide with a `DILocalVariable` allocated later (a fixed `!700` did).
        lines.append(f'!{nxt} = !{{i32 4, !"cf-protection-branch", i32 1}}')
        lines = [line.replace('__CFID__', f'!{nxt}') for line in lines]
        nxt += 1
    # **The register file, as a type, because a decompiler reads the signature.** Every statement of
    # a recovered function writes `ctx + <offset>`, and without a type that is
    # `*(undefined8 *)(param_1 + 0x20)` -- measured, 33 228 such dereferences in one decompile of this
    # payload. With `tvm_ctx_t` as the first parameter's type a decompiler shows `param_1->t_rsp`, which
    # is both shorter and *true*: the listing's register names are what those slots hold.
    member_id: dict[str, str] = {}
    members = []
    for reg, off in sorted(G.CTX_LAYOUT.items(), key=lambda kv: kv[1]):
        vid = f'!{nxt}'
        nxt += 1
        lines.append(f'{vid} = !DIDerivedType(tag: DW_TAG_member, name: "t_{reg}", '
                     f'baseType: {type_id[64]}, size: 64, offset: {off * 8})')
        members.append(vid)
        member_id[reg] = vid
    ctx_struct = f'!{nxt}'
    nxt += 1
    ctx_bits = max(G.CTX_LAYOUT.values()) + 8
    lines.append(f'{ctx_struct} = !DICompositeType(tag: DW_TAG_structure_type, name: "tvm_ctx_t", '
                 f'size: {ctx_bits * 8}, file: !4, elements: !{{{", ".join(members)}}})')
    ctx_ptr = f'!{nxt}'
    nxt += 1
    lines.append(f'{ctx_ptr} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: {ctx_struct}, '
                 'size: 64)')
    # `(tvm_ctx_t*, u64, u64) -> u64`: **the first type in a `DISubroutineType` is the return type**,
    # so the context pointer goes second. Getting that wrong is not a syntax error: putting it first
    # makes every function return `tvm_ctx_t *` while its parameters stay untyped, which a decompiler
    # shows as `tvm_ctx_t * __cdecl tvm_14014b40e(ulong64 param_1, ...)` and a body full of
    # `return (tvm_ctx_t *)0x2;` -- measured, before this was corrected.
    sub_type = f'!{nxt}'
    lines.append(f'{sub_type} = !DISubroutineType(types: !{{{type_id[64]}, {ctx_ptr}, '
                 f'{type_id[64]}, {type_id[64]}}})')
    nxt += 1

    mapping: dict[str, str] = {}
    # **One `DISubprogram` per emitted machine function.** LLVM rejects a subprogram attached to more
    # than one function -- `DISubprogram attached to more than one function` -- and then discards the
    # module's *whole* debug information, which is the difference between a PDB with 69 500 names and
    # one with none (measured: 118 784 bytes of empty PDB, no symbol records at all). So the key is
    # the emitted function's own key: the root for a whole-function build, root plus piece index under
    # `--split` -- which also means a decompiler shows `tvm_14014b40e_p3`, exactly the linker symbol.
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for m in manifests:
        for entry in m.get('dbg_names', []):
            key = entry.get('key') or m.get('dbg_key') or f'{m["root"]:x}'
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(entry)
    # **Every emitted function needs a node, even one with nothing named inside it.** The emitter
    # writes `!dbg !SPFILE_<key>` on *every* function's definition, and a piece whose statements happen
    # to define no named value contributed no entry above -- so the placeholder was left unresolved and
    # `attach_debug_metadata` refused the build ("debug placeholder !SPFILE_145b97b10_b7_p0 was emitted
    # but never defined"). The keys are known: `whole.emit_chunked` names them per piece.
    for m in manifests:
        keys = ([p['function'][len('tvm_'):] for p in m.get('pieces') or [] if p.get('function')]
                or [m.get('dbg_key') or f'{m["root"]:x}'])
        for key in keys:
            if key not in groups:
                groups[key] = []
                order.append(key)
    for key in order:
        sp, loc = f'!{nxt}', f'!{nxt + 1}'
        nxt += 2
        mapping[f'!SPFILE_{key}'] = sp
        mapping[f'!LOCFILE_{key}'] = loc
        var_ids: list[str] = []
        var_lines: list[str] = []
        for i, entry in enumerate(groups[key]):
            vid = f'!{nxt}'
            nxt += 1
            mapping[f'!VARFILE_{entry["pid"]}'] = vid
            var_ids.append(vid)
            # A name has to be safe inside a metadata string; the listing's names are already
            # [A-Za-z0-9_], but the quote is escaped in case a future one is not.
            safe = entry['name'].replace('\\', '\\\\').replace('"', '\\"')
            var_lines.append(f'{vid} = !DILocalVariable(name: "{safe}", scope: {sp}, file: !4, '
                             f'line: {i + 1}, type: {type_id.get(entry["width"], type_id[64])})')
        lines.append(
            f'{sp} = distinct !DISubprogram(name: "tvm_{key}", scope: !0, file: !4, '
            f'line: 1, type: {sub_type}, scopeLine: 1, unit: !0, '
            f'spFlags: DISPFlagDefinition, retainedNodes: !{{{", ".join(var_ids)}}})')
        lines.append(f'{loc} = !DILocation(line: 1, column: 1, scope: {sp})')
        lines.extend(var_lines)

    # One regex pass, not `str.replace` per key: substituting 123 553 placeholders one key at a time
    # over a 22 MB module is quadratic and took 495 s (measured) instead of 9 s.
    placeholder = re.compile(r'!(?:VARFILE|LOCFILE|SPFILE)_[0-9A-Za-z_]+')

    def _sub(match: 're.Match[str]') -> str:
        key = match.group(0)
        try:
            return mapping[key]
        except KeyError:
            raise BuildRefusal(f'debug placeholder {key} was emitted but never defined: the PDB '
                               'would carry a name that never resolves') from None

    out = placeholder.sub(_sub, text)
    leftover = placeholder.findall(out)
    if leftover:
        raise BuildRefusal(f'{len(leftover)} debug placeholder(s) not substituted, first '
                           f'{leftover[0]!r}')
    return out + '\n' + '\n'.join(lines) + '\n'


# -------------------------------------------------------------------------- generated C sources

def write_gen_header(path: Path, manifests: list[dict], table: list[str]) -> None:
    """`tvm_gen.h`: the limits, the state offsets, and the recovered functions' prototypes."""
    lines = [
        '/* GENERATED by llvm/payload.py -- do not edit. */',
        '#ifndef TVM_GEN_H',
        '#define TVM_GEN_H',
        '',
        f'#define TVM_SLOTS {SLOTS}',
        f'#define TVM_SLOT_BITS {SLOT_BITS}',
        f'#define TVM_NFUNC {len(manifests)}',
        f'#define TVM_FUEL {FUEL}ULL',
        f'#define TVM_STATE_BYTES {STATE_BYTES}',
        '',
    ]
    for key, off in OFF.items():
        lines.append(f'#define TVM_OFF_{key.upper()} {off}')
    lines += [
        '',
        '#include "tvm_sealed.h"',
        '',
        'void tvm_payload_main(tvm_state_t* s);',
        'void tvm_state_reset(tvm_state_t* s);',
        'void tvm_state_clear_mem(tvm_state_t* s);',
        'u64 tvm_state_bytes(void);',
        'u64 tvm_state_slot_bits(void);',
        'int tvm_seed_byte(u64 a);',
        'void tvm_report_fill(tvm_report_t* r, u32 nfunc, u64 fuel, u64 total_steps, u32 overflow,',
        '                     u64 loads, u64 stores);',
        '',
        '/* Analysis ABI: 00=normal steps, 10=terminal|steps, 11=~steps on fuel-out. */',
    ]
    for m in manifests:
        lines.append(f'i64 {m["function"]}(void* ctx, tvm_state_t* state, i64 fuel);')
        if m.get('split'):
            # A chunked function is a dispatcher over pieces; the pieces are what a decompiler sees
            # as separate functions, and the root is what `tvm_payload_main` calls.
            lines.append(f'/* {m["function"]}: {m["n_pieces"]} piece(s), one per '
                         f'{m.get("split")} statements */')
            for p in m['pieces']:
                lines.append(f'i64 {p["function"]}(void* ctx, tvm_state_t* state, i64* spill);')
    lines += [
        '',
        '/* Read-only tables referenced from the emitted code. */',
        'const int tvm_witness[];',
        'const unsigned tvm_witness_count;',
        'extern const unsigned char tvm_ctx_seed[TVM_NFUNC][TVM_CTX_BYTES];',
        '',
        'typedef char tvm_gen_nfunc_fits[TVM_NFUNC <= TVM_NFUNC_MAX ? 1 : -1];',
        '',
        '#endif /* TVM_GEN_H */',
        '',
    ]
    path.write_text('\n'.join(lines), encoding='ascii')


def _dispatcher_source(m: dict, index: int) -> list[str]:
    """The C that turns a chunked recovered function's pieces back into one callable function.

    Three things are deliberate:

    **The dispatch is a branch chain, not a jump table.** A table of function pointers, or of absolute
    addresses, would put DIR64 fixups in the payload image -- which `check_payload_image` refuses,
    because the whole position-independence argument rests on there being none. A balanced `if/else`
    over the piece index is code, has no data references at all, and costs about seven branches for
    the largest function in this corpus.

    **Fuel and step accounting stay exactly where they were.** A piece that starts a recovered block
    checks the fuel and a piece that ends one counts the step, so the observable step count and the
    point of fuel exhaustion are identical to the unchunked emission -- which is what lets the
    differential test compare the two builds' transcripts at all.

    **The spill lives on the dispatcher's stack.** The pieces hand their live values through
    `spill[]`, sized by the largest live set the plan measured, so `tvm_state_t` does not grow and the
    sealed accessors are untouched.
    """
    root = m['root']
    tag = f'{root:x}'
    pieces = m['pieces']
    n = len(pieces)
    spill = max(1, int(m.get('max_live_out', 0)))
    out = ['', f'/* ---- root {root:#x}: {n} piece(s), at most {m.get("split")} statement(s) each '
           f'----', ' * The recovered block boundaries are visible here: `start` marks the piece that',
           ' * begins a recovered block and `end` the piece that finishes one. */',
           f'#define TVM_NP_{tag} {n}',
           f'#define TVM_SPILL_{tag} {spill}',
           f'static const unsigned char tvm_{tag}_start_tbl[TVM_NP_{tag}] = {{'
           + ', '.join('1' if p['start'] == 0 else '0' for p in pieces) + '};',
           f'static const unsigned char tvm_{tag}_end_tbl[TVM_NP_{tag}] = {{'
           + ', '.join('1' if p['last'] else '0' for p in pieces) + '};']
    for p in pieces:
        out.append(f'i64 {p["function"]}(void* ctx, tvm_state_t* s, i64* spill);')
    out.append(f'i64 tvm_{tag}_dispatch(i64 pc, void* ctx, tvm_state_t* s, u64* spill) {{')
    # Balanced binary search over the piece index: no table, so no relocation.
    def emit_search(lo: int, hi: int, indent: str) -> None:
        if lo == hi:
            out.append(f'{indent}return {pieces[lo]["function"]}(ctx, s, (i64*)spill);')
            return
        mid = (lo + hi) // 2
        out.append(f'{indent}if (pc <= {mid}) {{')
        emit_search(lo, mid, indent + '    ')
        out.append(f'{indent}}} else {{')
        emit_search(mid + 1, hi, indent + '    ')
        out.append(f'{indent}}}')

    emit_search(0, n - 1, '    ')
    out.append('}')
    out += [
        '',
        f'i64 tvm_{tag}(void* ctx, tvm_state_t* s, i64 fuel) {{',
        f'    u64 spill[TVM_SPILL_{tag}];',
        '    i64 pc = 0;',
        '    i64 steps = 0;',
        '    for (;;) {',
        f'        if (tvm_{tag}_start_tbl[pc]) {{',
        '            if (fuel == 0) {',
        '                return ~steps;      /* the same encoding whole.py uses */',
        '            }',
        '            fuel--;',
        '        }',
        f'        {{ i64 nxt = tvm_{tag}_dispatch(pc, ctx, s, spill);',
        f'          if (tvm_{tag}_end_tbl[pc]) {{',
        '              steps++;',
        '          }',
        f'          if (nxt == TVM_NP_{tag}) {{',
        '              return steps;',
        '          }',
        f'          if (nxt == TVM_NP_{tag} + 1) {{',
        '              return (i64)(0x8000000000000000ULL | (u64)steps);',
        '          }',
        '          pc = nxt; }',
        '    }',
        '}',
        '',
    ]
    return out


def write_gen_source(path: Path, manifests: list[dict], table: list[str]) -> None:
    """`tvm_gen.c`: the witness table, the seeded contexts, and the run the stub performs."""
    seed = _ctx_seed()
    ctx_rows: list[list[int]] = []
    for _m in manifests:
        row = bytearray(136)  # TVM_CTX_BYTES
        for reg, off in G.CTX_LAYOUT.items():
            v = seed[reg] & 0xFFFFFFFFFFFFFFFF
            row[off:off + 8] = v.to_bytes(8, 'little')
        ctx_rows.append(list(row))
    out = [
        '/* GENERATED by llvm/payload.py -- do not edit. */',
        '#include "tvm_gen.h"',
        '#include "tvm_sealed.h"',
        '',
        '/* Filled in by llvm/peinject.py: the host RVA of the entry point, of the original entry',
        ' * point, and (in the kernel shape) of the state block. RVAs, not VAs, so no base relocation',
        ' * is needed; `extern` in tvm_entry.c so the compiler has to load them instead of folding the',
        ' * zero they are built with. */',
        'const u64 tvm_entry_rva = 0;',
        'const u64 tvm_oep_rva = 0;',
        'const u64 tvm_state_rva = 0;',
        '',
        f'/* {len(table)} witness(es), assignment recorded in the payload header: every value = '
        f'{WITNESS_VALUE}. */',
        f'const unsigned tvm_witness_count = {len(table)}u;',
    ]
    if table:
        vals = ', '.join(str(WITNESS_VALUE) for _ in table)
        out.append(f'const int tvm_witness[] = {{ {vals} }};')
    else:
        out.append('const int tvm_witness[] = { 0 };')
    out += [
        '',
        'int tvm_undef_flag(int which) {',
        '    return (which >= 0 && (unsigned)which < tvm_witness_count) ? tvm_witness[which] : 0;',
        '}',
        '',
        '/* The seeded register file per function. Exported (not `static`) so the linker map gives it an',
        ' * RVA: the differential harness then reads the seed *out of the payload* instead of',
        ' * recomputing it, so the two readings cannot disagree about the initial state. */',
        'const unsigned char tvm_ctx_seed[TVM_NFUNC][TVM_CTX_BYTES] = {',
    ]
    for row in ctx_rows:
        body = ', '.join(f'{b}' for b in row)
        out.append(f'    {{ {body} }},')
    out += [
        '};',
        '',
    ]
    # Chunked payloads: per recovered function, the piece graph that a decompiler can actually chew.
    # Every chunked function gets a wrapper, including the ones that happen to fit in one piece, so
    # `tvm_<root>` exists under the same name in both shapes.
    for i, m in enumerate(manifests):
        if not m.get('split'):
            continue
        out.extend(_dispatcher_source(m, i))
    out += [
        '/* The run the injected entry point performs. One function at a time, with the sealed map',
        ' * cleared first, so each function is the same independent experiment the differential',
        ' * harness performs by hand: same seed register file, same witness assignment, same fuel,',
        ' * same initial memory. That is what makes the payload\'s report comparable to a Python run',
        ' * instead of merely plausible. */',
        'void tvm_payload_main(tvm_state_t* s) {',
        '    u64 steps;',
        '    int j;',
        '    tvm_state_reset(s);',
    ]
    for i, m in enumerate(manifests):
        out.append(f'    /* {i}: root {m["root"]:#x}, {len(m["blocks"])} block(s), '
                   f'{m["statements"]} statement(s) */')
        out.append('    {')
        out.append('        /* `volatile`, because at -O1 MSVC turns this 136-byte copy into a call to')
        out.append('         * `memcpy` and the payload has no CRT (LNK2019). Measured at all three')
        out.append('         * levels by llvm/payload_readability.py. */')
        out.append('        volatile unsigned char* dst = s->ctx;')
        out.append('        const unsigned char* src = tvm_ctx_seed[%d];' % i)
        out.append('        for (j = 0; j < TVM_CTX_BYTES; j++) {')
        out.append('            dst[j] = src[j];')
        out.append('        }')
        out.append('    }')
        out.append('    tvm_state_clear_mem(s);')
        out.append(f'    steps = (u64){m["function"]}(s->ctx, s, (i64)TVM_FUEL);')
        out.append(f'    s->result[{i}] = steps;')
        out.append('    s->total_steps += (steps >> 62) == 3 ? ~steps : '
                   '(steps & 0x3FFFFFFFFFFFFFFFULL);')
        out.append('    s->calls++;')
    out += [
        '    tvm_report_fill(&s->report, TVM_NFUNC, TVM_FUEL, s->total_steps, s->overflow,',
        '                    s->loads, s->stores);',
        '}',
        '',
    ]
    path.write_text('\n'.join(out), encoding='ascii')


# ------------------------------------------------------------------------------- compilation

def _ascii(text: str) -> str:
    """Console output from a non-UTF-8 toolchain: this project's artifacts stay pure ASCII."""
    return text.encode('ascii', 'replace').decode('ascii')


def _run(cmd: list[str], what: str, cwd: Path | None = None) -> str:
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                       capture_output=True, text=True, errors='replace')
    if p.returncode != 0:
        raise BuildRefusal(_ascii(
            f'{what} failed ({p.returncode}):\n{" ".join(str(c) for c in cmd)}\n'
            f'{p.stdout}\n{p.stderr}'))
    return p.stdout


def _cl(work: Path, src: Path, obj: Path, defines: tuple[str, ...] = (),
        opt: int = 0) -> str:
    cmd = [_tool('cl.exe'), '/c', '/nologo', f'/O{opt}' if opt else '/Od', '/GS-', '/Zl', '/W4',
           '/WX-', f'/I{work}', f'/I{RUNTIME}', f'/Fo{obj}']
    cmd += [f'/D{d}' for d in defines]
    cmd.append(str(src))
    return _run(cmd, f'cl.exe {src.name}', cwd=work)


def compile_objects(work: Path, ir_text: str, opt: int = 0,
                    state: str = 'stack') -> dict[str, Path]:
    """Every object the payload needs, and the assembly the linker will see.

    `state` decides *where the 82 688-byte state block lives*, and it is a compile-time choice rather
    than a runtime one: `stack` (the user-mode shape -- `tvm_run_payload` owns the frame) or `section`
    (the kernel shape -- the state is an appended writable section and the frame holds nothing). See
    `runtime/tvm_entry.c` for why a driver cannot use the first.
    """
    work.mkdir(parents=True, exist_ok=True)
    objs: dict[str, Path] = {}

    ll = work / 'payload.ll'
    ll.write_text(ir_text, encoding='utf-8')
    obj = work / 'payload.obj'
    toolchain.emit_object(ir_text, str(obj), opt=opt)
    objs['ir'] = obj

    entry_defs: tuple[str, ...] = ()
    if state == 'section':
        entry_defs = ('TVM_STATE_SECTION',)
    elif state != 'stack':
        raise BuildRefusal(f'state mode {state!r} is neither stack nor section')
    specs = [
        ('runtime', RUNTIME / 'tvm_runtime.c', ('TVM_PAYLOAD',)),
        ('sealed', RUNTIME / 'tvm_sealed.c', ()),
        ('gen', work / 'tvm_gen.c', ()),
        ('entry', RUNTIME / 'tvm_entry.c', entry_defs),
    ]
    for key, src, defines in specs:
        obj = work / f'tvm_{key}.obj'
        _cl(work, src, obj, defines, opt)
        objs[key] = obj

    chk = work / 'tvm_chkstk.obj'
    _run(['ml64.exe', '/c', '/nologo', f'/Fo{chk}', str(RUNTIME / 'tvm_chkstk.asm')],
         'ml64.exe tvm_chkstk.asm', cwd=work)
    objs['chkstk'] = chk
    return objs


def link_payload(work: Path, objs: dict[str, Path], opt: int = 0,
                 pdb: Path | None = None, alt_name: str = 'payload.pdb') -> tuple[Path, Path]:
    """Link the payload. The image is only ever used as a source of section bytes.

    **One link, and that is the fix for the PDB defect.** `--pdb` used to run a *second* link of the
    same objects and keep only its PDB, so the payload image and the file the PDB described were two
    different images -- measured: their `.rdata` differed by 196 bytes. A symbol table that describes
    another build cannot be right about this one, so with `/DEBUG` on the *same* link the PDB describes
    exactly the section bytes that `.neko` receives. What the image gains is the debug directory (28
    bytes in `.rdata`, which is copied) and the two CodeView sections (`.debug$S`/`.debug$T`, which are
    not copied -- see `_not_copied`); the *code* and every function RVA are unchanged, which
    `test_codegen_is_identical_with_and_without_the_pdb` asserts against the linker map.

    `/PDBALTPATH` is what makes the PDB findable at all: without it the RSDS record inside the image
    carries this build directory's absolute path, and a decompiler on another machine searches a path
    that does not exist. With it the record says `payload.pdb`, and the file that ships next to the
    image is the one it names.

    `/Brepro` is what makes two builds of the same inputs byte-identical. Without it link.exe puts its
    own wall-clock timestamp into the "repro" debug record it always writes into `.rdata`, and two
    consecutive builds of the payload differ by **exactly one byte** at RVA 0xE200C (measured); with it
    the timestamp is a hash of the inputs and the section is reproducible.
    """
    dll = work / 'payload.dll'
    mp = work / 'payload.map'
    base = [_tool('link.exe'), '/DLL', '/NOENTRY', '/NODEFAULTLIB', '/MACHINE:X64', '/SUBSYSTEM:WINDOWS',
            '/FIXED:NO', '/OPT:NOREF', '/OPT:NOICF', '/INCREMENTAL:NO', '/MANIFEST:NO', '/Brepro']
    objs_all = [str(objs[k]) for k in ('ir', 'runtime', 'sealed', 'gen', 'entry', 'chkstk')]
    dbg_flags: list[str] = []
    if pdb is not None:
        dbg_flags = ['/DEBUG:FULL', '/DEBUGTYPE:CV', f'/PDB:{pdb}', f'/PDBALTPATH:{alt_name}']
    _run(base + dbg_flags + [f'/MAP:{mp}', f'/OUT:{dll}'] + objs_all,
         'link.exe payload', cwd=work)
    if pdb is not None and not pdb.exists():
        raise BuildRefusal(f'link.exe reported success but wrote no PDB at {pdb}; the names '
                           'would be missing without any error to say so')
    return dll, mp


def parse_map(map_path: Path, image_base: int) -> dict[str, int]:
    """Public symbol -> RVA, from the linker's map file."""
    out: dict[str, int] = {}
    text = map_path.read_text(encoding='utf-8', errors='replace')
    pat = re.compile(r'^\s+[0-9a-fA-F]{4}:[0-9a-fA-F]{8}\s+(\S+)\s+([0-9a-fA-F]{16})\s')
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            va = int(m.group(2), 16)
            out.setdefault(m.group(1), va - image_base)
    return out


# ------------------------------------------------------------------------------ image packing

def _is_debug_section(name: str) -> bool:
    """Sections the payload never copies: debug data and the relocation table that goes with it."""
    return name.startswith('.debug') or name == '.reloc'


def _not_copied(name: str, unwind: bool = False) -> bool:
    """Everything `.neko` leaves behind.

    **`.pdata` is dropped in the user-mode build and kept in the kernel one.** For a DLL the payload
    never registers an exception directory (the host's `Dir[3]` still points at its own table), nothing
    in the code reads the unwind info, and the payload cannot throw: every access goes through a sealed
    accessor and there is no SEH handler. The documented consequence: an exception or a stack walk
    *inside* payload frames finds no unwind information.

    **In a kernel driver that consequence is a bugcheck.** The kernel unwinder (`RtlVirtualUnwind`,
    behind every exception, page fault and bugcheck report) needs a `RUNTIME_FUNCTION` entry for *every*
    frame it walks, and code with none does not raise an exception -- it takes the machine down. So a
    kernel build copies `.pdata` into `.neko` (`unwind=True`) and `peinject` merges those records into
    the host's exception directory, rebased. Measured on `ACE-CORE.sys3`: the payload carries **163**
    records for 27 functions.

    **Debug sections are dropped because `--pdb` puts them there and they are not code.** Measured:
    with `--pdb`, link.exe keeps DWARF (`.debug_l .debug_a .debug_i .debug_s .debug_p`) *inside* the
    image, because a PDB has no place for DWARF -- `/DEBUGTYPE:CV` does not change that. The PDB it
    also writes does carry the CodeView records with the listing's names, which is the file an analyst
    loads. So the split is: names in the PDB, nothing debug-related in `.neko`, and the intermediate
    image's debug sections are simply not copied.
    """
    if _is_debug_section(name):
        return True
    if name == '.pdata':
        return not unwind
    return False


def image_blob(pe: peedit.Pe, unwind: bool = False) -> tuple[bytes, int]:
    """The linked DLL's mapped image from its first section onward, as one byte string.

    Sections are concatenated **at their own RVAs**, so every relative reference inside the payload
    keeps the distance the linker computed. That is what makes repacking a copy and not a rewrite.
    Sections that are not copied must therefore come last, which is asserted rather than assumed.
    """
    body = [s for s in pe.sections if not _not_copied(s.name, unwind)]
    skipped = [s for s in pe.sections if _not_copied(s.name, unwind)]
    first = min(s.rva for s in body)
    last = max(s.rva + max(s.vsize, s.rsize) for s in body)
    for s in skipped:
        if s.rva < last:
            raise BuildRefusal(f'{s.name} at rva {s.rva:#x} precedes the last copied section at '
                               f'{last:#x}: skipping it would not preserve the payload layout')
    out = bytearray(last - first)
    for s in body:
        span = max(s.vsize, s.rsize)
        if not span:
            continue
        base = s.rva - first
        out[base:base + span] = pe.image_bytes(s.rva, span)
    return bytes(out), first


def check_payload_image(pe: peedit.Pe, unwind: bool = False) -> dict:
    """Refuse an image that is not what the payload claims to be.

    Three ways the "R+X, no host dependency, position independent" claim can be false, each named with
    the mechanism that would have to handle it instead. A base relocation is tolerated **only** when
    it lands in a section that is not copied, which is exactly the case `--pdb` creates (link.exe
    writes DIR64 fixups into the debug sections it keeps).
    """
    problems = []
    writable = [s.name for s in pe.sections if s.writable and not _not_copied(s.name, unwind)]
    if writable:
        problems.append(f'writable section(s) {writable}: .neko is R+X, so a mutable global would '
                        'fault on the first write instead of being carried by the state block')
    if pe.dirs and len(pe.dirs) > peedit.DIR_IMPORT and pe.dirs[peedit.DIR_IMPORT] != (0, 0):
        problems.append(f'import directory {pe.dirs[peedit.DIR_IMPORT]}: the payload must need '
                        'nothing from the host')
    relocs = pe.dirs[peedit.DIR_BASERELOC] if len(pe.dirs) > peedit.DIR_BASERELOC else (0, 0)
    copied_sites = 0
    if relocs != (0, 0):
        for blk in peedit.read_relocation_blocks(pe):
            page, size = struct.unpack_from('<II', blk, 0)
            body = blk[8:size]
            for k in range(0, len(body) - 1, 2):
                ent = struct.unpack_from('<H', body, k)[0]
                if ent >> 12 == peedit.REL_ABSOLUTE:
                    continue
                site = page + (ent & 0xFFF)
                sec = pe.section_at(site)
                if sec is not None and not _not_copied(sec.name, unwind):
                    copied_sites += 1
        if copied_sites:
            problems.append(f'{copied_sites} base relocation(s) land inside a copied section: the '
                            'payload must be position independent (see peedit.Patch.add_relocations '
                            'for the fixup path)')
    if problems:
        raise BuildRefusal('the linked payload is not position independent:\n  '
                           + '\n  '.join(problems))
    return {'writable_sections': writable, 'imports': 0, 'relocations': copied_sites}


def build_header(manifests: list[dict], table: list[str], symbols: dict[str, int],
                 first_rva: int, span: int, fuel: int, opt: int = 0,
                 mem_mode: str = 'sealed') -> bytes:
    """The self-describing trailer: limits, state offsets, API offsets, and the function table."""
    missing = [n for n in API_SYMBOLS if n not in symbols]
    if missing:
        raise BuildRefusal(f'symbols missing from the linker map: {missing}')

    def off(sym: str) -> int:
        rva = symbols[sym]
        if rva < first_rva:
            raise BuildRefusal(f'{sym} at rva {rva:#x} precedes the first section {first_rva:#x}')
        return rva - first_rva

    note = (f'write-devirt payload: {len(manifests)} recovered function(s), {mem_mode} guest '
            f'memory, {len(table)} witness(es) all = {WITNESS_VALUE}, fuel {fuel}, codegen -O{opt}, '
            f'split {manifests[0].get("split", 0) if manifests else 0} statement(s) per function. '
            f'State is the caller\'s; this section is R+X only.').encode('ascii')
    header_bytes = HEADER_FIXED + FUNC_RECORD * len(manifests) + len(note)
    buf = bytearray(header_bytes)
    struct.pack_into('<4sIIIIIIIIII', buf, 0, HEADER_MAGIC, HEADER_VERSION, header_bytes,
                     len(manifests), SLOTS, SLOT_BITS, len(table), WITNESS_VALUE, opt,
                     MEM_MODE_CODE[mem_mode], 0)
    struct.pack_into('<QQQQ', buf, HEADER_Q_BASE, fuel, STATE_BYTES, off('tvm_payload_entry'), span)
    for i, (_key, value) in enumerate(OFF.items()):
        struct.pack_into('<Q', buf, OFF_BASE + 8 * i, value)
    for i, name in enumerate(API_SYMBOLS):
        struct.pack_into('<Q', buf, API_BASE + 8 * i, off(name))
    p = HEADER_FIXED
    for m in manifests:
        name = m['function'].encode('ascii')[:24].ljust(24, b'\0')
        # `pieces` is how many machine functions the recovered function was cut into: 1 when the
        # payload is not chunked, and the number a decompiler sees as separate functions otherwise.
        struct.pack_into('<QQQQQ24s', buf, p, off(m['function']), m['root'],
                         len(m['blocks']), m['statements'], m.get('n_pieces', 1), name)
        p += FUNC_RECORD
    buf[p:p + len(note)] = note
    return bytes(buf)


def read_header(section: bytes) -> dict:
    """Parse the trailer a `build()` produced. Raises BuildRefusal on a shape it does not know."""
    if len(section) < 32 or section[-16:-12] != TRAILER_MAGIC:
        raise BuildRefusal('no payload trailer: the last 16 bytes are not a NKPT record')
    header_off, header_bytes = struct.unpack_from('<II', section, len(section) - 12)
    blob = section[header_off:header_off + header_bytes]
    if blob[:4] != HEADER_MAGIC:
        raise BuildRefusal(f'header at {header_off:#x} does not start with NKPH')
    (magic, version, hb, nfunc, slots, slot_bits, nwit, wval, opt, mem_mode, _reserved) = \
        struct.unpack_from('<4sIIIIIIIIII', blob, 0)
    fuel, state_bytes, entry_off, span = struct.unpack_from('<QQQQ', blob, HEADER_Q_BASE)
    if version != HEADER_VERSION:
        raise BuildRefusal(f'payload header version {version} != {HEADER_VERSION} (the layout changed '
                           'when the codegen level was added; rebuild the payload)')
    if hb != header_bytes or len(blob) != header_bytes:
        raise BuildRefusal(f'header claims {hb} bytes, trailer says {header_bytes}')
    offs = {}
    for i, key in enumerate(OFF):
        offs[key] = struct.unpack_from('<Q', blob, OFF_BASE + 8 * i)[0]
    api = {}
    for i, name in enumerate(API_SYMBOLS):
        api[name] = struct.unpack_from('<Q', blob, API_BASE + 8 * i)[0]
    funcs = []
    p = HEADER_FIXED
    for _ in range(nfunc):
        f_off, root, blocks, stmts, pieces, nm = struct.unpack_from('<QQQQQ24s', blob, p)
        funcs.append({'offset': f_off, 'root': root, 'blocks': blocks, 'statements': stmts,
                      'pieces': pieces,
                      'name': nm.split(b'\0', 1)[0].decode('ascii')})
        p += FUNC_RECORD
    note = blob[p:].decode('ascii', 'replace')
    return {'version': version, 'nfunc': nfunc, 'slots': slots, 'slot_bits': slot_bits,
            'n_witness': nwit, 'witness_value': wval, 'opt': opt,
            'mem_mode': MEM_MODE_NAME.get(mem_mode, f'#{mem_mode}'), 'fuel': fuel,
            'state_bytes': state_bytes,
            'entry_offset': entry_off, 'image_span': span, 'state_offsets': offs, 'api': api,
            'functions': funcs, 'note': note, 'header_offset': header_off,
            'header_bytes': header_bytes}


# ----------------------------------------------------------------------------------- the build

def build(paths: list[Path], out_dir: Path, opt: int = 0, fuel: int = FUEL,
          slots: int = SLOTS, mem_mode: str = 'sealed', pdb: str | Path | None = None,
          split: int = DEFAULT_SPLIT, state: str = 'stack', unwind: bool = False,
          cf_protection: bool = False) -> dict:
    """Build the payload and write `payload.bin`, `payload.json`, `payload.ll` and the build logs.

    `opt` is the LLVM codegen level and it is a **readability dial, not a correctness dial** -- the
    differential is run at whatever level is built, and every level is expected to agree. What it
    changes is how much of the recovered statement structure survives into the machine code:

    * `-O0` (**default**): 1.50 machine instructions per recovered statement and one fuel-check plus
      one body basic block per recovered block, so a decompiler shows the listing's statements one
      per instruction and the recovered block structure survives. It costs ~3x section size
      (923 952 B against 301 360 B) and frames of up to 11 752 B.
    * `-O2`: 0.47 instructions per statement. LLVM folds the listing's temporaries away, merges and
      threads basic blocks, and a decompiler shows a much smaller function with no statement-by-
      statement correspondence to the listing -- which is why it is no longer the default.
    * `-O1` sits between the two, and on this corpus produces the same instruction count as `-O2`.

    The numbers are from `llvm/payload_readability.py` (log:
    `llvm/measurements/write-devirt-readability.log`), which also runs the differential at each level
    so "more readable" never quietly means "less correct".

    **`pdb` is the only debug-info switch, and it produces a separate file.** `--pdb=<name>` makes the
    emitter attach the listing's own variable names and links with `/DEBUG:FULL /DEBUGTYPE:CV`, whose
    product is the PDB. It is combinable with `--split`: the PDB then describes the piece functions,
    which is the only version of it that describes the shipped code. `/PDBALTPATH` puts a bare
    `payload.pdb` in the image's RSDS record; `payload.json`'s `codeview` block carries that record's
    identity and section offset so `peinject` can point the *host's* debug directory at it and
    `pdbedit.py` can rebase the PDB onto the host's section table.
    """
    global SLOTS, SLOT_BITS, STATE_BYTES, FUEL
    if slots != SLOTS:
        raise BuildRefusal('the sealed-map capacity is baked into runtime/tvm_sealed.c; rebuild '
                           'that file to change it')
    if mem_mode not in ('sealed', 'direct'):
        raise BuildRefusal(f'memory mode {mem_mode!r} is neither sealed nor direct')
    if opt not in (0, 1, 2, 3):
        raise BuildRefusal(f'codegen level -O{opt} is not one of 0, 1, 2, 3')
    # Absolute, because the compiler and linker run with the build directory as their working
    # directory and a relative `/Fo` or `/OUT` would then resolve twice.
    out_dir = Path(out_dir).resolve()
    paths = [Path(p).resolve() for p in paths]
    pdb_path = Path(pdb).resolve() if pdb else None
    FUEL = fuel
    t0 = time.time()
    mod = emit_module(paths, fuel, dbg=pdb_path is not None, split=split,
                     cf_protection=cf_protection)
    problems = hazard_lint(mod['text'])
    if problems:
        raise BuildRefusal('hazard lint on the payload IR: ' + '; '.join(problems[:5]))

    work = (out_dir / 'build') if out_dir else BUILD / 'write-devirt'
    work.mkdir(parents=True, exist_ok=True)
    write_gen_header(work / 'tvm_gen.h', mod['manifests'], mod['table'])
    write_gen_source(work / 'tvm_gen.c', mod['manifests'], mod['table'])

    t1 = time.time()
    objs = compile_objects(work, mod['text'], opt=opt, state=state)
    dll, mp = link_payload(work, objs, opt=opt, pdb=pdb_path)
    t2 = time.time()

    pe = peedit.Pe.from_path(dll)
    checks = check_payload_image(pe, unwind)
    symbols = parse_map(mp, pe.image_base)
    blob, first_rva = image_blob(pe, unwind)
    header = build_header(mod['manifests'], mod['table'], symbols, first_rva, len(blob), fuel,
                          opt=opt, mem_mode=mem_mode)
    # **The unwind records, when the build keeps them.** `.pdata` is a table of
    # `(begin, end, unwind-info)` RVAs; both the functions and the unwind info move with `.neko`, so
    # `peinject` rebases every field by the same delta and merges the result into the host's exception
    # directory. Recording where the table sits inside the section is all this needs to say.
    unw = None
    if unwind:
        pdata = pe.section('.pdata')
        exc_dir = pe.dirs[peedit.DIR_EXCEPTION] if len(pe.dirs) > peedit.DIR_EXCEPTION else (0, 0)
        if pdata is None or exc_dir[1] % 12 or not exc_dir[1]:
            raise BuildRefusal(f'a kernel build needs a `.pdata` table: the image has '
                               f'{pdata.name if pdata else "none"} and an exception directory of '
                               f'{exc_dir[1]} bytes')
        unw = {'section': '.pdata', 'rva_in_section': pdata.rva - first_rva,
               'entries': exc_dir[1] // 12,
               'exception_rva': exc_dir[0], 'bytes': exc_dir[1]}
    # The CodeView record the image points at, in *section* coordinates, so `peinject` can move it
    # into `.neko` and repoint the host's debug directory at it. Without that the image keeps naming
    # the vendor's PDB and a decompiler never looks at ours.
    cv = pdbedit.codeview(pe)
    if pdb_path is not None:
        if cv is None:
            raise BuildRefusal('--pdb was asked for but the linked image carries no CodeView record; '
                               'no loader would find the PDB from the image')
        info = pdbedit.Pdb.from_path(pdb_path).info()
        if info['guid_bytes'] != cv['guid_bytes'] or info['age'] != cv['age']:
            raise BuildRefusal(
                f'the PDB identity does not match the image: image RSDS {cv["guid"]} age {cv["age"]}, '
                f'PDB {info["guid"]} age {info["age"]}. A debugger-facing loader checks these two '
                'against each other and refuses the mismatch')

    section = bytearray(blob)
    section += bytes(peedit.align_up(len(section), 16) - len(section))
    header_off = len(section)
    section += header
    section += bytes(peedit.align_up(len(section), 16) - len(section))
    section += TRAILER_MAGIC + struct.pack('<II', header_off, len(header)) + b'\0\0\0\0'

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'payload.bin').write_bytes(bytes(section))
    (out_dir / 'payload.ll').write_text(mod['text'], encoding='utf-8')
    if pdb is not None:
        # The PDB's symbols are at the *payload image's* RVAs, so the image whose layout it describes
        # travels with it: that pair is the readable artifact (`payload.dll` + `payload.pdb`). The
        # shipping pair is different -- `SGuardSvc64.neko.exe` + a PDB `peinject` rebases onto the
        # host's section table -- and `llvm/pdbedit.py` changes the one into the other.
        import shutil
        shutil.copyfile(dll, out_dir / 'payload.dll')
    parsed = read_header(bytes(section))
    meta = {
        'section': SECTION_NAME, 'section_name_padded': list(SECTION_NAME_PADDED),
        'characteristics': SECTION_CHARS, 'section_bytes': len(section),
        'image_span': len(blob), 'entry_offset': parsed['entry_offset'],
        'header_offset': header_off, 'header_bytes': len(header),
        'functions': parsed['functions'], 'nfunc': parsed['nfunc'],
        'terminals': {f'{m["root"]:#x}': m.get('terminals', []) for m in mod['manifests']
                      if m.get('terminals')},
        'status_encoding': __import__('execution_status').ENCODING,
        'refusals': mod['refusals'], 'slots': SLOTS, 'slot_bits': SLOT_BITS,
        'state_bytes': STATE_BYTES, 'state_offsets': OFF, 'fuel': fuel,
        'witnesses': len(mod['table']), 'witness_value': WITNESS_VALUE,
        'api': parsed['api'], 'links': checks,
        'pdb': str(pdb_path) if pdb_path else None,
        'debug_names': (sum(len(m['dbg_names']) for m in mod['manifests'])
                        if pdb_path else 0),
        'pdb_bytes': (pdb_path.stat().st_size if pdb_path and pdb_path.exists() else 0),
        # Everything `peinject` needs to point the host's debug directory at this CodeView record and
        # to rebase the PDB onto the host's section table. `rva_in_section` is relative to the start of
        # `.neko`; the injector adds the section's RVA and its file offset.
        'codeview': (None if cv is None else {
            'rva_in_section': cv['record_rva'] - first_rva, 'size': cv['size'],
            'dir_rva_in_section': cv['dir_rva'] - first_rva, 'dir_size': cv['dir_size'],
            'guid': cv['guid'], 'age': cv['age'], 'path': cv['path'],
            'entry_index': cv['entry_index']}) if pdb_path else None,
        'image_first_rva': first_rva,
        'unwind': unw,
        'cf_protection': bool(cf_protection),
        'state_section': (STATE_BYTES if state == 'section' else None),
        'dll': {'sections': [{'name': s.name, 'rva': s.rva, 'vsize': s.vsize, 'rsize': s.rsize,
                              'chars': s.chars} for s in pe.sections],
                'image_base': pe.image_base, 'bytes': len(pe.data)},
        'statements': sum(m['statements'] for m in mod['manifests']),
        'blocks': sum(len(m['blocks']) for m in mod['manifests']),
        'split': split,
        'pieces': sum(m.get('n_pieces', 1) for m in mod['manifests']),
        # `--split` only cuts what has to be cut, so the mix is a fact worth recording.
        'chunked_functions': sum(1 for m in mod['manifests'] if m.get('split')),
        'whole_functions': sum(1 for m in mod['manifests'] if not m.get('split')),
        'max_whole_statements': MAX_WHOLE_STATEMENTS,
        'function_pieces': [{'root': m['root'], 'name': m['function'],
                             'statements': m['statements'], 'split': m.get('split', 0),
                             'pieces': m.get('n_pieces', 1)} for m in mod['manifests']],
        'max_piece_statements': max((m.get('max_statements', 0) for m in mod['manifests']),
                                    default=0),
        'max_live_out': max((m.get('max_live_out', 0) for m in mod['manifests']), default=0),
        'ir_bytes': len(mod['text']), 'opt': opt, 'mem_mode': mem_mode,
        'seconds': {'emit': round(t1 - t0, 2), 'compile': round(t2 - t1, 2),
                    'total': round(time.time() - t0, 2)},
    }
    (out_dir / 'payload.json').write_text(json.dumps(meta, indent=2), encoding='ascii')
    return meta


def main(argv: list[str]) -> int:
    # Defaults come from `main/.env` (then the environment), so a local checkout configures its
    # codegen level and memory mode once instead of on every command line. A flag still wins.
    _l = _env('TVM_LISTINGS')
    _o = _env('TVM_OUT')
    listings = Path(_l) if _l else DEFAULT_LISTINGS
    out = (Path(_o) / 'write-devirt') if _o else (BUILD / 'write-devirt')
    opt = _env_int('TVM_PAYLOAD_OPT', 0)
    fuel = _env_int('TVM_PAYLOAD_FUEL', FUEL)
    mem_mode = _env('TVM_PAYLOAD_MEM', 'sealed')
    pdb: str | None = None
    split = _env_int('TVM_PAYLOAD_SPLIT', DEFAULT_SPLIT)
    state = 'stack'
    unwind = False
    cf_protection = False
    args = list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith('--listings='):
            listings = Path(a.split('=', 1)[1])
        elif a.startswith('--out='):
            out = Path(a.split('=', 1)[1])
        elif a.startswith('--opt='):
            opt = int(a.split('=', 1)[1], 0)
        elif a.startswith('--fuel='):
            fuel = int(a.split('=', 1)[1], 0)
        elif a.startswith('--mem='):
            mem_mode = a.split('=', 1)[1].strip().lower()
        elif a == '--cf-protection':
            cf_protection = True
        elif a == '--unwind':
            unwind = True
        elif a == '--kernel':
            state = 'section'
            unwind = True
        elif a.startswith('--split='):
            split = int(a.split('=', 1)[1], 0)
        elif a.startswith('--pdb='):
            pdb = a.split('=', 1)[1]
        elif a == '--pdb':
            # Also accept the space-separated spelling: `--pdb payload.pdb`.
            if i + 1 >= len(args):
                print('--pdb needs a file name')
                return 2
            i += 1
            pdb = args[i]
        elif a in ('-h', '--help'):
            print(__doc__)
            return 2
        i += 1
    paths = sorted(Path(listings).glob('0x*.txt'))
    if not paths:
        print(f'no listings in {listings}')
        return 2
    try:
        meta = build(paths, out, opt=opt, fuel=fuel, mem_mode=mem_mode, pdb=pdb, split=split,
                     state=state, unwind=unwind, cf_protection=cf_protection)
    except BuildRefusal as exc:
        print(f'REFUSED: {exc}')
        return 1
    print(f"{meta['nfunc']} function(s), {meta['blocks']} block(s), {meta['statements']} "
          f"statement(s) -> {meta['section_bytes']} byte {meta['section']} section "
          f"(entry at +{meta['entry_offset']:#x})")
    print(f"  IR {meta['ir_bytes']} B, state {meta['state_bytes']} B, "
          f"witnesses {meta['witnesses']}, slots {meta['slots']}, memory {meta['mem_mode']}, "
          f"codegen -O{meta['opt']}, timing {meta['seconds']}")
    for r in meta['refusals']:
        print(f"  REFUSED {r['root']}: {r['reason']}")
    if meta['pdb']:
        print(f"  pdb {meta['pdb']}: {meta['pdb_bytes']} B, {meta['debug_names']} named value(s), "
              'external -- the image and .neko carry no debug data')
    print(f"  wrote {out / 'payload.bin'} and {out / 'payload.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
