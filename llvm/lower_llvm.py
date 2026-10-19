"""Transcribe a recovered block into LLVM IR, and refuse to write anything the recovery did not say.

## The hazard posture, which is the reason this module is shaped the way it is

LLVM is not a verified compiler, and its IR language is a *contract* in which undefined behaviour
licenses arbitrary output. A transpiler that emits machine-level semantics into it therefore has to
assume it will be miscompiled and remove every lever the optimiser has. The rules below are
consequences of that, not style:

* **No `nsw`, `nuw`, `exact`, `inbounds`, `noalias`, `nonnull`, `dereferenceable`, `noundef`.**
  Each one turns a value into `poison` on some input, and poison is a licence to produce anything.
  All arithmetic here is plain wrapping `add`/`sub`/`mul`, which is total -- and is also the correct
  reading, since x86 addition wraps.
* **`align 1` on every guest and context access.** The recovery records widths and addresses, never
  alignment. Claiming `align 8` would be a claim about a buffer the emitter has never seen, and a
  false alignment assumption is the classic UB that survives one optimiser version and crashes on
  the next.
* **No `undef` and no `poison` text anywhere, ever.** A value the recovery does not know becomes a
  call to an external function with a documented contract (`@tvm_undef_flag`). `undef` would let the
  optimiser delete the whole block that reads it; an external call cannot be folded away, and the
  decision about what an architecturally undefined x86 flag "is" gets made once, at a visible
  boundary, instead of being made silently by the optimiser.
* **Shift amounts are masked before use.** `shl i64 %x, %n` is poison for `%n >= 64`. x86 already
  masks the count -- `core/lift64.py::_shift` implements `k & 0x3F` at 64 bits and `k & 0x1F` below
  -- so the mask restores the lifted semantics rather than imposing one.
* **Division goes through a helper.** `udiv` is UB on a zero divisor and the guest would fault;
  `@tvm_udiv` reproduces the fault. There are **zero** division operations in the certified corpus
  (`/` and `%` occur 0 times in 170 078 statements), so this path is written but unexercised -- and
  it is written as a call rather than as an instruction precisely so that it cannot be UB if it is
  ever reached.
* **No floating point, no vectors.** Measured, not assumed: every width in the corpus is 8, 16, 32
  or 64 (`llvm/census.py`; zero statements wider than 64), so the guest's SSE instructions never
  reach this level. There is no `fadd` and no `<2 x i64>` in the output.

`hazard_lint()` re-checks all of it on the finished text, so these are enforced by a test rather
than by the emitter's good intentions.

## What a block becomes

One LLVM function per recovered block:

    define void @tvm_<root>_b<idx>(ptr %ctx)

`%ctx` is the guest register file: 17 eight-byte slots at the offsets in
`llvm/guestir.py::CTX_LAYOUT`. Guest *memory* is not a parameter -- the recovered addresses are
absolute, so the runtime maps that region at the recovered address and the emitted code addresses it
literally. The region the module needs is emitted as `@tvm_guest_lo` / `@tvm_guest_hi`, so a harness
enforces the precondition instead of assuming it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from guestir import CTX_LAYOUT, FLAG_BITS, Block, Function

I1, I8, I16, I32, I64 = 'i1', 'i8', 'i16', 'i32', 'i64'
_TY = {1: I1, 8: I8, 16: I16, 32: I32, 64: I64}

# Every spelling that hands the optimiser a licence this emitter has not earned.
FORBIDDEN = (
    ('nsw', re.compile(r'\bnsw\b')),
    ('nuw', re.compile(r'\bnuw\b')),
    ('exact', re.compile(r'\bexact\b')),
    ('inbounds', re.compile(r'\binbounds\b')),
    ('noalias', re.compile(r'\bnoalias\b')),
    ('nonnull', re.compile(r'\bnonnull\b')),
    ('dereferenceable', re.compile(r'\bdereferenceable\b')),
    ('noundef', re.compile(r'\bnoundef\b')),
    ('aligned-attr', re.compile(r'\baligned\b')),
    ('undef', re.compile(r'\bundef\b')),
    ('poison', re.compile(r'\bpoison\b')),
    ('sdiv', re.compile(r'\bsdiv\b')),
    ('srem', re.compile(r'\bsrem\b')),
    ('udiv-instr', re.compile(r'(?<![@\w])udiv\b(?!\s*[,(])')),
    ('float', re.compile(r'\b(?:fadd|fsub|fmul|fdiv|frem|fneg|fcmp|fpext|fptrunc|half|float|'
                         r'double|fp128|x86_fp80)\b')),
    ('vector', re.compile(r'<\s*\d+\s*x\s*i\d+\s*>')),
    ('byval', re.compile(r'\bbyval\b')),
    ('sret', re.compile(r'\bsret\b')),
    ('readonly', re.compile(r'\breadonly\b')),
    ('readnone', re.compile(r'\breadnone\b')),
    ('inreg', re.compile(r'\binreg\b')),
)


class HazardError(Exception):
    """The emitter would have had to write a construct the posture forbids."""


def _require_parsed(fn: Function) -> None:
    """Do not turn an unsupported side effect into an executable omission."""
    if 'entry_context' in fn.header:
        location = str(fn.path) if fn.path is not None else '<listing>'
        if fn.header.get('entry_context_line'):
            location += f':{fn.header["entry_context_line"]}'
        raise HazardError(f'{location}: caller-specialized entry context requires a guarded '
                          'call-site bridge; unconditional LLVM emission is refused')
    if fn.unparsed:
        location = str(fn.path) if fn.path is not None else '<listing>'
        if fn.unparsed_line_numbers:
            location += f':{fn.unparsed_line_numbers[0]}'
        raise HazardError(f'{location}: {len(fn.unparsed)} unparsed listing line(s); '
                          f'first: {fn.unparsed[0]}')


def hazard_lint(text: str) -> list[str]:
    """Every forbidden construct in `text`, as `line: reason` strings.

    Also checks the two structural rules that make the claims checkable: every `load`/`store` states
    `align 1`, and nothing claims a tighter alignment.
    """
    problems: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split(';', 1)[0]
        for name, pat in FORBIDDEN:
            if pat.search(code):
                problems.append(f'{i}: forbidden {name}: {line.strip()[:150]}')
        if re.search(r'\b(?:load|store)\b', code):
            if not re.search(r'align 1(?!\d)', code):
                problems.append(f'{i}: access without `align 1`: {line.strip()[:150]}')
    return problems


def _bare(operand: str) -> str:
    """`i64 %x` -> `%x`, `i64 5` -> `5`: an operand without its type, for a typed instruction."""
    return operand.split(' ', 1)[1] if ' ' in operand else operand


def _fnv(name: str) -> int:
    """A stable 32-bit identity for a name. Kept for callers that need a name-derived id."""
    h = 0x811C9DC5
    for b in name.encode('utf-8'):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def _san(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.]', '_', name)


class Widths:
    """The width of every temporary, from the `vN:W` spelling the listing uses.

    The listing states a width only on definitions, so an operand's width comes from looking the
    name up; an operand that is not a definition is 64-bit (a context slot, an entry snapshot, a
    post-call register) or one bit (a flag), which `width_of` decides syntactically.
    """

    def __init__(self, block: Block):
        self.declared: dict[str, int] = {}
        for st in block.stmts:
            if st.kind in ('def', 'ctx-load') and st.width:
                self.declared[st.dest] = st.width

    _CMP = ('eq', 'ne', 'ult', 'ule', 'slt', 'sle', 'ugt', 'uge', 'sgt', 'sge')

    def width_of(self, e, hint: int) -> int:
        if not isinstance(e, tuple) or not e or not isinstance(e[0], str):
            return hint
        op = e[0]
        if op == 'const':
            return hint
        if op in ('slot', 'entry', 'reg', 'callreg', 'callret'):
            return 64
        if op == 'temp':
            return self.declared.get(e[1]) or hint
        if op in ('flagbit', 'undef'):
            return 1
        if op == 'witness':
            return 64
        if op == 'bin':
            # **A comparison's declared width is the width of the *result*, and in this engine that
            # is the operand width, not one bit.** `v429:64 = a <u b` is a 64-bit value holding 0 or
            # 1 (`core/expr.py::_BOOL_RESULT_OPS`), so the `icmp` is zero-extended to the declared
            # width instead of being returned as i1. Emitting i1 where the listing says i64 was a
            # hard mismatch, not a style choice.
            return hint if hint > 1 else 1
        if op == 'un':
            return hint
        if op == 'cast':
            return e[4]
        if op == 'sel':
            return hint
        if op in ('load', 'hload'):
            return e[1]
        return hint


_BIN = {'add': 'add', 'sub': 'sub', 'mul': 'mul', 'and': 'and', 'or': 'or', 'xor': 'xor',
        'shl': 'shl', 'lshr': 'lshr', 'ashr': 'ashr'}
_CMP = {'eq': 'eq', 'ne': 'ne', 'ult': 'ult', 'ule': 'ule', 'slt': 'slt', 'sle': 'sle',
        'ugt': 'ugt', 'uge': 'uge', 'sgt': 'sgt', 'sge': 'sge'}

#: `i64 5` -> 5. Only literals the emitter itself wrote (`value('const')` masks to the width, so they
#: are always non-negative) match, and `_bare` has already stripped the type.
_LITERAL = re.compile(r'^-?\d+$')


def _int_const(operand: str, width: int) -> int | None:
    """The value of a literal operand, or None if it is a register."""
    bare = _bare(operand)
    if not _LITERAL.match(bare):
        return None
    return int(bare) & ((1 << width) - 1)


def _fold(op: str, a: int, b: int, width: int) -> int | None:
    """Constant-fold `a op b` at `width` **with the guest's own semantics**, or None.

    Wrapping arithmetic, not `nsw` arithmetic: the interpreter (`guestir.Machine`) computes in Python
    integers masked to the width, so folding has to mask the result too. Division is not folded at
    all -- `tvm_udiv` is a call into the runtime's divide-by-zero policy, and folding the easy case
    would emit code that answers that policy differently from the interpreter.
    """
    m = (1 << width) - 1
    if op == 'add':
        return (a + b) & m
    if op == 'sub':
        return (a - b) & m
    if op == 'mul':
        return (a * b) & m
    if op == 'and':
        return a & b
    if op == 'or':
        return a | b
    if op == 'xor':
        return a ^ b
    return None


def _compare(op: str, a: int, b: int, width: int) -> bool:
    """The signed/unsigned comparison the `icmp` predicate names."""
    if op in ('slt', 'sle', 'sgt', 'sge'):
        sign = 1 << (width - 1)
        a = a - (1 << width) if a & sign else a
        b = b - (1 << width) if b & sign else b
    return {'eq': a == b, 'ne': a != b, 'ult': a < b, 'ule': a <= b, 'slt': a < b, 'sle': a <= b,
            'ugt': a > b, 'uge': a >= b, 'sgt': a > b, 'sge': a >= b}[op]


class Emitter:
    """One block -> one LLVM function. Stateful only within one `emit`."""

    def __init__(self, block: Block, fn: Function, undef_table: list[str] | None = None,
                 attrs: str = '', prefix: str = '', mem_mode: str = 'direct',
                 state_ptr: str = '', dbg: bool = False, dbg_key: str = '',
                 alias: dict[str, tuple[str, int]] | None = None):
        from valueflow import prepare, ValueFlowError
        try:
            if fn.blocks:
                fn = prepare(fn)
                block = next(b for b in fn.blocks if b.index == block.index)
            else:
                from valueflow import resolve_block
                block = resolve_block(block)
        except ValueFlowError as exc:
            raise HazardError(str(exc)) from exc
        self.b = block
        self.fn = fn
        # **How guest memory is addressed.** `direct` is the default and is what every existing
        # artifact was measured on: an absolute guest address becomes `inttoptr i64 <addr>` and the
        # load/store happens at that address, which is only meaningful inside the harness that
        # mapped it. `sealed` routes the same address through `@tvm_ld<bits>`/`@tvm_st<bits>`
        # instead, so the code can be executed inside a *foreign* process (the patched image,
        # `write-devirt`) without touching any address the analysis invented -- see
        # `llvm/runtime/tvm_sealed.c`, which reproduces `interp.Mem`'s seeded model exactly. The
        # address itself is still an immediate operand in the instruction stream, so the emitted
        # code says which address the recovery named.
        if mem_mode not in ('direct', 'sealed'):
            raise HazardError(f'memory mode {mem_mode!r} is neither direct nor sealed')
        self.mem_mode = mem_mode
        # In `sealed` mode every guest access needs the caller-owned state block, so its LLVM name
        # is threaded in rather than assumed: the payload module takes it as a parameter, and the
        # emitter must say `ptr %state` at each call site so LLVM's verifier checks the arity.
        if mem_mode == 'sealed' and not state_ptr:
            raise HazardError('sealed memory needs the state pointer operand')
        self.state_ptr = state_ptr
        # **Debug names, when a PDB is asked for.**
        #
        # A decompiler invents its own numbering (`v134`), which matches nothing: not the listing,
        # not the recovered IR. Every statement destination here already carries the listing's own
        # name (`v429`, `guest_rbx_12`), so with `dbg` the emitter records each definition and emits
        # `llvm.dbg.value` for it. Codegen turns those into CodeView records and the linker into a
        # **separate** PDB (`--pdb`); the image itself carries no debug data -- see
        # `payload.link_payload`, which links twice for exactly that reason.
        self.dbg = dbg
        #: The key the debug placeholders are keyed by. **One `DISubprogram` per emitted machine
        #: function**: LLVM refuses to attach one subprogram to two functions and then drops the
        #: module's whole debug information, so under `--split` this carries the piece index too.
        self.dbg_key = dbg_key or f'{fn.root:x}'
        self.named: list[dict] = []
        # **A listing temporary is a name, not an instruction.**
        #
        # `v429:8 = <expr>` in the listing defines a name for a value the same block already
        # computed. The first version of this emitter made it a real definition by emitting
        # `%v429 = add i64 <value>, 0` -- an identity instruction that LLVM does not fold at `-O0`,
        # so the machine code carried 32 313 `add reg, 0` instructions (measured) and a decompiler
        # rendered them as `v429 = v428 + 0`, which is exactly the "this is a mess" complaint. The
        # name is recorded here instead, and every later mention of `v429` resolves to the operand it
        # was bound to, so the value keeps its name in the PDB and costs nothing in the code.
        #
        # Shared across the blocks of one function (`whole_function_text`) because a listing name is
        # unique per function and a condition or operand can be defined in an earlier block; a fresh
        # map per piece in `whole.emit_chunked`, because a piece is a separate machine function.
        self.alias: dict[str, tuple[str, int]] = {} if alias is None else alias
        #: Which statements of the block this emitter renders. `None` is the whole block; a `range`
        #: selects a *slice*, which is what the chunked emitter in `whole.py` uses to keep every
        #: machine function below a size a decompiler will accept.
        self.stmt_range: range | None = None
        # Namespace for the values this emitter invents.
        #
        # **Necessary once several blocks share one function.** `vN` names come from the listing's own
        # numbering and are already unique across a function, but the emitter's own temporaries
        # (`%a1`, `%l2`, ...) restart at 1 in every block, and so do the prologue's `%s_<reg>` and
        # `%in_<reg>`. In per-block emission that cannot collide because each block is its own module;
        # in whole-function emission it does, and LLVM rejects it with "multiple definition of local
        # value named 's_r12'". Empty by default, so the per-block IR is byte-identical.
        self.prefix = prefix
        # Function attributes for the emitted definition, used to state the *stack* promise.
        #
        # At `O0` LLVM spills every SSA value, so a block with thousands of statements gets a frame
        # past the 4 KiB stack-probe threshold and its object references `__chkstk` -- which does not
        # exist when the module is linked `/NODEFAULTLIB`, so the link fails with LNK2019 (observed
        # on `0x140169e9d` block 5, and only at `O0`). `"stack-probe-size"="4294967295"` says that no
        # frame this emitter can produce needs a probe. That is true of a frame bounded by one block's
        # temporaries, and it is a claim written into the module rather than a flag hidden in the
        # harness. See `emit_function` for why the two `probe-stack` spellings do not work.
        self.attrs = attrs
        # **The module-wide witness table, built before any block is emitted.** An index into it is
        # the identity the runtime keys on, and it has to be the *same* index in every block, so it
        # cannot be derived from a per-block ordering.
        self.undef_table: list[str] = undef_table if undef_table is not None else []
        self.w = Widths(block)
        self.lines: list[str] = []
        self.n = 0
        self.slot_ptr: dict[str, str] = {}
        #: Emitted operand -> `(context slot, the slot's write generation at the load)`. A
        #: `home`/`ctx-write` that stores such an operand back into the slot it came from is a no-op
        #: **only while the generation is unchanged**; see `_slot_origin`.
        self.load_origin: dict[str, tuple[int, int]] = {}
        #: Register -> the prologue's operand for its block-entry value (`%in_rax`), so a store of that
        #: value back can be recognised as a no-op too.
        self.snapshot_operands: dict[str, str] = {}
        #: Slot offset -> True once this block has written it. Only consulted for entry snapshots,
        #: whose value is stale exactly when the slot has been written since.
        self.slot_written: dict[int, bool] = {}
        #: Slot offset -> how many times this block has written it. A value loaded from a slot is
        #: only "what the slot holds" until the next write, and this is what says whether one has
        #: happened.
        self.slot_gen: dict[int, int] = {}
        self.entry_snap: dict[str, str] = {}
        self.needs_slots: set[str] = set()
        self.needs_entry: set[str] = set()
        self.undef_names: list[str] = []
        self.uses: set[str] = set()
        self.guest_lo = 0
        self.guest_hi = 0
        self.guest_addrs: list[int] = []
        self.call_sites: list[int] = []
        self.warnings: list[str] = []

    def fresh(self, tag: str = 't') -> str:
        self.n += 1
        return f'%{self.prefix}{tag}{self.n}'

    def selected_stmts(self) -> list:
        """The statements this emitter renders -- the whole block, or the slice `stmt_range` names.

        A slice is how a recovered block with tens of thousands of statements becomes a *series* of
        machine functions small enough for a decompiler. The slice is only ever cut at a statement
        boundary, and whatever crosses the cut is spilled (see `whole.emit_chunked`).
        """
        if self.stmt_range is None:
            return self.b.stmts
        return self.b.stmts[self.stmt_range.start:self.stmt_range.stop]

    def say(self, line: str) -> None:
        self.lines.append('  ' + line)

    def ref(self, name: str) -> tuple[str, int]:
        """What a recovered name currently denotes: `(bare operand, width)`.

        A name bound by a `def` resolves through `self.alias`; anything else is its own LLVM value,
        which is what the prologue and the chunked emitter's spill-in code define.
        """
        hit = self.alias.get(name)
        if hit is not None:
            return hit
        return f'%{_san(name)}', self.w.declared.get(name, 64)

    def name_value(self, operand: str, friendly: str, width: int) -> None:
        """Attach the listing's own name to a value, for the debug record.

        Metadata node ids are not known while the body is emitted -- they are allocated when the whole
        module is assembled -- so the line carries a placeholder that `payload.attach_debug_metadata`
        replaces. **The placeholder carries the function**, not just the block: `0_0` occurs in every
        function, and a per-function-keyed mapping then resolves one function's `dbg.value` to another
        function's `DILocalVariable`. LLVM's diagnostic names that symptom exactly --
        `mismatched subprogram between #dbg record variable and DILocation` -- and then discards the
        *whole module's* debug information, which is how a PDB ends up with function names and no
        variable names at all. The same failure is recorded as F15 in `PROBLEMS.md`. `self.dbg_key`
        extends the key to the *emitted function*, for the same reason one level down.
        """
        if not self.dbg:
            return
        ty = _TY[width]
        pid = f'{self.dbg_key}_{self.b.index}_{len(self.named)}'
        self.named.append({'pid': pid, 'name': friendly, 'width': width, 'llvm': operand,
                           'key': self.dbg_key})
        self.say(f'call void @llvm.dbg.value(metadata {ty} {operand}, '
                 f'metadata !VARFILE_{pid}, metadata !DIExpression()), '
                 f'!dbg !LOCFILE_{self.dbg_key}')
        self.uses.add('dbg')

    # ------------------------------------------------------------------ prescan
    def prescan(self) -> None:
        """Find every capability, slot and entry snapshot the block needs, before emitting."""
        def walk(e):
            if not isinstance(e, tuple) or not e or not isinstance(e[0], str):
                if isinstance(e, tuple):
                    for y in e:
                        walk(y)
                return
            op = e[0]
            if op in ('slot', 'reg'):
                self.needs_slots.add(e[1])
                self.uses.add('ctx')
            elif op == 'entry':
                self.needs_entry.add(e[1])
                self.uses.add('ctx')
            elif op == 'flagbit':
                self.uses.add('flagbit')
                if e[2] == 'entry':
                    self.needs_entry.add('rflags')
                else:
                    self.uses.add('callreg')
            elif op == 'undef':
                if e[1] not in self.undef_names:
                    self.undef_names.append(e[1])
                self.uses.add('undef')
            elif op == 'witness':
                # The same witness table, read at register width; see `value`.
                if e[1] not in self.undef_names:
                    self.undef_names.append(e[1])
                self.uses.add('undef')
            elif op == 'callreg':
                self.uses.add('callreg')
            elif op in ('load', 'hload'):
                self.uses.add('guest_mem')
                if self.mem_mode == 'sealed':
                    self.uses.add('sealed_mem')
                if op == 'hload':
                    self.uses.add('history')
            elif op == 'call':
                self.uses.add('call')
                if e[3] not in self.call_sites:
                    self.call_sites.append(e[3])
            elif op == 'pack':
                self.uses.add('pack')
            for x in e[1:]:
                if isinstance(x, tuple):
                    walk(x)

        for st in self.selected_stmts():
            if st.kind in ('ctx-write', 'ctx-partial', 'home', 'ctx-load'):
                self.uses.add('ctx')
                if st.extra and st.extra[0] in CTX_LAYOUT:
                    self.needs_slots.add(st.extra[0])
            if st.kind == 'store':
                self.uses.add('guest_mem')
                if self.mem_mode == 'sealed':
                    self.uses.add('sealed_mem')
            if st.kind == 'ctx-partial':
                self.uses.add('insert')
            if isinstance(st.expr, tuple):
                walk(st.expr)

    # ------------------------------------------------------------------ values
    def fit(self, operand: str, aw: int, w: int) -> tuple[str, int]:
        """Coerce an operand to an operation's width, exactly as the arena does.

        **The listing prints `a OP b` and drops the coercion, but the arena applies one.**
        `core/expr.py::Arena._fit` truncates an operand wider than the operation and zero-extends one
        that is narrower, and `Arena.bin`/`Arena.un` call it before building the node -- so
        `v429:8 = <64-bit value> & <8-bit value>` is not a 64-bit operation and not a type error; it
        is `trunc8` of the first operand. Reading the text without re-applying `_fit` therefore gets
        the arithmetic wrong whenever the widths differ, and it is a reconstruction the text alone
        cannot settle -- which is why it is stated here rather than done quietly.
        """
        if aw == w:
            return operand, w
        d = self.fresh('f')
        if aw > w:
            self.say(f'{d} = trunc {_TY[aw]} {_bare(operand)} to {_TY[w]}')
        else:
            self.say(f'{d} = zext {_TY[aw]} {_bare(operand)} to {_TY[w]}')
        return f'{_TY[w]} {d}', w

    def value(self, e, hint: int) -> tuple[str, int]:
        """Emit instructions for `e`; return `(type-qualified operand, width)`."""
        w = self.w.width_of(e, hint)
        if w not in _TY:
            raise HazardError(f'width {w} is not one of {sorted(_TY)}')
        ty = _TY[w]
        op = e[0]
        if op == 'const':
            return f'{ty} {e[1] & ((1 << w) - 1)}', w
        if op == 'temp':
            if e[1] not in self.w.declared and e[1] not in self.alias:
                self.warnings.append(f'operand %{e[1]} has no definition in this block')
            bare, aw = self.ref(e[1])
            return f'{_TY[aw]} {bare}', aw
        if op in ('slot', 'reg'):
            reg = e[1]
            self.uses.add('ctx')
            p = self.slot_ptr.get(reg)
            if p is None:
                raise HazardError(f'no context slot for {reg!r}')
            d = self.fresh('l')
            self.say(f'{d} = load i64, ptr {p}, align 1')
            self.load_origin[d] = (CTX_LAYOUT[reg], self.slot_gen.get(CTX_LAYOUT[reg], 0))
            return f'i64 {d}', 64
        if op == 'entry':
            reg = e[1]
            if reg not in CTX_LAYOUT:
                raise HazardError(f'`in.{reg}` names no context slot')
            return f'i64 {self.entry_snap[reg]}', 64
        if op == 'flagbit':
            return self._flagbit(e[1], e[2])
        if op == 'undef':
            idx = self.undef_table.index(e[1]) if e[1] in self.undef_table else 0
            r = self.fresh('u')
            self.say(f'{r} = call i32 @tvm_undef_flag(i32 {idx})')
            d = self.fresh('ub')
            self.say(f'{d} = trunc i32 {r} to i1')
            return f'i1 {d}', 1
        if op == 'witness':
            # `box<N>#<reg>`, an unmodelled instruction's result: the same witness call, **zero
            # extended to the register width instead of truncated to a bit**. The table and the value
            # are shared with the flag witnesses; only the width differs, and the width is what the
            # interpreter and the payload disagreed about.
            idx = self.undef_table.index(e[1]) if e[1] in self.undef_table else 0
            r = self.fresh('u')
            self.say(f'{r} = call i32 @tvm_undef_flag(i32 {idx})')
            d = self.fresh('uw')
            self.say(f'{d} = zext i32 {r} to i64')
            return f'i64 {d}', 64
        if op == 'callreg':
            if e[1] not in CTX_LAYOUT:
                raise HazardError(f'`callr{e[1]}#{e[2]}` names no register')
            r = self.fresh('q')
            self.say(f'{r} = call i64 @tvm_call_reg(i32 {e[2]}, i32 {CTX_LAYOUT[e[1]]})')
            return f'i64 {r}', 64
        if op == 'callret':
            # The return value of a call this block did not make. An input, exactly like
            # `callrXX#N`: a call to the runtime rather than a guess.
            self.uses.add('callreg')
            r = self.fresh('kr')
            self.say(f'{r} = call i64 @tvm_call_ret(i32 {e[1]})')
            return f'i64 {r}', 64
        if op == 'cast':
            _op, kind, inner, _srcw, dstw = e
            if kind not in ('trunc', 'zext', 'sext'):
                raise HazardError(f'cast {kind!r} has no transcription')
            src, sw = self.value(inner, dstw)
            if sw == dstw:
                # A same-width cast is the identity; emitting `trunc i32 %x to i32` would be noise
                # the optimiser deletes and the lint cannot distinguish from a real conversion.
                return f'{_TY[dstw]} {_bare(src)}', dstw
            d = self.fresh('c')
            if sw > dstw:
                # **`_fit` wins over the printed name.** `Arena.zext64(x)` with a wider `x` first
                # truncates; emitting a widening `zext` here would be an impossible instruction.
                self.say(f'{d} = trunc {_TY[sw]} {_bare(src)} to {_TY[dstw]}')
            elif kind == 'sext':
                self.say(f'{d} = sext {_TY[sw]} {_bare(src)} to {_TY[dstw]}')
            else:
                self.say(f'{d} = zext {_TY[sw]} {_bare(src)} to {_TY[dstw]}')
            return f'{_TY[dstw]} {d}', dstw
        if op == 'un':
            src, sw = self.fit(*self.value(e[2], w), w)
            d = self.fresh('n')
            if e[1] == 'not':
                self.say(f'{d} = xor {_TY[sw]} {_bare(src)}, -1')
            elif e[1] == 'neg':
                self.say(f'{d} = sub {_TY[sw]} 0, {_bare(src)}')
            else:
                raise HazardError(f'unary {e[1]!r} has no transcription')
            return f'{_TY[sw]} {d}', sw
        if op == 'bin':
            return self._bin(e[1], e[2], e[3], hint)
        if op == 'sel':
            c, cw = self.value(e[1], w)
            t, tw = self.value(e[2], w)
            t, tw = self.fit(t, tw, w)
            f, fw = self.value(e[3], w)
            f, fw = self.fit(f, fw, w)
            b = self.fresh('p')
            # `c ? a : b` tests NONZERO (`mach.py`'s branch semantics), so the condition is an
            # explicit comparison: reinterpreting a wider value as i1 would be wrong for any
            # condition that is not already 0 or 1.
            self.say(f'{b} = icmp ne {_TY[cw]} {_bare(c)}, 0')
            d = self.fresh('s')
            self.say(f'{d} = select i1 {b}, {_TY[tw]} {_bare(t)}, {_TY[fw]} {_bare(f)}')
            return f'{_TY[tw]} {d}', tw
        if op in ('load', 'hload'):
            bits = e[1]
            addr, _aw = self.value(e[2], 64)
            if self.mem_mode == 'sealed':
                d = self.fresh('g')
                self.say(f'{d} = call {_TY[bits]} @tvm_ld{bits}(ptr {self.state_ptr}, '
                         f'i64 {_bare(addr)})')
                return f'{_TY[bits]} {d}', bits
            p = self.fresh('p')
            self.say(f'{p} = inttoptr i64 {_bare(addr)} to ptr')
            d = self.fresh('g')
            self.say(f'{d} = load {_TY[bits]}, ptr {p}, align 1')
            return f'{_TY[bits]} {d}', bits
        if op == 'pack':
            pairs, base = e[1], e[2]
            acc = self.fresh('k')
            self.say(f'{acc} = add i64 0, {base}')
            for flag, sub in pairs:
                bit = FLAG_BITS.get(flag)
                if bit is None:
                    raise HazardError(f'pack of unknown flag {flag!r}')
                fv, fw = self.value(sub, 1)
                if fw != 1:
                    raise HazardError(f'pack input {flag} is {fw} bits, expected 1')
                z = self.fresh('k')
                self.say(f'{z} = zext i1 {_bare(fv)} to i64')
                sh = self.fresh('k')
                self.say(f'{sh} = shl i64 {z}, {bit}')
                nx = self.fresh('k')
                self.say(f'{nx} = or i64 {acc}, {sh}')
                acc = nx
            return f'i64 {acc}', 64
        if op == 'insert':
            raise HazardError('insert_bits is a statement-level form, not a value')
        if op == 'call':
            return self._call(e)
        if op == 'opaque_name':
            # A name the parser recognised as a value but could not attribute to a context slot, a
            # flag, a post-call register or a witness. Emitting anything for it would be invention;
            # naming it is what makes the refusal actionable.
            raise HazardError(f'value {e[1]!r} has no transcription')
        raise HazardError(f'value {op!r} has no transcription')

    def _flagbit(self, flag: str, source) -> tuple[str, int]:
        bit = FLAG_BITS.get(flag)
        if bit is None:
            raise HazardError(f'unknown flag {flag!r}')
        if source == 'entry':
            base = self.entry_snap['rflags']
        else:
            _kind, n = source
            base = self.fresh('f')
            self.say(f'{base} = call i64 @tvm_call_flags(i32 {n})')
        sh = self.fresh('b')
        self.say(f'{sh} = lshr i64 {base}, {bit}')
        an = self.fresh('b')
        self.say(f'{an} = and i64 {sh}, 1')
        d = self.fresh('b')
        self.say(f'{d} = trunc i64 {an} to i1')
        return f'i1 {d}', 1

    def _bin(self, op: str, left, right, hint: int) -> tuple[str, int]:
        w = hint if hint in _TY else 64
        a, aw = self.value(left, w)
        a, aw = self.fit(a, aw, w)
        b, bw = self.value(right, w)
        b, bw = self.fit(b, bw, w)
        if aw != w or bw != w:
            raise HazardError(f'{op} operands are {aw} and {bw} bits under a {w}-bit hint')
        ka, kb = _int_const(a, w), _int_const(b, w)
        if op in ('shl', 'lshr', 'ashr'):
            # **Masked, because an unmasked shift is poison at or above the width.** This is also
            # the guest's own rule: `core/lift64.py::_shift` uses `k & 0x3F` at 64 bits and
            # `k & 0x1F` below it, so the mask restores the lifted semantics rather than adding one.
            # When the amount is a literal the mask is arithmetic on a constant, so it is folded
            # here instead of costing an `and` instruction the decompiler then has to read past.
            k = (kb & (w - 1)) if kb is not None else None
            if k == 0:
                return f'{_TY[w]} {_bare(a)}', w
            if k is not None and ka is not None:
                folded = _fold(op, ka, k, w)
                if folded is not None:
                    return f'{_TY[w]} {folded}', w
            m = self.fresh('m') if k is None else repr(k)
            if k is None:
                self.say(f'{m} = and {_TY[w]} {_bare(b)}, {w - 1}')
            d = self.fresh('h')
            self.say(f'{d} = {_BIN[op]} {_TY[w]} {_bare(a)}, {m}')
            return f'{_TY[w]} {d}', w
        if op in _CMP:
            if ka is not None and kb is not None:
                bit = 1 if _compare(op, ka, kb, w) else 0
                # The listing's `vN:W` is the truth about the value's width: a comparison yields a
                # W-bit 0/1, which is what the `icmp` + `zext` pair below produces.
                return (f'{_TY[w]} {bit}', w) if w > 1 else (f'i1 {bit}', 1)
            d = self.fresh('q')
            self.say(f'{d} = icmp {_CMP[op]} {_TY[w]} {_bare(a)}, {_bare(b)}')
            if w > 1:
                z = self.fresh('z')
                self.say(f'{z} = zext i1 {d} to {_TY[w]}')
                return f'{_TY[w]} {z}', w
            return f'i1 {d}', 1
        if op in ('udiv', 'urem'):
            fn = 'tvm_udiv' if op == 'udiv' else 'tvm_urem'
            d = self.fresh('d')
            self.say(f'{d} = call {_TY[w]} @{fn}({_TY[w]} {_bare(a)}, {_TY[w]} {_bare(b)})')
            return f'{_TY[w]} {d}', w
        if op not in _BIN:
            raise HazardError(f'operator {op!r} has no transcription')
        if ka is not None and kb is not None:
            folded = _fold(op, ka, kb, w)
            if folded is not None:
                return f'{_TY[w]} {folded}', w
        d = self.fresh('a')
        self.say(f'{d} = {_BIN[op]} {_TY[w]} {_bare(a)}, {_bare(b)}')
        return f'{_TY[w]} {d}', w

    def _call(self, e) -> tuple[str, int]:
        _op, target, args, site = e
        vals = []
        for _reg, sub in args:
            v, vw = self.value(sub, 64)
            # **An argument has to be fitted to the call's width.** `value()` returns the operand with
            # the width it actually has, and a flag or an undefined witness is *one bit*; handing
            # `i1 %b1_ub2` to a callee that takes `i64` is not a coercion LLVM performs implicitly.
            # Measured on `ACE-CORE.sys3`: `error: '%b1_ub2' defined with type 'i1' but expected 'i64'`
            # on a `tvm_native_call` whose second argument is a witness. The Svc corpus never had a
            # narrow value in argument position, which is why 13 functions did not surface it.
            v, _vw = self.fit(v, vw, 64)
            vals.append(_bare(v))
        while len(vals) < 4:
            vals.append('0')
        d = self.fresh('r')
        # The call is addressed by its **site**, not by an index into a per-module table: the table
        # is built after the blocks are emitted, so a block cannot know its position in it.
        self.say(f'{d} = call i64 @tvm_native_call(i64 {site}, '
                 f'{", ".join("i64 " + v for v in vals[:4])})')
        return f'i64 {d}', 64

    # ------------------------------------------------------------------ statements
    def _slot_origin(self, e) -> tuple[int, bool, int | None] | None:
        """`(slot offset, is it read now, load generation)` for a value that reads the context.

        Four spellings reach a context slot, and the three-way return is what keeps the
        elimination below sound:

        * `('reg', 'rax')`/`('slot', 'rax')` -- a load emitted *now*, so what it returns is what the
          slot holds at this point in the block: storing it back is a no-op unconditionally
          (`is it read now` = True, generation None);
        * `('temp', 'v123')` bound by a *load* -- a value that was read at some **earlier point**.
          Storing it back is a no-op only if nothing has written the slot since, which is what the
          generation is for. Treating this as "read now" was **wrong**, and it was wrong on real
          output: on `0x145be0c7d` a `home rcx = v138` whose value came from an rcx load two writes
          earlier was dropped, the payload's slot sequence then diverged from the interpreter's, and
          the two readings of the same listing produced different values in `rax`
          (`evidence/E-DRIVER-CORE3.md` section 6.1);
        * `('entry', 'rax')` -- the value the slot held at *block entry*, so storing it back is a
          no-op only if nothing has written the slot since (the generation test covers this too, and
          `slot_written` is kept for the report).
        """
        if not isinstance(e, tuple) or not e:
            return None
        if e[0] in ('slot', 'reg'):
            off = CTX_LAYOUT.get(e[1])
            return (off, True, None) if off is not None else None
        if e[0] == 'entry':
            off = CTX_LAYOUT.get(e[1])
            return (off, False, None) if off is not None else None
        if e[0] == 'temp':
            hit = self.alias.get(e[1])
            if hit is None:
                return None
            bare = hit[0]
            if bare in self.load_origin:
                off, gen = self.load_origin[bare]
                return off, True, gen
            for reg, operand in self.snapshot_operands.items():
                if operand == bare:
                    return CTX_LAYOUT[reg], False, None
        return None

    def _still_holds(self, hit: tuple[int, bool, int | None], off: int) -> bool:
        """Is `hit`'s value still what slot `off` contains, so a store back is a no-op?

        * read-now (`('reg', ...)` used as the store's own operand): yes, unconditionally -- the load
          is emitted immediately before the store, so it cannot be stale;
        * bound to an earlier load or to the entry snapshot: only while the slot has not been written
          since, which the generation says exactly.
        """
        if hit[0] != off:
            return False
        if hit[1] and hit[2] is None:
            return True
        if hit[2] is not None:
            return hit[2] == self.slot_gen.get(off, 0)
        return not self.slot_written.get(off)

    def stmt(self, st) -> None:
        if st.kind in ('bare', 'comment', 'entry-note', 'call-block', 'forward-note', 'read-note'):
            return
        if st.kind == 'def':
            v, vw = self.value(st.expr, st.width or 64)
            w = st.width or vw
            if w != vw:
                raise HazardError(f'{st.dest}: declared {w} bits but the value is {vw}')
            # A name, not an instruction: the value is already in an SSA operand (or is a literal),
            # and binding the listing's name to it is what carries the name into the PDB. Emitting
            # `%v = add <value>, 0` instead -- which is what this used to do -- costs a machine
            # instruction per definition at `-O0` and shows up in a decompiler as `v429 = v428 + 0`.
            self.alias[st.dest] = (_bare(v), w)
            self.name_value(_bare(v), st.dest, w)
            return
        if st.kind in ('ctx-write', 'home'):
            _reg, off, _site = st.extra
            if off is None:
                raise HazardError(f'{st.kind} for an unmapped register {st.extra[0]!r}')
            # **`home rbx = rbx` is not an instruction.** The listing states the register file
            # explicitly at block boundaries, so a block that leaves a register alone says so with a
            # self-assignment. Emitting `store (load slot), slot` is faithful and, at `-O0`, real code:
            # measured in a decompiler, 17 lines of every function were
            # `*(undefined8 *)(param_1 + 0x68) = *(undefined8 *)(param_1 + 0x68);` -- and once the
            # context was typed, `param_1->t_rbx = param_1->t_rbx;`.
            #
            # Reading a register and storing it back to the same slot is a no-op **unconditionally**,
            # not only when nothing has written the slot yet: the read happens immediately before the
            # write, so it returns what the write puts there. Nothing the emitter produces can alias
            # the context either -- guest memory goes through the sealed accessors, whose callees take
            # the state block, not the context.
            #
            # The value usually arrives as a *listing temporary* bound by a `ctx-load`
            # (`home rax = v123`, where `v123` was `= rax`), so the check resolves the binding: an
            # earlier version compared only direct register operands, which caught the direct form and
            # left every indirect one in place.
            if st.expr is not None and isinstance(st.expr, tuple) and len(st.expr) >= 2:
                hit = self._slot_origin(st.expr)
                if hit is not None and hit[0] == off and self._still_holds(hit, off):
                    return
            v, vw = self.value(st.expr, 64)
            w = min(64, vw)
            p = self.fresh('w')
            self.say(f'{p} = getelementptr i8, ptr %ctx, i64 {off}')
            self.say(f'store {_TY[w]} {_bare(v)}, ptr {p}, align 1')
            self.slot_written[off] = True
            self.slot_gen[off] = self.slot_gen.get(off, 0) + 1
            return
        if st.kind == 'ctx-load':
            _reg, off, _site = st.extra
            if off is None:
                raise HazardError(f'context load from an unmapped offset in {st.dest}')
            bits = st.width
            p = self.fresh('w')
            self.say(f'{p} = getelementptr i8, ptr %ctx, i64 {off}')
            d = self.fresh('l')
            self.say(f'{d} = load {_TY[bits]}, ptr {p}, align 1')
            self.load_origin[d] = (off, self.slot_gen.get(off, 0))
            self.alias[st.dest] = (d, bits)
            self.name_value(d, st.dest, bits)
            return
        if st.kind == 'ctx-partial':
            _reg, off, _site = st.extra
            if off is None:
                raise HazardError(f'partial write to an unmapped register {st.extra[0]!r}')
            _op, _prev, lo_bits, nb_bits, val = st.expr
            lo_byte, nb_byte = lo_bits[1] // 8, nb_bits[1] // 8
            if nb_byte * 8 != nb_bits[1] or lo_byte * 8 != lo_bits[1]:
                raise HazardError(f'partial write {lo_bits[1]}+{nb_bits[1]} bits is not byte-aligned')
            v, vw = self.value(val, nb_bits[1])
            if vw != nb_bits[1]:
                raise HazardError(f'partial value is {vw} bits, the slice is {nb_bits[1]}')
            p = self.fresh('w')
            self.say(f'{p} = getelementptr i8, ptr %ctx, i64 {off + lo_byte}')
            # **A narrow store, not a merge.** The bytes outside the slice are the slot's own
            # contents, which the machine leaves alone by not writing them; taking the previous value
            # from the propagated state instead would import that state into the program.
            self.say(f'store {_TY[nb_bits[1]]} {_bare(v)}, ptr {p}, align 1')
            self.slot_written[off] = True
            self.slot_gen[off] = self.slot_gen.get(off, 0) + 1
            return
        if st.kind == 'store':
            bits = st.width
            addr, value = st.expr
            a, _aw = self.value(addr, 64)
            v, vw = self.value(value, bits)
            if self.mem_mode == 'sealed':
                self.say(f'call void @tvm_st{bits}(ptr {self.state_ptr}, i64 {_bare(a)}, '
                         f'{_TY[bits]} {_bare(v)})')
                return
            p = self.fresh('p')
            self.say(f'{p} = inttoptr i64 {_bare(a)} to ptr')
            self.say(f'store {_TY[bits]} {_bare(v)}, ptr {p}, align 1')
            return
        raise HazardError(f'statement kind {st.kind!r} has no transcription')

    # ------------------------------------------------------------------ the block
    def emit(self) -> tuple[str, dict]:
        _require_parsed(self.fn)
        if self.b.term in ('noreturn', 'fastfail'):
            raise HazardError('terminal block requires the whole-function analysis status ABI')
        self.prescan()
        name = f'tvm_{self.fn.root:x}_b{self.b.index}'
        # **A forking block returns which successor the machine takes; every other block returns
        # void.** Not a `br` between two labels, because the two successors are separate recovered
        # blocks with their own entries, and joining them here would be whole-function emission,
        # which needs every block of the function to be settled. Returning the index keeps the unit
        # one block while making the branch observable -- and comparable.
        forks = self.b.branches
        i0 = self.b.succ_index(self.b.when0) if forks else -1
        i1 = self.b.succ_index(self.b.when1) if forks else -1
        if forks and (i0 < 0 or i1 < 0):
            forks = False
        ret_ty = 'i32' if forks else 'void'
        prologue = self.build_prologue()
        self.build_body()
        self.body_terminator(forks, i0, i1)
        out = [f'define {ret_ty} @{name}(ptr %ctx){self.attrs} {{', 'entry:'] + prologue \
            + self.lines + ['}']
        manifest = self.manifest(name, ret_ty, forks, i0, i1)
        return '\n'.join(out), manifest

    # ------------------------------------------------------------------ pieces
    #
    # Split out so a whole-function emitter can place the same prologue at the top of *each* block's
    # basic block. That placement is not cosmetic: `in.<reg>` is the register as of the block's entry,
    # and **129 reads of `in.<reg>` in the corpus occur after that same register was written in the
    # same block**, so reading the live slot instead would be wrong. The snapshot has to be taken
    # before the block's first statement, in the block, and nowhere else.
    def build_prologue(self) -> list[str]:
        prologue: list[str] = []
        for reg in sorted(self.needs_slots):
            if reg not in CTX_LAYOUT:
                raise HazardError(f'no context slot for register {reg!r}')
            p = f'%{self.prefix}s_{_san(reg)}'
            self.slot_ptr[reg] = p
            prologue.append(f'  {p} = getelementptr i8, ptr %ctx, i64 {CTX_LAYOUT[reg]}')
        for reg in sorted(self.needs_entry):
            if reg not in CTX_LAYOUT:
                raise HazardError(f'no context slot for `in.{reg}`')
            p = f'%{self.prefix}in_{_san(reg)}'
            self.entry_snap[reg] = p
            self.snapshot_operands[reg] = p
            sp = f'%{self.prefix}snap_{_san(reg)}'
            prologue.append(f'  {sp} = getelementptr i8, ptr %ctx, i64 {CTX_LAYOUT[reg]}')
            prologue.append(f'  {p} = load i64, ptr {sp}, align 1')
        return prologue

    def build_body(self) -> None:
        lo, hi = 1 << 64, 0
        addrs: set[int] = set()
        for st in self.selected_stmts():
            for addr in _abs_addresses(st):
                addrs.add(addr)
                lo = min(lo, addr)
                hi = max(hi, addr + 8)
        self.guest_addrs = sorted(addrs)
        self.guest_lo = 0 if hi == 0 else lo
        self.guest_hi = 0 if hi == 0 else hi
        for st in self.selected_stmts():
            self.stmt(st)

    def cond(self) -> tuple[str, int]:
        """The branch condition: `(bare operand, width)`, resolved through the name bindings.

        The width comes from the listing's own `cond_width` when it states one, otherwise from the
        value the condition names. A condition whose width is not stated is a defect in the listing,
        not something to guess at.

        **A condition can be a flag, and a flag is not an SSA name.** `; branch condition: flag_cf`
        names the architectural carry as an *input*; `guestir` parses that spelling as a witness
        (`('undef', 'flag_cf')`), and `%flag_cf` is a name nothing defines -- measured as
        `error: use of undefined value '%flag_cf'` once a width was supplied for it. So the operand is
        materialised through the same path the body uses: a `tvm_undef_flag` read.
        """
        hit = self.alias.get(self.b.cond)
        if hit is not None:
            bare, aw = hit
            return bare, (self.b.cond_width or aw)
        name = self.b.cond.split(':', 1)[0]
        if re.match(r'^box\d+#\w+$', name):
            # A boxed instruction's result is a *witness*, at register width -- not a flag bit.
            v, vw = self.value(('witness', name), 64)
            return _bare(v), (self.b.cond_width or vw)
        if name.startswith(('flag_', 'init_', 'mul_undefined_', 'imul_undefined_')):
            v, vw = self.value(('undef', name), 1)
            return _bare(v), (self.b.cond_width or vw)
        cw = self.b.cond_width or self.w.declared.get(self.b.cond)
        if cw is None:
            raise HazardError(f'branch condition {self.b.cond!r} has no width')
        return f'%{_san(self.b.cond)}', cw

    def body_terminator(self, forks: bool, i0: int, i1: int) -> None:
        if forks:
            # The condition is a temporary the body already defines, so nothing is recomputed; the
            # test is NONZERO, which is what `mach.py`'s branch semantics are and what an `is_bool`
            # node makes safe to test either way.
            bare, cw = self.cond()
            t = self.fresh('nz')
            self.say(f'{t} = icmp ne {_TY[cw]} {bare}, 0')
            pick = self.fresh('sw')
            self.say(f'{pick} = select i1 {t}, i32 {i1}, i32 {i0}')
            self.say(f'ret i32 {pick}')
        else:
            self.say('ret void')

    def manifest(self, name: str, ret_ty: str, forks: bool, i0: int, i1: int) -> dict:
        return {
            'function': name, 'root': self.fn.root, 'block': self.b.index, 'label': self.b.label,
            'kind': self.b.kind, 'term': self.b.term, 'succs': list(self.b.succs),
            'returns': ret_ty,
            'branch': ({'cond': self.b.cond, 'index_when_0': i0, 'index_when_1': i1,
                        'succ_when_0': self.b.when0, 'succ_when_1': self.b.when1}
                       if forks else None),
            'statements': getattr(self.b, '_source_statement_count', len(self.b.stmts)),
            'value_statements': len(self.b.stmts), 'uses': sorted(self.uses),
            'undef_table': list(self.undef_table),
            'undef_used': sorted({u for st in self.b.stmts for u in _undef_names(st)}),
            'call_sites': [hex(x) for x in self.call_sites],
            'guest_lo': self.guest_lo, 'guest_hi': self.guest_hi,
            'guest_addrs': list(self.guest_addrs),
            'warnings': list(self.warnings),
        }
        return '\n'.join(out), manifest


def _undef_names(st) -> list[str]:
    """The undefined-flag witnesses a statement reads, in first-occurrence order.

    `('witness', ...)` is the same table read at register width, so it belongs here too: the payload
    header's table is what `tvm_undef_flag` indexes, and an index assigned on one side and not the
    other would make the two readings use different values.
    """
    out: list[str] = []

    def walk(e):
        if not isinstance(e, tuple) or not e:
            return
        if not isinstance(e[0], str):
            for y in e:
                walk(y)
            return
        if e[0] in ('undef', 'witness') and e[1] not in out:
            out.append(e[1])
        for x in e[1:]:
            if isinstance(x, tuple):
                walk(x)

    if isinstance(st.expr, tuple):
        walk(st.expr)
    return out


def _abs_addresses(st) -> list[int]:
    """Absolute guest-memory addresses a statement names, so the runtime region is known."""
    out: list[int] = []

    def walk(e):
        if not isinstance(e, tuple) or not e:
            return
        if not isinstance(e[0], str):
            for y in e:
                walk(y)
            return
        if e[0] in ('load', 'hload') and isinstance(e[2], tuple) and e[2][0] == 'const':
            out.append(e[2][1])
        for x in e[1:]:
            if isinstance(x, tuple):
                walk(x)

    if st.kind == 'store' and isinstance(st.expr, tuple):
        if st.expr[0][0] == 'const':
            out.append(st.expr[0][1])
        for part in st.expr:
            walk(part)
    elif isinstance(st.expr, tuple):
        walk(st.expr)
    return out


def _type_hint(uses) -> list[str]:
    decls = []
    if 'undef' in uses:
        decls.append('declare i32 @tvm_undef_flag(i32)')
    if 'callreg' in uses or 'flagbit' in uses:
        decls.append('declare i64 @tvm_call_reg(i32, i32)')
        decls.append('declare i64 @tvm_call_flags(i32)')
        decls.append('declare i64 @tvm_call_ret(i32)')
    if 'call' in uses:
        decls.append('declare i64 @tvm_native_call(i32, i64, i64, i64, i64)')
    if 'sealed_mem' in uses:
        # All four widths on both sides, whether or not a given module uses them: an unused
        # declaration is free, and a missing one is a link error that names the helper rather than
        # the statement that needed it. The return type is the *access* width, so the call site's
        # declared width (which the listing states) is the type LLVM checks.
        for b in (8, 16, 32, 64):
            decls.append(f'declare {_TY[b]} @tvm_ld{b}(ptr, i64)')
        for b in (8, 16, 32, 64):
            decls.append(f'declare void @tvm_st{b}(ptr, i64, {_TY[b]})')
    decls.append('declare i64 @tvm_udiv(i64, i64)')
    decls.append('declare i64 @tvm_urem(i64, i64)')
    if 'dbg' in uses:
        decls.append('declare void @llvm.dbg.value(metadata, metadata, metadata)')
    return decls


def emit_function(fn: Function, blocks=None,
                  attrs: str = ' "stack-probe-size"="4294967295"') -> tuple[list[str], list[dict]] | tuple[str, list[dict]]:
    """Emit the selected blocks of `fn` as one module. Returns `(text, manifests)`.

    `attrs` defaults to disabling the stack probe, for the reason given on `Emitter.__init__`: at
    `O0` a large block's frame crosses the 4 KiB threshold, LLVM then calls `__chkstk`, and a module
    linked `/NODEFAULTLIB` has no CRT to resolve it against. Pass `attrs=''` to reproduce the failure.

    **`"stack-probe-size"="4294967295"` is the spelling that works; `"probe-stack"="none"` and
    `"probe-stack"="inline-asm"` do not.** Measured on `0x140169e9d` block 5: the first suppresses the
    `__chkstk` reference at both codegen levels and links, the other two remove the `__chkstk` symbol
    and leave an *unnamed* unresolved one. A value of 2^32 -- one past the signed range -- does not
    parse as intended either and brings the probe back.
    """
    _require_parsed(fn)
    from valueflow import prepare, ValueFlowError
    try:
        fn = prepare(fn)
    except ValueFlowError as exc:
        raise HazardError(str(exc)) from exc
    selected = [b for b in fn.blocks if blocks is None or b.index in blocks]
    # **The witness table is built first, over every selected block.** `@tvm_undef_flag` is indexed,
    # and the index has to mean the same name in every block of the module, so it cannot come from a
    # per-block ordering discovered during emission.
    undef_table: list[str] = []
    for b in selected:
        for st in b.stmts:
            for name in _undef_names(st):
                if name not in undef_table:
                    undef_table.append(name)
    undef_table.sort()
    if len(undef_table) > 8192:
        raise HazardError(f'{len(undef_table)} distinct undefined flags exceeds the runtime table')
    bodies = []
    for b in selected:
        sub = Emitter(b, fn, undef_table, attrs)
        bodies.append(sub.emit())
    if not bodies:
        return '', []
    alluses = {u for _t, m in bodies for u in m['uses']}
    lo = min((m['guest_lo'] for _t, m in bodies if m['guest_hi']), default=0)
    hi = max((m['guest_hi'] for _t, m in bodies if m['guest_hi']), default=0)
    undefs: list[str] = list(undef_table)
    sites: list[str] = []
    for _t, m in bodies:
        for s in m['call_sites']:
            if s not in sites:
                sites.append(s)
    head = [
        '; transcribed from the recovered IR by llvm/lower_llvm.py',
        ';',
        '; HAZARD POSTURE: no nsw/nuw/exact/inbounds/noalias/nonnull/dereferenceable/noundef,',
        '; no undef, no poison, `align 1` on every access, shifts masked, division via a call.',
        '; See the module docstring; `hazard_lint` re-checks it on the finished text.',
        '',
        'target triple = "x86_64-pc-windows-msvc"',
        '',
        f'@tvm_guest_lo = constant i64 {lo}',
        f'@tvm_guest_hi = constant i64 {hi}',
    ]
    for i, u in enumerate(undefs):
        head.append(f'; undef-flag {i} = {u}')
    for i, s in enumerate(sites):
        head.append(f'; call-site {i} = {s}')
    parts = head + [''] + _type_hint(alluses) + ['']
    for text, _m in bodies:
        parts.append(text)
        parts.append('')
    return '\n'.join(parts), [m for _t, m in bodies]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    import guestir as G
    fn = G.parse_listing(path.read_text(encoding='utf-8', errors='replace'), path)
    only = None
    out_path = None
    for a in argv[1:]:
        if a.startswith('--block='):
            only = {int(a.split('=', 1)[1], 0)}
        elif not a.startswith('--'):
            out_path = Path(a)
    text, manifests = emit_function(fn, only)
    problems = hazard_lint(text)
    warns = [w for m in manifests for w in m['warnings']]
    print(f'{path.name}: {len(manifests)} block(s), {len(text.splitlines())} lines, '
          f'{len(problems)} hazard problem(s), {len(warns)} warning(s)')
    for p in problems[:20]:
        print('  HAZARD', p)
    for w in warns[:10]:
        print('  WARN', w)
    if out_path:
        out_path.write_text(text, encoding='utf-8')
        Path(str(out_path) + '.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')
        print(f'wrote {out_path}')
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
