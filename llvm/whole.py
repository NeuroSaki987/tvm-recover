"""One recovered **function**, as one LLVM function.

`lower_llvm` emits one LLVM function per recovered block, which makes each block testable on its own
and made the branch work observable. This module closes the loop: every block of a function becomes a
basic block of a single LLVM function, the terminators become real `br`/`ret`, and the compiled
function can therefore be **run end to end** and compared against the interpreter walking the same
CFG.

Four things had to be right for that to be sound, and each is a measurement rather than an assumption:

1. **Entry snapshots move into the block.** `in.<reg>` is the register as of that block's entry, and
   **129 reads of `in.<reg>` in the corpus occur after the same register was written in the same
   block** -- so reading the live context slot instead would be wrong, and taking the snapshot once at
   function entry would be wrong for every block but the first. It is taken at the top of each
   block's basic block, before that block's first statement.
2. **The context block is the merge.** Every guest register write is a store to `%ctx`, so a join
   needs no phi: the memory holds whichever predecessor ran, which is what the machine does.
3. **Cycles are real.** Two recovered blocks are `backedge` kind, so the CFG can loop. The function
   therefore takes a **fuel** budget and stops when it runs out, returning a distinguishable value;
   an unbounded loop would hang the harness rather than tell it anything.
4. **A block whose exit the listing does not settle refuses the whole function.** Emitting the settled
   ones and inventing the rest would be exactly the fabrication this project rejects, so the refusal
   names the block.

Signature:

    define i64 @tvm_<root>(ptr %ctx, i64 %fuel)

returns `steps` on a normal finish, `~steps` on fuel exhaustion, or
`0x8000000000000000 | steps` on an analysis terminal. A terminal never resumes
the recovered guest CFG; the safe harness receives its status instead.
"""

from __future__ import annotations

from pathlib import Path
from execution_status import TERMINAL_BIT, TERMINAL_KINDS, ENCODING, terminal_descriptor

from guestir import CTX_LAYOUT, Block, Function
from lower_llvm import (Emitter, HazardError, _abs_addresses, _san, _TY, _type_hint,
                        _undef_names, _require_parsed)


class WholeRefusal(HazardError):
    """The function cannot be emitted as one piece, and the reason is named."""


def _prepare_values(fn):
    from valueflow import prepare, ValueFlowError
    try:
        return prepare(fn)
    except ValueFlowError as exc:
        raise WholeRefusal(str(exc)) from exc


def _terminator(block: Block) -> str:
    """How a block's exit is spelled in the listing, as a claim about what we can emit."""
    if block.term in TERMINAL_KINDS:
        return block.term
    if block.term in ('return', 'tail'):
        # A `ret` and a tail call are both the *end* of the recovered function: the tail call's
        # callee returns to our caller, so there is nothing left to emit after it.
        return 'return'
    if block.term == 'goto' and len(block.succs) == 1:
        return 'goto'
    if block.branches:
        return 'branch'
    return 'unsettled'


def _witness_table(fn: Function) -> list[str]:
    """Every undefined-flag witness the function reads, sorted, as one module-wide index space."""
    table: list[str] = []
    for b in fn.blocks:
        for st in b.stmts:
            for name in _undef_names(st):
                if name not in table:
                    table.append(name)
    table.sort()
    if len(table) > 8192:
        raise WholeRefusal(f'{len(table)} distinct undefined flags exceeds the runtime table')
    return table


# --------------------------------------------------------------------- chunked emission (IDA-sized)

def _expr_uses(e) -> set[str]:
    """Every `%name` an expression reads: temporaries *and* block-entry snapshots."""
    out: set[str] = set()

    def walk(x):
        if not isinstance(x, tuple) or not x:
            return
        if not isinstance(x[0], str):
            for y in x:
                walk(y)
            return
        if x[0] in ('temp', 'reg', 'opaque_name'):
            out.add(x[1])
        elif x[0] == 'entry':
            # `in.<reg>` is defined once, at the block's entry, and must cross a cut like any other
            # value: reloading it from the context in a later piece would read whatever the block's
            # own earlier statements had already written there.
            out.add(f'in_{x[1]}')
        for y in x[1:]:
            if isinstance(y, tuple):
                walk(y)

    walk(e)
    return out


def _stmt_uses(st) -> set[str]:
    return _expr_uses(st.expr) if isinstance(st.expr, tuple) else set()


def chunk_plan(fn: Function, split: int) -> list[dict]:
    """Cut the function into pieces of at most `split` statements, and record what crosses each cut.

    A piece is a maximal run of statements inside **one** recovered block, so no cut ever lands in the
    middle of a control-flow decision -- a block is straight-line except for its terminator. What
    crosses a cut is the set of values defined before it and read after it, and that set is small
    enough to spill into a caller-provided array: measured over the whole corpus, **at most 17
    values** are live at any cut (`llvm/build/measure_cuts.py`).

    Returns one record per piece: `{'block', 'start', 'stop', 'entry_span', 'exit_span', 'live_in',
    'live_out', 'first', 'last'}` where `live_in`/`live_out` map a name to its spill slot.
    """
    # Every name the whole function *defines as an SSA value* has a width; the entry snapshots are
    # 64-bit. `ctx-write`, `home` and `ctx-partial` carry a `dest` too, but they emit a *store* and no
    # `%dest` value -- so they are not definitions here, and treating them as such produced a spill of
    # a name that does not exist (measured on `0x14015fc2e`, `%v3494`).
    fn = _prepare_values(fn)
    widths: dict[str, int] = {}
    for b in fn.blocks:
        for st in b.stmts:
            if st.kind in ('def', 'ctx-load') and st.dest:
                widths[st.dest] = st.width or 64
        for st in b.stmts:
            for name in _stmt_uses(st):
                if name.startswith('in_') and name[3:] in CTX_LAYOUT:
                    widths.setdefault(name, 64)

    pieces: list[dict] = []
    for b in fn.blocks:
        n = len(b.stmts)
        # Everything available at the block's *entry*: the `in.<reg>` snapshots `build_prologue` takes
        # before the first statement. Control arrives from another block with nothing in registers, so
        # a block's first piece has an empty live-in by construction.
        entry_names = {f'in_{reg}' for reg in CTX_LAYOUT if _snapshot_needed(b, reg)}

        def defined_before(cut: int) -> set[str]:
            out = set(entry_names)
            for st in b.stmts[:cut]:
                if st.kind in ('def', 'ctx-load') and st.dest:
                    out.add(st.dest)
            return out

        uses_after: list[set[str]] = [set() for _ in range(n + 1)]
        acc: set[str] = set()
        # The terminator reads the branch condition at the *end* of the block, so for the purposes of
        # liveness the condition is used after every cut -- not only after the last one. Getting this
        # wrong left a 62-statement gap where the condition was live but unspilled (`0x14015fc2e`
        # block 15, condition `v3494`).
        cond_names = {b.cond} if _terminator(b) == 'branch' else set()
        for i in range(n - 1, -1, -1):
            acc = acc | _stmt_uses(b.stmts[i])
            uses_after[i] = acc | cond_names

        def live_at(cut: int) -> list[str]:
            """Values defined strictly before `cut` and read at or after it. A pure function of the
            cut, which is what makes a piece's live-in equal the previous piece's live-out."""
            return sorted(defined_before(cut) & uses_after[cut])

        starts = list(range(0, max(n, 1), split)) or [0]
        block_pieces: list[dict] = []
        for k, start in enumerate(starts):
            stop = min(start + split, n)
            last = stop >= n
            used = set(live_at(stop))
            if last and _terminator(b) == 'branch':
                used.add(b.cond)        # read by the terminator, not by a statement
            live_out = sorted(defined_before(stop) & used)
            block_pieces.append({
                'block': b.index, 'label': b.label, 'start': start, 'stop': stop,
                'last': last, 'statements': stop - start,
                'live_in': [] if k == 0 else live_at(start),
                'live_out': live_out,
            })
        pieces.extend(block_pieces)
    for p in pieces:
        p['first'] = p['live_in'] == [] and p['start'] == 0
        p['live_in_widths'] = [widths.get(n_, 64) for n_ in p['live_in']]
        p['live_out_widths'] = [widths.get(n_, 64) for n_ in p['live_out']]
    return pieces


def _snapshot_needed(block: Block, reg: str) -> bool:
    """Whether this block reads `in.<reg>` at all."""
    for st in block.stmts:
        if f'in_{reg}' in _stmt_uses(st):
            return True
    return False


def emit_chunked(fn: Function, split: int, undef_table: list[str] | None = None,
                 mem_mode: str = 'sealed',
                 attrs: str = ' "stack-probe-size"="4294967295"',
                 dbg: bool = False) -> tuple[list[str], dict, set]:
    """Emit one recovered function as a **series** of machine functions of bounded size.

    Why: a recovered block can be tens of thousands of statements (`0x140169e9d` block 5 is 36 073),
    which compiles to a function no decompiler will take -- IDA refuses it as too big, which makes the
    artifact useless for the analysis it exists to support. So the block is cut at statement
    boundaries into pieces of at most `split` statements, each piece becomes its own LLVM function,
    and a generated dispatcher (`payload.py`) walks the piece graph.

    Three things keep this honest rather than merely smaller:

    * **The cut is at a statement boundary inside one block**, never across a control-flow decision: a
      block is straight-line up to its terminator, so the pieces of a block are a plain sequence.
    * **What crosses a cut is spilled explicitly**, into an array the *caller* owns (the dispatcher's
      stack frame) rather than into the payload's state, so `tvm_state_t` does not grow and the
      sealed accessors are untouched. The spilled set is measured, not guessed: at most 17 values over
      the whole corpus.
    * **The step and fuel accounting stays where it was**: it moves to the dispatcher, which knows
      from a per-piece flag which pieces start and end a block, so the observable step count and the
      fuel-exhaustion point are unchanged -- which the differential test then verifies.

    A piece returns the next piece, `n_pieces` for a normal finish, or
    `n_pieces + 1` for an analysis terminal.
    """
    fn = _prepare_values(fn)
    try:
        _require_parsed(fn)
    except HazardError as exc:
        raise WholeRefusal(str(exc)) from exc
    unsettled = [(b.index, b.term) for b in fn.blocks if _terminator(b) == 'unsettled']
    if unsettled:
        raise WholeRefusal(
            'block(s) whose exit the listing does not settle: '
            + ', '.join(f'#{i} {t}' for i, t in unsettled))
    by_label = {b.label: b for b in fn.blocks}
    for b in fn.blocks:
        for s in b.succs:
            if s not in by_label:
                raise WholeRefusal(f'block #{b.index} names successor {s}, which is not a block '
                                   f'of this function')
    table = _witness_table(fn) if undef_table is None else list(undef_table)
    for b in fn.blocks:
        for st in b.stmts:
            for name in _undef_names(st):
                if name not in table:
                    raise WholeRefusal(f'witness {name!r} of block #{b.index} is not in the shared '
                                       'table; a shared index space has to cover every function '
                                       'that uses it')

    pieces = chunk_plan(fn, split)
    first_piece_of_block = {}
    for i, p in enumerate(pieces):
        p['index'] = i
        if p['first']:
            first_piece_of_block[p['block']] = i
    n = len(pieces)
    sealed = mem_mode == 'sealed'
    uses: set[str] = set()
    lines: list[str] = []
    names: list[dict] = []
    # With `dbg`, the function names its `DISubprogram` and one ordinary instruction carries a
    # location; both are placeholders substituted when the module's metadata is assembled. A module
    # whose only debug attachments are `dbg.value` calls emits no CodeView function record at all
    # (measured: 112 bytes of `.debug$S` and no names), which is why the terminator has one too.
    # **The key carries the piece index**: LLVM refuses to attach one `DISubprogram` to two functions
    # ("DISubprogram attached to more than one function") and then drops the module's entire debug
    # information, so every piece gets its own, named exactly as the linker symbol.
    for p in pieces:
        # The name says what the function *is*: `b<block>_p<k>` is piece k of recovered block
        # `<block>`, so an analyst reading a decompiler's function list sees the recovered control
        # flow's shape without opening a single one. `p["index"]` is global; the name is per block.
        local = p['index'] - first_piece_of_block[p['block']]
        name = f'tvm_{fn.root:x}_b{p["block"]}_p{local}'
        key = f'{fn.root:x}_b{p["block"]}_p{local}'
        sub = f' !dbg !SPFILE_{key}' if dbg else ''
        loc = f', !dbg !LOCFILE_{key}' if dbg else ''
        b = next(x for x in fn.blocks if x.index == p['block'])
        # One binding map per piece: a piece is its own machine function, and what crosses the cut
        # travels through the spill array instead.
        aliases: dict[str, tuple[str, int]] = {}
        em = Emitter(b, fn, table, attrs, prefix=f'p{p["index"]}_', mem_mode=mem_mode,
                     state_ptr='%state' if sealed else '', dbg=dbg, dbg_key=key,
                     alias=aliases)
        em.stmt_range = range(p['start'], p['stop'])
        em.prescan()
        # Entry snapshots need one decision per piece: a piece that *spills out* an `in.<reg>` has to
        # define it (from the context -- valid only on the block's first piece), and a piece that
        # *spills in* one must not also define it, because the spill load already did.
        for nm in p['live_out']:
            if nm.startswith('in_') and nm[3:] in CTX_LAYOUT:
                em.needs_entry.add(nm[3:])
        for nm in p['live_in']:
            if nm.startswith('in_') and nm[3:] in CTX_LAYOUT:
                em.needs_entry.discard(nm[3:])
        # The spill-in lines are built **before** the body, because they are what defines the values
        # the body is about to read; they are emitted at the top of the function below.
        def llvm_name(nm: str) -> str:
            """The LLVM operand this piece uses for a recovered name.

            A name the piece bound to a value (a listing temporary, or a value that came in through
            the spill array) resolves to that value; an entry snapshot the prologue defines resolves
            to the emitter's spelling of it, and anything else is its own LLVM name.
            """
            hit = em.alias.get(nm)
            if hit is not None:
                return hit[0]
            if nm.startswith('in_') and nm[3:] in CTX_LAYOUT:
                return f'%{em.prefix}in_{_san(nm[3:])}'
            return f'%{_san(nm)}'

        spill_in: list[str] = []
        for k, (nm, wd) in enumerate(zip(p['live_in'], p['live_in_widths'])):
            sp = f'%p{p["index"]}_in_spill_{k}'
            spill_in.append(f'  {sp} = getelementptr i8, ptr %spill, i64 {8 * k}')
            raw = f'%p{p["index"]}_in_raw_{k}'
            spill_in.append(f'  {raw} = load i64, ptr {sp}, align 1')
            # **A binding, not `%name = add i64 %raw, 0`.** The loaded value *is* the recovered value;
            # naming it with an identity instruction costs one instruction per live-in value at `-O0`
            # and shows up in a decompiler as `v12 = v11 + 0`.
            if wd == 64:
                em.alias[nm] = (raw, 64)
            else:
                z = f'%p{p["index"]}_in_trunc_{k}'
                spill_in.append(f'  {z} = trunc i64 {raw} to {_TY[wd]}')
                em.alias[nm] = (z, wd)
            if nm.startswith('in_') and nm[3:] in CTX_LAYOUT:
                # An entry snapshot: `build_prologue` would have loaded it from the context, but the
                # context may already have been written by this block's earlier statements.
                em.entry_snap[nm[3:]] = em.alias[nm][0]
        prologue = em.build_prologue()
        em.build_body()
        # **The condition is resolved *before* the body is flushed.** cond() can *emit* an
        # instruction (a flag condition reads a witness), and emitting it after lines.extend(em.lines)
        # would leave the instruction out of the function while the terminator still used its name --
        # measured as error: use of undefined value %b2_ub2`.
        cond_ready = em.cond() if _terminator(b) == 'branch' else None
        uses |= em.uses
        names.extend(em.named)
        args = 'ptr %ctx, ptr %state, ptr %spill' if sealed else 'ptr %ctx, ptr %spill'
        lines.append('')
        lines.append(f'define i64 @{name}({args}){attrs}{sub} {{')
        lines.append('entry:')
        lines.extend(spill_in)
        lines.extend(prologue)
        lines.extend(em.lines)
        # Spill-out: the values the next piece will need.
        for k, (nm, wd) in enumerate(zip(p['live_out'], p['live_out_widths'])):
            sp = f'%p{p["index"]}_out_spill_{k}'
            lines.append(f'  {sp} = getelementptr i8, ptr %spill, i64 {8 * k}')
            if wd == 64:
                lines.append(f'  store i64 {llvm_name(nm)}, ptr {sp}, align 1')
            else:
                z = f'%p{p["index"]}_out_zext_{k}'
                lines.append(f'  {z} = zext {_TY[wd]} {llvm_name(nm)} to i64')
                lines.append(f'  store i64 {z}, ptr {sp}, align 1')
        kind = _terminator(b)
        if not p['last']:
            # A piece that is not the block's last hands over to the next piece of the same block.
            lines.append(f'  ret i64 {p["index"] + 1}{loc}')
        elif kind == 'return':
            lines.append(f'  ret i64 {n}{loc}')
        elif kind in TERMINAL_KINDS:
            # A distinct dispatcher sentinel, never an ordinary return or a piece index.
            lines.append(f'  ret i64 {n + 1}{loc}')
        elif kind == 'goto':
            lines.append(f'  ret i64 {first_piece_of_block[by_label[b.succs[0]].index]}{loc}')
        else:
            bare, cw = cond_ready
            t = f'%p{p["index"]}_nz'
            lines.append(f'  {t} = icmp ne {_TY[cw]} {bare}, 0')
            pick = f'%p{p["index"]}_next'
            lines.append(f'  {pick} = select i1 {t}, i64 '
                         f'{first_piece_of_block[by_label[b.when1].index]}, i64 '
                         f'{first_piece_of_block[by_label[b.when0].index]}')
            lines.append(f'  ret i64 {pick}{loc}')
        lines.append('}')
        p['function'] = name

    manifest = {
        'function': f'tvm_{fn.root:x}', 'root': fn.root, 'blocks': [b.index for b in fn.blocks],
        'statements': sum(getattr(b, '_source_statement_count', len(b.stmts)) for b in fn.blocks),
        'value_statements': sum(len(b.stmts) for b in fn.blocks),
        'undef_table': table, 'mem_mode': mem_mode, 'split': split,
        'pieces': [{k: v for k, v in p.items() if k != 'live_in_widths'} for p in pieces],
        'n_pieces': n,
        'terminal_piece_status': n + 1,
        'terminals': [terminal_descriptor(b) for b in fn.blocks if b.term in TERMINAL_KINDS],
        'status_encoding': ENCODING,
        'block_first_piece': {f'{k}': v for k, v in first_piece_of_block.items()},
        'max_live_out': max((len(p['live_out']) for p in pieces), default=0),
        'max_statements': max((p['statements'] for p in pieces), default=0),
        'uses': sorted(uses),
        'guest_addrs': sorted({a for b in fn.blocks for st in b.stmts for a in _abs_addresses(st)}),
        # Every named value carries the *emitted function* it belongs to (`key`), because one
        # `DISubprogram` may only ever be attached to one machine function.
        'dbg_names': names, 'dbg_key': f'{fn.root:x}',
    }
    return lines, manifest, uses


def whole_function_text(fn: Function, undef_table: list[str] | None = None,
                        mem_mode: str = 'direct',
                        attrs: str = ' "stack-probe-size"="4294967295"',
                        dbg: bool = False) -> tuple[list[str], dict, set]:
    """The `define ... }` of one recovered function, as lines. Returns `(lines, manifest, uses)`.

    Split out of `emit_whole` so that **several functions can share one module** -- the payload
    written by `write-devirt` needs one translation unit, so the `target triple`, the `declare`s and
    (critically) the witness index space have to be emitted once for the whole set rather than once
    per function. `undef_table` is that shared index space; when it is omitted the table is built
    from this function alone, which is exactly what the differential test has always done.
    """
    try:
        _require_parsed(fn)
    except HazardError as exc:
        raise WholeRefusal(str(exc)) from exc
    fn = _prepare_values(fn)
    if not fn.blocks:
        raise WholeRefusal('no blocks')
    unsettled = [(b.index, b.term) for b in fn.blocks if _terminator(b) == 'unsettled']
    if unsettled:
        raise WholeRefusal(
            'block(s) whose exit the listing does not settle: '
            + ', '.join(f'#{i} {t}' for i, t in unsettled))

    by_label = {b.label: b for b in fn.blocks}
    for b in fn.blocks:
        for s in b.succs:
            if s not in by_label:
                raise WholeRefusal(f'block #{b.index} names successor {s}, which is not a block '
                                   f'of this function')

    # The witness table is module-wide, as for the per-block emitter: an index has to mean the same
    # name everywhere, so it is built over all blocks before any is emitted.
    table = _witness_table(fn) if undef_table is None else list(undef_table)
    if len(table) > 8192:
        raise WholeRefusal(f'{len(table)} distinct undefined flags exceeds the runtime table')
    for b in fn.blocks:
        for st in b.stmts:
            for name in _undef_names(st):
                if name not in table:
                    raise WholeRefusal(
                        f'witness {name!r} of block #{b.index} is not in the shared table; a shared '
                        'index space has to cover every function that uses it')

    addrs: set[int] = set()
    uses: set[str] = set()
    for b in fn.blocks:
        for st in b.stmts:
            addrs.update(_abs_addresses(st))

    name = f'tvm_{fn.root:x}'
    # **The sealed signature carries the state block.** In `direct` mode guest memory is an absolute
    # address and needs no operand, so the signature is the one every earlier artifact was measured
    # with: `(ptr %ctx, i64 %fuel)`. In `sealed` mode the accessors need the caller's state, and
    # passing it as a parameter (rather than through a global) is what keeps the injected payload
    # free of writable globals -- see `llvm/runtime/tvm_sealed.h`.
    sealed = mem_mode == 'sealed'
    params = 'ptr %ctx, ptr %state, i64 %fuel' if sealed else 'ptr %ctx, i64 %fuel'
    # With `dbg`, the definition names its `DISubprogram`; the placeholder is substituted when the
    # module's metadata is assembled. Codegen needs it to attach the values to a function.
    sub = f' !dbg !SPFILE_{fn.root:x}' if dbg else ''
    lines: list[str] = [
        f'define i64 @{name}({params}){attrs}{sub} {{',
        'entry:',
        '  %fuelp = alloca i64, align 1',
        '  %stepsp = alloca i64, align 1',
        '  store i64 %fuel, ptr %fuelp, align 1',
        '  store i64 0, ptr %stepsp, align 1',
        f'  br label %b{fn.blocks[0].index}',
    ]

    total_stmts = 0
    names: list[dict] = []
    # One name-binding map for the whole function: a listing name is unique per function, and a
    # condition or operand can be defined in an earlier block than the one that reads it.
    aliases: dict[str, tuple[str, int]] = {}
    for b in fn.blocks:
        em = Emitter(b, fn, table, attrs, prefix=f'b{b.index}_', mem_mode=mem_mode,
                     state_ptr='%state' if sealed else '', dbg=dbg, alias=aliases)
        em.prescan()
        prologue = em.build_prologue()
        em.build_body()
        cond_ready = em.cond() if _terminator(b) == 'branch' else None
        total_stmts += getattr(b, '_source_statement_count', len(b.stmts))
        uses |= em.uses
        names.extend(em.named)

        kind = _terminator(b)
        # With `dbg`, a real `!DILocation` on ordinary instructions as well as on the `dbg.value`
        # calls. **Required, and measured**: a module whose only debug attachments are the intrinsic
        # calls emits no CodeView function record at all (112 bytes of `.debug$S`, no names), while
        # the same module with a location on a `load`/`add`/`ret` emits them. Those three opcodes all
        # take metadata attachments, so nothing about the code's shape changes.
        loc = f', !dbg !LOCFILE_{fn.root:x}' if dbg else ''
        lines.append('')
        lines.append(f'b{b.index}:')
        lines.extend(prologue)
        # Fuel, checked per block so a cycle ends instead of hanging. `i64` and not `i32`: the count
        # comes from the caller and is not bounded by anything in the listing.
        lines.append(f'  %fu{b.index} = load i64, ptr %fuelp, align 1{loc}')
        lines.append(f'  %fz{b.index} = icmp eq i64 %fu{b.index}, 0')
        lines.append(f'  br i1 %fz{b.index}, label %exhausted, label %b{b.index}_run')
        lines.append('')
        lines.append(f'b{b.index}_run:')
        lines.extend(em.lines)
        lines.append(f'  %fs{b.index} = sub i64 %fu{b.index}, 1')
        lines.append(f'  store i64 %fs{b.index}, ptr %fuelp, align 1')
        lines.append(f'  %sn{b.index} = load i64, ptr %stepsp, align 1')
        lines.append(f'  %sn{b.index}n = add i64 %sn{b.index}, 1{loc}')
        lines.append(f'  store i64 %sn{b.index}n, ptr %stepsp, align 1')
        if kind == 'return':
            lines.append('  br label %done')
        elif kind in TERMINAL_KINDS:
            # This returns to the analysis harness, with no guest continuation.
            terminal_status = em.fresh('terminal_status')
            lines.append(f'  {terminal_status} = or i64 %sn{b.index}n, {TERMINAL_BIT}')
            lines.append(f'  ret i64 {terminal_status}{loc}')
        elif kind == 'goto':
            lines.append(f'  br label %b{by_label[b.succs[0]].index}')
        else:
            bare, cw = cond_ready
            t = em.fresh('nz')
            lines.append(f'  {t} = icmp ne {_TY[cw]} {bare}, 0')
            hi = by_label[b.when1].index
            lo_ = by_label[b.when0].index
            # A fork whose two successors coincide is a plain jump; emitting a `br` to two identical
            # labels would be accepted by LLVM but would say something the listing does not.
            lines.append(f'  br i1 {t}, label %b{hi}, label %b{lo_}')

    lines.extend([
        '',
        '; **Fuel exhausted is not a normal return.** The step count is returned inverted, so a',
        '; harness cannot mistake "ran out of budget" for "finished at step N".',
        'exhausted:',
        '  %xe = load i64, ptr %stepsp, align 1',
        '  %xen = xor i64 %xe, -1',
        f'  ret i64 %xen{loc}',
        '',
        'done:',
        '  %xd = load i64, ptr %stepsp, align 1',
        f'  ret i64 %xd{loc}',
        '}',
    ])
    body = lines

    lo = min(addrs) if addrs else 0
    hi = max(addrs) + 8 if addrs else 0
    manifest = {
        'function': name, 'root': fn.root, 'blocks': [b.index for b in fn.blocks],
        'statements': total_stmts, 'undef_table': table,
        'guest_lo': lo, 'guest_hi': hi, 'guest_addrs': sorted(addrs),
        'uses': sorted(uses), 'mem_mode': mem_mode, 'dbg_names': names,
        'fuel_encoded_as': 'steps on return, ~steps on exhaustion',
        'status_encoding': ENCODING,
        'terminals': [terminal_descriptor(b) for b in fn.blocks if b.term in TERMINAL_KINDS],
    }
    return body, manifest, uses


def emit_whole(fn: Function, attrs: str = ' "stack-probe-size"="4294967295"',
               undef_table: list[str] | None = None,
               mem_mode: str = 'direct') -> tuple[str, dict]:
    """Emit every block of `fn` as one LLVM function. Returns `(text, manifest)`."""
    body, manifest, uses = whole_function_text(fn, undef_table, mem_mode, attrs)
    table = manifest['undef_table']
    addrs = manifest['guest_addrs']
    lo, hi = manifest['guest_lo'], manifest['guest_hi']
    head = [
        '; one recovered function as one LLVM function, via llvm/whole.py',
        ';',
        '; HAZARD POSTURE: as in llvm/lower_llvm.py -- no nsw/nuw/exact/inbounds/noalias/nonnull/',
        '; dereferenceable/noundef, no undef, no poison, `align 1` on every access, shifts masked,',
        '; division via a call. `hazard_lint` re-checks it on the finished text.',
        '',
        'target triple = "x86_64-pc-windows-msvc"',
        '',
        f'@tvm_guest_lo = constant i64 {lo}',
        f'@tvm_guest_hi = constant i64 {hi}',
    ]
    for i, u in enumerate(table):
        head.append(f'; undef-flag {i} = {u}')
    for i, addrx in enumerate(sorted(addrs)):
        head.append(f'; guest address {i} = {addrx:#x}')
    text = '\n'.join(head + [''] + _type_hint(uses) + [''] + body) + '\n'
    return text, manifest


def main(argv: list[str]) -> int:
    import json
    import sys
    if not argv:
        print(__doc__)
        return 2
    import guestir as G
    from lower_llvm import hazard_lint
    p = Path(argv[0])
    fn = G.parse_listing(p.read_text(encoding='utf-8', errors='replace'), p)
    out = Path(argv[1]) if len(argv) > 1 else p.with_suffix('.whole.ll')
    try:
        text, man = emit_whole(fn)
    except WholeRefusal as exc:
        print(f'{p.name}: REFUSED: {exc}')
        return 1
    problems = hazard_lint(text)
    print(f'{p.name}: {len(man["blocks"])} block(s), {man["statements"]} statement(s), '
          f'{len(text.splitlines())} lines, {len(problems)} hazard problem(s)')
    for q in problems[:10]:
        print('  HAZARD', q)
    out.write_text(text, encoding='utf-8')
    print(f'wrote {out}')
    if '--manifest' in argv:
        Path(str(out) + '.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main(__import__('sys').argv[1:]))
