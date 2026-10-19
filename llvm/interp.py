"""An independent evaluator for a recovered block: the oracle the LLVM path is checked against.

This module deliberately shares **no code** with `lower_llvm.py` beyond the parsed statements. It
computes what the block does to the guest register file and to guest memory by walking the listing
directly, and it is the thing the compiled code is compared against. That is the whole design:
LLVM is not trusted to be correct, so its output is checked against a second, hand-written reading
of the same source. A disagreement is a finding either way -- if the two agree and the listing is
right, the transcription is faithful; if they disagree, one of the two readings is wrong and the
first differing byte says which.

The semantics implemented here are the *memory* semantics described in `guestir.py`:

* the guest register file is 17 eight-byte slots in the context block;
* a context read is a load from its slot, a full write a store, a partial write a narrow store;
* `in.<reg>` / `init_<flag>` come from an entry snapshot taken before the first statement;
* guest memory is a byte-addressed region at the recovered absolute addresses;
* a bare `flag_<f>` is an *input*, not a derivation -- the caller supplies it, because
  `evidence/E-FLAG-ORIGIN-AND-OF.md` establishes it is an architectural-undefined witness.

Run:  python llvm/interp.py <listing.txt> --block=N  (prints the block's effect on a sample state)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import guestir as G  # noqa: E402

CTX_BYTES = 8 * len(G.CTX_LAYOUT)


def seed_free(addr: int) -> int:
    """A deterministic byte for any address, so two readings of a block see the same memory.

    Defined here rather than in the harness because both the interpreter and the native side have to
    use *the same* function, and because a test that invents its own seed is a test of a different
    initial state. It is a pure function of the address, so it needs no state to reproduce.
    """
    return (addr * 37 + (addr >> 13) * 11 + 5) & 0xFF


class Mem:
    """Little-endian byte memory at absolute addresses, **bounded by the regions that exist**.

    The bound is the part that matters. A dictionary-backed memory models *infinite* address space,
    so an interpreter over it will happily read address `-1` and report a value, while the native
    code faults. The first version of this class did exactly that, and the differential test then
    crashed with an access violation on a block whose address came out as `0xffffffffffffffff`
    instead of reporting the disagreement. An access outside every provided region is therefore an
    error here too, which makes the two readings comparable.
    """

    def __init__(self, regions=(), default=None):
        self.bytes: dict[int, int] = {}
        self.regions = list(regions)
        self.default = default or (lambda a: 0)

    def _check(self, addr: int, nbytes: int) -> None:
        for lo, hi in self.regions:
            if lo <= addr and addr + nbytes <= hi:
                return
        raise MemoryError(f'access at {addr:#x}+{nbytes} is outside every provided region')

    def load(self, addr: int, bits: int) -> int:
        n = bits // 8
        self._check(addr, n)
        v = 0
        for i in range(n):
            v |= (self.bytes.get(addr + i, self.default(addr + i)) & 0xFF) << (8 * i)
        return v

    def store(self, addr: int, bits: int, value: int) -> None:
        n = bits // 8
        self._check(addr, n)
        value &= (1 << bits) - 1
        for i in range(n):
            self.bytes[addr + i] = (value >> (8 * i)) & 0xFF

    def span(self) -> tuple[int, int] | None:
        if not self.bytes:
            return None
        return min(self.bytes), max(self.bytes) + 1

    def dump(self, lo: int, hi: int) -> bytes:
        return bytes(self.bytes.get(a, self.default(a)) & 0xFF for a in range(lo, hi))


class Ctx:
    """The guest register file: 8-byte slots at the offsets `guestir.CTX_LAYOUT` records."""

    def __init__(self, slots: dict[str, int] | None = None):
        self.mem = bytearray(CTX_BYTES)
        for reg, v in (slots or {}).items():
            self.write(G.CTX_LAYOUT[reg], 64, v)

    def read(self, off: int, bits: int) -> int:
        v = 0
        for i in range(bits // 8):
            v |= self.mem[off + i] << (8 * i)
        return v

    def write(self, off: int, bits: int, value: int) -> None:
        value &= (1 << bits) - 1
        for i in range(bits // 8):
            self.mem[off + i] = (value >> (8 * i)) & 0xFF

    def slots(self) -> dict[str, int]:
        return {reg: self.read(off, 64) for reg, off in G.CTX_LAYOUT.items()}

    def snapshot(self) -> bytes:
        return bytes(self.mem)


class Inputs:
    """Everything outside the block that the block may read.

    All four sources are explicit because none of them is derivable from the listing, and pretending
    otherwise is exactly the fabrication this project refuses.

    `strict` decides what happens when a source is missing. In strict mode (the default) a missing
    value is an error, which is what an audit of the *listing* wants. The differential test uses
    `strict=False`, where the missing value comes from the documented default below --
    **and the C runtime implements the same formulas**, so the two readings of the block are
    compared on identical inputs instead of on two different guesses at what a call returned.
    """

    def __init__(self, undef: dict[str, int] | None = None,
                 callreg: dict[tuple[str, int], int] | None = None,
                 callflags: dict[int, int] | None = None,
                 callresult: dict[int, int] | None = None,
                 strict: bool = True):
        self.undef = dict(undef or {})
        self.callreg = dict(callreg or {})
        self.callflags = dict(callflags or {})
        self.callresult = dict(callresult or {})
        self.strict = strict
        self.reads: dict[str, int] = {}

    def note(self, what: str) -> None:
        self.reads[what] = self.reads.get(what, 0) + 1

    def flag(self, name: str) -> int:
        self.note('undef:' + name)
        if name not in self.undef:
            raise KeyError(f'the block reads the undefined flag {name!r} and no value was supplied')
        return self.undef[name] & 1

    def witness(self, name: str) -> int:
        """The whole witness value, for `box<N>#<reg>` -- an unmodelled instruction's result.

        The same table as `flag`, without the `& 1`: a boxed value was written to a *register*, and
        the payload's `tvm_undef_flag` returns the same scalar, so both readings use the same number
        at the same width.
        """
        self.note('witness:' + name)
        if name not in self.undef:
            raise KeyError(f'the block reads the undefined value {name!r} and no value was supplied')
        return self.undef[name]


_M64 = (1 << 64) - 1


def default_callreg(call: int, reg: int) -> int:
    """Mirror of `tvm_call_reg` in `llvm/runtime/tvm_runtime.c`."""
    return (((call * 0x9E3779B97F4A7C15) & _M64) + reg * 0x100) & _M64 ^ 0x5A5A5A5A


def default_callflags(call: int) -> int:
    """Mirror of `tvm_call_flags`."""
    return 0x202 | (call & 7)


def default_callresult(site: int) -> int:
    """Mirror of `tvm_native_call`, injective in `site`."""
    return ((site * 0x9E3779B97F4A7C15) & _M64) ^ 0xC0FFEE


class Result:
    def __init__(self):
        self.ctx: Ctx | None = None
        self.mem: Mem | None = None
        self.temps: dict[str, int] = {}
        self.reads: dict[str, int] = {}
        self.errors: list[str] = []
        # Which successor the fork condition selects, as an index into `block.succs`, or None when
        # the block does not fork. `None` and `0` are different answers.
        self.succ: int | None = None
        # Set when the block accessed memory the caller did not provide. That is not a transcription
        # error and not a completed run: the machine would fault, so the interpreter stops there and
        # says which address. Comparing a faulted run against a completed one would compare two
        # different things, which is how a differential test starts reporting fiction.
        self.faulted: bool = False
        self.fault_addr: int | None = None
        self.terminal: dict | None = None


def _width(e, hint: int, declared: dict[str, int]) -> int:
    return G_width_width(e, hint, declared)


def G_width_width(e, hint: int, declared: dict[str, int]) -> int:
    if not isinstance(e, tuple) or not e or not isinstance(e[0], str):
        return hint
    op = e[0]
    if op == 'const':
        return hint
    if op in ('slot', 'entry', 'reg', 'callreg', 'callret'):
        return 64
    if op == 'temp':
        return declared.get(e[1]) or hint
    if op in ('flagbit', 'undef'):
        return 1
    if op == 'witness':
        # `box<N>#<reg>`: an unmodelled instruction's result. **Register-wide, not one bit** -- see
        # the note in `guestir._named`; a one-bit reading of it made the interpreter write zero bytes
        # into a 64-bit slot while the payload wrote one, and the two readings of `0x145c2102c`
        # then disagreed on `rsp` by exactly its low byte.
        return 64
    if op == 'bin':
        # A comparison's declared width is the width of the result, and in this engine that is the
        # operand width: `v429:64 = a <u b` is a 64-bit 0/1. See `lower_llvm.Widths.width_of`.
        return hint if op else hint
    if op == 'un':
        return hint
    if op == 'cast':
        return e[4]
    if op == 'sel':
        return hint
    if op in ('load', 'hload'):
        return e[1]
    return hint


def run_block(block, ctx: Ctx, mem: Mem, inputs: Inputs,
              declared: dict[str, int] | None = None, reading: str = 'value') -> Result:
    """Evaluate one block. `ctx` and `mem` are read and updated in place.

    The default, 'value', applies the explicit capture/entry-parameter pass.
    'slot' and 'binding' retain the old diagnostic readings solely to reproduce
    the audit counterexample; neither is the executable default. A standalone
    block with unproved cross-block inputs refuses. Whole-function execution
    validates those inputs against all CFG predecessors before evaluating blocks.
    """
    if reading == 'value':
        from valueflow import resolve_block
        block = resolve_block(block)
    elif reading not in ('slot', 'binding'):
        raise ValueError(f'unknown register reading {reading!r}')
    r = Result()
    r.ctx, r.mem = ctx, mem
    if declared is None:
        declared = {st.dest: st.width for st in block.stmts
                    if st.kind in ('def', 'ctx-load') and st.width}
    entry: dict[str, int] = {reg: ctx.read(off, 64) for reg, off in G.CTX_LAYOUT.items()}
    #: name -> value, for the 'binding' reading only. Block-local: a name defined in an earlier
    #: block falls back to the slot, which is what the composition already assumes (`in.<reg>` is
    #: the snapshot mechanism across blocks, not a name).
    bound: dict[str, int] = {}

    def ev(e, hint: int) -> tuple[int, int]:
        w = G_width_width(e, hint, declared)
        mask = (1 << w) - 1
        op = e[0]

        def fit(v: int, vw: int, want: int) -> tuple[int, int]:
            """Coerce an operand to an operation's width, matching `core/expr.py::Arena._fit`.

            The listing prints `a OP b` without the coercion the arena applied, so a transcription
            that skips it gets the arithmetic wrong whenever the operand widths differ. This is the
            same reconstruction as `lower_llvm.Emitter.fit`; the two are kept in step deliberately,
            because the differential test compares them.
            """
            if vw == want:
                return v, want
            return v & ((1 << want) - 1), want

        if op == 'const':
            return e[1] & mask, w
        if op == 'temp':
            if e[1] not in r.temps:
                raise KeyError(f'temporary {e[1]!r} read before definition')
            return r.temps[e[1]] & mask, w
        if op == 'slot':
            return ctx.read(G.CTX_LAYOUT[e[1]], 64), 64
        if op == 'reg':
            # The version number is not used: `guest_rax_1` and `guest_rax_9` are the same machine
            # access, and the slot's current contents are what it reads.
            if reading == 'binding':
                got = bound.get(f'guest_{e[1]}_{e[2]}')
                if got is not None:
                    return got & _M64, 64
            return ctx.read(G.CTX_LAYOUT[e[1]], 64), 64
        if op == 'entry':
            return entry[e[1]], 64
        if op == 'flagbit':
            flag, source = e[1], e[2]
            word = entry['rflags'] if source == 'entry' else _callflags(source, inputs)
            return (word >> G.FLAG_BITS[flag]) & 1, 1
        if op == 'undef':
            return inputs.flag(e[1]), 1
        if op == 'witness':
            return inputs.witness(e[1]) & _M64, 64
        if op == 'callreg':
            key = (e[1], e[2])
            inputs.note(f'callreg:{e[1]}#{e[2]}')
            if key in inputs.callreg:
                return inputs.callreg[key] & _M64, 64
            if inputs.strict:
                raise KeyError(f'no value supplied for callr{e[1]}#{e[2]}')
            return default_callreg(e[2], G.CTX_LAYOUT[e[1]]) & _M64, 64
        if op == 'callret':
            inputs.note(f'callret#{e[1]}')
            if e[1] in inputs.callresult:
                return inputs.callresult[e[1]] & _M64, 64
            if inputs.strict:
                raise KeyError(f'no value supplied for call_ret#{e[1]}')
            return default_callresult(e[1]) & _M64, 64
        if op == 'cast':
            _o, kind, inner, _sw, dstw = e
            v, sw = ev(inner, dstw)
            dm = (1 << dstw) - 1
            if sw == dstw:
                return v & dm, dstw
            if sw > dstw:
                return v & dm, dstw
            if kind == 'sext' and v & (1 << (sw - 1)):
                return (v | ~((1 << sw) - 1)) & dm, dstw
            return v & dm, dstw
        if op == 'un':
            v, sw = fit(*ev(e[2], w), w)
            return ((~v) if e[1] == 'not' else (-v)) & ((1 << sw) - 1), sw
        if op == 'bin':
            name = e[1]
            if name in ('eq', 'ne', 'ult', 'ule', 'slt', 'sle', 'ugt', 'uge', 'sgt', 'sge'):
                a, aw = fit(*ev(e[2], hint), hint)
                b, bw = fit(*ev(e[3], hint), hint)
                if aw != bw:
                    raise KeyError(f'comparison operands {aw}/{bw}')
                sa = a - (1 << aw) if (aw and a >> (aw - 1)) else a
                sb = b - (1 << bw) if (bw and b >> (bw - 1)) else b
                res = {'eq': a == b, 'ne': a != b,
                       'ult': a < b, 'ule': a <= b, 'ugt': a > b, 'uge': a >= b,
                       'slt': sa < sb, 'sle': sa <= sb, 'sgt': sa > sb, 'sge': sa >= sb}[name]
                # The listing's declared width governs: a 0/1 in a `width`-bit value, not an i1.
                return int(res), (hint if hint > 1 else 1)
            if name in ('udiv', 'urem'):
                a, aw = fit(*ev(e[2], w), w)
                b, bw = fit(*ev(e[3], w), w)
                if aw != bw:
                    raise KeyError(f'division operands {aw}/{bw}')
                if b == 0:
                    raise ZeroDivisionError('guest division by zero (the guest would fault)')
                return (a // b if name == 'udiv' else a % b) & ((1 << aw) - 1), aw
            a, aw = fit(*ev(e[2], w), w)
            b, bw = fit(*ev(e[3], w), w)
            if aw != bw:
                raise KeyError(f'{name} operands {aw}/{bw}')
            m = (1 << w) - 1
            if name == 'add':
                return (a + b) & m, w
            if name == 'sub':
                return (a - b) & m, w
            if name == 'mul':
                return (a * b) & m, w
            if name == 'and':
                return a & b, w
            if name == 'or':
                return a | b, w
            if name == 'xor':
                return a ^ b, w
            if name == 'shl':
                # **Masked exactly as the emitter masks it**, so the two are comparable: x86 keeps
                # count bits `log2(width)` and `core/lift64.py::_shift` implements the same rule.
                return (a << (b & (w - 1))) & m, w
            if name == 'lshr':
                return a >> (b & (w - 1)), w
            if name == 'ashr':
                s = a - (1 << w) if a >> (w - 1) else a
                return (s >> (b & (w - 1))) & m, w
            raise KeyError(f'operator {name!r}')
        if op == 'sel':
            c, _cw = ev(e[1], 32)
            t, tw = fit(*ev(e[2], w), w)
            f, fw = fit(*ev(e[3], w), w)
            if tw != fw:
                raise KeyError(f'select arms {tw}/{fw}')
            return (t if c != 0 else f), tw
        if op in ('load', 'hload'):
            a, _aw = ev(e[2], 64)
            return mem.load(a, e[1]), e[1]
        if op == 'pack':
            pairs, base = e[1], e[2]
            acc = base
            for flag, sub in pairs:
                v, vw = ev(sub, 1)
                if vw != 1:
                    raise KeyError(f'pack input {flag} is {vw} bits')
                acc |= (v & 1) << G.FLAG_BITS[flag]
            return acc & ((1 << 64) - 1), 64
        if op == 'insert':
            # insert_bits(prev, lo, n, v): `prev` with bits `lo..lo+n` replaced. Used directly as a
            # value only outside a context write, where the emitter refuses it; here it is exact.
            prev, _pw = ev(e[1], 64)
            lo, _lw = ev(e[2], 32)
            n, _nw = ev(e[3], 32)
            v, _vw = ev(e[4], n)
            mask = ((1 << n) - 1) << lo
            return (prev & ~mask) | ((v << lo) & mask), 64
        if op == 'call':
            _o, target, args, site = e
            for _reg, sub in args:
                ev(sub, 64)
            inputs.note(f'call:{site:#x}')
            if site in inputs.callresult:
                return inputs.callresult[site] & _M64, 64
            if inputs.strict:
                raise KeyError(f'no result supplied for the call at {site:#x}')
            return default_callresult(site) & _M64, 64
        raise KeyError(f'expression {op!r}')

    for st in block.stmts:
        try:
            if st.kind in ('bare', 'comment', 'entry-note', 'call-block', 'forward-note', 'read-note'):
                continue
            if st.kind == 'def':
                v, vw = ev(st.expr, st.width or 64)
                w = st.width or vw
                if w != vw:
                    raise KeyError(f'{st.dest}: declared {w} bits, value {vw}')
                r.temps[st.dest] = v & ((1 << w) - 1)
                if st.dest.startswith('guest_'):
                    bound[st.dest] = r.temps[st.dest]
                continue
            if st.kind == 'ctx-load':
                _reg, off, _site = st.extra
                r.temps[st.dest] = ctx.read(off, st.width)
                if st.dest.startswith('guest_'):
                    bound[st.dest] = r.temps[st.dest]
                continue
            if st.kind in ('ctx-write', 'home'):
                _reg, off, _site = st.extra
                v, vw = ev(st.expr, 64)
                ctx.write(off, min(64, vw), v)
                if st.dest.startswith('guest_'):
                    bound[st.dest] = ctx.read(off, 64)
                continue
            if st.kind == 'ctx-partial':
                _reg, off, _site = st.extra
                _o, _prev, lo_bits, nb_bits, val = st.expr
                lo, nb = lo_bits[1], nb_bits[1]
                v, vw = ev(val, nb)
                if vw != nb:
                    raise KeyError(f'partial value {vw} bits, slice {nb}')
                ctx.write(off + lo // 8, nb, v)
                if st.dest.startswith('guest_'):
                    bound[st.dest] = ctx.read(off, 64)
                continue
            if st.kind == 'store':
                bits = st.width
                addr, value = st.expr
                a, _aw = ev(addr, 64)
                v, _vw = ev(value, bits)
                mem.store(a, bits, v)
                continue
            raise KeyError(f'statement {st.kind!r}')
        except MemoryError as exc:
            # The machine faults here; carry on and the run stops describing the machine.
            r.errors.append(f'{st.kind} {st.dest}: {exc}')
            r.faulted = True
            r.fault_addr = _first_address(str(exc))
            break
        except (KeyError, ZeroDivisionError) as exc:
            r.errors.append(f'{st.kind} {st.dest}: {exc}')

    # **The fork, evaluated last, because the machine evaluates it after the block's work.**
    if block.branches:
        try:
            v = _cond_value(block, r, inputs)
            taken = block.when1 if v != 0 else block.when0
            idx = block.succ_index(taken)
            if idx < 0:
                r.errors.append(f'branch target {taken} is not among the block\'s successors')
            else:
                r.succ = idx
        except KeyError:
            r.errors.append(f'branch condition {block.cond!r} was never defined')
    r.reads = dict(inputs.reads)
    if not r.errors:
        from execution_status import terminal_descriptor
        r.terminal = terminal_descriptor(block)
    return r


def _cond_value(block, r: Result, inputs: Inputs) -> int:
    """The value a fork tests, **materialised the way the payload materialises it**.

    A condition is usually a temporary the body defined, and `r.temps` has it. It can also be a
    *witness spelling* -- `; branch condition: flag_cf` names the architectural carry as an input, not
    as an SSA name -- and then nothing in the block defines it. The payload's `Emitter.cond` reads such
    a name through `tvm_undef_flag` (see `lower_llvm.cond`), so an interpreter that only looks in
    `r.temps` refuses the fork the payload takes: on `ACE-CORE.sys2` root `0x140108ba8` that was
    `#3: branch condition 'flag_zf' was never defined`, the walk stopped at step 3, and the payload
    ran 4 -- a differential failure that was entirely the harness's.
    """
    if block.cond in r.temps:
        return r.temps[block.cond]
    short = block.cond.split(':', 1)[0]
    if short.startswith(('flag_', 'init_', 'mul_undefined_', 'imul_undefined_')):
        return inputs.flag(short)
    if re.match(r'^box\d+#\w+$', short):
        return inputs.witness(short)
    raise KeyError(block.cond)


class FunctionRun:
    """The result of walking a whole recovered CFG."""

    def __init__(self):
        self.steps = 0
        self.finished = False      # reached a block whose listing terminator is `return`
        self.exhausted = False     # ran out of fuel
        self.stuck = ''            # why it stopped otherwise
        self.errors: list[str] = []
        self.faulted = False
        self.visited: list[int] = []
        self.terminal: dict | None = None

    @property
    def encoded_status(self) -> int | None:
        from execution_status import MASK64, TERMINAL_BIT
        if self.terminal:
            return TERMINAL_BIT | self.steps
        if self.exhausted:
            return (~self.steps) & MASK64
        return self.steps if self.finished else None


def run_function(fn, ctx: Ctx, mem: Mem, inputs: Inputs, fuel: int = 10_000,
                 reading: str = 'value') -> FunctionRun:
    """Walk the recovered CFG, executing each block in turn on the same context and memory.

    The composition claim, stated exactly: block B's `in.<reg>` is the register value at *B's entry*,
    so each block takes its own entry snapshot from the live context -- which is why this reuses the
    same `Ctx` object across blocks rather than copying state forward. **129 reads of `in.<reg>` in
    the corpus follow a write to that same register inside the same block**, so an entry snapshot is
    not the same thing as a live slot read and the distinction has to survive composition.

    The walk is bounded: a recovered CFG can contain a `backedge`, and an unbounded loop would hang
    the harness rather than tell it anything.
    """
    if reading == 'value':
        from valueflow import prepare
        fn = prepare(fn)
    run = FunctionRun()
    if not fn.blocks:
        run.stuck = 'no blocks'
        return run
    by_label = {b.label: b for b in fn.blocks}
    b = fn.blocks[0]
    while True:
        if run.steps >= fuel:
            run.exhausted = True
            return run
        rr = run_block(b, ctx, mem, inputs, reading=reading)
        run.steps += 1
        run.visited.append(b.index)
        if rr.errors:
            run.errors.extend(f'#{b.index}: {e}' for e in rr.errors)
            if rr.faulted:
                run.faulted = True
                return run
        if b.term in ('return', 'tail'):
            run.finished = True
            return run
        if rr.terminal:
            run.terminal = rr.terminal
            return run
        if b.term == 'goto' and len(b.succs) == 1:
            nxt = b.succs[0]
        elif b.branches and rr.succ is not None:
            nxt = b.succs[rr.succ]
        else:
            run.stuck = (f'block #{b.index} exits as {b.term!r} with {len(b.succs)} '
                         f'successor(s) and no fork condition')
            return run
        if nxt not in by_label:
            run.stuck = f'block #{b.index} names successor {nxt}, which is not a block of this function'
            return run
        b = by_label[nxt]


def _first_address(text: str) -> int | None:
    """The hex address a `MemoryError` message reports, for reporting it back."""
    import re as _re
    m = _re.search(r'at (0x[0-9a-f]+)', text)
    return int(m.group(1), 16) if m else None


def _callflags(source, inputs: Inputs) -> int:
    _kind, n = source
    inputs.note(f'callflags#{n}')
    if n in inputs.callflags:
        return inputs.callflags[n]
    if inputs.strict:
        raise KeyError(f'no flags supplied for call {n}')
    return default_callflags(n)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    which = 0
    for a in argv:
        if a.startswith('--block='):
            which = int(a.split('=', 1)[1], 0)
    fn = G.parse_listing(path.read_text(encoding='utf-8', errors='replace'), path)
    block = next(b for b in fn.blocks if b.index == which)
    ctx = Ctx({reg: 0x1000 + 8 * i for i, reg in enumerate(G.CTX_LAYOUT)})
    ctx.write(G.CTX_LAYOUT['rflags'], 64, 0x202)
    mem = Mem()
    inputs = Inputs(undef={'flag_af': 0})
    before = ctx.snapshot()
    r = run_block(block, ctx, mem, inputs)
    print(f'{path.name} block #{which} {block.label} [{block.kind}] {len(block.stmts)} stmts')
    print(f'inputs read: {r.reads}')
    print(f'errors: {r.errors or "none"}')
    print(f'temporaries: {len(r.temps)}')
    print(f'ctx changed slots:')
    for reg, off in G.CTX_LAYOUT.items():
        a = int.from_bytes(before[off:off + 8], 'little')
        b = ctx.read(off, 64)
        if a != b:
            print(f'  {reg:<7} {a:#018x} -> {b:#018x}')
    sp = mem.span()
    print(f'guest memory touched: {sp and (hex(sp[0]), hex(sp[1]))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
