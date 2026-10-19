"""Parse the recovered listing without discarding value provenance.

`guest_<reg>_<version>` denotes a captured value, while `guest_<reg>` is
an explicit live slot operand. A `#N:` prefix on a versioned operand records
its defining block and survives parsing. Raw statements stay available for
historical census tools.

Executable consumers apply `valueflow.prepare`: local values become immutable
temporaries, recorded context reads capture at their read point, and cross-block
values become block-entry parameters only when all incoming paths prove that
the slot transports the named definition. Unproved continuity refuses emission.
Partial writes still update only the recorded byte slice. `in.<reg>` and
`init_<flag>` are snapshots at each recovered block's entry.

The legacy canonical/version-renaming probe is only an alpha-renaming census;
it is not a proof that register versions may be ignored during execution.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- tables

REGS = ("rax", "rbx", "rcx", "rdx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rflags")
# The context-block layout, as every listing states it. Verified uniform across all 14 certified
# roots (`[ctx+0x00 = rax]` .. `[ctx+0x80 = rflags]`, 69 843 bracketed occurrences).
CTX_LAYOUT = {reg: 8 * i for i, reg in enumerate(REGS)}
FLAG_BITS = {"cf": 0, "pf": 2, "af": 4, "zf": 6, "sf": 7, "df": 10, "of": 11}
PACK_BASE = 2  # RFLAGS bit 1 is always set on x86-64

_INFIX = {
    # Lowest precedence first. The listing prints two-operand lines with already-named operands, so
    # precedence almost never binds -- but `(guest_rax >> 32)` from the narrow-read rule does have
    # parentheses, and a parser that could not nest them would be a parser for one example.
    "|": ("or", 10), "^": ("xor", 20), "&": ("and", 30),
    "<<": ("shl", 40), ">>": ("lshr", 40), ">>a": ("ashr", 40), "/": ("udiv", 45), "%": ("urem", 45),
    "+": ("add", 50), "-": ("sub", 50), "*": ("mul", 60),
    "==": ("eq", 5), "!=": ("ne", 5), "<u": ("ult", 5), "<=u": ("ule", 5),
    "<s": ("slt", 5), "<=s": ("sle", 5), ">u": ("ugt", 5), ">=u": ("uge", 5),
    ">s": ("sgt", 5), ">=s": ("sge", 5),
}
# `>>a` must be tried before `>>`, and two-character comparisons before one-character ones.
_OPERATORS = sorted(_INFIX, key=len, reverse=True)

_RE_HEX = re.compile(r'0x[0-9a-fA-F]+')
_RE_DEC = re.compile(r'\d+')
_RE_NAME = re.compile(r'[A-Za-z_][\w.#]*')


class ParseError(Exception):
    """A line or expression the grammar does not cover.

    Raised rather than skipped: a line that is dropped is a semantic that is lost silently, and this
    project's recurring failure mode is exactly that.
    """


# --------------------------------------------------------------------------- expressions
#
# Expressions are tuples, because they are compared, counted and printed far more often than they
# are introspected, and a 40 000-line body of dataclasses is slower to build for no benefit.
#
#   ('const', value, width)
#   ('slot', reg)                  live read of the context slot  -- memory at this point
#   ('entry', reg)                 `in.<reg>`  -- entry snapshot
#   ('temp', name, width)          `v123:64` / `guest_rax_2` (via 'reg')
#   ('reg', reg, version)          `guest_<reg>_<k>` used as a value
#   ('flagbit', flag, source)      source='entry' | ('call', n) | 'current'
#   ('undef', name)                bare `flag_af` / `mul_undefined_pf#0` -- architecturally undefined,
#                                  and **one bit** wide (a flag is a bit)
#   ('witness', name)              `box<N>#<reg>` -- an unmodelled instruction's result: an unknown
#                                  whose width is the register's, so 64 bits, not one
#   ('callreg', reg, n)            `callrsp#1`  -- register value after call n
#   ('bin', op, a, b, width)
#   ('un', op, a, width)           op in {'not', 'neg'}
#   ('cast', op, a, srcw, dstw)    op in {'trunc','zext','sext'}; dstw from the printed name
#   ('sel', cond, a, b, width)     `c ? a : b`, condition tested for NONZERO
#   ('load', bits, addr)           guest memory
#   ('hload', bits, addr, ver)     `load64 [x] @m262` -- a load at a memory *version*
#   ('pack', pairs, base)          `pack_flags(cf=..., ...)`
#   ('insert', prev, lo, bits, value)
#   ('call', target, args, site)   `call 0x14000d4e0(rcx=..., ...) @0x14014c690`


class ExprParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def peek(self) -> str:
        while self.pos < len(self.text) and self.text[self.pos] == ' ':
            self.pos += 1
        return self.text[self.pos:]

    def eof(self) -> bool:
        return not self.peek()

    def take(self, s: str) -> bool:
        if self.peek().startswith(s):
            self.pos += len(s)
            return True
        return False

    def expect(self, s: str) -> None:
        if not self.take(s):
            raise ParseError(f'expected {s!r} at {self.text[self.pos:]!r} in {self.text!r}')

    # ---------------------------------------------------------------- precedence climb
    def parse(self, min_prec: int = 0):
        left = self.unary()
        while True:
            self.peek()
            op = None
            for cand in _OPERATORS:
                if self.text.startswith(cand, self.pos):
                    # `<<` must not be read as `<` then `<`; the sorted-by-length order fixes that,
                    # but `>>a` versus `>>` needs the longer one first, which it is.
                    op = cand
                    break
            if op is None:
                return left
            name, prec = _INFIX[op]
            if prec < min_prec:
                return left
            self.pos += len(op)
            right = self.parse(prec + 1)
            left = ('bin', name, left, right, None)

    def unary(self):
        if self.take('~'):
            return ('un', 'not', self.unary(), None)
        if self.take('-'):
            return ('un', 'neg', self.unary(), None)
        return self.primary()

    def primary(self):
        cur = self.peek()
        if not cur:
            raise ParseError(f'unexpected end of expression: {self.text!r}')
        if cur[0] == '(':
            self.pos += 1
            inner = self.parse(0)
            self.expect(')')
            return self._ternary(inner)
        if self.text.startswith('load', self.pos):
            m = re.compile(r'load(\d+) \[').match(self.text, self.pos)
            if m:
                self.pos = m.end()
                addr = self.parse(0)
                self.expect(']')
                ver = None
                mv = re.compile(r' @m(\d+)').match(self.text, self.pos)
                if mv:
                    self.pos = mv.end()
                    ver = int(mv.group(1))
                bits = int(m.group(1))
                return ('hload', bits, addr, ver) if ver is not None else ('load', bits, addr)
        for op in ('trunc', 'zext', 'sext'):
            m = re.compile(op + r'(\d+)\(').match(self.text, self.pos)
            if m:
                self.pos = m.end()
                inner = self.parse(0)
                self.expect(')')
                return ('cast', op, inner, None, int(m.group(1)))
        if self.text.startswith('pack_flags(', self.pos):
            self.pos += len('pack_flags(')
            pairs = []
            while True:
                if self.take(')'):
                    break
                m = re.compile(r'([a-z]+)=').match(self.text, self.pos)
                if not m:
                    raise ParseError(f'bad pack_flags argument in {self.text!r}')
                self.pos = m.end()
                pairs.append((m.group(1), self.parse(0)))
                self.take(',')
            base = PACK_BASE
            mb = re.compile(r' \| (0x[0-9a-f]+)').match(self.text, self.pos)
            if mb:
                self.pos = mb.end()
                base = int(mb.group(1), 16)
            return ('pack', tuple(pairs), base)
        if self.text.startswith('insert_bits(', self.pos):
            self.pos += len('insert_bits(')
            args = []
            for _ in range(4):
                args.append(self.parse(0))
                self.take(',')
            self.expect(')')
            return ('insert', args[0], args[1], args[2], args[3])
        if self.text.startswith('input64(state.', self.pos):
            m = re.compile(r'input64\(state\.(\w+)\)').match(self.text, self.pos)
            self.pos = m.end()
            return ('entry', m.group(1))
        if self.text.startswith('call ', self.pos):
            m = re.compile(r'call ([^\s(]+)\(').match(self.text, self.pos)
            if m:
                self.pos = m.end()
                args = []
                if not self.take(')'):
                    while True:
                        self.peek()
                        arg = re.compile(r'(\w+)\s*=').match(self.text, self.pos)
                        if not arg:
                            raise ParseError(f'expected named call argument at '
                                             f'{self.text[self.pos:]!r} in {self.text!r}')
                        self.pos = arg.end()
                        # Nested expressions consume their own parentheses and commas.
                        args.append((arg.group(1), self.parse(0)))
                        if self.take(')'):
                            break
                        self.expect(',')
                site = re.compile(r'\s+@(0x[0-9a-f]+)').match(self.text, self.pos)
                if site:
                    self.pos = site.end()
                return ('call', m.group(1), tuple(args),
                        int(site.group(1), 16) if site else 0)
        m = _RE_HEX.match(self.text, self.pos)
        if m:
            self.pos = m.end()
            return self._ternary(('const', int(m.group(0), 16), None))
        if cur[0].isdigit():
            m = _RE_DEC.match(self.text, self.pos)
            self.pos = m.end()
            return self._ternary(('const', int(m.group(0)), None))
        m = re.compile(r'#\d+:').match(self.text, self.pos)
        if m:
            # Preserve the owner of a versioned value. It must not accidentally
            # bind to a same-numbered local register version in the consumer block.
            self.pos = m.end()
            value = self.primary()
            if value[0] == 'reg':
                value = (*value, int(m.group(0)[1:-1]))
            return self._ternary(value)
        if self.text.startswith('in.', self.pos):
            m = re.compile(r'in\.(\w+)').match(self.text, self.pos)
            self.pos = m.end()
            return self._ternary(('entry', m.group(1)))
        m = _RE_NAME.match(self.text, self.pos)
        if m:
            name = m.group(0)
            self.pos = m.end()
            return self._ternary(self._named(name))
        raise ParseError(f'cannot parse {self.text[self.pos:]!r} in {self.text!r}')

    def _named(self, name: str):
        w = None
        m = re.compile(r':(\d+)').match(self.text, self.pos)
        if m:
            self.pos = m.end()
            w = int(m.group(1))
        if name.startswith('guest_'):
            tail = name[len('guest_'):]
            head, _, last = tail.rpartition('_')
            if last.isdigit() and head in CTX_LAYOUT:
                return ('reg', head, int(last))
            if tail in CTX_LAYOUT:
                return ('slot', tail)
            return ('opaque_name', name)
        if name.startswith('callr'):
            # The listing spells a post-call register as `callr` plus the register's **short** name:
            # `callrcx#1`, `callrdx#1`, `callr8#1`, `callrsp#1`. A lazy `\w+?` captured `cx` there and
            # the emitter then refused the register; the short form is normalised once, here.
            m = re.compile(r'callr(\w+)#(\d+)').match(name)
            if m:
                return ('callreg', _full_reg(m.group(1)), int(m.group(2)))
            return ('opaque_name', name)
        if name.startswith('call_ret#'):
            # `v427:64 = call_ret#3` -- the return value of the engine's third call, used here in a
            # block that did not make that call. It is an input, not a derivation: the listing says
            # where it came from (`; from guest_rdi via rdi`, a carry) and nothing more.
            return ('callret', int(name.split('#', 1)[1]))
        if name.startswith('init_'):
            flag = name[len('init_'):]
            if flag in FLAG_BITS:
                return ('flagbit', flag, 'entry')
            return ('opaque_name', name)
        if name.startswith('flag_'):
            body = name[len('flag_'):]
            mc = re.compile(r'([a-z]+)#call(\d+)').match(body)
            if mc and mc.group(1) in FLAG_BITS:
                # `('aftercall', n)`, **not** `('call', n)`: the source of a flag bit would otherwise
                # be indistinguishable from a call expression, and a walker that keys on the first
                # element then indexes a four-element call tuple with two elements. That is a real
                # crash this parser produced, not a hypothetical.
                return ('flagbit', mc.group(1), ('aftercall', int(mc.group(2))))
            if body in FLAG_BITS:
                return ('undef', name)
            return ('opaque_name', name)
        if name.startswith('mul_undefined_') or name.startswith('imul_undefined_'):
            return ('undef', name)
        if name.startswith('box'):
            # **The result of an instruction the lifter does not model.** `core/lift64.py::_box`
            # records the instruction and continues with an opaque value per written register, spelled
            # `box<step>#<reg>` (`box7625#edx`: step 7625 wrote edx with a value nothing here knows).
            # That is precisely what the witness table exists for -- "every value the recovery does not
            # know, behind a documented function" -- so it transcribes as one: a `tvm_undef_flag`
            # read whose assignment the payload header records (all zero, measured), not a fabricated
            # value and not a refusal. The listing still says which instruction produced it, and the
            # boxed-instruction log in the engine (`Boxed(ip, mnemonic, bytes, defs, uses)`) names it.
            #
            # **It is a `witness`, not an `undef` bit.** A boxed value fills a whole register, and the
            # two consumers disagreed about what a one-bit value stored into a 64-bit slot means: the
            # interpreter wrote `0` bytes (`ctx.write` of 1 bit rounds to nothing) while the payload
            # emitted `store i1`, which writes one byte -- so on `0x145c2102c` the rsp slot kept
            # `0x7ffffffe1130` on one side and became `0x7ffffffe1100` on the other, a difference of
            # exactly the low byte. A flag witness is one bit by definition; an unmodelled
            # instruction's result is as wide as the register it was written to.
            if re.match(r'^box\d+#\w+$', name):
                return ('witness', name)
            return ('opaque_name', name)
        if name.startswith('v') and name[1:].isdigit():
            return ('temp', name, w)
        return ('opaque_name', name)

    def _ternary(self, cond):
        if self.take('?'):
            a = self.parse(0)
            self.expect(':')
            b = self.parse(0)
            return ('sel', cond, a, b, None)
        return cond


def _split_args(text: str) -> list[str]:
    """Split a comma-separated argument list at nesting depth zero."""
    out, depth, cur = [], 0, ''
    for ch in text:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
            continue
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_expr(text: str):
    p = ExprParser(text)
    e = p.parse(0)
    if not p.eof():
        raise ParseError(f'trailing text {p.text[p.pos:]!r} in {text!r}')
    return e


# --------------------------------------------------------------------------- statements
#
#   ('ctx-write', reg, width, expr, site)      `guest_rax_3 = expr [ctx+0x00 = rax] ; write to ctx`
#   ('home',      reg, width, expr)            `guest_rax_3 = expr ; home rbx`
#   ('def',       dest, width, expr, origin)   `v12:64 = expr`
#   ('store',     bits, addr, value, site)     `store64 [addr], v`
#   ('bare',      expr, origin)                a line that defines nothing (a folded read)
#   ('comment',   text)


@dataclass
class Stmt:
    kind: str
    dest: str = ''
    width: int = 0
    expr: object = None
    extra: object = None
    origin: str = ''
    text: str = ''


@dataclass
class Block:
    label: str
    handler: int
    ctx: int
    kind: str
    index: int
    stmts: list = field(default_factory=list)
    succs: list = field(default_factory=list)
    term: str = ''
    term_text: str = ''
    entry_note: str = ''
    calls: list = field(default_factory=list)
    # The fork condition, when the listing states one it has also lowered:
    #   `; branch condition: v63:32   ; successor when 0: A   ; successor when 1: B   ; from ...`
    # `cond` is the name of a temporary defined in this block's body, so it carries a width and
    # needs no inference; `when0`/`when1` are successor keys as printed, mapped to an index into
    # `succs` so the emitted code returns a position rather than a key.
    cond: str = ''
    cond_width: int = 0
    when0: str = ''
    when1: str = ''
    cond_text: str = ''
    #: The engine proved this block unreachable: its exit is
    #: `<infeasible: the accumulated path conditions are unsatisfiable>`. Such a block has no
    #: successor (and no exit to transcribe), and any edge *into* it is a contradiction on the way in.
    #: It is dropped from the function, and the drop is recorded -- see `Function.infeasible_blocks`.
    infeasible: bool = False
    terminal_detail: str = ''
    source_index: int | None = None

    @property
    def branches(self) -> bool:
        """True only when a conditional branch can be emitted without inventing anything."""
        return bool(self.cond and self.when0 and self.when1 and len(self.succs) > 1)

    def succ_index(self, key: str) -> int:
        try:
            return self.succs.index(key)
        except ValueError:
            return -1


@dataclass
class Function:
    root: int
    path: Path | None = None
    header: dict = field(default_factory=dict)
    blocks: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    unparsed: list = field(default_factory=list)
    unparsed_line_numbers: list[int] = field(default_factory=list)
    #: Blocks the engine proved unreachable and the transcription therefore drops, with the engine's
    #: own reason text. Empty for every function that needs no such edit; a reader who sees a
    #: statement count lower than the listing's `lines` line can find out why here.
    infeasible_blocks: list = field(default_factory=list)

    @property
    def ctx_base(self) -> int:
        bases = {b.ctx for b in self.blocks}
        return bases.pop() if len(bases) == 1 else 0


_RE_BLOCK = re.compile(r'^(0x[0-9a-f]+)@ctx(0x[0-9a-f]+):\s+\[(\w+)\]\s+;\s+#(\d+)$')
_RE_COND = re.compile(r'^branch condition:\s*(\S+)')
_RE_WHEN = re.compile(r'successor when (\d):\s*(0x[0-9a-f]+@ctx0x[0-9a-f]+)')
_RE_FROM = re.compile(r'\s{2,};\sfrom\s+(.*)$')
_RE_SPLIT = re.compile(r'\s{2,};\s')
_RE_SLOT = re.compile(r'\[ctx\+(0x[0-9a-f]+)(?: = (\w+))?(?:\+(\d+))?\]\s*$')
_RE_DEF = re.compile(r'^([A-Za-z_][\w.]*?)(?::(\d+))?\s*=\s*(.+)$')
_RE_STORE = re.compile(r'^store(\d+) \[(.*)\], (.*)$')
_RE_LOAD = re.compile(r'^load(\d+) \[(.*)\]$')
_RE_TERM = re.compile(r'^-> ?(.*)$')
_RE_CALLBLOCK = re.compile(r'^call\s+(\S+)\s+@(0x[0-9a-f]+)$')


def parse_listing(text: str, path: Path | None = None) -> Function:
    lines = text.splitlines()
    root = 0
    header = {}
    fn = None
    cur: Block | None = None
    body_started = False

    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        context = re.match(r'^;\s*entry context:\s*(.*)$', raw.strip())
        if context:
            location = f'{path or "<listing>"}:{line_number}'
            if 'entry_context' in header:
                raise ParseError(f'{location}: duplicate entry context contract')
            try:
                contract = json.loads(context.group(1))
            except json.JSONDecodeError as exc:
                raise ParseError(f'{location}: invalid entry context JSON: {exc.msg}') from exc
            if not isinstance(contract, dict):
                raise ParseError(f'{location}: entry context must be a JSON object')
            # Preserve the complete precondition even though emission cannot yet
            # enforce it. Recognize it before indentation can turn it into a note.
            header['entry_context'] = contract
            header['entry_context_line'] = line_number
            continue
        if not raw.startswith(' '):
            s = raw.strip()
            if s.startswith(';'):
                m = re.match(r'^; devirtualized (0x[0-9a-f]+)', s)
                if m:
                    root = int(m.group(1), 16)
                    continue
                m = re.match(r'^; blocks (\d+)\s+unresolved (\d+)\s+instructions (\d+)$', s)
                if m:
                    header['blocks'] = int(m.group(1))
                    header['unresolved'] = int(m.group(2))
                    header['instructions'] = int(m.group(3))
                    continue
                m = re.match(r'^; calls (\d+)$', s)
                if m:
                    header['calls'] = int(m.group(1))
                    continue
                m = re.match(r'^; lines (\d+)\s+state (\d+)\s+computation kept (\d+)'
                             r'\s+\(of which (\d+) do not reach a home\)\s+suppressed (\d+)$', s)
                if m:
                    header['lines'] = int(m.group(1))
                    header['state'] = int(m.group(2))
                    header['kept'] = int(m.group(3))
                    continue
                if 'TRUNCATED' in s:
                    header['truncated'] = True
                continue
            m = _RE_BLOCK.match(s)
            if m:
                body_started = True
                cur = Block(label=s.split(':')[0], handler=int(m.group(1), 16),
                            ctx=int(m.group(2), 16), kind=m.group(3), index=int(m.group(4)))
                if fn is None:
                    fn = Function(root=root, path=path, header=header)
                fn.blocks.append(cur)
                continue
            if not body_started:
                continue
            # Once a block starts, indentation cannot hide executable text.
            # Parse unindented/tabbed lines through the same grammar and record
            # unknown effects with their source location for emitter refusal.
        if fn is None:
            continue
        body = raw.strip()
        if cur is None:
            # A note before the first block header (the header block and the ownership lines).
            fn.notes.append(body)
            continue
        st = _parse_stmt(body)
        if st is None:
            if fn.unparsed is not None:
                fn.unparsed.append(body)
                fn.unparsed_line_numbers.append(line_number)
            continue
        if st.kind == 'term':
            cur.term_text = st.text
            cur.term = st.dest
            if st.dest in ('noreturn', 'fastfail'):
                cur.terminal_detail = st.extra[0]
            cur.succs = [x.strip() for x in st.expr] if st.expr else []
            # **`<infeasible: ...>` is the engine proving the block cannot be reached**, not a gap in
            # the recovery: the path conditions accumulated on the way in are unsatisfiable, so the
            # edge into this block is a contradiction and the block has no successor to transcribe.
            # Marked here and resolved by `_drop_infeasible`, which removes the block *and* every edge
            # into it and records the drop on the function, because a transcription that silently
            # removed a block would be a lie and one that refused the whole function (which is what
            # this used to do) threw away 95 blocks on `ACE-CORE.sys3` root 0x145b97b10 for one
            # unreachable one.
            if cur.term.startswith('<infeasible'):
                cur.infeasible = True
            continue
        if st.kind == 'call-block':
            cur.calls.append((st.text, st.expr))
            continue
        if st.kind == 'branch-cond':
            m = _RE_COND.match(st.text)
            if m and m.group(1) not in ('<none', '<unnamed>'):
                # The name is printed with its width (`v63:32`), the same spelling the body uses for
                # a definition, so the width comes for free rather than being inferred.
                head_, _, tail_ = m.group(1).rpartition(':')
                if head_ and tail_.isdigit():
                    cur.cond, cur.cond_width = head_, int(tail_)
                else:
                    cur.cond = m.group(1)
                    # **A condition that is a flag is one bit wide, and that is not a guess.** The
                    # listing prints `; branch condition: flag_cf` with no width because a flag has no
                    # width to print -- and `whole.cond()` then refused the function with "branch
                    # condition 'flag_cf' has no width": three driver roots lost to a width that is
                    # 1 by definition (`test` consumes bits, and `interp` reads a flag as one bit too).
                    if cur.cond.startswith('flag_') or cur.cond.startswith('init_'):
                        cur.cond_width = 1
            for vm_ in _RE_WHEN.finditer(st.text):
                if vm_.group(1) == '0':
                    cur.when0 = vm_.group(2)
                else:
                    cur.when1 = vm_.group(2)
            mf = _RE_FROM.search(st.text)
            if mf:
                cur.cond_text = mf.group(1)
            continue
        if st.kind == 'entry-note':
            cur.entry_note = st.text
            continue
        cur.stmts.append(st)
    if fn is None:
        fn = Function(root=root, path=path, header=header, unparsed=[])
    if fn.unparsed is None:
        fn.unparsed = []
    _drop_infeasible(fn)
    return fn


def _drop_infeasible(fn: Function) -> None:
    """Remove the blocks the engine proved unreachable, and every edge into them.

    **This is a CFG edit justified by the engine's own proof, and it is recorded.** A block whose exit
    is `<infeasible: the accumulated path conditions are unsatisfiable>` can only be entered by a
    contradicting path, so on every path that can happen the edge is never taken. Dropping it makes a
    forking block a plain jump to its remaining successor -- which is exactly the proven semantics,
    since the dropped side is what the contradiction excludes -- and lets the rest of the function be
    transcribed instead of being refused wholesale.

    What is *not* claimed: the proof itself is not re-checked here. `Function.infeasible_blocks`
    carries the labels and the engine's own text, the payload manifest repeats them, and the
    evidence document names the root they came from, so a reader can see which blocks a recovered
    function is missing rather than discovering it from a statement count.
    """
    dropped = [b for b in fn.blocks if b.infeasible]
    if not dropped:
        return
    gone = {b.label for b in dropped}
    fn.blocks = [b for b in fn.blocks if not b.infeasible]
    fn.infeasible_blocks = [{'label': b.label, 'block': b.index, 'reason': b.term,
                             'statements': len(b.stmts), 'label_text': b.label}
                            for b in dropped]
    for b in fn.blocks:
        if not b.succs:
            continue
        kept = [s for s in b.succs if s not in gone]
        if len(kept) != len(b.succs):
            # An edge to a provably unreachable block: the branch becomes a jump to the other side.
            b.succs = kept
            if b.when0 in gone:
                b.when0 = ''
            if b.when1 in gone:
                b.when1 = ''
            b.cond = '' if not (b.when0 and b.when1) else b.cond
            b.cond_width = b.cond_width if b.cond else 0
    for i, b in enumerate(fn.blocks):
        if b.source_index is None:
            b.source_index = b.index
        b.index = i


def _parse_stmt(body: str):
    s = body.strip()
    if s.startswith(';'):
        tag = s[1:].strip()
        if tag.startswith('entry state'):
            return Stmt(kind='entry-note', text=tag)
        if tag.startswith('branch condition:'):
            return Stmt(kind='branch-cond', text=tag)
        return Stmt(kind='comment', text=tag)
    m = _RE_TERM.match(s)
    if m:
        rest = m.group(1).strip()
        terminal = re.fullmatch(r'<(noreturn|fastfail):\s*(.*)>', rest)
        if terminal:
            return Stmt(kind='term', dest=terminal.group(1), text=s,
                        extra=(terminal.group(2),))
        if rest.startswith('<tail'):
            # `-> <tail: tail call to ntoskrnl.exe!PsTerminateSystemThread>`: the block hands control
            # to a named callee that returns to *our* caller, so the recovered function **ends** here.
            # Spelled as its own term so a reader can tell a tail exit from a `ret`, and treated as a
            # terminal by every consumer (`whole._terminator`, `interp.run_function`).
            return Stmt(kind='term', dest='tail', text=s)
        if rest.startswith('<'):
            return Stmt(kind='term', dest=rest, text=s)
        if rest == 'return':
            return Stmt(kind='term', dest='return', text=s)
        return Stmt(kind='term', dest='goto', text=s,
                    expr=tuple(x.strip() for x in rest.split(',')))
    head, comment = (_RE_SPLIT.split(s, maxsplit=1) + [''])[:2]
    head = head.strip()
    # **A bare value with only a comment after it is a memory-forwarding note, not a statement.**
    #
    # `#0:callr8#3   ; read from guest memory @0x145b9e338   [guest_rax]` is what `core/lower.py`
    # prints when a guest load reads an address it has already seen a store to: the load result *is*
    # that stored value, so the engine names it and the following statement uses the name
    # (`guest_r9_19 = #0:callr8#3`). Nothing is assigned on the line, so there is nothing to transcribe
    # -- and the value it names is already a value this parser understands (`callr8#3` with a
    # cross-block prefix). Treating it as an unknown line is what made `ACE-CORE.sys3` root
    # 0x145b9dbfc refuse whole: **one** forwarding note, 604 statements.
    if comment.strip() and '=' not in head and head and not head.startswith('->'):
        if re.match(r'^(?:#\d+:)?[\w.#]+$', head):
            return Stmt(kind='forward-note', text=s, dest=head.strip())
    # **`ld:<site> reads [ <address> ] [ guest registers: … ]` is an annotation of a read, not a
    # statement.** The engine prints it for a guest memory read it does not bind to a value (the value
    # is not used by anything the listing later names), so there is nothing to assign. Treating it as an
    # unknown line cost ten driver roots, 604 statements among them. A read has no effect in the sealed
    # model -- `interp.Mem.load` and `tvm_ld<bits>` both only return a value -- so dropping the
    # annotation cannot change what the recovered function does; it only says where the guest looked.
    if re.match(r'^ld:0x[0-9a-f]+\s+reads\s+\[', head):
        return Stmt(kind='read-note', text=s, dest=head.split()[0])
    m = _RE_CALLBLOCK.match(head)
    if m:
        return Stmt(kind='call-block', text=m.group(1), expr=int(m.group(2), 16))
    slot_off = slot_reg = None
    ms = _RE_SLOT.search(head)
    if ms:
        # **The offset is the machine fact; the register name is what the derived layout calls it.**
        # `[ctx+0x04 = rax+4]` is a byte slice four bytes into `rax`'s slot, and the two are not
        # interchangeable, so both are kept and the offset is what the emitter uses.
        slot_off = int(ms.group(1), 16)
        slot_reg = ms.group(2)
        head = head[:ms.start()].strip()
    tag = comment.strip()
    # A bracket **in the head** describes the address operand of this very line; a bracket in the
    # trailing comment describes where a value already expressed through a register name came from.
    # Merging the two is a real bug and this parser had it: `v132:32 = load32 [0x7ffffffef740]
    # [ctx+0x80 = rflags]` is a *context* read, and emitting it as an absolute guest-memory load
    # pointed the transposed program at the wrong object entirely.
    tag_off = tag_reg = None
    if slot_off is None:
        mt = re.search(r'\[ctx\+(0x[0-9a-f]+)(?: = (\w+))?\]', tag)
        if mt:
            tag_off = int(mt.group(1), 16)
            tag_reg = mt.group(2)
    m = _RE_STORE.match(head)
    if m:
        bits = int(m.group(1))
        addr = parse_expr(m.group(2))
        value = parse_expr(m.group(3))
        rel = re.search(r'\[(guest_[a-z0-9]+(?:[+-]0x[0-9a-f]+)?)\]', tag)
        return Stmt(kind='store', width=bits, expr=(addr, value),
                    extra=(rel.group(1) if rel else '', _site_of(tag)), text=s)
    m = _RE_DEF.match(head)
    if m:
        dest, width, rhs = m.group(1), m.group(2), m.group(3).strip()
        width = int(width) if width else 0
        # A call line carries its site *after* the argument list with no semicolon:
        # `v196:64 = call 0x14000d4e0(rcx=in.rcx, ...)   @0x14014c690`. Strip it into the site and
        # leave the expression clean, or `parse_expr` sees trailing text.
        msite = re.search(r'\s+@(0x[0-9a-f]+)$', rhs)
        site = 0
        if msite:
            site = int(msite.group(1), 16)
            rhs = rhs[:msite.start()].strip()
        e = parse_expr(rhs)
        if not site:
            site = _site_of(tag)
        if slot_reg is None:
            # A `PARTIAL write to ctx` line carries no `[ctx+0xNN = reg]` bracket -- the listing only
            # brackets the full-slot writes. The register is still named in the destination
            # (`guest_rflags_1`), and the offset then comes from the derived layout, which is the same
            # table the bracket would have been read from.
            reg = _reg_of_dest(dest)
            if reg in CTX_LAYOUT:
                slot_reg, slot_off = reg, CTX_LAYOUT[reg]
        if tag.startswith('write to ctx'):
            return Stmt(kind='ctx-write', dest=dest, width=64, expr=e,
                        extra=(slot_reg, slot_off, site), text=s)
        if tag.startswith('PARTIAL write to ctx'):
            return Stmt(kind='ctx-partial', dest=dest, width=64, expr=e,
                        extra=(slot_reg, slot_off, site), text=s)
        if tag.startswith('RFLAGS bits'):
            return Stmt(kind='ctx-write', dest=dest, width=64, expr=e,
                        extra=('rflags', CTX_LAYOUT['rflags'], site), text=s)
        if tag.startswith('home '):
            reg = _reg_of_dest(dest)
            return Stmt(kind='home', dest=dest, width=width or 64, expr=e,
                        extra=(reg, CTX_LAYOUT.get(reg), 0), text=s)
        if ms is not None and _RE_LOAD.match(rhs):
            # `v132:32 = load32 [0x7ffffffef740]   [ctx+0x80 = rflags]` -- a load whose address the
            # line itself brackets as a context slot. The width is the load's, the offset the slot's.
            bits = int(_RE_LOAD.match(rhs).group(1))
            return Stmt(kind='ctx-load', dest=dest, width=bits, expr=('const', bits, None),
                        extra=(slot_reg, slot_off, site), text=s)
        origin = ('ctx-read' if tag.startswith('read from ctx') else
                  'guest-read' if tag.startswith('guest memory read') else
                  'flags' if tag.startswith('RFLAGS') else
                  'history' if re.search(r'@m\d+$', rhs) else
                  'forward' if tag.startswith('from guest_') else
                  'entry' if tag.startswith('block entry backing') else '')
        return Stmt(kind='def', dest=dest, width=width or 0, expr=e, origin=origin, text=s)
    m = re.compile(r'^(?:#\d+:)?([A-Za-z_][\w.]*?)(?::(\d+))?$').match(head)
    if m:
        return Stmt(kind='bare', dest=m.group(1), expr=None, origin=tag, text=s)
    if re.match(r'^0x[0-9a-f]+$', head):
        return Stmt(kind='bare', dest=head, expr=('const', int(head, 16), None),
                    origin=tag, text=s)
    return None


def _site_of(tag: str) -> int:
    m = re.search(r'@(0x[0-9a-f]+)', tag)
    return int(m.group(1), 16) if m else 0


def _reg_of_slot(slot: str | None, tag: str) -> str:
    m = re.search(r'\[ctx\+0x[0-9a-f]+ = (\w+)', tag)
    if m:
        return m.group(1)
    m = re.search(r'@(0x[0-9a-f]+)', tag)
    return ''


def _guest_width(name: str) -> int:
    tail = name[len('guest_'):] if name.startswith('guest_') else name
    head, _, last = tail.rpartition('_')
    return 64 if last.isdigit() and head in CTX_LAYOUT else 0


def _reg_of_dest(dest: str) -> str:
    """The guest register a destination name names: `guest_rflags_1` -> `rflags`."""
    if not dest.startswith('guest_'):
        return ''
    tail = dest[len('guest_'):]
    head, _, last = tail.rpartition('_')
    if last.isdigit() and head in CTX_LAYOUT:
        return head
    return tail if tail in CTX_LAYOUT else ''


def _full_reg(short: str) -> str:
    """`cx` -> `rcx`, `sp` -> `rsp`, `8` -> `r8`, and a name that is already full stays put."""
    if short in CTX_LAYOUT:
        return short
    if ('r' + short) in CTX_LAYOUT:
        return 'r' + short
    return short


# --------------------------------------------------------------------------- features
#
# Which capabilities a block needs from the backend. Attributed per block so that "can this be
# emitted" is a statement about a population rather than about a listing.


def features(block: Block) -> dict:
    f = {k: 0 for k in ('ctx', 'home', 'guest_mem', 'history', 'call', 'callreg', 'flagbit',
                        'undef', 'witness', 'select', 'narrow', 'pack', 'insert', 'term_multi',
                        'term_unresolved')}

    def walk(e):
        if not isinstance(e, tuple):
            return
        op = e[0]
        if op in ('slot', 'entry'):
            f['ctx'] += 1
        elif op == 'reg':
            f['ctx'] += 1
        elif op == 'callreg':
            f['callreg'] += 1
        elif op == 'flagbit':
            f['flagbit'] += 1
        elif op == 'undef':
            f['undef'] += 1
        elif op == 'witness':
            f['witness'] += 1
        elif op == 'sel':
            f['select'] += 1
        elif op == 'load':
            f['guest_mem'] += 1
        elif op == 'hload':
            f['history'] += 1
        elif op == 'pack':
            f['pack'] += 1
        elif op == 'insert':
            f['insert'] += 1
        elif op == 'call':
            f['call'] += 1
        elif op == 'cast':
            if op == 'trunc':
                pass
            walk(e[2])
            return
        for sub in e[1:]:
            if isinstance(sub, tuple):
                walk(sub)
            elif isinstance(sub, list) or isinstance(sub, tuple):
                for x in sub:
                    if isinstance(x, tuple):
                        walk(x)

    for st in block.stmts:
        if st.kind == 'ctx-write' or st.kind == 'ctx-partial':
            f['ctx'] += 1
        if st.kind == 'home':
            f['home'] += 1
        if st.kind == 'store':
            f['guest_mem'] += 1
        if isinstance(st.expr, tuple) and st.expr and isinstance(st.expr[0], str):
            walk(st.expr)
        elif isinstance(st.expr, tuple):
            for part in st.expr:
                walk(part)
    if len(block.succs) > 1:
        f['term_multi'] = len(block.succs)
        if block.branches:
            # A fork whose condition and 0/1 association the listing states is *core*: it is an
            # `icmp` and a `select` over a temporary the body already defines.
            f['branch'] = 1
            f.pop('term_multi', None)
    if block.term.startswith('<'):
        f['term_unresolved'] = 1
    return {k: v for k, v in f.items() if v}


# --------------------------------------------------------------------------- xstate independence


def _render_canonical(e) -> str:
    """Render an expression with every analysis-supplied name erased."""
    if not isinstance(e, tuple):
        return repr(e)
    if not e:
        return '()'
    if not isinstance(e[0], str):
        # An argument list or a flag pair: a plain sequence, not an expression.
        return '(' + ', '.join(_render_canonical(x) for x in e) + ')'
    op = e[0]
    if op == 'reg':
        # **The version number is dropped here and nowhere else.** `guest_rax_1` and `guest_rax_9`
        # are the same machine access -- a load of the slot the derived context layout calls `rax` --
        # and only `core/xstate.py` distinguishes them.
        return f'reg:{e[1]}'
    if op == 'temp':
        return f'tmp:{e[1]}'
    parts = [str(op)]
    for x in e[1:]:
        if isinstance(x, tuple):
            parts.append(_render_canonical(x))
        else:
            parts.append(repr(x))
    return '(' + ' '.join(parts) + ')'


def canonical(fn: Function) -> str:
    """The parsed structure with every analysis-supplied name erased.

    Register **version numbers** and block-owner prefixes are dropped, because a transcription that
    depends on them depends on `core/xstate.py`, which the listing itself disclaims. What is left is
    the machine's own record: slot offsets, widths, sites, literals and operations.
    """
    out = [f'root {fn.root:#x}']
    for b in fn.blocks:
        out.append(f'block {b.label} [{b.kind}] term={b.term} succs={b.succs} '
                   f'cond={b.cond} w0={b.when0} w1={b.when1}')
        for st in b.stmts:
            out.append(f'  {st.kind} {st.width} {_render_canonical(st.expr)} {st.extra}')
    return '\n'.join(out)


_RE_VERSION = re.compile(r'\bguest_([a-z0-9]+)_(\d+)\b')


def _permute_versions(text: str) -> str:
    """Rename every `guest_<reg>_<k>` to `guest_<reg>_<f(k)>`, consistently within the file."""
    seen: dict[tuple[str, int], int] = {}

    def sub(m):
        reg, k = m.group(1), int(m.group(2))
        if (reg, k) not in seen:
            # A permutation that is not order-preserving and not the identity, so an accidental
            # dependence on the numbering cannot survive it.
            seen[(reg, k)] = (k * 7 + 13) % 97 + 1000
        return f'guest_{reg}_{seen[(reg, k)]}'

    return _RE_VERSION.sub(sub, text)


def version_invariance(text: str) -> tuple[bool, str]:
    """Parse the listing, then parse it again with every version number permuted, and compare.

    This is the falsifier for the claim in the module docstring. If the two canonical forms differ,
    something in the transcription is reading a name that came from the propagated register file
    rather than from the machine -- and the discrepancy is printed rather than tolerated.
    """
    a = parse_listing(text)
    b = parse_listing(_permute_versions(text))
    ca, cb = canonical(a), canonical(b)
    if ca == cb:
        return True, ''
    al, bl = ca.splitlines(), cb.splitlines()
    for i, (x, y) in enumerate(zip(al, bl)):
        if x != y:
            return False, f'line {i}:\n  with version numbers: {x}\n  with them permuted:  {y}'
    return False, f'different lengths: {len(al)} vs {len(bl)}'


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    fn = parse_listing(path.read_text(encoding='utf-8', errors='replace'), path)
    total = sum(len(b.stmts) for b in fn.blocks)
    print(f'{path.name}  root {fn.root:#x}  ctx {fn.ctx_base:#x}  '
          f'blocks {len(fn.blocks)}  statements {total}  unparsed {len(fn.unparsed)}')
    for b in fn.blocks:
        f = features(b)
        print(f'  #{b.index:<3} {b.label:<40} [{b.kind}] {len(b.stmts):>3} stmt  '
              f'{b.term} {b.succs if b.succs else ""}  {f}')
    if fn.unparsed:
        print('UNPARSED:')
        for u in fn.unparsed[:20]:
            print('   ', u)
    if '--json' in argv:
        print(json.dumps({'root': fn.root, 'blocks': [
            {'label': b.label, 'kind': b.kind, 'stmts': len(b.stmts),
             'features': features(b), 'succs': b.succs, 'term': b.term}
            for b in fn.blocks]}, indent=2))
    return 0 if not fn.unparsed else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
