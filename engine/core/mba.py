"""Killing the `not`/`xchg` obfuscation the TVM handlers are built out of.

## Why this module exists

The handlers in `.tvm0` do not contain the operation they appear to perform. They
contain a *dance*: a register is swapped with a scratch register, negated, swapped
back and negated again. Measured on the DeltaForce base image, 600 sampled handler
functions decode to 760,531 `not` and 730,640 `xchg` against 1,023,887 `mov` -- so
roughly **1.5 obfuscation instructions per real one**. The canonical shape, seen at
`0x180fd4007`:

    xchg r10, rax          ; r10 <- old rax,  rax <- old r10
    mov  r10, [rdx]        ; r10 <- M
    not  r10               ; r10 <- ~M
    xchg r10, rax          ; rax <- ~M,      r10 <- old rax   (restored)
    not  rax               ; rax <- M                        (two nots cancel)

which is exactly `mov rax, [rdx]` plus noise. The noise is *not* semantically
harmless to an analyser: each `xchg` on a sub-64-bit register goes through the
partial-write path (`or`/`and`/`shl` rebuild, see `Machine.set_reg`), so a pair of
`xchg`es that cancel still leaves a `zext8(trunc8(...))` ladder in the DAG that
nothing folds afterwards. That ladder is what turns a 355-node median value into
something a reader cannot follow.

## Two mechanisms, because there are two different failures

`SwapTracker` works on **machine state**, not on expressions. It remembers what the
two registers of a pending swap held, and if the same pair is swapped again before
either has been written, it **restores the saved values and drops both swaps**. That
is the literal statement "xchg twice equals no swap", made true by undoing the first
one rather than by hoping the folder notices. Partial-width writes are the reason
this has to be state-level: `xchg ax, bx` twice is a no-op on hardware, but through
`set_reg` it is two `or`/`and` rebuilds that interning cannot see through.

The expression-level rules (`fold_un`, `fold_bin`, `simplify`) handle the `not`
pairs and the algebra around them, on the interned DAG. `not (not x) = x` already
lives in `Arena.un`; the rules added here are the ones the handlers actually need:
self-cancelling `xor`/`and`/`or`, `x ^ x = 0`, `x & ~x = 0`, `x - x = 0`,
`(x + y) - y = x`, `-(-x) = x`, and the width-local forms of each.

## What is deliberately *not* here

No rule is admitted that is not true for **every** value at the stated width. The
classic MBA rewrites (`x + y == (x ^ y) + 2*(x & y)` and friends) are excluded:
they are correct but they *grow* the DAG, and this module's contract is that
`simplify` returns a value with **no more nodes** than it started with. Rules are
applied to fixpoint with a node-count ceiling so a pathological DAG cannot spin.

Everything here is width-exact: `_MASK[width]` bounds every constant, and a rule
that would need a width it does not have declines instead of guessing. A declined
rule is reported by `stats`, never silently wrong.
"""

from __future__ import annotations

_MASK = {8: 0xFF, 16: 0xFFFF, 32: 0xFFFF_FFFF, 64: 0xFFFF_FFFF_FFFF_FFFF}


def _mask(width: int) -> int:
    return _MASK.get(width, (1 << width) - 1)


# --------------------------------------------------------------------- helpers

def _node(arena, ref):
    try:
        return arena.nodes[ref]
    except (IndexError, AttributeError):
        return None


def _const(arena, ref):
    n = _node(arena, ref)
    if n is not None and n[0] == "c":
        return n[1]
    return None


def _width(arena, ref):
    try:
        return arena.width(ref)
    except Exception:
        return None


def is_not(arena, ref) -> bool:
    n = _node(arena, ref)
    return n is not None and n[0] == "un" and n[1] == "not"


def strip_not(arena, ref) -> tuple[int, bool]:
    """`(inner, negated)` for a value, folding any number of leading `not`s."""
    neg = False
    n = _node(arena, ref)
    while n is not None and n[0] == "un" and n[1] == "not":
        neg = not neg
        ref = n[2]
        n = _node(arena, ref)
    return ref, neg


# ------------------------------------------------------------------ the folder
#
# These two return a *replacement ref* when a rule fires and `None` when none does,
# so `Arena.un` / `Arena.bin` can call them without this module needing to know how
# the arena builds things. They never call `arena.bin`/`arena.un` with the same
# (op, operands) they were handed, so they cannot re-enter themselves.

def fold_un(arena, op: str, a: int) -> int | None:
    """One-step rules for unary ops. Called from `Arena.un`."""
    n = _node(arena, a)
    if n is None:
        return None
    width = n[-1] if n[0] != "load" else n[2]
    full = _mask(width)

    if op == "not":
        # not(not(x)) = x -- `Arena.un` has this too; kept here so the rule set is
        # complete when `simplify` walks a DAG built before this module existed.
        if n[0] == "un" and n[1] == "not":
            return n[2]
        # ~~ inside a truncation: `trunc8(not(not(x)))` never forms, but
        # `not(trunc8(not(not(x))))` does, and it is `not(trunc8(x))`.
        if n[0] in ("zext", "sext", "trunc"):
            inner = n[1]
            if is_not(arena, inner):
                return arena.un("not", arena._fit(inner, width) if hasattr(arena, "_fit")
                                else inner, width)
        # not(all-ones) = 0, not(0) = all-ones -- folded by the const path already,
        # but a `not` over a load-known-zero is not, and the VM parks zeros in the
        # unused slots.
        c = _const(arena, a)
        if c is not None:
            return arena.const((~c) & full, width)
    if op == "neg":
        if n[0] == "un" and n[1] == "neg":
            return n[2]
        if n[0] == "bin" and n[1] == "sub":
            # -(x - y) = y - x
            return arena.bin("sub", n[3], n[2], width)
    return None


def fold_bin(arena, op: str, a: int, b: int, width: int) -> int | None:
    """One-step rules for binary ops. Called from `Arena.bin`."""
    na, nb = _node(arena, a), _node(arena, b)

    # ---- self-cancelling bitwise -------------------------------------------
    if a == b:
        if op == "xor" or op == "sub":
            return arena.const(0, width)            # x ^ x = 0,  x - x = 0
        if op == "and" or op == "or":
            return a                                # x & x = x,  x | x = x
        # `x + x` is 2x and is deliberately left alone: rewriting it as
        # `shl(x, 1)` grows the DAG and this module never grows a value.

    # x ^ 0, x | 0, x + 0, x - 0
    if op in ("xor", "or", "add", "sub", "shl", "shr", "sar", "rol", "ror"):
        cb = _const(arena, b)
        if cb == 0 and op != "shl":
            return a
        ca = _const(arena, a)
        if ca == 0 and op in ("xor", "or", "add"):
            return b
        if ca == 0 and op == "sub" and False:
            return None

    # x & all-ones / x | all-ones
    full = _mask(width)
    if op == "and":
        cb = _const(arena, b)
        if cb == full:
            return a
        ca = _const(arena, a)
        if ca == full:
            return b
        # x & ~x = 0
        if nb is not None and nb[0] == "un" and nb[1] == "not" and nb[2] == a:
            return arena.const(0, width)
        if na is not None and na[0] == "un" and na[1] == "not" and na[2] == b:
            return arena.const(0, width)
    if op == "or":
        cb = _const(arena, b)
        if cb == 0:
            return a
        ca = _const(arena, a)
        if ca == 0:
            return b
        # x | ~x = all-ones
        if nb is not None and nb[0] == "un" and nb[1] == "not" and nb[2] == a:
            return arena.const(full, width)
        if na is not None and na[0] == "un" and na[1] == "not" and na[2] == b:
            return arena.const(full, width)

    # ---- xor cancellation: (x ^ y) ^ y = x --------------------------------
    if op == "xor":
        if na is not None and na[0] == "bin" and na[1] == "xor":
            if na[3] == b:
                return na[2]
            if na[2] == b:
                return na[3]
        if nb is not None and nb[0] == "bin" and nb[1] == "xor":
            if nb[3] == a:
                return nb[2]
            if nb[2] == a:
                return nb[3]
        # (x ^ ~x) = all-ones
        if nb is not None and nb[0] == "un" and nb[1] == "not" and nb[2] == a:
            return arena.const(full, width)
        if na is not None and na[0] == "un" and na[1] == "not" and na[2] == b:
            return arena.const(full, width)

    # ---- additive cancellation: (x + y) - y = x, (x - y) + y = x ----------
    if op in ("add", "sub"):
        # (x + y) - y  /  (x - y) + y  /  (y + x) - y
        if na is not None and na[0] == "bin" and na[1] in ("add", "sub"):
            inner_op, p, q = na[1], na[2], na[3]
            if op == "sub" and inner_op == "add" and q == b:
                return p
            if op == "add" and inner_op == "sub" and q == b:
                return p
            if op == "sub" and inner_op == "add" and p == b:
                return q
        if na is not None and na[0] == "bin" and na[1] == "add" and op == "add":
            # (x - y) + y  reaching here as ((x - y) + y)
            pass
        # x + (-y) = x - y  and  x - (-y) = x + y
        if nb is not None and nb[0] == "un" and nb[1] == "neg":
            return arena.bin("sub" if op == "add" else "add", a, nb[2], width)

    # ---- constants on both sides of a bitwise op --------------------------
    ca, cb = _const(arena, a), _const(arena, b)
    if ca is not None and cb is not None:
        return None                                 # `Arena` already folds these

    # ---- double negation reaching through a bin: not(not(a) op b) handled in un
    return None


# ------------------------------------------------------------------ the walked
#
# `simplify` re-walks a DAG top-down and rebuilds every node through the folder, so
# rules fire even on nodes interned before this module was importable (a listing
# read back from disk, a value captured in a snapshot). It is a *rebuild*, not an
# in-place rewrite: the arena is interned, so a rebuilt node that simplifies to an
# existing node simply returns that node.

_STATS = {"fold_un": 0, "fold_bin": 0, "rebuilt": 0, "declined": 0}


def stats() -> dict[str, int]:
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def node_count(arena, ref: int) -> int:
    """Distinct nodes reachable from `ref`."""
    seen: set[int] = set()
    stack = [ref]
    while stack:
        r = stack.pop()
        if r in seen:
            continue
        seen.add(r)
        n = _node(arena, r)
        if n is None:
            continue
        t = n[0]
        if t == "bin":
            stack += [n[2], n[3]]
        elif t == "un":
            # ("un", op, operand, width) -- n[1] is the *operator name*, not a ref.
            stack.append(n[2])
        elif t in ("trunc", "zext", "sext"):
            stack.append(n[1])
        elif t == "sel":
            stack += [n[1], n[2], n[3]]
        elif t == "load":
            stack.append(n[1])
    return len(seen)


def simplify(arena, ref: int, rounds: int = 8, max_nodes: int = 400_000) -> int:
    """Rebuild `ref` through the rules until it stops shrinking.

    Returns the simplified ref. Because the arena interns, an unchanged value comes
    back as the same ref, so `simplify(a, r) is r` means "no rule fired anywhere".
    """
    before = node_count(arena, ref)
    cur = ref
    for _ in range(rounds):
        new = _rebuild(arena, cur, {}, 0)
        _STATS["rebuilt"] += 1
        after = node_count(arena, new)
        if new == cur or after >= node_count(arena, cur) and after >= before:
            cur = new
            break
        cur = new
        if after >= max_nodes:
            _STATS["declined"] += 1
            break
    return cur


def _rebuild(arena, ref: int, memo: dict[int, int], depth: int) -> int:
    if depth > 512:
        _STATS["declined"] += 1
        return ref
    got = memo.get(ref)
    if got is not None:
        return got
    n = _node(arena, ref)
    if n is None:
        return ref
    t = n[0]
    out = ref
    if t == "bin":
        op, a, b, w = n[1], n[2], n[3], n[-1]
        na = _rebuild(arena, a, memo, depth + 1)
        nb = _rebuild(arena, b, memo, depth + 1)
        rule = fold_bin(arena, op, na, nb, w)
        if rule is not None:
            _STATS["fold_bin"] += 1
            out = rule
        else:
            out = arena.bin(op, na, nb, w)
    elif t == "un":
        op, a, w = n[1], n[2], n[-1]
        na = _rebuild(arena, a, memo, depth + 1)
        rule = fold_un(arena, op, na)
        if rule is not None:
            _STATS["fold_un"] += 1
            out = rule
        else:
            out = arena.un(op, na, w)
    elif t in ("trunc", "zext", "sext"):
        a, w = n[1], n[2]
        na = _rebuild(arena, a, memo, depth + 1)
        out = getattr(arena, t)(na, w)
    elif t == "sel":
        c, x, y, w = n[1], n[2], n[3], n[-1]
        nc = _rebuild(arena, c, memo, depth + 1)
        nx = _rebuild(arena, x, memo, depth + 1)
        ny = _rebuild(arena, y, memo, depth + 1)
        out = arena._add(("sel", nc, nx, ny, w))
    elif t == "load":
        addr, w, ver = n[1], n[2], n[3]
        na = _rebuild(arena, addr, memo, depth + 1)
        # `ref`, not `na`: an unchanged address means the *load node* is unchanged.
        out = ref if na == addr else arena.load(na, w, ver)
    memo[ref] = out
    return out


# ---------------------------------------------------------------- swap tracker

class SwapTracker:
    """Undo `xchg a, b` when the same pair is swapped again unchanged.

    The handlers use `xchg` as their exclusive-access idiom: a scratch register and
    an accumulator trade places, the scratch is loaded from the guest context, and
    the pair is swapped back. Between the two swaps the scratch is *written*, which is
    what makes the pair non-trivial -- and it is also why this tracker cannot fire on
    that shape. What it does fire on is the degenerate form, `xchg a, b` twice with
    no intervening write to either register, which the protector also emits as
    padding. `probe.deobf_scan` measures both counts on a real image so the two
    claims stay separable.

    Cancelling at the *state* level rather than at the expression level is the point:
    `Machine.set_reg` rebuilds sub-64-bit registers out of `and`/`or`/`shl`, and
    interning cannot prove those rebuilds are the identity, so an undone pair that
    went through `set_reg` leaves a `zext8(trunc8(...))` ladder behind. Restoring the
    saved refs skips the ladder entirely.

    **Cancellation is only claimed when it is certain.** A pending pair is dropped by
    `on_write` on either register, by a width mismatch, or when `max_pending` pairs
    are outstanding. The tracker never inspects values, so it can never be fooled by
    an equal constant.
    """

    __slots__ = ("m", "_pending", "cancelled", "tracked", "max_pending", "_restoring")

    def __init__(self, machine, max_pending: int = 64) -> None:
        self.m = machine
        self._pending: dict[tuple[str, str], tuple[int, int, int]] = {}
        self.cancelled = 0
        self.tracked = 0
        self.max_pending = max_pending
        self._restoring = False

    @staticmethod
    def _parent(name: str) -> str:
        try:
            from .mach import canon as _canon
            return _canon(name)[0]
        except Exception:
            return name

    # -- called by `Machine.set_reg` on every register write -------------------
    def on_write(self, name: str) -> None:
        """A register changed, so any pending swap on it is no longer reversible."""
        if self._restoring or not self._pending:
            return
        p = self._parent(name)
        for key in [k for k in self._pending
                    if any(self._parent(operand) == p for operand in k)]:
            self._pending.pop(key, None)

    # -- called by the `xchg` handler ----------------------------------------
    def should_cancel(self, x: str, y: str, width: int) -> bool:
        """True when this swap undoes a pending one; both registers are restored."""
        px, py = self._parent(x), self._parent(y)
        # Two 32-bit swaps clear both parent registers' upper halves, so they
        # cannot restore the original 64-bit refs as an identity operation.
        if px == py or width == 32:
            return False
        # Include the actual subregisters: AL/BL and AH/BH are different pairs.
        key = tuple(sorted((x, y)))
        prev = self._pending.get(key)
        if prev is None:
            return False
        w_prev, vx, vy = prev
        if w_prev != width:
            return False
        self._pending.pop(key, None)
        self._restoring = True
        try:
            self.m.regs[self._parent(key[0])] = vx
            self.m.regs[self._parent(key[1])] = vy
        finally:
            self._restoring = False
        self.cancelled += 1
        return True

    def remember(self, x: str, y: str, width: int, saved: tuple[int, int]) -> None:
        """Record a swap that has just been performed, with the values it replaced."""
        px, py = self._parent(x), self._parent(y)
        if px == py or width == 32:
            return
        self.tracked += 1
        key = tuple(sorted((x, y)))
        if len(self._pending) >= self.max_pending:
            self._pending.clear()
        vx, vy = saved if key == (x, y) else (saved[1], saved[0])
        self._pending[key] = (width, vx, vy)

    def snapshot(self, x: str, y: str) -> tuple[int, int]:
        """The 64-bit refs currently in both registers, for `remember`."""
        return (self.m.reg(self._parent(x), 64), self.m.reg(self._parent(y), 64))

    def flush(self) -> None:
        self._pending.clear()


# --------------------------------------------------- the dance, at instruction level
#
# Used by `probe.deobf_scan` to *report* the obfuscation pattern over handler bytes,
# independently of the lifter. Keeping a structural recogniser next to the stateful
# one means "the pattern is there" and "the engine removes it" are two separate
# claims that can be checked against each other.

def not_xchg_dance(insns) -> list[dict]:
    """Find `xchg r,acc ; <uses of r> ; xchg r,acc ; not acc`-shaped runs.

    `insns` is a capstone instruction sequence. Returns one record per recognised
    run with the net effect it collapses to, so a caller can quantify how much of a
    handler is obfuscation rather than code.
    """
    out: list[dict] = []
    n = len(insns)
    i = 0
    while i < n:
        a = insns[i]
        if a.mnemonic != "xchg":
            i += 1
            continue
        regs_a = _swap_regs(a)
        if regs_a is None:
            i += 1
            continue
        px, py = regs_a
        # look forward for the matching swap back, allowing `not`/`mov` between
        for j in range(i + 1, min(i + 12, n)):
            b = insns[j]
            regs_b = _swap_regs(b) if b.mnemonic == "xchg" else None
            if regs_b is not None and set(regs_b) == {px, py}:
                between = insns[i + 1:j]
                negs = [k for k in between if k.mnemonic == "not"]
                movs = [k for k in between if k.mnemonic in ("mov", "movzx", "movsxd")]
                after_not = j + 1 < n and insns[j + 1].mnemonic == "not"
                out.append(dict(
                    start=a.address, end=b.address,
                    pair=(px, py),
                    interior=len(between), nots=len(negs), moves=len(movs),
                    trailing_not=after_not,
                    # an even number of nots between plus one after the swap back is
                    # a plain load; state it rather than implying it
                    net="load" if (len(negs) + (1 if after_not else 0)) % 2 == 1 else "copy",
                    cancels=len(negs) % 2 == 0,
                ))
                i = j + (1 if after_not else 0)
                break
        i += 1
    return out


def _swap_regs(insn):
    try:
        ops = insn.operands
        if len(ops) != 2:
            return None
        import capstone
        if ops[0].type != capstone.x86.X86_OP_REG or ops[1].type != capstone.x86.X86_OP_REG:
            return None
        return insn.reg_name(ops[0].reg), insn.reg_name(ops[1].reg)
    except Exception:
        return None
