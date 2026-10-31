"""Hash-consed expression DAG: the algebra that makes MBA obfuscation collapse.

The recovery engine never asks "is this expression equal to that one"; it asks
whether two `Ref`s are *the same integer*. Every node is interned, so structural
equality is pointer equality, and `sub(x, x)` folds to zero without any solver.
That single property is what turns the protector's obfuscated dispatch
arithmetic -- which is built entirely out of self-cancelling terms -- into a
constant. The core folder remains solver-free; recovery can additionally use
the optional bounded bit-vector backend in `bitproof.py`.

Three things live here:

* interning plus constant folding, including the absorption and De Morgan
  identities the protector's register-shuffling idiom produces;
* per-node known-bits (`zbits`, `obits`), which is how the engine decides that a
  subexpression is a predicate rather than data;
* a structural hash, which is how blocks are identified later.

Widths are tracked because x86 sub-register writes are not width-agnostic.
"""

from __future__ import annotations

import sys
from typing import Iterable

sys.setrecursionlimit(200_000)

# --------------------------------------------------------------------- widths

W8, W16, W32, W64 = 8, 16, 32, 64
_MASK = {W8: 0xFF, W16: 0xFFFF, W32: 0xFFFFFFFF, W64: 0xFFFFFFFFFFFFFFFF}
_SIGN = {W8: 1 << 7, W16: 1 << 15, W32: 1 << 31, W64: 1 << 63}

# Binary operators. `udiv`/`urem` are kept symbolic on purpose: the VM's own
# arithmetic folds, and a guest division is not a dispatch input.
BIN_OPS = frozenset(
    {"add", "sub", "mul", "and", "or", "xor", "shl", "shr", "sar", "udiv", "urem",
     "eq", "ne", "ult", "ule", "slt", "sle"}
)
UN_OPS = frozenset({"not", "neg"})

# Operations whose result is provably 0 or 1 regardless of the operands.
_BOOL_RESULT_OPS = frozenset({"eq", "ne", "ult", "ule", "slt", "sle"})


def mask(width: int) -> int:
    return _MASK[width]


def sign_bit(width: int) -> int:
    return _SIGN[width]


def sext_value(value: int, width: int) -> int:
    """`value` reinterpreted as signed at `width`, returned as a Python int."""
    m = _MASK[width]
    value &= m
    if value & _SIGN[width]:
        value -= m + 1
    return value


class Arena:
    """Interned expression nodes. `Ref` is an index into `nodes`."""

    def __init__(self) -> None:
        self.nodes: list[tuple] = []
        self._intern: dict[tuple, int] = {}
        self._zbits: list[int] = []
        self._obits: list[int] = []
        self._shash: list[int] = []
        # Re-entry guard for the bit-slicing folder; see `_guarded_slice`.
        self._slice_depth = 0
        # Optional `(address, width) -> Ref | None` hook that supplies the value
        # of a read at a *constant* address. The arena has no image and no memory
        # to answer that question with; the recovery driver installs a hook that
        # does, under the same rule the read site uses. See `_fold_known_load`.
        self.load_folder = None
        self.versioned_load_folder = None
        self._memory_serial = 0
        # Persistent write history: each state links to its parent. Forks share
        # history without copying the whole byte map on every symbolic read.
        self.memory_versions: dict[int, tuple] = {}
        self.readonly_writes: list[tuple[int, int]] = []
        # Node -> its rebuild under unchanged operands. See `Arena._normal`.
        self._norm: dict[int, int] = {}
        # Root -> its descendants in topological order. See `_descendants_ordered`.
        self._desc: dict[int, list[int]] = {}
        # Immutable source refs only. None marks a source set above the bound;
        # mutable census labels must always be looked up by the caller afresh.
        self._leaf_memo: dict[int, tuple[int, ...] | None] = {}
        # Node 0 is a convenience zero so that "no value" is never a valid Ref.
        self.zero_w64 = self.const(0, W64)

    # ------------------------------------------------------------ primitives

    def _add(self, node: tuple) -> int:
        got = self._intern.get(node)
        if got is not None:
            return got
        ref = len(self.nodes)
        self.nodes.append(node)
        self._intern[node] = ref
        self._zbits.append(-1)
        self._obits.append(-1)
        self._shash.append(0)
        return ref

    def _rebuild_nofold(self, node: tuple) -> int:
        """Intern a node without re-entering the folder, but still fold it when
        every operand is a constant.

        The bit-slicing paths must not call the folder, because building a node
        lands back in the slicer. But interning blindly is worse than it looks:
        it can wrap a constant in a `trunc`/`zext` that never folds, and that
        wrapper is then a permanent barrier -- everything built on top of it
        stays symbolic forever. That is not a hypothetical: an unfolded
        `zext64(trunc16(0x3d0))` in the VM's register-shuffle path is what kept a
        guest stack pointer symbolic, and through it the whole dispatch.
        """
        t = node[0]
        if t in ("trunc", "zext", "sext"):
            c = self.as_const(node[1])
            if c is not None:
                w = node[2]
                if t == "trunc":
                    return self.const(c & _MASK[w], w)
                if t == "zext":
                    return self.const(c, w)
                return self.const(sext_value(c, self.width(node[1])) & _MASK[w], w)
        elif t == "bin":
            ca, cb = self.as_const(node[2]), self.as_const(node[3])
            if ca is not None and cb is not None:
                return self.const(_fold_bin(node[1], ca, cb, node[-1]), node[-1])
        elif t == "un":
            ca = self.as_const(node[2])
            if ca is not None:
                w = node[-1]
                return self.const((~ca) & _MASK[w] if node[1] == "not" else (-ca) & _MASK[w], w)
        elif t == "sel":
            cc = self.as_const(node[1])
            if cc is not None:
                return self._fit(node[2] if cc else node[3], node[-1])
        elif t == "load":
            # Same rule as `_rebuild`: a read whose address is a constant has a
            # value, and interning it as a load instead is how a correct number
            # becomes a permanent barrier. Reached here only from the bit slicer,
            # which is exactly where such a node would otherwise survive.
            folded = self._fold_known_load(node[1], node[2], node[3])
            if folded is not None:
                return folded
        return self._add(node)

    def node(self, r: int) -> tuple:
        return self.nodes[r]

    def op(self, r: int) -> str:
        return self.nodes[r][0]

    def width(self, r: int) -> int:
        n = self.nodes[r]
        t = n[0]
        if t in ("c", "bin", "un", "trunc", "zext", "sext", "sel"):
            return n[-1]
        if t in ("init", "param", "opaque"):
            return n[-1]
        if t == "load":
            # Fixed index, not `n[-1]`: the node is `("load", addr, width, version)` and the last
            # field is the version. Reading `n[-1]` here would silently return the version as a
            # bit width, which is the class of bug this change exists to remove.
            return n[2]
        raise KeyError(f"node {r} has no width: {n}")

    def __len__(self) -> int:
        return len(self.nodes)

    # -------------------------------------------------------------- builders

    def const(self, value: int, width: int) -> int:
        return self._add(("c", value & _MASK[width], width))

    def init_reg(self, reg: str) -> int:
        return self._add(("init", reg, W64))

    def param(self, blockref: int, reg: str) -> int:
        return self._add(("param", blockref, reg, W64))

    def opaque(self, name: str, width: int = W64) -> int:
        return self._add(("opaque", name, width))

    def load(self, addr: int, width: int, version: int = 0) -> int:
        """A read of `width` bits at `addr`, **at memory version `version`**.

        The version is what makes a read of a mutable location differ from an earlier read of the
        same location: same address and same width do not mean the same value at two different
        times. `Machine` bumps its version on every store, and two reads with no intervening store
        share a version and therefore intern to the same node -- which is the property the
        protector own self-cancelling arithmetic depends on (`load(A) - load(A)` must still fold),
        so this does not cost that.
        """
        return self._add(("load", addr, width, version))

    def new_memory_version(self) -> int:
        """Unique across machines, forks and restored snapshots sharing this arena."""
        self._memory_serial += 1
        return self._memory_serial

    def _fold_known_load(self, addr_ref: int, width: int, version: int = 0) -> int | None:
        """Value of a read whose address has *just become* a constant.

        A `load` node is built once, when the read happens, and at that moment the
        address may still be symbolic -- so the machine correctly records a load
        rather than a value. Substitution can fold the address afterwards, and the
        rebuilt node is then a read at a **constant** address that nothing ever
        folds, because `Arena.load` has no image to fold against.

        That node is a permanent barrier: every expression built on it stays
        symbolic forever, and it is indistinguishable from an honest unknown in
        every diagnostic. It is the same failure mode as the unfolded constant
        wrappers and the untagged bytes before it, and it is what kept a dispatch
        in this module unresolved while the reference resolved it -- the address
        folded to `0x140141c14`, a `.tvm0` code address, and the four bytes there
        were never read.

        The hook is supplied by the driver, which owns the image and the memory
        map, and it returns a value only when the read is one the read site itself
        would have folded.
        """
        if self.load_folder is None and self.versioned_load_folder is None:
            return None
        c = self.as_const(addr_ref)
        if c is None:
            return None
        if self.versioned_load_folder is not None:
            return self.versioned_load_folder(c, width, version)
        return self.load_folder(c, width)

    # --------------------------------------------------------------- queries

    def as_const(self, r: int) -> int | None:
        n = self.nodes[r]
        if n[0] == "c":
            return n[1]
        return None

    def is_const(self, r: int) -> bool:
        return self.nodes[r][0] == "c"

    def is_bool(self, r: int) -> bool:
        """True when every bit except bit 0 is provably zero, i.e. the value is 0 or 1."""
        if self.nodes[r][0] in ("c",):
            return self.nodes[r][1] in (0, 1)
        return (self.zbits(r) | 1) == _MASK[self.width(r)]

    def zbits(self, r: int) -> int:
        """Bits of `r` that are provably zero."""
        cached = self._zbits[r]
        if cached >= 0:
            return cached
        self._zbits[r] = 0  # cycle guard; the DAG is acyclic so this is never re-read
        v = self._compute_zbits(r) & _MASK[self.width(r)]
        self._zbits[r] = v
        return v

    def obits(self, r: int) -> int:
        """Bits of `r` that are provably one."""
        cached = self._obits[r]
        if cached >= 0:
            return cached
        self._obits[r] = 0
        v = self._compute_obits(r) & _MASK[self.width(r)]
        self._obits[r] = v
        return v

    def _compute_zbits(self, r: int) -> int:
        n = self.nodes[r]
        t = n[0]
        w = self.width(r)
        full = _MASK[w]
        if t == "c":
            return (~n[1]) & full
        if t in ("init", "param", "opaque", "load"):
            return 0
        if t == "bin":
            op, a, b = n[1], n[2], n[3]
            za, oa = self.zbits(a), self.obits(a)
            zb, ob = self.zbits(b), self.obits(b)
            if op in _BOOL_RESULT_OPS:
                return full ^ 1
            if op == "and":
                return za | zb
            if op == "or":
                return za & zb
            if op == "xor":
                return (za & zb) | (oa & ob)
            if op in ("shl", "shr", "sar"):
                if self.is_const(b):
                    sh = n[3]
                    k = self.as_const(sh)
                    if k is not None:
                        if k >= w:
                            return full if op != "sar" or za & _SIGN[w] else 0
                        if op == "shl":
                            return ((za << k) | ((1 << k) - 1)) & full
                        if op == "shr":
                            return (za >> k) | ((full << (w - k)) & full)
                        fill = ((full << (w - k)) & full) if za & _SIGN[w] else 0
                        return (za >> k) | fill
                return 0
            if op in ("add", "sub"):
                zero, one = _arithmetic_known_bits(za, oa, zb, ob, w, op == 'sub')
                self._obits[r] = one
                return zero
            if op == "mul":
                return full if (za & zb) == full else 0
            return 0
        if t == "un":
            op, a = n[1], n[2]
            if op == "not":
                return self.obits(a)
            if op == "neg":
                # -x == ~x + 1: zero bits survive only in the all-zero case.
                return 0
            return 0
        if t == "trunc":
            return self.zbits(n[1]) & full
        if t == "zext":
            return self.zbits(n[1]) | (full ^ _MASK[self.width(n[1])])
        if t == "sext":
            inner = self.width(n[1])
            z = self.zbits(n[1])
            # Bits above the source sign bit are all zero or all one.
            if z & _SIGN[inner]:
                return z | (full ^ _MASK[inner])
            return z
        if t == "sel":
            return self.zbits(n[2]) & self.zbits(n[3])
        return 0

    def _compute_obits(self, r: int) -> int:
        n = self.nodes[r]
        t = n[0]
        w = self.width(r)
        full = _MASK[w]
        if t == "c":
            return n[1]
        if t in ("init", "param", "opaque", "load"):
            return 0
        if t == "bin":
            op, a, b = n[1], n[2], n[3]
            za, oa = self.zbits(a), self.obits(a)
            zb, ob = self.zbits(b), self.obits(b)
            if op in _BOOL_RESULT_OPS:
                if op == "eq" and a == b:
                    return 1
                if op == "ne" and a == b:
                    return 0
                if op in ("ule", "sle") and a == b:
                    return 1
                return 0
            if op == "and":
                return oa & ob
            if op == "or":
                return oa | ob
            if op == "xor":
                return (oa & zb) | (za & ob)
            if op in ("shl", "shr", "sar"):
                k = self.as_const(b)
                if k is not None and k >= w:
                    return full if op == 'sar' and oa & _SIGN[w] else 0
                if k is not None and k < w:
                    if op == "shl":
                        return (oa << k) & full
                    if op == "shr":
                        return oa >> k
                    fill = ((full << (w - k)) & full) if oa & _SIGN[w] else 0
                    return (oa >> k) | fill
                return 0
            if op in ('add', 'sub'):
                zero, one = _arithmetic_known_bits(za, oa, zb, ob, w, op == 'sub')
                self._zbits[r] = zero
                return one
            if op == "mul":
                if ob & 1 and self.is_const(a) and self.as_const(a) == 1:
                    return ob
                return 0
            return 0
        if t == "un":
            op, a = n[1], n[2]
            if op == "not":
                return self.zbits(a)
            return 0
        if t == "trunc":
            return self.obits(n[1]) & full
        if t == "zext":
            return self.obits(n[1])
        if t == "sext":
            inner = self.width(n[1])
            o = self.obits(n[1])
            if o & _SIGN[inner]:
                return o | (full ^ _MASK[inner])
            return o
        if t == "sel":
            return self.obits(n[2]) & self.obits(n[3])
        return 0

    # ------------------------------------------------------------- structural

    def structural_hash(self, r: int) -> int:
        """Collision-resistant-enough 64-bit hash of the whole sub-DAG.

        Full depth, memoised per node: this engine refuses the depth-bounded
        variant because a bounded hash merges two different block states into one
        identity, which silently fabricates a back edge.
        """
        cached = self._shash[r]
        if cached:
            return cached
        n = self.nodes[r]
        h = 0xCBF29CE484222325
        for item in n:
            if isinstance(item, int) and item in self._shash and not isinstance(item, bool):
                # A child Ref: mix its hash (already computed, DAG order guarantees this).
                sub = self._shash[item] or self._compute_shash(item)
            else:
                sub = _mix_scalar(item)
            h ^= sub & 0xFFFFFFFFFFFFFFFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        self._shash[r] = h or 1
        return self._shash[r]

    def _compute_shash(self, r: int) -> int:
        self.structural_hash(r)
        return self._shash[r]

    _LEAF_MEMO_LIMIT = 32

    def leaves(self, r: int) -> list[int]:
        """Sources in the original DFS order, with bounded immutable summaries.

        Small source sets are common even in very large arithmetic DAGs. Build
        their summaries bottom-up so a new parent reuses its cached children.
        A wide subtree keeps only an overflow marker and uses the original DFS;
        no source is dropped, and storage stays bounded per native node.
        """
        memo = self._leaf_memo
        if r not in memo:
            pending = [(r, False)]
            while pending:
                ref, ready = pending.pop()
                if ref in memo:
                    continue
                node = self.nodes[ref]
                children = _children(node)
                if not ready:
                    missing = [child for child in children if child not in memo]
                    if missing:
                        pending.append((ref, True))
                        pending.extend((child, False) for child in missing)
                        continue
                # The old DFS pushes operands in their native order and pops
                # the last first. A load contributes itself before its address.
                ordered = {ref: None} if node[0] in ('init', 'param', 'opaque', 'load') else {}
                overflow = False
                for child in reversed(children):
                    sources = memo[child]
                    if sources is None:
                        overflow = True
                        break
                    for source in sources:
                        ordered[source] = None
                        if len(ordered) > self._LEAF_MEMO_LIMIT:
                            overflow = True
                            break
                    if overflow:
                        break
                memo[ref] = None if overflow else tuple(ordered)
        sources = memo[r]
        return self._leaves_uncached(r) if sources is None else list(sources)

    def _leaves_uncached(self, r: int) -> list[int]:
        """Exact unbounded DFS fallback for source sets above the memo cap."""
        out: list[int] = []
        seen: set[int] = set()
        stack = [r]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            n = self.nodes[cur]
            t = n[0]
            if t in ("c", "init", "param", "opaque", "load"):
                if t != "c":
                    out.append(cur)
                if t == "load":
                    stack.append(n[1])
                continue
            if t == "bin":
                stack.append(n[2])
                stack.append(n[3])
            elif t == "un":
                stack.append(n[2])
            elif t in ("trunc", "zext", "sext"):
                stack.append(n[1])
            elif t == "sel":
                stack.append(n[1])
                stack.append(n[2])
                stack.append(n[3])
        return out

    def leaf_path_counts(self, r: int) -> dict[int, int]:
        """How many distinct paths reach each leaf, capped at 2.

        The dispatch resolver tries up to 256 candidate predicates against one address, and nearly
        every one of them cannot possibly work: the candidates are ordered deepest-first, so the
        first two hundred are the individual *bits* of the value the VM read, and pinning a bit
        cannot make the address constant when the discriminating predicate is a composite sitting
        above it. Each attempt costs a full substitution over a several-hundred-node DAG, which is
        where `probe/profile_root.py` puts 85 % of the recovery time (12.43 s of 14.54 s).

        The test this supports is exact rather than heuristic. A leaf the pin does not touch can
        only disappear from the rebuilt expression by cancelling against an *equal subterm*
        (`x ^ x`, `x - x`, `x & x`), and that needs at least two occurrences. A leaf reached by
        exactly one path therefore cannot vanish, so if such a leaf survives the pin the rebuilt
        expression still depends on it and is not a constant -- proved, not guessed, and the
        substitution is skipped without being run.

        Cap-2 accumulation in topological order, so a DAG with shared subterms costs one pass
        rather than one pass per path.
        """
        counts: dict[int, int] = {r: 1}
        order = self._descendants_ordered(r)
        for cur in reversed(order):
            mult = counts.get(cur)
            if mult is None:
                continue
            n = self.nodes[cur]
            t = n[0]
            if t == "c" or t == "init" or t == "param" or t == "opaque":
                continue
            if t == "bin":
                kids = (n[2], n[3])
            elif t == "sel":
                kids = (n[1], n[2], n[3])
            elif t == "load":
                kids = (n[1],)
            elif t == "un":
                kids = (n[2],)
            else:
                kids = (n[1],)
            for k in kids:
                counts[k] = min(2, counts.get(k, 0) + mult)
        return counts

    def pin_can_fold(self, leaves: list[int], counts: dict[int, int], cand: int) -> bool:
        """Can pinning `cand` turn an expression with these leaves into a constant at all?

        False is a proof that it cannot, so a caller may skip the substitution. True means "not
        ruled out", which is the honest strength of the test: a surviving leaf with two or more
        paths *might* cancel, and only running the substitution decides it.
        """
        cand_leaves = set(self.leaves(cand))
        for lf in leaves:
            if lf in cand_leaves:
                continue
            if counts.get(lf, 1) < 2:
                return False
        return True

    def depends_on(self, r: int, refs: Iterable[int]) -> bool:
        wanted = set(refs)
        if not wanted:
            return False
        seen: set[int] = set()
        stack = [r]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in wanted:
                return True
            n = self.nodes[cur]
            t = n[0]
            if t == "c":
                continue
            if t == "bin":
                stack.append(n[2])
                stack.append(n[3])
            elif t == "un":
                stack.append(n[2])
            elif t in ("trunc", "zext", "sext"):
                stack.append(n[1])
            elif t == "sel":
                stack.append(n[1])
                stack.append(n[2])
                stack.append(n[3])
            elif t == "load":
                stack.append(n[1])
        return False

    # ------------------------------------------------------------- rewriting

    def _descendants_ordered(self, r: int) -> list[int]:
        """Refs reachable from `r`, including `r`, in increasing index order.

        Node indices are already a topological order: `_add` appends a node only
        after every node it references exists, so a child's index is always below
        its parent's. That makes this list a valid bottom-up order, and a
        substitution can walk it directly instead of maintaining an explicit
        post-order stack -- two pushes, a `ready` flag and a children list per
        node, which on a several-hundred-node address repeated a hundred times
        costs more than the folding it is there to perform.

        Cached per root, with a small bound: a resolver substitutes the *same*
        address a hundred times in a row, so a handful of entries capture almost
        all of the benefit without retaining a list per distinct expression the
        run ever touches.
        """
        got = self._desc.get(r)
        if got is not None:
            return got
        seen = {r}
        stack = [r]
        while stack:
            cur = stack.pop()
            for k in _children(self.nodes[cur]):
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
        out = sorted(seen)
        if len(self._desc) > 64:
            self._desc.clear()
        self._desc[r] = out
        return out

    def normalize(self, r: int) -> int:
        """Rebuild only newly seen immutable nodes, using normalized children.

        No input is pinned. Historical load versions remain part of each node;
        changing the folding hook starts a fresh cache. Slice-depth-dependent
        folds use the uncached substitution path instead.
        """
        if self._slice_depth:
            return self.substitute(r, {})
        hooks = (self.load_folder, self.versioned_load_folder)
        if getattr(self, '_full_normal_hooks', None) != hooks:
            self._full_normal_hooks = hooks
            self._full_normal = {}
        memo = self._full_normal
        todo = [(r, False)]
        while todo:
            ref, ready = todo.pop()
            if ref in memo:
                continue
            node = self.nodes[ref]
            children = _children(node)
            if not ready and any(child not in memo for child in children):
                todo.append((ref, True))
                todo.extend((child, False) for child in children if child not in memo)
                continue
            memo[ref] = (self._rebuild(node, [memo[child] for child in children])
                         if children else ref)
        return memo[r]

    def substitute(self, r: int, pinmap: dict[int, int], memo: dict[int, int] | None = None,
                   order: list[int] | None = None) -> int:
        """Rebuild `r` with pinned leaves replaced, re-folding as it goes.

        Interning makes this both the substitution and the constant propagator:
        a pinned predicate turns `sel(c, A, B)` into `A` or `B` merely by
        rebuilding it, and everything built on top folds with it.

        Only the part of the DAG that can change is rebuilt. A node under which
        no pinned ref appears rebuilds to its own normal form, which is a
        function of the node and the arena alone -- so it is computed once
        (`_normal`) and reused by every subsequent substitution, instead of being
        recomputed on each of the hundreds of trials a resolver makes against one
        address. The result is unchanged; only the repetition is removed.

        That distinction matters more than it looks. A dispatch resolver
        substitutes a *hundred candidates* into a several-hundred-node address
        expression and throws nearly all of them away, so the repeated rebuild of
        the untouched part of the DAG, not the folding that actually decides
        anything, is where the time goes.

        `order` is the post-order traversal of `r`, which a resolver substituting the same address
        many times already has. Passing it in removes one full DAG walk per trial; the result and the
        emitted node are identical either way.
        """
        if memo is None:
            memo = {}
        if order is None:
            order = self._descendants_ordered(r)
        touched: dict[int, bool] = {}
        for cur in order:
            if cur in memo:
                continue
            pin = pinmap.get(cur)
            if pin is not None:
                memo[cur] = pin
                touched[cur] = True
                continue
            n = self.nodes[cur]
            t = n[0]
            if t == "c" or t == "init" or t == "param" or t == "opaque":
                memo[cur] = cur
                touched[cur] = False
                continue
            if t == "bin":
                kids: tuple[int, ...] = (n[2], n[3])
            elif t == "sel":
                kids = (n[1], n[2], n[3])
            elif t == "load":
                kids = (n[1],)
            else:  # trunc / zext / sext / un
                kids = (n[2],) if t == "un" else (n[1],)
            hit = False
            for k in kids:
                if touched[k]:
                    hit = True
                    break
            if hit:
                memo[cur] = self._rebuild(n, [memo[k] for k in kids])
                touched[cur] = True
            else:
                memo[cur] = self._normal(cur, n, kids)
                # Normalization can change an unpinned child too (including
                # a historically foldable load). Its parents must see the
                # rebuilt ref rather than silently retaining the old child.
                touched[cur] = memo[cur] != cur
        return memo[r]

    # ------------------------------------------------------ direct evaluation

    def eval_exact(self, r: int, env: dict[int, int]) -> tuple[int | None, bool]:
        """The constant `r` takes with `env` pinned, and whether that is final.

        Returns `(value, exact)`. When `exact` is True the answer is *the* answer:
        `value` is the constant, or there is none -- `as_const(substitute(...))`
        would return exactly the same. When `exact` is False the evaluator reached
        a node whose folding can depend on reasoning it does not perform, so a
        None means "ask the folder" rather than "does not fold".

        This exists because of a measurement, not a hunch. A dispatch resolver
        substitutes one candidate predicate at a time into an address expression
        and keeps only the trials that fold, so a run costs `trials x cost of a
        trial`. Instrumenting that cost showed the pin's ancestor cone covers
        **78 % of the addressed DAG** (286 of 368 nodes): the rebuild is not
        skipping anything, it is doing real work, and no cache removes it. The
        only lever left is for a trial not to build nodes at all -- the value is
        all the caller needs, and folding through the interner is a means, not
        the end.

        Two families of rule are deliberately *not* reproduced, because both need
        the rebuilt operand rather than its value: `_decide_compare`, which
        decides a comparison from the operands' known bits, and the slice rules in
        `and` and the shifts. Reaching one of those with an unknown operand is
        what marks the node inexact. Everything else the folder can conclude about
        a node with an unknown operand is an identity restated here exactly.

        Written as one flat loop on purpose. A first version called a helper per
        node and built a `(value, exact)` tuple per node; it was *slower* than the
        substitution it replaced, because per-node Python overhead, not the
        folding, is the cost. Iterating the cached topological order with plain
        dictionary lookups and a membership set that is empty in the common case
        is what makes it pay.

        Sound by construction: it returns a value only where the value is forced.
        That an `exact` True is really final is measured, not asserted --
        `probe.eval_census` answers both ways on a real recovery and counts every
        violation (64,095 trials: zero disagreements, 180 folds this finds that
        the folder does not, 852 the folder finds that this declines), and
        `tests/test_eval.py` does the same over generated DAGs.

        Deliberately **not** on the resolver's hot path. Putting it there was
        implemented and measured, and it was slower -- 31.7 s against 25.3 s on a
        real function -- because a trial visits every node of the addressed
        expression whether or not it rebuilds any. It is used where the trial
        count is orders of magnitude larger and each answer is all that is wanted:
        the 256-value domains `unique_dispatch` and `enumerate_dispatch` drive.
        """
        memo: dict[int, int] = {}
        bad: set[int] = set()
        nodes = self.nodes
        masks = _MASK
        for cur in self._descendants_ordered(r):
            if env:
                ev = env.get(cur)
                if ev is not None:
                    memo[cur] = ev
                    continue
            n = nodes[cur]
            t = n[0]
            if t == "bin":
                op = n[1]
                k0 = n[2]
                k1 = n[3]
                w = n[4]
                x = memo.get(k0)
                y = memo.get(k1)
                if x is not None and y is not None:
                    memo[cur] = _fold_bin(op, x, y, w)
                    if bad and (k0 in bad or k1 in bad):
                        bad.add(cur)
                    continue
                full = masks[w]
                val: int | None = None
                if op == "and":
                    if x == 0 or y == 0:
                        val = 0
                    elif x == full:
                        val = y
                    elif y == full:
                        val = x
                    else:
                        c = x if y is None else y
                        if c != 0 and c != full:
                            bad.add(cur)
                elif op == "or":
                    if x == full or y == full:
                        val = full
                    elif x == 0:
                        val = y
                    elif y == 0:
                        val = x
                elif op == "xor" or op == "add":
                    if x == 0:
                        val = y
                    elif y == 0:
                        val = x
                elif op == "sub":
                    if y == 0:
                        val = x
                elif op == "mul":
                    if x == 0 or y == 0:
                        val = 0
                    elif x == 1:
                        val = y
                    elif y == 1:
                        val = x
                elif op == "shl" or op == "shr" or op == "sar":
                    if y == 0:
                        val = x
                    elif y is not None and y >= w:
                        if op != "sar":
                            val = 0
                        elif x is not None:
                            val = full if (x >> (w - 1)) & 1 else 0
                    elif x == 0:
                        val = 0
                    if val is None and y is not None and 0 < y < w and x is None:
                        bad.add(cur)
                elif op in _BOOL_RESULT_OPS:
                    if y == 0 and op == "ult":
                        val = 0
                    elif y == full and op == "ule":
                        val = 1
                    else:
                        # `_decide_compare` can settle this from known bits.
                        bad.add(cur)
                if k0 == k1:
                    # The folder's self-identities are stated over rebuilt
                    # operands; equal originals do not imply equal rebuilds.
                    bad.add(cur)
                if val is not None:
                    memo[cur] = val
                    continue
                if bad and (k0 in bad or k1 in bad):
                    bad.add(cur)
                continue
            if t == "c":
                memo[cur] = n[1]
                continue
            if t == "init" or t == "param" or t == "opaque":
                continue
            if t == "sel":
                c = memo.get(n[1])
                if c is not None:
                    a1 = n[2] if c else n[3]
                    v = memo.get(a1)
                    if v is not None:
                        memo[cur] = v
                    if bad and a1 in bad:
                        bad.add(cur)
                    continue
                v1 = memo.get(n[2])
                v2 = memo.get(n[3])
                if v1 is not None and v1 == v2:
                    memo[cur] = v1
                    continue
                if n[2] == n[3] or (bad and (n[1] in bad or n[2] in bad or n[3] in bad)):
                    bad.add(cur)
                continue
            if t == "load":
                c = memo.get(n[1])
                if c is None or (self.load_folder is None and self.versioned_load_folder is None):
                    if bad and n[1] in bad:
                        bad.add(cur)
                    continue
                got = (self.versioned_load_folder(c, n[2], n[3])
                       if self.versioned_load_folder is not None else self.load_folder(c, n[2]))
                if got is None:
                    continue
                v = self.as_const(got)
                if v is not None:
                    memo[cur] = v
                continue
            # un / trunc / zext / sext. The width sits at a different index in
            # each spelling: `("un", op, child, w)` but `("trunc", child, w)`.
            if t == "un":
                k0 = n[2]
                w = n[3]
            else:
                k0 = n[1]
                w = n[2]
            x = memo.get(k0)
            if x is not None:
                full = masks[w]
                if t == "un":
                    memo[cur] = (~x) & full if n[1] == "not" else (-x) & full
                elif t == "trunc":
                    memo[cur] = x & full
                elif t == "zext":
                    memo[cur] = x
                else:
                    memo[cur] = sext_value(x, self.width(k0)) & full
                continue
            if bad and k0 in bad:
                bad.add(cur)
        return memo.get(r), r not in bad

    def eval_const(self, r: int, env: dict[int, int]) -> int | None:
        """`eval_exact`, with the exactness flag dropped. See `eval_exact`."""
        return self.eval_exact(r, env)[0]

    def _normal(self, r: int, n: tuple, kids: tuple[int, ...]) -> int:
        """`_rebuild(n, kids)` for a node whose operands did not change, cached.

        The folder is a pure function of the node and the interned arena -- it
        compares operand *refs* and adds canonically interned results -- so this
        is the same value on every call, and computing it once is not an
        approximation of the original code but the original code with the
        repetition removed.

        Only cached at the top level. Inside a slice, `_slice_depth` bounds how
        far the folder is allowed to go, so the answer depends on the call
        context and must not be shared.
        """
        if self._slice_depth:
            return self._rebuild(n, list(kids))
        got = self._norm.get(r)
        if got is None:
            got = self._rebuild(n, list(kids))
            self._norm[r] = got
        return got

    def _rebuild(self, n: tuple, kids: list[int]) -> int:
        t = n[0]
        if t == "bin":
            return self.bin(n[1], kids[0], kids[1], n[-1])
        if t == "un":
            return self.un(n[1], kids[0], n[-1])
        if t == "trunc":
            return self.trunc(kids[0], n[-1])
        if t == "zext":
            return self.zext(kids[0], n[-1])
        if t == "sext":
            return self.sext(kids[0], n[-1])
        if t == "sel":
            return self.select(kids[0], kids[1], kids[2], n[-1])
        if t == "load":
            folded = self._fold_known_load(kids[0], n[2], n[3])
            if folded is not None:
                return folded
            return self.load(kids[0], n[2], n[3])
        return self._add(n)

    # ------------------------------------------------------------ bit slicing

    def _guarded_slice(self, r: int, lo: int, hi: int) -> int | None:
        """`bit_slice` under a re-entry guard.

        Slicing builds nodes, and building nodes calls the folder, which slices
        again. The guard turns that mutual recursion into a bounded descent
        instead of a hang; a slice that needs more than a few levels to pay off
        is not worth taking.

        Memoising this was tried and **removed**: the results are interned, so a
        cache is sound, but on the real workload the top-level `(node, lo, hi)`
        triples essentially never repeat, and the dictionary lookups cost what
        the slices saved. Recorded as a null result rather than kept as
        plausible-looking complexity.
        """
        if self._slice_depth > 24:
            return None
        self._slice_depth += 1
        try:
            return self.bit_slice(r, lo, hi)
        finally:
            self._slice_depth -= 1

    def bit_slice(self, r: int, lo: int, hi: int) -> int | None:
        """Bits `[lo, hi)` of `r`, zero-extended into a 64-bit value.

        Sound and deliberately partial: it distributes over the bitwise
        operators and constant shifts, which is exactly the shape the protector
        emits, and returns None the moment an add/sub/mul or an un-sliceable
        leaf would make the answer a guess.
        """
        if self._slice_depth > 32:
            return None
        w = self.width(r)
        if lo >= hi or lo >= w:
            return self.const(0, W64)
        if hi > w:
            hi = w
        span = (1 << (hi - lo)) - 1

        n = self.nodes[r]
        t = n[0]
        if t == "c":
            return self.const((n[1] >> lo) & span, W64)

        if t == "bin":
            op = n[1]
            a, b = n[2], n[3]
            if op == "and":
                # Both operand orders must be handled: the folder canonicalises
                # commutative nodes, so a mask can end up on either side and a
                # one-sided test silently misses half of them.
                ca = self.as_const(a)
                cb = self.as_const(b)
                span_mask = span << lo
                if ca is not None and ((ca >> lo) & span) == 0:
                    return self.const(0, W64)
                if cb is not None and ((cb >> lo) & span) == 0:
                    return self.const(0, W64)
                # Same reasoning as for `or`: a side whose bits are all known
                # zero in this range annihilates the whole conjunction.
                if (self.zbits(a) & span_mask) == span_mask:
                    return self.const(0, W64)
                if (self.zbits(b) & span_mask) == span_mask:
                    return self.const(0, W64)
                sa = self.bit_slice(a, lo, hi)
                sb = self.bit_slice(b, lo, hi)
                if sa is not None and sb is not None:
                    return self.bin("and", sa, sb, W64)
                if sa is not None and cb is not None:
                    return self.bin("and", sa, self.const((cb >> lo) & span, W64), W64)
                if sb is not None and ca is not None:
                    return self.bin("and", sb, self.const((ca >> lo) & span, W64), W64)
                return None
            if op in ("or", "xor"):
                # A term cannot supply a bit that is known zero in it. Testing
                # that first is what makes the protector's masking idioms
                # collapse: the VM constantly builds values like
                # `((x & ~0xffff) | zext16(y)) & 0xffff`, where the outer mask
                # annihilates the first term entirely. Requiring *both* sides of
                # an `or` to be independently sliceable -- which is what this did
                # before -- fails on exactly those, and takes the projection of
                # the whole packed-RFLAGS word down with it.
                span_mask = span << lo
                za = self.zbits(a)
                zb = self.zbits(b)
                if op == "or":
                    a_dead = (za & span_mask) == span_mask
                    b_dead = (zb & span_mask) == span_mask
                    if a_dead and b_dead:
                        return self.const(0, W64)
                    if a_dead:
                        return self.bit_slice(b, lo, hi)
                    if b_dead:
                        return self.bit_slice(a, lo, hi)
                sa = self.bit_slice(a, lo, hi)
                if sa is None:
                    return None
                sb = self.bit_slice(b, lo, hi)
                if sb is None:
                    return None
                return self.bin(op, sa, sb, W64)
            if op == "shl":
                k = self.as_const(b)
                if k is None or k >= w:
                    return None if k is None else self.const(0, W64)
                if hi <= k:
                    return self.const(0, W64)
                if lo >= k:
                    return self.bit_slice(a, lo - k, hi - k)
                # Bits below `k` are zero; the rest arrive shifted down by
                # `k - lo`, because the slice is re-based to bit 0.
                inner = self.bit_slice(a, 0, hi - k)
                if inner is None:
                    return None
                return self.bin("shl", inner, self.const(k - lo, W64), W64)
            if op == "shr":
                k = self.as_const(b)
                if k is None:
                    return None
                return self.bit_slice(a, lo + k, hi + k)
            if op == "sar":
                k = self.as_const(b)
                if k is not None and hi + k <= w and lo + k <= w:
                    inner = self.bit_slice(a, lo + k, hi + k)
                    if inner is not None:
                        return inner
                return None
            return None

        if t == "un":
            if n[1] != "not":
                return None
            sa = self.bit_slice(n[2], lo, hi)
            if sa is None:
                return None
            return self.bin("and", self.un("not", sa, W64), self.const(span, W64), W64)

        if t == "trunc":
            return self.bit_slice(n[1], lo, hi if hi <= n[2] else n[2])
        if t == "zext":
            return self.bit_slice(n[1], lo, hi)
        if t == "sext":
            if hi <= self.width(n[1]):
                return self.bit_slice(n[1], lo, hi)
            return None
        if t == "sel":
            sa = self.bit_slice(n[2], lo, hi)
            if sa is None:
                return None
            sb = self.bit_slice(n[3], lo, hi)
            if sb is None:
                return None
            return self.select(n[1], sa, sb, W64)
        if t in ("init", "param", "opaque", "load"):
            if lo == 0 and hi >= w:
                return self.zext(r, W64)
            # A leaf projected at its low end is that leaf masked. This has to be
            # answered rather than refused: `bit_slice` distributes over `or`, and
            # an `or`-tree slice fails as a whole if *any* branch fails. One
            # un-sliceable flag leaf would otherwise take down the projection of
            # the entire packed-RFLAGS word, which is why
            # `unpack_flags(pack_flags())` did not return the flags it started
            # with -- the single most important identity in the whole fold.
            if lo == 0:
                return self.bin("and", r, self.const((1 << hi) - 1, W64), W64)
            return None

        # A *general* low-end fallback was also tried here -- project bits [0,hi)
        # of any node at all, including arithmetic -- and was measured and
        # rejected: on the benchmark root it took the expression DAG from 12,003
        # nodes to 162,272 and the walk from 543 ms to 2,496 ms, for the same 796
        # transitions. Masking every un-sliceable branch multiplies nodes without
        # ever letting the tree collapse. Restricting it to leaves is what the
        # `or`-distribution actually needs, and it is nearly free.
        return None

    # -------------------------------------------------------- folding builders

    def bin(self, op: str, a: int, b: int, width: int) -> int:
        assert op in BIN_OPS, op
        wa, wb = self.width(a), self.width(b)
        if wa != width:
            a = self._fit(a, width)
        if wb != width:
            b = self._fit(b, width)
        full = _MASK[width]
        ca, cb = self.as_const(a), self.as_const(b)

        # --- both constant: fold exactly
        if ca is not None and cb is not None:
            return self.const(_fold_bin(op, ca, cb, width), width)

        # --- structural self-identities (the whole point of interning)
        if a == b:
            if op in ("sub", "xor"):
                return self.const(0, width)
            if op in ("and", "or"):
                return a
            if op in ("eq", "ule", "sle"):
                return self.const(1, width)
            if op in ("ne", "ult", "slt"):
                return self.const(0, width)

        # --- result-is-boolean operators: canonicalise against constants
        if op in _BOOL_RESULT_OPS:
            # `ne` is `not eq`; canonicalising here keeps one node kind per
            # comparison so that two spellings of the same test intern together.
            if op == "ne":
                return self.bin("xor", self.bin("eq", a, b, width), self.const(1, width), width)
            # A comparison against zero is how the protector tests a single bit
            # of a byte the VM read. Normalising the arithmetic under it is what
            # makes `(X + -0x40) == 0` recognisable as `X == 0x40`, and therefore
            # what makes a two-entry table index recognisable as a predicate at
            # all. Without it the index stays a wide expression and the dispatch
            # is reported unresolved.
            if op == "eq":
                for x, y in ((a, b), (b, a)):
                    if self.as_const(y) != 0:
                        continue
                    xn = self.nodes[x]
                    if xn[0] != "bin":
                        continue
                    if xn[1] == "add":
                        c = self.as_const(xn[3])
                        if c is not None:
                            return self.bin("eq", xn[2], self.const((-c) & _MASK[width], width), width)
                    elif xn[1] in ("sub", "xor"):
                        return self.bin("eq", xn[2], xn[3], width)
            if op == "ult" and cb == 0:
                return self.const(0, width)
            if op == "ule" and cb == full:
                return self.const(1, width)
            if op == "ule" and ca == 0:
                return self.const(1, width)
            # Forced constants: `x `u< 0` and the like are handled above, but a
            # comparison between two values whose known bits already decide it
            # folds without enumerating anything.
            decided = _decide_compare(op, self.zbits(a), self.obits(a), self.zbits(b), self.obits(b), width)
            if decided is not None:
                return self.const(decided, width)
            return self._add(("bin", op, a, b, width))

        # --- annihilators and identities
        if op == "and":
            if ca == 0 or cb == 0:
                return self.const(0, width)
            if ca == full:
                return b
            if cb == full:
                return a
            # Flatten nested masks: and(and(x, c1), c2) == and(x, c1 & c2). The
            # protector builds stacked masks, and without this the innermost
            # value stays buried under a chain of separately-interned nodes --
            # which is exactly what the packed-flags round trip produces, four
            # levels deep. The constant may be on either side of the inner `and`
            # because the folder canonicalises commutative operands.
            for x, c in ((a, cb), (b, ca)):
                if c is None:
                    continue
                xn = self.nodes[x]
                if xn[0] == "bin" and xn[1] == "and":
                    ic = self.as_const(xn[3])
                    inner = xn[2]
                    if ic is None:
                        ic = self.as_const(xn[2])
                        inner = xn[3]
                    if ic is not None:
                        return self.bin("and", inner, self.const(ic & c, width), width)
            # Absorption: and(x, or(x, y)) == x
            for x, y in ((a, b), (b, a)):
                yn = self.nodes[y]
                if yn[0] == "bin" and yn[1] == "or" and (yn[2] == x or yn[3] == x):
                    return x
            # A mask applied to a shifted field is itself a bit selection:
            # `(X << 1) & 1` is zero because bit 0 of a left shift is zero, and
            # only a slice can see that.
            #
            # This belongs to `and` and *only* to `and`. An earlier revision had
            # it one indentation level deeper, inside the shift branch, where it
            # read the shift *count* as if it were a mask and silently rewrote
            # `shr(v, 6)` as `v & 6`. That is not a small error: it corrupts every
            # shift, and it is invisible in any value-level diagnostic because the
            # result is a perfectly ordinary-looking expression.
            if self._slice_depth == 0:
                c = ca if ca is not None else cb
                other = b if ca is not None else a
                if c is not None and c:
                    hi = min(c.bit_length(), width)
                    sliced = self._guarded_slice(other, 0, hi)
                    if sliced is not None and sliced != other:
                        sc = self.as_const(sliced)
                        if sc is not None:
                            return self.const(sc & c, width)
                        cand = self._rebuild_nofold(("bin", "and", sliced, self.const(c, width), width))
                        if cand != self._rebuild_nofold(("bin", "and", a, b, width)):
                            return cand
        if op == "or":
            if ca == full or cb == full:
                return self.const(full, width)
            if ca == 0:
                return b
            if cb == 0:
                return a
            for x, y in ((a, b), (b, a)):
                yn = self.nodes[y]
                if yn[0] == "bin" and yn[1] == "and" and (yn[2] == x or yn[3] == x):
                    return x
        if op == "xor":
            if ca == 0:
                return b
            if cb == 0:
                return a
            if ca == full:
                return self.un("not", b, width)
            if cb == full:
                return self.un("not", a, width)
        if op == "add":
            if ca == 0:
                return b
            if cb == 0:
                return a
        if op == "sub":
            if cb == 0:
                return a
        if op == "mul":
            if ca == 0 or cb == 0:
                return self.const(0, width)
            if ca == 1:
                return b
            if cb == 1:
                return a
        if op in ("shl", "shr", "sar"):
            k = cb
            if k is not None:
                if k == 0:
                    return a
                if k >= width:
                    if op == "sar":
                        # arithmetic shift right to nothing is the sign broadcast
                        return self.select(
                            self.bin("slt", a, self.const(0, width), width),
                            self.const(full, width),
                            self.const(0, width),
                            width,
                        )
                    return self.const(0, width)
                # Bit-level projection. This is what collapses the VM's
                # `pushfq` / `popfq` round trip: packing RFLAGS builds an
                # `or`-tree of shifted flag bits, and unpacking one flag is a
                # slice of that tree. Without the projection the flag stays an
                # opaque 30-node expression and every branch on it stops the
                # walk.
                #
                # Only attempted at the outermost level, and the candidate is
                # built with `_add` rather than re-entering the folder. Slicing
                # builds nodes and building nodes lands back here, so recursing
                # would be an unbounded cycle rather than a fixpoint.
                if self._slice_depth == 0 and op in ('shl', 'shr'):
                    lo_s, hi_s = (k, width) if op == "shr" else (0, width - k)
                    sliced = self._guarded_slice(a, lo_s, hi_s)
                    if sliced is not None:
                        if op == "shr":
                            cand = sliced if self.width(sliced) == width else self._rebuild_nofold(("trunc", sliced, width))
                        else:
                            inner = self._rebuild_nofold(("bin", "shl", sliced, self.const(k, W64), W64))
                            cand = inner if width == W64 else self._rebuild_nofold(("trunc", inner, width))
                        plain = self._rebuild_nofold(("bin", op, a, self.const(k, width), width))
                        if cand != plain:
                            return cand
            if ca == 0:
                return self.const(0, width)

        # --- canonical operand order for commutative ops, so that interning
        # merges `A + B` written two ways.
        if op in ("add", "mul", "and", "or", "xor") and b < a:
            a, b = b, a
        return self._add(("bin", op, a, b, width))

    def _not_bool(self, r: int, width: int) -> int:
        return self.bin("xor", r, self.const(1, width), width)

    def _fit(self, r: int, width: int) -> int:
        w = self.width(r)
        if w == width:
            return r
        return self.zext(r, width) if w < width else self.trunc(r, width)

    def un(self, op: str, a: int, width: int) -> int:
        assert op in UN_OPS, op
        if self.width(a) != width:
            a = self._fit(a, width)
        ca = self.as_const(a)
        full = _MASK[width]
        if ca is not None:
            return self.const((~ca) & full if op == "not" else (-ca) & full, width)
        if op == "not":
            n = self.nodes[a]
            if n[0] == "un" and n[1] == "not":
                return n[2]
            # De Morgan, so absorption can see through the negation. The
            # protector writes `~(a|b)` and then `| ~a`, which only collapses if
            # the inner negation is pushed down first.
            if n[0] == "bin":
                if n[1] == "or":
                    return self.bin("and", self.un("not", n[2], width), self.un("not", n[3], width), width)
                if n[1] == "and":
                    return self.bin("or", self.un("not", n[2], width), self.un("not", n[3], width), width)
                if n[1] == "xor" and self.as_const(n[3]) is not None:
                    return self.bin("xor", self.un("not", n[2], width), n[3], width)
        if op == "neg":
            n = self.nodes[a]
            if n[0] == "un" and n[1] == "neg":
                return n[2]
        return self._add(("un", op, a, width))

    def trunc(self, a: int, width: int) -> int:
        wa = self.width(a)
        if wa == width:
            return a
        if wa < width:
            return a
        ca = self.as_const(a)
        if ca is not None:
            return self.const(ca & _MASK[width], width)
        n = self.nodes[a]
        # Masking off bits that the truncation discards anyway is redundant:
        # `trunc8(x & 0xff)` is `trunc8(x)`. The slicer introduces these masks
        # when it projects a leaf, so without this rule every projection leaves
        # one behind.
        if n[0] == "bin" and n[1] == "and":
            for x, c in ((n[2], self.as_const(n[3])), (n[3], self.as_const(n[2]))):
                if c is not None and (c & _MASK[width]) == _MASK[width]:
                    return self.trunc(x, width)
        if n[0] in ("zext", "sext"):
            inner = n[1]
            wi = self.width(inner)
            if wi == width:
                return inner
            if wi > width:
                return self.trunc(inner, width)
            # Width algebra: narrowing a widening that is already no wider than
            # the target is the widening itself. Without this rule the VM's
            # repeated byte extraction builds `zext64(trunc8(zext64(trunc8(..))))`
            # chains that never reduce, and a two-entry table index buried in one
            # of them is unrecognisable as a 0/1 value.
            if n[0] == "zext":
                return self.zext(inner, width)
            return self.sext(inner, width)
            # `trunc` of a widening: only a slice of the source survives. Built
            # with `_add` so this cannot re-enter the slicer.
            if self._slice_depth == 0:
                narrowed = self._guarded_slice(inner, 0, width)
                if narrowed is not None:
                    cand = narrowed if self.width(narrowed) == width else self._rebuild_nofold(("trunc", narrowed, width))
                    if cand != self._rebuild_nofold(("trunc", a, width)):
                        return cand
            return self._rebuild_nofold(("trunc", a, width))
        if n[0] == "trunc" and n[2] >= width:
            return self.trunc(n[1], width)
        # A truncation is bit selection, and bit selection is what makes an
        # `or`-tree of shifted fields readable again. Outermost level only, and
        # built with `_add`, for the same re-entrancy reason as in `bin`.
        if self._slice_depth == 0:
            sliced = self._guarded_slice(a, 0, width)
            if sliced is not None and sliced != a:
                if self.width(sliced) == width:
                    return sliced
                cand = self._rebuild_nofold(("trunc", sliced, width))
                if cand != self._rebuild_nofold(("trunc", a, width)):
                    return cand
        return self._rebuild_nofold(("trunc", a, width))

    def zext(self, a: int, width: int) -> int:
        wa = self.width(a)
        if wa == width:
            return a
        if wa > width:
            return self.trunc(a, width)
        ca = self.as_const(a)
        if ca is not None:
            return self.const(ca, width)
        n = self.nodes[a]
        if n[0] == "zext" and n[2] <= width:
            return self.zext(n[1], width)
        if n[0] == "trunc":
            # Zero-extending a value that was narrowed to at least its own width
            # is the value itself: the truncation removed nothing.
            inner = n[1]
            if n[2] >= self.width(inner):
                return self.zext(inner, width)
            # ...and zero-extending a truncation of a zero-extension is that
            # extension narrowed then widened, which reduces to the truncation.
            inode = self.nodes[inner]
            if inode[0] == "zext" and inode[2] >= n[2]:
                return self.zext(self.trunc(inode[1], n[2]), width)
        return self._add(("zext", a, width))

    def sext(self, a: int, width: int) -> int:
        wa = self.width(a)
        if wa == width:
            return a
        if wa > width:
            return self.trunc(a, width)
        ca = self.as_const(a)
        if ca is not None:
            return self.const(sext_value(ca, wa) & _MASK[width], width)
        n = self.nodes[a]
        if n[0] == "sext" and n[-1] <= width:
            return self.sext(n[1], width)
        if n[0] == "zext" and self.zbits(a) == _MASK[wa]:
            # A value already known to be non-negative sign-extends like a zext.
            return self.zext(n[1], width)
        return self._add(("sext", a, width))

    def select(self, c: int, a: int, b: int, width: int) -> int:
        # Two identities, and the order they are tested in is the whole of this
        # function's history.
        #
        # The constant test must come first. It used to come after a width test
        # that refused any condition whose type was not exactly eight bits, and
        # `select` returns the *unfolded* node in that case -- so a select whose
        # condition had folded to a constant stayed unfolded forever whenever the
        # constant was not W8. That is a permanent barrier of exactly the kind this
        # engine keeps rediscovering: the value is right, the shape is a wall, and
        # nothing downstream can see through it. Fixing it is what closed the CFG.
        #
        # The `a == b` test has now been found in the same trap, one line further
        # down. The width test returned early, so `cond ? X : X` folded only when
        # the condition happened to be exactly one byte -- and the branch it
        # returned from built *the identical node* the fallthrough would have
        # built, so the width test did nothing at all except skip the identity.
        #
        # It was not a rare shape. With native calls interpreted rather than
        # stubbed, seventeen dropped store addresses in one function render as
        # `trunc16((0xa770 xor ((1 and zext64(flag_af)) ? 0xa7f8 : 0xa7f8)))` --
        # a constant, held symbolic by a select whose arms are the same number.
        cc = self.as_const(c)
        if cc is not None:
            return self._fit(a if cc else b, width)
        af = self._fit(a, width)
        bf = self._fit(b, width)
        if af == bf:
            return af
        return self._add(("sel", c, af, bf, width))


def boolean_subexpressions(a: "Arena", r: int, limit: int = 64) -> list[int]:
    """Subexpressions of `r` that provably hold only 0 or 1, deepest first.

    The VM compiles `cond ? A : B` into arithmetic, so a predicate ends up below
    a chain of shifts and truncations rather than as a bare value. Deepest first
    tries the innermost predicate before the composites built on top of it, which
    is the one that actually discriminates.

    Lives here rather than in the recovery driver because the *lifter* needs it
    too: a load whose address takes two values is resolved by splitting on one of
    these, and that has to happen where the load is performed, not later.
    """
    out: list[tuple[int, int]] = []
    seen: set[int] = set()
    stack = [(r, 0)]
    while stack:
        cur, d = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if not a.is_const(cur) and a.is_bool(cur):
            out.append((d, cur))
        n = a.nodes[cur]
        t = n[0]
        if t == "bin":
            stack.append((n[2], d + 1))
            stack.append((n[3], d + 1))
        elif t == "un":
            stack.append((n[2], d + 1))
        elif t in ("trunc", "zext", "sext"):
            stack.append((n[1], d + 1))
        elif t == "sel":
            stack.append((n[1], d + 1))
            stack.append((n[2], d + 1))
            stack.append((n[3], d + 1))
        elif t == "load":
            stack.append((n[1], d + 1))
    out.sort(key=lambda kv: -kv[0])
    return [r for _, r in out[:limit]]


def _arithmetic_known_bits(za, oa, zb, ob, width, subtract=False):
    """Sound interval-prefix and exact low-prefix facts, including carry/borrow.

    A shared zero in the operands is NOT a known zero of their sum. The result
    is bounded by extrema, and only their common high prefix is invariant.
    Low bits whose input bits are all fixed can additionally be computed exactly.
    """
    space = 1 << width
    full = space - 1
    za, oa, zb, ob = za & full, oa & full, zb & full, ob & full
    lo, hi = ((oa - (full ^ zb), (full ^ za) - ob) if subtract
              else (oa + ob, (full ^ za) + (full ^ zb)))
    zero = one = 0
    if lo // space == hi // space:
        lo, hi = lo & full, hi & full
        varying = (1 << (lo ^ hi).bit_length()) - 1
        one = lo & ~varying & full
        zero = ~hi & ~varying & full
    known = (za | oa) & (zb | ob)
    low_mask = ((known ^ (known + 1)) >> 1) & full
    exact_low = (oa - ob if subtract else oa + ob) & low_mask
    return zero | (low_mask ^ exact_low), one | exact_low


def _decide_compare(op: str, za: int, oa: int, zb: int, ob: int, width: int) -> int | None:
    """Decide a comparison from known bits alone, or None when it is undecided.

    Sound, never complete: it only fires when the two operands are *provably*
    different at some bit. That is enough for the case that matters -- a
    comparison against zero where the other side is forced by a mask -- without
    ever inventing a verdict the bits do not support.
    """
    diff = (oa & zb) | (za & ob)
    if op == "eq":
        return 0 if diff else None
    if op == "ne":
        return 1 if diff else None
    full = (1 << width) - 1
    def interval(zero, one, signed):
        lo, hi = one & full, full ^ (zero & full)
        if not signed:
            return lo, hi
        sign = 1 << (width - 1)
        if zero & sign:
            return lo, hi
        if one & sign:
            return lo - (1 << width), hi - (1 << width)
        return (lo | sign) - (1 << width), hi & ~sign
    if op in ('ult', 'ule', 'slt', 'sle'):
        alo, ahi = interval(za, oa, op.startswith('s'))
        blo, bhi = interval(zb, ob, op.startswith('s'))
        if op.endswith('lt'):
            if ahi < blo:
                return 1
            if alo >= bhi:
                return 0
        else:
            if ahi <= blo:
                return 1
            if alo > bhi:
                return 0
    return None


def _children(n: tuple) -> list[int]:
    t = n[0]
    if t == "bin":
        return [n[2], n[3]]
    if t == "un":
        return [n[2]]
    if t in ("trunc", "zext", "sext"):
        return [n[1]]
    if t == "sel":
        return [n[1], n[2], n[3]]
    if t == "load":
        return [n[1]]
    return []


def _mix_scalar(item) -> int:
    if isinstance(item, str):
        h = 0xCBF29CE484222325
        for ch in item.encode():
            h ^= ch
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h
    if isinstance(item, int):
        return (item * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    return 0x1234567890ABCDEF


def _fold_bin(op: str, a: int, b: int, width: int) -> int:
    m = _MASK[width]
    if op == "add":
        return (a + b) & m
    if op == "sub":
        return (a - b) & m
    if op == "mul":
        return (a * b) & m
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "shl":
        return 0 if b >= width else (a << b) & m
    if op == "shr":
        return 0 if b >= width else (a & m) >> b
    if op == "sar":
        sa = sext_value(a, width)
        if b >= width:
            return m if sa < 0 else 0
        return (sa >> b) & m
    if op == "udiv":
        return 0 if b == 0 else (a // b) & m
    if op == "urem":
        return 0 if b == 0 else (a % b) & m
    if op == "eq":
        return 1 if a == b else 0
    if op == "ne":
        return 1 if a != b else 0
    if op == "ult":
        return 1 if a < b else 0
    if op == "ule":
        return 1 if a <= b else 0
    if op == "slt":
        return 1 if sext_value(a, width) < sext_value(b, width) else 0
    if op == "sle":
        return 1 if sext_value(a, width) <= sext_value(b, width) else 0
    raise KeyError(op)


def render(arena: Arena, r: int, limit: int = 200, *, max_chars: int = 4096) -> str:
    """Bounded diagnostic text; never materialize the expanded expression tree.

    ``limit`` bounds node visits, including repeated occurrences of a shared node.
    ``max_chars`` bounds the returned string. ``...`` marks any omitted content;
    expression refs used for recovery and lowering are left untouched.
    """
    if max_chars < 3:
        raise ValueError("max_chars must leave room for the omission marker")
    parts: list[str] = []
    length = visits = 0
    # Tokens are emitted directly, rather than concatenating rendered children.
    # The depth bound also keeps the pending token stack small on deep DAGs.
    pending = [(True, r, 0)]
    truncated = False
    while pending:
        is_ref, value, depth = pending.pop()
        if is_ref:
            if visits >= limit:
                truncated = True
                break
            visits += 1
            if depth > 24:
                pending.append((False, "...", depth))
                continue
            n = arena.nodes[value]
            t = n[0]
            child = lambda ref: (True, ref, depth + 1)
            text = lambda token: (False, token, depth)
            if t == "c":
                tokens = [text(f"{n[1]:#x}" if n[1] > 9 else str(n[1]))]
            elif t == "init":
                tokens = [text("in."), text(n[1])]
            elif t == "param":
                tokens = [text("p"), text(str(n[1])), text("."), text(str(n[2]))]
            elif t == "opaque":
                tokens = [text(n[1])]
            elif t == "load":
                tokens = [text("*"), child(n[1]), text(f":{n[2]}")]
            elif t == "bin":
                tokens = [text("("), child(n[2]), text(" "), text(n[1]),
                          text(" "), child(n[3]), text(")")]
            elif t == "un":
                tokens = [text(n[1]), text("("), child(n[2]), text(")")]
            elif t in ("trunc", "zext", "sext"):
                tokens = [text(f"{t}{n[2]}("), child(n[1]), text(")")]
            elif t == "sel":
                tokens = [text("("), child(n[1]), text(" ? "), child(n[2]),
                          text(" : "), child(n[3]), text(")")]
            else:
                # Do not repr an arbitrary node tuple: its payload may be large.
                tokens = [text("<"), text(t), text(f" n{value}>")]
            pending.extend(reversed(tokens))
        else:
            remaining = max_chars - length
            if len(value) > remaining:
                parts.append(value[:remaining])
                truncated = True
                break
            parts.append(value)
            length += len(value)
    result = "".join(parts)
    return result[:max_chars - 3] + "..." if truncated else result
