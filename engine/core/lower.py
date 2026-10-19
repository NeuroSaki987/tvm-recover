"""Lowering a recovered guest value into three-address code.

The recovery produces a DAG of lifted operations. This module preserves those operations in
an ordered, block-local value stream. It does not prove the lifter correct, infer missing guest
instruction boundaries, or turn sampled expression identification into an equivalence proof.

The reference's `lower` shows what the artifact should look like:

    v2:64 = opaque call_ret @Rax
    v3:64 = And v2, 0xffffffff
    v12:64 = Add rcx, rcx
    store64 [v16], v11

Every sub-expression gets a name, and each line is one operation. Global historical guest names
are not substituted for result expressions: only earlier definitions in this block may be reused.

Three properties are deliberate:

* **post-order**, so a name is always defined before it is used, and the emitted order is a
  topological order of the DAG rather than the order a walk happened to visit it;
* **honest about what it cannot lower** -- an unknown node shape emits a named opaque line rather
  than being inlined or dropped. Iterative traversal handles deep DAGs; the explicit node budget
  reports incompleteness instead of silently cutting lines;
* **guest register names survive** -- a leaf that is guest state prints as `guest_rax`, not as a
  number, because that is the whole point of the recovery.
"""

from __future__ import annotations

# The one implementation of a narrow write's merge, shared with the dataflow so both sides produce
# the *same* node -- see `xstate.partial`. Imported at module level rather than inside the method that
# uses it: a mid-body import is what an earlier edit got the indentation wrong on, and removing that
# line then left `leaf_name` undefined, which crashed the emission for a whole root.
from .xstate import partial  # noqa: E402
from .guestir import leaf_name  # noqa: E402


def is_reference(text: str) -> bool:
    """Whether a rendered state value is a **name a reader can follow**, not an expression.

    A ctx read whose value the incoming state already names may be emitted as a bare line, because
    `#3:guest_rax_1` is a *reference* to a definition in another block. A **join** is not: the value
    `xstate` hands back at a block with two predecessors is `phi(a, b)`, and emitting that as a bare
    line produces a body line that neither defines a name nor mentions an operand -- the acceptance
    gate's check 3 counted 79 of them on four of the fourteen reviewed roots, every one a `phi(...)`
    with no destination. The read site binds such a value to a temporary instead, so the line is a
    definition and the join is followable rather than dangling. Only the *text shape* is decided
    here; whether the value deserves a cross-block name stays `devirt`'s owners-and-dominance rule.
    """
    name = text.split(':', 1)[1] if text.startswith('#') else text
    if ':' in name and name.rsplit(':', 1)[1].isdigit():
        name = name.rsplit(':', 1)[0]
    return bool(name) and (name[0].isalpha() or name[0] == '_') and all(
        ch.isalnum() or ch in '_.' for ch in name)

# Binops whose text is the operator itself. Anything else prints as a call-shaped form so an
# unusual operator is visible rather than silently mis-spelled.
INFIX = {
    "add": "+", "sub": "-", "mul": "*", "and": "&", "or": "|", "xor": "^",
    "shl": "<<", "shr": ">>", "sar": ">>a", "div": "/", "mod": "%",
    "eq": "==", "ne": "!=", "ult": "<u", "ule": "<=u", "slt": "<s", "sle": "<=s",
    "ugt": ">u", "uge": ">=u", "sgt": ">s", "sge": ">=s",
}
PREFIX = {"not": "~", "neg": "-"}

# The architectural RFLAGS bit of each flag the lifter models. The pack recognizer checks a
# candidate against these positions rather than assuming them.
FLAG_BITS = {"cf": 0, "pf": 2, "af": 4, "zf": 6, "sf": 7, "df": 10, "of": 11}

# The guest register names, so a rendered `guest_<x>` can be told from a rendered `guest_<x>_<n>`.
REGISTER_NAMES = frozenset(
    ("rax", "rbx", "rcx", "rdx", "rsp", "rbp", "rsi", "rdi",
     "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rflags"))


def split_guest_name(name: str) -> tuple[str, str | None]:
    """`("guest_r8_2") -> ("r8", "r8")`; the register a rendered name belongs to, or None.

    Shared with `core/devirt.py` when it collects which block defines what, so both sides classify
    `guest_r8` (the VM-entry value of a register) and `guest_r8_2` (a write to it) the same way.
    """
    if not name.startswith("guest_"):
        return name, None
    tail = name[len("guest_"):]
    if "_" in tail:
        head, _, last = tail.rpartition("_")
        if last.isdigit() and head in REGISTER_NAMES:
            return head, head
    return tail, tail if tail in REGISTER_NAMES else None

MAX_NODES = 100_000


def guest_rel(reg: str, off: int) -> str:
    """A guest-relative address: `guest_rcx+0x18`, `guest_rsp-0x8`, `guest_rbx`.

    The form a recompiler needs. `[rsp+0x10]` in the reference's output is this; an absolute
    `0x7ffffffef700` is a property of one process's stack and cannot be emitted for another.
    """
    if off == 0:
        return f"guest_{reg}"
    sign = "+" if off > 0 else "-"
    return f"guest_{reg}{sign}{abs(off):#x}"


class Lowering:
    """One function's lowering: names are unique across it, as in a normal listing."""

    def __init__(self, arena, guest_values, ctx=None, identity=None, entry=None) -> None:
        self.a = arena
        self.gv = guest_values
        # Only identity claims name a register. See `name()`.
        self.identity: dict[int, str] = identity if identity is not None else {}
        # The guest context, when the caller has it. It turns an absolute frame address into the
        # slot it names -- `[ctx+0x28]` and which guest register lives there -- which is both
        # readable and exact, and it is the first step of the guest-memory work a recompiler
        # would need: a backend cannot use `load64 [0x7ffffffef700]`, because that address is a
        # property of this process's stack and not of the guest's.
        self.ctx = ctx
        # **The guest register file this block starts from**, as `{reg: name}`, from
        # `core/xstate.py`. It is what turns a read of a context slot into the *name* of the value
        # that slot holds -- a value defined by an earlier block, or a phi at a join -- instead of
        # an absolute frame load. Without it the per-block listings are a bag of fragments that
        # happen to sit next to each other; with it a reader (and a backend) can follow a guest
        # register from one block to the next. See `events()`.
        self.entry: dict[str, str] = entry or {}
        # `{reg: node}` beside `entry`, because a partial write has to merge into the register's prior
        # *value*, and building that merge from a different input than the dataflow did yields a
        # different node -- which then has no definition in the block that emitted it. See
        # `xstate.entry_nodes` and `xstate.partial`.
        self.entry_nodes: dict[str, int] = {}
        self.register_nodes: dict[str, int] = {}
        # **Guest memory, at this block's entry.** `{(base register name, offset): value name}`, from
        # `core/xstate.py`, and this block's own stores are added as they are emitted. A read whose
        # key is present prints the name of the value that was stored -- which is the guest-memory
        # counterpart of what `entry` does for the register file -- and one whose key is absent is
        # read from memory as before. Keys match only exactly (same base name, same offset), so a
        # store that *might* alias never suppresses a read.
        self.guest_mem: dict[tuple, tuple] = {}
        # The block's number, so a store's value can be named `#3:v12` for later blocks.
        self.block_index = 0
        self.mem_reads_forwarded = 0
        self.mem_reads_from_memory = 0
        self.names: dict[int, str] = {}
        # Every definition this block emits, `{node: name}` -- including constants, which `names`
        # omits. See `_bind`.
        self.defs: dict[int, str] = {}
        self.lines: list[str] = []
        self._n = 0
        self._cond_n = 0
        self.cond_mode = False
        self._versions: dict[str, int] = {}
        self.registers: dict[str, str] = {}
        self.truncated = 0
        # Reads answered from the entry state, and reads that had to go to memory after all. Both
        # are exposed because "how much of the body is expressed in terms of the incoming state" is
        # a measurement of whether this is a program or an annotation.
        self.state_reads = 0
        self.memory_reads = 0

    def _ctx_rel_int(self, c: int | None) -> str:
        """`[ctx+0xNN = reg]` for a raw address that is a context slot, else empty.

        Takes a resolved address, not a node: `GuestContext.addr_of` returns an `int`, and the
        first version of this passed it to `as_const`, which indexes the node table.
        """
        if self.ctx is None or c is None:
            return ""
        off = c - self.ctx.base
        span = self.ctx.span[1] - self.ctx.base
        if not (0 <= off < span):
            return ""
        reg = self.ctx.reg_at(c)
        if reg is None:
            # **A byte or word slice of a slot, and the offset is the whole point.** `[ctx+0x04]`
            # says nothing; `[ctx+0x04 = rax+4]` is the high half of `rax`, which is exactly the case
            # where a reader (and an emitter) can go wrong -- a four-byte read at the slot base and
            # one four bytes in are different halves of the same register.
            inside = c - self.ctx.base
            slot = (inside // 8) * 8
            if inside % 8:
                owner = self.ctx.reg_at(self.ctx.base + slot)
                if owner:
                    return f"   [ctx+{off:#04x} = {owner}+{inside % 8}]"
        return f"   [ctx+{off:#04x}{' = ' + reg if reg else ''}]"

    def _ctx_rel(self, addr) -> str:
        """The same, for an address that is still a value node."""
        return self._ctx_rel_int(self.a.as_const(addr) if addr is not None else None)

    # **A separate numbering range for fork conditions, and it is not cosmetic.** `_n` runs across
    # the whole function, so naming a condition from the same counter shifts every temporary in every
    # *later* block -- which changed 2,732 lines of `0x140171e29` when the only intended change was to
    # *add* comment lines. A block-local identifier has no business perturbing its neighbours, and the
    # certified artifacts are only worth keeping if a change to them is auditable as an addition.
    # The base is far above any observed body temp count (largest root: ~20 000), and a collision
    # would surface as a duplicate temporary definition in the well-formedness checker rather than as
    # a silent aliasing bug.
    COND_BASE = 900_000

    def fresh(self, width: int | None) -> str:
        if self.cond_mode:
            self._cond_n += 1
            n = self.COND_BASE + self._cond_n
        else:
            self._n += 1
            n = self._n
        return f"v{n}:{width}" if width else f"v{n}"

    def _guest_rel(self, reg: str, off: int) -> str:
        """The guest-relative address of a memory event, for the comment only."""
        return guest_rel(reg, off) if reg else "guest memory"

    def _slot_inside(self, reg: str, addr, nbytes: int) -> int | None:
        """How far into the register's slot a narrow read lands, or None if that cannot be told.

        Little-endian, so the offset selects which bits were read: 0 is the low half, 4 the high half.
        `None` when the address is not concrete, because then the offset is unknown and the read must
        go to memory rather than be guessed -- a guess here reports the wrong half of a register, which
        is a *wrong* value rather than a missing one.
        """
        if self.ctx is None or addr is None:
            return None
        c = self.a.as_const(addr)
        if c is None:
            return None
        base = self.ctx.addr_of(reg)
        if base is None:
            return None
        inside = c - base
        return inside if 0 <= inside <= 8 - nbytes else None

    def _guest_rel_note(self, reg: str, off: int) -> tuple[str, bool]:
        """The guest-relative form of a memory event's address, **and whether it is one**.

        Printing `[guest_rcx+0x7ffffffef698]` would be worse than printing nothing: that offset is a
        host stack address, not a guest offset, so the line would present a decomposition as a
        guest-relative address it is not. The boundary is the address space, not a tuned constant --
        a guest offset fits in 32 bits and a host address does not -- and the two cases are counted
        separately because the second is a finding about the classification rather than a formatting
        choice. `PROBLEMS.md` B11 records the same shape from the load side: an address the VM built
        on its *own* stack, classified as guest traffic because it transitively depends on a guest
        register.
        """
        if not reg:
            return "", False
        if 0 <= off < (1 << 32):
            return f"   [{guest_rel(reg, off)}]", True
        return (f"   [base {reg} + {off:#x}: offset is host-sized, so this is not a "
                f"guest-relative address]", False)

    def _global_ref(self, name: str) -> str:
        """How a name defined *in this block* must be written to be resolvable from another block.

        **Not every rendered operand is a definition, and prefixing the ones that are not produced
        dangling references.** The independent IR checker found three lines like `#0:0x7ffffffef...`
        in a real listing: a stored *constant* had been recorded as if it were a block-local name, and
        a constant is an operand, not something a block defines. Three cases need no prefix --
        a literal, a VM-entry value (`in.rax`), and a value already qualified (`#k:...`) or a global
        register identity (`guest_rax`) -- and everything else does, because the block that emitted
        `v12:64` or `guest_r8_1` is the only one that defines it.
        """
        if name.startswith("#") or name.startswith("in."):
            return name
        if name[:2] == "0x" or name.isdigit():
            return name
        # `guest_<reg>` with no write suffix is the register's VM-entry value, which needs no block.
        _reg, owner_reg = split_guest_name(name)
        if name.startswith("guest_") and owner_reg is not None and "_" not in name[len("guest_"):]:
            return name
        return f"#{self.block_index}:{name}"

    def _mem_key(self, reg: str, off: int) -> tuple:
        """The guest-memory key for an access based on `reg`, **named as the state names it**.

        The register's *name* rather than the register, because that is what makes two accesses the
        same location: after `mov rcx, rbx` a store through `rcx` and a load through `rbx` share the
        key, and a block that has written `rax` twice keys the second write differently from the
        first. An unnamed base produces a key that can never match a state entry, which is the
        conservative direction -- it costs a forwarding opportunity and cannot produce a wrong value.
        """
        base = self.registers.get(reg) or self.entry.get(reg) or f"?{reg}"
        return (base, off)

    def _entry_operand(self, reg: str) -> str:
        """Bind an unnamed/merged boundary value as an explicit block input.

        A partial prior or unbound phi is a state parameter, not an invented
        predecessor expression. Display-only placeholders stay in comments.
        """
        existing = self.registers.get(reg) or self.entry.get(reg)
        if existing is not None and is_reference(existing):
            return existing
        raw = self.entry_nodes.get(reg)
        concrete = self.a.as_const(raw) if raw is not None else None
        if concrete is not None:
            return f'{concrete:#x}'
        name = self.fresh(64)
        self.lines.append(f'{name} = input64(state.{reg})   ; block entry backing value')
        self.entry[reg] = name
        return name

    def name(self, node: int, depth: int = 0) -> str:
        """Iterative post-order: expression depth is not a reason to lose semantics."""
        from .expr import _children
        stack = [(node, False)]
        visited = 0
        while stack:
            current, ready = stack.pop()
            if current in self.names or self.a.is_const(current):
                continue
            n = self.a.node(current)
            if n[0] in ('opaque', 'init', 'param'):
                continue
            if ready:
                self._name_ready(current)
                continue
            visited += 1
            if visited > MAX_NODES:
                self.truncated += 1
                nm = self.fresh(self.a.width(node))
                self.lines.append(f'{nm} = <expression exceeds {MAX_NODES} visited nodes>')
                self.names[node] = nm
                return nm
            stack.append((current, True))
            stack.extend((child, False) for child in reversed(_children(n)))
        return self._name_ready(node)

    def _name_ready(self, node: int) -> str:
        """Render a node whose composite children already have definitions."""
        c = self.a.as_const(node)
        if c is not None:
            return f"{c:#x}"
        if node in self.names:
            return self.names[node]
        # Global identity is historical: even a justified home-update describes
        # the result AFTER assignment. Only a definition already emitted in this
        # block may substitute a value. Otherwise keep the actual expression.
        n = self.a.node(node)
        op = n[0]
        if op == "opaque":
            return n[1]
        if op == "init":
            return f"in.{n[1]}"
        if op == "param":
            return leaf_name(self.a, node)
        w = self.a.width(node)
        fresh = self.fresh(w)
        if op == "load":
            addr = self._name_ready(n[1])
            # **`n[2]` is already in bits.** `Arena.load(addr, width)` takes a width, and
            # `mach.py` calls it as `a.load(addr, nbytes * 8)`. The previous version of this line
            # multiplied by 8 again and printed `load512` for a 64-bit load, because it assumed
            # the node stored bytes. Two units were in play in this one function: the trace events
            # carry **bytes** (the lifter's `nbytes`) and the arena nodes carry **bits**. Both are
            # now named explicitly at the point of use rather than inferred from a variable name.
            self.lines.append(f"{fresh} = load{n[2]} [{addr}] @m{n[3]}")
        elif op == "bin":
            x = INFIX.get(n[1])
            a = self._name_ready(n[2])
            b = self._name_ready(n[3])
            self.lines.append(f"{fresh} = {a} {x} {b}" if x
                              else f"{fresh} = {n[1]}({a}, {b})")
        elif op == "un":
            a = self._name_ready(n[2])
            sym = PREFIX.get(n[1])
            self.lines.append(f"{fresh} = {sym}{a}" if sym
                              else f"{fresh} = {n[1]}({a})")
        elif op in ("trunc", "zext", "sext"):
            a = self._name_ready(n[1])
            self.lines.append(f"{fresh} = {op}{n[2]}({a})")
        elif op == "sel":
            cnd = self._name_ready(n[1])
            t = self._name_ready(n[2])
            f = self._name_ready(n[3])
            self.lines.append(f"{fresh} = {cnd} ? {t} : {f}")
        else:
            self.lines.append(f"{fresh} = <unlowered {op}>")
        self.names[node] = fresh
        return fresh

    # --------------------------------------------------------------- block level

    def flags_pack(self, node: int):
        """Recognize an architectural flags assembly, and **prove** it is one before collapsing it.

        The VM builds the flags word by hand, one flag at a time:

            v1:64 = zext64(init_cf)      v2:64 = 0x1 & v1:64      v5:64 = 0x2 | v4:64
            v8:64 = v7:64 << 0x2         v9:64 = v5:64 | v8:64    ... 25 lines for one word

        That is a quarter of a listing's line count for an operation a reader already knows the
        meaning of, and it hides the one thing that matters about it -- which flags went in.

        This is not a pattern match. Every leaf must be a flag, and the node is then **evaluated on
        every assignment of those leaves** through `substitute`; the assembly is accepted only when
        each single-flag evaluation differs from the all-zero evaluation by exactly one bit, that bit
        is the flag's architectural position, and every assignment reproduces the linear form. A
        pack that reads a call return is refused, which is the common case for the flags word and the
        one where collapsing would be a fabrication.

        Returns `(text, positions)` or None.
        """
        a = self.a
        leaves = a.leaves(node)
        if not leaves or len(leaves) > 8:
            return None
        named: list[tuple[int, str]] = []
        for lf in leaves:
            n = a.node(lf)
            if n[0] not in ("init", "opaque"):
                return None
            raw = n[1]
            for prefix in ("init_", "flag_"):
                if raw.startswith(prefix) and raw[len(prefix):] in FLAG_BITS:
                    named.append((lf, raw[len(prefix):]))
                    break
            else:
                return None
        if not named:
            return None
        zero = {lf: a.const(0, a.width(lf)) for lf, _f in named}
        try:
            base = a.as_const(a.substitute(node, zero))
        except (TypeError, IndexError):
            return None
        if base is None:
            return None
        pos: dict[str, int] = {}
        for lf, flag in named:
            one = dict(zero)
            one[lf] = a.const(1, a.width(lf))
            v = a.as_const(a.substitute(node, one))
            if v is None:
                return None
            delta = v ^ base
            if delta == 0 or (delta & (delta - 1)) != 0:
                return None
            bit = delta.bit_length() - 1
            if bit != FLAG_BITS[flag]:
                return None
            pos[flag] = bit
        # Every assignment, not a sample: with at most eight leaves that is 256 evaluations of a
        # twenty-five node DAG, and an equivalence that is checked rather than argued.
        for mask in range(1 << len(named)):
            env = {lf: a.const((mask >> i) & 1, a.width(lf))
                   for i, (lf, _f) in enumerate(named)}
            got = a.as_const(a.substitute(node, env))
            if got is None:
                return None
            want = base
            for i, (_lf, flag) in enumerate(named):
                if (mask >> i) & 1:
                    want |= 1 << pos[flag]
            if got != want:
                return None
        order = [f for f in FLAG_BITS if f in pos]
        args = ", ".join(f"{f}={self._flag_operand(node, f, named)}" for f in order)
        text = f"pack_flags({args})"
        if base != 0x2:
            text += f" | {base:#x}"
        return text, tuple(sorted(pos.values()))

    def _flag_operand(self, node: int, flag: str, named) -> str:
        """How this pack's `flag` input is spelled in the IR."""
        for lf, f in named:
            if f != flag:
                continue
            return leaf_name(self.a, lf)
        return flag

    def _destination(self, reg: str) -> str:
        version = self._versions.get(reg, 0) + 1
        self._versions[reg] = version
        name = f"guest_{reg}_{version}"
        self.registers[reg] = name
        return name

    def _bind(self, value, name):
        # Constants stay constants; all other bindings refer to an existing def.
        #
        # **`defs` records the definition even when `names` does not.** `names` skips constants so
        # the listing inlines them, but a caller asking "which block defines this value, and under
        # what name" needs the constant case too -- `guest_r8_2 = 0xb` is a real definition, and
        # leaving it out made the dataflow refuse to name a register whose value happened to be a
        # literal. `defs` is what `core/xstate.py` collects its `owners` map from, so a name it uses
        # is a definition that exists rather than one it predicted.
        self.defs[value] = name
        if not self.a.is_const(value):
            self.names[value] = name

    def events(self, block, order: bool = True) -> list[str]:
        """The lowered guest effect of one block, in execution order.

        `block.guest_trace` carries `(step, kind, payload)` for every guest event the census
        recorded while lifting the block. That ordering is the whole reason the trace exists:
        the four per-kind lists answer attribution questions and cannot be interleaved.

        **The per-block counters are reset before the empty-trace return, not after it.** They used to
        be initialised only on the path that had a trace, so a block with no guest trace returned `[]`
        and left the attributes undefined for the caller: `core/devirt.py:417` reads
        `lower.packs_collapsed` after every block, and on `ACE-CORE.sys3` -- a kernel driver whose 4th
        root has blocks with an empty trace -- that raised
        `AttributeError: 'Lowering' object has no attribute 'packs_collapsed'` and aborted the whole
        module's devirtualisation after four of 144 roots. A block with no trace has zero of
        everything, so resetting first makes `events` total.
        """
        out: list[str] = []
        self.entry = dict(self.entry)
        self.names = {}
        self.defs = {}
        self.registers = {}
        self.register_nodes = dict(self.entry_nodes)
        # **Per block, and it has to be.** `_destination` numbers a register's writes, and a
        # cross-block reference is written `#N:guest_<reg>_<n>` -- with `n` counted *within* block N.
        # Leaving this counter running across blocks made the numbers global, so `#0:guest_rflags_35`
        # could name a write that block 1 made, and the independent checker found the dangling
        # reference that follows. It also made the names depend on how many blocks a *previous* pass
        # had already lowered, which is why the owner map and the final emission disagreed.
        self._versions = {}
        # Per block, like the names above: a caller summing them over a function would otherwise be
        # adding cumulative totals and reporting a number that is not a count of anything.
        self.state_reads = 0
        self.memory_reads = 0
        self.mem_reads_forwarded = 0
        self.mem_reads_from_memory = 0
        # A key that matched but whose width did not: the audit's second reproduction, counted
        # separately from "nothing was parked at that location".
        self.mem_forward_refused = 0
        # Memory events whose address really is guest-relative, and those whose "offset" is a host
        # address -- see `_guest_rel_note`. The second is a classification finding, not a formatting
        # one, so it is counted rather than smoothed over.
        self.mem_rel_ok = 0
        self.mem_rel_host = 0
        # Flags assemblies collapsed from their expansion, and the lines that saved. Both are
        # reported, because a compression whose size is not measured cannot be audited.
        self.packs_collapsed = 0
        self.pack_lines_saved = 0
        trace = getattr(block, "guest_trace", None) or []
        if not trace:
            return []
        for _step, kind, payload in sorted(trace, key=lambda t: t[0]):
            # Values have immutable node ids. Keep their earlier definitions so
            # a load result remains the same value when later events use it.
            self.lines = []
            if kind == "load":
                site, reg = payload[0], payload[1]
                addr = payload[3] if len(payload) > 3 else None
                nbytes = payload[4] if len(payload) > 4 else 8
                value = payload[2]
                # **A read of a slot whose value already has a name *is* that name.** If the block
                # has written the register already, it is this block's version; otherwise it is
                # what the register held on entry, which `core/xstate.py` propagated here from the
                # predecessors -- a value from an earlier block, or a phi at a join. That is the
                # difference between a listing that annotates effects and one that binds them:
                # `load64 [0x7ffffffef6a8]` is a property of this process's stack, while
                # `#1:guest_rbx_2` is a fact about the guest that a reader can follow across blocks.
                named = self.registers.get(reg) if reg else None
                if named is None and reg:
                    named = self.entry.get(reg)
                if named is not None and not is_reference(named):
                    named = self._entry_operand(reg)
                if named is not None and nbytes == 8:
                    out.extend(self.lines)
                    self.state_reads += 1
                    if not is_reference(named):
                        # A join or a placeholder is an expression, so it needs a destination: see
                        # `is_reference`. This is the one place the two shapes have to be told
                        # apart, and telling them apart by looking for a parenthesis would have
                        # missed `<value n62233>`.
                        dest = self.fresh(64)
                        out.append(f"{dest} = {named}   ; read from ctx @{site:#x}"
                                   f"{self._ctx_rel(addr) if addr is not None else ''}")
                        named = dest
                    else:
                        out.append(f"{named}   ; read from ctx @{site:#x}"
                                   f"{self._ctx_rel(addr) if addr is not None else ''}")
                    self._bind(value, named)
                    continue
                if named is not None and 0 < nbytes < 8:
                    # **A narrow read of a slot is a truncation of the state, not a frame load.**
                    # It cannot be a bare name -- the low four bytes of `guest_rax` are not
                    # `guest_rax` -- but it can be expressed in terms of it, which removes the
                    # absolute frame address and links the value to the register it came from.
                    #
                    # **The in-slot offset is the part that is easy to get wrong, and the first
                    # version did.** `_guest_slot` accepts byte and word *slices*, so a four-byte
                    # read may land at the slot base (the low half) or four bytes in (**the high
                    # half**), and emitting `trunc32(guest_rax)` for both would silently report the
                    # low half where the machine read the high one. Little-endian, so a read at
                    # `inside` bytes into the slot is bits `8*inside` upward.
                    inside = self._slot_inside(reg, addr, nbytes)
                    if inside is not None:
                        out.extend(self.lines)
                        self.state_reads += 1
                        dest = self.fresh(nbytes * 8)
                        src = named if inside == 0 else f"({named} >> {inside * 8})"
                        out.append(f"{dest} = trunc{nbytes * 8}({src})"
                                   f"   ; read from ctx @{site:#x}"
                                   f"{self._ctx_rel(addr) if addr is not None else ''}")
                        self._bind(value, dest)
                        continue
                src = self.name(addr) if addr is not None else "guest memory"
                out.extend(self.lines)
                self.memory_reads += 1
                # **A read is not a write, so it does not take a register version.** This used
                # `_destination(reg)`, which named the loaded value `guest_<reg>_<n>` as if the
                # register had been updated -- and, worse, made the numbering depend on whether the
                # read was answered from the incoming state or from memory, so the same block's
                # versions shifted with the dataflow result. The tag `core/xstate.py` attaches to a
                # written value is a count of *writes*, and this is what keeps the two in step.
                dest = self.fresh(nbytes * 8)
                out.append(f"{dest} = load{nbytes * 8} [{src}]"
                           f"{self._ctx_rel(addr) if addr is not None else ''}"
                           f"   ; read from ctx @{site:#x}")
                self._bind(value, dest)
            elif kind == "store":
                site, reg, value = payload
                # **A flags assembly that is provably one becomes one line.** See `flags_pack`: the
                # 25 lines it replaces are a deterministic expansion, so nothing is lost, and the
                # one fact a reader wants -- which flags went in -- is what the compact line says.
                # A pack this refuses is emitted in full, which is the honest outcome for a flags
                # word built from a call return.
                if reg == "rflags":
                    pack = self.flags_pack(value)
                    if pack is not None:
                        text, positions = pack
                        out.extend(self.lines)
                        # What the expansion *would* have cost: one line per composite node. Measured
                        # rather than guessed, because `self.lines` is reset per event and is empty
                        # here -- the count from it was silently zero.
                        expansion = sum(
                            1 for nd in self.a._descendants_ordered(value)
                            if self.a.node(nd)[0] not in ("c", "init", "param", "opaque"))
                        dest = self._destination(reg)
                        self._bind(value, dest)
                        out.append(f"{dest} = {text}   ; RFLAGS bits {positions}")
                        self.packs_collapsed += 1
                        self.pack_lines_saved += max(0, expansion - 1)
                        continue
                v = self.name(value)
                out.extend(self.lines)
                # **Not "guest memory write".** A `store` event fires when the value goes into
                # the context slot that the derived layout calls `reg`, so this is the guest
                # register being updated -- labelling it a memory write would misname the one
                # operation the whole recovery is about.
                slot = self.ctx.addr_of(reg) if self.ctx is not None else None
                rel = self._ctx_rel_int(slot)
                dest = self._destination(reg)
                out.append(f"{dest} = {v}{rel}   ; write to ctx @{site:#x}")
                self._bind(value, dest)
                self.register_nodes[reg] = value
            elif kind == "home":
                guest, host, value = payload
                v = self.name(value)
                out.extend(self.lines)
                dest = self._destination(guest)
                out.append(f"{dest} = {v}   ; home {host}")
                self._bind(value, dest)
                self.register_nodes[guest] = value
            elif kind == "carry":
                guest, host, _prev, value = payload
                v = self.name(value)
                out.extend(self.lines)
                dest = self.fresh(self.a.width(value))
                out.append(f"{dest} = {v}   ; from guest_{guest} via {host}")
                self._bind(value, dest)
            elif kind == "gload":
                # **Two units meet here.** The trace payload carries the lifter's `nbytes`, which
                # is *bytes*; an arena load node carries *bits*. They are not interchangeable, and
                # conflating them is what produced `load512` in the node path.
                site, reg, off, nbytes, value = payload[:5]
                # **Forwarding is conditional, and the conditions are the four the audit
                # reproduced.** A matching key is necessary and nowhere near sufficient: the width
                # must match (a one-byte write does not answer an eight-byte read), the entry must
                # not have been invalidated by an overlapping or possibly-aliasing write, and nothing
                # may have been parked across a native call. `_walk` maintains the map under those
                # rules; this side re-checks the width because the map is per-location and the read
                # width is known only here.
                entry = self.guest_mem.get(self._mem_key(reg, off))
                if entry is not None and entry[1] == nbytes:
                    out.extend(self.lines)
                    self.mem_reads_forwarded += 1
                    out.append(f"{entry[0]}   ; read from guest memory @{site:#x}"
                               f"   [{guest_rel(reg, off)}]")
                    self._bind(value, entry[0])
                    continue
                self.mem_forward_refused += 1 if entry is not None else 0
                addr = self.name(payload[5]) if len(payload) > 5 else guest_rel(reg, off)
                out.extend(self.lines)
                self.mem_reads_from_memory += 1
                dest = self.fresh(nbytes * 8)
                # **The census knows which guest pointer and offset this address is, and the line
                # used to throw that away** whenever the address was a computed node: `[v5071:64]`
                # says nothing, while `[guest_rcx+0x18]` is the guest-relative address a recompiler
                # can use. The node stays the operand -- it is what was actually computed -- and the
                # guest form is stated beside it rather than replacing it, because the two are only
                # the same when the node *is* that expression.
                rel, credible = self._guest_rel_note(reg, off)
                self.mem_rel_ok += 1 if credible else 0
                self.mem_rel_host += 0 if credible else 1
                out.append(f"{dest} = load{nbytes * 8} [{addr}]"
                           f"   ; guest memory read @{site:#x}{rel}")
                self._bind(value, dest)
            elif kind == "gstore":
                site, reg, off, nbytes, value = payload[:5]
                addr = self.name(payload[5]) if len(payload) > 5 else guest_rel(reg, off)
                v = self.name(value)
                out.extend(self.lines)
                rel, credible = self._guest_rel_note(reg, off)
                self.mem_rel_ok += 1 if credible else 0
                self.mem_rel_host += 0 if credible else 1
                out.append(f"store{nbytes * 8} [{addr}], {v}"
                           f"   ; guest memory write @{site:#x}{rel}")
                # **Record it, so a later read of the same location is that name** -- subject to the
                # same invalidation rules the dataflow applies, which this side has to mirror because
                # in-block reads are answered here rather than from the propagated state.
                for other in [k for k in self.guest_mem
                              if k[0] != self._mem_key(reg, off)[0]
                              or (k[1] < off + nbytes and off < k[1] + self.guest_mem[k][1])]:
                    del self.guest_mem[other]
                self.guest_mem[self._mem_key(reg, off)] = (self._global_ref(v), nbytes)
            elif kind == "part":
                # A narrow store into a context slot: it writes part of the register's home and
                # leaves the rest. Rendered as a sub-range so the listing does not claim the whole
                # register was replaced.
                #
                # **The prior value comes from the entry state when the block has not written the
                # register yet.** `param.<reg>` was the old fallback, and it is a name with no
                # definition anywhere -- the upper bytes a partial write preserves are exactly the
                # bytes the incoming state holds, so the state's name is both correct and followable.
                site, reg, lo, nbytes, value = payload
                v = self.name(value)
                previous = self._entry_operand(reg)
                out.extend(self.lines)
                dest = self._destination(reg)
                out.append(f"{dest} = insert_bits({previous}, {lo * 8}, {nbytes * 8}, {v})"
                           f"   ; PARTIAL write to ctx @{site:#x}")
                # **The merged node is recorded, and it is the *same* node the dataflow holds.** A
                # partial write's result used to be bound to nothing, so the value the state carried
                # had no definition anywhere and its name was refused -- 25 of 306 entry-state values
                # on `0x140155dc1`, the flags slot being the worst because it is written four bytes at
                # a time. `xstate.partial` is the one implementation of the merge, used by both sides.
                prev_node = self.register_nodes.get(reg)
                if prev_node is not None:
                    merged = partial(self.a, prev_node, lo, nbytes, value)
                    self.defs[merged] = dest
                    self.register_nodes[reg] = merged
            elif kind == "call":
                # **A native call may write guest memory, so nothing parked before it survives.**
                # This branch used to be shadowed by a second `elif kind == "call"` added below it,
                # which silently deleted the call *line* from every listing -- `test_lowering_repair`
                # caught it as a missing `test!call`. One branch, both duties.
                self.guest_mem.clear()
                args = [f"{reg}={self.name(value)}" for reg, value in payload.get('args', [])]
                out.extend(self.lines)
                ret = payload['ret']
                dest = self.fresh(self.a.width(ret))
                out.append(f"{dest} = call {payload['name']}({', '.join(args)})"
                           f"   @{payload['site']:#x}")
                self._bind(ret, dest)
            else:
                out.append(f"; unknown guest event {kind}")
        return out
