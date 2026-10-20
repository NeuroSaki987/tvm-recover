"""x86-64 semantics for the VM's own code, at the granularity dispatch needs.

This is not a general-purpose emulator. It models the AMD64 subset that the TVM
handler code is built from -- data movement, arithmetic, the obfuscator's
inverted-register idiom, stack, flags, and control transfer -- and *boxes*
everything else: an unmodelled instruction is recorded with its operands turned
into opaque values and evaluation continues. Boxing is what makes the engine
robust across ACE builds, where a handler may contain an instruction this file
has never seen but which cannot influence dispatch.

The one deliberate fidelity choice is that every partial-register write follows
x86 exactly. The protector mixes `mov rax` / `mov eax` / `mov ax` on the same
register inside a `not`/`xchg` dance; a lifter that treats them all as 64-bit
produces a wrong dispatch address rather than merely a less precise one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from capstone import CS_ARCH_X86, CS_MODE_64, CS_OP_IMM, CS_OP_MEM, CS_OP_REG, Cs

from .expr import W8, W16, W32, W64, Arena, boolean_subexpressions, sext_value
from .guestctx import MATERIALISATION, GuestCache, GuestContext
from .mach import GP64 as GP64_NAMES
from .mach import SUB, FLAGS, Machine, UnmodelledRegister, canon, condition
from . import x86sum

JCC = {
    "jo": "o", "jno": "no", "jb": "b", "jc": "b", "jnae": "b", "jae": "a", "jnb": "ae",
    "jnc": "ae", "je": "e", "jz": "e", "jne": "ne", "jnz": "ne", "jbe": "be", "jna": "be",
    "ja": "a", "jnbe": "a", "js": "s", "jns": "ns", "jp": "p", "jpe": "p", "jnp": "np",
    "jpo": "np", "jl": "l", "jnge": "l", "jge": "ge", "jnl": "ge", "jle": "le",
    "jng": "le", "jg": "g", "jnle": "g",
}
SETCC = {
    "seto": "o", "setno": "no", "setb": "b", "setc": "b", "setnae": "b", "setae": "ae",
    "setnb": "ae", "setnc": "ae", "sete": "e", "setz": "e", "setne": "ne", "setnz": "ne",
    "setbe": "be", "setna": "be", "seta": "a", "setnbe": "a", "sets": "s", "setns": "ns",
    "setp": "p", "setpe": "p", "setnp": "np", "setpo": "np", "setl": "l", "setnge": "l",
    "setge": "ge", "setnl": "ge", "setle": "le", "setng": "le", "setg": "g", "setnle": "g",
}
CMOVCC = {
    "cmovo": "o", "cmovno": "no", "cmovb": "b", "cmovc": "b", "cmovnae": "b",
    "cmovae": "ae", "cmovnb": "ae", "cmovnc": "ae", "cmove": "e", "cmovz": "e",
    "cmovne": "ne", "cmovnz": "ne", "cmovbe": "be", "cmovna": "be", "cmova": "a",
    "cmovnbe": "a", "cmovs": "s", "cmovns": "ns", "cmovp": "p", "cmovpe": "p",
    "cmovnp": "np", "cmovpo": "np", "cmovl": "l", "cmovnge": "l", "cmovge": "ge",
    "cmovnl": "ge", "cmovle": "le", "cmovng": "le", "cmovg": "g", "cmovnle": "g",
}

# Mnemonics that must never be boxed: they are control transfer or traps, and
# treating them as ordinary data movement would send evaluation somewhere the
# VM cannot actually go.
UNBOXABLE = frozenset(
    {"iret", "iretd", "iretq", "sysenter", "sysexit", "syscall", "sysret", "sysretq",
     "int", "int3", "int1", "into", "ud0", "ud1", "ud2", "hlt", "hltq"}
)

# Instructions whose memory operand is only ever read. Used by the boxed path so
# that `nop dword ptr [rax]` does not erase a guest variable.
#
# The SSE moves were removed from this set when they gained handlers: listing a
# store here is what made `movdqu [rsp+X], xmm0` discard its write silently. What
# remains are the instructions that genuinely never write the memory they name --
# and `movd`/`movq`-family forms that reach the boxed path after all are now
# invalidated rather than preserved, which is the conservative direction.
_READ_ONLY_MEM = frozenset(
    {"nop", "cmp", "test", "bt", "jmp", "call", "push", "prefetch", "prefetchnta",
     "prefetcht0", "prefetcht1", "prefetcht2", "lea", "movntdqa",
     "pinsrq", "pinsrd", "pextrq", "pextrd", "movbe"}
)

# Instructions that touch only the CET shadow stack or the pipeline. The VM emits
# them around its own non-returning calls; they have no guest-visible semantics,
# so treating them as no-ops is exact, not an approximation. The reference
# implementation reaches the same conclusion through a `assume_cet_disabled`
# switch.
_CET_NOOP = ("incsspq", "rdsspq", "saveprevssp", "rstorssp", "wrssq", "wrussq",
             "clrssbsy", "setssbsy", "endbr64", "endbr32")


@dataclass
class Stop:
    kind: str
    site: int
    dest: int | None = None           # Ref
    pred: int | None = None           # Ref
    taken: int = 0
    not_taken: int = 0
    text: str = ""
    info: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Stop({self.kind}@{self.site:#x})"


@dataclass
class Boxed:
    site: int
    text: str
    bytes_: bytes
    defs: list[tuple[str, int]]
    uses: list[tuple[str, int]]


class Lifter:
    #: Whether a bare `int3` counts as the VM leaving the virtualized function, when its
    #: shape says so. See `Lifter._guest_exit_trap`. Switchable so the A/B stays available.
    guest_exit_trap = True
    # A saved-stack restore and complete re-entry frame identify the guest ABI
    # boundary. The callee remains symbolic; this can be disabled for A/B probes.
    computed_reentry_calls = True

    """Single-steps `.tvm0` code over a `Machine`, stopping where control is opaque."""

    # Default for AF after a logical operation: unknown (see `_logic_flags`).
    logical_af_zero = False

    # Class-level so that a probe can force native-call summaries off with one
    # assignment and A/B the same recovery. An optimisation that cannot be
    # switched off cannot be shown to earn its place.
    #
    # **Off by default, because the A/B came out flat.** With summaries on, the
    # module recovers 733 blocks / 27 unresolved / 1,098,044 instructions across
    # its fourteen functions; with them off, exactly the same three numbers. Sixteen
    # callees *are* interpreted to a real value in 33,331 steps, and none of those
    # values changes the recovered graph. Three percent of the run for a measured
    # zero is not worth paying by default, so the module stays available, wired in
    # and counted (`--nosum` is the control) rather than deleted: it is correct,
    # bounded, and the reject reasons below are the map of what is still unknown.
    # **Off was the wrong default, and it made a whole subsystem invisible.** With
    # `summaries_default = False`, every corpus figure in this workspace was measured with the
    # intra-image summary interpreter switched off -- a module with its own rejection counters that
    # had never run in an acceptance run. Turning it on costs nothing measurable in coverage and is
    # strictly more exact where it succeeds: a summarized callee performs its own stores on the
    # machine, so its write set is *exact* rather than forgotten. Measured on the corpus:
    # `--sum --nocallmem` is 14/14, the same as with summaries off.
    summaries_default = True
    # The per-callee interpretation cap, and the run-wide total. Both are class defaults read in
    # `__init__` rather than literals, because a literal cannot be varied from outside: a probe that
    # set `Lifter.summary_limit` had no effect at all, since `__init__` overwrote it with `4000`, and
    # the measurement silently reported the same figure for a 20 000-step limit as for 400 000. The
    # limit is the difference between "this callee cannot be summarized" and "it was not given the
    # steps", which is the difference between a capability and a stub.
    summary_limit_default = 4000
    summary_budget_default = 400_000

    def __init__(self, img, machine: Machine, arena: Arena, vm_lo: int, vm_hi: int,
                 box_unmodelled: bool = True):
        self.img = img
        self.m = machine
        # The `not`/`xchg` deobfuscation, installed on the machine so `set_reg` can
        # report writes without `mach.py` needing to know what they are for.
        try:
            from .mba import SwapTracker
            self.swaps = SwapTracker(machine)
            machine.swap_tracker = self.swaps
        except Exception:                              # pragma: no cover
            self.swaps = None
        self.a = arena
        # Small constants the flag builders ask for on **every** arithmetic instruction, kept as
        # node ids. `a.const(k, width)` builds a tuple and hashes it, and a profile of one
        # 317 k-instruction root put **10.5 M `const` calls against 5.5 M `bin` calls** -- the flag
        # helpers were asking the interner for the same sixteen constants hundreds of thousands of
        # times. Node ids are stable inside a run (the arena is append-only and nothing rewrites a
        # node), so this is a cache and not a second model: `a.const(k, width)` would return the
        # same id, it would just cost a hash to say so.
        self._consts: dict[int, tuple[int, ...]] = {}
        self.vm_lo = vm_lo
        self.vm_hi = vm_hi
        self.box_unmodelled = box_unmodelled
        self._md = Cs(CS_ARCH_X86, CS_MODE_64)
        self._md.detail = True
        self._cache: dict[int, tuple[int, object, bytes]] = {}
        self.events: list[object] = []
        self.boxed: list[Boxed] = []
        # (site, destination) for every *indirect* control transfer that folded
        # to a concrete address. These are the VM's handler transitions: the
        # bytecode program, read out of the protection. Recording them at the
        # jump rather than inferring them from a block graph is what makes the
        # engine directly comparable to an independent tracer.
        self.dispatches: list[tuple[int, int]] = []
        self.steps = 0
        self.trace: list[int] = []
        self._current_ip = None
        self._previous_ip = None
        self.call_count = 0
        self.imports = img.imports
        if hasattr(img, 'section_for_va'):
            def readonly(addr, size):
                section = img.section_for_va(addr)
                return (section is not None and not section.writable
                        and addr + size <= section.end
                        and not img.is_loader_bound(addr, size))
            machine.readonly = readonly
        # A `.tvm0` pointer the VM parks at [rbp+0] for the unwinder. Set by the
        # recovery driver once measured; excluded from block identity because it
        # never moves.
        self.unwind_slot: int | None = None
        # Native-call summaries. See `x86sum.py`: a callee in this image's own
        # `.text` is interpreted to its return value on a trial basis, and rolled
        # back completely if any step is undecidable, so the caller falls back to
        # the opaque behaviour. The depth and budget caps bound the cost, and
        # `summary_rejects` records *why* summaries were abandoned -- the reason
        # is the interesting part.
        self.summary_enabled = self.summaries_default
        self.summary_max_depth = 3
        self.summary_limit = self.summary_limit_default
        self.summary_budget = self.summary_budget_default
        self.summary_depth = 0
        self.summary_steps = 0
        self.summary_ok = 0
        self.summary_rejects: dict[str, int] = {}
        # Direct-call targets that are one-instruction thunks. See
        # `_resolve_call_thunk`; `-1` means "not looked at yet", None means "not
        # a thunk".
        self._thunk_cache: dict[int, int | None] = {}
        # Calls through a writable image slot the program fills at run time, by
        # slot address. See `_runtime_bound_slot`. Counted rather than merely
        # permitted, because "how often is this rule the reason a block closed"
        # is the question that decides whether it is worth having.
        self.runtime_bound_calls: dict[int, int] = {}

        # ------------------------------------------------- guest-context census
        #
        # The VM keeps the guest register file at a fixed offset from its own
        # frame, so a load or store that lands on a slot **is** a guest register
        # read or write, and naming it turns "the handler read 8 bytes at
        # 0x7ffffffef338" into "the handler read the guest `rsi`". See
        # `core/guestctx.py` for where the block is and how it was located.
        #
        # **Off by default, and measured.** Wrapping the machine costs a Python
        # call on every memory access of every lifted instruction, for a census
        # most runs do not read, so it is installed by `enable_guest_census`
        # rather than in the constructor -- the same way `x86sum` is off by
        # default for the same reason.
        #
        # The cost is **1-2 %**, and the first figure written here was wrong. It
        # said five percent, from comparing 18.2 s before the change against
        # 19.1 s after -- two runs in different machine states, which is not an
        # A/B. Interleaving `census=False, True, False, True` inside one process
        # gives 20.09 / 20.54 / 20.54 / 20.64 s on the benchmark root: the whole
        # box had drifted by 1.7 s between sessions, and that drift was larger
        # than the effect being claimed. Recovery output is identical either way.
        #
        # When on, the machine is wrapped rather than the nine call sites edited:
        # a store reaches memory through `write_op`, three push/pop helpers and
        # the call helper, and a census that has to be added to each of them is a
        # census with holes in it.
        self.guest_ctx: GuestContext | None = None
        self.guest_reads: list[tuple[int, str]] = []
        self.guest_writes: list[tuple[int, str, int]] = []
        self.guest_cache = GuestCache(MATERIALISATION)
        # Which *values* are guest state: node -> guest register. Keyed on the value
        # node rather than on a register, and that is the whole design.
        #
        # Keying on the register does not work, and the measurement said so: with
        # register-keyed provenance the attribution covered **the same 5 blocks** that
        # already write the context, out of 149. The handler-time cache is exactly a
        # value *persisting across dispatches*, and register provenance has to be
        # discarded at every block boundary -- a machine snapshot restores registers,
        # not a side table -- which destroys precisely the case it exists for. Keying
        # on the node has neither problem: node ids are globally unique in the shared
        # arena, so an entry can never leak from one block's state to another's, and
        # no invalidation is needed because nothing is restored over it.
        #
        # `add(load(ctx+rsi), 8)` has the load among its leaves, so a value derived
        # from a guest register propagates without any extra bookkeeping.
        self.guest_values: dict[int, str] = {}
        # **Why each mark exists, because `guest_values` is a mixture of two relations and
        # that was invisible until round 37.** There are four places that write the map and
        # they do not mean the same thing:
        #
        #   `context-load`   the value that came out of slot `G` **is** `G`   -- identity
        #   `home-update`    the new value in `G`'s home **is** `G`           -- identity
        #   `context-store`  the value going *into* slot `G` is assigned to `G`, which is a
        #                    statement about where it is going, not about what it is
        #   `derived`        a value computed from exactly one guest value, marked so the
        #                    trail survives a step through scratch -- traceability, not identity
        #
        # Every consumer has been reading the whole map as identity, including
        # `render_value`, which substitutes the marked leaf and asserts the result is a
        # function of `guest_<g>`. This records the mechanism so the two can be separated
        # and the consequence measured rather than argued.
        self.gv_prov: dict[int, tuple[str, int]] = {}
        # **The first mechanism, which is the informative one.** A node marked by a store
        # and later returned by a load keeps only the load in `gv_prov`, because the load
        # overwrites -- and that hides the fact that a *store* is what put the value there.
        # The first version of this recorded one mechanism and therefore could not tell
        # "the slot holds this value" (load) from "this value was assigned to that slot"
        # (store), which is the whole question the provenance exists to answer.
        self.gv_first: dict[int, tuple[str, int]] = {}
        # **The identity relation, stored separately from the map that gets overwritten.**
        #
        # `guest_values` holds one register per node and is written by four mechanisms, so a
        # traceability mark can overwrite an identity claim about the same value: the block
        # seeding asserts "host rbx currently holds guest rbx" and a later `derived` mark on the
        # same node replaces it, after which nothing can tell that an identity was ever
        # justified. That is why round 39's guest-memory criterion found 166 identity marks in
        # 4464 and recovered almost no guest accesses. An identity claim is about the *value*
        # and does not stop being true when the register it arrived in is reused, so it is
        # recorded here where nothing overwrites it.
        self.guest_identity: dict[int, str] = {}
        # Stores whose address is a direct guest pointer, i.e. the ones the VM's private frame is
        # treated as disjoint from. Exposed so the assumption's blast radius is a number rather
        # than a claim. See `_guest_store`.
        self.guest_pointer_stores = 0
        # Candidates `_load_via_predicate_split` proved cannot fold the address and therefore never
        # substituted. A number, not a claim: it is the pruning that the speed-up consists of.
        self.split_pruned = 0
        # (guest register, byte width) for every read of a context slot. See `_guest_load`: this is
        # what decides whether a slot's narrow writes are complete or partial, and it is measured
        # rather than assumed per slot.
        self.slot_read_widths: list[tuple[str, int]] = []
        # (step, guest register, host register, new value) for every host-register
        # write whose value depended on exactly one known guest value.
        self.guest_reg_writes: list[tuple[int, str, str, int]] = []
        # Loads whose address is a guest register plus an offset -- a guest memory read.
        # See `load_at`. Gated on the census because the test costs a leaf walk per load.
        self.guest_loads: list[tuple[int, int]] = []
        # **The guest events in execution order, which is what a listing needs.**
        #
        # `guest_writes`, `guest_reads`, `guest_reg_writes` and `guest_derived` are four lists
        # that each lose the interleaving: a block that reads guest memory, computes with it and
        # then writes it back produces one entry in each, and nothing said in which order. A
        # devirtualized listing is an *ordered* stream, so the order is recorded once, here, as
        # `(step, kind, payload)`. The four lists are kept because attribution and the census
        # ask per-kind questions; this is the one a reader asks.
        self.guest_trace: list[tuple[int, str, tuple]] = []
        # Explicit origins, distinct from names of incoming unknown values.
        # Attempts roll these back together with the machine observations.
        self.local_effects: list[tuple[int, dict]] = []
        # (site, address, nbytes, value) for **every** store to a concrete address, context
        # window or not. See `_guest_store`; the control for round 38's classifier.
        self.all_stores: list[tuple[int, int, int, int]] = []
        self.guest_load_rejects: dict[str, int] = {}
        # Why candidate attributions were refused, by reason, so the filter's
        # contribution is a number rather than a claim.
        self.grw_rejected: dict[str, int] = {}
        # (step, guest register, host register, the value it held, the new value) for
        # every write that computed with a *copy* of a guest register rather than
        # updating it. See `_guest_set_reg`; this is the "carrying" quantity separated
        # from the "state" one, and it is emitted as computation.
        self.guest_derived: list[tuple[int, str, str, int, int]] = []
        self.handler_cache: GuestCache | None = None
        # (guest register, value, step number) for the context read made by the
        # instruction currently being lifted. The step number is what confines the
        # correlation to one instruction -- **not** a clear at the top of `step`,
        # which was the first version and cost 1.8 s of a 20 s run: `step` executes
        # tens of millions of times per recovery, and an attribute store on every
        # one of them is paid by every caller for a census that is off by default.
        # Recording the step here instead costs nothing, because this line only
        # runs when the census is installed.
        self._ctx_read: tuple[str, int, int] | None = None
        self._machine = machine
        self._m_load = machine.load
        self._m_store = machine.store
        self._m_set_reg = machine.set_reg

    def enable_guest_census(self, ctx: GuestContext,
                            handler_cache: GuestCache | None = None) -> None:
        """Start naming guest register reads and writes. Costs 1-2 % of a run.

        `handler_cache` is the **handler-time** map, if the caller has derived it with
        `Recoverer.derive_guest_cache`. It is what makes an attribution trustworthy:
        the value chain says "this register was carrying a guest value and is being
        updated", and the cache says "that register is where *this* guest register
        lives". Round 10 measured why one alone is not enough -- the value chain on its
        own attributed 6942 writes of which **97.6 %** contradicted the cache, the
        dominant case being `guest rax` spread across six scratch registers.
        """
        self.guest_ctx = ctx
        self.handler_cache = handler_cache
        self._machine.load = self._guest_load
        self._machine.store = self._guest_store
        self._machine.set_reg = self._guest_set_reg
        # **The census is evidence, handed to the memory model once.** `guest_identity` is created in
        # `__init__` and only ever added to, so the reference stays live; passing it here is what
        # makes a `load` leaf the census has justified as *being* a guest register count as guest
        # state in `guest_only_address`, without any per-store flag changing how a store behaves.
        self._machine.guest_marks = self.guest_identity

    def _guest_set_reg(self, name, value, width=None):
        # Provenance, tracked **as a chain from a context read** rather than read off
        # the global cache map. `derive_guest_cache` measures which host register
        # carries which guest register, and it is unambiguous when it fires -- but it
        # fires on 170 of 798 boundaries for `rsi` on the benchmark root, so the map
        # says "when this register holds a guest value, it is guest rcx" and not "this
        # register is guest rcx". Attributing every write to `rsi` through that map
        # would mis-name 628 boundaries' worth of work.
        #
        # So a register acquires provenance when a context read of slot G lands in it,
        # and keeps it only while its value still depends on what it held -- a
        # `rsi = rsi + 8` update keeps it, an unrelated `rsi = <scratch>` drops it.
        # That is conservative in the right direction: it under-reports rather than
        # mis-reports, and the count of unattributed writes is what measures the gap.
        parent = canon(name)[0] if name else name
        if width is None or width == 64:
            # **Marking and attribution are different jobs, and only attribution must be
            # strict.** Marking asks "is this value guest state?" and may be liberal: a
            # value computed from exactly one known guest value *is* derived from it, and
            # marking it keeps a chain of computations traceable across a block. Until
            # this existed the mark was set only when the *destination* already carried a
            # guest value, so a chain that stepped through a scratch register lost the
            # trail at the first step -- which is what left 139 blocks with nothing to say.
            #
            # Attribution asks "does this write update guest state?" and stays strict, for
            # the reason rounds 10 and 18 both measured: a liberal attribution called 6942
            # writes updates and 97.6 % of them contradicted the cache.
            _derived = {self.guest_values[n] for n in self.a.leaves(value)
                        if n in self.guest_values}
            if len(_derived) == 1 and value not in self.guest_values:
                self.guest_values[value] = next(iter(_derived))
                self.gv_prov[value] = ("derived", self.steps)
                self.gv_first.setdefault(value, ("derived", self.steps))
        if width is None or width == 64:
            # A guest register is updated when a register that was **carrying** it is
            # overwritten with something derived from what it held -- not when a value
            # that merely came from a guest register is written somewhere.
            #
            # The looser rule, "the new value depends on some guest value", was
            # measured and is wrong: it attributed 6942 writes across 142 of 149
            # blocks, and **6774 of them were inconsistent with the cache** that round
            # 1 measured. The dominant case was `guest rax` appearing in `r9`, `rcx`,
            # `rax`, `rdx` and `r10` -- the VM doing arithmetic with a guest value and
            # parking intermediates in scratch. That is computation, not a guest
            # register being updated, and round 1 already said so: guest `rax` is not
            # cache-resident on this function.
            prev = self._machine.reg(parent, 64)
            carried = self.guest_values.get(prev)
            if carried is not None and (value == prev
                                        or prev in self.a.leaves(value)):
                # Second condition: the destination register must be where *this*
                # guest register lives, per the handler-time cache. Without it the
                # chain attributes scratch work -- measured at 6774 of 6942 writes.
                if self.handler_cache is not None:
                    if self.handler_cache.guest_of(parent) != carried:
                        # Not `G`'s home: the register held `G`'s value and is being
                        # overwritten with something derived from it. That is a
                        # **computation with a copy of `G`**, not an update of `G` --
                        # and it is exactly what a devirtualizer wants emitted, so it
                        # is recorded rather than discarded.
                        self._reject("not the cache register for that guest")
                        self.guest_derived.append(
                            (self.steps, carried, parent, prev, value))
                        self.guest_trace.append((self.steps, "carry",
                                                 (carried, parent, prev, value)))
                        return self._m_set_reg(name, value, width)
                else:
                    self._reject("no handler-time cache derived")
                self.guest_values[value] = carried
                self.gv_prov[value] = ("home-update", self.steps)
                self.gv_first.setdefault(value, ("home-update", self.steps))
                self.guest_identity.setdefault(value, carried)
                self.guest_reg_writes.append(
                    (self.steps, carried, parent, value))
                self.guest_trace.append((self.steps, "home",
                                         (carried, parent, value)))
            elif carried is not None:
                self._reject("overwritten by a value not derived from it")
        return self._m_set_reg(name, value, width)

    def _reject(self, reason: str) -> None:
        self.grw_rejected[reason] = self.grw_rejected.get(reason, 0) + 1

    def _guest_slot(self, addr: int, nbytes: int) -> str | None:
        """The context slot containing this access, including byte/word slices.

        Memory writes preserve bytes outside the written range. The caller must
        distinguish a complete slot from a partial access; x86's EAX zero-extension
        rule does not apply to a four-byte write to memory.
        """
        ctx = self.guest_ctx
        if ctx is None or nbytes not in (1, 2, 4, 8):
            return None
        c = self.a.as_const(addr)
        if c is None:
            return None
        offset = c - ctx.base
        if offset < 0 or offset + nbytes > ctx.span[1] - ctx.base:
            return None
        inside = offset % 8
        return ctx.reg_at(c - inside) if inside + nbytes <= 8 else None

    def _wide(self, value: int, nbytes: int) -> int:
        """The value a guest register holds after an `nbytes`-wide write to its home."""
        return value if nbytes == 8 else self.a.zext(value, 64)

    def _guest_ptr(self, addr: int) -> tuple[str, int] | None:
        """`(guest register, byte offset)` if this address is `guest_G + k`.

        **The sharper criterion round 36 designed and deliberately deferred**, because changing a
        criterion while its consequences are unmeasured is how this project's earlier over-claims
        happened. Two restrictions make it sound where the round-35 rule was not:

        * **only the top-level term is tested**, so the dependence is *direct*. The round-35 rule
          ("the address depends on an `init_*` leaf") is transitive through a flags-word
          reconstruction, and measured 40 of 41 addresses were a VM-stack constant indexed by flag
          bits -- the VM's own traffic, not guest pointers.
        * **only identity marks count** (`context-load`, `home-update`). A `derived` mark says a
          value was computed from a guest value, which is not a licence to use it as a pointer
          base; on the roots measured 88-96 % of marks are of that kind.

        What is left is the shape a compiler emits for a guest access -- `[guest_rbx + 0x18]`,
        `[guest_rcx]` -- which is what a devirtualized listing is missing and what a recompiler
        would need, since an absolute stack address cannot be emitted for another process.
        """
        a = self.a
        n = a.node(addr)
        if addr in self.guest_identity:
            return self.guest_identity[addr], 0
        # **A bare `init:<reg>` is a guest pointer, and this case is added on evidence rather than
        # on hope.** `0x140155dc1` loses 535 of 769 reference transitions because a store whose
        # address is exactly `in.rcx` has unbounded `address_bounds`, so it deletes 980 tracked
        # frame bytes, the VM's saved RSP becomes unknown, and `0x14015695c pushfq` cannot resolve.
        # `init_*` is the register file at the start of the walk, so `in.rcx` is the guest
        # function's incoming register -- a guest pointer.
        #
        # An earlier attempt added `init` handling to recover guest *accesses* and measured zero
        # gain, and was reverted for exactly that reason. This is a different question with a
        # different measurement attached: does scoping the clobber restore the coverage?
        if n[0] == "init":
            return n[1], 0
        if n[0] != "bin" or n[1] not in ("add", "sub"):
            return None
        base = off = None
        if n[1] == "add":
            for x, y in ((n[2], n[3]), (n[3], n[2])):
                c = a.as_const(y)
                if c is not None:
                    base, off = x, c
                    break
        else:
            c = a.as_const(n[3])
            if c is not None:
                base, off = n[2], -c
        if base is None:
            return None
        bn = a.node(base)
        if bn[0] == "init":
            return bn[1], off
        if base not in self.guest_identity:
            return None
        return self.guest_identity[base], off

    def _is_identity(self, node: int) -> bool:
        """Was this value ever justified as *being* a guest register? See `guest_identity`."""
        return node in self.guest_identity

    def _guest_load(self, addr, nbytes, site, image_u8):
        reg = self._guest_slot(addr, nbytes)
        v = self._m_load(addr, nbytes, site, image_u8)
        if reg is not None:
            self.guest_reads.append((site, reg))
            # Widths actually read from each context slot. This is the evidence that decides
            # whether a slot's 32-bit writes are complete or partial: if every read of a slot is
            # 4 bytes wide, its upper half is never observed and the 4-byte write is effectively
            # the whole value; if any read is 8 bytes wide, it is not.
            self.slot_read_widths.append((reg, nbytes))
            # Correlation stays on the width the instruction used, because the register write
            # it is matched against is the same width; the *guest* value is the widened one,
            # because that is what the register holds afterwards.
            self.guest_trace.append((self.steps, "load", (site, reg, v, addr, nbytes)))
            if nbytes != 8:
                # This is a slice of a slot, not the whole guest register.
                self.guest_values.setdefault(v, reg)
                self.gv_prov.setdefault(v, ("context-slice", self.steps))
                self.gv_first.setdefault(v, ("context-slice", self.steps))
                return v
            self._ctx_read = (reg, v, self.steps)
            wide = v
            # The value that came out of the slot **is** that guest register, by
            # definition -- this is the one place where no inference is involved.
            self.guest_values[wide] = reg
            self.gv_prov[wide] = ("context-load", self.steps)
            self.gv_first.setdefault(wide, ("context-load", self.steps))
            self.guest_identity.setdefault(wide, reg)
        else:
            # Not a context slot. **Is it a guest access?** This is the case the trace had no
            # event for at all: the guest's own memory traffic, which is the bulk of what a
            # devirtualized listing is supposed to show.
            p = self._guest_ptr(addr)
            if p is not None:
                self.guest_trace.append((self.steps, "gload",
                                         (site, p[0], p[1], nbytes, v, addr)))
        return v

    def _guest_store_slot(self, value: int, addr: int) -> None:
        """The value written *into* a context slot **is** that guest register.

        The other half of the same fact, and it went unused for eight rounds: a context
        **load** of slot `G` says the value that came out is `G`, and a context **store**
        to slot `G` says the value that went in is `G`. Both are definitional rather than
        inferred, both are point-wise, and neither needs the global cache or its
        stability statistics. Marking on the store side extends the trail to values that
        reach a slot before anything loads it back.
        """
        ctx = self.guest_ctx
        if ctx is None:
            return
        c = self.a.as_const(addr)
        if c is None:
            return
        reg = ctx.reg_at(c)
        if reg is not None and value not in self.guest_values:
            # **This one is not an identity claim.** The comment above said the stored value
            # "is" that guest register; what a store establishes is that the register is
            # *becoming* this value. Marking it identity is what lets a value that is merely
            # on its way into a slot be rendered later as `guest_<g>`.
            self.guest_values[value] = reg
            self.gv_prov[value] = ("context-store", self.steps)
            self.gv_first.setdefault(value, ("context-store", self.steps))

    def _guest_store(self, addr, value, nbytes, site):
        """Note whether this store goes through a guest pointer, then record it.

        **The count is a count and nothing else.** This method used to set
        `Machine.frame_separated` around the call, which is how the census made a guest-pointer store
        spare the VM's frame -- and that made *recording an observation* change *program behaviour*,
        so enabling the census could change the analysis. The decision now lives in `Machine.store`,
        where the store is performed, and it takes the census's identity map as an **input**
        (`Machine.guest_marks`) rather than as a flag switched on for the duration of one call. The
        criterion is `guest_only_address`, which reads the marks when they exist and the `init`
        leaves when they do not.
        """
        if (self._guest_slot(addr, nbytes) is None
                and self._guest_ptr(addr) is not None):
            self.guest_pointer_stores += 1
        return self._guest_store_record(addr, value, nbytes, site)

    def _guest_store_record(self, addr, value, nbytes, site):
        reg = self._guest_slot(addr, nbytes)
        # **Every concrete store, so the context window can be compared against the rest of
        # memory rather than only against itself.** A classifier that only ever looks at
        # context slots cannot show that it discriminates -- round 38 needs the same test
        # applied to frame stores, where the answer is expected to be the opposite. This is
        # free in the default configuration: `_guest_store` is installed only under the
        # census.
        c = self.a.as_const(addr)
        if c is not None:
            self.all_stores.append((site, c, nbytes, value))
        if reg is not None and nbytes == 8:
            wide = self._wide(value, nbytes)
            self.guest_writes.append((site, reg, wide))
            self.guest_trace.append((self.steps, "store", (site, reg, wide)))
            self._guest_store_slot(wide, addr)
        elif reg is not None:
            # **A narrow store to a context slot is a PARTIAL write, and treating it as a full
            # register update was wrong.** The slot is 8 bytes; `mov [ctx+8i], eax` writes 4 of
            # them and leaves the upper 4 as they were, so the register does *not* become
            # `zext64(value)`. The review's repro: a slot holding `0x1122334455667788` becomes
            # `0x11223344deadbeef` after a 4-byte write of `0xdeadbeef`, while the census recorded
            # `0xdeadbeef`. Round 39 generalised from the flags slot, where 32 bits *is* the whole
            # architectural width -- a convention that has to be proven per slot, not assumed for
            # all of them.
            #
            # So the write is recorded as partial and is used for **neither** identity marking nor
            # full-register attribution. `slot_read_widths` accumulates the widths actually read
            # from each slot, which is the evidence that would justify treating a specific slot's
            # 4-byte writes as complete.
            lo = c - self.guest_ctx.addr_of(reg)
            self.guest_trace.append((self.steps, "part",
                                     (site, reg, lo, nbytes, self.a.trunc(value, nbytes * 8))))
        else:
            p = self._guest_ptr(addr)
            if p is not None:
                self.guest_trace.append((self.steps, "gstore",
                                         (site, p[0], p[1], nbytes, value, addr)))
        return self._m_store(addr, value, nbytes, site)

    # Marks that assert *identity*: the node's value **is** that guest register.
    # `context-load` because the slot layout defines it; `home-update` because a register
    # shown by the handler-time cache to be `G`'s home, overwritten with a value derived
    # from what it held, now holds `G`. The other two are relations about *flow* -- where a
    # value is going, or what it was computed from -- and reading them as identity is the
    # over-claim `render_value` has been making since round 24.
    IDENTITY_MARKS = ("context-load", "home-update", "cache-seed")

    def identity_marks(self) -> dict[int, str]:
        """Only the marks that assert identity, for consumers that name guest registers."""
        return {n: r for n, r in self.guest_values.items()
                if self.gv_prov.get(n, ("unknown", -1))[0] in self.IDENTITY_MARKS}

    def mark_census(self) -> dict[str, int]:
        """How many marks of each mechanism exist -- the mixture, as a number."""
        out: dict[str, int] = {}
        for n in self.guest_values:
            why = self.gv_prov.get(n, ("unknown", -1))[0]
            out[why] = out.get(why, 0) + 1
        return out

    def guest_effect(self, since_read: int, since_write: int
                     ) -> tuple[list[tuple[int, str]], list[tuple[int, str, int]]]:
        """Guest reads and writes recorded since a pair of earlier marks.

        A block's effect is obtained by marking the lists before it is
        interpreted and asking afterwards, which keeps the accounting out of the
        interpreter's own state and makes it exact per block.
        """
        return (self.guest_reads[since_read:], self.guest_writes[since_write:])


    # ------------------------------------------------------------- decoding

    def in_vm(self, va: int) -> bool:
        return self.vm_lo <= va < self.vm_hi

    def executable(self, va: int) -> bool:
        sec = self.img.section_for_va(va)
        return sec is not None and sec.executable

    def decode(self, ip: int):
        got = self._cache.get(ip)
        if got is not None:
            return got
        body = self.img.image_bytes(ip, 16)
        if not body:
            self._cache[ip] = (0, None, b"")
            return self._cache[ip]
        insn = next(self._md.disasm(body, ip, 1), None)
        raw = body[: insn.size] if insn is not None else b""
        self._cache[ip] = (insn.size if insn is not None else 0, insn, raw)
        return self._cache[ip]

    # ------------------------------------------------------------ operand IO

    def _read_at(self, addr: int, nbytes: int, site: int) -> int:
        a = self.a
        if not self.img.foldable(addr, nbytes):
            # The file is not a valid initial value here, but an explicit write
            # performed on this path still is. Consult the overlay without a file fallback.
            return self.m.load(a.const(addr, W64), nbytes, site, lambda _: None)
        return self.m.load(a.const(addr, W64), nbytes, site, self.img.image_u8)

    def _load_via_predicate_split(self, addr: int, nbytes: int, site: int) -> int | None:
        """Resolve a load whose address takes exactly two values.

        The VM reads its dispatch tables with `mov eax, [C + (cond << k)]`, where
        `cond` is a predicate the evaluator cannot decide. Pinning it both ways
        makes the address constant each way, so the read becomes
        `cond ? [C+k] : [C]` -- two constants and a selector -- instead of an
        opaque load at a symbolic address.

        What this unblocks is not the value but everything above it: the VM folds
        a condition with `load(A) - load(A)`, and those are two different nodes
        whenever a store landed between them. Once each resolves to the same pair
        of constants they intern to one `select`, the subtraction collapses, and
        the dispatch above it becomes a two-way branch that recovery can split.

        Only a two-way split is attempted. An address with more freedom is left
        symbolic: enumerating it needs a sound bound on the predicate, and
        guessing would fabricate edges.

        Answering each candidate by *direct evaluation* instead of by
        substitution was implemented and measured, and it was **slower** -- 31.7 s
        against 25.3 s on the same function -- so the substitution stayed. The
        evaluator is sound and slightly more precise in places (64,095 trials:
        zero disagreements, 180 folds the folder misses, 852 it makes that the
        evaluator declines), but a trial here does not rebuild the expression so
        much as visit every node of it, and the visit costs what the rebuild
        saved. The measurement is in `probe.eval_census`; the numbers belong in
        the record so that the experiment is not repeated.
        """
        from .memory import finite_addresses
        a = self.a
        before = getattr(a, 'address_search_stats', {}).get('pruned', 0)
        # Existing dispatch-table loads can require DAGs above 4096 nodes or
        # more than 16384 rebuild work. Preserve their established precision;
        # the optional store-side budget must not truncate these load proofs.
        split = finite_addresses(a, addr, candidate_limit=256, prune=True,
                                 prefer_selectors=False, max_nodes=None,
                                 max_rebuild_work=None)
        self.split_pruned += a.address_search_stats['pruned'] - before
        if split is None:
            return None
        condition, zero, one = split
        # Address discovery is pure. Read memory only after both alternatives
        # are proved, and always from the current machine epoch. Cache no data.
        left = self._read_at(zero, nbytes, site)
        right = self._read_at(one, nbytes, site)
        return a.select(condition, right, left, nbytes * 8)

    def note_guest_load(self, addr: int, site: int) -> None:
        """Is this read going through a guest register? See `load_at`.

        The test is `init_*` presence and nothing else, deliberately: `init_*` **is** the
        register file at VM entry, so an address containing one is computed from a guest
        register, which is what a guest load is. A frame-relative read does not contain
        one -- the frame base is a constant the walk already knows -- so the VM's own
        traffic is excluded by the same test rather than by a second rule.
        """
        if self.guest_ctx is None:
            return
        a = self.a
        if any(a.op(n) == "init" for n in a.leaves(addr)):
            self.guest_loads.append((site, addr))
        else:
            self.guest_load_rejects["not through a guest register"] = (
                self.guest_load_rejects.get("not through a guest register", 0) + 1)

    def load_at(self, addr: int, nbytes: int, site: int) -> int:
        """Read `nbytes` from `addr`, folding to a file value only when that is sound.

        A read from a writable image section, or from an IAT slot, must not fold
        to the byte in the file: the loader or the runtime writes the real value
        there before this code runs. It becomes an interned symbolic load at a
        constant address instead, which keeps two reads of the same address the
        same node -- so `load(A) - load(A)` still cancels and the protector's
        self-cancelling dispatch arithmetic still collapses.

        **Guest memory reads are recorded here.** A load whose address depends on an
        `init_*` node is a read through a guest register: `init_*` is the register file
        at VM entry, and the virtualized function is called normally, so those are its
        own registers. A guest `mov eax, [rbx+8]` is exactly this, and the register
        census cannot see it -- the address is neither a context slot nor a concrete
        frame address, so it fell outside every region `probe/guestmem` bucketed.
        Measured: `0x1401505d5` reads `[entry_rcx + 0x10]` **105 times**.
        """
        self.note_guest_load(addr, site)
        a = self.a
        c = a.as_const(addr)
        if c is not None:
            return self._read_at(c, nbytes, site)
        split = self._load_via_predicate_split(addr, nbytes, site)
        if split is not None:
            return split
        self.m.sym_loads.append((site, nbytes))
        return self.m.load(addr, nbytes, site, self.img.image_u8)

    def op_width(self, insn, i: int) -> int:
        return insn.operands[i].size * 8

    def read_op(self, insn, i: int) -> int:
        op = insn.operands[i]
        if op.type == CS_OP_REG:
            return self.m.reg(insn.reg_name(op.reg), op.size * 8)
        if op.type == CS_OP_IMM:
            w = op.size * 8
            return self.a.const(op.imm & ((1 << w) - 1), w)
        if op.type == CS_OP_MEM:
            addr = self.mem_address(insn)
            return self.load_at(addr, op.size, insn.address)
        raise NotImplementedError(f"operand type {op.type}")

    def write_op(self, insn, i: int, value: int, width: int | None = None) -> None:
        op = insn.operands[i]
        w = width or op.size * 8
        if op.type == CS_OP_REG:
            self.m.set_reg(insn.reg_name(op.reg), value, w)
            return
        if op.type == CS_OP_MEM:
            self.m.store(self.mem_address(insn), value, op.size, insn.address)
            return
        raise NotImplementedError("cannot write a non-writable operand")

    def mem_address(self, insn) -> int:
        op = next((o for o in insn.operands if o.type == CS_OP_MEM), None)
        assert op is not None
        a = self.a
        acc: int | None = None
        base_name = insn.reg_name(op.mem.base) if op.mem.base else None
        if base_name == "rip":
            # RIP-relative: the effective address is the *next* instruction plus
            # the displacement, and the base register contributes nothing. This
            # must be handled before touching the register file, which has no
            # `rip` entry by design.
            return a.const((insn.address + insn.size + op.mem.disp) & ((1 << 64) - 1), W64)
        if op.mem.base:
            acc = self.m.reg(base_name, 64)
        if op.mem.index:
            idx = self.m.reg(insn.reg_name(op.mem.index), 64)
            scaled = self.a.bin("mul", idx, self.a.const(op.mem.scale, W64), W64)
            acc = scaled if acc is None else self.a.bin("add", acc, scaled, W64)
        if op.mem.disp:
            dv = self.a.const(op.mem.disp & ((1 << 64) - 1), W64)
            acc = dv if acc is None else self.a.bin("add", acc, dv, W64)
        if acc is None:
            acc = self.a.const(0, W64)
        return acc

    # ---------------------------------------------------------------- flags

    def _const_table(self, width: int) -> tuple[int, ...]:
        """`(0, 1, ..., 7, width - 1)` as node ids at `width`, built once per width.

        Index `k` for `k` in `0..7`; index 8 is `width - 1`. See `Lifter.__init__` for why these
        are hoisted instead of interned at the point of use.
        """
        got = self._consts.get(width)
        if got is None:
            a = self.a
            got = self._consts[width] = tuple(a.const(k, width) for k in range(8)) + (
                a.const(width - 1, width),)
        return got

    def _zf_sf_pf(self, result: int, width: int) -> None:
        a = self.a
        k = self._const_table(width)
        self.m.set_flag("zf", a.bin("eq", result, k[0], width))
        self.m.set_flag("sf", a.trunc(a.bin("shr", result, k[8], width), W8))
        # Parity is the XOR of the low eight bits, inverted.
        #
        # This used to be left opaque on the grounds that it costs about two
        # dozen DAG nodes per arithmetic instruction and only `jp`/`jnp`/`setp`
        # read it. That reasoning was wrong, and expensively so: `pushfq` packs
        # every modelled flag into one word, the VM stores that word and later
        # reads bits out of it to choose its next handler, so an opaque PF sits
        # *on the dispatch path*. An opaque flag is not a cheap flag; it is a
        # flag that makes every expression built on the packed word unfoldable.
        #
        # **Three halving folds, not seven shifts.** XOR-ing the low byte onto itself shifted by
        # 4, then by 2, then by 1 leaves the XOR of all eight bits in bit 0, and bit 0 of an
        # eight-bit value is that value's parity. Seven operations where the bit-at-a-time loop
        # took 31 plus sixteen `const` calls -- and this runs on every arithmetic instruction, so
        # on a 317 k-instruction root the loop alone was ~4.3 M interner operations. The two forms
        # are equal over the whole domain that matters (all 256 values of the low byte, at every
        # width), and `tests/test_flag_cost.py` checks that rather than asserting it.
        low = result if width == W8 else a.trunc(result, W8)
        eight = self._const_table(W8)
        x = a.bin("xor", low, a.bin("shr", low, eight[4], W8), W8)
        x = a.bin("xor", x, a.bin("shr", x, eight[2], W8), W8)
        x = a.bin("xor", x, a.bin("shr", x, eight[1], W8), W8)
        self.m.set_flag("pf", a.bin("xor", a.bin("and", x, eight[1], W8), eight[1], W8))

    def _af_from(self, x: int, y: int, result: int, width: int) -> int:
        """Auxiliary carry: the carry out of bit 3.

        `x ^ y ^ result` has bit 3 set exactly when a carry crossed that
        boundary, for both addition and subtraction. The shift has to be masked
        to bit 0 of the result, not truncated: `trunc(shr(t,4), 8)` keeps bits
        4..11 and returns 31 for `0xff + 1`, where `(t >> 4) & 1` returns 1.
        """
        a = self.a
        k = self._const_table(width)
        t = a.bin("xor", x, y, width)
        t = a.bin("xor", t, result, width)
        shifted = a.bin("shr", t, k[4], width)
        return a.trunc(a.bin("and", shifted, k[1], width), W8)

    def _logic_flags(self, result: int, width: int) -> None:
        self._zf_sf_pf(result, width)
        k = self._const_table(W8)
        self.m.set_flag("cf", k[0])
        self.m.set_flag("of", k[0])
        # AF is architecturally undefined after a logical operation, and this
        # models it as unknown -- the reference implementation's choice as well.
        #
        # Setting it to zero was tried once before, on the grounds that every
        # Intel and AMD part clears it in practice, and reverted: it removed one
        # symbolic leaf from the benchmark root's dispatch address and changed
        # **nothing** measurable. It is switchable now because the same question
        # came back on a different root, where AF is one of only two symbolic
        # leaves in a dispatch the *reference resolves*, and a hypothesis that was
        # rejected on one root has to be re-measured on another rather than
        # assumed to fail everywhere. See `Lifter.logical_af_zero`.
        if self.logical_af_zero:
            self.m.set_flag("af", k[0])
        else:
            self.m.set_flag("af", self.a.opaque("flag_af", W8))

    def _bit_of(self, value: int, index: int, width: int) -> int:
        """Bit `index` of `value`, as a W8 value in {0, 1}."""
        a = self.a
        bit = value if index <= 0 else a.bin("shr", value, a.const(index, width), width)
        return a.trunc(a.bin("and", bit, a.const(1, width), width), W8)

    def _msb(self, value: int, width: int) -> int:
        return self._bit_of(value, width - 1, width)

    def _shift_flags(self, op_name: str, x: int, result: int, width: int,
                     count: int) -> None:
        """CF and OF after SHL/SHR/SAR, per Intel SDM, and witnesses where it refuses.

        `CF` is the last bit shifted out: bit `width - count` of the source for `shl`,
        bit `count - 1` for `shr`/`sar`. `OF` is defined **only for a count of one** —
        `shl`: MSB(source) XOR MSB(result); `shr`: MSB(source); `sar`: 0 — and a count of
        two or more leaves it architecturally undefined, so it stays a witness rather
        than being invented.

        A count at or above the operand size also leaves `CF` undefined, which is the one
        case here that stays a witness on both flags. Both witnesses are the *named*
        nodes the rest of the engine already uses, so they intern with every other
        undefined `CF` rather than creating a private one per site.
        """
        a = self.a
        if count >= width:
            self.m.set_flag("cf", a.opaque("flag_cf", W8))
            self.m.set_flag("of", a.opaque("flag_of", W8))
            return
        source_bit = width - count if op_name == "shl" else count - 1
        self.m.set_flag("cf", self._bit_of(x, source_bit, width))
        if count != 1:
            self.m.set_flag("of", a.opaque("flag_of", W8))
        elif op_name == "shl":
            self.m.set_flag("of", a.bin("xor", self._msb(x, width),
                                        self._msb(result, width), W8))
        elif op_name == "shr":
            self.m.set_flag("of", self._msb(x, width))
        else:  # sar
            self.m.set_flag("of", a.const(0, W8))

    def _rotate_flags(self, op_name: str, x: int, result: int, width: int,
                      count: int) -> None:
        """CF and OF after ROL/ROR.

        `CF` is the last bit rotated out — bit `width - count` of the source for `rol`,
        bit `count - 1` for `ror` — and unlike a shift it is defined for every count.
        `OF` is defined only for a count of one: `rol`: MSB(result) XOR CF; `ror`:
        MSB(result) XOR the bit below it.
        """
        a = self.a
        cf = self._bit_of(x, width - count if op_name == "rol" else count - 1, width)
        self.m.set_flag("cf", cf)
        if count != 1:
            self.m.set_flag("of", a.opaque("flag_of", W8))
        elif op_name == "rol":
            self.m.set_flag("of", a.bin("xor", self._msb(result, width), cf, W8))
        else:  # ror
            self.m.set_flag("of", a.bin("xor", self._msb(result, width),
                                        self._bit_of(result, width - 2, width), W8))

    def _add_flags(self, x: int, y: int, result: int, width: int) -> None:
        a = self.a
        self._zf_sf_pf(result, width)
        # CF: an unsigned sum is below either addend exactly when it wrapped.
        self.m.set_flag("cf", a.bin("ult", result, x, width))
        k = self._const_table(width)
        sx = a.bin("slt", x, k[0], width)
        sy = a.bin("slt", y, k[0], width)
        sr = a.bin("slt", result, k[0], width)
        # **OF: both signs alike and the result's sign different.** This used to
        # read `(sx ^ sy) ^ (sx ^ sr)`, which collapses to `sy ^ sr` and is not the
        # overflow predicate at all: it disagrees with Python on **16,384 of 65,536**
        # ordered byte pairs, exhaustively. `tests/test_arithmetic_flags` is the
        # instrument, and it is exhaustive on the 8-bit domain rather than sampled.
        #
        # The reason this mattered enough to fix rather than note: `pushfq` packs
        # every flag into one word and the VM stores that word and reads bits out of
        # it to choose its next handler, so a wrong OF is a wrong *dispatch* -- and a
        # silent one, because the bit is plausible in every diagnostic. The corpus
        # still reproduces all 15,909 reference transitions after the fix, which says
        # the discipline path in this build does not read OF; that is a fact about
        # this module, not a licence for the rule to be wrong.
        agree = a.bin("xor", a.bin("xor", sx, sy, W8), k[1], W8)     # 1 when sx == sy
        flipped = a.bin("xor", sr, sx, W8)                           # 1 when sr != sx
        self.m.set_flag("of", a.bin("and", agree, flipped, W8))
        self.m.set_flag("af", self._af_from(x, y, result, width))

    def _sub_flags(self, x: int, y: int, result: int, width: int) -> None:
        a = self.a
        self._zf_sf_pf(result, width)
        self.m.set_flag("cf", a.bin("ult", x, y, width))
        k = self._const_table(width)
        sx = a.bin("slt", x, k[0], width)
        sy = a.bin("slt", y, k[0], width)
        sr = a.bin("slt", result, k[0], width)
        # **OF: the signs differed, and the result took neither side's sign.**
        # The previous `(sx ^ sy) ^ (sr ^ sy)` collapses to `sx ^ sr`; it disagreed
        # with Python on 16,384 of 65,536 pairs. See `_add_flags`.
        differ = a.bin("xor", sx, sy, W8)                            # 1 when sx != sy
        flipped = a.bin("xor", sr, sx, W8)                           # 1 when sr != sx
        self.m.set_flag("of", a.bin("and", differ, flipped, W8))
        self.m.set_flag("af", self._af_from(x, y, result, width))

    # ----------------------------------------------------------------- step

    def step(self, ip: int) -> tuple[str, object]:
        size, insn, raw = self.decode(ip)
        if insn is None:
            return "stop", Stop("unreadable", ip)
        self.steps += 1
        self._previous_ip, self._current_ip = self._current_ip, ip
        self.trace.append(ip)
        if len(self.trace) > 4096:
            del self.trace[:2048]
        nxt = ip + size
        mn = insn.mnemonic

        if mn in _CET_NOOP:
            # CET bookkeeping and endbr markers: no architectural effect the
            # guest can observe.
            return "next", nxt

        try:
            handler = _HANDLERS.get(mn)
            if handler is not None:
                return handler(self, insn, ip, nxt)
            if mn in JCC:
                return self._jcc(insn, ip, nxt)
            if mn in SETCC:
                v = condition(self.a, self.m, SETCC[mn])
                self.write_op(insn, 0, v, 8)
                return "next", nxt
            if mn in CMOVCC:
                c = condition(self.a, self.m, CMOVCC[mn])
                src = self.read_op(insn, 1)
                dst = self.read_op(insn, 0)
                self.write_op(insn, 0, self.a.select(c, src, dst, self.op_width(insn, 0)))
                return "next", nxt
            if mn in UNBOXABLE:
                if (mn == 'int' and len(insn.operands) == 1
                        and insn.operands[0].type == CS_OP_IMM
                        and insn.operands[0].imm == 0x29):
                    # Windows __fastfail(unsigned int): this is a real terminal
                    # effect, not an unsupported instruction or a normal return.
                    code = self.m.reg('rcx', W32)
                    return 'stop', Stop('fastfail', ip, text='Windows fast fail (int 0x29)',
                                        info={'code': code, 'mechanism': 'int 0x29'})
                trap = self._guest_exit_trap(insn, ip, nxt)
                if trap is not None:
                    return "stop", trap
                return "stop", Stop("unsupported", ip, text=mn)
            if self.box_unmodelled:
                return self._box(insn, ip, nxt, raw)
            return "stop", Stop("unsupported", ip, text=mn)
        except _Unsupported as e:
            if self.box_unmodelled:
                return self._box(insn, ip, nxt, raw)
            return "stop", Stop("unsupported", ip, text=f"{mn}: {e}")
        except UnmodelledRegister as e:
            # **A register the model does not carry is an unsupported instruction, not a crash.**
            # The VM's own handler code executes `mov ..., cr8` on `ACE-CORE.sys3`; letting the
            # `KeyError` escape killed an entire 143-root corpus run. A named stop is the honest
            # answer and the recovery already knows how to report one.
            return "stop", Stop("unsupported", ip, text=f"{mn}: unmodelled register {e.args[0]}")
        except _Refuse as e:
            return "stop", Stop(e.kind, ip, text=e.text, info=e.info)

    # --------------------------------------------------------- control flow

    @staticmethod
    def _reg_name(insn, reg: int) -> str | None:
        """The name of a capstone register id, or None for "no register"."""
        return insn.reg_name(reg) if reg else None

    def _reentry_prologue(self, va: int) -> bool:
        """True when `va` begins the VM's re-entry prologue.

        The measured three-instruction shape, identical on both builds this engine has been
        run against (frame 0xaa0 on `SGuardSvc64.exe`, 0xa70 on `SGuardAgent64.dll`):

            lea rsp, [rsp - IMM]
            mov [rsp + 0x10], rbp
            mov rbp, rsp

        Operand-based rather than byte-based, so a different frame size, displacement or
        encoding still matches, and `lea` off any base other than `rsp` does not.
        """
        size, insn, _raw = self.decode(va)
        if insn is None or insn.mnemonic != "lea" or len(insn.operands) != 2:
            return False
        dst, src = insn.operands
        if (dst.type != CS_OP_REG or self._reg_name(insn, dst.reg) != "rsp"
                or src.type != CS_OP_MEM or self._reg_name(insn, src.mem.base) != "rsp"):
            return False
        va += size
        size, insn, _raw = self.decode(va)
        if insn is None or insn.mnemonic != "mov" or len(insn.operands) != 2:
            return False
        dst, src = insn.operands
        if (dst.type != CS_OP_MEM or self._reg_name(insn, dst.mem.base) != "rsp"
                or src.type != CS_OP_REG or self._reg_name(insn, src.reg) != "rbp"):
            return False
        va += size
        size, insn, _raw = self.decode(va)
        if insn is None or insn.mnemonic != "mov" or len(insn.operands) != 2:
            return False
        dst, src = insn.operands
        return (dst.type == CS_OP_REG and self._reg_name(insn, dst.reg) == "rbp"
                and src.type == CS_OP_REG and self._reg_name(insn, src.reg) == "rsp")

    def _guest_exit_trap(self, insn, ip: int, nxt: int) -> Stop | None:
        """A bare `int3` that is really the VM leaving the virtualized function.

        **The mnemonic is not the test.** `int3` in ordinary code is a breakpoint or a patch
        point and must stay an honest hole, so this fires only when the three instructions
        after it are the VM re-entry prologue — a shape no debugger-bait `int3` carries.
        A `return` claims no successor, so the rule cannot fabricate an edge; the worst it
        can do is call a trap an exit.

        The measurement, on `SGuardAgent64.dll` (probe/disassembly plus the reference as an
        output oracle): four blocks at handler `0x1800b30d7` run exactly **372** native
        instructions and stop at a single `0xCC` byte, and the reference terminates those
        same four blocks as `return (372 steps)` — four for four, at the same step count.
        The byte before it is `mov rsp,[rsp]`, which has just switched `rsp` to the guest's
        stack, and the bytes after it are the re-entry prologue. On the build this project
        started with, the same stub ends in `call qword ptr [slot]`; see
        `_runtime_bound_slot`. Two spellings of one idiom, and this build's spelling is a
        trap.
        """
        if not self.guest_exit_trap or insn.mnemonic != "int3":
            return None
        if not self._reentry_prologue(nxt):
            return None
        return Stop("return", ip, text="int3 guest-exit trap")

    def _jcc(self, insn, ip: int, nxt: int) -> tuple[str, object]:
        target = insn.operands[0].imm
        c = condition(self.a, self.m, JCC[insn.mnemonic])
        cc = self.a.as_const(c)
        if cc is not None:
            return "next", target if cc else nxt
        return "stop", Stop("native_branch", ip, pred=c, taken=target, not_taken=nxt)

    def _indirect_slot(self, insn) -> int | None:
        """The IAT slot an indirect branch goes through, when it goes through one.

        `call qword ptr [slot]` cannot be resolved from the file: the slot holds
        an unbound hint that the loader replaces with a live address. It does not
        need resolving -- the *slot* is the identity of the callee, which is what
        the import table is for. Recognising this turns an unresolvable indirect
        call into an ordinary call to a named import.
        """
        op = insn.operands[0]
        if op.type != CS_OP_MEM:
            return None
        addr = self.a.as_const(self.mem_address(insn))
        if addr is None:
            return None
        return addr if addr in self.imports else None

    def _jmp(self, insn, ip: int, nxt: int) -> tuple[str, object]:
        op = insn.operands[0]
        if op.type == CS_OP_IMM:
            target = op.imm & ((1 << 64) - 1)
            if not self.executable(target):
                return "stop", Stop("bad_target", ip, dest=self.a.const(target, W64))
            return "next", target
        slot = self._indirect_slot(insn)
        if slot is not None:
            mod, sym = self.imports[slot]
            # **A tail call is still a call, and dropping it loses the only thing the block did.**
            # `jmp qword ptr [slot]` hands control to a named import and never comes back to this
            # frame, so the *transfer* is a tail exit -- but the call itself happens, and the payload
            # has to make it. Recording the event first is what puts `call ntoskrnl.exe!X @site` in
            # the listing before the `-> <tail: ...>` arrow; without it, `0x145bd7fa4` (whose whole
            # body is `ExAcquireFastMutex`/`ExReleaseFastMutex` traffic ending in
            # `jmp [ExReleaseFastMutex]`) would have returned without releasing the mutex.
            self._guest_call(insn, ip, nxt, slot)
            return "stop", Stop("tail_call", ip, text=f"{mod}!{sym}",
                                info={"import_slot": slot, "symbol": f"{mod}!{sym}"})
        dest = self.read_op(insn, 0)
        c = self.a.as_const(dest)
        if c is None:
            return "stop", Stop("dispatch", ip, dest=dest)
        if not self.executable(c):
            return "stop", Stop("bad_target", ip, dest=dest, text=f"{c:#x}")
        self.dispatches.append((ip, c))
        return "next", c

    def _resolve_call_thunk(self, target: int) -> int:
        """Follow a one-instruction thunk at a direct call target.

        A `call` to a `.text` address is not necessarily a call to a native
        routine, and the engine used to assume it was. Measuring every call it
        refused to model on one root gave three kinds:

        * `jmp qword ptr [rip+disp]` -- a jump through an IAT slot. The engine
          already has the machinery for a *named import*; it only applied it to
          indirect calls, so a direct call to an import thunk became an opaque.
          **Resolved here.**
        * `jmp rel32` into `.tvm0` -- a trampoline to a *virtualized function*.
          **Deliberately not resolved.** Following it makes the engine inline the
          callee, and a call to another virtualized function is a guest-level
          call: its handler dispatches belong to the callee, not to the caller,
          so inlining them appends the callee's transitions to the caller's
          block. Measured: `0x14014b40e` stops reproducing the reference's
          sequence at transition 0, because the reference's first transition is
          no longer the caller's first. The engine already inlines calls that
          *stay* inside `.tvm0` -- that is the VM's own subroutines -- and this
          distinction is exactly what keeps the two apart.
        * real functions in `.text`, which are what `x86sum` is for.

        Chases are bounded: a thunk that jumps to a thunk is followed a few times
        and then left alone.
        """
        for _ in range(4):
            got = self._thunk_cache.get(target, -1)
            if got == -1:
                got = self._decode_import_thunk(target)
                self._thunk_cache[target] = got
            if got is None:
                return target
            target = got
        return target

    def _decode_import_thunk(self, va: int) -> int | None:
        """The IAT slot the single `jmp` at `va` reads, when that is what it is."""
        size, insn, _raw = self.decode(va)
        if insn is None or insn.mnemonic != "jmp" or len(insn.operands) != 1:
            return None
        if insn.operands[0].type != CS_OP_MEM:
            return None
        addr = self.a.as_const(self.mem_address(insn))
        if addr is not None and addr in self.imports:
            return addr
        return None

    def vm_trampoline_target(self, va: int) -> int | None:
        """The `.tvm0` entry a `jmp rel32` trampoline at `va` reaches, if any.

        Not used by the call path -- see `_resolve_call_thunk` for why following
        it during recovery is wrong. Exposed because it names what a call target
        *is*, which is what turned an unresolvable opaque into a classified
        three-way split.
        """
        size, insn, _raw = self.decode(va)
        if insn is None or insn.mnemonic != "jmp" or len(insn.operands) != 1:
            return None
        op = insn.operands[0]
        if op.type != CS_OP_IMM:
            return None
        tgt = op.imm & ((1 << 64) - 1)
        return tgt if self.in_vm(tgt) else None

    def _runtime_bound_slot(self, tref: int) -> int | None:
        """The `.data` slot a computed call target is read from, when that is what it is.

        Some calls in this module do not go through the IAT and do not go to a
        computed VM destination: they go through a **writable image slot the
        program fills at run time**. The measured shape, twice on `0x14015b1ba`:

            call qword ptr [0x14011c548]        ; 8 bytes at [.data] = 00 x 8 on disk
            call qword ptr [0x14011c538]        ; same

        `PeImage.foldable` is right to refuse those bytes -- the file's zero is not
        the process's value -- so the read correctly comes back as a `load` node and
        the target does not fold. What follows from reading only the image is
        nevertheless decisive: the value is *not* in the image, therefore it is
        supplied from outside, therefore this is a call to an external callee and
        not a VM-internal computed destination. The engine already treats exactly
        this situation as an ordinary call when the slot happens to be a named IAT
        entry (`_indirect_slot` -> `_guest_call`); a slot the image cannot name is
        the same situation minus a name.

        The test is deliberately the narrow one and nothing wider. A `load`
        whose address is **constant** and which `foldable` refuses. That excludes:

        * a read at a *symbolic* address (the address itself is unknown, so nothing
          says where the pointer came from);
        * a read at a constant address **outside the image**, which `foldable`
          accepts -- notably the scratch-window loads the imprecise retry builds at
          concrete frame addresses, which are the VM's own bytes and not an
          external pointer;
        * any computed expression over registers or guest memory.

        Those keep the honest `dispatch_call` stop. Population, and the falsifier
        that had to survive before this rule was written:
        `probe/calltarget_census` -- **2 distinct (root, site) pairs** over the 14
        roots and 2,695,317 interpreted instructions, both `load@constant-nonfoldable`
        in writable `.data`, zero on disk. The falsifier was "if any symbolic call
        destination is not a read at a constant non-foldable address, this is wrong";
        none was, and the population is 2, so the rule rests on two measurements and
        says so.
        """
        a = self.a
        n = a.node(tref)
        if n[0] != "load":
            return None
        addr = a.as_const(n[1])
        if addr is None:
            return None
        if self.img.foldable(addr, max(1, n[2] // 8)):
            return None
        return addr

    def _computed_call_reentry(self, insn, ip: int, nxt: int) -> bool:
        """Recognise the measured saved-stack / Win64 call / VM-frame idiom.

        This is a returning guest-call ABI contract, not a proof that an
        arbitrary unknown callee returns. Require the complete boundary and
        retain the actual target expression in the event.
        """
        if not self.computed_reentry_calls:
            return False
        if self._previous_ip != ip - 4:
            return False
        op = insn.operands[0]
        if (op.type != CS_OP_MEM or op.size != 8 or not op.mem.base
                or self._reg_name(insn, op.mem.base) == 'rip'
                or op.mem.index or op.mem.segment
                or op.mem.disp < 0 or op.mem.disp % 8):
            return False
        size, previous, raw = self.decode(ip - 4)
        if previous is None or size != 4 or raw != bytes.fromhex('488b2424'):
            return False
        size, frame, _ = self.decode(nxt)
        if frame is None or frame.mnemonic != 'lea' or len(frame.operands) != 2:
            return False
        dst, src = frame.operands
        if (dst.type != CS_OP_REG or dst.size != 8 or self._reg_name(frame, dst.reg) != 'rsp'
                or src.type != CS_OP_MEM or self._reg_name(frame, src.mem.base) != 'rsp'
                or src.mem.index or src.mem.segment or src.mem.disp > -8
                or src.mem.disp % 8):
            return False
        frame_size = -src.mem.disp
        size2, save, _ = self.decode(nxt + size)
        if save is None or save.mnemonic != 'mov' or len(save.operands) != 2:
            return False
        dst, src = save.operands
        if (dst.type != CS_OP_MEM or dst.size != 8
                or self._reg_name(save, dst.mem.base) != 'rsp'
                or dst.mem.index or dst.mem.segment or dst.mem.disp < 0
                or dst.mem.disp % 8 or dst.mem.disp + 8 > frame_size
                or src.type != CS_OP_REG or src.size != 8
                or self._reg_name(save, src.reg) != 'rbp'):
            return False
        _, setup, _ = self.decode(nxt + size + size2)
        if setup is None or setup.mnemonic != 'mov' or len(setup.operands) != 2:
            return False
        dst, src = setup.operands
        return (dst.type == CS_OP_REG and dst.size == 8
                and self._reg_name(setup, dst.reg) == 'rbp'
                and src.type == CS_OP_REG and src.size == 8
                and self._reg_name(setup, src.reg) == 'rsp')

    def _call(self, insn, ip: int, nxt: int) -> tuple[str, object]:
        op = insn.operands[0]
        if op.type == CS_OP_IMM:
            target = op.imm & ((1 << 64) - 1)
            target = self._resolve_call_thunk(target)
        else:
            slot = self._indirect_slot(insn)
            if slot is not None:
                return self._guest_call(insn, ip, nxt, slot)
            tref = self.read_op(insn, 0)
            tconst = self.a.as_const(tref)
            if tconst is None:
                bound = self._runtime_bound_slot(tref)
                if bound is not None:
                    self.runtime_bound_calls[bound] = \
                        self.runtime_bound_calls.get(bound, 0) + 1
                    return self._guest_call(insn, ip, nxt, bound)
                if self._computed_call_reentry(insn, ip, nxt):
                    return self._guest_call(insn, ip, nxt, None, target_expr=tref,
                                            continuation_contract='saved-stack Win64 call / VM re-entry')
                return "stop", Stop("dispatch_call", ip, dest=tref)
            target = self._resolve_call_thunk(tconst)

        if self.executable(target) and self.in_vm(target):
            # A call that stays inside the VM is VM control flow: inline it and
            # let the concrete push/pop keep the return address on the machine
            # stack, so the matching `ret` simply falls out of the stack model.
            #
            # Inlining *native* `.text` callees as well was implemented and
            # measured, and it is a **regression**: on the benchmark root the
            # recovered graph stops reproducing the reference's transition
            # sequence at 230 of 802, where restricting inlining to `.tvm0`
            # reproduces all 802, and both roots derail on a conditional branch
            # inside `.text` (`0x14000d51a`) that the engine forks on. The reason
            # it looked attractive is real -- the VM calls native helpers, and an
            # opaque return for those severs every value derived from them -- but
            # interpreting a callee means treating its internal branches as CFG
            # forks, and a native routine's branch is not a virtualized program
            # point. The experiment and its numbers are kept here so it is not
            # repeated; `probe/scratch_slot.py` is the probe that measures it.
            from .stackops import push
            push(self, self.a.const(nxt, W64), 8, ip)
            return "next", target

        return self._guest_call(insn, ip, nxt, target)

    def _guest_call(self, insn, ip: int, nxt: int, target: int | None,
                    *, target_expr: int | None = None,
                    continuation_contract: str | None = None) -> tuple[str, object]:
        """A call out of the virtualized function: a guest-visible event.

        Before falling back to an opaque return, a call to code this image
        contains is offered to the summary interpreter. That is the third option
        between "throw the value away" and "inline the callee and adopt its
        branches as CFG edges" -- the second was measured and is a regression.
        See `x86sum.py`.
        """
        failed_stores: list = []
        if self.summary_enabled and self.summary_depth < self.summary_max_depth:
            worth = (target is not None and target not in self.imports
                     and self.executable(target)
                     and not self.in_vm(target)
                     and self.summary_steps < self.summary_budget)
            if worth:
                self.summary_depth += 1
                try:
                    res = x86sum.summarize(self, target, nxt, self.summary_limit)
                finally:
                    self.summary_depth -= 1
                self.summary_steps += res.steps
                if res.ok:
                    self.summary_ok += 1
                    return "next", nxt
                self.summary_rejects[res.reason] = \
                    self.summary_rejects.get(res.reason, 0) + 1
                # **A failed summary is still a measurement of where the callee writes.** The
                # partial interpretation performed `res.stores` stores before it stopped, and those
                # addresses are the evidence the write region is built from -- see
                # `Machine.forget_call_writes`. Discarding them is what forced the earlier guess of
                # "everything above the pointer", which `probe/callcost.py` shows points the wrong
                # way: the callees here write below their pointer argument.
                failed_stores = res.stores

        self.call_count += 1
        args = [(r, self.m.reg(r, 64)) for r in ("rcx", "rdx", "r8", "r9")]
        # **Read the writable pointers BEFORE the clobber.** `rcx`/`rdx`/`r8`/`r9` are volatile, so
        # once the callee's destruction is modelled the argument values are gone -- and they are the
        # only thing that says which memory the callee could name.
        # The target may itself be an IAT slot address, in which case the call
        # goes through the import table and the callee is a named import.
        import_slot = target if target in self.imports else None
        ret = self.a.opaque(f"call_ret#{self.call_count}", W64)
        event = {
            "kind": "call",
            "site": ip,
            "target": target,
            "args": args,
            "ret": ret,
            "import_slot": import_slot,
        }
        if target_expr is not None:
            event['target_expr'] = target_expr
            event['continuation_contract'] = continuation_contract
        from .callcontracts import import_noreturn
        contract = (import_noreturn(*self.imports[import_slot])
                    if import_slot is not None else None)
        if contract is not None:
            event['noreturn'] = True
            event['noreturn_contract'] = contract.name
            if contract.symbol == 'KeBugCheckEx':
                address = self.a.bin('add', self.m.reg('rsp', 64), self.a.const(0x20, 64), 64)
                event['stack_args'] = {'4': self.m.load(address, 8, ip, self.img.image_u8)}
        self.events.append(event)
        if self.guest_ctx is not None:
            from .ir import call_target
            ordered = dict(event)
            ordered['name'] = call_target(self, event)
            self.guest_trace.append((self.steps, 'call', ordered))
        if contract is not None:
            return 'stop', Stop('noreturn', ip, text=f'{contract.name}: {contract.reason}',
                                info={'event': event, 'callee': contract.name,
                                      'reason': contract.reason, 'source': contract.source})
        call_writes = self.m.call_write_targets(args)
        # Win64: volatile registers are clobbered by the callee.
        for r in ("rax", "rcx", "rdx", "r8", "r9", "r10", "r11"):
            self.m.set_reg(r, self.a.opaque(f"call{r}#{self.call_count}", W64), 64)
        # Win64 also makes XMM0..5 volatile. Materialise both halves even if
        # previously unseen so a later read cannot revive its incoming value.
        for index in range(6):
            for half in range(2):
                self.m.xmm[(f'xmm{index}', half)] = self.a.opaque(
                    f'callxmm{index}.{half}#{self.call_count}', W64)
        for flag in ('cf', 'pf', 'af', 'zf', 'sf', 'of'):
            self.m.set_flag(flag, self.a.opaque(f'flag_{flag}#call{self.call_count}', W8))
        self.m.set_reg("rax", ret, 64)
        # **A native call can write memory, and the model said it could not.** Only the registers
        # and the flags were clobbered, so a read after a call returned the pre-call value. What the
        # callee writes is now described rather than guessed: a named import from its signature
        # (`core/callsum.py`), an intra-image callee from the stores its (possibly failed) partial
        # interpretation performed, and only when neither is available does it fall back to a
        # bounded region below the pointer. `probe/callmem.py` and `probe/callcost.py` are the
        # measurements behind each of those three.
        described = None
        if import_slot is not None:
            mod, sym = self.imports[import_slot]
            from . import callsum
            described = callsum.import_region(f"{mod}!{sym}",
                                              {r: self.a.as_const(v) for r, v in args})
        self.m.forget_call_writes(ip, call_writes, observed=failed_stores,
                                  described=described)
        return "next", nxt

    def _ret(self, insn, ip: int, nxt: int) -> tuple[str, object]:
        from .stackops import adjust, pop
        dest = pop(self, 8, ip)
        if insn.operands:
            adjust(self, insn.operands[0].imm & 0xffff)
        c = self.a.as_const(dest)
        if c is None:
            return "stop", Stop("return", ip, dest=dest)
        if self.in_vm(c):
            # Return from a VM-internal subroutine.
            return "next", c
        return "stop", Stop("return", ip, dest=dest, info={"target": c})

    # --------------------------------------------------------------- SSE moves

    def xmm_halves(self, name: str) -> tuple[int, int]:
        """`(low64, high64)` for an XMM register, creating opaques on first use.

        The register file has no XMM state at all -- `Machine.xmm` was declared
        and never used -- so every SSE instruction was boxed and, because
        `_READ_ONLY_MEM` calls their memory operand read-only, every SSE *store*
        was silently discarded. Two 64-bit halves are enough for the move group
        and avoid needing a 128-bit width in the arena.
        """
        out = []
        for half in (0, 1):
            key = (name, half)
            v = self.m.xmm.get(key)
            if v is None:
                v = self.a.opaque(f"xmm{name}#{half}", W64)
                self.m.xmm[key] = v
            out.append(v)
        return out[0], out[1]

    def _sse_move(self, insn, ip: int, nxt: int) -> tuple[str, object]:
        dst, src = insn.operands[0], insn.operands[1]
        if dst.type not in (CS_OP_REG, CS_OP_MEM) or src.type not in (CS_OP_REG, CS_OP_MEM):
            return self._box(insn, ip, nxt, insn.bytes)
        if dst.type == CS_OP_MEM and src.type == CS_OP_MEM:
            return self._box(insn, ip, nxt, insn.bytes)
        nb, tail = _SSE_MOVES[insn.mnemonic]
        a = self.a

        # --- source
        if src.type == CS_OP_REG:
            nm = insn.reg_name(src.reg)
            if nm.startswith("xmm"):
                lo, hi = self.xmm_halves(nm)
            else:
                lo, hi = self.m.reg(nm, 64), a.const(0, W64)
        else:
            base = self.mem_address(insn)

            def at(off: int, n: int) -> int:
                ad = base if off == 0 else a.bin("add", base, a.const(off, W64), W64)
                return self.load_at(ad, n, ip)

            if nb == 16:
                lo, hi = at(0, 8), at(8, 8)
            elif nb == 8:
                lo, hi = at(0, 8), None
            else:
                lo, hi = a.zext(at(0, nb), 64), None

        # --- destination
        if dst.type == CS_OP_REG:
            nm = insn.reg_name(dst.reg)
            if nm.startswith("xmm"):
                cur_lo, cur_hi = self.xmm_halves(nm)
                if nb == 16:
                    new_lo, new_hi = lo, hi if hi is not None else cur_hi
                elif nb == 8:
                    new_lo = lo
                    new_hi = a.const(0, W64) if tail == "zero" else cur_hi
                else:
                    keep = (1 << 64) - 1 - ((1 << (nb * 8)) - 1)
                    new_lo = a.bin("or", a.bin("and", cur_lo, a.const(keep, W64), W64),
                                   lo, W64)
                    new_hi = cur_hi
                self.m.xmm[(nm, 0)] = new_lo
                self.m.xmm[(nm, 1)] = new_hi
            else:
                # `movq rax, xmm0` and friends.
                w = dst.size * 8
                self.m.set_reg(nm, a.trunc(lo, w) if w < 64 else lo, w)
            return "next", nxt

        base = self.mem_address(insn)
        self.m.store(base, lo if nb >= 8 else a.trunc(lo, nb * 8), min(nb, 8), ip)
        if nb == 16:
            self.m.store(a.bin("add", base, a.const(8, W64), W64),
                         hi if hi is not None else a.const(0, W64), 8, ip)
        return "next", nxt

    # ------------------------------------------------------------- boxed ops

    def _box(self, insn, ip: int, nxt: int, raw: bytes) -> tuple[str, object]:
        """Record an unmodelled instruction and continue with opaque results.

        The set of registers to clobber comes from capstone's own access
        analysis, never from "operand 0 is the destination". That assumption is
        wrong for the CET encodings the VM emits -- `nop r15, rcx` lists two
        register operands and writes neither -- and getting it wrong turns a
        flag-preserving no-op into a clobber of a live guest register, which is
        exactly the kind of silent corruption the next branch then acts on.
        """
        defs: list[tuple[str, int]] = []
        uses: list[tuple[str, int]] = []
        regs_read, regs_written = insn.regs_access()
        md = self._md
        for rid in regs_written:
            name = md.reg_name(rid)
            if name not in SUB or SUB[name][0] not in GP64_NAMES:
                continue
            ref = self.a.opaque(f"box{self.steps}#{name}", W64)
            self.m.set_reg(name, ref, 64)
            defs.append((name, ref))
        for rid in regs_read:
            name = md.reg_name(rid)
            if name in SUB and SUB[name][0] in GP64_NAMES:
                uses.append((name, self.m.reg(name, 64)))
        # Memory: only operand 0 can be a destination, and only for instructions
        # that are not pure reads. `nop`/`cmp`/`test` and the control transfers
        # never write the memory they name.
        if insn.operands and insn.operands[0].type == CS_OP_MEM and insn.mnemonic not in _READ_ONLY_MEM:
            addr = self.mem_address(insn)
            c = self.a.as_const(addr)
            if c is not None:
                self.m.forget(addr, insn.operands[0].size)
        self.m.set_flags_unknown(("cf", "pf", "af", "zf", "sf", "of"))
        self.boxed.append(Boxed(ip, f"{insn.mnemonic} {insn.op_str}", raw, defs, uses))
        return "next", nxt


class _Unsupported(Exception):
    pass


class _Refuse(Exception):
    def __init__(self, kind: str, text: str, info: dict | None = None):
        super().__init__(text)
        self.kind = kind
        self.text = text
        self.info = info or {}


# ------------------------------------------------------------------ handlers
#
# Each takes (lifter, insn, ip, next_ip) and returns ("next", ip) or ("stop", Stop).


def _h_mov(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    v = L.read_op(insn, 1)
    L.write_op(insn, 0, v, w)
    return "next", nxt


def _h_movabs(L: Lifter, insn, ip, nxt):
    return _h_mov(L, insn, ip, nxt)


def _h_lea(L: Lifter, insn, ip, nxt):
    addr = L.mem_address(insn)
    L.write_op(insn, 0, addr, 64)
    return "next", nxt


def _h_movzx(L: Lifter, insn, ip, nxt):
    sw = L.op_width(insn, 1)
    v = L.read_op(insn, 1)
    L.write_op(insn, 0, L.a.zext(v, L.op_width(insn, 0)))
    return "next", nxt


def _h_movsx(L: Lifter, insn, ip, nxt):
    v = L.read_op(insn, 1)
    L.write_op(insn, 0, L.a.sext(v, L.op_width(insn, 0)))
    return "next", nxt


def _h_movsxd(L: Lifter, insn, ip, nxt):
    v = L.read_op(insn, 1)
    L.write_op(insn, 0, L.a.sext(v, 64))
    return "next", nxt


# The SSE move group, in bytes, and what the instruction does to the half of an
# XMM destination that it does not write.
#
# Every one of these used to be boxed, and `_READ_ONLY_MEM` then told the boxed
# path that its memory operand is only read -- so `movdqu xmmword ptr [rsp+X],
# xmm0` **silently wrote nothing**. A later read of that slot came back unknown
# for a reason the engine never recorded anywhere, which is the same failure this
# workspace keeps meeting: a real effect in a shape nothing can see.
#
# It cost the last short root its last 137 transitions. The dispatch at
# `0x1401de67b` reads a spill slot that one of these instructions was supposed to
# write; the engine reports the slot unwritten and stops, and the reference
# resolves the branch. The value was not unknowable -- it was never stored.
#
# Registers are modelled as two 64-bit halves, which is enough for a move group
# and needs no 128-bit width in the arena. `zero`/`keep` is what the ISA says
# happens to the untouched half: MOVQ to an XMM register clears bits 127:64, and
# the scalar SSE moves leave the rest of the register alone.
_SSE_MOVES = {
    "movdqu": (16, "all"),
    "movups": (16, "all"),
    "movaps": (16, "all"),
    "movdqa": (16, "all"),
    "movq": (8, "zero"),
    "movsd": (8, "keep"),
    "movss": (4, "keep"),
}


def _h_sse_move(L: Lifter, insn, ip, nxt):
    return L._sse_move(insn, ip, nxt)


# `xorps xmm0, xmm0` is the standard way to zero an XMM register, and boxing it
# leaves the register holding whatever it held before -- so every 16-byte spill of
# it stores garbage. With the move group modelled, these were the *only* two
# instruction kinds left boxed on the last short root.
_XMM_XORS = ("xorps", "xorpd", "pxor", "vxorps", "vxorpd", "vpxor")


def _h_sse_xor(L: Lifter, insn, ip, nxt):
    """Bitwise XOR of two XMM values, which is per-half and therefore exact."""
    if len(insn.operands) != 2:
        return L._box(insn, ip, nxt, insn.bytes)
    dst, src = insn.operands[0], insn.operands[1]
    if dst.type != CS_OP_REG or src.type != CS_OP_REG:
        return L._box(insn, ip, nxt, insn.bytes)
    dn, sn = insn.reg_name(dst.reg), insn.reg_name(src.reg)
    if not dn.startswith("xmm") or not sn.startswith("xmm"):
        return L._box(insn, ip, nxt, insn.bytes)
    d_lo, d_hi = L.xmm_halves(dn)
    s_lo, s_hi = L.xmm_halves(sn)
    a = L.a
    L.m.xmm[(dn, 0)] = a.bin("xor", d_lo, s_lo, W64)
    L.m.xmm[(dn, 1)] = a.bin("xor", d_hi, s_hi, W64)
    return "next", nxt


def _binop(op_name):
    def go(L: Lifter, insn, ip, nxt):
        w = L.op_width(insn, 0)
        x = L.read_op(insn, 0)
        y = L.read_op(insn, 1)
        r = L.a.bin(op_name, x, y, w)
        L.write_op(insn, 0, r, w)
        if op_name == "add":
            L._add_flags(x, y, r, w)
        elif op_name == "sub":
            L._sub_flags(x, y, r, w)
        elif op_name in ("and", "or", "xor"):
            L._logic_flags(r, w)
        return "next", nxt

    return go


def _h_not(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    L.write_op(insn, 0, L.a.un("not", L.read_op(insn, 0), w), w)
    return "next", nxt


def _h_neg(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    x = L.read_op(insn, 0)
    r = L.a.un("neg", x, w)
    L.write_op(insn, 0, r, w)
    L._sub_flags(L.a.const(0, w), x, r, w)
    return "next", nxt


def _h_inc(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    x = L.read_op(insn, 0)
    r = L.a.bin("add", x, L.a.const(1, w), w)
    L.write_op(insn, 0, r, w)
    # **INC and DEC leave CF exactly as it was** (Intel SDM: "The CF flag is not
    # affected"), and this used to clobber it with the addition's carry-out. The
    # previous form was `set_flag("cf", opaque("flag_cf"))` followed by
    # `_add_flags(...)`, i.e. an attempt to record "unknown" that the very next call
    # overwrote with a carry-out that x86 does not compute -- so the engine held a
    # definite, definite-*wrong* bit on a register that `pushfq` packs straight into
    # the dispatch word. Preserving the flag is both the documented semantics and the
    # honest one: it needs no unknown at all.
    carried = L.m.flag("cf")
    L._add_flags(x, L.a.const(1, w), r, w)
    L.m.set_flag("cf", carried)
    return "next", nxt


def _h_dec(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    x = L.read_op(insn, 0)
    r = L.a.bin("sub", x, L.a.const(1, w), w)
    L.write_op(insn, 0, r, w)
    carried = L.m.flag("cf")          # preserved, as above
    L._sub_flags(x, L.a.const(1, w), r, w)
    L.m.set_flag("cf", carried)
    return "next", nxt


def _h_cmp(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    x = L.read_op(insn, 0)
    y = L.read_op(insn, 1)
    L._sub_flags(x, y, L.a.bin("sub", x, y, w), w)
    return "next", nxt


def _h_test(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    r = L.a.bin("and", L.read_op(insn, 0), L.read_op(insn, 1), w)
    L._logic_flags(r, w)
    return "next", nxt


def _h_xchg(L: Lifter, insn, ip, nxt):
    """`xchg` -- and the one place a pair of them can be proved to do nothing.

    The handlers are built from `xchg`/`not` pairs (760 k `not` and 730 k `xchg`
    against 1.02 M `mov` in 600 sampled handler functions of the DeltaForce build),
    so a cancelled pair is worth removing at the source: `Machine.set_reg` rebuilds
    sub-64-bit registers out of `and`/`or`/`shl`, and those rebuilds cannot be folded
    afterwards, which is how a two-instruction no-op becomes a permanent widening
    ladder in the value DAG.

    Cancellation is delegated to `mba.SwapTracker`, which only claims it when the
    same register pair is swapped twice with no write to either in between. When it
    fires, both registers have already been restored to the refs they held before the
    first swap, so the right thing to do is nothing at all.
    """
    w = L.op_width(insn, 0)
    pair = _xchg_register_pair(insn)
    tracker = getattr(L.m, "swap_tracker", None)
    if pair is not None and tracker is not None:
        if tracker.should_cancel(pair[0], pair[1], w):
            return "next", nxt
        saved = tracker.snapshot(pair[0], pair[1])
    else:
        saved = None
    x = L.read_op(insn, 0)
    y = L.read_op(insn, 1)
    L.write_op(insn, 0, y, w)
    L.write_op(insn, 1, x, w)
    if pair is not None and tracker is not None and saved is not None:
        # Recorded *after* the writes: the two writes are this swap's own effect and
        # must not invalidate the record the way a third-party write would.
        tracker.remember(pair[0], pair[1], w, saved)
    return "next", nxt


def _xchg_register_pair(insn):
    """`(reg0, reg1)` when both operands are registers, else None."""
    try:
        ops = insn.operands
        if len(ops) != 2 or ops[0].type != CS_OP_REG or ops[1].type != CS_OP_REG:
            return None
        return insn.reg_name(ops[0].reg), insn.reg_name(ops[1].reg)
    except Exception:
        return None


def _shift(op_name):
    def go(L: Lifter, insn, ip, nxt):
        w = L.op_width(insn, 0)
        x = L.read_op(insn, 0)
        cnt = L.read_op(insn, 1) if len(insn.operands) > 1 else L.a.const(1, w)
        k = L.a.as_const(cnt)
        if k is None:
            # A symbolic shift count makes the result unmodellable; the
            # protector uses `cl` only in code that cannot feed dispatch.
            raise _Unsupported("symbolic shift count")
        # **x86 masks the count, and the mask width depends on the operand.** Five bits for
        # an 8-, 16- or 32-bit operand, six for a 64-bit one, so `shl r32, 33` shifts by one
        # on hardware. This used `k % 64` at every width, which shifts by thirty-three and
        # therefore produces 0 where the hardware produces `r32 << 1`.
        #
        # **Latent in this module, measured before the change**: `probe.shift_counts` over
        # all 14 roots counts **2,171** executed shift/rotate flag evaluations across 11
        # distinct (op, width, count) combinations, and every one is either a 64-bit operand
        # or a count below 32 — `shl 32bit count 2` (590 executions), `shr 64bit` counts
        # 1/2/4/63, `sar 64bit` 3/5/23, `rol`/`ror` 64bit count 16. So no execution reaches
        # the deviation and the corpus result of this change is a **null result**, not a
        # recovery improvement. It is fixed because a wrong result is wrong.
        #
        # Rotations need no equivalent change: `_rotate` reduces the count modulo the operand
        # width, and every sub-64-bit width here divides 32, so masking to five bits first
        # cannot change `k % w`.
        eff = k & (0x3F if w == 64 else 0x1F)
        r = L.a.bin(op_name, x, L.a.const(eff, w), w)
        L.write_op(insn, 0, r, w)
        if eff:
            L._zf_sf_pf(r, w)
            L._shift_flags(op_name, x, r, w, eff)
        return "next", nxt

    return go


def _rotate(op_name):
    def go(L: Lifter, insn, ip, nxt):
        w = L.op_width(insn, 0)
        x = L.read_op(insn, 0)
        cnt = L.read_op(insn, 1) if len(insn.operands) > 1 else L.a.const(1, w)
        k = L.a.as_const(cnt)
        if k is None:
            raise _Unsupported("symbolic rotate count")
        eff = k % w
        # **The direction is `op_name`, and it used to be ignored.** The body was
        # `shl(x, eff) | shr(x, w - eff)` for both spellings, i.e. every `ror` was
        # lifted as a `rol`. A linear census of `.tvm0` reported 884 `ror` and no
        # `rol`, which would have made that enormous; measured on executed code
        # (`probe.executed_mnemonics`, all 14 roots) it is **29 executions at 1
        # distinct site**, and `rol` is 29 at 1 site as well. Small population, wrong
        # value, no ambiguity about which.
        if op_name == "rol":
            r = L.a.bin("or",
                        L.a.bin("shl", x, L.a.const(eff, w), w),
                        L.a.bin("shr", x, L.a.const((w - eff) % w, w), w), w)
        else:  # ror
            r = L.a.bin("or",
                        L.a.bin("shr", x, L.a.const(eff, w), w),
                        L.a.bin("shl", x, L.a.const((w - eff) % w, w), w), w)
        L.write_op(insn, 0, r, w)
        if eff:
            # Rotates leave ZF/SF/PF alone and define CF for every count, OF only at 1.
            L._rotate_flags(op_name, x, r, w, eff)
        return "next", nxt

    return go


def _h_adc(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    x = L.read_op(insn, 0)
    y = L.read_op(insn, 1)
    cf = L.a.zext(L.m.flag("cf"), w)
    t = L.a.bin("add", x, y, w)
    r = L.a.bin("add", t, cf, w)
    L.write_op(insn, 0, r, w)
    L._add_flags(t, cf, r, w)
    return "next", nxt


def _h_sbb(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    x = L.read_op(insn, 0)
    y = L.read_op(insn, 1)
    cf = L.a.zext(L.m.flag("cf"), w)
    t = L.a.bin("sub", x, y, w)
    r = L.a.bin("sub", t, cf, w)
    L.write_op(insn, 0, r, w)
    L._sub_flags(t, cf, r, w)
    return "next", nxt


def _h_imul(L: Lifter, insn, ip, nxt):
    from .widemul import high_product, set_multiply_flags, write_double_result
    if len(insn.operands) == 1:
        w = L.op_width(insn, 0)
        x = L.m.reg("rax", w)
        y = L.read_op(insn, 0)
        r = L.a.bin("mul", x, y, w)
        high = high_product(L.a, x, y, w, signed=True)
        write_double_result(L, r, high, w)
        set_multiply_flags(L, r, high, w, signed=True, site=ip)
        return "next", nxt
    w = L.op_width(insn, 0)
    if len(insn.operands) == 2:
        x = L.read_op(insn, 0)
        y = L.read_op(insn, 1)
    else:
        x = L.read_op(insn, 1)
        y = L.read_op(insn, 2)
    r = L.a.bin("mul", x, y, w)
    L.write_op(insn, 0, r, w)
    set_multiply_flags(L, r, high_product(L.a, x, y, w, signed=True), w, signed=True, site=ip)
    return "next", nxt


def _h_mul(L: Lifter, insn, ip, nxt):
    from .widemul import high_product, set_multiply_flags, write_double_result
    w = L.op_width(insn, 0)
    x = L.m.reg("rax", w)
    y = L.read_op(insn, 0)
    r = L.a.bin("mul", x, y, w)
    high = high_product(L.a, x, y, w)
    write_double_result(L, r, high, w)
    set_multiply_flags(L, r, high, w, signed=False, site=ip)
    return "next", nxt


def _h_push(L: Lifter, insn, ip, nxt):
    from .stackops import push
    w = L.op_width(insn, 0)
    v = L.read_op(insn, 0)
    n = w // 8 if w in (16, 64) else 8
    push(L, v, n, ip)
    return "next", nxt


def _h_pushfq(L: Lifter, insn, ip, nxt):
    from .stackops import push
    push(L, L.m.pack_flags(), 8, ip)
    return "next", nxt


def _h_popfq(L: Lifter, insn, ip, nxt):
    from .stackops import pop
    v = pop(L, 8, ip)
    L.m.unpack_flags(v)
    return "next", nxt


def _h_pop(L: Lifter, insn, ip, nxt):
    from .stackops import pop
    w = L.op_width(insn, 0)
    n = w // 8 if w in (16, 64) else 8
    v = pop(L, n, ip)
    L.write_op(insn, 0, v, w)
    return "next", nxt


def _h_leave(L: Lifter, insn, ip, nxt):
    from .stackops import pop
    L.m.set_reg("rsp", L.m.reg("rbp", 64), 64)
    width = 16 if 0x66 in insn.prefix else 64
    v = pop(L, width // 8, ip)
    L.m.set_reg("bp" if width == 16 else "rbp", v, width)
    return "next", nxt


def _h_cdqe(L: Lifter, insn, ip, nxt):
    L.m.set_reg("rax", L.a.sext(L.m.reg("eax", 32), 64), 64)
    return "next", nxt


def _h_cwde(L: Lifter, insn, ip, nxt):
    L.m.set_reg("rax", L.a.sext(L.m.reg("ax", 16), 32), 32)
    return "next", nxt


def _h_cbw(L: Lifter, insn, ip, nxt):
    L.m.set_reg("rax", L.a.sext(L.m.reg("al", 8), 16), 16)
    return "next", nxt


def _h_bswap(L: Lifter, insn, ip, nxt):
    w = L.op_width(insn, 0)
    x = L.read_op(insn, 0)
    acc: int | None = None
    for i in range(w // 8):
        byte = L.a.bin("shr", x, L.a.const(i * 8, w), w) if i else x
        byte = L.a.bin("and", byte, L.a.const(0xFF, w), w)
        byte = L.a.bin("shl", byte, L.a.const((w // 8 - 1 - i) * 8, w), w)
        acc = byte if acc is None else L.a.bin("or", acc, byte, w)
    assert acc is not None
    L.write_op(insn, 0, acc, w)
    return "next", nxt


def _h_nop(L: Lifter, insn, ip, nxt):
    return "next", nxt


def _h_jmp(L: Lifter, insn, ip, nxt):
    return L._jmp(insn, ip, nxt)


def _h_call(L: Lifter, insn, ip, nxt):
    return L._call(insn, ip, nxt)


def _h_ret(L: Lifter, insn, ip, nxt):
    return L._ret(insn, ip, nxt)


_HANDLERS = {
    "mov": _h_mov,
    "movabs": _h_movabs,
    "lea": _h_lea,
    "movzx": _h_movzx,
    "movsx": _h_movsx,
    "movsxd": _h_movsxd,
    "movdqu": _h_sse_move,
    "movups": _h_sse_move,
    "movaps": _h_sse_move,
    "movdqa": _h_sse_move,
    "movq": _h_sse_move,
    "movsd": _h_sse_move,
    "movss": _h_sse_move,
    "xorps": _h_sse_xor,
    "xorpd": _h_sse_xor,
    "pxor": _h_sse_xor,
    "add": _binop("add"),
    "sub": _binop("sub"),
    "and": _binop("and"),
    "or": _binop("or"),
    "xor": _binop("xor"),
    "not": _h_not,
    "neg": _h_neg,
    "inc": _h_inc,
    "dec": _h_dec,
    "cmp": _h_cmp,
    "test": _h_test,
    "xchg": _h_xchg,
    "adc": _h_adc,
    "sbb": _h_sbb,
    "imul": _h_imul,
    "mul": _h_mul,
    "shl": _shift("shl"),
    "sal": _shift("shl"),
    "shr": _shift("shr"),
    "sar": _shift("sar"),
    "rol": _rotate("rol"),
    "ror": _rotate("ror"),
    "push": _h_push,
    "pop": _h_pop,
    "pushfq": _h_pushfq,
    "popfq": _h_popfq,
    "leave": _h_leave,
    "cdqe": _h_cdqe,
    "cwde": _h_cwde,
    "cbw": _h_cbw,
    "bswap": _h_bswap,
    "nop": _h_nop,
    "jmp": _h_jmp,
    "call": _h_call,
    "ret": _h_ret,
}
