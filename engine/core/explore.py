"""CFG recovery by guided state forking.

A single path walk stops the moment control depends on a guest value, which is
correct but useless on its own: virtualized functions branch on guest data
constantly. This module turns one path into a graph.

Three devices do the work.

* **Candidate pinning.** A dispatch address is almost always `table + f(cond)`
  where `cond` is a value the folder can prove is 0 or 1. Substituting 0 and 1
  into the address and re-folding asks the VM directly where each side goes --
  no solver, and nothing enumerated that the VM cannot actually reach.
* **Block identity by (handler, vip).** Tencent's VM shares handler code between
  program points, so a handler address alone is ambiguous. The pair is not. The
  vip is a hash over the in-`.tvm0` pointers in the VM context window, which is
  exactly the state that distinguishes one program point from another -- and it
  deliberately excludes guest data, so a block does not get a new identity for
  every distinct guest value that reaches it.
* **Path-conditioned snapshots.** Each successor specializes registers, flags,
  frame bytes, deferred writes and provenance under its own predicate. A pin
  used only at the next dispatcher leaves the rest of the machine ambiguous.
* **Finite-image proofs.** An optional bounded bit-vector solver proves complete
  multiway target sets and rejects inconsistent paths before sampled heuristics.

Nothing here re-executes an already recovered block, and a jump to a block on
the current path is recorded as a back edge rather than re-walked.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .canon import MUTATIONS, identify, influence_set
from .expr import W64, Arena, boolean_subexpressions, render
from .guestctx import (ARCH, FLAGS, GUEST_CTX_FROM_ENTRY_RBP, HANDLER_TIME,
                       GuestCache, GuestContext, from_working_frame)
from .ir import call_target
from .lift64 import Lifter, Stop
from .mach import Machine
from .state import MachineSnapshot, LifterCheckpoint
from .peimg import PeImage
from . import cfgio
from .gueststate import observe as observe_guest_state

STACK_BASE = 0x7FFF_FFFE_FDA0
CONTEXT_WINDOW = 0x90
#: The second, slower attempt at a dispatch's target image. `bitproof.finite_values` defaults to a
#: 350 ms deadline and a 64-address table because it runs on the path of every dispatch; these are
#: the numbers the *retry* uses, and only after the fast answer failed. Measured on `ACE-CORE.sys3`:
#: five blocks of root `0x145bccd2d` are `too_many` at 64 addresses and `complete` with one target
#: at 256 (`probe/block_exits.py --prove=64`), which is 5 of that root's 7 unresolved blocks.
WIDE_TABLE_LIMIT = 256
WIDE_TIMEOUT_MS = 5000
# **The canonical per-block step budget, defined once.** `core.devirt` and
# `probe.semantic_regression` both derive from this rather than carrying their own copy, and
# `tests/test_budget_defaults.py` fails if they ever drift apart.
#
# The value is a measurement, not a preference: at 400,000 the root `0x140169e9d` truncated a
# block and reported 1 unresolved; at 900,000 it reports 17 blocks / 0 unresolved with a clean
# verifier, and corpus-wide the change is 139 -> 150 blocks and 2 -> 1 unresolved with no root
# worse (`evidence/E-BUDGET-OR-LOOP.md`). A caller that passes its own smaller budget is
# choosing to truncate, which is legitimate — but doing it *by accident* is what this constant
# exists to prevent.
DEFAULT_STEP_BUDGET = 900_000
# Site marker for stores the *engine* performs rather than the VM, so a census can
# tell its own writes from the program's. Also used by the probes.
SEED_SITE = -1
# The host general registers, in the order the guest cache is reported.
HOST_REGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
             "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
@dataclass(frozen=True)
class BlockKey:
    """A block's identity: the handler, and a hash of the VM context window.

    **The second field is not a vip, and calling it one was a plain error in
    labelling that survived three rounds.** It is `Recoverer.identity(L)` -- a 64-bit
    hash of every in-`.tvm0` value in the context window plus the vip slot -- and it
    is printed as `@vip0x...`, which is the syntax the reference's `lower` uses for
    something else entirely. Output from this engine and from the reference therefore
    *looked* comparable while the two fields meant different things.

    The measurement that settles it is `probe/vipslot`: taking the `(handler, vip)`
    pairs the reference prints and scanning for those exact values found **nothing** --
    not in the frame window `rbp-0x200..rbp+0x200`, not in any of the fifteen general
    registers, on 802 dispatch boundaries. So the engine cannot compute the
    reference's vip at all; it is computed at the dispatch or encoded through a
    transform, and no field of this engine's state holds it.

    What follows is that blocks from the two tools can be compared **by handler and
    not by key**, which is a correction to round 3's comparison as well as to this
    label. The field is left in place, and its name with it, because `cfgio`
    serialises it and a rename would churn a working format for no gain -- but it is a
    context hash and the rendering below no longer claims otherwise.
    """

    handler: int
    vip: int

    def __str__(self) -> str:
        return f"{self.handler:#x}@ctx{self.vip:#018x}"


@dataclass
class Block:
    key: BlockKey
    preds: list[BlockKey] = field(default_factory=list)
    succs: list[BlockKey] = field(default_factory=list)
    kind: str = "unresolved"
    detail: str = ""
    predicate: str = ""
    # **Which successor each value of `predicate` selects**, one entry per `succs` entry, or None
    # where the association was not established. The listing has always printed the successors of a
    # two-way block and never said which is taken when, so a backend reading it could not emit a
    # conditional branch without inventing the mapping. `resolve_dispatch` returns its targets in
    # resolver order, which is *not* predicate-value order for a dispatch -- `_terminate` verifies
    # the association by pinning the predicate both ways and substituting, and the answer was being
    # thrown away after the edge pins were built.
    edge_cond: list = field(default_factory=list)
    pred_node: int = 0
    retry: str = ""
    hypothesis: bool = False
    cost: int = 0
    events: int = 0
    # Address of the instruction that ended the block. An edge is "the branch at
    # `site` lands on `succ`", not "the block entered at `handler` reaches
    # `succ`" -- a block spans many instructions and the transfer happens at its
    # last one. Comparing the wrong pair makes every edge look wrong.
    site: int = 0
    # Every *indirect* transfer the lifter folded while walking this block, in
    # order. A block ends only at a transfer the evaluator could not resolve, so
    # it contains all the ones it could -- and those are exactly what an
    # independent tracer records. Without this list an edge can only be compared
    # by its terminator, and a coarse block makes 100 % of its interior
    # transitions invisible.
    transfers: list[tuple[int, int]] = field(default_factory=list)
    # Every native call the lifter made while walking this block, as
    # `(site, target name)`. The lifter already records these in `Lifter.events`
    # and clears the list at the start of each block, so this is the moment they
    # exist; until now only their *count* was kept and the events were dropped, which
    # is why an emitted IR could not be compared against the reference's `lower`
    # output on the one thing that is objective about it.
    calls: list[tuple[int, str]] = field(default_factory=list)
    # Guest register writes made while walking this block, as `(guest register, value
    # node)`. Only populated when the census is on (`guest_census=True`). The arena is
    # shared across blocks -- a block restores a machine snapshot, not a fresh arena --
    # so a node id recorded here stays valid for a renderer afterwards.
    guest_writes: list[tuple[str, int]] = field(default_factory=list)
    # Host-register writes this block made to a register that was carrying a *known*
    # guest register, as `(guest register, host register, value node)`. This is what
    # reaches the blocks that never touch the context -- 144 of 149 on the benchmark
    # root -- and the provenance is a chain from a context read, invalidated by any
    # write that does not depend on the previous value, so an attribution here means
    # the register really was carrying guest state at that point.
    guest_reg_writes: list[tuple[str, str, int]] = field(default_factory=list)
    # Writes that computed with a **copy** of a guest register rather than updating it,
    # as `(guest register, host register, the copied value, the new value)`. These are
    # computation, not state, and they are what the emitter renders as
    # `t<host> = f(guest r12)`.
    guest_derived: list[tuple[str, str, int, int]] = field(default_factory=list)
    # Loads through a guest register -- a guest memory read. See
    # `Lifter.note_guest_load`; captured per block so the emitter can show them.
    guest_loads: list[tuple[int, int]] = field(default_factory=list)
    # Every guest event in execution order: (step, kind, payload). See `Lifter.guest_trace`.
    guest_trace: list[tuple[int, str, tuple]] = field(default_factory=list)
    guest_values: dict[int, str] = field(default_factory=dict)
    guest_identity: dict[int, str] = field(default_factory=dict)
    # Physical context state, independent of census attribution. The Guest
    # transfer compiler cuts these values at stable block parameters.
    guest_entry: dict[str, int] = field(default_factory=dict)
    guest_exit: dict[str, int] = field(default_factory=dict)
    memory_entry: int = 0
    memory_exit: int = 0
    guest_edges: dict = field(default_factory=dict)
    native_return: dict[str, int] = field(default_factory=dict)
    native_return_phase: str = ''
    native_return_flags: dict = field(default_factory=dict)
    terminal: dict = field(default_factory=dict)


@dataclass
class Cfg:
    entry: BlockKey | None = None
    blocks: dict[BlockKey, Block] = field(default_factory=dict)
    order: list[BlockKey] = field(default_factory=list)
    steps: int = 0
    forks: int = 0
    hypotheses: int = 0
    unresolved: int = 0
    # Native-call summaries: how many callees were interpreted to a value, and
    # why the rest were abandoned. See `x86sum.py`.
    summary_ok: int = 0
    summary_steps: int = 0
    summary_rejects: dict[str, int] = field(default_factory=dict)
    # Search-side cost, so that "this blows up" can be attributed to the search
    # or to the graph rather than guessed at.
    work_peak: int = 0
    snapshot_cells_peak: int = 0
    arena_nodes: int = 0
    transfers: int = 0

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for b in self.blocks.values():
            out[b.kind] = out.get(b.kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def edges(self) -> list[tuple[int, int]]:
        """(dispatch site, successor entry) for every edge.

        Keyed by the address of the transferring instruction, not by the block's
        own entry: a block runs many instructions and only its last one branches.
        """
        return sorted({(b.site, s.handler) for b in self.blocks.values()
                       for s in b.succs if b.site})


def hypothesis_candidates(a: Arena, r: int, max_leaves: int = 4, max_masks: int = 10):
    """Bit hypotheses for a dispatch that no provably-boolean value can resolve.

    The VM's dispatcher picks the next handler out of a table, and the index is
    read from a value it cannot fold -- a bytecode operand, a guest variable, a
    call result. What the *address arithmetic* then does with that value is
    opaque, but the VM still only ever needs one bit of it to choose a branch,
    because a two-entry table is two entries.

    So rather than give up, the engine names the question: "does bit `k` of this
    leaf look like this?" It rewrites the leaf as
    `select(bit_k(leaf), leaf | m, leaf & ~m)`, pins that boolean to 0 and 1, and
    re-folds the address. A hypothesis is only kept when the folded target is
    executable code, and a two-sided hypothesis is only kept when *both* sides
    are -- so a wrong guess shows up as a dead end rather than as a fabricated
    edge.

    Yields (leaf, mask, boolean_ref) in the order they are worth trying.
    """
    out = []
    seen_leaves = []
    stack = [r]
    seen = set()
    while stack and len(seen_leaves) < max_leaves * 4:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        n = a.nodes[cur]
        t = n[0]
        if t in ("load", "opaque"):
            seen_leaves.append(cur)
        elif t == "bin":
            stack.extend((n[2], n[3]))
        elif t == "un":
            stack.append(n[2])
        elif t in ("trunc", "zext", "sext"):
            stack.append(n[1])
        elif t == "sel":
            stack.extend((n[1], n[2], n[3]))
        elif t == "load":
            stack.append(n[1])

    for leaf in seen_leaves[:max_leaves]:
        w = a.width(leaf)
        # Sign bit first: a two-entry table indexed by a value the VM read is
        # usually discriminated by its top bit, which is also what the reference
        # implementation ends up splitting on at these sites.
        bits = [w - 1, 7, 6, 5, 4, 3, 2, 1, 0, 15]
        for k in bits[:max_masks]:
            if k >= w:
                continue
            m = 1 << k
            masked = a.bin("and", leaf, a.const(m, w), w)
            bit = a.bin("eq", masked, a.const(m, w), w)
            rewritten = a.select(
                bit,
                a.bin("or", leaf, a.const(m, w), w),
                a.bin("and", leaf, a.const(((1 << w) - 1) ^ m, w), w),
                w,
            )
            out.append((leaf, m, bit, rewritten))
    return out


class Recoverer:
    def __init__(self, img: PeImage, vm_lo: int, vm_hi: int, stack_base: int = STACK_BASE,
                 step_budget: int = DEFAULT_STEP_BUDGET, block_budget: int = 512,
                 max_forks: int = 8192,
                 journal: str | None = None, guest_census: bool = False,
                 path_sensitive: bool = True, proof_dispatch: bool = True,
                 entry_context=None):
        self.img = img
        self.vm_lo = vm_lo
        self.vm_hi = vm_hi
        self.stack_base = stack_base
        self.step_budget = step_budget
        self.block_budget = block_budget
        self.max_forks = max_forks
        self.path_sensitive = path_sensitive
        self.conditioned_edges = 0
        self.proof_dispatch = proof_dispatch
        self.entry_context = entry_context
        self.entry_contract = {}
        self.proof_stats: dict[str, int] = {}
        self._last_dispatch_proven = False
        self._last_dispatch_infeasible = False
        # Path to write the graph to as it is recovered, one record per block,
        # flushed per block. A run that is cut short keeps what it had.
        # See `cfgio`.
        self.journal = journal
        self.journal_truncated = False
        self.vip_slot: int | None = None
        self.unwind_slot: int | None = None
        self.vip_samples = 0
        # The guest register image, once a frame has been sampled. Set in
        # `_prepare`; `None` means the walk reports addresses, not register names.
        self.guest_ctx: GuestContext | None = None
        self.guest_census = guest_census
        # Filled by `derive_guest_cache`, which is a separate measurement and not
        # run by `recover`. `None` means "not derived", not "no cache".
        self.handler_cache: GuestCache | None = None
        self.handler_cache_walk: dict = {}
        # Raw register writes from the seeded walk and the arena that owns them,
        # interpreted by `guest_operations` once the cache is known.
        self.guest_ops: list[tuple[int, str, int, int]] = []
        self._op_arena: Arena | None = None

    # ------------------------------------------------------------------ setup

    def _lifter(self, arena: Arena, machine: Machine) -> Lifter:
        L = Lifter(self.img, machine, arena, self.vm_lo, self.vm_hi)
        L.unwind_slot = self.unwind_slot
        return L

    def _install_load_folder(self, arena: Arena, L: Lifter) -> None:
        """Let the folder answer reads at a constant address.

        A read is lifted once, and at that moment its address may still be
        symbolic, so the machine records a `load` node rather than a value --
        correctly. Substitution can fold the address afterwards, and the rebuilt
        node is then a read at a *constant* address that nothing folds, because
        the arena owns neither the image nor the memory map.

        The rule here is deliberately the read site's own rule, no wider:

        * the same `foldable` test, so nothing the loader or the runtime rewrites
          is ever folded -- the set of addresses that may fold is unchanged by
          this hook, only the moment at which the question is asked;
        * the read's own memory state, so later stores cannot change historical reads;
        * a value only if the result is a constant; a read that is still symbolic
          keeps its node, so two reads of one address still cancel.
        """
        img = self.img

        def fold(addr: int, width: int, version: int) -> int | None:
            if width not in (8, 16, 32, 64):
                return None
            if version < 0:
                return None
            nbytes = width // 8
            if not img.foldable(addr, nbytes):
                return None
            # A conditioned snapshot can have different byte values under the
            # same historical epoch. Arena normalization is shared by siblings,
            # so even a current-epoch node must use immutable write history.
            return L.m.historical_constant(addr, nbytes, version, img.image_u8)

        arena.versioned_load_folder = fold

    def _initial_state(self, root):
        if self.entry_context is not None:
            if self.entry_context.target_root != root:
                raise ValueError('caller context targets a different root')
            arena, machine, contract = self.entry_context.replay(
                self.img, self.vm_lo, self.vm_hi, self.stack_base)
            self.entry_contract = contract
            return arena, machine
        arena = Arena()
        return arena, Machine(arena, self.stack_base)

    def prepare(self, root: int) -> tuple[Arena, Machine, Lifter]:
        """Locate the frame invariants on a throwaway run, then start clean."""
        pa, pm = self._initial_state(root)
        probe = self._lifter(pa, pm)
        sample: dict[int, set[int]] = {}
        ip = root
        for i in range(20_000):
            if i % 16 == 0:
                rbp = pa.as_const(pm.reg("rbp", 64))
                if rbp is not None:
                    for off in range(0, CONTEXT_WINDOW, 8):
                        v = pm.load(pa.const(rbp + off, 64), 8, 0, self.img.image_u8)
                        c = pa.as_const(v)
                        if c is not None and self.vm_lo <= c < self.vm_hi:
                            sample.setdefault(off, set()).add(c)
            kind, payload = probe.step(ip)
            if kind != "next":
                break
            ip = payload
        # The vip is the slot whose in-`.tvm0` value moves most, because
        # executing a VM instruction is what advances it.
        if sample:
            self.vip_slot = max(sample.items(), key=lambda kv: (len(kv[1]), -kv[0]))[0]
            self.vip_samples = len(sample)
        # The unwind slot is a `.tvm0` pointer the VM parks at [rbp+0] for the
        # unwinder; it never moves, so it is not the vip and must stay out of
        # the block identity.
        rbp = pa.as_const(pm.reg("rbp", 64))
        if rbp is not None:
            u = pa.as_const(pm.load(pa.const(rbp, 64), 8, 0, self.img.image_u8))
            if u is not None and self.vm_lo <= u < self.vm_hi:
                self.unwind_slot = u

        arena, machine = self._initial_state(root)
        # The machine the *sampling* walk ran on. Kept because its diagnostics --
        # `sym_stores`, `sym_store_windows`, `unmapped_loads` -- describe the whole
        # path, while the per-block machines are transient and describe one block.
        self.machine = machine
        # Kept for the same reason: `Block.guest_writes` stores value nodes, and a
        # renderer needs the arena that owns them.
        self.arena = arena
        L = self._lifter(arena, machine)
        # And the lifter, so a caller can read its diagnostics -- `grw_rejected`,
        # `sym_stores`, `guest_reads` -- after a recovery rather than during one.
        self.lifter = L
        self._install_load_folder(arena, L)
        # The guest register image sits at a fixed offset above the frame this
        # walk sampled, so a block interpreted from here on can name guest
        # register reads and writes instead of reporting bare addresses. The
        # offset is a measurement with a stated derivation, not a magic number:
        # `core/guestctx.py` and `evidence/E-GUEST-CONTEXT-LOCATED.md`.
        #
        # The census itself is opt-in because it costs about five percent of a
        # run; `guest_census=True` is for callers that read it.
        if rbp is not None:
            self.guest_ctx = from_working_frame(rbp)
        elif self.handler_cache is not None and getattr(self.handler_cache, "ctx", None):
            # **A symbolic first-block `rbp` is not the same as no frame at all, and treating
            # them as the same produced an empty listing.** `derive_guest_cache` samples `rbp`
            # at every dispatch boundary and builds the context the first time it is concrete;
            # this walk samples it once, in the first block. On `0x1401687d5` the first sample
            # is symbolic, so the census was never installed, and a 15187-instruction function
            # devirtualised to a single `none` line with no indication that anything had gone
            # wrong -- while the cache derivation on the same root had found the frame and 460
            # boundaries. The frame is a property of the function, not of the block the walk
            # happened to start in, so the one that was found is used.
            self.guest_ctx = self.handler_cache.ctx
            self.guest_ctx_source = "derived by derive_guest_cache (first-block rbp symbolic)"
        else:
            self.guest_ctx_source = "none (no concrete rbp sampled)"
        if self.guest_ctx is not None and self.guest_census:
            L.enable_guest_census(self.guest_ctx, self.handler_cache)
        return arena, machine, L

    # --------------------------------------------------------------- identity

    def identity(self, L: Lifter) -> int:
        """Hash of the VM's own pointers in the context window.

        The vip slot enters as a value when it is concrete and as a structural
        hash when it is not, so a program point reached with a symbolic vip
        still gets a stable identity. Every other slot enters only when it holds
        a concrete in-`.tvm0` pointer. Guest data is deliberately excluded: an
        identity that moved with guest data would re-recover a block once per
        guest value, which is precisely the explosion this avoids.
        """
        a = L.a
        rbp = a.as_const(L.m.reg("rbp", 64))
        if rbp is None:
            return 0
        h = 0xCBF29CE484222325
        for off in range(0, CONTEXT_WINDOW, 8):
            v = L.m.load(a.const(rbp + off, 64), 8, 0, L.img.image_u8)
            c = a.as_const(v)
            if off == self.vip_slot:
                # The vip is part of the identity whether or not it is known.
                mix = c if c is not None else a.structural_hash(v)
                h ^= (off * 0x9E3779B97F4A7C15) ^ mix
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
                continue
            if c is None or not (self.vm_lo <= c < self.vm_hi):
                continue
            if c == self.unwind_slot:
                continue
            h ^= (off * 0x9E3779B97F4A7C15) ^ c
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h or 1

    # ------------------------------------------------------- guest cache

    def derive_guest_cache(self, root: int, handlers: int = 400,
                           steps: int = 400_000) -> GuestCache:
        """Measure which host registers cache which guest registers, per function.

        The census derives the *materialisation layout* for free, and
        `E-GUEST-CACHE-AND-MATERIALISATION.md` shows that map is **not** the one a
        handler reads: on `0x14015fc2e` it says `rsi <- guest rsi` and the handler
        actually reads guest `rcx` out of `rsi`. The handler-time map cannot be
        inferred from the symbolic state -- it is a runtime property of the
        permutation the protector chose for this function -- so it is measured.

        The method is the one `probe/guestcache.py` validated:

        1. walk the root until the VM has filled the context (the census reports
           the writes, so this needs no extra instrumentation);
        2. overwrite the seventeen slots with fresh symbols, **except `rsp`**;
        3. at each dispatch boundary, any host register holding exactly one of
           those symbols, as a bare symbol, is caching that guest register.

        `rsp` is excluded because seeding it stops the walk: measured, 7104
        instructions against 28151, with `no_concrete_rsp`. The VM derives its own
        stack pointer from the guest `rsp` slot, so seeding it does not measure the
        VM, it halts it.

        The result is a *different* kind of statement from `Lifter.guest_cache`,
        and the two are labelled: this returns a `GuestCache` whose `source` is
        `HANDLER_TIME`. Ambiguous registers are kept as staging rather than
        resolved by majority.

        This is a second bounded walk and it is **not** run by `recover`, so it
        costs nothing to a caller that does not want it.
        """
        arena, machine = self._initial_state(root)
        L = self._lifter(arena, machine)
        self._install_load_folder(arena, L)

        ctx: GuestContext | None = None
        symbols: dict[int, str] = {}
        cache = GuestCache(HANDLER_TIME)
        seeded = False
        ip, at, seen, boundaries, stop = root, 0, 0, 0, None
        handler = 0
        # Host-register writes whose new value mentions a seeded symbol. Collected
        # over the whole seeded walk and interpreted against the finished cache,
        # because the cache is not known until the walk ends.
        #
        # A *boundary* snapshot only sees what survived to the boundary, and that is
        # why an intermediate result looks invisible: it lives in a scratch register
        # for a few instructions and is written back before the next dispatch. So
        # the register file is differenced at every instruction instead -- fifteen
        # comparisons a step, and the log becomes complete rather than sampled.
        prev = {r: machine.reg(r, 64) for r in HOST_REGS}
        ops: list[tuple[int, str, int, int]] = []
        # How many host registers carry a seeded symbol at each boundary, in order.
        # The seeding is **one-shot**: symbols are written into the context once and
        # the guest file keeps evolving afterwards. Any slot recomputed from
        # non-symbolic data loses its marker permanently, so if this decays the map
        # below is a picture of the walk's opening rather than of the function -- and
        # the "conditional" counts it reports would be measuring marker loss rather
        # than the VM's behaviour.
        carry_counts: list[int] = []
        # Which host registers *ever* carry a seeded symbol, anywhere in the walk --
        # not only at dispatch boundaries. This is the measurement round 12 said was
        # needed: the cache map samples at boundaries, so a register carrying a guest
        # value strictly inside a block is invisible to it **by construction**, and
        # whether that blindness is small or is most of the walk is the difference
        # between "~138 blocks do not touch guest state" and "the map is a projection".
        #
        # Recorded when a register's value *changes* to something carrying a symbol,
        # which is both cheaper than re-testing every register every step and the
        # semantically right question: when does a register acquire a guest value.
        carriers: set[str] = set()
        bare_carries: dict[str, int] = {}
        derived_carries: dict[str, int] = {}
        # And how long each bare acquisition is *held*. "Where a guest register lives"
        # is a claim about persistence -- a value that survives a dispatch is cached, a
        # value that appears and is overwritten inside one block is being moved. The
        # boundary map answers that question only at boundaries; this answers it for
        # every acquisition, and it is what decides whether the five registers the map
        # never names are staging (short holds, none reaching a dispatch) or a genuine
        # omission (holds comparable to the named ones, spanning dispatches).
        hold: dict[str, tuple[int, str, int]] = {}
        hold_stats: dict[str, list[tuple[int, int]]] = {}

        while at < steps and boundaries < handlers:
            kind, payload = L.step(ip)
            at += 1
            cur = {r: machine.reg(r, 64) for r in HOST_REGS}
            if seeded:
                for r in HOST_REGS:
                    if cur[r] == prev[r]:
                        continue
                    v = cur[r]
                    if arena.op(v) == "opaque" and v in symbols:
                        carriers.add(r)
                        bare_carries[r] = bare_carries.get(r, 0) + 1
                        hold[r] = (at, symbols[v], seen)
                    elif any(n in symbols for n in arena.leaves(v)):
                        carriers.add(r)
                        derived_carries[r] = derived_carries.get(r, 0) + 1
                        if r in hold:
                            start, _g, d0 = hold.pop(r)
                            hold_stats.setdefault(r, []).append(
                                (at - start, seen - d0))
                    elif r in hold:
                        start, _g, d0 = hold.pop(r)
                        hold_stats.setdefault(r, []).append((at - start, seen - d0))
                    if arena.op(v) == "opaque" and v in symbols:
                        ops.append((handler, r, prev[r], v))
                    elif any(n in symbols for n in arena.leaves(v)):
                        ops.append((handler, r, prev[r], v))
            prev = cur
            if ctx is None:
                b = arena.as_const(machine.reg("rbp", 64))
                if b is not None:
                    ctx = GuestContext(b + GUEST_CTX_FROM_ENTRY_RBP)
                    L.enable_guest_census(ctx)
            if kind != "next":
                stop = payload
                break
            ip = payload
            if len(L.dispatches) <= seen:
                continue
            seen = len(L.dispatches)
            handler = L.dispatches[-1][1]
            if not seeded:
                if ctx is not None and len({r for _s, r, _v
                                            in L.guest_writes}) >= 14:
                    seeded = True
                    for i, reg in enumerate(ARCH):
                        if reg == "rsp":
                            continue
                        s = arena.opaque(f"H_{reg}", W64)
                        symbols[s] = reg
                        machine.store(arena.const(ctx.base + 8 * i, W64),
                                      s, 8, SEED_SITE)
                    s = arena.opaque(f"H_{FLAGS}", W64)
                    symbols[s] = FLAGS
                    machine.store(arena.const(ctx.addr_of(FLAGS), W64),
                                  s, 8, SEED_SITE)
                    prev = {r: machine.reg(r, 64) for r in HOST_REGS}
                continue
            boundaries += 1
            n_carry = 0
            for r in HOST_REGS:
                v = machine.reg(r, 64)
                if arena.op(v) != "opaque":
                    continue
                if v in symbols:
                    cache.note(r, symbols[v])
                    n_carry += 1
            carry_counts.append(n_carry)
        self.handler_cache = cache if seeded else None
        if self.handler_cache is not None:
            self.handler_cache.boundaries = boundaries
        self.guest_ops = ops
        self._op_arena = arena
        self.handler_cache_walk = {"steps": at, "boundaries": boundaries,
                                   "seeded": seeded,
                                   "stop": stop.kind if stop else "budget",
                                   "carry_timeline": carry_counts,
                                   "carriers": sorted(carriers),
                                   "bare_carries": dict(sorted(
                                       bare_carries.items(),
                                       key=lambda kv: -kv[1])),
                                   "derived_carries": dict(sorted(
                                       derived_carries.items(),
                                       key=lambda kv: -kv[1])),
                                   "hold_stats": {r: v for r, v in
                                                  sorted(hold_stats.items())}}
        # Carry the frame this walk found, so a recovery whose own first-block sample was
        # symbolic can use it instead of silently running with no census at all.
        cache.ctx = ctx
        cache.walk_status = {
            'seeded': seeded, 'steps': at, 'boundaries': boundaries,
            'step_budget': steps, 'handler_budget': handlers,
            'stop': stop.kind if stop is not None else ('handler_limit' if boundaries >= handlers else 'step_limit'),
        }
        return cache

    def guest_operations(self, limit: int = 0) -> list[tuple[int, str, str]]:
        """The seeded walk's register writes, expressed as guest operations.

        Each host register write whose value mentions a seeded guest symbol is
        mapped back through `handler_cache`, so `rsi <- H_rcx + 8` is reported as
        `guest rcx <- rcx + 8`. Only handlers on the walked path appear, and a host
        register the cache calls *staging* is reported as such rather than being
        given a guest name it does not have.
        """
        cache = self.handler_cache
        if cache is None or not self.guest_ops or self._op_arena is None:
            return []
        ops = self.guest_ops[:limit] if limit else self.guest_ops
        out: list[tuple[int, str, str]] = []
        for handler, host, _before, after in ops:
            guest = cache.guest_of(host)
            name = f"guest {guest}" if guest else f"staging {host}"
            out.append((handler, name,
                        render(self._op_arena, after, limit=24).replace("H_", "")))
        return out

    def canonical_operations(self, samples: int = 64, limit: int = 0
                             ) -> list[tuple[int, str, str, int]]:
        """Guest register writes identified as *guest operations*.

        For each recovered value the leaves that actually influence it are measured
        (`core.canon.influence_set`) and, when exactly one leaf does, the value is
        identified as a simple function of it by building candidates and evaluating
        them against the original on every sample.

        The leaf is **not required to be a guest register**, and that is the point.
        Measured on `0x14015fc2e`, the biggest expressions in the run are not
        functions of guest state at all: 36 values with a median DAG size of 358 are
        **constants**, and 82 with a median of 422 are functions of a single
        non-guest leaf -- a call return or a flag. Restricting the search to guest
        registers identified only the small ones (median DAG size 2 of 19 found) and
        left every large expression unexplained.

        Returns `(handler, guest register, form, samples checked)`, where the form
        names its own leaf: `~guest rcx`, `call_ret#100 & 0xffffffff`, `= const`. A
        form of `None` means the value depends on several leaves and is reported as
        unidentified rather than approximated. An empty influence set is confirmed
        as constant by re-randomising the whole environment, not from one base.
        """
        arena = self._op_arena
        cache = self.handler_cache
        if arena is None or cache is None or not self.guest_ops:
            return []
        ops = self.guest_ops[:limit] if limit else self.guest_ops
        out: list[tuple[int, str, str, int]] = []
        for handler, host, _before, value in ops:
            if arena.op(value) == "opaque":
                continue
            inf = influence_set(arena, value, MUTATIONS, 0x5EED_0001)
            if inf is None:
                continue
            guest = cache.guest_of(host) or host
            if not inf:
                out.append((handler, guest, "= const", 16))
                continue
            if len(inf) != 1:
                out.append((handler, guest, None, len(inf)))
                continue
            leaf = next(iter(inf))
            r = random.Random(0xC0FFEE ^ value)
            env = {lf: r.getrandbits(arena.width(lf))
                   for lf in arena.leaves(value)}
            _node, label, checked = identify(arena, value, leaf, env,
                                             samples=samples)
            if label is None:
                out.append((handler, guest, None, 0))
                continue
            out.append((handler, guest, f"{self._leaf_name(arena, leaf, cache)} "
                                        f"-> {label}", checked))
        return out

    @staticmethod
    def _leaf_name(arena: Arena, leaf: int, cache: GuestCache) -> str:
        """How a leaf is written in an identified form."""
        n = arena.node(leaf)
        if n[0] == "opaque":
            if n[1].startswith("H_"):
                return f"guest {n[1][2:]}"
            return n[1]
        if n[0] == "load":
            addr = arena.as_const(n[1])
            return f"load[{addr:#x}]" if addr is not None else "load[?]"
        return n[0]

    def _executable(self, va: int) -> bool:
        sec = self.img.section_for_va(va)
        return sec is not None and sec.executable

    def _scratch_window(self, L: Lifter) -> tuple[int, int] | None:
        """The VM's own scratch area: its frame, above the register image.

        The register image (`[rbp+8 .. rbp+0x88)`) is deliberately excluded: the
        vip lives in it, and a symbolic vip would make block identity path
        dependent. Everything above it is the VM's private working area, and its
        bytes are exactly what cannot be assumed to be fixed.
        """
        rbp = L.a.as_const(L.m.reg("rbp", 64))
        if rbp is None:
            return None
        return (rbp + 0x90, rbp + 0xB28)

    def _brief(self, L: Lifter, stop: Stop) -> str:
        target = L.a.as_const(stop.dest) if stop.dest is not None else None
        return (f"{stop.kind} at {stop.site:#x}"
                + (f" -> {target:#x}" if target is not None else ""))

    # ------------------------------------------------------------- resolution

    def symbolic_leaves(self, a: Arena, r: int, limit: int = 8) -> list[int]:
        """Leaves of `r` that are genuinely unknown: loads, opaques, parameters.

        Entry registers are deliberately excluded. They are unknown too, but they
        are unknown *per function*, and enumerating one of them would enumerate
        the function's arguments rather than a dispatch index.
        """
        out: list[int] = []
        seen: set[int] = set()
        stack = [r]
        while stack and len(out) < limit:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            n = a.nodes[cur]
            t = n[0]
            if t in ("load", "opaque", "param"):
                out.append(cur)
            elif t == "init":
                continue
            elif t == "bin":
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
        return out[:limit]

    def unique_dispatch(self, L: Lifter, dest: int, pins: dict[int, int], samples: int = 8):
        """Resolve a dispatch that is symbolic yet has only one possible value.

        The case this exists for: the address is built from a value the VM read
        and a status flag, so it is symbolic -- but every value either of them can
        take leads to the *same* address. Nothing to pin, nothing to enumerate,
        and the correct answer is an unconditional jump. A resolver that only
        accepts two-way splits rejects it and reports "unresolved", which is what
        this engine did for five rounds on every function in the module.

        **This rule was unsound when first written, and the check that caught it
        was external.** It sampled a spread of values per leaf and accepted a
        single target if every sample agreed. On one site the reference resolves
        fourteen different destinations, and the sampled spread landed on the
        same one every time -- because the value that actually discriminates the
        address was not in the spread. The CFG then claimed an edge the reference
        contradicts, and 46 of 68 checkable edges were wrong.

        What makes it sound enough to keep is that the *whole* domain of the
        discriminating leaf is now enumerated, not sampled: all 256 values for
        each candidate leaf, with the others driven over a spread of their own.
        A leaf whose value moves the address can no longer hide between samples.
        What remains unproven is a dependence that no sampled combination of the
        *other* leaves reaches, and that residue is why these edges are recorded
        as hypotheses and held to account by the contradiction test.
        """
        a = L.a
        base = a.substitute(dest, pins) if pins else dest
        if a.as_const(base) is not None:
            return None
        leaves = self.symbolic_leaves(a, base, limit=6)
        if not leaves:
            return None
        spread = (0, 1, 2, 3, 0x7F, 0x80, 0xFF, 0x100, 0x10000, 0x7FFFFFFF, 0xFFFFFFFF,
                  0xFFFF, 0x5555, 0xAAAA)

        def fold_with(values: dict[int, int]):
            """The address with `values` pinned, as a concrete number or None.

            Answered by direct evaluation, falling back to a real substitution
            only where the evaluator says it cannot be sure. The domains below
            drive each leaf through 256 values, so this loop runs hundreds of
            thousands of times per site and each iteration wants a *number*, not
            a rebuilt expression. (On the two-per-candidate path this same swap
            was a loss; here the trial count is three orders of magnitude larger
            and it pays. Both measurements are in the record.)
            """
            trial = dict(pins)
            env: dict[int, int] = {}
            for lf, v in values.items():
                w = a.width(lf)
                env[lf] = v & ((1 << w) - 1)
            got, exact = a.eval_exact(dest, env)
            if not exact:
                for lf, v in env.items():
                    trial[lf] = a.const(v, a.width(lf))
                return a.as_const(a.substitute(dest, trial))
            return got

        others = leaves[1:4]
        # Every combination of the *other* leaves over their spread, so that a
        # value that only shows up when two unknowns move together is still seen.
        combos: list[dict[int, int]] = [{}]
        for lf in others:
            nxt = []
            for row in combos:
                for v in spread[:6]:
                    r = dict(row)
                    r[lf] = v
                    nxt.append(r)
                if len(nxt) > 512:
                    break
            combos = nxt[:512]

        for first in leaves[:3]:
            w = a.width(first)
            if w < 8:
                continue
            got = None
            ok = True
            for base_row in combos:
                for v in range(256):
                    row = dict(base_row)
                    row[first] = v
                    folded = fold_with(row)
                    if folded is None:
                        ok = False
                        break
                    if got is None:
                        got = folded
                    elif folded != got:
                        ok = False
                        break
                if not ok:
                    break
            # A high-bit dependence of the enumerated leaf has to be ruled out as
            # well; the falsifier for that is the same one enumeration uses.
            if ok:
                for v in (0x100, 0x1000, 0x10000, 0x7FFFFFFF, 0xFFFFFFFF):
                    if v >= (1 << w):
                        continue
                    for base_row in combos[:8]:
                        row = dict(base_row)
                        row[first] = v
                        folded = fold_with(row)
                        if folded is None or folded != got:
                            ok = False
                            break
                    if not ok:
                        break
            if ok and got is not None and self._executable(got):
                return got
        return None

    def enumerate_dispatch(self, L: Lifter, dest: int, pins: dict[int, int],
                           bound: int = 256, max_combos: int = 8192):
        """Resolve a dispatch by enumerating the narrow values it depends on.

        The third device, and the one the reference calls `narrow_indices`. An
        address of the form `base + f(table[g(x)])` has no boolean discriminant
        to pin -- `g(x)` is a whole byte, so nothing folds -- but the byte has
        only 256 values and every one of them can simply be tried.

        **All the unknown values must be enumerated together.** Substituting one
        leaf and leaving another free leaves the address symbolic, so the fold
        returns nothing and the rule silently never fires -- which is exactly
        what it did until this was noticed: the dispatch depended on a call
        return *and* on a flag, and the rule was varying only the first.

        Leaf domains differ in width: a dispatch index is a byte, and a status
        flag is a bit. The first leaf gets [0, 256) and the others {0, 1}, with
        the product capped. That restriction is a guess about the excluded
        values, so it is *checked*: values outside each domain are sampled
        afterwards and the whole result is discarded if any of them produces an
        executable destination that the enumeration did not already find.
        Without that check the rule would fabricate arms, which is worse than an
        unresolved block.
        """
        a = L.a
        base_dest = a.substitute(dest, pins) if pins else dest
        if a.as_const(base_dest) is not None:
            return None

        leaves = [lf for lf in self.symbolic_leaves(a, base_dest) if a.width(lf) >= 8]
        if not leaves:
            return None

        for first in leaves[:3]:
            others = [lf for lf in leaves if lf != first][:3]
            combos: list[list[tuple[int, int]]] = [[(first, v)] for v in range(bound)]
            for lf in others:
                nxt: list[list[tuple[int, int]]] = []
                for row in combos:
                    for v in (0, 1):
                        nxt.append(row + [(lf, v)])
                    if len(nxt) > max_combos:
                        break
                combos = nxt[:max_combos]
            if len(combos) * (len(others) + 1) > max_combos * 2:
                continue

            found: dict[int, int] = {}
            all_fold = True
            for row in combos:
                folded = self._fold_row(a, dest, pins, row)
                if folded is None:
                    all_fold = False
                    break
                if self._executable(folded):
                    found[folded] = folded
            if not all_fold or len(found) < 2:
                continue

            # Falsifier: values outside the domains tried must not reach code the
            # enumeration missed.
            samples: list[list[tuple[int, int]]] = []
            for v in (0x100, 0x180, 0x10000, 0x7FFFFFFF, 0xFFFFFFFF):
                if v < (1 << a.width(first)):
                    samples.append([(first, v)] + [(lf, 0) for lf in others])
                    samples.append([(first, v)] + [(lf, 1) for lf in others])
            for lf in others:
                for v in (2, 3, 0x100, 0x101, 0xFF):
                    if v < (1 << a.width(lf)):
                        samples.append([(first, 0), (lf, v)]
                                       + [(o, 0) for o in others if o != lf])
                        samples.append([(first, 1), (lf, v)]
                                       + [(o, 0) for o in others if o != lf])
            ok = True
            for row in samples:
                folded = self._fold_row(a, dest, pins, row)
                if folded is None:
                    ok = False
                    break
                if self._executable(folded) and folded not in found:
                    ok = False
                    break
            if ok:
                return sorted(found), first
        return None

    def _fold_row(self, a: Arena, dest: int, pins: dict[int, int],
                  row) -> int | None:
        """The dispatch address with one assignment of its leaves, or None.

        Answered by direct evaluation, falling back to a real substitution only
        where the evaluator says it cannot be sure. `enumerate_dispatch` drives a
        leaf through its whole 256-value domain and does so for several leaves and
        several combinations, so this runs hundreds of thousands of times per
        site and each iteration wants a *number*, not a rebuilt expression.

        Swapping the substitution for the evaluator on the two-per-candidate path
        in the lifter was measured to be a **loss** (31.7 s against 25.3 s on a
        real function); here the trial count is three orders of magnitude larger
        and the same swap pays. Same code, opposite verdicts, which is why both
        measurements are kept.
        """
        env: dict[int, int] = {}
        for lf, v in row:
            w = a.width(lf)
            env[lf] = v & ((1 << w) - 1)
        got, exact = a.eval_exact(dest, env)
        if exact:
            return got
        trial = dict(pins)
        for lf, v in env.items():
            trial[lf] = a.const(v, a.width(lf))
        return a.as_const(a.substitute(dest, trial))

    def resolve_dispatch(self, L: Lifter, dest: int, pins: dict[int, int]):
        """Where an indirect branch can go.

        Returns (targets, pins, predicate). `targets` are concrete executable
        destinations; `pins` is the assertion to carry forward; `predicate` is
        the guest value that decides between two targets, or None when the
        address simply folded.
        """
        a = L.a
        self._last_dispatch_proven = False
        self._last_dispatch_infeasible = False
        dest = a.substitute(dest, pins) if pins else dest
        c = a.as_const(dest)
        if c is not None:
            return ([c], dict(pins), None, False) if self._executable(c) else ([], dict(pins), None, False)

        for cand in boolean_subexpressions(a, dest):
            # Both sides must *fold*. A pin that leaves the other side symbolic
            # has not determined the address, and taking the one side that
            # happened to land on code is a guess -- the wrong guess, in fact: on
            # a site where the reference resolves fourteen destinations, pinning
            # one bit of the index left the rest of it free, one side folded onto
            # code and the other did not fold at all, and the engine claimed a
            # single unconditional edge. The contradiction test found 46 of them.
            folded_sides: dict[int, int | None] = {}
            exec_sides: dict[int, int] = {}
            for val in (0, 1):
                trial = dict(pins)
                trial[cand] = a.const(val, a.width(cand))
                folded = a.as_const(a.substitute(dest, trial))
                folded_sides[val] = folded
                if folded is not None and self._executable(folded):
                    exec_sides[val] = folded
            if any(v is None for v in folded_sides.values()):
                continue
            if len(exec_sides) == 2 and len(set(exec_sides.values())) == 2:
                return (list(exec_sides.values()), dict(pins), cand, False)
            if len(exec_sides) == 1:
                val, tgt = next(iter(exec_sides.items()))
                np = dict(pins)
                np[cand] = a.const(val, a.width(cand))
                return ([tgt], np, None, False)

        # Solve the full finite target image before guessing narrow leaf domains.
        # The memory tables and all target alternatives must be exhaustive.
        if self.proof_dispatch:
            from .bitproof import finite_values
            proof = finite_values(a, dest, pins)
            self.proof_stats[proof.status] = self.proof_stats.get(proof.status, 0) + 1
            # **A timeout is not a verdict, and neither is the table cap.** The narrow call runs with
            # a 350 ms deadline and a 64-address table because it is on the path of *every* dispatch;
            # what it cannot answer, a slower call sometimes can, because materialising a memory
            # table needs the load's address image inside `table_limit`. Measured on `ACE-CORE.sys3`:
            # five blocks of `0x145bccd2d` come back `too_many` at 64 addresses and `complete` with
            # **one** executable target at 256 (`probe/block_exits.py --prove=64`). So the narrow
            # answer is retried once with the wider budget before the block is called unresolved, and
            # both attempts are counted separately so the cost is visible.
            if proof.status in ('timeout', 'too_many', 'node_limit'):
                proof = finite_values(a, dest, pins, table_limit=WIDE_TABLE_LIMIT,
                                     timeout_ms=WIDE_TIMEOUT_MS)
                self.proof_stats['wide:' + proof.status] = \
                    self.proof_stats.get('wide:' + proof.status, 0) + 1
            if proof.status == 'complete' and all(self._executable(t) for t in proof.values):
                self._last_dispatch_proven = True
                self.proof_stats['accepted'] = self.proof_stats.get('accepted', 0) + 1
                self.proof_stats['targets'] = self.proof_stats.get('targets', 0) + len(proof.values)
                return (list(proof.values), dict(pins), None, False)
            if proof.status == 'infeasible':
                self._last_dispatch_infeasible = True
                return ([], dict(pins), None, False)

        # No provably-boolean discriminant: fall back to naming the bit the VM
        # must be testing, and keep the hypothesis only if it lands on code.
        for leaf, _m, bit, rewritten in hypothesis_candidates(a, dest):
            sub = dict(pins)
            sub[leaf] = rewritten
            trial_dest = a.substitute(dest, sub)
            if a.as_const(trial_dest) is not None or trial_dest == dest:
                continue
            got = {}
            for val in (0, 1):
                t2 = dict(sub)
                t2[bit] = a.const(val, a.width(bit))
                folded = a.as_const(a.substitute(dest, t2))
                got[val] = folded
            if any(v is None for v in got.values()):
                continue
            ex = {k: v for k, v in got.items() if self._executable(v)}
            if len(ex) == 2 and len(set(ex.values())) == 2:
                np = dict(sub)
                np[bit] = a.const(0, a.width(bit))
                return (list(ex.values()), np, bit, True)
            if len(ex) == 1:
                val, tgt = next(iter(ex.items()))
                np = dict(sub)
                np[bit] = a.const(val, a.width(bit))
                return ([tgt], np, None, True)

        # No discriminant to pin. Before enumerating, ask the cheaper question:
        # does the address have only one possible value at all?
        unique = self.unique_dispatch(L, dest, pins)
        if unique is not None:
            return ([unique], dict(pins), None, True)

        # Last resort: a narrow index with no boolean to pin. Enumerate it.
        enumerated = self.enumerate_dispatch(L, dest, pins)
        if enumerated is not None:
            targets, leaf = enumerated
            return (targets, dict(pins), leaf, True)
        return ([], dict(pins), None, False)

    # ------------------------------------------------------------------- run

    def recover(self, root: int) -> Cfg:
        arena, machine, L = self.prepare(root)
        cfg = Cfg()
        start = BlockKey(root, self.identity(L))
        cfg.entry = start
        journal = None
        if self.journal:
            journal = cfgio.Journal(
                self.journal, getattr(self.img, "path", "?"), root, cfg,
                {"step_budget": self.step_budget, "block_budget": self.block_budget})

        work: list[tuple[MachineSnapshot, int, BlockKey, dict[int, int]]] = [
            (MachineSnapshot(machine, lifter=L), root, start, {})
        ]
        queued: set[BlockKey] = {start}

        while work:
            if len(cfg.blocks) >= self.block_budget or len(cfg.blocks) >= self.max_forks:
                break
            snap, resume, key, pins = work.pop()
            if key in cfg.blocks:
                continue
            snap.restore(machine, L)
            L.events = []
            L.dispatches.clear()
            L.boxed.clear()

            block = Block(key=key)
            if self.guest_census:
                block.guest_entry = observe_guest_state(machine, self.guest_ctx, self.img.image_u8)
                block.memory_entry = machine.mem_version
            cfg.blocks[key] = block
            cfg.order.append(key)

            ip = resume
            stop: Stop | None = None
            steps_here = 0
            gw_mark = len(L.guest_writes)
            grw_mark = len(L.guest_reg_writes)
            gd_mark = len(L.guest_derived)
            gl_mark = len(L.guest_loads)
            gt_mark = len(L.guest_trace)
            effect_mark = len(L.local_effects)
            # Provenance is restored from this path's snapshot. Cache frequency
            # is not proof that a host register holds its guest at this point.
            attempt_start = LifterCheckpoint(L)
            loop_period = 0
            checked = -1
            candidate = None       # (period, dispatch count when seen, identity then)
            for _ in range(self.step_budget):
                kind, payload = L.step(ip)
                steps_here += 1
                if kind == "next":
                    ip = payload
                    n = len(L.dispatches)
                    if candidate is not None and n >= candidate[1] + candidate[0]:
                        # **The path is not enough; the state has to come back too.** A repeating
                        # transfer tail is also what a virtualized loop *inside* a handler looks
                        # like, which is how a path-only rule cut `0x1401505d5` from 8 blocks to 3
                        # (A32). `identity` is this engine's own answer to "same point, same state" --
                        # the hash of the VM's pointers in the context window that a BlockKey is
                        # built from -- so requiring it to return after exactly one period is the
                        # same test the explorer already applies to decide two blocks are one.
                        if self.identity(L) == candidate[2]:
                            loop_period = candidate[0]
                            break
                        candidate = None
                    elif (self.loop_cutpoint_default and n != checked
                            and n % self.LOOP_CHECK_EVERY == 0):
                        checked = n
                        period = self._repeating_dispatch_period(L)
                        if period:
                            candidate = (period, n, self.identity(L))
                    continue
                stop = payload
                break
            block.loop_period = loop_period

            # Validity-checked precision. A dispatch that folded to a concrete
            # address which is not executable is a proof that some concrete
            # value it used is not actually fixed -- most often a byte read out
            # of the VM's own scratch area. Rather than weaken the precise pass
            # globally, which would cost accuracy everywhere to fix one site,
            # the block is re-run once with that scratch treated as unknowable,
            # and the retry is used only if it yields a real control transfer.
            retry_note = ""
            if stop is not None and stop.kind == "bad_target":
                win = self._scratch_window(L)
                if win is not None:
                    retry_note = f"precise pass reached {self._brief(L, stop)}"
                    precise_state = LifterCheckpoint(L)
                    attempt_start.restore(L)
                    # The two attempts are alternative interpretations, not two
                    # executions. Only the selected attempt contributes effects.
                    L.events = []
                    L.boxed.clear()
                    # The precise pass's transfers describe a path this block is
                    # not going to take any more, so they must not be mixed with
                    # the retry's.
                    L.dispatches.clear()
                    machine.symbolic_window = win
                    try:
                        ip2, stop2, steps2 = resume, None, 0
                        for _ in range(self.step_budget):
                            kind, payload = L.step(ip2)
                            steps2 += 1
                            if kind == "next":
                                ip2 = payload
                            else:
                                stop2 = payload
                                break
                    finally:
                        machine.symbolic_window = None
                    if stop2 is not None and stop2.kind in ("dispatch", "native_branch", "dispatch_call"):
                        stop, steps_here = stop2, steps2
                        retry_note += "; scratch made symbolic, control recovered"
                    else:
                        precise_state.restore(L)
                        retry_note += f"; scratch made symbolic, still {stop2.kind if stop2 else 'budget'}"

            cfg.steps += steps_here
            block.cost = steps_here
            block.events = len(L.events)
            block.retry = retry_note
            block.transfers = list(L.dispatches)
            block.calls = [(e["site"], call_target(L, e)) for e in L.events
                           if e.get("kind") == "call"]
            block.guest_writes = [(reg, value)
                                  for _site, reg, value in L.guest_writes[gw_mark:]]
            block.guest_reg_writes = [(guest, host, value)
                                      for _step, guest, host, value
                                      in L.guest_reg_writes[grw_mark:]]
            block.guest_derived = [(guest, host, prev, value)
                                   for _step, guest, host, prev, value
                                   in L.guest_derived[gd_mark:]]
            block.guest_loads = [(site, addr) for site, addr in L.guest_loads[gl_mark:]]
            block.guest_trace = L.guest_trace[gt_mark:]
            block.local_effects = dict(L.local_effects[effect_mark:])
            block.guest_values = dict(L.guest_values)
            block.guest_identity = dict(L.guest_identity)
            if self.guest_census:
                block.guest_exit = observe_guest_state(machine, self.guest_ctx, self.img.image_u8)
                block.memory_exit = machine.mem_version
                if stop is not None and stop.kind == 'return':
                    # At an observed native return the host architectural state
                    # is the externally visible result. It may differ from the
                    # last backing-slot values after the VM restores registers.
                    block.native_return = {r: machine.reg(r, 64) for r in ARCH}
                    block.native_return_flags = dict(machine.flags)
                    block.native_return_phase = ('before-ret-symbolic-rsp' if machine.concrete_rsp() is None
                                                 else 'after-lifted-ret-stack-adjustment')
            if stop is not None:
                block.site = stop.site

            succs = self._terminate(L, stop, key, block, pins, cfg)
            for nk, nsnap, npins in succs:
                block.succs.append(nk)
                if self.guest_census:
                    block.guest_edges.setdefault(nk, []).append(dict(
                        values=observe_guest_state(machine, self.guest_ctx, self.img.image_u8, nsnap),
                        pins=dict(npins), termination=stop))
                if nk in cfg.blocks:
                    cfg.blocks[nk].preds.append(key)
                elif nk not in queued:
                    queued.add(nk)
                    work.append((nsnap, nk.handler, nk, npins))
            if len(work) > cfg.work_peak:
                cfg.work_peak = len(work)
                # Snapshots share nothing, so the pending cost is the sum of
                # their entry counts; this is the number that decides whether the
                # *search* is what grows.
                cfg.snapshot_cells_peak = sum(s.cells for s, _, _, _ in work)

            # Persist the block only now, when its successors are known, so a
            # record on disk is a complete statement about that block. See
            # `cfgio`: an interrupted run keeps every block it finished, which is
            # the whole point -- recovery of a real function is tens of seconds.
            if journal is not None:
                journal.block(block)

        for b in cfg.blocks.values():
            b.succs = list(dict.fromkeys(b.succs))
            b.preds = list(dict.fromkeys(b.preds))
        cfg.summary_ok = L.summary_ok
        cfg.summary_steps = L.summary_steps
        cfg.summary_rejects = dict(L.summary_rejects)
        cfg.arena_nodes = len(arena)
        cfg.transfers = sum(len(b.transfers) for b in cfg.blocks.values())
        # A run stopped by its own budget did not finish the function, and the
        # file says so rather than looking like a complete graph.
        self.journal_truncated = bool(work) or len(cfg.blocks) >= self.block_budget
        # **A truncated search must say so, and it did not.**
        #
        # A block whose lifting loop exhausts `step_budget` leaves `stop` as None, is
        # recorded as `unresolved`, and the run then looks entirely complete: a full CFG, a
        # full corpus, and `COMPLETE` from the reference comparison. Round 37 found that
        # every probe in this tree defaults to 200000 steps while `0x1401505d5` needs 309036,
        # so a large fraction of the figures published for that root over eight rounds came
        # from a partial search of 11 blocks out of 89 -- and the check that would have caught
        # it, *did any block hit the budget*, was available in every one of those runs and
        # never asked. This exposes it so a probe can print it.
        self.budget_stopped = [b.key for b in cfg.blocks.values()
                               if b.kind == 'unresolved'
                               and b.detail.startswith('step budget reached')]
        self.truncated = bool(self.budget_stopped) or self.journal_truncated
        if journal is not None:
            journal.finish(cfg, self.truncated)
        # Materialise observation DAGs only after all control-flow decisions.
        # Search heuristics use native node IDs as tie-breakers, so even pure
        # observation during search could otherwise alter candidate selection.
        if self.guest_census:
            from copy import copy
            for observed in cfg.blocks.values():
                observed.guest_entry = observed.guest_entry.materialize()
                observed.guest_exit = observed.guest_exit.materialize()
                for target, variants in observed.guest_edges.items():
                    for edge in variants:
                        edge['pins'] = self.guest_edge_pins(L, edge.pop('termination'), target, edge['pins'])
                        edge['values'] = edge['values'].materialize()
                if observed.native_return_flags:
                    view = copy(machine)
                    view.flags = observed.native_return_flags
                    observed.native_return['rflags'] = view.pack_flags()
        return cfg

    @staticmethod
    def guest_edge_pins(lifter, stop, target, pins):
        """Edge validity is independent of whether snapshots are conditioned."""
        guards = dict(pins)
        if stop is not None and stop.kind in ('dispatch', 'dispatch_call'):
            guards[stop.dest] = lifter.a.const(target.handler, lifter.a.width(stop.dest))
        elif stop is not None and stop.kind == 'native_branch' and stop.taken != stop.not_taken:
            guards[stop.pred] = lifter.a.const(int(target.handler == stop.taken), lifter.a.width(stop.pred))
        return guards

    # ---------------------------------------------------- repeating-path cutpoints
    #
    # **A block whose dispatch sequence repeats is finished, not truncated.** `0x140157674` was the
    # root that made this unavoidable: it lifts 400,000 instructions into one block and is then
    # reported as `unresolved` / `truncated` with a step budget reached, because the virtualized
    # loop it is executing never ends inside the budget. What it actually does is settle into an
    # **exact 226-transfer cycle** from boundary 372 (10,965 of 10,965 compared pairs equal) and
    # never leave it -- measured over 520,000 engine steps and 49 different data bytes, so the
    # *path* is data-independent while the value the body writes is not
    # (`evidence/E-LOOP-CUTPOINTS.md`). Lifting further iterations therefore adds no new control
    # flow, and the honest block is a cutpoint that says "repeats every N transfers; no exit
    # observed", with a self-loop as its successor so the transition comparison can follow the
    # repetition rather than the search running out of budget and calling that a hole.
    #
    # Three identical consecutive periods are required, not two: a two-period match is cheap
    # coincidence, and the cost of being wrong here is a missed exit edge.
    LOOP_CHECK_EVERY = 64
    LOOP_MIN_PERIOD = 8
    LOOP_MAX_PERIOD = 512
    #: **Off by default, and the reason is a measurement this file made on itself.** Enabled, the
    #: check finds `0x140157674`'s real 226-transfer cycle and takes that root from a 400,000-step
    #: truncation to a 1,088-transfer cutpoint (36,240 instructions instead of 400,000). It also
    #: fires on `0x1401505d5`, whose dispatch sequence is *locally* 239-periodic while the block is
    #: still going: there it cuts the block from **8 blocks / 317,585 instructions to 3 / 45,843**.
    #: The subagent evidence had already said why -- every *state* fingerprint it tried was
    #: falsified, so path periodicity alone does not mean the state has returned to a cutpoint, and
    #: a virtualized loop nested inside a handler is enough to make the transfer tail repeat. A
    #: criterion that needs the state as well is the version worth having; this one is kept, off,
    #: as the switch that reproduces both numbers.
    loop_cutpoint_default = False
    loop_cutpoints = 0
    loop_cutpoint_blocks: list = []

    def _repeating_dispatch_period(self, L) -> int:
        """Smallest period whose trailing three repetitions are identical, or 0.

        Compares `L.dispatches`, the `(site, target)` transfers the lifter folded while walking this
        block -- the same sequence `probe.corpus_compare` is judged on.
        """
        d = L.dispatches
        n = len(d)
        for period in range(self.LOOP_MIN_PERIOD, self.LOOP_MAX_PERIOD + 1):
            if n < 3 * period:
                return 0
            tail = d[n - period:]
            if tail == d[n - 2 * period:n - period] and tail == d[n - 3 * period:n - 2 * period]:
                return period
        return 0

    def _terminate(self, L, stop: Stop | None, key, block, pins, cfg):
        a = L.a
        out: list[tuple[BlockKey, MachineSnapshot, dict]] = []
        base = MachineSnapshot(L.m, len(L.events), lifter=L)

        def mk(target: int, npins: dict) -> tuple:
            snap = MachineSnapshot(L.m, len(L.events), lifter=L)
            if self.path_sensitive and npins:
                snap.restrict(a, npins)
                try:
                    snap.restore(L.m, L)
                    identity = self.identity(L)
                finally:
                    base.restore(L.m, L)
                self.conditioned_edges += 1
            else:
                identity = self.identity(L)
            return (BlockKey(target, identity), snap, npins)

        if stop is None:
            period = getattr(block, "loop_period", 0)
            if period:
                # See `LOOP_CHECK_EVERY`: the lifting loop ended by *evidence*, not by a budget.
                # The successor is this block itself when the specialized identity returns to the
                # current key -- the identity was measured periodic on the same root -- so the
                # transition comparison can keep following the repetition. When it does not
                # return, no successor is claimed and the renderer prints a terminal `<loop: ...>`
                # arrow instead, because an invented edge would be worse than a stated cutpoint.
                entry = mk(key.handler, {})
                if entry[0] == key:
                    out.append(entry)
                block.kind = "loop"
                block.detail = (f"dispatch sequence repeats every {period} transfer(s); no exit "
                                f"observed in {len(block.transfers)} transfer(s)")
                self.loop_cutpoints += 1
                self.loop_cutpoint_blocks.append((key, period))
                return out
            block.kind = "unresolved"
            # The budget is *named*, so a diagnostic that prints `detail` says which number
            # to raise instead of leaving the reader to find the call site. The prefix is
            # stable because `budget_stopped` filters on it.
            block.detail = f"step budget reached ({self.step_budget})"
            cfg.unresolved += 1
            return out

        if stop.kind in ("dispatch", "dispatch_call"):
            targets, npins, pred, hypothesised = self.resolve_dispatch(L, stop.dest, pins)
            if not targets:
                if self._last_dispatch_infeasible:
                    block.kind = 'infeasible'
                    block.detail = 'path constraints proved unsatisfiable in the lifted model'
                    return out
                block.kind = "unresolved"
                word = "call" if stop.kind == "dispatch_call" else "branch"
                block.detail = f"unresolved {word} at {stop.site:#x}"
                shown = a.substitute(stop.dest, pins) if pins else stop.dest
                block.predicate = render(a, shown)
                # The node is kept as well as its rendering: a diagnostic that wants to re-run the
                # image proof with a wider budget (`probe/block_exits.py`) needs the expression, and
                # the rendering is a rewrite of it, not the thing the resolver failed on.
                block.pred_node = shown
                cfg.unresolved += 1
                return out
            if hypothesised:
                block.hypothesis = True
                cfg.hypotheses += 1
            if len(targets) == 1:
                if self.path_sensitive and self._last_dispatch_proven:
                    npins = dict(npins)
                    npins[stop.dest] = a.const(targets[0], a.width(stop.dest))
                entry = mk(targets[0], npins)
                block.kind = "backedge" if entry[0] in cfg.blocks or entry[0] == key else "jump"
                out.append(entry)
                return out
            block.kind = "branch"
            # **Not truncated any more.** The condition is what a backend has to emit, and the 220
            # character cap -- a display choice from when nothing consumed this -- cut 4 of 4
            # conditions on `0x14016e8ee` (587 to 1220 characters each). A truncated condition is
            # indistinguishable, to a reader, from one that could be transcribed.
            block.predicate = render(a, pred) if pred is not None else ""
            block.pred_node = pred if pred is not None else 0
            cfg.forks += 1
            # **The successor/predicate-value association, recorded rather than discarded.**
            # `resolve_dispatch` returns targets in resolver order; the loop below establishes which
            # value of `pred` produces which target by pinning it both ways and substituting. Until
            # now that answer only reached the edge pins, so nothing downstream could say "successor
            # 1 is the pred=1 side" -- which is the one fact a conditional branch needs.
            assoc: list = []
            for t in targets:
                edge_pins = dict(npins)
                if self.path_sensitive and self._last_dispatch_proven:
                    edge_pins[stop.dest] = a.const(t, a.width(stop.dest))
                value = None
                if self.path_sensitive and pred is not None and a.is_bool(pred):
                    # A two-way resolver returns targets in predicate-value order;
                    # verify the association instead of trusting list ordering.
                    for val in (0, 1):
                        trial = dict(npins)
                        trial[pred] = a.const(val, a.width(pred))
                        if a.as_const(a.substitute(stop.dest, trial)) == t:
                            edge_pins = trial
                            value = val
                            break
                assoc.append(value)
                out.append(mk(t, edge_pins))
            block.edge_cond = assoc
            return out

        if stop.kind == "native_branch":
            block.kind = "branch"
            block.predicate = render(a, stop.pred)
            block.pred_node = stop.pred
            cfg.forks += 1
            # For a native branch the order is fixed by construction -- `taken` is the pred=1 side --
            # so the association is stated rather than searched for.
            assoc: list = []
            for val, t in ((1, stop.taken), (0, stop.not_taken)):
                if self._executable(t):
                    edge_pins = dict(pins)
                    if self.path_sensitive and a.is_bool(stop.pred):
                        existing = pins.get(stop.pred)
                        if existing is not None and a.as_const(existing) != val:
                            continue
                        edge_pins[stop.pred] = a.const(val, a.width(stop.pred))
                    assoc.append(val)
                    out.append(mk(t, edge_pins))
            block.edge_cond = assoc
            if not out:
                block.kind = "unresolved"
                block.detail = f"branch with no executable side at {stop.site:#x}"
                cfg.unresolved += 1
            return out

        if stop.kind in ('noreturn', 'fastfail'):
            from .terminals import from_stop
            block.kind = stop.kind
            block.detail = stop.text or stop.info.get('callee', stop.kind)
            block.terminal = from_stop(stop)
            return out

        if stop.kind == "return":
            block.kind = "return"
            tgt = stop.info.get("target")
            # A stop that names its own reason (e.g. the `int3` guest-exit trap) wins over
            # the generic rendering, so the classification is visible where the block is.
            block.detail = stop.text or (f"ret to {tgt:#x}" if tgt is not None
                                         else "ret to a symbolic destination")
            return out

        if stop.kind == "tail_call":
            block.kind = "tailcall"
            block.detail = f"tail call to {stop.text}"
            return out

        block.kind = "unresolved"
        block.detail = f"{stop.kind} at {stop.site:#x} {stop.text}".strip()
        cfg.unresolved += 1
        return out
