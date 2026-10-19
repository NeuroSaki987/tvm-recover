"""The engine's devirtualisation entry point: module in, guest-level listings out.

Everything needed for this has existed since round 40 -- `Recoverer.derive_guest_cache` derives
the per-function handler-time cache, `Recoverer.recover` builds the CFG, `emit_function` renders
each block's effect in guest register terms -- and none of it had an entry point. The capability
lived in four probes that each rebuilt the same pipeline, so "devirtualise this module" was a
thing a human assembled by hand rather than a thing the engine did.

This is that entry point. It is in `core` rather than in `probe` because it is the product: a
capability that only a probe can exercise is not one the engine has.

    python -m core.devirt <module>                 # every root, listing to stdout
    python -m core.devirt <module> --out DIR       # one file per function
    python -m core.devirt <module> --root 0x14015fc2e --show

Notes that matter for reading the output:

* **Truncation is printed before the listing.** A run stopped by its step budget still produces a
  listing that looks complete, and round 37 exists because that went unnoticed for eight rounds.
* **The listing says what it cannot name.** A block with no guest effect prints `none` with the
  reason rather than being omitted, and a value that was not identified prints an explicit
  `(opaque: ...)` with the reason -- never a raw DAG, and never silence.
* **The listing is the whole function.** Every recovered block appears, including the ones that
  changed nothing, because a reader must be able to tell "no guest effect" from "not analysed".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .explore import DEFAULT_STEP_BUDGET, STACK_BASE, Recoverer
from .guestir import Emitted, emit_function
from .lower import Lowering, split_guest_name
from .peimg import PeImage
from .gueststate import build as build_guest_state

# Steps per block. **Derived, not copied** — `core.explore.DEFAULT_STEP_BUDGET` is the one
# definition, and `tests/test_budget_defaults.py` fails if these drift apart. The value is a
# measurement: at 400,000 the root `0x140169e9d` truncated a block and reported 1 unresolved;
# at 900,000 it reports 17 blocks / 0 unresolved with a clean verifier, and corpus-wide the
# change is 139 -> 150 blocks and 2 -> 1 unresolved with no root worse
# (`evidence/E-BUDGET-OR-LOOP.md`). The one root that still truncates is `0x140157674`, a
# genuine 226-transfer loop that this engine and the reference both give up on at any budget.
# The cost is a measured number; see that document, section 6.1.
# `TVM_STEPS` / `TVM_BLOCKS` in `main/.env` override these defaults for a local checkout; a command
# line flag still wins. Nothing is read from the file when no `.env` exists.
import os as _os


def _budget(name: str, fallback: int) -> int:
    try:
        raw = _os.environ.get(name)
        value = int(raw, 0) if raw else fallback
    except (TypeError, ValueError):
        value = fallback
    return value if value > 0 else fallback


DEFAULT_STEPS = _budget("TVM_STEPS", DEFAULT_STEP_BUDGET)
DEFAULT_BLOCKS = _budget("TVM_BLOCKS", 512)


@dataclass
class Function:
    """One virtualized function, devirtualised."""

    root: int
    blocks: int
    unresolved: int
    instructions: int
    truncated: bool
    emitted: Emitted
    text: str = ""
    # Per-block: (label, kind, line range, did it change guest state).
    effects: list[tuple[str, str, int, int, bool]] = field(default_factory=list)
    # Label -> block, so the listing can print each block's calls inline. Calls are the densest
    # semantic content the recovery already produces and the reference's `lower` prints them.
    blocks_by_label: dict = field(default_factory=dict)
    calls: int = 0
    # Where the guest context came from, or why there is none. A listing produced without a
    # frame is empty and must say so rather than looking like a function with no guest effect.
    frame: str = "unknown"
    # The arena and the guest map the listing is rendered against. Kept on the result so
    # `render` is a pure function of it rather than needing the recoverer to still be alive.
    arena: object = None
    gv: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    ctx: object = None
    cache_status: dict = field(default_factory=dict)
    lowering_truncated: int = 0
    recovery: dict = field(default_factory=dict)
    guest_state: object = None
    entry_contract: dict = field(default_factory=dict)

    @property
    def lines(self) -> int:
        return len(self.emitted.lines) if self.emitted else 0

    @property
    def summary(self) -> str:
        t = " TRUNCATED" if self.truncated else ""
        return (f"{self.root:#x}  {self.blocks:>4} blocks  {self.unresolved:>3} unresolved  "
                f"{self.instructions:>8} instr  {self.lines:>6} lines  "
                f"{self.emitted.blocks_with_effect:>4} with effect  "
                f"{self.emitted.opaque:>4} opaque ({self.emitted.opaque_rate * 100:>4.1f}%)"
                f"{t}")


def devirtualize(img: PeImage, vm, root: int, handlers: int = 900,
                 steps: int = DEFAULT_STEPS, blocks: int = DEFAULT_BLOCKS,
                 path_sensitive: bool = True, proof_dispatch: bool = True,
                 entry_context=None) -> Function:
    """Devirtualise one function.

    Two recoveries, and the first is not optional: `derive_guest_cache` measures which host
    register carries which guest register by seeding the context with fresh symbols and watching
    where they land. Without it the census refuses every attribution and the listing is empty --
    which round 36 mistook for a property of the map rather than of the configuration.
    """
    probe = Recoverer(img, vm.va, vm.end, STACK_BASE, guest_census=True,
                      entry_context=entry_context)
    cache = probe.derive_guest_cache(root, handlers=handlers, steps=steps)

    R = Recoverer(img, vm.va, vm.end, STACK_BASE, step_budget=steps,
                  block_budget=blocks, guest_census=True,
                  path_sensitive=path_sensitive, proof_dispatch=proof_dispatch,
                  entry_context=entry_context)
    R.handler_cache = cache
    cfg = R.recover(root)
    em = emit_function(R.arena, cfg, R.lifter)

    order = list(cfg.blocks.values())
    kinds = {f"{b.key}": b.kind for b in order}
    with_effect = {b.key for b in order
                   if b.guest_writes or b.guest_reg_writes or b.guest_derived
                   or any(k in ('part', 'call', 'gstore') for _, k, _ in b.guest_trace)}

    fn = Function(root=root, blocks=len(cfg.blocks), unresolved=cfg.unresolved,
                  instructions=cfg.steps, truncated=R.truncated, emitted=em)
    fn.entry_contract = dict(R.entry_contract)
    fn.blocks_by_label = {f"{b.key}": b for b in order}
    # The graph itself, so a caller (or a probe) does not have to recover it a second time to ask a
    # question about the same run. `render` is handed it separately; this makes it reachable.
    fn.cfg = cfg
    fn.calls = sum(len(b.calls) for b in order)
    fn.arena = R.arena
    fn.gv = R.lifter.guest_values
    fn.ctx = R.lifter.guest_ctx
    fn.identity = R.lifter.guest_identity
    fn.cache_status = dict(cache.walk_status)
    fn.recovery = {'path_sensitive': path_sensitive, 'proof_dispatch': proof_dispatch,
                   'conditioned_edges': R.conditioned_edges, 'proofs': dict(R.proof_stats),
                   'loop_cutpoints': R.loop_cutpoints}
    fn.guest_state = build_guest_state(cfg, R.arena)
    fn.recovery['guest_state'] = fn.guest_state.validation
    fn.frame = getattr(R, "guest_ctx_source",
                       "from the working frame sampled at the root")
    # **The conditions the recovery is carrying, in the listing rather than in a docstring.**
    # The frame exemption is an assumption, so a reader has to be able to see how many times it was
    # relied on, how many bytes it spared, and whether any of it was withdrawn.
    fn.isolation = R.machine.isolation_report() if hasattr(R, "machine") else {}
    fn.guest_state.assumptions = dict(
        context_base=fn.ctx.base if fn.ctx is not None else None,
        context_layout='measured x64 working-context layout for this protector build',
        frame_isolation=fn.isolation, private_stack_window=R.machine.stack_window,
        native_cfg='first-arrival abstraction; native states at joins are not reanalysed',
        memory='lifted store/havoc history; readonly image and call-summary assumptions apply; '
               'backing parameters and physical context bytes must agree',
        stack_model='fixed synthetic native stack; relocation to arbitrary Guest RSP is not established',
        partial=fn.truncated, unresolved=fn.unresolved,
        guest_architecture='backing slots are exact observations of the lifted machine; '
                           'cache ownership and Guest instruction cutpoints remain unproved')
    if fn.entry_contract:
        fn.guest_state.assumptions['entry_context'] = fn.entry_contract
    for i, (label, start, end) in enumerate(em.spans):
        b = order[i]
        fn.effects.append((label, kinds.get(label, "?"), start, end,
                           label in {f"{k}" for k in with_effect}))
    fn.text = render(fn, cfg)
    return fn


def bounded_text(arena, ref: int, node_budget: int = 28, max_depth: int = 3) -> str:
    """A short description of a value, for entry-state comments only.

    `<value n112331>` identifies an expression in this run's arena and nothing a reader can follow,
    and **355 of the 967 entry-state values in the corpus print that way** (36.7 percent): the join
    is the one part of a listing that is still an annotation rather than something a reader can
    trace. A *comment* may describe a value where a *body line* may not, so this is used only by the
    `entry state` line; `_namer`, which the body's environment also consults, is untouched and still
    returns a placeholder.

    Bounded on node count and depth, and it shows where it stopped: a full render of a dispatch
    expression is thousands of characters, which is presumably why the placeholder was there. An
    expression that cannot be walked falls back to the placeholder rather than raising -- the
    placeholder is the safe answer, it is just not the only one.
    """
    spent = [0]

    def walk(r: int, depth: int) -> str:
        if spent[0] >= node_budget or depth > max_depth:
            return ".."
        spent[0] += 1
        try:
            c = arena.as_const(r)
        except Exception:
            return f"<value n{r}>"
        if c is not None:
            return f"{c:#x}"
        n = arena.node(r)
        kind = n[0]
        if kind in ("opaque", "init"):
            return str(n[1])
        if kind == "param":
            return str(n[2])
        if kind == "load":
            return f"load{arena.width(r)}({walk(n[1], depth + 1)})"
        if depth == max_depth:
            # At the limit the *operator* is the informative part; printing the bare kind read as
            # `(0x2 add ((0x4 add bin) add (0x2 and bin)))`, which says less than `shr..` does.
            return f"{n[1]}.." if kind == "bin" else kind
        if kind == "bin":
            return f"({walk(n[2], depth + 1)} {n[1]} {walk(n[3], depth + 1)})"
        if kind == "un":
            return f"{n[1]}({walk(n[2], depth + 1)})"
        if kind in ("trunc", "zext", "sext"):
            return f"{kind}{arena.width(r)}({walk(n[1], depth + 1)})"
        if kind == "sel":
            return (f"sel({walk(n[1], depth + 1)}, {walk(n[2], depth + 1)}, "
                    f"{walk(n[3], depth + 1)})")
        return kind

    return walk(ref, 0)


def settle_owners(initial, rebuild, limit=16):
    """Only a complete-map fixed point supplies verified listing references."""
    owners = initial
    seen = set()
    for round_ in range(1, limit + 1):
        signature = tuple(sorted((node, tuple(values)) for node, values in owners.items()))
        if signature in seen:
            return {}, round_ - 1, False
        seen.add(signature)
        grown = rebuild(owners)
        if grown == owners:
            return grown, round_, True
        owners = grown
    return {}, limit, False


def render(fn: Function, cfg=None, width: int = 150) -> str:
    """The listing: a header, then every block with its calls and its guest effect.

    Calls are printed because the reference's `lower` prints them and because they are the
    single densest semantic content we already have: `block.calls` is `(site, target)` for every
    native call the interpreter made, and leaving them out of the listing while the reference
    shows `call KERNEL32.dll!GetTickCount` makes our output look emptier than the recovery is.
    """
    em = fn.emitted
    iso = getattr(fn, "isolation", None) or {}
    head = [
        f"; devirtualized {fn.root:#x}",
        f"; recovery {fn.recovery}",
        f"; blocks {fn.blocks}   unresolved {fn.unresolved}   "
        f"instructions {fn.instructions}",
        f"; lines {len(em.lines)}   state {em.state_lines}   "
        f"computation kept {em.kept} (of which {em.forwards_kept} do not reach a home)   "
        f"suppressed {em.suppressed}",
        f"; opaque {em.opaque} of {em.state_lines} state lines "
        f"({em.opaque_rate * 100:.1f} percent)",
        f'; exact unsimplified multi-source expressions {em.expressions}; partial writes {em.partial_writes}',
        f"; blocks with a guest effect {em.blocks_with_effect}, "
        f"without {em.blocks_without_effect}",
        f"; calls {fn.calls}",
        f"; guest frame: {fn.frame}",
        f"; frame-isolation conditions {iso.get('conditions', 0)} "
        f"(sparing {iso.get('spared_bytes', 0)} byte(s)), refuted {iso.get('refuted', 0)}; "
        f"pending guest-pointer writes {iso.get('pending_isolated', 0)}, "
        f"guarded at read time {iso.get('guards_applied', 0)}, guards dropped "
        f"{iso.get('guards_dropped', 0)}; native calls checked {iso.get('calls_checked', 0)} "
        f"(writes forgotten {iso.get('call_writes_forgotten', 0)})   [sampling walk]",
        f"; cache sample: {fn.cache_status}",
        "; IR values are block-local; identities like `v12` are not shared across blocks, "
        "but the guest register file at each entry is propagated (see the `entry state` line)",
    ]
    if fn.truncated:
        head.append("; *** TRUNCATED: this listing is a lower bound on the function ***")
    if fn.emitted.blocks_with_effect == 0:
        head.append("; *** NO GUEST EFFECT RECOVERED: treat this listing as missing, not "
                    "as a function that does nothing ***")
    lower = Lowering(fn.arena, fn.gv, fn.ctx, fn.identity)

    def _namer(node: int) -> str:
        # **A placeholder, never a reference.** This used to look up a function-global node -> name
        # map, which is last-writer-wins and checks neither that the defining block emitted the name
        # nor that it dominates -- so a comment could cite `#1:v1244:64` for a value the body could
        # not name. Every `#N:` in the listing now comes from `xstate.resolve`, which enforces both
        # conditions; anything else prints as a placeholder that is visibly not a reference. The node
        # id identifies an expression in this run's arena, not IR code, and the spelling says so.
        return f"<value n{node}>"

    def _comment_namer(node: int) -> str:
        """The `entry state` line's own namer: describe the value, do not just number it.

        The placeholder above is right for a *body* position, where anything that is not a
        definition or a reference would be a lie about the IR. An entry state is annotation, so it
        can say what the value *is* -- and 36.7 percent of its clauses were `<value nNNNN>`, which
        is what made the join the one part of a listing a reader could not trace.
        """
        return bounded_text(fn.arena, node)

    # **Cross-block guest state, which the header used to disclaim.** `core/xstate.py` propagates
    # the guest register file across the CFG, and the lowering is then given that state as its base
    # environment, so a read of a context slot prints the *name* of the value the slot holds -- one
    # defined by an earlier block, or a phi at a join -- instead of loading from an absolute frame
    # address that belongs to this process's stack. That is the difference between a listing that
    # annotates effects and one that binds them into a program a reader can follow across blocks.
    from . import xstate
    block_index = {fn.blocks_by_label[label].key: i
                   for i, (label, _k, _s, _e, _eff) in enumerate(fn.effects)
                   if label in fn.blocks_by_label}

    # **Pass A: find out what each block actually defines, before naming anything.**
    #
    # The first design had the dataflow *predict* the name (`guest_<reg>_<write number>`) from its own
    # count of a block's writes. On `0x140155dc1` the two counts disagreed for **138 of 162** recorded
    # names, and because the tag map was keyed by node and last-write-wins, a value stored 139 times
    # in one block was tagged by its 139th write -- producing dangling references (`#4:guest_r8_2`)
    # and sibling-branch ones. Names are now read off the blocks' own output, so a name that is used
    # is a definition that exists.
    #
    # The name set is stable between the passes: a read no longer consumes a register version (that
    # was removed when reads stopped being writes), so lowering *without* an entry state emits exactly
    # the register definitions the second pass will.
    probe_lower = Lowering(fn.arena, fn.gv, fn.ctx, fn.identity)

    def _owners_from(envs, node_envs):
        """Lower every block and collect what it defines **in this configuration**.

        The configuration matters: a partial write's merged node depends on the register's *prior
        value*, which the block only knows when it starts from the propagated state. A first pass with
        no entry state therefore cannot define those nodes, and their names are refused -- 25 of 306
        entry-state values on `0x140155dc1`. So the map is rebuilt until it stops changing, which is
        what makes the naming a fixed point rather than a guess.
        """
        found: dict = {}
        # A fresh counter per call: the register version numbers are per block (see
        # `Lowering.events`), and a counter carried over from a previous round would number this
        # round's blocks differently from the emission that follows it.
        probe_lower._versions = {}
        for i, (label, _kind, _s, _e, _eff) in enumerate(fn.effects):
            block = fn.blocks_by_label.get(label)
            if block is None:
                continue
            probe_lower.entry = envs.get(label, {})
            probe_lower.entry_nodes = node_envs.get(label, {})
            probe_lower.guest_mem = {}
            probe_lower.block_index = i
            probe_lower.events(block)
            for node, name in probe_lower.defs.items():
                # **Register-named definitions only, so a cross-block reference is self-describing.**
                # Every reference the state emits is then `#N:guest_<reg>[_<k>]`, which says which
                # register the value is. A temp is block-local by construction; emitting
                # `#4:v1809:64` from elsewhere is the one shape a reader and an independent checker
                # both have to special-case. A value whose only definition is a temp gets no name and
                # the read falls back to memory, which is honest rather than clever.
                _base, reg = split_guest_name(name)
                if reg is None:
                    continue
                found.setdefault(node, []).append((i, name, reg))
        return found

    root_key = None
    for label, *_rest in fn.effects:
        block = fn.blocks_by_label.get(label)
        if block is not None:
            root_key = block.key
            break
    dominates = (xstate.dominators(cfg, root_key, block_index=block_index)
                 if cfg is not None else None)

    owners = _owners_from({}, {})
    flow = (xstate.build(cfg, fn.arena, block_index=block_index, gv=fn.gv, owners=owners)
            if cfg is not None else {"entry": {}, "converged": True})
    entry_states = flow.get("entry", {})
    mem_entry = flow.get("mem_in", {})
    def rebuild_owners(current):
        envs = {}
        node_envs = {}
        for i, (label, *_rest) in enumerate(fn.effects):
            block = fn.blocks_by_label.get(label)
            if block is None or block.key not in entry_states:
                continue
            state = entry_states[block.key]
            envs[label] = xstate.entry_names(fn.arena, fn.gv, state, owners=current,
                                             dominates=dominates, block_index=i)
            node_envs[label] = xstate.entry_nodes(state)
        return _owners_from(envs, node_envs)
    owners, fn.owner_rounds, fn.owner_converged = settle_owners(
        owners, rebuild_owners, limit=max(4, len(fn.effects) + 1))
    if not fn.owner_converged:
        mem_entry = {}  # stale names must not survive in a forwarding cache
    fn.state_flow = flow

    # **Lower with the incoming state as the base environment**, so the body is expressed in terms
    # of values that cross blocks rather than of absolute frame addresses. The counts are printed
    # because "how much of the body is in terms of the entry state" is the measurement of whether
    # this is a program or an annotation.
    bodies: dict = {}
    fork_lines: dict = {}
    block_numbers: dict = {}
    cond_names: dict = {}
    terminal_names: dict = {}
    cond_lowered = 0
    cond_lines_total = 0
    state_reads = memory_reads = 0
    mem_forwarded = mem_from_memory = 0
    packs = pack_saved = 0
    rel_ok = rel_host = 0
    for i, (label, _kind, _s, _e, _eff) in enumerate(fn.effects):
        block_numbers[label] = i
        block = fn.blocks_by_label.get(label)
        if block is None:
            continue
        env = (xstate.entry_names(fn.arena, fn.gv, entry_states.get(block.key, {}),
                                  namer=_namer, owners=owners, dominates=dominates,
                                  block_index=i)
               if block.key in entry_states else {})
        lower.entry = env
        lower.entry_nodes = xstate.entry_nodes(entry_states.get(block.key, {}))
        lower.block_index = i
        lower.guest_mem = dict(mem_entry.get(block.key, {}))
        bodies[label] = lower.events(block)
        state_reads += lower.state_reads
        memory_reads += lower.memory_reads
        mem_forwarded += lower.mem_reads_forwarded
        mem_from_memory += lower.mem_reads_from_memory
        packs += lower.packs_collapsed
        pack_saved += lower.pack_lines_saved
        rel_ok += lower.mem_rel_ok
        rel_host += lower.mem_rel_host
        # **The fork condition, lowered in the listing's own vocabulary so it carries widths.**
        #
        # `Block.predicate` is `expr.render` output, which prints a DAG without widths: `(a or b)`
        # does not say how many bits either side is, and reconstructing that is a guess -- the same
        # guess `Arena._fit` makes inside the engine and a backend cannot make outside it. Naming the
        # condition through the same `Lowering` that produced the body gives `vN:W = ...` lines in the
        # grammar the body already uses, so nothing downstream needs a second parser and no width is
        # inferred.
        #
        # The names are reused: `lower.name` consults `lower.names`, which `events()` filled, so a
        # subexpression the body already defined is referenced rather than re-emitted.
        cond_name = ""
        if block is not None and len(block.succs) > 1 and getattr(block, "pred_node", 0):
            lower.lines = []
            lower.cond_mode = True
            try:
                cond_name = lower.name(block.pred_node)
            except Exception:  # a node the lowering cannot name; the fallback is honest
                cond_name = ""
            finally:
                lower.cond_mode = False
            if cond_name:
                cond_lines = list(lower.lines)
                # Kept apart from the body rather than concatenated, because the renderer emits a
                # `no guest effect recorded` marker for a block whose body is empty and that
                # information is still true when the only lines added are the fork's own: a block
                # that records no guest write and only decides still recorded no guest write.
                fork_lines[label] = cond_lines
                cond_lowered += 1
                cond_lines_total += len(cond_lines)
        cond_names[label] = cond_name
        if getattr(block, 'terminal', None):
            terminal = block.terminal
            lower.lines = []
            lower.cond_mode = True
            try:
                terminal_names[label] = (lower.name(terminal['code'])
                                         if 'code' in terminal else '')
            finally:
                lower.cond_mode = False
            fork_lines.setdefault(label, []).extend(lower.lines)
    fn.state_reads = state_reads
    fn.memory_reads = memory_reads
    fn.mem_forwarded = mem_forwarded
    fn.mem_from_memory = mem_from_memory
    fn.packs = packs
    fn.pack_saved = pack_saved
    # **`out` is assembled only now, after the body has been lowered.** The header reports how much
    # of the body was answered from the incoming state, and a header built before that is measured
    # prints zeros -- a number that is not a count of anything, which is worse than printing nothing.
    if fn.entry_contract:
        import json
        head.append('; entry context: ' + json.dumps(fn.entry_contract, sort_keys=True))
    out = head + [""]
    out.append(f'; name ownership: {fn.owner_rounds} rounds; converged={fn.owner_converged}')
    out.append(f"; cross-block state: {state_reads} context read(s) expressed from the incoming "
               f"register file (a narrow read as `truncN(name)`), {memory_reads} still read from "
               f"the frame")
    out.append(f"; guest memory: {mem_forwarded} read(s) answered from a named stored value, "
               f"{mem_from_memory} read from memory")
    out.append(f"; flags assemblies collapsed to one line: {packs} "
               f"({pack_saved} expansion line(s) not emitted)")
    out.append(f"; fork conditions lowered: {cond_lowered} "
               f"({cond_lines_total} line(s) added to the bodies)")
    out.append(f"; guest memory events (reads and writes): {rel_ok + rel_host} total, "
               f"{rel_ok} with a guest-relative address, {rel_host} with a host-sized offset "
               f"(classified as guest traffic but not guest-relative -- see PROBLEMS.md B11)")
    out.append('; Entry states summarize recorded effects; recovery still uses first-arrival '
               'machine states at joins. These annotations do not establish whole-function SSA.')
    if cfg is not None and not flow.get("converged", True):
        out.append("; *** ENTRY-STATE JOIN DID NOT CONVERGE: some registers are reported as "
                   "unknown rather than at a fixed point ***")
    for i, (label, kind, start, end, eff) in enumerate(fn.effects):
        block = fn.blocks_by_label.get(label)
        out.append(f"{label}:   [{kind}]   ; #{i}")
        if block is not None and block.key in entry_states:
            out.append("    ; " + xstate.render(fn.arena, fn.gv, entry_states[block.key],
                                                namer=_comment_namer, owners=owners,
                                                dominates=dominates, block_index=i))
        if block is not None and not any(kind == 'call' for _, kind, _ in block.guest_trace):
            for site, name in block.calls:
                out.append(f"    call     {name}   @{site:#x}")
        # **The body is the lowered three-address stream**, in execution order, whenever the
        # block recorded a guest event. `em.lines[start:end]` remains the fallback -- and is
        # what supplies the tally in the header -- because it is the rendering whose counts are
        # audited; this is the rendering a reader uses.
        body = bodies.get(label) or []
        if body:
            out.extend("    " + ln for ln in body)
        else:
            for k, reg, text in em.lines[start:end]:
                # **The empty-block marker is an annotation, not IR.** `none  no guest effect
                # recorded; ...` is a statement *about* the block, and the independent checker is
                # right to call it a line that defines nothing and uses nothing. Emitting it as a
                # comment keeps the information (a reader must be able to tell a block with no guest
                # effect from one that was never analysed) without putting a non-instruction into the
                # instruction stream.
                if k == "none":
                    out.append(f"    ; {text}")
                    continue
                tag = (f"{reg:<6}") if reg else " " * 6
                out.append(f"    {k:<8} {tag} {text}")
        # **Control flow, which the recovery has always had and the listing never showed.** A
        # devirtualized function without its successors is a bag of blocks; the reference's
        # `lower` ends each block with `br ? A : B`. The successors are the CFG's own edges, and
        # an unresolved block says so rather than being rendered as a fall-through.
        if block is not None:
            # The fork condition's own definitions, after the body (it may reuse the body's temps)
            # and before the terminator comment that names it.
            for ln in fork_lines.get(label) or ():
                out.append("    " + ln)
            if len(block.succs) > 1:
                out.append("    " + branch_condition(block, cond_names.get(label, "")))
            if block.succs:
                out.append("    -> " + ", ".join(str(s) for s in block.succs))
            elif block.kind in ('noreturn', 'fastfail'):
                if block.kind == 'fastfail':
                    detail = 'code=' + terminal_names.get(label, '<unavailable>')
                else:
                    detail = block.terminal.get('callee', block.detail)
                out.append(f"    -> <{block.kind}: {detail} @{block.terminal['site']:#x}>")
            elif block.kind == "tailcall":
                # **A tail call ends the block, and the arrow is what says so.** `jmp [import]` leaves
                # this function for a named callee that returns to our caller, so the recovered
                # function ends here -- but a block with no arrow at all reads as a block whose exit
                # was lost, which is why two driver roots (`0x145bd7fa4`, `0x145c24c0b`) were refused
                # for "an exit the listing does not settle" while the engine had settled it.
                out.append(f"    -> <tail: {block.detail}>")
            elif block.kind == "unresolved":
                out.append("    -> <unresolved: the dispatcher target was not recovered>")
            elif block.kind == "return":
                out.append("    -> return")
            elif block.kind == "infeasible":
                # **A proved-infeasible path still ends.** The block has no successors because the
                # accumulated conditions are unsatisfiable, and printing nothing left the text with
                # a block that simply stopped -- the acceptance gate read that as a missing
                # successor line, which is the same shape a *bug* in the edge builder would have.
                # A terminal arrow says which of the two it is.
                out.append("    -> <infeasible: the accumulated path conditions are unsatisfiable>")
            elif block.kind == "loop":
                # Reached only when the self-loop key did not come back, so no successor is
                # claimed; the arrow states the cutpoint instead of leaving the block looking
                # like one whose edge was lost.
                out.append("    -> <loop: the dispatch sequence repeats; no exit was observed>")
    fn.lowering_truncated = lower.truncated
    if lower.truncated:
        out.insert(0, f"; *** INCOMPLETE EXPRESSIONS: {lower.truncated} depth-limit cut(s) ***")
    return "\n".join(out) + "\n"


def branch_condition(block, cond_name: str = "") -> str:
    """The condition that chooses between a two-way block's successors, as one comment line.

    A listing has always printed a forking block's successors and never said which is taken when, so
    a reader could not follow the branch and a backend could not emit one -- the only honest options
    were to refuse or to invent, and refusing was right. The condition was in the CFG all along
    (`Block.predicate`); what was missing was the **association**, which `Block.edge_cond` now keeps
    because `_terminate` had to establish it to build the edge pins and then threw it away.

    Two forms, and the difference is a claim about what a backend can do:

    * `... ; condition vN:1 ; successor when 0: A ; successor when 1: B` -- the condition was lowered
      into the listing's own vocabulary, so it has a width and a definition in this block, and the
      0/1 association was verified. **This is transcribable.**
    * `... ; one successor of N selected by this condition` -- the condition is known but was not
      lowered (or no 0/1 association was established), so nothing downstream may turn it into a
      branch. The line says so instead of implying one.

    Deliberately a *comment*, placed immediately before the `->` line, so every existing consumer of
    the listing -- including the IR acceptance gate and any parser that keys on `-> ` -- sees exactly
    the line it saw before.
    """
    text = getattr(block, 'predicate', '') or ''
    cond = list(getattr(block, 'edge_cond', ()) or ())
    succs = list(block.succs or ())
    if not text and not cond_name:
        return '; branch condition: <none recovered for this fork>'
    usable = (cond_name and len(cond) == len(succs) and all(v is not None for v in cond)
              and set(cond) == {0, 1})
    if usable:
        by_value = {v: str(s) for v, s in zip(cond, succs)}
        line = (f'; branch condition: {cond_name}'
                f'   ; successor when 0: {by_value[0]}'
                f'   ; successor when 1: {by_value[1]}')
        if text:
            line += f'   ; from {text[:400]}'
        return line
    return ('; branch condition: ' + (text or '<unnamed>')
            + f'   ; one successor of {len(succs)} selected by this condition')


def module_roots(img: PeImage):
    """The VM entry points, in address order."""
    from .entries import vm_roots
    return vm_roots(img)


def main() -> int:
    import os
    import sys

    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    out_dir = None
    only = None
    caller_entry = call_site = None
    show = 0
    steps, blocks = DEFAULT_STEPS, DEFAULT_BLOCKS
    legacy = '--legacy-recovery' in sys.argv
    proof = not legacy and '--no-proof-dispatch' not in sys.argv
    for x in sys.argv[1:]:
        if x.startswith("--out="):
            out_dir = x.split("=", 1)[1]
        if x.startswith("--root="):
            only = int(x.split("=", 1)[1], 0)
        if x.startswith('--caller-entry='):
            caller_entry = int(x.split('=', 1)[1], 0)
        if x.startswith('--call-site='):
            call_site = int(x.split('=', 1)[1], 0)
        if x.startswith("--show"):
            show = int(x.split("=", 1)[1], 0) if "=" in x else 60
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--blocks="):
            blocks = int(x.split("=", 1)[1], 0)
    if not args:
        print(__doc__)
        return 2

    img = PeImage(args[0])
    context = None
    if caller_entry is not None or call_site is not None:
        if only is None or caller_entry is None or call_site is None:
            raise ValueError('caller context requires --root, --caller-entry and --call-site together')
        from .entrycontext import EntryContext
        context = EntryContext(caller_entry, call_site, only)
    vm = img.section_by_name(".tvm0")
    roots = [only] if only is not None else module_roots(img)
    print(f"{args[0]}   .tvm0 {vm.va:#x}..{vm.end:#x}   roots {len(roots)}")
    print()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_lines = total_effect = 0
    for root in roots:
        fn = devirtualize(img, vm, root, steps=steps, blocks=blocks,
                         path_sensitive=not legacy, proof_dispatch=proof,
                         entry_context=context)
        print(fn.summary)
        total_lines += fn.lines
        total_effect += fn.emitted.blocks_with_effect
        if out_dir:
            path = os.path.join(out_dir, f"{root:#x}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(fn.text)
            import json
            state_path = os.path.join(out_dir, f"{root:#x}.guest-state.json")
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(fn.guest_state.to_dict(), fh, indent=2)
        if show:
            print()
            print("\n".join(fn.text.splitlines()[:show]))
            print()
    print()
    print(f"TOTAL {total_lines} lines over {len(roots)} functions, "
          f"{total_effect} blocks with a guest effect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
