"""The guest register file at every block's entry, propagated across the CFG.

The listing has always ended with "IR values are block-local; cross-block phi/equivalence is not
established", and that disclaimer was the honest description of the artifact: each block was
described as a run of guest operations starting from an unnamed state, so a reader could not tell
what `guest_rax` held when a block began, and a backend could not be built on it at all.

This module supplies the missing half. A block's **exit** state is its entry state with its own
recorded guest writes applied in trace order; an **entry** state is the merge of its predecessors'
exit states. That is the standard forward problem, and what comes out is the guest register file at
every program point the recovery reached.

Three deliberate choices, each because the alternative would be a claim the engine cannot support:

* **A disagreement is a phi, not a fabricated value.** When predecessors disagree the entry value is
  recorded as the *set* of incoming values. Synthesising `select(cond, a, b)` needs the condition on
  the edge, and the CFG does not carry one -- inventing it would put a fabricated predicate into the
  IR, which is the failure this workspace keeps recording. A phi is exact and a reader can act on it.
* **Termination is enforced, and non-convergence is printed.** The merge can oscillate, so a register
  that is revisited too often is widened to `unknown` with the reason, rather than the fixed point
  being assumed.
* **A partial write with no known prior value is unknown, not zero-extended.** `mov [ctx+8i], eax`
  leaves the upper half as it was; if that half was never established, the register's value is not
  known and saying so is the whole point of the recovery.
"""

from __future__ import annotations

from collections import deque

from .expr import W64

REGS = ("rax", "rbx", "rcx", "rdx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rflags")
ALL = REGS
MAX_UPDATES = 6


def _entry(reg: str) -> tuple:
    return ("in", reg)


def _unknown(why: str) -> tuple:
    return ("unknown", why)


def _value(node: int) -> tuple:
    """A known value.

    The *name* of a value -- `#3:guest_rax_1`, defined in block 3 -- lives in a separate map rather
    than in this tuple, because it is metadata about a node rather than part of its identity: two
    writes that produced the same node produced the same value, so one name describes both. Keeping
    it out of the tuple also keeps the state values comparable by construction, which is what the
    join tests assert.
    """
    return ("value", node)


def _partial(arena, prev: tuple, lo: int, nbytes: int, value: int) -> tuple:
    """`prev` with bytes `lo..lo+nbytes` replaced -- the semantics of a narrow ctx store."""
    if prev[0] == "unknown":
        return prev
    if prev[0] == "phi":
        arms = tuple((pred, _partial(arena, incoming, lo, nbytes, value))
                     for pred, incoming in prev[1])
        if all(incoming == arms[0][1] for _, incoming in arms):
            return arms[0][1]
        return ("phi", arms)
    if prev[0] == "in":
        base = arena.init_reg(prev[1])
    else:
        base = prev[1]
    shift = lo * 8
    bits = nbytes * 8
    keep = ((1 << 64) - 1) ^ (((1 << bits) - 1) << shift)
    low = arena.bin("and", base, arena.const(keep, W64), W64)
    wide = arena.zext(value, W64) if arena.width(value) < 64 else value
    if shift:
        wide = arena.bin("shl", wide, arena.const(shift, W64), W64)
    placed = arena.bin("and", wide, arena.const(((1 << bits) - 1) << shift, W64), W64)
    return _value(arena.bin("or", low, placed, W64))


def exit_state(arena, block, entry: dict[str, tuple], block_index: int = 0,
               owners: dict | None = None, gv=None, mem: dict | None = None):
    """The entry state with this block's guest writes applied, in execution order.

    **`dict(mem)` is not tidiness.** `_walk` updates the memory map as it goes, so passing the
    caller's dictionary made the *entry* state and the *exit* state the same object: a load before a
    store in one block was resolved with the value the later store wrote. The audit reproduced it (a
    load before a store returned the store's value), and separating the two is the only reason the
    recorded entry state means what it says.

    When `mem` is given, guest memory is advanced in the same pass and the return value is
    `(registers, memory)`; see `_walk` for why the two cannot be computed separately.
    """
    regs, out_mem = _walk(arena, block, dict(entry),
                          dict(mem) if mem is not None else None,
                          gv if gv is not None else {},
                          owners if owners is not None else {}, block_index)
    return (regs, out_mem) if mem is not None else regs


def _key(value: tuple):
    """Comparable identity for a state value, so two entries can be tested for agreement."""
    if value[0] == "phi":
        return ("phi", tuple((str(k), v) for k, v in value[1]))
    return value


def merge(values: list[tuple]) -> tuple:
    """The entry value implied by a set of predecessor values."""
    if not values:
        return _unknown("no predecessor state")
    first = _key(values[0])
    if all(_key(v) == first for v in values):
        return values[0]
    seen = []
    for _pk, v in ((None, v) for v in values):
        if _key(v) not in [_key(x[1]) for x in seen]:
            seen.append((None, v))
    if len(seen) == 1:
        return seen[0][1]
    return ("phi", tuple(seen))


def build(cfg, arena, root_key=None, block_index=None, gv=None, owners=None) -> dict:
    """`{block key: {reg: state value}}` at every block's entry, to a fixed point.

    `block_index` maps a block key to the number the listing prints for it; it is what lets a value
    be named `#3:guest_rax_1` -- defined in block 3. Without it the state is exact but anonymous.

    Returns a dict with `entry`, `exit`, `updates` and `converged`, because a dataflow result that
    cannot say whether it reached a fixed point is not a result.
    """
    block_index = block_index or {}
    owners = owners or {}
    blocks = {b.key: b for b in cfg.blocks.values()}
    if not blocks:
        return {"entry": {}, "exit": {}, "updates": {}, "converged": True}
    if root_key is None:
        root_key = getattr(cfg, 'entry', None)
    if root_key is None:
        roots = [k for k, b in blocks.items() if not b.preds]
        root_key = roots[0] if roots else next(iter(blocks))

    if root_key not in blocks:
        raise ValueError('CFG entry is not a recovered block')
    seed = {r: _entry(r) for r in ALL}
    entry: dict = {}
    exit_: dict = {}
    updates: dict = {}
    # Guest memory, at each block's entry and exit. `{key: (value name, nbytes)}`, where a key is
    # `(base register name, offset)` -- see `mem_key`.
    mem_entry: dict = {}
    exit_mem: dict = {}
    converged = True

    predecessors = {key: [] for key in blocks}
    for pred, b in blocks.items():
        for succ in b.succs:
            if succ in predecessors and pred not in predecessors[succ]:
                predecessors[succ].append(pred)
    work = deque([root_key])
    seen_queue = {root_key}
    while work:
        key = work.popleft()
        seen_queue.discard(key)
        block = blocks[key]
        incoming = [(pred, exit_[pred]) for pred in predecessors[key] if pred in exit_]
        if key == root_key:
            incoming.insert(0, (None, seed))
        if not incoming:
            continue
        incoming_mem = [(pred, exit_mem.get(pred, {})) for pred in predecessors[key]
                        if pred in exit_]
        if key == root_key:
            incoming_mem.insert(0, (None, {}))
        merged = {}
        for reg in ALL:
            arms = tuple((pred, state[reg]) for pred, state in incoming)
            values = [value for _, value in arms]
            unknown = next((v for v in values if v[0] == 'unknown'), None)
            candidate = (unknown if unknown is not None else values[0]
                         if all(v == values[0] for v in values) else ('phi', arms))
            old = entry.get(key, {}).get(reg)
            if old == _unknown('join did not converge'):
                candidate = old  # widening is sticky
            elif old is not None and candidate != old:
                updates[(key, reg)] = updates.get((key, reg), 0) + 1
                if updates[(key, reg)] >= MAX_UPDATES:
                    candidate = _unknown('join did not converge')
                    converged = False
            merged[reg] = candidate
        # **Guest memory joins by must-match only.** A location survives a join when *every*
        # predecessor holds the same value there; anything else is dropped. That is the sound
        # direction for an exact-key map -- a location only some predecessors agree on cannot be
        # forwarded, and saying so costs a naming opportunity rather than a wrong value.
        mem_arms = [m for _pred, m in incoming_mem]
        candidate_mem: dict = {}
        if mem_arms:
            common = set(mem_arms[0])
            for m in mem_arms[1:]:
                common &= set(m)
            for k in common:
                first = mem_arms[0][k]
                if all(m[k] == first for m in mem_arms[1:]):
                    candidate_mem[k] = first
        entry[key] = merged
        mem_in = candidate_mem
        regs_out, mem_out = exit_state(arena, block, merged, block_index.get(key, 0), owners,
                                       gv, mem_in)
        mem_entry[key] = mem_in
        out = regs_out
        if key in exit_ and out == exit_[key] and mem_out == exit_mem.get(key):
            continue
        exit_[key] = out
        exit_mem[key] = mem_out
        for succ in block.succs:
            if succ not in blocks:
                continue
            if succ not in seen_queue:
                work.append(succ)
                seen_queue.add(succ)

    return {"entry": entry, "exit": exit_, "updates": updates, "converged": converged,
            "mem_in": mem_entry, "mem_out": exit_mem}


def canonical(gv, owners, node: int, reg: str | None = None) -> str | None:
    """A name for a value that the *dataflow* and the *lowering* both compute the same way.

    **The name comes from what the lowering actually emitted, not from a prediction.** The first
    design had the dataflow predict the name (`guest_<reg>_<write number>`) from its own count of a
    block's writes, and the two counts disagreed on a real function: of 162 recorded tags, **138**
    named a definition their block never emitted, and because the tag map was keyed by node and
    written last-wins, a value stored 139 times in one block was tagged by its 139th write. The audit
    found the resulting dangling references (`#4:guest_r8_2`) and the sibling-branch variant.

    `owners` is `{node: [(block index, name, register), ...]}` collected from the blocks' own output,
    so a name returned here **is** a definition that exists. `reg` prefers an owner that defined the
    value as that register, which keeps a rename readable.
    """
    if node in gv:
        return f"guest_{gv[node]}"
    found = owners.get(node)
    if not found:
        return None
    if reg is not None:
        for bi, name, owner_reg in found:
            if owner_reg == reg:
                return f"#{bi}:{name}"
    bi, name, _reg = found[0]
    return f"#{bi}:{name}"


def mem_key(base: str, off: int) -> tuple:
    """A guest-memory location, as `(name of the base register, offset)`.

    Register *names* rather than nodes, because the two sides of the propagation have to agree on
    the identity of an address without sharing an arena-level handle. Two accesses share a key only
    when their base registers hold the same value -- which is what makes the match an exact address
    match rather than a guess, and what makes it sound to forward one to the other.
    """
    return (base, off)


def _base_name(value: tuple, gv, owners) -> str | None:
    """The canonical name of a register value, or None when it has none.

    **The spelling must match `entry_names` exactly, and the first version did not**: this returned
    `in.rcx` for a register still at its VM-entry value while `entry_names` returns `guest_rcx`, so
    every key computed here was unmatchable by the lowering and cross-block guest-memory forwarding
    fired **zero** times on a real root -- silently, since a key that never matches looks exactly
    like a location that was never stored. The unit test caught it; the corpus could not, because
    both spellings are internally consistent.
    """
    if value is None:
        return None
    if value[0] == "in":
        return f"guest_{value[1]}"
    if value[0] == "unknown":
        return None
    if value[0] == "phi":
        arms = [_base_name(v, gv, owners) for _pk, v in value[1]]
        if any(a is None for a in arms):
            return None
        return "phi(" + ", ".join(arms) + ")"
    return canonical(gv, owners, value[1])


def _walk(arena, block, regs: dict, mem: dict | None, gv, owners, block_index: int):
    """One pass over a block's trace, advancing the register view and (optionally) memory.

    Registers and guest memory have to be advanced **together**, because a memory entry's key is the
    *name of the base register at the point of use*: a block that does `mov rcx, rbx` and then stores
    through `rcx` writes where `rbx` points, and a walk that computed the register view separately
    would key it on the wrong value.

    **`block_index` is passed for reporting only, and the write counter that used to live here is
    gone.** It existed so the dataflow could *predict* the name `guest_<reg>_<n>` the emitter would
    produce, and on a real function the two counts disagreed for 138 of 162 recorded tags. Names now
    come from what the emitter actually emitted (`owners`), so no counter has to agree with anything.
    """
    for _step, kind, payload in sorted(getattr(block, "guest_trace", ()) or (),
                                       key=lambda t: t[0]):
        if kind in ("store", "home", "part"):
            if kind == "store":
                _site, reg, value = payload
                produced = value
            elif kind == "home":
                reg, _host, produced = payload
            else:
                _site, reg, lo, nbytes, value = payload
                if reg not in regs:
                    continue
                regs[reg] = _partial(arena, regs[reg], lo, nbytes, value)
                continue
            if reg in regs:
                regs[reg] = _value(produced)
            continue
        if mem is None or kind not in ("gstore", "gload", "call"):
            continue
        if kind == "call":
            # **A native call may write guest memory, so nothing parked here survives it.** The
            # audit reproduced this: after an unsummarized call the lowering went on forwarding the
            # pre-call value. Guest memory is not the machine's overlay, so `Machine.forget_call_writes`
            # -- which forgets the *frame* -- does not cover it.
            mem.clear()
            continue
        reg, off, nbytes = payload[1], payload[2], payload[3]
        base = _base_name(regs.get(reg), gv, owners)
        if base is None:
            continue
        key = mem_key(base, off)
        if kind == "gstore":
            value = payload[4]
            # **Every entry the write may reach is dropped, not just the one it exactly covers.**
            # Three separate reproductions sit here: the same base with an overlapping range (a
            # one-byte write over the front of a qword), a *different* base that may point at the
            # same address (nothing here can prove two register values disjoint), and a write of a
            # narrower width than a later read. A location that is not provably untouched cannot be
            # forwarded, and "not provably" is the common case when the base name differs.
            for other in [k for k in mem
                          if k[0] != base
                          or (k[1] < off + nbytes and off < k[1] + mem[k][1])]:
                del mem[other]
            name = canonical(gv, owners, value, reg)
            if name is not None:
                mem[key] = (name, nbytes)
        elif key in mem:
            # A load of a location whose value is already named: keep the pair only at the width it
            # was written with, so the lowering can check it before printing a name.
            mem[key] = (mem[key][0], mem[key][1])
    return regs, mem


def entry_nodes(entry: dict) -> dict:
    """`{reg: node}` for the registers whose entry state is a known value.

    The lowering needs the **nodes**, not just the names, to compute the result of a partial write
    the same way the dataflow does: `part` merges the written bytes into the register's prior value,
    and if the two sides build that merge from different inputs they produce different nodes -- so the
    value the state holds has no definition in the block that emitted it and its name is refused.
    Measured on `0x140155dc1`: 25 of 306 entry-state values were refused for exactly this reason, and
    the flags slot is the biggest contributor because it is written four bytes at a time, 47 times.
    """
    out: dict = {}
    for reg in ALL:
        value = entry.get(reg)
        if value is None:
            continue
        if value[0] == "in":
            continue          # the lowering knows `in.<reg>` by name; there is no node to pin
        if value[0] == "value":
            out[reg] = value[1]
    return out


def partial(arena, prev_node, lo: int, nbytes: int, value: int) -> int:
    """The node a narrow write produces from a known prior node. Shared by both sides."""
    shift = lo * 8
    bits = nbytes * 8
    keep = ((1 << 64) - 1) ^ (((1 << bits) - 1) << shift)
    low = arena.bin("and", prev_node, arena.const(keep, W64), W64)
    wide = arena.zext(value, W64) if arena.width(value) < 64 else value
    if shift:
        wide = arena.bin("shl", wide, arena.const(shift, W64), W64)
    placed = arena.bin("and", wide, arena.const(((1 << bits) - 1) << shift, W64), W64)
    return arena.bin("or", low, placed, W64)


def dominators(cfg, root_key, block_index=None):
    """`dominates(a, b)` over printed block numbers, for the cross-block reference check.

    **A name defined on a sibling path cannot be used at a join.** The audit reproduced exactly that:
    the right branch's entry state referred to a definition the left branch had made, and the right
    branch's own execution never passes through the left one, so the reference is undefined on the
    path that uses it. Equal values do not make a definition available -- only dominance does.

    `block_index` is the same `{block key: printed number}` map the caller uses for `#N:`, and it is
    required whenever the caller passes numbers in. The first version numbered blocks by *CFG
    dictionary order* while the callers passed the *listing* index; the two agreed on small graphs and
    silently disagreed on real ones, refusing every reference -- which showed up not as a wrong answer
    but as an empty population in the independent checker's first rule.

    Standard iterative dataflow: `dom(root) = {root}`, `dom(n) = {n} union the intersection of the
    predecessors' sets`, to a fixed point. A block with no predecessors keeps only itself.
    """
    blocks = {b.key: b for b in cfg.blocks.values()}
    preds: dict = {k: [] for k in blocks}
    for k, b in blocks.items():
        for succ in b.succs:
            if succ in preds and k not in preds[succ]:
                preds[succ].append(k)
    if block_index:
        number_of = {num: key for key, num in block_index.items() if key in blocks}
    else:
        number_of = {i: k for i, k in enumerate(blocks)}
    dom = {k: ({k} if k == root_key else set(blocks)) for k in blocks}
    changed = True
    guard = 0
    while changed and guard < len(blocks) + 2:
        guard += 1
        changed = False
        for k in blocks:
            if k == root_key:
                continue
            incoming = [dom[p] for p in preds[k] if p in dom]
            if not incoming:
                new = {k}
            else:
                new = {k} | set.intersection(*incoming)
            if new != dom[k]:
                dom[k] = new
                changed = True

    def dominates(a: int, b: int) -> bool:
        ka, kb = number_of.get(a), number_of.get(b)
        if ka is None or kb is None:
            return False
        return ka in dom.get(kb, set())

    return dominates


def resolve(value: tuple, reg: str | None, gv, owners, dominates, block_index) -> str | None:
    """The single name resolver, used by **both** the entry-state comment and the body.

    They used to disagree. The body resolved through `owners` and dominance; the comment fell back to
    a function-global node map that checked neither, so on `0x140155dc1` every entry-state comment
    cited a `#N:v…` reference the body did not make -- unverified in exactly the way the audit found.
    One resolver, one answer: whatever a comment says a register holds, the body names the same way.
    """
    if value is None:
        return None
    if value[0] == "in":
        return f"guest_{value[1]}"
    if value[0] == "unknown":
        return None
    if value[0] == "phi":
        arms = [resolve(v, None, gv, owners, dominates, block_index) for _pk, v in value[1]]
        if any(a is None for a in arms):
            return None
        return "phi(" + ", ".join(arms) + ")"
    node = value[1]
    named = gv.get(node)
    if named is not None:
        return f"guest_{named}"
    found = owners.get(node) or ()
    for want_reg in ((reg, None) if reg is not None else (None,)):
        for bi, owner_name, owner_reg in found:
            if want_reg is not None and owner_reg != want_reg:
                continue
            if block_index is not None and bi == block_index:
                continue  # this definition is not available at its own entry
            if dominates is not None and block_index is not None and not dominates(bi, block_index):
                continue
            return f"#{bi}:{owner_name}"
    return None


def entry_names(arena, gv, entry: dict, namer=None, owners=None, dominates=None,
                block_index: int | None = None) -> dict[str, str]:
    """`{reg: name}` for the registers whose entry value can be *named* in the IR.

    This is the bridge from the propagated state to the lowering. A block that reads a context slot
    before writing it prints the name of the value that slot holds -- a value defined by an earlier
    block, or a phi at a join -- rather than an absolute frame load that is a property of this
    process's stack. `unknown` is deliberately absent: a value the dataflow could not establish has
    no name, and the read falls back to the memory load rather than being labelled with something the
    engine does not know.

    **Two conditions, and the audit found a listing that violated both.** The name must come from a
    block that actually emitted it (`owners`, collected from the blocks' own output), and that block
    must **dominate** the use -- otherwise a joining block refers to a definition on a sibling path
    that its own execution never passes through. Both are checked here rather than assumed; a name
    that fails either is dropped and the read goes to memory.
    """
    owners = owners or {}
    out: dict[str, str] = {}
    for reg in ALL:
        value = entry.get(reg)
        if value is None:
            continue
        if value[0] == "in":
            out[reg] = f"guest_{value[1]}"
        elif value[0] == "value":
            chosen = resolve(value, reg, gv, owners, dominates, block_index)
            if chosen is not None:
                out[reg] = chosen
            # **No unverified fallback here.** The old one resolved a name from a function-global node
            # map that is last-writer-wins and checked neither that the defining block emitted the
            # name nor that it dominates -- the same two conditions this function exists to enforce.
            # On `0x140155dc1` it produced `#4:v1809:64` at a block block 4 does not dominate, and
            # `<expression n62233>` where no name existed at all: one dangling reference and one line
            # that was not IR. A value with no resolvable name simply gets none, and the read falls
            # back to memory.
        elif value[0] == "phi":
            shown = [_names(arena, gv, v, namer) for _pk, v in value[1][:2]]
            more = "" if len(value[1]) <= 2 else f", +{len(value[1]) - 2}"
            out[reg] = "phi(" + ", ".join(shown) + more + ")"
    return out


def _names(arena, gv, value: tuple, namer=None) -> str:
    if value[0] == "in":
        return f"in.{value[1]}"
    if value[0] == "unknown":
        return f"<unknown: {value[1]}>"
    if value[0] == "phi":
        shown = [_names(arena, gv, v, namer) for _pk, v in value[1][:3]]
        more = "" if len(value[1]) <= 3 else f", +{len(value[1]) - 3}"
        return "phi(" + ", ".join(shown) + more + ")"
    node = value[1]
    c = arena.as_const(node)
    if c is not None:
        return f"{c:#x}"
    reg = gv.get(node)
    if reg is not None:
        return f"guest_{reg}"
    if namer is not None:
        got = namer(node)
        if got:
            return got
    return "<expression>"


def render(arena, gv, entry: dict, width: int = 150, namer=None, owners=None,
           dominates=None, block_index: int | None = None) -> str:
    """One line per block: what the guest register file holds on entry, and how much is untouched.

    The renames that say nothing are dropped. `guest_rax = guest_rax` is what an identity mark
    produces for a register the block never touched, and printing seventeen of those is noise rather
    than information. What is shown is a rename, a constant, or a phi; what is counted is everything
    else.

    **The same resolver as the body.** `namer` is only a display fallback for a value that has no
    verified name, and it is used solely when `resolve` returns None -- so a clause that quotes a
    `#N:name` is quoting a reference the body could also make.
    """
    interesting: list[str] = []
    aliases: list[str] = []
    unchanged = 0
    for reg in ALL:
        value = entry.get(reg)
        if value is None:
            continue
        resolved = resolve(value, reg, gv, owners or {}, dominates, block_index)
        shown = resolved if resolved is not None else _names(arena, gv, value, namer)
        if value[0] == "in" and value[1] == reg:
            unchanged += 1
            continue
        if shown == f"guest_{reg}":
            unchanged += 1          # a rename to itself: the register did not change
            continue
        text = f"guest_{reg} = {shown}"
        if resolved is not None and value[0] == "value" and gv.get(value[1]) is not None:
            aliases.append(text)
        else:
            interesting.append(text)
    tail = f"   ({unchanged} register(s) unchanged)"
    parts = interesting + aliases
    if not parts:
        return "entry state: unchanged from VM entry" + tail
    # **Every clause, however long the line gets.** The line used to be truncated to `width`, which
    # silently dropped clauses: the registers they named were then neither listed nor counted as
    # unchanged, and the independent checker found seven entry states summing to 16 instead of 17. The
    # count of registers the line accounts for is the one property that has to survive, and a long
    # complete line beats a short one whose arithmetic does not close. `width` is kept as an argument
    # because a caller may still want to know what it would have been.
    _ = width
    return "entry state: " + " ; ".join(parts) + tail
