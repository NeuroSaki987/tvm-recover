"""Three-address emission for a recovered block.

Everything up to this round made values *nameable*; nothing was ever written down.
A devirtualizer that cannot emit is a tracer, so this turns the interpreter's own
record of a walk into statements an analyst can read, and into a form that can be
compared against the reference's `lower` command.

The design decision that matters is **what granularity to emit at**, and it is
settled by measurement rather than taste. The engine's block identity is
`(handler, vip)` -- one handler execution -- and on `0x14015fc2e` that yields 149
blocks. The reference's `lower` prints **50** blocks with the same key syntax. The
two are not in conflict and they are not the same thing: the reference emits *guest*
basic blocks, terminated by a decision, and its `0x14015fc2e` block runs from the
prologue through `GetTickCount`, several calls and a branch. A guest basic block
therefore spans several handler executions, and the emitter reports both.

What is emitted is what the interpreter *did*, not a re-derivation: every call with
its target resolved to a named import where the IAT slot says so and its four Win64
argument registers, and the block's exit as a transfer kind and successor.

Nothing is scheduled, register-allocated or simplified. Calling this `lower`-equivalent
would be false; it is the same *level* of description with less work done to it, and
the part that is directly comparable is objective: the multiset of calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .expr import W64, render


@dataclass
class IrBlock:
    """One handler block's effect, as the interpreter recorded it."""

    handler: int
    vip: int
    instructions: int = 0
    calls: list[tuple[int, str, list[tuple[str, int]]]] = field(default_factory=list)
    exit_kind: str = "?"
    successor: int | None = None

    @property
    def key(self) -> str:
        return f"{self.handler:#x}@vip{self.vip:#x}"


_HEX = None


def classify_predicate(text: str, vm_lo: int, vm_hi: int) -> str:
    """What a block's terminating condition is a condition *on*.

    This is the question round 5's block-count puzzle turned into. The engine's block
    ends at every dispatch; the reference's ends at a guest decision; and no
    structural rule on the recovered graph separates the two, because the handler
    graph is densely shared and a guest decision is not a graph property.

    The reference's own output says what it is instead, and it is measurable: of its
    50 block predicates, **31 depend on a call result, 9 on a load, and 0 mention a
    `.tvm0` address**. So the classification is:

    * **`vm`** -- the predicate mentions a literal inside `.tvm0`. That is VM
      bookkeeping by construction: the section holds handlers and bytecode, and a
      decision on one is the dispatcher deciding, not the guest;
    * **`guest`** -- the predicate mentions `call_ret#N` or a `load[`, i.e. a value
      that came from outside the VM's own machinery;
    * **`other`** -- neither, which is where a constant-folded or flag-only predicate
      lands and is reported rather than forced into one of the two.

    The `vm` test uses the two addresses the engine already derives rather than a
    spelling like `0x1401`, so it stays correct across builds: a classifier keyed on a
    pattern would be a pattern, not a measurement.
    """
    import re as _re

    global _HEX
    if _HEX is None:
        _HEX = _re.compile(r"0x[0-9a-fA-F]+")
    for m in _HEX.finditer(text):
        try:
            v = int(m.group(0), 16)
        except ValueError:
            continue
        if vm_lo <= v < vm_hi:
            return "vm"
    if "call_ret#" in text or "load[" in text or "load64[" in text:
        return "guest"
    return "other"


def classify_blocks(cfg, vm_lo: int, vm_hi: int) -> dict[str, list]:
    """Every block whose terminator is a *decision*, classified.

    A decision is a block that does not resolve to exactly one successor -- a
    symbolic branch, an unresolved dispatch, or a call the VM branches on. A block
    that falls through to a single destination is a dispatch, and the engine's
    ordinary `jump` blocks are exactly that.
    """
    out: dict[str, list] = {"guest": [], "vm": [], "other": [],
                            "not_a_decision": []}
    for key in cfg.order:
        b = cfg.blocks.get(key)
        if b is None:
            continue
        if b.kind == "jump":
            out["not_a_decision"].append(b)
            continue
        out[classify_predicate(b.predicate or "", vm_lo, vm_hi)].append(b)
    return out



def call_target(lifter, event) -> str:
    """A call's name: the import symbol when the target is an IAT slot."""
    slot = event.get("import_slot")
    if slot is not None:
        mod, sym = lifter.imports[slot]
        return f"{mod}!{sym}"
    if event.get('target') is None and event.get('target_expr') is not None:
        return f"indirect@{event['site']:#x}"
    return f"{event['target']:#x}"


def _calls(lifter, events) -> list[tuple[int, str, list[tuple[str, int]]]]:
    return [(e["site"], call_target(lifter, e),
             [(r, e["args"][i][1]) for i, (r, _v) in enumerate(e["args"])])
            for e in events if e.get("kind") == "call"]


def emit_walk(lifter, root: int, vip_slot: int | None = None,
              handlers: int = 900, steps: int = 400_000
              ) -> tuple[list[IrBlock], str]:
    """Walk `root` and return one `IrBlock` per handler execution.

    The slicing is exact rather than statistical, the same way
    `Recoverer.guest_operations` slices: `Lifter.events` and `Lifter.dispatches` are
    append-only, so a mark taken at each dispatch boundary makes the events since the
    previous mark exactly the effect of the handler dispatched there.

    `vip_slot` is the offset of the vip in the frame, if the caller has measured it
    (`Recoverer.vip_slot`); when it is `None` the vip is reported as `0` and the key
    is still unique on the handler.
    """
    blocks: list[IrBlock] = []
    ip, at, seen, stop = root, 0, 0, None
    ev_mark, instr_mark = 0, 0
    cur: IrBlock | None = None
    a = lifter.a

    def vip_now() -> int:
        if vip_slot is None:
            return 0
        rbp = a.as_const(lifter.m.reg("rbp", 64))
        if rbp is None:
            return 0
        v = lifter.m.load(a.const(rbp + vip_slot, W64), 8, -1, lifter.img.image_u8)
        c = a.as_const(v)
        return c if c is not None else 0

    while at < steps and len(blocks) < handlers:
        kind, payload = lifter.step(ip)
        at += 1
        if kind != "next":
            stop = payload
            break
        ip = payload
        if len(lifter.dispatches) <= seen:
            continue
        seen = len(lifter.dispatches)
        _site, dest = lifter.dispatches[-1]
        if cur is not None:
            cur.calls = _calls(lifter, lifter.events[ev_mark:])
            cur.instructions = at - instr_mark
            cur.exit_kind = "dispatch"
            cur.successor = dest
            blocks.append(cur)
        ev_mark, instr_mark = len(lifter.events), at
        cur = IrBlock(handler=dest, vip=vip_now())
    if cur is not None:
        cur.calls = _calls(lifter, lifter.events[ev_mark:])
        cur.instructions = at - instr_mark
        cur.exit_kind = stop.kind if stop else "budget"
        blocks.append(cur)
    return blocks, (stop.kind if stop else "budget")


def render_block(lifter, blk: IrBlock) -> list[str]:
    """The block as text, laid out like the reference's `lower` output."""
    a = lifter.a
    out = [f"{blk.key}:"]
    for _site, name, args in blk.calls:
        rendered = ", ".join(f"{r}={render(a, v, limit=8)}" for r, v in args)
        out.append(f"    call {name}({rendered})")
    out.append(f"    exit: {blk.exit_kind}"
               + (f" -> {blk.successor:#x}" if blk.successor else ""))
    return out


def guest_block_leaders(cfg) -> list:
    """The blocks that start a *guest* basic block.

    The engine's block is one handler execution and ends at every dispatch; a guest
    basic block ends at a decision. So a guest block is a chain of handler blocks
    joined by ordinary one-way dispatches, and the chains are counted by their
    leaders -- the standard basic-block definition, applied to the recovered graph:

    * the entry block;
    * any block with other than exactly one predecessor;
    * any successor of a block that is not a plain one-way dispatch.

    The first attempt at this grouped the recovery's blocks in the order they came
    out of the worklist and cut on `b.calls or b.kind != "branch"`, which produced
    **149 runs of length 1** -- it cut everywhere and explained nothing, because a
    call is not a decision and worklist order is not control flow. This walks the
    graph instead.

    The prediction it exists to test: the reference prints **50** blocks for
    `0x14015fc2e` where this engine recovers 149 handler blocks, and ~3 handlers per
    guest block is what a 50-against-149 difference looks like.
    """
    leaders = []
    for key in cfg.order:
        b = cfg.blocks.get(key)
        if b is None:
            continue
        if key == cfg.entry or len(b.preds) != 1:
            leaders.append(key)
            continue
        pred = cfg.blocks.get(b.preds[0])
        if pred is None or pred.kind != "jump":
            leaders.append(key)
    return leaders


def guest_blocks(blocks: list[IrBlock], stop_at_call: bool = False) -> list[list[int]]:
    """Group a *linear* block list into runs. Kept for the walk-based emitter.

    This is the grouping that does **not** work for a recovered graph: it has no
    control flow to follow, so it can only cut on local properties, and on
    `0x14015fc2e` that gave 149 runs of length 1. Use `guest_block_leaders` for a
    `Cfg`; this remains only for a single linear walk, where a run really is a run.
    """
    groups: list[list[int]] = []
    run: list[int] = []
    for i, b in enumerate(blocks):
        run.append(i)
        if stop_at_call and b.calls:
            groups.append(run)
            run = []
        elif b.exit_kind != "dispatch":
            groups.append(run)
            run = []
    if run:
        groups.append(run)
    return groups
