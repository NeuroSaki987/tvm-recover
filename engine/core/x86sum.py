"""Bounded interpretation of a native callee, producing a *summary*.

The VM calls helpers that live in this image's own `.text`. Treating those calls
as opaque throws away a return value the engine is holding the code for; treating
them as ordinary inlined control flow -- which was implemented and measured --
throws away the graph, because a native routine's internal `jcc` then becomes a
CFG fork and a native routine is not a virtualized program point. On the benchmark
root that experiment took the graph's reproduction of the reference's transition
sequence from 802 of 802 down to 230 of 802.

This module is the third option, and the reason it is its own module rather than a
branch inside the lifter: it does not extend the CFG at all. It interprets the
callee on the caller's machine to a **summary** -- whatever the callee did, with
`rax` holding its result -- and hands control straight back to the return address.
The callee's internal branches are resolved or the summary is abandoned; they
never become edges.

The contract, which is what makes it safe to call from anywhere:

* it runs **on the caller's machine**, mutating it, because a callee's side
  effects on memory and registers are real and wanted;
* if every step is decidable and the callee reaches its return address, the
  effects are **kept** and the summary is `ok`;
* otherwise the machine, the memory map and the lifter's diagnostic lists are
  **restored exactly**, and the summary reports why it failed, so the caller can
  fall back to what it did before. No value is ever invented.

"What could not be decided" is deliberately broad: a conditional branch on a
symbolic condition, an indirect transfer with a symbolic destination, an
instruction capstone cannot read, a return to somewhere other than the caller, a
nested call that itself fails, or the step budget running out. Each is counted by
reason, because the reason is the interesting part -- a helper that always fails
on one instruction is a lifter gap worth knowing about, and one that fails on a
symbolic branch is genuinely data-dependent.

Three things are deliberately *not* rolled back:

* `arena` nodes. The arena is append-only and nodes are interned; abandoning a run
  cannot un-create them and does not need to.
* `call_count`. Rolling it back would let two different calls receive the same
  `call_ret#N`, and those intern to the *same* opaque node -- two unrelated return
  values silently unified. That is the one thing here that would be unsound.
* `steps`, `trace`. They measure work done, and the work was done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .expr import W64
from .state import LifterCheckpoint


@dataclass
class Summary:
    ok: bool
    reason: str = ""
    steps: int = 0
    # **What the callee wrote, which is the thing callers actually need.** A summary that only says
    # "it returned" leaves the caller to guess the write set, and guessing it as "everything above
    # the pointer argument" is what costs `0x1401687d5` its coverage. Recorded on failure too: a
    # partial interpretation still shows where the callee writes, which is the difference between a
    # bounded rule and an unbounded one.
    stores: list = field(default_factory=list)
    loaded: int = 0


def summarize(L, entry: int, ret_addr: int, limit: int) -> Summary:
    """Interpret `entry` until it returns to `ret_addr`, or give up.

    `L` is the live lifter; the callee runs on `L.m` and its effects are kept on
    success. See the module docstring for the contract.
    """
    a = L.a
    m = L.m
    rsp = m.concrete_rsp()
    if rsp is None:
        return Summary(False, "caller rsp is symbolic")

    snap = LifterCheckpoint(L)
    # A native routine's internal indirect transfers are not VM handler
    # transitions, and recording them would corrupt the transition comparison the
    # whole workspace is judged by. Swap the list out for the duration.
    saved_dispatches = L.dispatches
    L.dispatches = []

    # Record the callee's stores as it makes them. `m.store` may be the census wrapper, so the
    # attribute is saved and restored rather than assumed to be `Machine.store`.
    real_store = m.store

    def _recording_store(addr, value, nbytes, site):
        written.append((addr, nbytes))
        return real_store(addr, value, nbytes, site)

    written: list[tuple] = []
    m.store = _recording_store

    m.set_reg("rsp", a.const(rsp - 8, W64), 64)
    m.store(a.const(rsp - 8, W64), a.const(ret_addr, W64), 8, entry)
    written.clear()  # the return-address push is the harness's, not the callee's

    ip = entry
    steps = 0
    reason = "step budget exhausted"
    ok = False
    try:
        while steps < limit:
            if ip == ret_addr:
                ok = True
                reason = ""
                break
            kind, payload = L.step(ip)
            steps += 1
            if kind == "next":
                ip = payload
                continue
            stop = payload
            if stop.kind == "return":
                # Returned somewhere that is not the caller: an unwound frame, a
                # tail call, or a return address the engine got wrong. Not a
                # summary of this call.
                reason = "returned elsewhere"
            elif stop.kind == "native_branch":
                reason = "symbolic conditional branch"
            elif stop.kind == "dispatch" or stop.kind == "dispatch_call":
                reason = "symbolic indirect transfer"
            elif stop.kind == "bad_target":
                reason = "control target is not executable"
            elif stop.kind == "unreadable":
                reason = f"unreadable instruction at {stop.site:#x}"
            else:
                reason = stop.kind
            break
    finally:
        m.store = real_store
        L.dispatches = saved_dispatches
        if not ok:
            snap.restore(L)

    if not ok:
        return Summary(False, reason, steps, stores=written)
    return Summary(True, "", steps, stores=written)
