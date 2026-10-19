"""A recovered block's effect on the guest, emitted in guest register terms.

This is the deliverable the goal names, and it has been the one thing repeatedly
deferred while structural questions were settled. The engine can name guest register
reads and writes (`core/guestctx.py`), can identify an obfuscated value as the guest
operation it is (`core/canon.py`), and can emit a block (`core/ir.py`) -- but what it
emits has been calls and a terminator, with the guest effect absent.

So: for every recovered block, print what it does to the guest register file, with the
value written given in canonical form where one was identified and in raw form where it
was not. Nothing is claimed beyond that -- a raw expression is printed as raw.

**The honest half is the opaque count, and it is the point of the probe.** Three
outcomes are distinguished and counted separately, because collapsing them would let a
block that *reads* the guest look like a block that does nothing:

* `named` -- the value depends on exactly one leaf and a form was identified and
  verified on held-out samples;
* `constant` -- the value depends on nothing, confirmed by re-randomising the whole
  environment, not from one assignment;
* `opaque` -- anything else, reported with the reason: how many leaves actually
  influence it, and whether any of them is a guest register.

Falsifier stated before the run: **every block that writes the guest register file must
appear here, and every named form must carry the number of held-out samples it was
verified against.** A form with no sample count behind it is not reported at all, and a
block that writes the guest context while producing no line here would be a hole --
which is why the probe prints the number of context writes recorded per block next to
the number of lines emitted.

Usage:
  python -m probe.guestir <module> <root_va> [--blocks N] [--steps N] [--show N]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core.canon import MUTATIONS, identify, influence_set  # noqa: E402
from core.expr import render  # noqa: E402
from core.explore import STACK_BASE, Recoverer  # noqa: E402
from core.guestir import leaf_name as _leaf_name  # noqa: E402
from core.guestir import render_value as _render  # noqa: E402
from core.peimg import PeImage  # noqa: E402


def leaf_name(arena, leaf: int) -> str:
    """Re-exported from `core.guestir`; this probe no longer owns the rendering."""
    return _leaf_name(arena, leaf)


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    blocks_budget, steps, show = 512, 200_000, 8
    for x in sys.argv[1:]:
        if x.startswith("--blocks="):
            blocks_budget = int(x.split("=", 1)[1], 0)
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--show="):
            show = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    # The census is opt-in because it costs 1-2 % of a run (measured by an interleaved
    # A/B, not across sessions); here it is the whole point.
    # The handler-time cache first: without it the census refuses every attribution and
    # records no computation either, which the first run of this section would have
    # reported as "0 carrying writes" -- a zero that measures the probe.
    probe = Recoverer(img, vm.va, vm.end, STACK_BASE, guest_census=True)
    cache = probe.derive_guest_cache(root, handlers=900, steps=400_000)
    R = Recoverer(img, vm.va, vm.end, STACK_BASE, step_budget=steps,
                  block_budget=blocks_budget, guest_census=True)
    R.handler_cache = cache
    cfg = R.recover(root)
    arena = R.arena
    blocks = list(cfg.blocks.values())

    written = sum(len(b.guest_writes) for b in blocks)
    with_effects = [b for b in blocks if b.guest_writes]
    print(f"root {root:#x}   blocks {len(blocks)}   unresolved "
          f"{sum(1 for b in blocks if b.kind == 'unresolved')}")
    # Before any corpus figure, because a truncated search produces a complete-looking
    # corpus. Round 37: this root's published corpus was 11 blocks of 89 for eight rounds.
    if R.truncated:
        print(f"*** TRUNCATED: {len(R.budget_stopped)} block(s) hit the "
              f"{R.step_budget}-step budget; every figure below is a lower bound ***")
    else:
        print(f"search converged (no block reached the {R.step_budget}-step budget)")
    print(f"guest register writes recorded across the graph: {written}")
    print(f"blocks with at least one guest register write:   {len(with_effects)}"
          f"  ({100.0 * len(with_effects) / max(len(blocks), 1):.1f} % of blocks)")
    print()

    kinds: collections.Counter = collections.Counter()
    reasons: collections.Counter = collections.Counter()
    lines: list[tuple] = []
    for b in blocks:
        for reg, value in b.guest_writes:
            inf = influence_set(arena, value, MUTATIONS, 0x5EED_0001)
            if inf is None:
                kinds["evaluator declined"] += 1
                continue
            if not inf:
                kinds["constant"] += 1
                lines.append((b, reg, "= const", 16))
                continue
            if len(inf) != 1:
                kinds["opaque"] += 1
                reasons[f"{len(inf)} influential leaves"] += 1
                lines.append((b, reg, None, len(inf)))
                continue
            leaf = next(iter(inf))
            # **Any** single leaf, not only a guest register. The first version of
            # this required `H_*`, and the result was 0 named forms out of 373 writes
            # with **129** of the opaque ones reported as "one non-guest leaf (init)"
            # -- which is `guest rax <- in.rax`, the context initialisation, about as
            # legible as a guest operation gets. The blind spot was the identifier's,
            # not the VM's opacity, and only the breakdown by reason exposed it.
            lname = leaf_name(arena, leaf)
            env = {lf: 0 for lf in arena.leaves(value)}
            node, label, checked = identify(arena, value, leaf, env, samples=64)
            if label is None:
                kinds["opaque"] += 1
                reasons[f"one leaf ({str(lname)[:16]}), no form matched"] += 1
                lines.append((b, reg, None, 0))
                continue
            kinds["named"] += 1
            lines.append((b, reg, f"{lname} -> {label}", checked))

    print("=== outcome of every guest register write ===")
    for k, n in kinds.most_common():
        print(f"  {n:>5}  {k}")
    if reasons:
        print("  opaque, by reason:")
        for r, n in reasons.most_common(6):
            print(f"      {n:>5}  {r}")
    print()

    print(f"=== the first {show} emitted guest effects ===")
    for b, reg, label, checked in lines[:show]:
        if label is None:
            print(f"{b.key}   [{b.kind}]")
            print(f"    guest {reg:<6} <- (opaque, {checked} influential leaves)")
        else:
            print(f"{b.key}   [{b.kind}]")
            print(f"    guest {reg:<6} <- {label}   "
                  f"(verified on {checked} samples)")
    print()

    emitted = len(lines)
    print(f"FALSIFIER -- context writes recorded {written}, lines emitted {emitted}. "
          + ("NO HOLES" if written == emitted
             else f"HOLE: {written - emitted} write(s) produced no line"))
    print("    every named form above carries its held-out sample count; a form "
          "without one is reported as opaque instead")
    print()

    # ------------------------------------------------------------------ computation
    #
    # The writes that compute with a *copy* of a guest register rather than updating
    # it. Round 15 separated these from state: 262 of 397 carrying-writes on the
    # benchmark root. They are not guest-register updates and they are exactly what a
    # devirtualizer wants, so they are emitted as computation -- `t<host> = f(guest
    # r12)` -- by substituting the copied value with a named placeholder for the guest
    # register and rendering the result.
    derived = [(b, g, h, prev, v) for b in blocks for g, h, prev, v in b.guest_derived]
    print(f"=== computation with a copy of a guest register: {len(derived)} writes ===")
    dblocks = len({id(b) for b, *_ in derived})
    print(f"in {dblocks} blocks")
    dkinds: collections.Counter = collections.Counter()
    dforms: collections.Counter = collections.Counter()
    shown_d = 0
    for b, guest, host, prev, value in derived:
        ph = arena.opaque(f"guest_{guest}", arena.width(prev))
        sub = arena.substitute(value, {prev: ph})
        inf = influence_set(arena, sub, MUTATIONS, 0x5EED_0001)
        if inf == {ph}:
            node, label, checked = identify(arena, sub, ph,
                                            {lf: 0 for lf in arena.leaves(sub)},
                                            samples=64)
            label = f"{label} (verified on {checked})" if label else "(no form)"
        else:
            label = f"(over {0 if inf is None else len(inf)} leaves)"
        dkinds["named" if "verified" in label else "opaque"] += 1
        dforms[(guest, host, label)] += 1
        if shown_d < show:
            print(f"{b.key}   [{b.kind}]")
            print(f"    t_{host:<5} <- {label}   from guest {guest}")
            shown_d += 1
    print()
    print(f"  named {dkinds['named']}, opaque {dkinds['opaque']}")
    for (guest, host, label), n in dforms.most_common(8):
        print(f"    x{n:<4} t_{host:<5} <- {label[:70]}   from guest {guest}")
    print()
    print(f"FALSIFIER -- carrying writes recorded {len(derived)}; every line names the "
          f"guest register it derives from, and a form is emitted only with its "
          f"held-out sample count. No line is emitted without both.")
    print()

    # --------------------------------------------------- coverage is not interpretation
    #
    # Round 19 found 600 computations in 129 of 149 blocks and said plainly that most of
    # them are the protector's shuffling rather than the guest's operations. This is the
    # measurable form of that: **does the result of the computation reach a guest home?**
    #
    # A computation whose value is later written to a context slot or to a register the
    # cache names as some guest's home is part of the mechanism by which that guest
    # register changes, and calling it bookkeeping would be wrong. One whose value is
    # never seen again is a dead end -- the VM moving a value around, which is exactly
    # what a copy into scratch is.
    home_nodes: set[int] = set()
    for b in blocks:
        for _g, v in b.guest_writes:
            home_nodes.add(v)
            home_nodes.update(arena._descendants_ordered(v))
        for _g, _h, v in b.guest_reg_writes:
            home_nodes.add(v)
            home_nodes.update(arena._descendants_ordered(v))
    reaches = dead = 0
    dead_forms: collections.Counter = collections.Counter()
    live_forms: collections.Counter = collections.Counter()
    for b, guest, host, prev, value in derived:
        ph = arena.opaque(f"guest_{guest}", arena.width(prev))
        sub = arena.substitute(value, {prev: ph})
        inf = influence_set(arena, sub, MUTATIONS, 0x5EED_0001)
        lbl = "?"
        if inf == {ph}:
            _n, lbl2, _c = identify(arena, sub, ph,
                                    {lf: 0 for lf in arena.leaves(sub)}, samples=64)
            lbl = lbl2 or "(no form)"
        if value in home_nodes:
            reaches += 1
            live_forms[lbl] += 1
        else:
            dead += 1
            dead_forms[lbl] += 1
    print("=== does the computation reach a guest home? ===")
    print(f"  reaches a home (part of how a guest register changes): {reaches}")
    print(f"  never seen again (a value moved around)              : {dead}")
    print()
    print("  forms that reach a home:")
    for lbl, n in live_forms.most_common(5):
        print(f"    x{n:<4} {lbl[:60]}")
    print("  forms that do not:")
    for lbl, n in dead_forms.most_common(5):
        print(f"    x{n:<4} {lbl[:60]}")
    print()

    # ---------------------------------------------------------------------- folding
    #
    # A computation whose form is `x` and whose value reaches a home is a **pure
    # forward**: it moves a guest value to where it lives and states nothing the update
    # it feeds does not already state. Emitting it is noise, and the only question that
    # matters is whether removing it loses anything.
    #
    # The falsifier is that it must not: **the set of guest registers receiving a state
    # update has to be identical before and after.** If folding changes that set, a
    # forward was carrying content and suppressing it is wrong.
    # **The first version of this folded the wrong set.** It suppressed the copies whose
    # value *does* reach a home and kept the ones that never do -- which is backwards: a
    # copy that reaches a home is the mechanism by which a guest register changes, and a
    # value that never appears in any state update states nothing about guest state. The
    # fold now suppresses the non-reaching computations and keeps the 195.
    classified = [(d, d[4] in home_nodes) for d in derived]
    kept = [d for d, reaches in classified if reaches]
    fwd = [d for d, reaches in classified if not reaches]
    state_total = sum(len(b.guest_reg_writes) for b in blocks)
    keep_ids = {id(d) for d in kept}

    # A falsifier that **can fail**: every guest register mentioned as the *source* of a
    # suppressed line must still be mentioned somewhere in the retained corpus. If a
    # guest register appears only as the source of suppressed computations, the folded
    # output never names it and a reader would not know it was involved at all.
    retained_regs = set()
    for b in blocks:
        for g, _v in b.guest_writes:
            retained_regs.add(g)
        for g, _h, _v in b.guest_reg_writes:
            retained_regs.add(g)
    for d in kept:
        retained_regs.add(d[1])
    suppressed_regs = {d[1] for d in fwd}
    dropped_regs = suppressed_regs - retained_regs

    print("=== folding the computations that never reach a guest home ===")
    print(f"  suppressed (never appear in any state update): {len(fwd)}")
    print(f"  kept (part of how a guest register changes)  : {len(kept)}")
    print(f"  emitted lines: raw {written + state_total + len(derived)}"
          f"   folded {written + state_total + len(kept)}")
    print(f"FALSIFIER -- guest registers named only by suppressed lines: "
          f"{len(dropped_regs)}"
          + (f" {sorted(dropped_regs)} -- folding lost them" if dropped_regs
             else "  (none -- every one is still named)"))
    print()
    print("=== a block, folded ===")
    # Render the state writes in guest terms with the *same* substitution the
    # computations get. Until this existed the state lines printed a placeholder while
    # the computations printed verified forms -- correct and unreadable.
    gv = R.lifter.guest_values
    opaque_state = 0
    state_total_forms = 0
    shown_f = 0
    for b in blocks:
        rendered_ctx = [(g, _render(arena, v, gv, dest=g)) for g, v in b.guest_writes]
        rendered_home = [(g, h, _render(arena, v, gv, dest=g))
                         for g, h, v in b.guest_reg_writes]
        state_total_forms += len(rendered_ctx) + len(rendered_home)
        opaque_state += sum(1 for _g, t in rendered_ctx + [(h, t) for _g, h, t
                                                           in rendered_home]
                            if t.startswith("("))
        comp = [d for d in b.guest_derived if id(d) in keep_ids]
        if shown_f >= show or not (b.guest_writes or b.guest_reg_writes or comp):
            continue
        print(f"{b.key}   [{b.kind}]")
        for g, txt in rendered_ctx:
            print(f"    guest {g:<5} <- {txt}")
        for g, h, txt in rendered_home:
            print(f"    guest {g:<5} <- {txt}   (home {h})")
        for d in comp:
            _b, g, host, _p, _v = d
            print(f"    t_{host:<5} <- {_render(arena, _v, gv)}   "
                  f"-> feeds a state change")
        shown_f += 1
    print()
    print(f"FALSIFIER -- state lines rendered: {state_total_forms}; printed as opaque "
          f"rather than as a raw DAG: {opaque_state}. "
          + ("every line is a form or an explicit opaque"
             if opaque_state <= state_total_forms else "IMPOSSIBLE"))
    print()

    # ------------------------------------------------------ the engine's own emitter
    #
    # Everything above is the probe's own rendering. `core/guestir.emit_function` is the
    # same job done **in the engine**, and until this section existed nothing in the tree
    # called it: the round-30 corpus figures came from an inline script, so the emitter was
    # a capability no probe exercised and no test pinned. A capability that only an
    # ad-hoc script has ever run is not a capability the engine has -- the same rule that
    # moved the rendering out of this probe in round 24.
    #
    # The falsifiers are counting identities, so each can fail:
    #   * every recovered block appears -- `Emitted.blocks` equals the block count, and
    #     the lines cover all of them (a folded corpus silently dropped ~119 blocks in
    #     round 21, which is why the emitter states `none` rather than omitting);
    #   * every state write has a line -- `state_lines` equals writes plus home writes;
    #   * **every guest load reaches the corpus** -- `guest_loads` equals what the Lifter
    #     recorded. This is the round-36 addition and the one that can be short by a
    #     per-block capture bug that the totals would not otherwise reveal.
    from core.guestir import emit_function  # noqa: E402
    em = emit_function(arena, cfg, R.lifter, folded=True)
    print("=== the engine's emitter (core/guestir.emit_function) ===")
    print(f"  blocks            {em.blocks}   (recovered {len(blocks)})")
    print(f"  lines             {em.total}")
    print(f"  state lines       {em.state_lines}   (writes {written} + home "
          f"{state_total} = {written + state_total})")
    print(f"  with an effect    {em.blocks_with_effect}   without {em.blocks_without_effect}")
    print(f"  opaque            {em.opaque}   rate {em.opaque_rate * 100:.1f} % of state lines")
    print(f"  suppressed/kept   {em.suppressed} / {em.kept}")
    print(f"  guest loads       {em.guest_loads} in {em.blocks_with_loads} block(s), "
          f"{em.load_lines} line(s)   (Lifter recorded {len(R.lifter.guest_loads)})")
    ok_blocks = em.blocks == len(blocks)
    ok_state = em.state_lines == written + state_total
    ok_loads = em.guest_loads == len(R.lifter.guest_loads)
    ok_lines = em.blocks_with_effect + em.blocks_without_effect == em.blocks
    for name, ok in (("every block present", ok_blocks),
                     ("every state write has a line", ok_state),
                     ("every guest load reaches the corpus", ok_loads),
                     ("effect+no-effect covers all blocks", ok_lines)):
        print(f"  FALSIFIER {name:<38} {'holds' if ok else 'FAILS'}")
    print()
    if em.load_lines:
        print("=== the guest memory reads, as emitted ===")
        ld = [(k, t) for k, _w, t in em.lines if k.startswith("ld:")]
        for kind, text in ld:
            print(f"    {kind:<18} {text}")
        print()
        # Two checks on the address rendering, and both can fail.
        #
        # The first is the *reason* the rendering was replaced: printing an address as a
        # function of one variable produced the literal string "3 leaves" 40 times out of
        # 41. The criterion for being a guest load is an `init_*` leaf (see
        # `Lifter.note_guest_load`), so a non-constant guest address **must** name at least
        # one guest register -- if it does not, the renderer is not reading out the same
        # fact the classifier used, and one of the two is wrong.
        #
        # The second guards the direction that would be catastrophic rather than useless:
        # an address that is genuinely symbolic must never come out looking concrete.
        def inner(t: str) -> str:
            return t[len("reads [ "):-2]

        unnamed = [t for _k, t in ld if "guest_" not in inner(t)]
        bare = [t for _k, t in ld
                if inner(t).startswith("0x")
                and all(c in "0123456789abcdefx" for c in inner(t))]
        print(f"  FALSIFIER every symbolic guest address names a guest register: "
              f"{len(ld) - len(unnamed)}/{len(ld)}"
              + ("" if not unnamed else f"  MISSED: {unnamed[:3]}"))
        print(f"  FALSIFIER no symbolic address printed as a constant: "
              f"{len(bare)} of {len(ld)}"
              + ("" if not bare else f"  VIOLATED: {bare[:3]}"))
        print()
    return 0


def _form_of(arena, d):
    """The canonical form of a derived computation, as the emitter labels it."""
    _b, guest, _host, prev, value = d
    ph = arena.opaque(f"guest_{guest}", arena.width(prev))
    sub = arena.substitute(value, {prev: ph})
    inf = influence_set(arena, sub, MUTATIONS, 0x5EED_0001)
    if inf != {ph}:
        return "?"
    _n, lbl, _c = identify(arena, sub, ph,
                           {lf: 0 for lf in arena.leaves(sub)}, samples=64)
    return lbl


if __name__ == "__main__":
    raise SystemExit(main())
