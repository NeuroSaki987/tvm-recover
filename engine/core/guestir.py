"""Rendering recovered values as guest operations.

The census in `lift64.py` records *which* values are guest state and which writes update
it; identification in `canon.py` says what function a value is. This module turns those
into text an analyst reads, and it lives in `core` rather than in a probe because that
text is the deliverable: a capability that exists only inside a probe is not one the
engine has.

Three things it does that the earlier probe-only versions got wrong, kept here because
each was a wrong rule rather than a typo:

* **substitute, then identify** -- a value written to a guest home is expressed by
  replacing the leaf the census marked as guest state with a placeholder naming that
  register, and identifying the result. Verifying the *substituted* form on held-out
  samples is what makes the printed form a claim rather than a rendering.
* **name the role, not the shape** -- the flags word is recognised because it is going
  into the `rflags` slot, not because it looks like a pack. A rule keyed on shape is a
  guess about how a value was built; a rule keyed on its role is a statement about what
  it is. Round 23 needed three attempts to learn this; round 9 (`H_*`-only) and round 15
  (state versus carrying) were the same mistake.
* **name what it cannot identify** -- an unidentifiable value is reported as an explicit
  opaque with the reason and the leaf count, never printed as a raw DAG. A reader must be
  able to tell "this has no guest form" from "this has a form I did not print".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .canon import MUTATIONS, identify, influence_set
from .expr import render as render_expr

SAMPLES = 64


def leaf_name(arena, leaf: int) -> str:
    """A leaf named as the analyst sees it, not as its operator is spelled.

    `init` alone is useless in a line that says `guest rbx <- init -> x`: it hides *which*
    initial register. `load` has the same problem for the same reason, since its second
    field is an address node rather than a name.
    """
    n = arena.node(leaf)
    op = n[0]
    if op == "opaque":
        return n[1]
    if op == "init":
        return f"in.{n[1]}"
    if op == "load":
        addr = arena.as_const(n[1])
        shown = f'{addr:#x}' if addr is not None else f'n{n[1]}'
        return f'load{arena.width(leaf)}[{shown}]@m{n[3]}'
    if op == "param":
        return f"param.{n[1]}.{n[2]}"
    return op


def _only_leaf(arena, value):
    leaves = arena.leaves(value)
    return leaves[0] if len(leaves) == 1 else None


def render_value(arena, value: int, guest_values: dict[int, str],
                 dest: str | None = None, samples: int = SAMPLES) -> str:
    """A recovered value as text: a verified form, or an explicit opaque.

    `guest_values` maps a value node to the guest register it is; `dest` is the guest
    register this value is being written to, which is what decides the flags case.
    """
    c = arena.as_const(value)
    if c is not None:
        return f"= {c:#x}"

    infl = influence_set(arena, value, MUTATIONS, 0x5EED_0001)
    if infl:
        names = sorted(leaf_name(arena, n) for n in infl)
        flags = [n for n in names if n.startswith(("flag_", "init_"))]
        # **The flags word is identified by its ROLE, not by the shape of its inputs.**
        #
        # The rule here used to demand `len(flags) >= 2`, and round 40 measured what that
        # refused: of 29 refused flags writes on the benchmark root, **11 are a single guest
        # flag bit** (`flag_af` alone, written straight into the flags slot) and were reported
        # as "not guest state" -- which is false, since a guest flag bit is guest state -- and
        # 18 depend on a call's return value.
        #
        # The destination names the role, so the rule does not need to guess it. What it still
        # has to decide is **how much of the content is known**, and that is what the opacity
        # count must keep measuring: a value assembled from flag bits is identified, a value
        # taken from a call return is not. So the 18 stay opaque -- and say what they are --
        # while the 11 become named.
        #
        # This is the same lesson the module docstring records from round 23 (name the role, not
        # the shape), which had never been applied to this rule because this rule was written
        # before it.
        if dest == "rflags":
            others = [n for n in names if n not in flags]
            # **`len(flags) >= 2` is the right test for a reconstruction, and dropping it cost
            # 10 lines.** The first version of this fix routed *any* case with a non-flag input
            # into the opaque branch, which took the 10 five-flag-bit reconstructions -- named
            # correctly before, table load and all -- and refused them, gaining 11 and losing
            # 10. The round-6 note above says why they must stay nameable: the reconstruction
            # reads a table, so the influential set is flags **plus** a load, and a rule
            # demanding only flags cannot match the thing it was written for.
            #
            # So the discriminator is not "are there other inputs" but **"is this a
            # reconstruction from flag bits, or is it taken from an unidentified source"**:
            # two or more flag bits is a reconstruction whatever else it reads; a single flag
            # bit alone is that bit; anything with fewer than two flag bits *and* another input
            # is taken from elsewhere and its content is not known.
            if flags and not others:
                return (f"rflags from {len(flags)} guest flag bit"
                        f"{'s' if len(flags) != 1 else ''}: " + ", ".join(flags))
            if len(flags) >= 2:
                return ("rflags assembled from " + ", ".join(flags)
                        + f"   ({len(flags)} flag bits, +{len(others)} other input"
                        + f"{'s' if len(others) != 1 else ''})")
            if others and not flags:
                return (f"(opaque: rflags taken from {', '.join(others[:3])} -- "
                        f"not identified)")
            if others:
                return (f"(opaque: rflags taken from {', '.join(others[:3])} and "
                        f"{len(flags)} flag bit"
                        f"{'s' if len(flags) != 1 else ''} ({', '.join(flags)}) -- "
                        f"not identified)")
            return "rflags"

    marked = [lf for lf in arena.leaves(value) if lf in guest_values]
    if marked:
        if len(marked) != 1:
            # The unary recognizer cannot simplify this value, but the DAG
            # still expresses its operations exactly. Do not call that unknown.
            from .lower import Lowering
            lower = Lowering(arena, {})
            result = lower.name(value)
            if lower.truncated:
                return f"(opaque: expression depth limit; {len(marked)} guest leaves)"
            return "expr: " + "; ".join(lower.lines + [f"result {result}"])
        lf = marked[0]
        guest = guest_values[lf]
        ph = arena.opaque(f"guest_{guest}", arena.width(lf))
        sub = arena.substitute(value, {lf: ph})
        sub_infl = influence_set(arena, sub, MUTATIONS, 0x5EED_0001)
        if sub_infl != {ph}:
            return (f"(opaque: "
                    f"{0 if sub_infl is None else len(sub_infl)} influential leaves)")
        _n, label, checked = identify(arena, sub, ph,
                                      {x: 0 for x in arena.leaves(sub)},
                                      samples=samples)
        if label is None:
            return f"(opaque: no form matched) from guest {guest}"
        return f"{label}   (verified on {checked}; over guest {guest})"

    return _render_unmarked(arena, value, samples)


def _render_unmarked(arena, value: int, samples: int) -> str:
    """A value that is not guest state: name the single leaf it depends on, or say why."""
    leaf = _only_leaf(arena, value)
    if leaf is None:
        return f"(opaque: not guest state; {len(arena.leaves(value))} leaves)"
    nm = leaf_name(arena, leaf)
    ph = arena.opaque(f"x_{nm}", arena.width(leaf))
    sub = arena.substitute(value, {leaf: ph})
    infl = influence_set(arena, sub, MUTATIONS, 0x5EED_0001)
    if infl != {ph}:
        return f"(opaque: not guest state; {nm} not the only influence)"
    _n, label, checked = identify(arena, sub, ph,
                                  {x: 0 for x in arena.leaves(sub)},
                                  samples=samples)
    if label is None:
        return f"(opaque: not guest state; {nm}, no form matched)"
    return f"{label}   (verified on {checked}; over {nm} -- not guest state)"


def render_address(arena, value: int, guest_values: dict[int, str],
                   limit: int = 24, width: int = 160) -> str:
    """A guest address, printed in guest register terms. **Printed, not identified.**

    `identify` answers "which function of one variable is this value", and that is the
    right question for a register and the wrong one for an address. An address is
    `base + index*scale + disp`: it usually depends on two or three guest registers, so
    forcing it through a single-leaf form fails -- 40 of 41 guest loads on `0x1401505d5`
    came out as the literal string "3 leaves", which is a line that tells an analyst
    nothing about where in guest memory the read lands.

    The criterion for "this is a guest address" is already definitional (`init_*` present,
    see `Lifter.note_guest_load`), so the rendering is the same fact read out loud rather
    than a second, weaker inference: every `init_<reg>` leaf **is** guest register `<reg>`
    at VM entry, and any leaf the census has marked is named after the register it carries.
    A `load` leaf stays a load, because it is one.

    **Two bounds, and the reason for them is a failed check, not tidiness.** Addresses here
    are not simple: they are the guest base plus a flags-word reconstruction, so the raw
    tree runs to kilobytes. Both `render`'s memo limit and a hard character cap therefore
    apply -- and the first version had neither, with the result that the one `init:rcx`
    leaf sat past the truncation point in 30 of 41 lines and the output named *no* guest
    register at all. So the registers involved are also stated **outside** the tree, in a
    suffix computed from the leaf set: a reader must never lose the answer to "which guest
    register is this" to a truncation of the expression that mentions it.
    """
    c = arena.as_const(value)
    if c is not None:
        return f"{c:#x}"
    subs: dict[int, int] = {}
    regs: list[str] = []
    for lf in arena.leaves(value):
        n = arena.node(lf)
        g = guest_values.get(lf)
        if g is None and n[0] == "init":
            g = n[1]
        if g is not None:
            # **When the census and the definition disagree, both are printed.** For an
            # `init_<host>` leaf the definitional reading is "this is guest register <host>
            # at VM entry", and on `0x1401505d5` the census map disagrees with that on **7 of
            # 15** hosts at the converged budget (11 of 15 on a truncated search -- this root
            # needs 309036 instructions and the probes default to 200000; see FINDINGS.md
            # section 9). The shape is a pair `rcx->rdi`, `rdx->r12` plus five hosts
            # collapsing onto `rsi`, i.e. one context slot written from five host registers
            # over the run.
            #
            # Round 37 measured what causes that: every `init` mark's **first** mechanism is a
            # `context-store`, so the disagreement is the VM filling guest slots from host
            # registers, which is what materialisation is -- not a corrupted map. What remains
            # open is whether that region is *also* used as scratch, in which case a store into
            # `rsi`'s slot would say nothing about guest `rsi`. So the two readings are still
            # not chosen between: the census's name stays primary (it is the name the state
            # lines use, and consistency is worth more than an unverified identity claim) and
            # the register the classifier tested for is printed beside it.
            name = f"guest_{g}"
            if n[0] == "init" and g != n[1]:
                name = f"guest_{g}(init:{n[1]})"
            if name not in regs:
                regs.append(name)
        else:
            name = leaf_name(arena, lf)
        subs[lf] = arena.opaque(name, arena.width(lf))
    txt = render_expr(arena, arena.substitute(value, subs), limit)
    if len(txt) > width:
        cut = txt.rfind(" ", 0, width)
        txt = txt[:cut if cut > width // 2 else width] + " ..."
    # The suffix is built from the same annotated names, and that is load-bearing rather
    # than cosmetic: it is the part that *survives* truncation, and the first version put
    # bare census names here, so on 40 of 41 addresses the register the classifier tested
    # for was the one piece of information the line threw away.
    if regs:
        return f"{txt}   [guest registers: {', '.join(regs)}]"
    return txt


@dataclass
class Emitted:
    """One function's guest semantics, as text."""

    lines: list[tuple[str, str, str]] = field(default_factory=list)
    blocks: int = 0
    blocks_with_effect: int = 0
    blocks_without_effect: int = 0
    blocks_emitted: set = field(default_factory=set)
    opaque: int = 0
    expressions: int = 0
    partial_writes: int = 0
    # **The denominator `opaque` needs, and whose absence produced a wrong number.**
    # `opaque` counts opaque *state* lines; dividing it by `total` (state plus
    # computation) understates the rate, and round 29 reported 0.8-9.6 % that way when
    # the true figures are 1.4-19.2 %. A ratio whose numerator and denominator count
    # different populations is not a measurement, so the populations are both exposed.
    state_lines: int = 0
    suppressed: int = 0
    kept: int = 0
    # How many of `kept` are guest operations that do **not** reach a guest home, i.e. the ones
    # the fold used to drop because it could not tell them from a pure forward. Exposed
    # separately so the densification is a number rather than a claim about the listing.
    forwards_kept: int = 0
    # **Guest memory reads are a third population.** `state_lines` counts writes to
    # guest state and `kept`/`suppressed` count computation; a load is neither, so it
    # gets its own counter rather than being folded into either -- the round-29 error
    # was exactly a ratio between two different populations. `guest_loads` is load
    # *instructions* classified as reading through a guest register; `load_lines` is the
    # distinct address expressions printed, which is the smaller number and the one that
    # decides how long the output is.
    guest_loads: int = 0
    load_lines: int = 0
    blocks_with_loads: int = 0
    # (block label, first line index, one past last line index) into `lines`, so a listing can
    # be rendered with block headers without a second pass over the CFG and without adding
    # header lines to `lines` itself -- adding them there would move every audited count.
    spans: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def opaque_rate(self) -> float:
        """Opaque state lines as a fraction of state lines. The honest ratio."""
        return self.opaque / self.state_lines if self.state_lines else 0.0

    @property
    def total(self) -> int:
        return len(self.lines)


def emit_function(arena, cfg, lifter, folded: bool = True,
                  samples: int = SAMPLES, marks: dict[int, str] | None = None) -> Emitted:
    """Render every recovered block's effect on the guest.

    `folded` suppresses the computations whose value never reaches a guest home -- those
    are the protector moving a value around rather than anything the guest did, and
    including them triples the line count without adding guest semantics.

    `marks` selects which guest map to name values with, defaulting to
    `lifter.guest_values`. It exists so that the map's two relations can be compared on the
    same input -- `Lifter.identity_marks()` against the whole map -- instead of the choice
    being baked in and argued about. See `Lifter.IDENTITY_MARKS`.
    """
    from .canon import MUTATIONS as _M
    gv = lifter.guest_values if marks is None else marks
    blocks = list(cfg.blocks.values())

    home_nodes: set[int] = set()
    for b in blocks:
        for _g, v in b.guest_writes:
            home_nodes.add(v)
            home_nodes.update(arena._descendants_ordered(v))
        for _g, _h, v in b.guest_reg_writes:
            home_nodes.add(v)
            home_nodes.update(arena._descendants_ordered(v))

    out = Emitted(blocks=len(blocks))
    for b in blocks:
        # Forks restore their own provenance. A later sibling's map must not
        # rename values in an earlier block when rendering the whole CFG.
        gv = b.guest_values if marks is None and hasattr(b, 'guest_values') else gv
        _start = len(out.lines)
        rows: list[tuple[str, str, str]] = []
        for g, v in b.guest_writes:
            rows.append(("ctx", g, render_value(arena, v, gv, dest=g, samples=samples)))
        for g, h, v in b.guest_reg_writes:
            rows.append(("home:" + h, g,
                         render_value(arena, v, gv, dest=g, samples=samples)))
        for _step, kind, payload in b.guest_trace:
            if kind == 'part':
                _site, reg, lo, nbytes, value = payload
                rows.append(('partial', reg,
                             f'bits {lo * 8}..{(lo + nbytes) * 8 - 1}: '
                             + render_value(arena, value, gv, samples=samples)))
                out.partial_writes += 1
        comp = []
        n_suppressed = 0
        for d in b.guest_derived:
            # Four fields, not five: the Recoverer strips the step number when it
            # captures the record per block. The extraction from the probe kept the
            # five-field unpack and this is the first time it ran.
            g, host, prev, value = d
            if folded and value not in home_nodes:
                # **A pure forward and a guest operation are not the same thing, and the fold
                # was treating them as one.** A computation whose value is *exactly the value
                # the register already held* (`value == prev`) states nothing: it is the
                # protector moving a guest register through scratch, and suppressing it is
                # right. A computation whose value is something else is a **guest operation**
                # -- `t <- guest_rax & 0xffffffff`, `t <- guest_rcx + 8` -- and dropping it
                # because its result happens not to reach a home is why 139 of 149 blocks on
                # the benchmark root emitted nothing but `none` from a 62822-instruction
                # function. The structural test is exact and costs nothing: `prev` is the node
                # the register held, so equality is identity of values, not a rendering
                # heuristic.
                if value == prev:
                    out.suppressed += 1
                    n_suppressed += 1
                    continue
                out.forwards_kept += 1
            out.kept += 1
            comp.append(("t_" + host, g,
                         render_value(arena, value, gv, dest=None, samples=samples)))
        # **Guest memory reads.** One line per *distinct* address expression per block,
        # with the number of load instructions that made it, because the same expression
        # recurs: `0x1401505d5` reads `[entry_rcx + 0x10]` 105 times as interned-node
        # occurrences, and printing that 105 times says nothing the count does not. (The
        # first version of this comment said "the benchmark root", which is `0x14015fc2e`
        # and reads it 4 times: the 105 belongs to `0x1401505d5`. A figure attached to the
        # wrong root is the error this file's history is mostly made of.) The address goes
        # through `render_address`, which prints it in guest terms instead of asking
        # `identify` for a one-variable form an address does not have.
        loads: list[tuple[str, str, str]] = []
        groups: dict[int, list[int]] = {}
        for site, addr in b.guest_loads:
            groups.setdefault(addr, []).append(site)
        for addr, sites in groups.items():
            txt = render_address(arena, addr, gv)
            where = f"{min(sites):#x}" if len(sites) == 1 else (
                f"{min(sites):#x}..{max(sites):#x} x{len(sites)}")
            loads.append(("ld:" + where, "", "reads [ " + txt + " ]"))
        out.guest_loads += len(b.guest_loads)  # instructions, not lines: grouping loses none
        out.load_lines += len(loads)
        if loads:
            out.blocks_with_loads += 1
        # **Every block is emitted, including the ones with nothing to say.** Round 21's
        # fold quietly removed about 119 blocks from the description, and a reader could
        # not tell a block with no guest effect from a block that was never analysed --
        # the same failure as printing a raw DAG where a form was wanted, and the same
        # rule round 9 established for unidentifiable values: report it, do not omit it.
        out.opaque += sum(1 for kind, _g, text in rows
                          if text.startswith('(') or (kind == 'partial' and ': (opaque:' in text))
        out.expressions += sum(1 for kind, _g, text in rows
                              if text.startswith('expr:') or (kind == 'partial' and ': expr:' in text))
        out.state_lines += len(rows)
        other_effects = [kind for _, kind, _ in b.guest_trace if kind in ('call', 'gstore')]
        if rows or comp or other_effects:
            out.blocks_with_effect += 1
        else:
            out.blocks_without_effect += 1
        if rows or comp or loads:
            out.lines.extend(rows)
            out.lines.extend(comp)
            out.lines.extend(loads)
        else:
            why = (f'{len(other_effects)} ordered call/memory effect(s)' if other_effects else
                   f'no guest effect recorded; {n_suppressed} copy/copies suppressed; '
                   'not proved inert')
            out.lines.append(('events' if other_effects else 'none', b.key.handler.__format__('x'), why))
        out.spans.append((f"{b.key}", _start, len(out.lines)))
    _ = _M
    return out
