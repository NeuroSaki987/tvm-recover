"""Turning an obfuscated value back into the guest operation it is.

The values the engine recovers for guest registers have a median DAG size of
**355 nodes** and depend syntactically on flags, loads and call results that
mostly do not matter: `probe/influence` measures that only **18.4 %** of those
leaves influence the answer at all, and for the flag leaves it is **6 %**. A value
that is a 355-node tangle over flags and constants is, semantically, `rcx + 0x10`.

So the job here is to *identify* the function, not to rewrite the DAG. The
identification is done by **building candidates in the arena and evaluating them**,
never by pattern-matching the DAG: for a value `v` known to be a function of one
leaf `x`, candidates are constructed from one sample and the answer comes from
comparing `v` against each candidate through `Arena.eval_exact` on all the other
samples.

Building the candidate rather than computing it in Python is deliberate. Python
arithmetic on masked ints is not the engine's arithmetic -- widths, `sext`, `trunc`
and the flag rules all differ -- so a form verified in Python would be a claim about
Python. Both sides go through the same evaluator, so agreement is agreement about
the thing being claimed.

A form is reported only when it agrees on **every** sample beyond the one used to
solve it. One sample proves nothing and is never reported as a match; the sample
count travels with the answer so a caller can see how much checking stands behind
it.

**What this is not.** It is not a soundness proof. For a 64-bit function, agreement
on 64 samples is strong evidence and not a theorem, and that is stated rather than
implied. It is also deliberately narrow: it identifies a value only when the value
depends on exactly one leaf, so a genuinely two-register operation is left alone
rather than approximated.
"""

from __future__ import annotations

import random

from .expr import Arena

# Widths a field might be narrowed to, for the byte/word extraction forms.
NARROW = (8, 16, 32)

# Values a leaf is varied over when measuring influence. Zero and all-ones are the
# boundaries bitwise identities live on; the rest poke individual bits, which is
# where a masked merge stops resembling a plain copy.
MUTATIONS = (0, (1 << 64) - 1, 1, 2, 0x10, 0xFF,
             0x8000_0000_0000_0000, 0x5555_5555_5555_5555)


def _const(arena: Arena, value: int, width: int) -> int:
    return arena.const(value & ((1 << width) - 1), width)


def candidates(arena: Arena, x: int, width: int, xs: list[int], ys: list[int]):
    """Candidate nodes equivalent to a function of `x`, labelled.

    **Constants are solved from the whole sample set, not from one observation**,
    and getting that wrong is why the first version of this identified nothing
    larger than nine nodes. For `y = x + K` or `y = x ^ K` one sample is enough --
    both operations are invertible in K. For `y = x & K` it is not: the sample
    `(0x12345678, 0x12345678)` is equally consistent with `K = 0x12345678` and with
    `K = 0xffffffff`, and solving from it picks the wrong one and then fails every
    remaining check. The mask has to be read off the whole set:

    * `x & K` -- bits that are never set in any `y` are zero in `K`, so `K` is the
      **union** of the observed results;
    * `x | K` -- bits that are always set in every `y` are one in `K`, so `K` is
      their **intersection**.

    A family still needs a second parameter that no single expression determines
    (`(x & K) | D` with both unknown) is solved from the union and the intersection
    jointly, which is exact for the shape it names and simply fails verification
    otherwise.
    """
    m = (1 << width) - 1
    x0, y0 = xs[0], ys[0]
    k_add, k_sub = (y0 - x0) & m, (x0 - y0) & m
    k_xor = (y0 ^ x0) & m
    k_and = 0
    for y in ys:
        k_and |= y
    k_or = m
    for y in ys:
        k_or &= y

    yield "x", x
    yield "~x", arena.un("not", x, width)

    def add(a: int, b: int) -> int:
        return arena.bin("add", a, b, width)

    def sub(a: int, b: int) -> int:
        return arena.bin("sub", a, b, width)

    def band(a: int, b: int) -> int:
        return arena.bin("and", a, b, width)

    def bor(a: int, b: int) -> int:
        return arena.bin("or", a, b, width)

    def bxor(a: int, b: int) -> int:
        return arena.bin("xor", a, b, width)

    def shl(a: int, k: int) -> int:
        return arena.bin("shl", a, arena.const(k, 64), width)

    def shr(a: int, k: int) -> int:
        return arena.bin("shr", a, arena.const(k, 64), width)

    yield f"x + {k_add:#x}", add(x, _const(arena, k_add, width))
    yield f"x - {k_sub:#x}", sub(x, _const(arena, k_sub, width))
    yield f"x & {k_and:#x}", band(x, _const(arena, k_and, width))
    yield f"x | {k_or:#x}", bor(x, _const(arena, k_or, width))
    yield f"x ^ {k_xor:#x}", bxor(x, _const(arena, k_xor, width))

    # `(x op K1) op K2` -- the shape a masked merge or a range check leaves behind.
    masked = band(x, _const(arena, k_and, width))
    yield f"(x & {k_and:#x}) + {k_add:#x}", add(masked, _const(arena, k_add, width))
    yield f"(x & {k_and:#x}) | {k_or:#x}", bor(masked, _const(arena, k_or, width))
    yield f"(x & {k_and:#x}) ^ {k_xor:#x}", bxor(masked, _const(arena, k_xor, width))
    yield f"(x & {k_and:#x}) + {k_or:#x}", add(masked, _const(arena, k_or, width))
    yield f"(x & {k_and:#x}) - {k_add:#x}", sub(masked, _const(arena, k_add, width))

    for k in range(min(width, 64)):
        yield f"x << {k}", shl(x, k)
        yield f"x >> {k}", shr(x, k)
        shifted = shr(x, k)
        yield f"(x >> {k}) & {k_and:#x}", band(shifted, _const(arena, k_and, width))
        yield f"(x >> {k}) | {k_or:#x}", bor(shifted, _const(arena, k_or, width))

    # `lea`-shaped forms: a scaled register plus a constant. Here the constant is
    # derived from the *residual over every sample* and only survives when that
    # residual is the same one everywhere -- so it is solved and checked at once,
    # rather than solved from one observation and hoped for. This is the family the
    # unidentified values were measured to be in: `size 4, x = 0 -> 0x10` is
    # `x * 2 + 0x10`, and one sample cannot tell that from `x + K`.
    for k in range(1, min(width, 64)):
        sh = shl(x, k)
        cs = {(y - ((xx << k) & m)) & m for xx, y in zip(xs, ys)}
        if len(cs) == 1:
            c = cs.pop()
            yield f"(x << {k}) + {c:#x}", add(sh, _const(arena, c, width))
        cs = {y & (~((xx << k) & m) & m) for xx, y in zip(xs, ys)}
        if len(cs) == 1:
            c = cs.pop()
            yield f"(x << {k}) | {c:#x}", bor(sh, _const(arena, c, width))

    # Multiplication, solved exactly when the sample makes it solvable: with an odd
    # `x0` the factor is `y0 * x0^-1` in modular arithmetic.
    if x0 & 1:
        k = (y0 * pow(x0, -1, 1 << width)) & m
        yield f"x * {k:#x}", arena.bin("mul", x, _const(arena, k, width), width)

    # Field extraction and extension: `movzx`/`movsx` on a byte or word, at any
    # offset. This is where an address computation usually comes from.
    for w in NARROW:
        if w > width:
            continue
        yield f"zext{w}(x)", arena.zext(arena.trunc(x, w), width)
        yield f"sext{w}(x)", arena.sext(arena.trunc(x, w), width)
        for byte in range(min(width // 8, 8)):
            field = band(shr(x, 8 * byte), arena.const((1 << w) - 1, width))
            if field == x:
                continue
            yield f"zext{w}(x >> {8 * byte})", arena.zext(arena.trunc(field, w), width)
            yield f"sext{w}(x >> {8 * byte})", arena.sext(arena.trunc(field, w), width)


def bitwise_form(arena: Arena, value: int, leaf: int, base_env: dict[int, int],
                 base_points: int = 3, seed: int = 0x5A17):
    """Derive `(x & K) ^ C` from the value itself, when that is what it is.

    The candidate list in `candidates` guesses a shape and solves its constants.
    This does the opposite: it **measures which input bit each output bit tracks**.
    Flip input bit `i` of `x` at a few base points and record which output bits
    move; if every output bit moves for at most one input bit, the function has to
    be `(x & K) ^ C` and both `K` and `C` fall out of the measurement rather than
    being guessed:

    * `K` is the set of input bits that any output bit tracks;
    * `C` is `f(x)` at a point where every bit of `K` is clear, because there each
      output bit is its constant term.

    Not every value is bitwise -- a carry chain makes one output bit depend on
    several input bits, and those are rejected rather than approximated. The
    point is coverage: a list of shapes has to contain the right one, whereas a
    derivation covers every member of the family at once. `(x & K) | D`, `~x`,
    `x & 0xffffffff` and `x` are all the same family, and the list found them one
    spelling at a time.

    Returns `(node, label)` or `(None, None)`.
    """
    width = arena.width(leaf)
    out_width = arena.width(value)
    r = random.Random(seed)
    env = dict(base_env)
    mask = (1 << width) - 1

    points = [0, mask, r.getrandbits(width)][:max(base_points, 1)]
    while len(points) < base_points:
        points.append(r.getrandbits(width))

    tracks: list[set[int]] = [set() for _ in range(out_width)]
    ys_at: dict[int, int] = {}
    for x in points:
        env[leaf] = x
        base, exact = arena.eval_exact(value, env)
        if not exact or base is None:
            return None, None
        ys_at[x] = base
        for i in range(width):
            env[leaf] = x ^ (1 << i)
            got, exact = arena.eval_exact(value, env)
            if not exact or got is None:
                return None, None
            diff = got ^ base
            j = 0
            while diff:
                if diff & 1:
                    tracks[j].add(i)
                diff >>= 1
                j += 1

    if any(len(t) > 1 for t in tracks):
        return None, None

    k = 0
    for t in tracks:
        for i in t:
            k |= 1 << i
    c = ys_at[points[0] & ~k] if (points[0] & ~k) in ys_at else None
    if c is None:
        env[leaf] = points[0] & ~k
        c, exact = arena.eval_exact(value, env)
        if not exact or c is None:
            return None, None

    node = arena.bin("and", leaf, _const(arena, k, width), width)
    label = f"x & {k:#x}" if k != mask else "x"
    if c:
        node = arena.bin("xor", node, _const(arena, c, out_width), out_width)
        label = f"({label}) ^ {c:#x}"
    if arena.width(node) != out_width:
        return None, None
    return node, label


def verify(arena: Arena, value: int, node: int, leaf: int, base_env: dict[int, int],
           samples: int = 64, seed: int = 0xFEED) -> int:
    """How many *held-out* samples `node` agrees with `value` on.

    Used as the gate on every claim: a form is only reported when this returns the
    full sample count. The seed is separate from the one used to derive the form, so
    agreement is not the derivation checking itself.
    """
    width = arena.width(leaf)
    xs = samples_for(width, samples, seed, base_env.get(leaf, 0))
    env = dict(base_env)
    for x in xs:
        env[leaf] = x
        want, e1 = arena.eval_exact(value, env)
        got, e2 = arena.eval_exact(node, env)
        if not e1 or not e2 or want != got:
            return 0
    return len(xs)


def samples_for(width: int, n: int, seed: int, x0: int):
    """`x0` plus `n-1` further values, weighted toward what bitwise forms turn on."""
    r = random.Random(seed)
    mask = (1 << width) - 1
    out = [x0]
    edge = [0, 1, mask, mask ^ 1, 0x80, 0xFF00, 0xFFFF_FFFF, 0x8000_0000_0000_0000]
    for e in edge:
        if len(out) < n:
            out.append(e & mask)
    while len(out) < n:
        out.append(r.getrandbits(width))
    return out


def identify(arena: Arena, value: int, leaf: int, base_env: dict[int, int],
             samples: int = 64, seed: int = 0x1234_5678):
    """`(node, label, checked)` when `value` is a simple function of `leaf`.

    Two strategies, and the order matters. The **direct derivation**
    (`bitwise_form`) measures the function rather than guessing a shape, so it
    covers a whole family at once; the **candidate list** is tried only if that
    fails, and covers arithmetic and field-extraction shapes the derivation cannot
    express.

    Every form is then gated on `verify`, which scores it against `value` on a
    **held-out sample set with a different seed**. A form solved and checked on the
    same samples is checking itself; the gate is what makes the answer a claim about
    the function rather than about the samples.

    `checked` is the number of held-out samples the surviving form agreed on, or `0`
    when nothing survived -- so a failed identification can never be mistaken for a
    verified one.
    """
    width = arena.width(leaf)
    xs = samples_for(width, samples, seed, base_env[leaf] if leaf in base_env else 0)
    ys: list[int] = []
    env = dict(base_env)
    for x in xs:
        env[leaf] = x
        got, exact = arena.eval_exact(value, env)
        if not exact or got is None:
            return None, None, 0
        ys.append(got)

    node, label = bitwise_form(arena, value, leaf, base_env, seed=seed ^ 0x9E37)
    if node is not None:
        checked = verify(arena, value, node, leaf, base_env,
                         samples=samples, seed=seed ^ 0x1111)
        if checked:
            return node, label, checked

    width_out = arena.width(value)
    for label, node in candidates(arena, leaf, width, xs, ys):
        if arena.width(node) != width_out:
            continue
        if all(_eq(arena, value, node, leaf, env, x, y) for x, y in zip(xs, ys)):
            checked = verify(arena, value, node, leaf, base_env,
                             samples=samples, seed=seed ^ 0x2222)
            if checked:
                return node, label, checked
    return None, None, 0


def _eq(arena: Arena, value: int, node: int, leaf: int, env: dict[int, int],
        x: int, y: int) -> bool:
    env[leaf] = x
    got, exact = arena.eval_exact(node, env)
    return bool(exact) and got == y


def profile_consistent(arena: Arena, value: int, leaf: int, samples: int = 32,
                       envs: int = 3, seed: int = 0xB0B0):
    """Evaluate `f(x)` under several independent assignments of the *other* leaves.

    The premise of every identification is that the value is a function of `leaf`
    alone once the rest are pinned -- measured by influence, which varies one leaf
    at a time and can therefore miss a dependence that only shows under a joint
    change. This re-measures that premise directly: if `f` really is a function of
    `leaf` given fixed others, the row for each environment must be *identical*, and
    any disagreement means the influence measurement was incomplete and the
    identification rests on nothing.

    Returns `(rows, agree)`.

    **The first version of `agree` was wrong and it fired on 134 of 134 values.**
    It read `all(v == rows[0][i] for v in row)` -- comparing *every* element of each
    row against the i-th element of row 0, which is false unless each row is
    constant. The rows were in fact identical. `probe/premise` caught it by printing
    the rows next to the verdict and by testing the alternative explanation first
    (the influence set is stable across four independent bases, 195 of 195), so the
    fault had to be here. A verdict that does not print the evidence for itself is a
    verdict that cannot be checked.
    """
    width = arena.width(leaf)
    xs = samples_for(width, samples, seed, 0)
    rows: list[list[int | None]] = []
    for e in range(envs):
        r = random.Random(seed + 0x1000 * e)
        env = {lf: r.getrandbits(arena.width(lf)) for lf in arena.leaves(value)}
        row: list[int | None] = []
        for x in xs:
            env[leaf] = x
            got, exact = arena.eval_exact(value, env)
            row.append(got if exact else None)
        rows.append(row)
    agree = all(row[i] == rows[0][i] for row in rows for i in range(len(xs)))
    return rows, agree


def influence_set(arena: Arena, value: int, mutations, seed: int,
                  extra_random: int = 6) -> set[int] | None:
    """Leaves whose mutation moves `value`, measured from a fresh base assignment.

    **The mutation values matter, and getting that wrong produced a wrong result
    that survived a reproducibility check.** The first version varied each leaf over
    eight fixed values -- `0`, all-ones, `1`, `2`, `0x10`, `0xFF`, a high bit, an
    alternating pattern -- on the reasoning that they are the boundaries bitwise
    identities live on. They are, and they are not enough: a leaf whose only effect
    is through, say, a byte the fixed set never varies in combination with the base
    assignment is reported as inert, reproducibly, from every seed. Reproducibility
    is not validity, and this measurement had the first without the second.

    `probe/influence`'s premise check is what caught it: re-evaluating the same `x`
    under independent *random* assignments of the other leaves disagreed on **134 of
    134** values that this function had called single-leaf. The fixed mutation set is
    kept, because boundary values are genuinely good at exposing bitwise effects,
    and `extra_random` fresh values per leaf are added, because they are what expose
    everything else.

    `None` when the evaluator declines. An empty set means the value is constant,
    which a caller should confirm by re-randomising the whole environment rather
    than trusting one base.
    """
    leaves = arena.leaves(value)
    if not leaves:
        return set()
    r = random.Random(seed)
    env = {lf: r.getrandbits(arena.width(lf)) for lf in leaves}
    base, exact = arena.eval_exact(value, env)
    if not exact or base is None:
        return None
    out: set[int] = set()
    for lf in leaves:
        w = arena.width(lf)
        keep = env[lf]
        tries = [m & ((1 << w) - 1) for m in mutations]
        tries += [r.getrandbits(w) for _ in range(extra_random)]
        for m in tries:
            env[lf] = m
            got, ex = arena.eval_exact(value, env)
            if not ex or got != base:
                out.add(lf)
                break
        env[lf] = keep
    return out
