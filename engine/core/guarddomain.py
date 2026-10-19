"""Propose a counter bound by bounded search; accept only universal BV proof.

Concrete probes are candidate generation only. Disconnected regions, wraparound
and high-bit counterexamples are covered by the final complete equivalence
query. Missing solver, unsupported expressions, budgets and timeouts refuse the
rule. The result concerns recorded path guards, never guard inductiveness.
"""
from .bitproof import finite_values


def derive_bound(arena, before, after, guards, sample, delta,
                 search_limit=4096, timeout_ms=1000):
    if not guards or sample is None or not 0 <= sample < search_limit or delta <= 0:
        return None
    a = arena
    condition = a.const(1, 8)
    for left, right in guards:
        equal = a.bin('eq', left, right, max(a.width(left), a.width(right)))
        condition = a.bin('and', condition, equal, 8)

    def accepts(value):
        result = a.as_const(a.substitute(condition, {before: a.const(value, a.width(before))}))
        return None if result is None else bool(result)

    if accepts(sample) is not True:
        return None
    low, high = sample, max(sample + 1, 1)
    for _ in range(64):
        if high > search_limit:
            return None
        accepted = accepts(high)
        if accepted is None:
            return None
        if not accepted:
            break
        low, high = high, max(high + 1, high * 2)
    else:
        return None
    while low + 1 < high:
        middle = (low + high) // 2
        accepted = accepts(middle)
        if accepted is None:
            return None
        if accepted:
            low = middle
        else:
            high = middle
    maximum = low + delta
    if maximum >= 1 << a.width(after):
        return None
    candidate = a.bin('ule', after, a.const(maximum, a.width(after)), a.width(after))
    mismatch = a.bin('ne', condition, candidate, 8)
    proof = finite_values(a, mismatch, {}, max_values=2, timeout_ms=timeout_ms)
    if proof.status != 'complete' or proof.values != (0,):
        return None
    return dict(kind='unsigned_output_bound', maximum_output=maximum,
                proof='complete_bitvector_equivalence', checks=proof.checks,
                scope='recorded path guards; entry specialization remains separate',
                conditional=True, loop_proved=False)
