"""Induction and termination of a serialized, deterministic register projection.

This is deliberately not native-loop recognition. Matching measured control
boundaries proposes a recurrence; SMT proves its guard domain, inductive entry
specializations and ranking function. Memory and discarded outputs are outside
this certificate. Unknown/unsupported/timeout never counts as a proof.
"""
import hashlib
import json

from .bitproof import finite_values
from .expr import Arena


def _translate(program, roots):
    a, refs = Arena(), {}
    for ref in program.dependencies(roots):
        n = program.nodes[ref]
        op, width = n['op'], n['width']
        args = [refs[r] for r in n['args']]
        if op == 'param':
            value = a.init_reg(n['reg'])
        elif op == 'c':
            value = a.const(n['value'], width)
        elif op in ('bin', 'un'):
            value = a._add((op, n['operator'], *args, width))
        elif op in ('trunc', 'zext', 'sext', 'sel'):
            value = a._add((op, *args, width))
        else:
            raise ValueError('projection contains ' + op)
        refs[ref] = value
    return a, refs


def prove(rows, programs, timeout_ms=1500):
    result = dict(loop_proved=False, scope='serialized_register_subsystem',
                  full_native_loop_proved=False, full_guest_state=False,
                  conditional=True)

    def refuse(reason):
        return dict(result, reason=reason)

    pairs = list(zip(rows, programs.values()))
    if len(rows) < 2 or len(rows) != len(programs):
        return refuse('two or more adjacent measured cuts required')
    if timeout_ms <= 0:
        return refuse('solver budget exhausted')
    for index, (row, p) in enumerate(pairs):
        if (not row.get('ok') or row.get('entry_ip') != hex(row.get('next_ip', -1))
                or row.get('entry_ip') != rows[0].get('entry_ip')
                or row.get('end', 0) <= row.get('start', 0)
                or (index and rows[index - 1]['end'] != row['start'])):
            return refuse('cuts do not share an adjacent recurrent control boundary')
        if p.verify():
            return refuse('invalid transfer graph')
    closures = [p.closure(p.entry) for _, p in pairs]
    if not closures[0]['signature'] or len({c['signature'] for c in closures}) != 1:
        return refuse('guarded closed register transfers differ')
    closed = closures[0]['registers']
    candidates = []
    for reg, relation in rows[0].get('relations', {}).items():
        update = relation.get('derived_update', {})
        bound = relation.get('path_guard_equivalence', {}).get('maximum_output')
        bits = update.get('bits', 0)
        if (reg in closed and update.get('delta') == 1 and bits in (8, 16, 32, 64)
                and isinstance(bound, int) and 0 < bound < (1 << bits)):
            candidates.append((reg, bits, bound))
    if len(candidates) != 1:
        return refuse('a unique bounded unit counter candidate is required')
    counter, bits, bound = candidates[0]
    obligations = {}
    digests = []
    for index, (row, p) in enumerate(pairs):
        b = p.blocks[p.entry]
        roots = [b['outputs'][r] for r in closed] + list(b['params'].values())
        roots += [r for pair in b['guards'] for r in pair]
        try:
            a, refs = _translate(p, roots)
        except (ValueError, KeyError):
            return refuse('unsupported projected expression')
        one = a.const(1, 8)

        def conjunction(items):
            value = one
            for item in items:
                value = a.bin('and', value, item, 8)
            return value

        def compare(op, l, r):
            # Arena.bin fits operands to its result width before comparison.
            # Compare complete values, never their low flag byte.
            return a.bin(op, l, r, max(a.width(l), a.width(r)))

        def eq(l, r):
            return compare('eq', l, r)

        def neg(value):
            return eq(value, a.const(0, 8))

        params = {r: refs[v] for r, v in b['params'].items()}
        outputs = {r: refs[b['outputs'][r]] for r in closed}
        guard = conjunction(eq(refs[l], refs[r]) for l, r in b['guards'])
        static = []
        for l, r in b['guards']:
            ln, rn = p.nodes[l], p.nodes[r]
            if ((ln['op'] == 'param' and rn['op'] == 'c') or
                    (rn['op'] == 'param' and ln['op'] == 'c')):
                static.append(eq(refs[l], refs[r]))
        x, following = params[counter], outputs[counter]
        maximum = a.const(bound, 64)
        invariant = conjunction([*static, compare('ule', x, maximum)])
        continues = compare('ult', x, maximum)
        active = conjunction([invariant, guard])
        replacements = {params[r]: outputs[r] for r in closed}
        next_invariant = a.substitute(invariant, replacements)
        expected = a.bin('add', a.trunc(x, bits), a.const(1, bits), bits)
        if bits < 64:
            expected = a.zext(expected, 64)
        rank = a.bin('sub', maximum, x, 64)
        next_rank = a.bin('sub', maximum, following, 64)
        rank_ok = conjunction([compare('ult', next_rank, rank),
                               eq(next_rank, a.bin('sub', rank, a.const(1, 64), 64))])
        bad = dict(
            guard_domain=conjunction([invariant, compare('ne', guard, continues)]),
            counter_update=conjunction([active, compare('ne', following, expected)]),
            invariant_preservation=conjunction([active, neg(next_invariant)]),
            rank_decrease=conjunction([active, neg(rank_ok)]),
            terminal_boundary=conjunction([invariant, neg(guard), compare('ne', x, maximum)]))
        for name, mismatch in bad.items():
            proof = finite_values(a, mismatch, {}, max_values=2, node_limit=8000,
                                  timeout_ms=timeout_ms)
            if proof.status != 'complete' or proof.values != (0,):
                return refuse(f'cut {index} {name}: {proof.status} {proof.values}')
            obligations[f'cut_{index}.{name}'] = 'unsat'
        observed = row.get('native_entry_constants', {})
        bindings = {params[r]: a.const(value, 64) for r, value in observed.items()
                    if r in params and value is not None}
        initial = a.substitute(conjunction([invariant, guard]), bindings)
        proof = finite_values(a, initial, {}, max_values=2, node_limit=8000,
                              timeout_ms=timeout_ms)
        if proof.status != 'complete' or proof.values != (1,):
            return refuse(f'cut {index} observed entry is outside the active invariant')
        obligations[f'cut_{index}.initialization'] = 'unsat'
        digests.append(hashlib.sha256(json.dumps(p.to_dict(), sort_keys=True).encode()).hexdigest())
    return dict(result, loop_proved=True, registers=closed, counter=counter,
                counter_bits=bits, invariant=f'0 <= {counter} <= {bound}; entry constant guards',
                continuation=f'{counter} < {bound}', terminal=f'{counter} == {bound}',
                ranking_function=f'{bound} - {counter}', obligations=obligations,
                maximum_iterations_from_observed_entry=bound - rows[0]['native_entry_constants'][counter],
                guarded_transfer_signature=closures[0]['signature'], transfer_sha256=digests,
                proof='universal bitvector induction and decreasing bounded rank',
                excluded='memory safety, other backing outputs, native cache recurrence and native loop equivalence')
