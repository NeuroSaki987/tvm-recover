"""Reconstruct a cached scalar from an exhaustively surviving set of copies.

No pointer-domain guard is added. Each representation has an executable exact
nonoverlap test. A universal BV proof must establish that at least one survives;
otherwise the historical dependency remains. Address tests may not read memory,
which prevents circular reasoning through the values being reconstructed.
"""
from .bitproof import finite_values
from .memory import may_alias


def _inverse(a, expression, target):
    plan = []
    for _ in range(10):
        if expression == target:
            return plan
        n = a.node(expression)
        if n[0] == 'un' and a.width(n[2]) == a.width(expression):
            plan.append((n[1], 0)); expression = n[2]
        elif n[0] == 'bin' and n[1] in ('xor', 'add', 'sub'):
            op, l, r = n[1:4]
            lc, rc = a.as_const(l), a.as_const(r)
            if rc is not None and a.width(l) == a.width(expression):
                plan.append(({'xor': 'xor', 'add': 'sub', 'sub': 'add'}[op], rc))
                expression = l
            elif lc is not None and a.width(r) == a.width(expression):
                plan.append(({'xor': 'xor', 'add': 'sub', 'sub': 'rsub'}[op], lc))
                expression = r
            else:
                break
        else:
            break
    return None


def _chain(a, entry, stop):
    records, seen = [], set()
    while entry != stop:
        if entry in seen or entry not in a.memory_versions or len(records) >= 40000:
            return None
        seen.add(entry)
        record = a.memory_versions[entry]
        records.append((entry, record))
        entry = record[0]
    return records


def _pure_address(a, ref):
    return all(a.node(r)[0] != 'load' for r in a._descendants_ordered(ref))


def _safe(a, address, size, records):
    safe = a.const(1, 64)
    for _version, (_parent, addr, value, count) in records:
        if value is None and count == 0:
            return None
        if not may_alias(a, addr, count, address, size):
            continue
        if not _pure_address(a, addr) or not 0 < count < (1 << 64):
            return None
        # Two modular intervals overlap iff either start lies in the other.
        left = a.bin('ule', a.const(size, 64), a.bin('sub', addr, address, 64), 64)
        right = a.bin('ule', a.const(count, 64), a.bin('sub', address, addr, 64), 64)
        safe = a.bin('and', safe, a.bin('and', left, right, 64), 64)
        if a.as_const(safe) == 0:
            return None
    return safe


def _expression(a, candidate, width, entry):
    result = a.load(candidate['address_ref'], width, entry)
    for op, value in candidate['inverse']:
        if op in ('neg', 'not'):
            result = a.un(op, result, width)
        elif op == 'rsub':
            result = a.bin('sub', a.const(value, width), result, width)
        else:
            result = a.bin(op, result, a.const(value, width), width)
    return result


def _covered(a, candidates, timeout_ms):
    condition = a.const(0, 64)
    for candidate in candidates:
        condition = a.bin('or', condition, candidate['condition_ref'], 64)
    proof = finite_values(a, condition, {}, max_values=2, timeout_ms=timeout_ms, node_limit=4000)
    return proof.status == 'complete' and proof.values == (1,)


def recover(a, ref, entry, timeout_ms=1500):
    n = a.node(ref)
    if n[0] != 'load' or timeout_ms <= 0:
        return None
    records = _chain(a, entry, n[3])
    if records is None:
        return None
    width, candidates = a.width(ref), []
    demands = []
    for i, (version, (_parent, addr, value, size)) in enumerate(records):
        if value is not None and a.width(value) == width and size * 8 >= width and a.is_const(addr):
            inverse = _inverse(a, value, ref)
            if inverse is not None:
                demands.append((version, addr, inverse, i))
    if _pure_address(a, n[1]):
        demands.append((n[3], n[1], [], len(records)))
    seen = set()
    for version, addr, inverse, i in demands:
        marker = (addr, tuple(inverse))
        if marker in seen:
            continue
        seen.add(marker)
        safe = _safe(a, addr, width // 8, records[:i])
        if safe is None:
            continue
        candidates.append(dict(memory_version=version, address_ref=addr,
                               inverse=inverse, condition_ref=safe))
        if len(candidates) >= 8:
            break
    if not candidates or not _covered(a, candidates, timeout_ms):
        return None
    result = _expression(a, candidates[-1], width, entry)
    for candidate in reversed(candidates[:-1]):
        result = a.select(candidate['condition_ref'], _expression(a, candidate, width, entry), result, width)
    return dict(kind='covered_entry_copies', native_expression=ref, entry_version=entry,
                result_ref=result, candidates=candidates,
                proof='universal bitvector coverage of unmodified representations')


def validate(a, proof, timeout_ms=1500):
    """Rewalk the history and reconstruct the certificate; trust no suffix count."""
    ref, entry = proof['native_expression'], proof['entry_version']
    n = a.node(ref)
    if n[0] != 'load':
        return False
    width = a.width(ref)
    for candidate in proof['candidates']:
        version, addr = candidate['memory_version'], candidate['address_ref']
        if version == n[3] and addr == n[1]:
            inverse = []
        else:
            if version not in a.memory_versions:
                return False
            _parent, address, value, size = a.memory_versions[version]
            if address != addr or value is None or size * 8 < width or a.width(value) != width:
                return False
            inverse = _inverse(a, value, ref)
        if inverse is None or [list(v) for v in inverse] != [list(v) for v in candidate['inverse']]:
            return False
        suffix = _chain(a, entry, version)
        if suffix is None or _safe(a, addr, width // 8, suffix) != candidate['condition_ref']:
            return False
    if not proof['candidates'] or not _covered(a, proof['candidates'], timeout_ms):
        return False
    result = _expression(a, proof['candidates'][-1], width, entry)
    for candidate in reversed(proof['candidates'][:-1]):
        result = a.select(candidate['condition_ref'], _expression(a, candidate, width, entry), result, width)
    return result == proof['result_ref']
