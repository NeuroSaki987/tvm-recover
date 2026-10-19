"""Bounded, complete bit-vector images of dispatch DAGs.

Memory is an uninterpreted function per width and historical version. Only a
proved finite address domain whose every read folds under that version is
materialized as a table. Unknown memory stays unknown; timeout never proves an
edge. This proves properties of the lifted model, not of the native lifter.
"""
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class ImageProof:
    status: str
    values: tuple[int, ...] = ()
    tables: int = 0
    checks: int = 0


def finite_values(arena, root, pins, max_values=16, table_limit=64,
                  node_limit=4000, timeout_ms=350):
    try:
        import z3
    except ImportError:
        return ImageProof('unavailable')
    if any(not arena.is_const(value) for value in pins.values()):
        return ImageProof('unsupported_pins')
    deadline = monotonic() + timeout_ms / 1000
    refs = {r for top in (root, *pins, *pins.values())
            for r in arena._descendants_ordered(top)}
    if len(refs) > node_limit:
        return ImageProof('node_limit')
    memo, loads = {}, []
    solver = z3.Solver()
    checks = tables = 0

    def fit(expr, width, signed=False):
        if expr.size() > width:
            return z3.Extract(width - 1, 0, expr)
        if expr.size() < width:
            return (z3.SignExt if signed else z3.ZeroExt)(width - expr.size(), expr)
        return expr

    def check():
        nonlocal checks
        remaining = int((deadline - monotonic()) * 1000)
        if remaining <= 0:
            return z3.unknown
        solver.set(timeout=max(1, remaining))
        checks += 1
        return solver.check()

    def image(expression, limit):
        values = []
        solver.push()
        try:
            while True:
                status = check()
                if status == z3.unsat:
                    return 'complete', tuple(sorted(values))
                if status != z3.sat:
                    return 'timeout', ()
                if len(values) == limit:
                    return 'too_many', ()
                value = solver.model().eval(expression, model_completion=True).as_long()
                values.append(value)
                solver.add(expression != value)
        finally:
            solver.pop()

    try:
        for ref in sorted(refs):
            if monotonic() >= deadline:
                return ImageProof('timeout')
            n = arena.node(ref)
            kind, width = n[0], arena.width(ref)
            bv = lambda v: z3.BitVecVal(v, width)
            if kind == 'c':
                expr = bv(n[1])
            elif kind in ('init', 'opaque', 'param'):
                expr = z3.BitVec(f'n{ref}', width)
            elif kind == 'load':
                address = memo[n[1]]
                memory = z3.Function(f'mem_{n[3]}_{width}_{address.size()}',
                                     address.sort(), z3.BitVecSort(width))
                expr = memory(address)
                loads.append(ref)
            elif kind in ('trunc', 'zext', 'sext'):
                child = memo[n[1]]
                if kind == 'trunc':
                    expr = z3.Extract(width - 1, 0, child)
                elif kind == 'zext':
                    expr = z3.ZeroExt(width - child.size(), child)
                else:
                    expr = z3.SignExt(width - child.size(), child)
            elif kind == 'sel':
                expr = z3.If(memo[n[1]] != 0, fit(memo[n[2]], width), fit(memo[n[3]], width))
            elif kind == 'un':
                expr = ~fit(memo[n[2]], width) if n[1] == 'not' else -fit(memo[n[2]], width)
            elif kind == 'bin':
                left, right, op = memo[n[2]], memo[n[3]], n[1]
                if op in ('and', 'or', 'xor') and max(left.size(), right.size()) > width:
                    bounded = [arena.width(r) <= width or
                               (arena.as_const(r) is not None and arena.as_const(r) < (1 << width))
                               for r in (n[2], n[3])]
                    if not (any(bounded) if op == 'and' else all(bounded)):
                        # The direct evaluator and normal builder disagree on
                        # overflow in malformed internal bitwise nodes. Refuse
                        # to prove anything about that ambiguity.
                        return ImageProof('unsupported', tables=tables, checks=checks)
                # Bit slicing can build mixed-width internal nodes. Match the
                # arena's integer-and-final-mask semantics, including wide counts.
                if op in ('eq', 'ne', 'ult', 'ule', 'udiv', 'urem'):
                    work_width = max(width, left.size(), right.size())
                elif op in ('shl', 'shr', 'sar'):
                    work_width = max(width, right.size())
                    left = fit(left, width)
                else:
                    work_width = width
                left = fit(left, work_width, signed=op == 'sar')
                right = fit(right, work_width)
                if op == 'add': expr = left + right
                elif op == 'sub': expr = left - right
                elif op == 'mul': expr = left * right
                elif op == 'and': expr = left & right
                elif op == 'or': expr = left | right
                elif op == 'xor': expr = left ^ right
                elif op == 'shl': expr = left << right
                elif op == 'shr': expr = z3.LShR(left, right)
                elif op == 'sar': expr = left >> right
                elif op == 'udiv': expr = z3.If(right == 0, z3.BitVecVal(0, work_width), z3.UDiv(left, right))
                elif op == 'urem': expr = z3.If(right == 0, z3.BitVecVal(0, work_width), z3.URem(left, right))
                else:
                    relations = {'eq': lambda: left == right, 'ne': lambda: left != right,
                                 'ult': lambda: z3.ULT(left, right), 'ule': lambda: z3.ULE(left, right),
                                 'slt': lambda: left < right, 'sle': lambda: left <= right}
                    expr = z3.If(relations[op](), bv(1), bv(0))
                expr = fit(expr, width)
            else:
                return ImageProof('unsupported')
            memo[ref] = expr
        for ref, value in pins.items():
            solver.add(memo[ref] == memo[value])
        for ref in loads:
            n = arena.node(ref)
            status, addresses = image(memo[n[1]], table_limit)
            if status == 'timeout':
                return ImageProof('timeout', tables=tables, checks=checks)
            if status != 'complete' or not addresses:
                continue
            values = []
            for addr in addresses:
                value = arena._fold_known_load(arena.const(addr, arena.width(n[1])), n[2], n[3])
                concrete = arena.as_const(value) if value is not None else None
                if concrete is None:
                    break
                values.append(concrete)
            if len(values) != len(addresses):
                continue
            expression = z3.BitVecVal(values[-1], n[2])
            for addr, value in reversed(list(zip(addresses[:-1], values[:-1]))):
                expression = z3.If(memo[n[1]] == addr, z3.BitVecVal(value, n[2]), expression)
            solver.add(memo[ref] == expression)
            tables += 1
        status, values = image(memo[root], max_values)
        if status == 'complete' and not values:
            status = 'infeasible'
        return ImageProof(status, values, tables, checks)
    except (z3.Z3Exception, KeyError, ValueError):
        return ImageProof('unsupported', tables=tables, checks=checks)
