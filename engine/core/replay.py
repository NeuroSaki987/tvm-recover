"""Transactional replay of an observed native segment under explicit guards.

Every instruction and every folded transfer is checked, including transfers
that need no intervention. Forcing a target is a path constraint, never evidence
that the path is input independent. A failed/budget-limited replay contributes
no machine or census effects. Arena nodes/work counters remain append-only.
"""
from dataclasses import dataclass, field

from .state import LifterCheckpoint, MachineSnapshot


@dataclass(frozen=True)
class TraceStep:
    site: int
    next_ip: int
    transfers: tuple = ()


@dataclass
class ReplayResult:
    ok: bool = False
    reason: str = ''
    steps: int = 0
    transfers: int = 0
    forced_dispatches: int = 0
    forced_branches: int = 0
    guards: list = field(default_factory=list)
    next_ip: int = 0


def replay(lifter, entry, guide, limit=400_000):
    """Re-lift exactly `guide`, keeping effects only if every step matches."""
    if limit <= 0:
        raise ValueError('replay budget must be positive')
    checkpoint = LifterCheckpoint(lifter)
    result = ReplayResult(next_ip=entry)
    arena = lifter.a
    bindings = {}
    ip = entry

    def constrain(ref, expected):
        constant = arena.as_const(ref)
        if constant is not None:
            return constant == expected
        if ref in bindings:
            return arena.as_const(bindings[ref]) == expected
        value = arena.const(expected, arena.width(ref))
        bindings[ref] = value
        result.guards.append((ref, value))
        snapshot = MachineSnapshot(lifter.m, lifter=lifter)
        snapshot.restrict(arena, {ref: value})
        snapshot.restore(lifter.m, lifter)
        return True

    try:
        for expected in guide:
            if result.steps >= limit:
                result.reason = 'step_budget'
                return result
            if ip != expected.site:
                result.reason = 'instruction_mismatch'
                return result
            mark = len(lifter.dispatches)
            kind, payload = lifter.step(ip)
            result.steps += 1
            actual = tuple(lifter.dispatches[mark:])
            if actual != expected.transfers[:len(actual)]:
                result.reason = 'transfer_mismatch'
                return result
            result.transfers += len(actual)
            if kind == 'next':
                if actual != expected.transfers:
                    result.reason = 'transfer_mismatch'
                    return result
                if payload != expected.next_ip:
                    result.reason = 'next_ip_mismatch'
                    return result
                ip = payload
            elif payload.kind == 'dispatch_call':
                # A stopped call has not pushed its return address. Moreover a
                # guide's next IP may be an external call's continuation, not
                # the callee. Guessing it as a jump target loses call effects.
                result.reason = 'unsupported_indirect_call'
                return result
            elif payload.kind == 'dispatch':
                missing = expected.transfers[len(actual):]
                if missing != ((payload.site, expected.next_ip),):
                    result.reason = 'dispatch_site_mismatch'
                    return result
                if not constrain(payload.dest, expected.next_ip):
                    result.reason = 'contradictory_guard'
                    return result
                lifter.dispatches.append((payload.site, expected.next_ip))
                result.transfers += 1
                result.forced_dispatches += 1
                ip = expected.next_ip
            elif payload.kind == 'native_branch':
                if actual != expected.transfers or expected.next_ip not in (payload.taken, payload.not_taken):
                    result.reason = 'branch_target_mismatch'
                    return result
                if payload.taken != payload.not_taken:
                    # Branch semantics tests nonzero, rather than assuming an
                    # opaque flag expression is restricted to exactly {0,1}.
                    predicate = (payload.pred if arena.is_bool(payload.pred) else
                                 arena.bin('ne', payload.pred, arena.const(0, arena.width(payload.pred)),
                                           arena.width(payload.pred)))
                    if not constrain(predicate, int(expected.next_ip == payload.taken)):
                        result.reason = 'contradictory_guard'
                        return result
                result.forced_branches += 1
                ip = expected.next_ip
            else:
                result.reason = 'stop:' + payload.kind
                return result
            result.next_ip = ip
        result.ok = True
        return result
    finally:
        if not result.ok:
            checkpoint.restore(lifter)
