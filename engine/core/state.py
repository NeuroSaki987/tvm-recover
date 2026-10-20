"""Shared architectural snapshots and transactional semantic observations.

**Restores mutate in place, and that is a correctness requirement rather than a style choice.**
`setattr(lifter, name, dict(values))` rebinds the attribute to a *new* object, so anything holding
the old one keeps reading the state the snapshot was supposed to undo. That is not hypothetical: the
machine holds `guest_marks` -- the census's identity map -- and after a fork, a retry or a native
summary the machine went on consulting an identity map that had been withdrawn, so a mark left by a
rejected attempt could still decide that an address was a guest address and spare the frame. The
reverse of the `frame_separated` flag, arrived at from the other direction: there, observation changed
behaviour through a switch; here, through a stale reference.

The owner of a map is its object identity. A snapshot restores *contents*, and every holder of the
reference sees the restored state because there is only ever one object.
"""
from copy import deepcopy


def _restore_dict(target, saved) -> None:
    """Put `saved` back into `target` without replacing the object."""
    target.clear()
    target.update(saved)


def _restore_list(target, saved) -> None:
    target[:] = saved


class CensusState:
    DICTS = ('guest_values', 'guest_identity', 'gv_prov', 'gv_first')

    def __init__(self, lifter):
        self.maps = {name: dict(getattr(lifter, name)) for name in self.DICTS}
        self.ctx_read = lifter._ctx_read

    def restore(self, lifter):
        for name, values in self.maps.items():
            _restore_dict(getattr(lifter, name), values)
        lifter._ctx_read = self.ctx_read


class MachineSnapshot:
    DICTS = ('regs', 'flags', 'mem', 'xmm', 'symbolic_mem', 'byte_versions')
    LISTS = ('sym_loads', 'sym_stores', 'unmapped_loads', 'sym_store_windows', 'clobbered_ranges')
    # **State the recovery's own assumptions live in, which has to roll back with everything else.**
    # `isolation_spared` and `pending_isolated` decide what a later read returns and whether
    # `refute_isolation` deletes memory, so a rejected attempt must not leave them behind; the
    # counters are rolled back with them because a call or a guard built during an attempt that was
    # undone did not happen.
    ASSUMPTION_LISTS = ('isolation_conditions', 'isolation_log', 'pending_isolated')
    ASSUMPTION_SCALARS = ('isolation_refuted', 'guards_built', 'guards_dropped',
                          'guards_applied', 'call_memory_checked', 'call_memory_forgotten')

    def __init__(self, machine, events: int = 0, lifter=None):
        for name in self.DICTS:
            setattr(self, name, dict(getattr(machine, name)))
        for name in self.LISTS:
            setattr(self, name, list(getattr(machine, name)))
        for name in self.ASSUMPTION_LISTS:
            setattr(self, name, list(getattr(machine, name)))
        self.isolation_spared = set(machine.isolation_spared)
        self.assumption_scalars = {n: getattr(machine, n) for n in self.ASSUMPTION_SCALARS}
        self.unmapped = self.unmapped_loads
        self.mem_version = machine.mem_version
        self.symbolic_window = machine.symbolic_window
        self.events = events
        self.census = CensusState(lifter) if lifter is not None else None
        self.predecessor = ((lifter._previous_ip, lifter._current_ip)
                            if lifter is not None else (None, None))

    @property
    def cells(self):
        return sum(len(getattr(self, name)) for name in self.DICTS + self.LISTS)

    def restrict(self, arena, pins):
        """Condition the whole saved machine, not just its next dispatch address.

        All roots share one topological rebuild. Memory epochs remain historical;
        this specializes current byte values without rewriting another path's DAG.
        """
        pins = {k: v for k, v in pins.items() if arena.is_const(v)}
        if not pins:
            return
        roots = set(self.regs.values()) | set(self.flags.values()) | set(self.xmm.values())
        roots.update(v[0] for v in self.mem.values() if not isinstance(v, int))
        for (addr, _size), value in self.symbolic_mem.items():
            roots.update((addr, value))
        for addr, _size, value, _epoch in self.pending_isolated:
            roots.update((addr, value))
        if self.census is not None:
            for values in self.census.maps.values():
                roots.update(values)
        if not roots:
            return
        order = sorted({ref for root in roots for ref in arena._descendants_ordered(root)})
        memo = {}
        arena.substitute(next(iter(roots)), pins, memo=memo, order=order)
        for name in ('regs', 'flags', 'xmm'):
            setattr(self, name, {k: memo[v] for k, v in getattr(self, name).items()})
        self.mem = {k: v if isinstance(v, int) else (memo[v[0]], v[1])
                    for k, v in self.mem.items()}
        self.symbolic_mem = {(memo[addr], size): memo[value]
                             for (addr, size), value in self.symbolic_mem.items()}
        self.pending_isolated = [(memo[addr], size, memo[value], epoch)
                                 for addr, size, value, epoch in self.pending_isolated]
        if self.census is not None:
            for name, values in self.census.maps.items():
                remapped, ambiguous = {}, set()
                for ref, value in values.items():
                    new = memo[ref]
                    if arena.is_const(new):
                        continue
                    if new in remapped and remapped[new] != value:
                        ambiguous.add(new)
                    remapped[new] = value
                self.census.maps[name] = {k: v for k, v in remapped.items() if k not in ambiguous}

    def restore(self, machine, lifter=None):
        # Cancellation evidence belongs to the execution path that produced it.
        # Register refs restored from a snapshot must not reuse discarded swaps.
        if machine.swap_tracker is not None:
            machine.swap_tracker.flush()
        for name in self.DICTS:
            _restore_dict(getattr(machine, name), getattr(self, name))
        for name in self.LISTS:
            _restore_list(getattr(machine, name), getattr(self, name))
        for name in self.ASSUMPTION_LISTS:
            _restore_list(getattr(machine, name), getattr(self, name))
        _restore_dict(machine.isolation_spared, self.isolation_spared)
        # `update` rather than a rebind, for the same reason as the census maps: these are scalars on
        # the dict's owner, and a rebind here would be a second object to keep in step.
        for name, value in self.assumption_scalars.items():
            setattr(machine, name, value)
        machine.mem_version = self.mem_version
        machine.symbolic_window = self.symbolic_window
        if self.census is not None and lifter is not None:
            self.census.restore(lifter)
        if lifter is not None:
            lifter._previous_ip, lifter._current_ip = self.predecessor


class LifterCheckpoint:
    """Only a selected execution attempt may contribute effects and provenance."""
    LISTS = ('events', 'boxed', 'dispatches', 'guest_reads', 'guest_writes',
             'guest_reg_writes', 'guest_derived', 'guest_loads', 'guest_trace',
             'all_stores', 'slot_read_widths', 'local_effects')
    DICTS = ('grw_rejected', 'guest_load_rejects')

    def __init__(self, lifter):
        self.machine = MachineSnapshot(lifter.m, lifter=lifter)
        self.lists = {name: list(getattr(lifter, name)) for name in self.LISTS}
        self.dicts = {name: dict(getattr(lifter, name)) for name in self.DICTS}
        self.cache = deepcopy(lifter.guest_cache)

    def restore(self, lifter):
        self.machine.restore(lifter.m, lifter)
        for name, values in self.lists.items():
            _restore_list(getattr(lifter, name), values)
        for name, values in self.dicts.items():
            _restore_dict(getattr(lifter, name), values)
        lifter.guest_cache = deepcopy(self.cache)
