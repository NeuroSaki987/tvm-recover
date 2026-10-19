"""Demand-driven execution of register DAGs and persistent byte-memory epochs.

A memory node is a lazy state update. Reading a byte examines the latest writes
first; disjoint values and overwritten writes need not be evaluated. Cross-block
parents are resolved by an iterative scheduler, not Python recursion. Missing
and forgotten bytes remain unknown, with no implicit memory fill.
"""
from dataclasses import dataclass
from bisect import bisect_right
from types import MappingProxyType

from .expr import _fold_bin, sext_value
from .xstate import REGS

MASK = (1 << 64) - 1


@dataclass(frozen=True)
class UnknownByte:
    address: int
    reason: str


class MissingByte(KeyError):
    def __init__(self, address, reason):
        self.address, self.reason = address, reason
        super().__init__(f'{address:#x}: {reason}')


class _Need(Exception):
    def __init__(self, owner, task):
        self.owner, self.task = owner, task


@dataclass(frozen=True, init=False)
class MemoryInput:
    """Frozen finite memory regions plus explicitly observed byte overrides.

    A PE image can back regions without expanding millions of bytes into a
    dictionary. Unmapped addresses remain absent. Regions must not overlap.
    """
    overrides: object
    regions: tuple
    starts: tuple

    def __init__(self, overrides=(), regions=()):
        object.__setattr__(self, 'overrides', MappingProxyType(dict(overrides)))
        object.__setattr__(self, 'regions', tuple(sorted((base, bytes(data)) for base, data in regions)))
        object.__setattr__(self, 'starts', tuple(base for base, _ in self.regions))
        previous_end = 0
        for base, data in self.regions:
            if base < previous_end or base < 0 or base + len(data) > 1 << 64:
                raise ValueError('overlapping or invalid initial memory regions')
            previous_end = base + len(data)

    def get(self, address, default=None):
        if address in self.overrides:
            return self.overrides[address]
        index = bisect_right(self.starts, address) - 1
        if index >= 0:
            base, data = self.regions[index]
            if address - base < len(data):
                return data[address - base]
        return default

    def __iter__(self):
        yield from self.overrides
        for base, data in self.regions:
            yield from range(base, base + len(data))


class MemoryState:
    """Persistent byte state; retains the invocation's frozen inputs/witnesses."""
    def __init__(self, owner, ref):
        self.owner, self.ref = owner, ref

    def __getitem__(self, address):
        value = self.owner.resolve(('byte', self.ref, address & MASK))
        if isinstance(value, UnknownByte):
            raise MissingByte(value.address, value.reason)
        return value

    def get(self, address, default=None):
        try:
            return self[address]
        except MissingByte:
            return default

    def materialize(self, addresses=None):
        """Resolve all live written/base bytes; overwritten values stay dead."""
        selected = None if addresses is None else set(addresses)
        addresses, stack, seen = set(), [(self.owner, self.ref)], set()
        if selected is not None:
            addresses.update(selected)
            stack.clear()
        while stack:
            owner, ref = stack.pop()
            marker = (owner, ref)
            if marker in seen:
                continue
            seen.add(marker)
            n = owner.program.nodes[ref]
            if n['op'] == 'havoc' and n['bytes'] == 0:
                continue
            if n['op'] in ('store', 'havoc'):
                if n['op'] == 'store':
                    address = owner.resolve(('expr', n['args'][1]))
                    addresses.update((address + i) & MASK for i in range(n['bytes']))
                stack.append((owner, n['args'][0]))
            else:
                base = owner.memory if n['op'] == 'memparam' else owner.externals[('memory', n['version'])]
                if isinstance(base, MemoryState):
                    stack.append((base.owner, base.ref))
                else:
                    addresses.update(base)
                if n['op'] == 'memparam':
                    addresses.update(owner.context)
        result = {}
        for address in sorted(addresses):
            value = self.owner.resolve(('byte', self.ref, address))
            if not isinstance(value, UnknownByte):
                result[address] = value
        return result


class Executor:
    def __init__(self, program, key, state, memory, externals=None):
        self.program, self.key = program, str(key)
        self.state = dict(state)
        self.memory = memory if isinstance(memory, (MemoryState, MemoryInput)) else dict(memory)
        self.externals = {k: (dict(v) if isinstance(v, dict) else v)
                          for k, v in (externals or {}).items()}
        self.cache, self.context = {}, {}
        base = program.assumptions.get('context_base')
        if base is not None:
            from .guestctx import GuestContext
            ctx = GuestContext(base)
            self.context = {ctx.addr_of(reg) + i: (self.state[reg] >> (8 * i)) & 255
                            for reg in REGS for i in range(8)}

    def need(self, task):
        if task not in self.cache:
            raise _Need(self, task)
        return self.cache[task]

    def resolve(self, task):
        pending = [(self, task)]
        while pending:
            owner, current = pending[-1]
            if current in owner.cache:
                pending.pop()
                continue
            try:
                owner.cache[current] = owner.compute(current)
            except _Need as missing:
                pending.append((missing.owner, missing.task))
        return self.cache[task]

    def base_byte(self, base, address):
        if isinstance(base, MemoryState):
            return base.owner.need(('byte', base.ref, address))
        return base.get(address, UnknownByte(address, 'missing entry byte'))

    def compute(self, task):
        tag, ref, *rest = task
        n = self.program.nodes[ref]
        op, width = n['op'], n['width']
        get = lambda r: self.need(('expr', r))
        if tag == 'byte':
            address = rest[0]
            # Scan disjoint writes without memoizing one byte at every epoch.
            # Only the requested epoch and the supplying expression are needed.
            # This bounds carried-cache growth for long VM-frame histories.
            while op in ('store', 'havoc'):
                if op == 'havoc' and n['bytes'] == 0:
                    return UnknownByte(address, 'havoc')
                address_node = self.program.nodes[n['args'][1]]
                start = address_node['value'] if address_node['op'] == 'c' else get(n['args'][1])
                offset = (address - start) & MASK
                if offset < n['bytes']:
                    if op == 'havoc':
                        return UnknownByte(address, 'havoc')
                    return (get(n['args'][2]) >> (8 * offset)) & 255
                ref = n['args'][0]
                n = self.program.nodes[ref]
                op = n['op']
            # Validate only the memory origin actually reached by this read.
            get(ref)
            if op == 'memparam':
                return self.context[address] if address in self.context else self.base_byte(self.memory, address)
            if op == 'history':
                return self.base_byte(self.externals[('memory', n['version'])], address)
            raise ValueError(f'unsupported memory token {op}')
        if op in ('memparam', 'history', 'store', 'havoc'):
            if op == 'memparam':
                if n['block'] != self.key:
                    raise ValueError('foreign block memory')
                for address, expected in self.context.items():
                    actual = self.base_byte(self.memory, address)
                    if not isinstance(actual, UnknownByte) and actual != expected:
                        raise ValueError('backing register and entry memory disagree')
            elif op == 'history':
                self.externals[('memory', n['version'])]
            return MemoryState(self, ref)
        if op == 'param':
            if n['block'] != self.key:
                raise ValueError('foreign block parameter')
            value = self.state[n['reg']]
        elif op == 'c':
            value = n['value']
        elif op == 'input':
            value = self.externals[('input', n['name'])]
        elif op == 'opaque':
            value = self.externals[('opaque', n['native_ref'])]
        elif op == 'effect':
            if n['block'] != self.key:
                raise ValueError('foreign block effect')
            value = self.externals[('effect', self.key, n['native_ref'])]
        elif op == 'bin':
            value = _fold_bin(n['operator'], get(n['args'][0]), get(n['args'][1]), width)
        elif op == 'un':
            child = get(n['args'][0])
            value = ~child if n['operator'] == 'not' else -child
        elif op in ('trunc', 'zext'):
            value = get(n['args'][0])
        elif op == 'sext':
            value = sext_value(get(n['args'][0]), self.program.nodes[n['args'][0]]['width'])
        elif op == 'sel':
            value = get(n['args'][1 if get(n['args'][0]) else 2])
        elif op == 'load':
            address = get(n['args'][0])
            value = 0
            for i in range(width // 8):
                byte = self.need(('byte', n['args'][1], (address + i) & MASK))
                if isinstance(byte, UnknownByte):
                    raise MissingByte(byte.address, byte.reason)
                value |= byte << (8 * i)
        else:
            raise ValueError(f'unsupported state node {op}')
        return value & ((1 << width) - 1)

    def matches(self, guards):
        return all(self.resolve(('expr', left)) == self.resolve(('expr', right))
                   for left, right in guards)

    def run(self, outputs=None, memory_ref=None, extra_guards=()):
        b = self.program.blocks[self.key]
        if not self.matches(list(b['guards']) + list(extra_guards)):
            raise ValueError('entry state violates a specialization guard')
        self.resolve(('expr', b['memory_in']))
        state = {reg: self.resolve(('expr', ref))
                 for reg, ref in (b['outputs'] if outputs is None else outputs).items()}
        memory = self.resolve(('expr', b['memory_out'] if memory_ref is None else memory_ref))
        base = self.program.assumptions.get('context_base')
        if base is not None:
            from .guestctx import GuestContext
            context = GuestContext(base)
            for reg in REGS:
                for i in range(8):
                    if memory[context.addr_of(reg) + i] != ((state[reg] >> (8 * i)) & 255):
                        raise ValueError('backing output and output memory disagree')
        return state, memory
