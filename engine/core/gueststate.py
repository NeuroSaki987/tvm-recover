"""Parametric transfers of the Guest *backing* register file and native memory.

The context layout is evidence; a host register merely carrying a copied Guest
value is not an architectural commit. We therefore project actual machine
boundary values, never the census's `home`/`carry` attribution. Cached values
remain candidates. Each block has 17 i64 parameters and a memory parameter.
Edges pass values rather than recursively expanding predecessor expressions.

This graph is a trace projection under the recovery's layout, path and memory
assumptions. It is not a proof that the first-arrival native CFG generalises to
all Guest states. Historical memory and residual native inputs stay explicit.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

from .expr import _children, _fold_bin, sext_value
from .mach import Machine
from .memory import may_alias, _base_offset, ADDRESS_MASK
from .xstate import REGS


@dataclass
class Observation:
    machine: object
    context: object
    image_u8: object

    def materialize(self):
        return capture(self.machine, self.context, self.image_u8)


def observe(machine, ctx, image_u8, snapshot=None):
    """Freeze only the slot bytes; allocate no expression during CFG search.

    Even pure expression allocation changes node IDs and hence the tie order
    of heuristic candidate sets. Defer materialisation until recovery finishes.
    """
    observer = copy(machine)
    source = snapshot or machine
    observer.mem = ({addr: value for addr, value in source.mem.items()
                     if ctx.base <= addr < ctx.base + 17 * 8} if ctx else {})
    observer.byte_versions = ({addr: version for addr, version in source.byte_versions.items()
                              if ctx.base <= addr < ctx.base + 17 * 8} if ctx else {})
    observer.pending_isolated = list(source.pending_isolated)
    observer.clobbered_ranges = list(source.clobbered_ranges)
    observer.mem_version = source.mem_version
    observer.symbolic_window = source.symbolic_window
    observer.sym_loads = []
    observer.unmapped_loads = []
    return Observation(observer, ctx, image_u8)


def capture(machine, ctx, image_u8, snapshot=None):
    """Read all physical slots on a private observer; no hooks or log mutations."""
    if ctx is None:
        return {}
    observer = copy(machine)
    # Loads append diagnostics and increment guard counters. A shallow machine
    # copy isolates scalars; copies of mutable logs isolate all load side effects.
    observer.sym_loads = []
    observer.unmapped_loads = []
    if snapshot is not None:
        for name in ('mem', 'mem_version', 'symbolic_window', 'byte_versions',
                     'pending_isolated', 'clobbered_ranges'):
            setattr(observer, name, getattr(snapshot, name))
    return {reg: Machine.load_concrete(observer, ctx.addr_of(reg), 8, -2, image_u8)
            for reg in REGS}


@dataclass
class Program:
    nodes: list[dict] = field(default_factory=list)
    blocks: dict = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    entry: str | None = None
    validation: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)
    _intern: dict = field(default_factory=dict, repr=False)

    def add(self, op, width, args=(), **attrs):
        key = (op, width, tuple(args), tuple(sorted(attrs.items())))
        if key not in self._intern:
            self._intern[key] = len(self.nodes)
            self.nodes.append(dict(op=op, width=width, args=list(args), **attrs))
        return self._intern[key]

    def dependencies(self, roots):
        seen, todo = set(), list(roots)
        while todo:
            ref = todo.pop()
            if ref not in seen:
                seen.add(ref)
                todo.extend(self.nodes[ref]['args'])
        return sorted(seen)

    def verify(self):
        """Check typing, local parameter scope and complete edge signatures."""
        errors = []
        for ref, node in enumerate(self.nodes):
            if any(not isinstance(a, int) or not 0 <= a < ref for a in node['args']):
                errors.append(f'node {ref}: forward or invalid argument')
                continue
            widths = [self.nodes[a]['width'] for a in node['args']]
            op, w = node['op'], node['width']
            good = True
            if op == 'bin':
                # Native Arena operators have an explicit result width and may
                # receive wider operands (notably masked 8-bit flags). Preserve
                # those operand types; silently truncating a shift count would
                # change its meaning. Evaluation follows Arena's width contract.
                good = len(widths) == 2 and all(t in (8, 16, 32, 64) for t in widths) and w in (8, 16, 32, 64)
            elif op == 'un':
                good = widths == [w]
            elif op == 'sel':
                good = len(widths) == 3 and widths[0] > 0 and widths[1:] == [w, w]
            elif op in ('trunc', 'zext', 'sext'):
                good = len(widths) == 1 and (widths[0] >= w if op == 'trunc' else widths[0] <= w)
            elif op == 'load':
                good = widths == [64, 0] and w in (8, 16, 32, 64)
            elif op == 'store':
                good = len(widths) == 3 and widths[:2] == [0, 64] and widths[2] >= node['bytes'] * 8 and w == 0
            elif op == 'havoc':
                good = widths == [0, 64] and w == 0
            if not good:
                errors.append(f'node {ref}: invalid {op} widths {widths} -> {w}')
        for key, b in self.blocks.items():
            if set(b['params']) != set(REGS) or set(b['outputs']) != set(REGS):
                errors.append(f'{key}: incomplete register signature')
            roots = list(b['outputs'].values()) + list(b.get('return_outputs', {}).values()) + [b['memory_out']]
            from .terminals import roots as terminal_roots
            terminal = b.get('terminal', {})
            if b.get('kind') in ('noreturn', 'fastfail') and not terminal:
                errors.append(f'{key}: missing terminal descriptor')
            if terminal:
                if terminal.get('kind') not in ('noreturn', 'fastfail') or terminal.get('kind') != b['kind']:
                    errors.append(f'{key}: inconsistent terminal kind')
                refs = terminal_roots(terminal)
                if any(type(ref) is not int or not 0 <= ref < len(self.nodes) for ref in refs):
                    errors.append(f'{key}: invalid terminal reference')
                    continue
                if terminal['kind'] == 'fastfail' and (
                        'code' not in terminal or self.nodes[terminal['code']]['width'] != 32):
                    errors.append(f'{key}: fastfail requires an i32 code')
                roots += refs
            roots += [v for pair in b['guards'] for v in pair]
            for ref in self.dependencies(roots):
                n = self.nodes[ref]
                if n['op'] in ('param', 'memparam', 'effect') and n['block'] != key:
                    errors.append(f'{key}: foreign parameter {ref}')
            if any(self.nodes[r]['width'] != 64 for r in b['outputs'].values()):
                errors.append(f'{key}: output is not i64')
            if self.nodes[b['memory_out']]['width'] != 0:
                errors.append(f'{key}: output memory is not a token')
        for edge in self.edges:
            if edge['source'] not in self.blocks:
                errors.append('edge has no source')
            if set(edge['args']) != set(REGS):
                errors.append('edge has incomplete register arguments')
            if not edge['recovered']:
                continue
            if edge['target'] not in self.blocks:
                errors.append('recovered edge has no target')
            roots = list(edge['args'].values()) + [edge['memory']]
            roots += [v for pair in edge['guards'] for v in pair]
            for ref in self.dependencies(roots):
                n = self.nodes[ref]
                if n['op'] in ('param', 'memparam', 'effect') and n['block'] != edge['source']:
                    errors.append('edge references a foreign parameter')
        return errors

    def _evaluate(self, key, state, memory, roots, externals=None):
        """Strict evaluator. Missing native inputs/bytes/historical memory fail."""
        externals = externals or {}
        values = {}
        pending = [(ref, False) for ref in roots]
        while pending:
            ref, ready = pending.pop()
            if ref in values:
                continue
            n = self.nodes[ref]
            op, w = n['op'], n['width']
            dependencies = n['args']
            if op == 'sel':
                condition = n['args'][0]
                if condition not in values:
                    pending.extend(((ref, False), (condition, False)))
                    continue
                chosen = n['args'][1 if values[condition] else 2]
                dependencies = [condition, chosen]
            if not ready or any(a not in values for a in dependencies):
                missing = [a for a in dependencies if a not in values]
                if missing:
                    pending.append((ref, True))
                    pending.extend((a, False) for a in reversed(missing))
                    continue
            args = [values[a] for a in dependencies]
            mask = (1 << w) - 1 if w else 0
            if op == 'param':
                if n['block'] != key:
                    raise ValueError('foreign block parameter')
                value = state[n['reg']]
            elif op == 'memparam':
                value = dict(memory)
                base = self.assumptions.get('context_base')
                if base is not None:
                    # The 17 backing values and the physical memory token are
                    # two views of one state. Materialise missing slot bytes and
                    # reject contradictory caller-supplied memory, rather than
                    # letting a slot read and its parameter disagree silently.
                    from .guestctx import GuestContext
                    context = GuestContext(base)
                    for reg in REGS:
                        for i in range(8):
                            addr = context.addr_of(reg) + i
                            byte = (state[reg] >> (8 * i)) & 255
                            if addr in value and value[addr] != byte:
                                raise ValueError('backing register and entry memory disagree')
                            value[addr] = byte
            elif op == 'c':
                value = n['value']
            elif op == 'input':
                value = externals[('input', n['name'])]
            elif op == 'opaque':
                value = externals[('opaque', n['native_ref'])]
            elif op == 'effect':
                if n['block'] != key:
                    raise ValueError('foreign block effect')
                # A witness belongs to this invocation, with correlation across
                # all uses of one definition. Never silently choose zero.
                value = externals[('effect', key, n['native_ref'])]
            elif op == 'history':
                value = externals[('memory', n['version'])]
            elif op == 'bin':
                value = _fold_bin(n['operator'], args[0], args[1], w)
            elif op == 'un':
                value = ~args[0] if n['operator'] == 'not' else -args[0]
            elif op in ('trunc', 'zext'):
                value = args[0]
            elif op == 'sext':
                value = sext_value(args[0], self.nodes[n['args'][0]]['width'])
            elif op == 'sel':
                value = args[1]
            elif op == 'load':
                addr, mem = args
                value = sum(mem[(addr + i) & ((1 << 64) - 1)] << (8 * i) for i in range(w // 8))
            elif op in ('store', 'havoc'):
                value = dict(args[0])
                addr = args[1]
                if op == 'store':
                    for i in range(n['bytes']):
                        value[(addr + i) & ((1 << 64) - 1)] = (args[2] >> (8 * i)) & 255
                elif n['bytes'] == 0:
                    value.clear()
                else:
                    for i in range(n['bytes']):
                        value.pop((addr + i) & ((1 << 64) - 1), None)
            else:
                raise ValueError(f'unsupported state node {op}')
            values[ref] = value & mask if w else value
        return values

    def run_block(self, key, state, memory, externals=None):
        return self._run_outputs(key, state, memory, 'outputs', externals)

    def closure(self, key):
        from .stateclosure import analyze
        return analyze(self, str(key))

    def memory_closure(self, key):
        from .stateclosure import analyze_memory
        return analyze_memory(self, str(key))

    def symbols(self, key):
        from .symbols import catalog
        return catalog(self, str(key))

    def bind_inputs(self, key, values):
        from .symbols import bind
        return bind(self, str(key), values)

    def run_named_state(self, key, state, memory, inputs):
        """Run the joint state using public scoped names instead of native IDs."""
        bindings = self.bind_inputs(key, inputs)
        try:
            return self.run_state(key, state, memory, bindings)
        except KeyError as error:
            for symbol in self.symbols(key):
                if symbol['binding'] is not None and error.args[0] == tuple(symbol['binding']):
                    raise KeyError(f'missing state input {symbol["name"]!r} in block {key}') from error
            raise

    def run_state(self, key, state, memory, externals=None):
        """Carry all backing registers and lazy persistent byte memory.

        Local effects/native unknowns still need explicit invocation witnesses.
        Missing memory fails at the actual read, never defaults to zero.
        Use memory.materialize() for a standalone byte dictionary.
        """
        from .statememory import Executor
        return Executor(self, key, state, memory, externals).run()

    def initial_memory(self, key):
        """Copy the measured concrete entry bytes, only when explicitly asked.

        This seed is evidence for one measured entry, not a general initializer
        and never an automatic reset of carried state.
        """
        return {int(addr): value for addr, value in
                self.blocks[str(key)].get('entry_memory_seed', {}).items()}

    def run_state_edge(self, source, target, state, memory, externals=None):
        """Select a guarded CFG edge and carry its values and persistent memory."""
        from .statememory import Executor
        source, target = str(source), str(target)
        executor = Executor(self, source, state, memory, externals)
        for edge in self.edges:
            if edge['source'] != source or edge['target'] != target:
                continue
            if not edge['recovered']:
                raise ValueError('target is not recovered')
            if executor.matches(self.blocks[source]['guards'] + edge['guards']):
                return executor.run(edge['args'], edge['memory'], edge['guards'])
        raise ValueError('no edge variant satisfies the state/path guards')

    def run_projection(self, key, state, memory, registers, externals=None):
        """Evaluate selected backing outputs under every block guard.

        Unrequested outputs and output memory are not evaluated. This permits
        a closed register subsystem to execute with only its own input state.
        It makes no claim about the discarded outputs or memory effects.
        """
        key = str(key)
        b = self.blocks[key]
        outputs = {reg: b['outputs'][reg] for reg in registers}
        self._check_guards(key, state, memory, externals)
        values = self._evaluate(key, state, memory, outputs.values(), externals)
        return {reg: values[ref] for reg, ref in outputs.items()}

    def _check_guards(self, key, state, memory, externals):
        guards = self.blocks[str(key)]['guards']
        values = self._evaluate(str(key), state, memory,
                                [v for pair in guards for v in pair], externals)
        if any(values[l] != values[r] for l, r in guards):
            raise ValueError('entry state violates a specialization guard')

    def run_return(self, key, state, memory, externals=None):
        """Native state at the lifter's return stop; see return_known_masks.

        Only seven arithmetic/direction flags plus fixed bit 1 are modelled.
        A symbolic RSP stops the lifter before its stack adjustment.
        """
        if not self.blocks[str(key)].get('return_outputs'):
            raise ValueError('block has no observed native return')
        return self._run_outputs(key, state, memory, 'return_outputs', externals)

    def _run_outputs(self, key, state, memory, output_field, externals):
        b = self.blocks[str(key)]
        outputs = b[output_field]
        roots = list(outputs.values()) + [b['memory_out']]
        self._check_guards(key, state, memory, externals)
        values = self._evaluate(str(key), state, memory, roots, externals)
        return {r: values[v] for r, v in outputs.items()}, dict(values[b['memory_out']])

    def run_edge(self, source, target, state, memory, externals=None):
        """Execute the selected edge including its recovered path constraints."""
        b = self.blocks[str(source)]
        for e in self.edges:
            if e['source'] != str(source) or e['target'] != str(target):
                continue
            if not e['recovered']:
                raise ValueError('target is not recovered')
            guards = b['guards'] + e['guards']
            values = self._evaluate(str(source), state, memory, [v for pair in guards for v in pair], externals)
            if any(values[l] != values[r] for l, r in guards):
                continue
            roots = list(e['args'].values()) + [e['memory']]
            values = self._evaluate(str(source), state, memory, roots, externals)
            return {r: values[v] for r, v in e['args'].items()}, dict(values[e['memory']])
        raise ValueError('no edge variant satisfies the state/path guards')

    def to_dict(self):
        return dict(schema=1, kind='guest-backing-state-transfer', entry=self.entry,
                    registers=list(REGS), nodes=self.nodes, blocks=self.blocks, edges=self.edges,
                    validation=self.validation, assumptions=self.assumptions,
                    symbols={key: self.symbols(key) for key in self.blocks},
                    contract='trace projection under measured context layout and recovery assumptions; '
                             'cached host values are candidates; full Guest equivalence unproved')

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 1 or data.get('kind') != 'guest-backing-state-transfer':
            raise ValueError('unsupported Guest transfer schema')
        p = cls(nodes=data['nodes'], blocks=data['blocks'], edges=data['edges'],
                entry=data['entry'], validation=data['validation'], assumptions=data.get('assumptions', {}))
        errors = p.verify()
        if errors:
            raise ValueError('; '.join(errors))
        if 'symbols' in data and data['symbols'] != {key: p.symbols(key) for key in p.blocks}:
            raise ValueError('serialized state symbol catalog does not match the graph')
        return p


class _Compiler:
    def __init__(self, program, arena, block):
        self.p, self.a, self.block = program, arena, block
        self.key = str(block.key)
        self.memo = {}
        self.originals = {}
        self.load_memory = {}
        self.slice_parents = {}
        self.copy_loads = {}
        self.copy_proofs = []
        self.entry_copies = None
        self.stale_cones = {}
        self.conditional_copies = {}
        self.conditional_copy_proofs = []
        self.params = {r: program.add('param', 64, block=self.key, reg=r) for r in REGS}
        self.memory = program.add('memparam', 0, block=self.key)
        self.memo[('m', block.memory_entry)] = self.memory
        self.guards = []
        for reg in REGS:
            raw = block.guest_entry[reg]
            if arena.is_const(raw):
                c = program.add('c', 64, value=arena.as_const(raw))
                self.guards.append([self.params[reg], c])
            elif ('e', raw) in self.memo:
                self.guards.append([self.params[reg], self.memo[('e', raw)]])
            else:
                self.memo[('e', raw)] = self.params[reg]
                self.originals[self.params[reg]] = raw
        # Only versions descended from this block's entry can use its memory
        # parameter. Older/sibling views must remain historical dependencies.
        self.local_versions = set()
        version = block.memory_exit
        while version != block.memory_entry and version in arena.memory_versions:
            if version in self.local_versions:
                raise ValueError('cyclic native memory history')
            self.local_versions.add(version)
            version = arena.memory_versions[version][0]
        if version != block.memory_entry:
            self.local_versions.clear()

    def deps(self, item):
        tag, ref = item
        if tag == 's':
            version, _addr, _width = ref
            _parent, addr, value, _size = self.a.memory_versions[version]
            return [self.slice_parents[ref], ('e', addr), ('e', value)]
        if tag == 'm':
            if ref not in self.local_versions:
                return []
            parent, addr, value, size = self.a.memory_versions[ref]
            return [('m', parent), ('e', addr)] + ([] if value is None else [('e', value)])
        n = self.a.node(ref)
        if self.conditional_copies.get(ref) is not None:
            return [('e', self.conditional_copies[ref])]
        if self.entry_copy_rematerialization and self._has_stale_read(ref):
            if ref not in self.copy_loads:
                copy = self._entry_copy(ref)
                if copy is not None:
                    self.copy_loads[ref] = copy
            if ref in self.copy_loads:
                return [('e', self.copy_loads[ref]), ('m', self.block.memory_entry)]
        out = [('e', child) for child in _children(n)]
        if n[0] == 'load':
            if ref not in self.load_memory:
                version = n[3]
                memory = ('m', version)
                if version != self.block.memory_entry and version not in self.local_versions:
                    target = self._retarget_stale(version, n[1], n[2]) if self.stale_retarget else None
                    if target is not None:
                        memory = ('m', target)
                    elif self.entry_copy_rematerialization and (copy := self._entry_copy(ref)) is not None:
                        self.copy_loads[ref] = copy
                        memory = ('m', self.block.memory_entry)
                    elif self.historical_slicing:
                        memory = self._historical_slice(version, n[1], n[2]) or memory
                    if memory == ('m', version) and self.covered_copy_rematerialization:
                        if ref not in self.conditional_copies:
                            from .memorycopies import recover
                            proof = recover(self.a, ref, self.block.memory_entry)
                            self.conditional_copies[ref] = proof['result_ref'] if proof else None
                            if proof:
                                self.conditional_copy_proofs.append(proof)
                        if self.conditional_copies[ref] is not None:
                            return [('e', self.conditional_copies[ref])]
                self.load_memory[ref] = memory
            if ref in self.copy_loads:
                out = [('e', self.copy_loads[ref])]
            out.append(self.load_memory[ref])
        return out

    #: A stale read that provably sees the entry's bytes is re-pointed at the entry token
    #: instead of being declared external history. Switchable so the A/B is available.
    stale_retarget = True
    historical_slicing = True
    entry_copy_rematerialization = True
    covered_copy_rematerialization = True

    def _has_stale_read(self, root):
        stack = [(root, False)]
        while stack:
            ref, ready = stack.pop()
            if ref in self.stale_cones:
                continue
            if ('e', ref) in self.memo:
                self.stale_cones[ref] = False
                continue
            n = self.a.node(ref)
            children = _children(n)
            if not ready:
                stack.append((ref, True))
                stack.extend((child, False) for child in children if child not in self.stale_cones)
                continue
            self.stale_cones[ref] = ((n[0] == 'load' and n[3] != self.block.memory_entry
                                      and n[3] not in self.local_versions)
                                     or any(self.stale_cones[c] for c in children))
        return self.stale_cones[root]

    def _index_entry_copies(self):
        copies, killed, uncertain, seen = {}, set(), [], set()
        cur, count = self.block.memory_entry, 0
        while cur not in seen:
            seen.add(cur)
            record = self.a.memory_versions.get(cur)
            if record is None:
                break
            parent, addr, value, size = record
            if value is None and size == 0:
                break
            address = self.a.as_const(addr)
            if address is not None and size <= 256:
                if value is not None and self.a.width(value) <= size * 8:
                    source = value
                    while True:
                        read_size = self.a.width(source) // 8
                        if (all((address + i) & ADDRESS_MASK not in killed for i in range(read_size))
                                and all(not may_alias(self.a, a, n, addr, read_size) for a, n in uncertain)):
                            copies.setdefault(source, (addr, cur, count))
                        n = self.a.node(source)
                        if (n[0] not in ('trunc', 'zext', 'sext')
                                or self.a.width(n[1]) > self.a.width(source)):
                            break
                        source = n[1]
                killed.update((address + i) & ADDRESS_MASK for i in range(size))
            else:
                uncertain.append((addr, size))
            cur, count = parent, count + 1
        self.entry_copies = copies

    def _entry_copy(self, ref):
        """Recover a historical scalar from a proved surviving entry-memory copy.

        A load carried in host state is often also saved in a native scratch
        slot. Exact stored-value identity and a complete nonaliasing suffix
        prove that this slot still contains the old value, even if its original
        source was overwritten. No sampling, extra guard or frame exemption.
        """
        if self.entry_copies is None:
            self._index_entry_copies()
        copy = self.entry_copies.get(ref)
        if copy is None:
            return None
        addr, version, count = copy
        self.copy_proofs.append(dict(kind='surviving_entry_copy', native_expression=ref,
                                     copy_store_version=version, entry_version=self.block.memory_entry,
                                     address=self.a.as_const(addr), bytes=self.a.width(ref) // 8,
                                     intervening_writes_checked=count))
        return addr

    def _historical_slice(self, version, addr_ref, width):
        """A read-local store view whose base bytes are provably unobservable.

        Walk backwards until every requested byte is definitely overwritten.
        Replay potential aliases in their original order, including symbolic
        stores; only proved disjoint writes are omitted. Entry memory supplies
        the base because every byte the load can read is overwritten in this
        view. The view never becomes the block's output memory. Incomplete
        coverage, cycles, precision views and relevant havoc remain historical.
        """
        if not isinstance(version, int) or version < 0:
            return None
        base, offset = _base_offset(self.a, addr_ref)
        uncovered = set(range(max(1, width // 8)))
        chain, seen = [], set()
        cur = version
        while uncovered:
            if cur in seen:
                return None
            seen.add(cur)
            record = self.a.memory_versions.get(cur)
            if record is None:
                return None
            parent, addr, value, size = record
            if (value is None and size == 0) or may_alias(self.a, addr, size, addr_ref, width // 8):
                if value is None:
                    return None
                chain.append(cur)
                write_base, write_offset = _base_offset(self.a, addr)
                if write_base == base:
                    uncovered = {i for i in uncovered
                                 if ((offset + i - write_offset) & ADDRESS_MASK) >= size}
                else:
                    # Different constant addresses have distinct Arena bases.
                    read_const, write_const = self.a.as_const(addr_ref), self.a.as_const(addr)
                    if read_const is not None and write_const is not None:
                        uncovered = {i for i in uncovered
                                     if ((read_const + i - write_const) & ADDRESS_MASK) >= size}
            cur = parent
        previous = ('m', self.block.memory_entry)
        for v in reversed(chain):
            key = (v, addr_ref, width)
            self.slice_parents[key] = previous
            previous = ('s', key)
        return previous

    def _retarget_stale(self, version, addr_ref, width):
        """The entry version, when a read at an older version provably sees the same bytes.

        `local_versions` covers only versions descended from this block's entry, so a load
        whose version is older is compiled as an external `history` dependency and the block
        is reported `memory_closed: false`. Measurement of why (probe/memory_chain, both
        `0x140171e29` and `0x140169e9d`): every such version is **older** than the entry
        version (57 of 57 and 100 % on the second root), no block's exit->entry chain is
        broken, and the same handful of versions recur across blocks — the signature of a
        value computed at an early epoch and carried forward, not of state that is missing.

        Such a read is declared external only because its epoch differs, not because the
        byte differs. Walking the recorded write chain from `memory_entry` back to `version`
        enumerates exactly the stores performed in `(version, entry]`; if none of them may
        touch the bytes this load reads, the byte at the entry equals the byte at `version`
        and reading through the entry token is the same value.

        Conservative by construction: unknown addresses widen through `address_bounds`, so a
        `havoc` or a symbolic-address store keeps the read historical. The rewrite can only
        remove a dependency that no recorded store supports.
        """
        a = self.a
        if not isinstance(version, int) or version < 0:
            return None                      # precision view or not a version at all
        size = max(1, width // 8)
        cur = self.block.memory_entry
        seen: set[int] = set()
        while cur != version:
            if cur in seen:
                return None                  # cyclic history, leave it external
            seen.add(cur)
            record = a.memory_versions.get(cur)
            if record is None:
                return None                  # chain does not reach `version`
            parent, addr, value, nbytes = record
            if (value is None and nbytes == 0) or may_alias(a, addr, nbytes, addr_ref, size):
                return None                  # a store in between may have changed these bytes
            cur = parent
        return self.block.memory_entry

    def compile(self, item):
        stack = [(item, False)]
        while stack:
            current, ready = stack.pop()
            if current in self.memo:
                continue
            children = self.deps(current)
            if not ready:
                stack.append((current, True))
                stack.extend((child, False) for child in reversed(children) if child not in self.memo)
                continue
            tag, ref = current
            args = [self.memo[child] for child in children]
            if tag in ('m', 's'):
                version = ref[0] if tag == 's' else ref
                if tag == 'm' and ref not in self.local_versions:
                    result = self.p.add('history', 0, version=ref)
                else:
                    parent, addr, value, size = self.a.memory_versions[version]
                    result = self.p.add('havoc' if value is None else 'store', 0, args,
                                        bytes=size, native_version=version)
            else:
                n = self.a.node(ref)
                op, width = n[0], self.a.width(ref)
                if self.conditional_copies.get(ref) is not None:
                    zero = self.p.add('c', width, value=0)
                    result = self.p.add('bin', width, [args[0], zero], operator='or',
                                        native_expression=ref,
                                        rematerialized_expression=self.conditional_copies[ref])
                elif ref in self.copy_loads:
                    result = self.p.add('load', width, args, native_expression=ref,
                                        native_version=self.block.memory_entry)
                elif op == 'c':
                    result = self.p.add('c', width, value=n[1])
                elif op == 'init':
                    result = self.p.add('input', width, name=n[1])
                elif op in ('opaque', 'param'):
                    origin = getattr(self.block, 'local_effects', {}).get(ref)
                    if origin is not None:
                        result = self.p.add('effect', width, native_ref=ref, block=self.key, **origin)
                    else:
                        result = self.p.add('opaque', width, native_ref=ref, name=str(n[1:-1]))
                elif op in ('bin', 'un'):
                    result = self.p.add(op, width, args, operator=n[1])
                elif op == 'load':
                    attrs = {'native_address': n[1]} if ref in self.copy_loads else {}
                    result = self.p.add('load', width, args, native_version=n[3], **attrs)
                elif op in ('trunc', 'zext', 'sext', 'sel'):
                    result = self.p.add(op, width, args)
                else:
                    raise ValueError(f'unsupported native expression {op}')
                self.originals[result] = ref
            self.memo[current] = result
        return self.memo[item]

    def expr(self, ref):
        return self.compile(('e', ref))

    def roundtrip(self, output_map, original_map):
        """Reverse parameter cuts into native nodes and check the projection.

        This verifies serialization/rebasing, not independent semantic accuracy.
        All operations are reconstructed in the native interner; the original
        input graph and the projection must coincide after parameters are bound.
        """
        native, errors = {}, []
        for ref in self.p.dependencies(output_map.values()):
            n = self.p.nodes[ref]
            op = n['op']
            if n['width'] == 0:
                continue
            args = [native[v] for v in n['args'] if self.p.nodes[v]['width']]
            if 'native_expression' in n:
                value = n['native_expression']
            elif op == 'param':
                value = self.block.guest_entry[n['reg']]
            elif op == 'c':
                value = self.a.const(n['value'], n['width'])
            elif op in ('input', 'opaque', 'effect'):
                value = self.originals[ref]
            elif op in ('bin', 'un'):
                value = self.a._add((op, n['operator'], *args, n['width']))
            elif op in ('sel', 'trunc', 'zext', 'sext'):
                # Reconstruction is structural. Re-running algebraic folding
                # here is not a round trip: some Arena normalisations expose
                # another fold on a second pass, despite equivalent values.
                value = self.a._add((op, *args, n['width']))
            elif op == 'load':
                value = (n['native_expression'] if 'native_expression' in n else
                         self.a.load(n.get('native_address', args[0]), n['width'], n['native_version']))
            native[ref] = value
        for reg, ref in output_map.items():
            if native[ref] != original_map[reg]:
                errors.append(reg)
        return errors


def build(cfg, arena):
    p = Program(entry=str(cfg.entry) if cfg.entry is not None else None)
    roundtrip_errors = []
    for key, b in cfg.blocks.items():
        if set(getattr(b, 'guest_entry', {})) != set(REGS) or set(getattr(b, 'guest_exit', {})) != set(REGS):
            continue
        compiler = _Compiler(p, arena, b)
        outputs = {r: compiler.expr(b.guest_exit[r]) for r in REGS}
        compiler.guards.extend([[compiler.expr(ref), compiler.expr(value)]
                                for ref, value in getattr(b, 'path_constraints', ())])
        return_outputs = {r: compiler.expr(v) for r, v in getattr(b, 'native_return', {}).items()}
        memory_out = compiler.compile(('m', b.memory_exit))
        errors = compiler.roundtrip(outputs, b.guest_exit)
        errors += ['return.' + r for r in compiler.roundtrip(return_outputs, getattr(b, 'native_return', {}))]
        roundtrip_errors.extend(f'{key}:{r}' for r in errors)
        roots = list(outputs.values()) + list(return_outputs.values()) + [memory_out]
        from .terminals import compile_terminal, roots as terminal_roots
        terminal = compile_terminal(compiler, b)
        if terminal:
            native_terminal = getattr(b, 'terminal', {})
            native_refs, compiled_refs = {}, {}
            if 'code' in terminal:
                native_refs['code'] = native_terminal['code']
                compiled_refs['code'] = terminal['code']
            for group in ('args', 'stack_args'):
                for name, value in terminal.get(group, {}).items():
                    compiled_refs[group + '.' + str(name)] = value
                    native_refs[group + '.' + str(name)] = native_terminal[group][name]
            terminal_errors = compiler.roundtrip(compiled_refs, native_refs)
            roundtrip_errors.extend(f'{key}:terminal.{name}' for name in terminal_errors)
        roots += terminal_roots(terminal)
        roots.extend(ref for pair in compiler.guards for ref in pair)
        candidates = sum(kind == 'home' for _, kind, _ in b.guest_trace)
        p.blocks[str(key)] = dict(params=compiler.params, memory_in=compiler.memory,
                                 outputs=outputs, memory_out=memory_out, guards=compiler.guards,
                                 return_outputs=return_outputs,
                                 return_known_masks={r: 0xCD7 if r == 'rflags' else (1 << 64) - 1
                                                     for r in return_outputs},
                                 return_phase=getattr(b, 'native_return_phase', 'lifter-stop'),
                                 residual_inputs=[], cache_candidates=candidates,
                                 kind=b.kind, hypothesis=getattr(b, 'hypothesis', False), roundtrip_errors=errors)
        if terminal:
            p.blocks[str(key)]['terminal'] = terminal
        for successor in dict.fromkeys(b.succs):
            # A conditioned edge can carry more precise values than the common
            # terminator. Its own predicate is retained, never silently assumed.
            observations = getattr(b, 'guest_edges', {}).get(successor)
            variants = observations if isinstance(observations, list) else [observations]
            for observation in variants:
                edge_values = observation['values'] if observation else b.guest_exit
                edge_pins = observation['pins'] if observation else {}
                arguments = {r: compiler.expr(edge_values[r]) for r in REGS}
                roundtrip_errors.extend(f'{key}->{successor}:{r}'
                                        for r in compiler.roundtrip(arguments, edge_values))
                guards = [[compiler.expr(ref), compiler.expr(value)] for ref, value in edge_pins.items()]
                roots.extend(arguments.values())
                roots.extend(v for pair in guards for v in pair)
                p.edges.append(dict(source=str(key), target=str(successor),
                                    args=arguments, memory=memory_out, guards=guards,
                                    hypothesis=getattr(b, 'hypothesis', False), recovered=False))
        p.blocks[str(key)]['residual_inputs'] = [r for r in p.dependencies(roots)
                                                if p.nodes[r]['op'] in ('input', 'opaque', 'history')]
        p.blocks[str(key)]['local_effects'] = [r for r in p.dependencies(roots)
                                              if p.nodes[r]['op'] == 'effect']
        p.blocks[str(key)]['memory_copy_proofs'] = compiler.copy_proofs
        p.blocks[str(key)]['conditional_memory_copy_proofs'] = compiler.conditional_copy_proofs
        if hasattr(b, 'entry_memory_seed'):
            p.blocks[str(key)]['entry_memory_seed'] = {
                str(addr): value for addr, value in b.entry_memory_seed.items()}
    for edge in p.edges:
        edge['recovered'] = edge['target'] in p.blocks
    p.validation = dict(blocks=len(p.blocks), register_outputs=17 * len(p.blocks),
                        roundtrip_errors=roundtrip_errors,
                        missing_blocks=len(cfg.blocks) - len(p.blocks),
                        residual_inputs=sum(len(b['residual_inputs']) for b in p.blocks.values()),
                        local_effects=sum(len(b['local_effects']) for b in p.blocks.values()),
                        cache_candidates=sum(b['cache_candidates'] for b in p.blocks.values()),
                        native_returns=sum(bool(b['return_outputs']) for b in p.blocks.values()),
                        guards=sum(len(b['guards']) for b in p.blocks.values()),
                        edges=len(p.edges),
                        missing_targets=sum(not e['recovered'] for e in p.edges))
    p.validation['verifier_errors'] = p.verify()
    return p
