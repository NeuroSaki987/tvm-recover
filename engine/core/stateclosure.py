"""Dependency closure of a guarded register transfer, independent of native IR.

The certificate describes the serialized transfer only. Its signature includes
the path guards, but establishes neither guard inductiveness nor cache ownership.
Memory is excluded from this register-only subsystem, never silently discarded
from the full Program.
"""
import hashlib
import json

from .xstate import REGS


def analyze(program, key):
    b = program.blocks[key]

    def cone(roots):
        deps = program.dependencies(roots)
        return dict(registers=[reg for reg in REGS if any(
                        program.nodes[ref]['op'] == 'param' and program.nodes[ref]['reg'] == reg
                        for ref in deps)],
                    residual_inputs=[r for r in deps if program.nodes[r]['op'] in ('input', 'opaque', 'history')],
                    local_effects=[r for r in deps if program.nodes[r]['op'] == 'effect'],
                    memory=any(program.nodes[r]['width'] == 0 or program.nodes[r]['op'] == 'load'
                               for r in deps))

    outputs = {reg: cone([ref]) for reg, ref in b['outputs'].items()}
    guards = cone([ref for pair in b['guards'] for ref in pair])

    def deterministic(c):
        return not (c['residual_inputs'] or c['local_effects'] or c['memory'])

    initial = {r for r, c in outputs.items() if deterministic(c)}
    closed = set(initial)
    while True:
        reduced = {r for r in closed if set(outputs[r]['registers']) <= closed}
        if reduced == closed:
            break
        closed = reduced
    transitive = [r for r in REGS if r in initial - closed]
    unavailable = [r for r in guards['registers'] if r not in closed]
    if unavailable or not deterministic(guards):
        closed.clear()
    registers = [r for r in REGS if r in closed]

    # Canonical Merkle hashes rename only local block labels. Every semantic
    # attribute of a deterministic register expression is retained. No native
    # node IDs, memory epochs or nondeterministic values are admitted here.
    signature = None
    if registers:
        roots = [b['outputs'][r] for r in registers] + [v for pair in b['guards'] for v in pair]
        hashes = {}
        for ref in program.dependencies(roots):
            n = program.nodes[ref]
            attrs = {k: v for k, v in n.items() if k not in ('args', 'block')}
            payload = [attrs, [hashes[a] for a in n['args']]]
            hashes[ref] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        payload = dict(outputs=[[r, hashes[b['outputs'][r]]] for r in registers],
                       guards=[[hashes[l], hashes[r]] for l, r in b['guards']])
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return dict(registers=registers, outputs=outputs, guards=guards,
                transitive_exclusions=transitive, unavailable_guard_registers=unavailable,
                signature=signature, conditional=True, memory_included=False,
                loop_proved=False)


def analyze_memory(program, key):
    """Joint register/memory interface; do not confuse effects with memory holes."""
    b = program.blocks[key]
    roots = list(b['outputs'].values()) + [b['memory_out']]
    from .terminals import roots as terminal_roots
    roots += terminal_roots(b.get('terminal', {}))
    roots += [r for pair in b['guards'] for r in pair]
    for edge in program.edges:
        if edge['source'] == key:
            roots += list(edge['args'].values()) + [edge['memory']]
            roots += [r for pair in edge['guards'] for r in pair]
    deps = program.dependencies(roots)
    histories = [r for r in deps if program.nodes[r]['op'] == 'history']
    native = [r for r in deps if program.nodes[r]['op'] in ('input', 'opaque')]
    effects = [r for r in deps if program.nodes[r]['op'] == 'effect']
    memory_roots = [r for r in deps if program.nodes[r]['op'] == 'memparam']
    memory_closed = not histories and all(r == b['memory_in'] for r in memory_roots)
    return dict(registers=list(REGS), memory_included=True, memory_closed=memory_closed,
                closed=memory_closed and not native,
                deterministic=not (native or effects),
                native_inputs=native, local_effects=effects, historical_memory=histories,
                outputs={reg: dict(loads=sum(program.nodes[r]['op'] == 'load'
                                            for r in program.dependencies([ref])))
                         for reg, ref in b['outputs'].items()},
                memory_origin='block entry byte memory plus ordered local writes',
                scope='block outputs, memory, terminal inputs and outgoing guarded edges',
                requires_entry_bytes=True, conditional=True, loop_proved=False)
