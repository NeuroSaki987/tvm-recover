"""Scoped, portable names for the serialized state interface.

Names describe inputs and origins; they do not resolve unknown semantics. Native
refs remain provenance and legacy binding keys, never the public spelling.
"""
import ast
from collections import Counter, defaultdict
from urllib.parse import quote

from .xstate import REGS


def _text(value):
    return quote(str(value), safe='._-') or 'empty'


def _native_name(node):
    raw = node.get('name', '')
    # Older schema-1 artifacts stored tuple reprs, rather than their leaf name.
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, tuple) and len(parsed) == 1 and isinstance(parsed[0], str):
            return parsed[0]
    except (ValueError, SyntaxError):
        pass
    return raw


def catalog(program, key):
    key = str(key)
    b = program.blocks[key]
    roots = list(b['outputs'].values()) + list(b.get('return_outputs', {}).values()) + [b['memory_out']]
    from .terminals import roots as terminal_roots
    roots += terminal_roots(b.get('terminal', {}))
    roots += [r for pair in b['guards'] for r in pair]
    for edge in program.edges:
        if edge['source'] == key:
            roots += list(edge['args'].values()) + [edge['memory']]
            roots += [r for pair in edge['guards'] for r in pair]
    deps = program.dependencies(roots)
    records = []

    def add(ref, name, kind, binding=None, source=None):
        records.append(dict(name=name, kind=kind, node=ref, block=key,
                            width=program.nodes[ref]['width'], binding=binding,
                            source={} if source is None else source))

    for reg in REGS:
        add(b['params'][reg], 'state.' + reg, 'register', source=dict(register=reg))
    add(b['memory_in'], 'memory.entry', 'memory')
    effects = sorted((r for r in deps if program.nodes[r]['op'] == 'effect'),
                     key=lambda r: (program.nodes[r]['step'], r))
    occurrences, effect_names = defaultdict(int), {}
    for ref in effects:
        n = program.nodes[ref]
        role = (n['kind'], n['site'], n['flag'])
        ordinal = occurrences[role]
        occurrences[role] += 1
        effect_names[ref] = (f'effect.{_text(n["kind"])}.{_text(n["flag"])}.'
                             f'at_{n["site"]:x}.occ_{ordinal}')
    history_index = 0
    for ref in deps:
        n = program.nodes[ref]
        op = n['op']
        if op == 'input':
            add(ref, 'native.entry.' + _text(n['name']), 'input', ['input', n['name']],
                dict(native_name=n['name']))
        elif op == 'opaque':
            raw = _native_name(n)
            add(ref, 'native.opaque.' + _text(raw), 'opaque', ['opaque', n['native_ref']],
                dict(native_name=raw, native_ref=n['native_ref']))
        elif op == 'effect':
            add(ref, effect_names[ref], 'effect', ['effect', key, n['native_ref']],
                {k: n[k] for k in ('kind', 'flag', 'site', 'step', 'native_ref')})
        elif op == 'history':
            add(ref, f'memory.history.{history_index}', 'history', ['memory', n['version']],
                dict(native_version=n['version']))
            history_index += 1
    counts = Counter(r['name'] for r in records)
    ordinal = defaultdict(int)
    for record in records:
        base = record['name']
        if counts[base] > 1:
            record['name'] = base + f'@distinct_{ordinal[base]}'
            ordinal[base] += 1
    # Escaping plus explicit duplicate ordinals must remain injective, including
    # adversarial names already containing the suffix spelling.
    if len({r['name'] for r in records}) != len(records):
        raise ValueError('ambiguous state symbol names')
    return records


def bind(program, key, values):
    names = {s['name']: tuple(s['binding']) for s in catalog(program, key) if s['binding'] is not None}
    result = {}
    for alias, value in values.items():
        if isinstance(alias, str):
            if alias not in names:
                raise ValueError(f'unknown state input name {alias!r} in block {key}')
            binding = names[alias]
        else:
            binding = alias
        if binding in result and result[binding] != value:
            raise ValueError(f'conflicting bindings for {alias!r}')
        result[binding] = value
    return result
