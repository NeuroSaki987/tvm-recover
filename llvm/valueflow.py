"""Resolve captured register values without silently reloading mutable slots.

Raw parsed listings remain available to audit tools. Executable consumers use this
pass. Cross-block slot transport is allowed only after a reaching-definition proof
over every incoming path; a historical value that needs a separate spill refuses.
"""
from dataclasses import replace
import re
import guestir as G


class ValueFlowError(ValueError):
    pass


def key(e, block):
    return (e[3] if len(e) > 3 else block, e[1], e[2])


def dest_key(st, block):
    m = re.fullmatch(r'guest_(\w+)_(\d+)', st.dest)
    return (block, m[1], int(m[2])) if m else None


def source_index(block):
    return block.index if block.source_index is None else block.source_index


def prepare(fn):
    if getattr(fn, '_values_resolved', False):
        return fn
    entries = {b.index: {r: set() for r in G.CTX_LAYOUT} for b in fn.blocks}
    exits = {b.index: {r: set() for r in G.CTX_LAYOUT} for b in fn.blocks}
    pred = {b.label: [] for b in fn.blocks}
    for b in fn.blocks:
        for s in b.succs:
            if s in pred:
                pred[s].append(b.index)
    changed = True
    while changed:
        changed = False
        for pos, b in enumerate(fn.blocks):
            incoming = {r: set().union(*(exits[p][r] for p in pred[b.label]))
                        for r in G.CTX_LAYOUT}
            if pos == 0:
                for r in incoming:
                    incoming[r].add(None)
            outgoing = {r: set(v) for r, v in incoming.items()}
            for st in b.stmts:
                if st.kind in ('ctx-write', 'home', 'ctx-partial'):
                    r = st.extra[0]
                    if r in outgoing:
                        outgoing[r] = {dest_key(st, source_index(b))}
            if outgoing != exits[b.index] or incoming != entries[b.index]:
                entries[b.index], exits[b.index] = incoming, outgoing
                changed = True
    result = replace(fn, blocks=[resolve_block(b, entries[b.index]) for b in fn.blocks])
    result._values_resolved = True
    return result


def resolve_block(block, entry_sources=None):
    if getattr(block, '_values_resolved', False):
        return block
    bound = {}
    output = []

    def resolve(e):
        if not isinstance(e, tuple) or not e:
            return e
        if e[0] == 'reg':
            if len(e) < 3 or e[2] is None:
                return ('slot', e[1])
            k = key(e, source_index(block))
            if k in bound:
                return ('temp', bound[k], 64)
            if len(e) > 3 and entry_sources and entry_sources[e[1]] == {k}:
                return ('entry', e[1])
            raise ValueFlowError(f'block #{block.index}: uncaptured register value '
                                 f'#{k[0]}:guest_{k[1]}_{k[2]}; '
                                 'cross-block continuity is not proved')
        return tuple(resolve(x) if isinstance(x, tuple) else x for x in e)

    for i, st in enumerate(block.stmts):
        # The read annotation carries an actual operation, even when its printed
        # source is just a forwarded name. Capture it before subsequent writes.
        if 'read from ctx @' in st.text and st.kind in ('bare', 'forward-note', 'def'):
            slot = re.search(r'\[ctx\+(0x[0-9a-f]+) = (\w+)(?:\+\d+)?\]', st.text)
            if slot and st.kind in ('bare', 'forward-note'):
                head = st.text.split(';', 1)[0].strip()
                try:
                    e = G.parse_expr(head)
                except G.ParseError:
                    e = None
                if e and e[0] == 'reg':
                    nm = f'capture_b{block.index}_read_{i}'
                    site = re.search(r'read from ctx @(0x[0-9a-f]+)', st.text)
                    output.append(G.Stmt('ctx-load', nm, 64,
                                         extra=(slot[2], int(slot[1], 16), int(site[1], 16)),
                                         text=st.text, origin='ctx-read'))
                    bound[key(e, source_index(block))] = nm
                    continue
            if slot and st.kind == 'def':
                # Expressions such as trunc32(#0:guest_rdi_2) read *now*.
                # Their byte slicing remains in the expression, not in a guessed phi.
                def live(e):
                    if not isinstance(e, tuple) or not e:
                        return e
                    if e[0] == 'reg' and e[1] == slot[2]:
                        return ('slot', e[1])
                    return tuple(live(x) if isinstance(x, tuple) else x for x in e)
                st = replace(st, expr=live(st.expr))
        expression = st.expr
        if st.kind == 'ctx-partial':
            # The prev field is provenance; the real operation is a narrow store.
            expression = (expression[0], ('slot', st.extra[0]), *expression[2:])
        expression = resolve(expression)
        k = dest_key(st, source_index(block))
        if k and st.kind in ('ctx-write', 'home', 'ctx-partial', 'def', 'ctx-load'):
            nm = f'capture_b{block.index}_{k[1]}_{k[2]}'
            if st.kind in ('ctx-write', 'home'):
                output.append(G.Stmt('def', nm, 64, expression, text=st.text))
                expression = ('temp', nm, 64)
                output.append(replace(st, expr=expression))
            else:
                output.append(replace(st, expr=expression))
                if st.kind == 'ctx-partial':
                    output.append(G.Stmt('ctx-load', nm, 64, extra=st.extra))
                else:
                    output[-1] = replace(output[-1], dest=nm)
            bound[k] = nm
        else:
            output.append(replace(st, expr=expression))
    result = replace(block, stmts=output)
    result._values_resolved = True
    result._source_statement_count = len(block.stmts)
    return result
