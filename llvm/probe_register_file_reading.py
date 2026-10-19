"""What does `guest_<reg>_<k>` mean: the slot now, or the value that name was bound to?

`stale_reg_refs.py` counts the references where the two readings disagree -- 440 of them across 50
of `ACE-CORE.sys3`'s 144 listings. Counting them is not answering them. Both consumers of the
listing (`interp.py` and `lower_llvm.py`) implement the **slot** reading, so the 120/120 differential
is silent about this: two readers that share a misreading agree with each other.

This probe does three things the count does not.

**1. It asks whether the question can change the result at all.** A reference to a stale name is
only worth arguing about where the slot and the register actually hold different values. For every
stale reference it records the three candidates -- the name's binding, the slot's current contents,
and the register's most recent definition -- and reports how many are *distinct*. A stale reference
whose three candidates coincide is a naming artefact with no semantic content.

**2. It classifies who is ahead.** The shipped reading treats the context block as the register
file. That is faithful exactly when the slot's last writer *is* the register's last definition. A
definition that never reaches the slot (a cache-only write) would leave the slot behind the register
file and make the shipped reading wrong; a slot write that is not a register definition would leave
it ahead. Each stale reference is classified `in-sync` / `register-ahead` / `slot-ahead`, per
register, so the answer is a measurement rather than a defence of the existing choice.

**3. It runs both readings.** `interp.run_block` now takes `reading='slot'|'binding'`; this probe
executes the same block twice on the same entry state and reports where the two readings produce a
different register file. A reading that never changes a result cannot be the reason the differential
passes; a reading that does change results is a live question, and the population it changes is the
population that needs an independent oracle.

    python llvm/probe_register_file_reading.py --listings=out/driver-core3-20261018
    python llvm/probe_register_file_reading.py --listings=out/driver-core3-20261018 --values
    python llvm/probe_register_file_reading.py --listing=PATH --example=4

The third mode prints the worked example: for one stale reference, the statement that defined the
name, every write to its slot between then and the use, and what each reading makes of the operand.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'toolchain_spike'))

import guestir as G  # noqa: E402
import interp as I  # noqa: E402

#: What the engine counts as "a write to the slot". `ctx-load` is included on purpose to reproduce
#: the published 440 exactly, but it is a **read** (`interp.run_block` does `ctx.read`), so counting
#: it as a write cannot leave a reference un-flagged -- it can only flag more. Both figures are
#: therefore reported: `stale_refs` (published definition) and `stale_reads_only` (writes that are
#: really writes).
WRITE_KINDS = ('ctx-write', 'home', 'ctx-partial', 'ctx-load')
TRUE_WRITE_KINDS = ('ctx-write', 'home', 'ctx-partial')


def defined_name(st) -> tuple[str, int] | None:
    dest = st.dest or ''
    if not dest.startswith('guest_'):
        return None
    tail = dest[len('guest_'):]
    head, _, last = tail.rpartition('_')
    if last.isdigit() and head in G.CTX_LAYOUT:
        return head, int(last)
    return None


def written_slot(st, kinds=WRITE_KINDS) -> int | None:
    if st.kind in kinds:
        extra = st.extra or ()
        if len(extra) > 1 and isinstance(extra[1], int):
            return extra[1]
    return None


def refs(expr) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    stack = [expr]
    while stack:
        e = stack.pop()
        if not isinstance(e, tuple) or not e:
            continue
        if isinstance(e[0], str) and e[0] == 'reg':
            out.append((e[1], e[2]))
            continue
        for x in e:
            if isinstance(x, tuple):
                stack.append(x)
            elif isinstance(x, list):
                stack.extend(x)
    return out


def classify(fn, kinds=WRITE_KINDS) -> dict:
    """Every stale reference with the three candidate values and a who-is-ahead verdict."""
    rows = []
    for b in fn.blocks:
        gen: dict[int, int] = {}
        bound: dict[str, tuple[int, int]] = {}          # name -> (slot, generation at definition)
        name_of_gen: dict[tuple[int, int], str] = {}    # (slot, generation) -> the name that made it
        latest_def: dict[str, tuple[str, int]] = {}     # reg -> (name, stmt index)
        latest_slot: dict[str, tuple[str | None, int]] = {}
        for i, st in enumerate(b.stmts):
            for reg, ver in refs(st.expr):
                name = f'guest_{reg}_{ver}'
                hit = bound.get(name)
                if hit is None:
                    continue
                off, at = hit
                now = gen.get(off, 0)
                if now == at:
                    continue
                # The three candidates.
                by_name = name
                by_slot = name_of_gen.get((off, now))
                by_reg = latest_def.get(reg, (None, -1))[0]
                slot_is_reg = latest_slot.get(reg, (None, -1))[0] == by_reg
                rows.append({
                    'block': b.index, 'stmt': i, 'reg': reg, 'name': name,
                    'slot': off, 'generation': now,
                    'reading_name': by_name, 'reading_slot': by_slot, 'reading_register': by_reg,
                    'distinct': len({by_name, by_slot, by_reg}),
                    'in_sync': slot_is_reg,
                    'register_ahead': by_reg is not None and not slot_is_reg
                                     and latest_def[reg][1] > latest_slot.get(reg, (None, -1))[1],
                    'slot_ahead': not slot_is_reg
                                  and latest_slot.get(reg, (None, -1))[1] > latest_def.get(reg, (None, -1))[1],
                    'kind': st.kind, 'dest': st.dest,
                    'text': f'{st.dest} = ' + G._render_canonical(st.expr) if st.expr else st.text.strip(),
                })
            off = written_slot(st, kinds)
            if off is not None:
                gen[off] = gen.get(off, 0) + 1
                d = defined_name(st)
                name_of_gen[(off, gen[off])] = d and f'guest_{d[0]}_{d[1]}'
            d = defined_name(st)
            if d is not None:
                reg, ver = d
                off = G.CTX_LAYOUT[reg]
                nm = f'guest_{reg}_{ver}'
                bound[nm] = (off, gen.get(off, 0))
                latest_def[reg] = (nm, i)
                if written_slot(st, kinds) == off:
                    latest_slot[reg] = (nm, i)
    return {'root': fn.root, 'blocks': len(fn.blocks), 'stale': rows}


def _entry_ctx() -> I.Ctx:
    """The same deterministic entry state `interp.main` uses, so a value here is reproducible."""
    ctx = I.Ctx({reg: 0x1000 + 8 * i for i, reg in enumerate(G.CTX_LAYOUT)})
    ctx.write(G.CTX_LAYOUT['rflags'], 64, 0x202)
    return ctx


def _mem() -> I.Mem:
    """`diff_payload`'s memory: every address readable, unwritten bytes a function of the address."""
    return I.Mem(regions=[(0, 1 << 64)], default=I.seed_free)


def _inputs() -> I.Inputs:
    return I.Inputs(strict=False,
                    undef={k: 0 for k in ('flag_af', 'flag_pf', 'flag_zf', 'flag_sf',
                                          'flag_cf', 'flag_of', 'flag_df')})


def run_both(block, probe=None) -> dict:
    """Execute one block under each reading and report the register file each produces.

    `probe=(reg, version)` appends one synthetic `__probe:64 = guest_<reg>_<version>` statement, so
    the operand's own value under each reading is reported instead of a slot that both readings may
    happen to leave equal.
    """
    blocks = {}
    if probe is not None:
        st = G.Stmt(kind='def', dest='__probe', width=64,
                    expr=('reg', probe[0], probe[1]), text='')
        block = dataclasses.replace(block, stmts=list(block.stmts) + [st])
    out = {}
    for reading in ('slot', 'binding'):
        ctx = _entry_ctx()
        r = I.run_block(block, ctx, _mem(), _inputs(), reading=reading)
        out[reading] = {'slots': ctx.slots(), 'errors': r.errors[:3],
                        'operand': r.temps.get('__probe')}
    differ = [reg for reg in G.CTX_LAYOUT
              if out['slot']['slots'][reg] != out['binding']['slots'][reg]]
    return {'differ': differ, **out}


def example(path: Path, want: int) -> int:
    """Print the worked example for the first `want` stale references of one listing."""
    fn = G.parse_listing(path.read_text(encoding='utf-8', errors='replace'), path)
    rec = classify(fn)
    rows = [r for r in rec['stale'] if r['distinct'] > 1]
    if not rows:
        print(f'{path.name}: no stale reference where the readings differ')
        return 0
    print(f'{path.name}: root {fn.root:#x}, {len(rec["stale"])} stale reference(s), '
          f'{len(rows)} where the readings can differ')
    by_block = {b.index: b for b in fn.blocks}
    for row in rows[:want]:
        b = by_block[row['block']]
        print(f'\n--- block #{row["block"]} stmt {row["stmt"]}  register {row["reg"]} '
              f'(slot {row["slot"]:#x}) ---')
        for i, st in enumerate(b.stmts):
            mark = '>>' if i == row['stmt'] else '  '
            if i <= row['stmt'] and st.kind in WRITE_KINDS + ('def',):
                extra = st.extra or ()
                slot = extra[1] if len(extra) > 1 else None
                if slot == row['slot'] or (defined_name(st) or ('', 0))[0] == row['reg']:
                    print(f'{mark} {i:4} {st.kind:<11} {st.text.strip()[:120]}')
        print(f'   readings at the use: name-bound {row["reading_name"]!r} | '
              f'slot-now {row["reading_slot"]!r} | '
              f'register-file {row["reading_register"]!r}')
        print(f'   in sync (slot last writer == register last definition): {row["in_sync"]}')
        # The values, on a concrete entry state.
        prefix = dataclasses.replace(b, stmts=b.stmts[:row['stmt'] + 1],
                                     cond='', when0='', when1='', succs=[])
        got = run_both(prefix, probe=(row['reg'], int(row['reading_name'].rsplit('_', 1)[1])))
        a = got['slot']['operand']
        c = got['binding']['operand']
        verdict = '** the two readings differ here **' if a != c else 'the two readings agree here'
        print(f'   operand value: slot-now {a:#018x} | name-bound {c:#018x}   {verdict}')
        full = run_both(b)
        if full['differ']:
            print(f'   the whole block ends differently on: {", ".join(full["differ"])}')
            for reg in full['differ']:
                print(f'     {reg:<7} slot-now {full["slot"]["slots"][reg]:#018x}  '
                      f'name-bound {full["binding"]["slots"][reg]:#018x}')
    return 0


def main(argv: list[str]) -> int:
    listings = None
    listing = None
    do_values = False
    ex = 0
    quiet = False
    for a in argv:
        if a.startswith('--listings='):
            listings = Path(a.split('=', 1)[1])
        elif a.startswith('--listing='):
            listing = Path(a.split('=', 1)[1])
        elif a == '--values':
            do_values = True
        elif a.startswith('--example='):
            ex = int(a.split('=', 1)[1])
        elif a == '--quiet':
            quiet = True
    if listing is None and listings is None:
        print(__doc__)
        return 2
    if ex:
        return example(listing, ex)
    paths = [listing] if listing else sorted(listings.glob('0x*.txt'))
    tot = {'files': 0, 'with_stale': 0, 'refs_published': 0, 'refs_reads_only': 0,
           'distinct': 0, 'in_sync': 0, 'register_ahead': 0, 'slot_ahead': 0,
           'value_roots': 0, 'value_registers': 0}
    value_rows = []
    for p in paths:
        fn = G.parse_listing(p.read_text(encoding='utf-8', errors='replace'), p)
        rec = classify(fn)
        strict = classify(fn, TRUE_WRITE_KINDS)
        tot['files'] += 1
        if rec['stale']:
            tot['with_stale'] += 1
        tot['refs_published'] += len(rec['stale'])
        tot['refs_reads_only'] += len(strict['stale'])
        tot['distinct'] += sum(1 for r in rec['stale'] if r['distinct'] > 1)
        tot['in_sync'] += sum(1 for r in rec['stale'] if r['in_sync'])
        tot['register_ahead'] += sum(1 for r in rec['stale'] if r['register_ahead'])
        tot['slot_ahead'] += sum(1 for r in rec['stale'] if r['slot_ahead'])
        if not quiet and rec['stale']:
            print(f"{p.name}  {len(rec['stale']):>4} stale ({len(strict['stale']):>4} counting only real "
                  f"writes)  {sum(1 for r in rec['stale'] if r['distinct'] > 1):>4} distinct  "
                  f"{sum(1 for r in rec['stale'] if not r['in_sync']):>3} out of sync")
        if do_values:
            for b in fn.blocks:
                if not any(r['block'] == b.index for r in rec['stale']):
                    continue
                got = run_both(b)
                if got['differ']:
                    tot['value_roots'] += 1
                    tot['value_registers'] += len(got['differ'])
                    value_rows.append({'root': f'{fn.root:#x}', 'block': b.index,
                                       'registers': got['differ'],
                                       'slot': {r: f'{got["slot"]["slots"][r]:#x}' for r in got['differ']},
                                       'binding': {r: f'{got["binding"]["slots"][r]:#x}'
                                                   for r in got['differ']}})
                    print(f'  ** block #{b.index}: the two readings disagree on '
                          f'{", ".join(got["differ"])}')
    print(f"\n{tot['files']} listing(s), {tot['with_stale']} with at least one stale reference")
    print(f"  {tot['refs_published']} reference(s) under the published count "
          f"(ctx-load counted as a write)")
    print(f"  {tot['refs_reads_only']} reference(s) when only real slot writes advance the generation")
    print(f"  {tot['distinct']} where the name, the slot and the register file are not all the same")
    print(f"  {tot['in_sync']} in sync (slot's last writer == register's last definition)")
    print(f"  {tot['register_ahead']} register-ahead, {tot['slot_ahead']} slot-ahead")
    if do_values:
        print(f"  readings differ on {tot['value_roots']} block(s), "
              f"{tot['value_registers']} register(s) of the final state")
        out = ROOT / 'out' / 'register-file-reading.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'totals': tot, 'value_rows': value_rows}, indent=2),
                       encoding='utf-8')
        print(f'  wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
