"""The whole-function differential test: run a recovered function, not just a block.

`diff.py` checks each block on its own. That is a real check and it is not the interesting one: a
devirtualized function is a *composition*, and a composition can be wrong in ways no block is. So this
emits each recovered function as one LLVM function (`llvm/whole.py`), compiles and links it, calls it
with a fuel budget, and compares against `interp.run_function` walking the same CFG.

Four things are compared, and the fourth is the point:

* the whole context block, byte for byte;
* every guest address either side named, byte for byte;
* **how many blocks each side executed** -- a transcription that computes the right state and takes
  the wrong exit is invisible to a memory-only comparison;
* whether the run finished, ran out of fuel, or stopped for a reason the listing does not settle.

The encoded return is `steps` on a normal finish and `~steps` on fuel exhaustion, so a budget stop can
never be mistaken for a completion.

Run:  python llvm/diff_whole.py [--root=HEX] [--fuel=N] [--opt=0,2] [--listings=DIR]
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'toolchain_spike'))

import diff  # noqa: E402
import guestir as G  # noqa: E402
import interp as I  # noqa: E402
import whole as W  # noqa: E402
from execution_status import decode  # noqa: E402

def _boot_env() -> None:
    """Load `main/.env` if present. `main/` is not on sys.path for a llvm/ script, so walk up."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'bootenv.py').is_file():
            text = str(parent)
            if text not in sys.path:
                sys.path.insert(0, text)
            try:
                from bootenv import boot
                boot(here)
            except Exception:
                pass
            return


_boot_env()

MAX_EXHAUSTIVE = 8
MAX_SAMPLES = 8

# Default input is the corpus whose forks state their condition; the certified listings cannot be
# composed into a function because most of their blocks' exits are not settled.
BRANCH_LISTINGS = Path(__file__).resolve().parent.parent / 'out' / 'branch-conditions'


def _cases(table: list[str]) -> tuple[list[dict], str]:
    """Witness assignments. Exhaustive while that is affordable, sampled otherwise -- and said."""
    if len(table) <= MAX_EXHAUSTIVE:
        return ([{w: (m >> i) & 1 for i, w in enumerate(table)}
                 for m in range(1 << len(table))],
                f'exhaustive over 2^{len(table)}')
    cases = [{w: 0 for w in table}, {w: 1 for w in table}]
    for s in range(MAX_SAMPLES):
        h = (s * 0x9E3779B1 + 0x5EED) & 0x7FFFFFFF
        a = {}
        for w in table:
            h = (h * 1103515245 + 12345) & 0x7FFFFFFF
            a[w] = (h >> 16) & 1
        cases.append(a)
    return cases, f'sampled {len(cases)} of 2^{len(table)}'


def _ctx_seed() -> dict[str, int]:
    seed = {}
    for i, (reg, _off) in enumerate(G.CTX_LAYOUT.items()):
        if reg == 'rsp':
            seed[reg] = 0x7FFFFFFEFDA0
        elif reg == 'rbp':
            seed[reg] = 0x7FFFFFFEFD40
        elif reg == 'rflags':
            seed[reg] = 0x202
        else:
            seed[reg] = diff.SCRATCH_LO + 0x1100 + 0x10 * i
    return seed


def compare_function(fn, fuel: int, opt: int = 2) -> dict:
    res = {'root': f'{fn.root:#x}', 'blocks': len(fn.blocks), 'verdict': 'REFUSED', 'opt': opt,
           'ok': False, 'cases': []}
    try:
        text, man = W.emit_whole(fn)
    except W.WholeRefusal as exc:
        res['detail'] = str(exc)
        return res
    problems = __import__('lower_llvm').hazard_lint(text)
    if problems:
        res['verdict'] = 'HAZARD'
        res['detail'] = '; '.join(problems[:3])
        return res

    dll = diff.build_module(text, f'whole_{fn.root:x}', opt=opt)
    lib = ctypes.WinDLL(str(dll))
    call = getattr(lib, man['function'])
    call.restype = ctypes.c_int64
    call.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    set_undef = lib.tvm_set_undef
    set_undef.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.tvm_clear_undef.argtypes = []
    table = man['undef_table']

    cases, mode = _cases(table)
    res['cases_mode'] = mode
    res['witness_count'] = len(table)
    res['cases'] = []
    addrs = man['guest_addrs']
    region = diff.Region(list(addrs) + [diff.SCRATCH_LO + k
                                        for k in range(0, diff.SCRATCH_SIZE, diff.PAGE)]) \
        if addrs else diff.Region([diff.SCRATCH_LO])
    ok = True
    try:
        for ci, vals in enumerate(cases):
            region.seed()
            # --- our reading
            ctx_a = I.Ctx(dict(_ctx_seed()))
            mem_a = I.Mem(regions=region.spans, default=diff.seed_byte)
            rr = I.run_function(fn, ctx_a, mem_a,
                                I.Inputs(undef=vals, strict=False), fuel=fuel)
            # --- the code LLVM produced
            ctx_b = I.Ctx(dict(_ctx_seed()))
            buf = ctypes.create_string_buffer(bytes(ctx_b.mem), len(ctx_b.mem))
            lib.tvm_clear_undef()
            for i, name in enumerate(table):
                set_undef(i, vals.get(name, 0))
            fault = ''
            ret = None
            try:
                ret = int(call(ctypes.cast(buf, ctypes.c_void_p), fuel))
            except OSError as exc:
                fault = f'native call faulted: {exc}'
            native_ctx = bytes(buf.raw[:len(ctx_b.mem)])
            native_status = decode(ret) if ret is not None else None
            native_finished = native_status is not None and native_status['finished']
            native_steps = native_status['steps'] if native_status is not None else -1

            both_faulted = rr.faulted and bool(fault)
            if both_faulted:
                res['cases'].append({'case': ci, 'verdict': 'not_exercised',
                                     'detail': rr.errors[:1]})
                continue

            same_ctx = ctx_a.snapshot() == native_ctx
            same_steps = rr.steps == native_steps
            same_end = (native_status is not None and rr.finished == native_finished
                        and rr.exhausted == native_status['exhausted']
                        and bool(rr.terminal) == native_status['terminal'])
            same_mem = True
            diff_mem = ''
            watched = set(addrs) | set(mem_a.bytes)
            for a in sorted(watched):
                for k in range(8):
                    want = mem_a.bytes.get(a + k, diff.seed_byte(a + k)) & 0xFF
                    try:
                        got = region.read(a + k)
                    except KeyError:
                        continue
                    if want != got:
                        same_mem = False
                        diff_mem = f'memory differs at {a + k:#x}: interp {want:#02x} native {got:#02x}'
                        break
                if not same_mem:
                    break
            first = ''
            if not same_ctx:
                exp = ctx_a.snapshot()
                for i in range(len(exp)):
                    if exp[i] != native_ctx[i]:
                        reg = next((r for r, o in G.CTX_LAYOUT.items() if o <= i < o + 8), '?')
                        first = (f'context differs at byte {i} ({reg}): '
                                 f'interp {exp[i]:#02x} native {native_ctx[i]:#02x}')
                        break
            good = same_ctx and same_mem and same_steps and same_end and not rr.errors and not fault
            ok = ok and good
            res['cases'].append({
                'case': ci, 'verdict': 'ok' if good else 'divergent',
                'ctx_match': same_ctx, 'mem_match': same_mem,
                'steps interp': rr.steps, 'steps native': native_steps,
                'finished interp': rr.finished, 'finished native': native_finished,
                'status native': native_status['status'] if native_status else 'fault',
                'terminal interp': rr.terminal,
                'detail': first or diff_mem or ('' if same_steps else 'step count differs'),
                'stuck': rr.stuck, 'errors': rr.errors[:2], 'fault': fault,
            })
    finally:
        region.free()

    exercised = [c for c in res['cases'] if c['verdict'] != 'not_exercised']
    res['exercised'] = len(exercised)
    res['unexercised'] = len(res['cases']) - len(exercised)
    res['visit_trace'] = rr.visited[:12]
    res['ok'] = bool(exercised) and ok
    res['verdict'] = ('OK' if res['ok'] else
                      'NOT EXERCISED' if not exercised else 'FAIL')
    return res


def main(argv: list[str]) -> int:
    only_root = None
    fuel = 2000
    opts = [2]
    d = BRANCH_LISTINGS
    for a in argv:
        if a.startswith('--root='):
            only_root = a.split('=', 1)[1].lower().replace('0x', '')
        elif a.startswith('--fuel='):
            fuel = int(a.split('=', 1)[1], 0)
        elif a.startswith('--opt='):
            opts = [int(x) for x in a.split('=', 1)[1].split(',')]
        elif a.startswith('--listings='):
            d = Path(a.split('=', 1)[1])
    files = sorted(d.glob('0x*.txt'))
    results = []
    for p in files:
        if only_root and only_root not in p.stem.lower():
            continue
        fn = G.parse_listing(p.read_text(encoding='utf-8', errors='replace'), p)
        per = {o: compare_function(fn, fuel, o) for o in opts}
        r = per[opts[0]]
        r['per_opt'] = {str(o): v['verdict'] for o, v in per.items()}
        r['ok'] = all(v['ok'] for v in per.values())
        results.append(r)
        tag = '/'.join(f'O{o}={v["verdict"]}' for o, v in per.items())
        if r['verdict'] == 'REFUSED':
            print(f'{r["root"]:<12} REFUSED   {len(str(r.get("detail","")))} chars: '
                  f'{str(r.get("detail",""))[:110]}')
            continue
        steps = r['cases'][0].get('steps interp') if r['cases'] else '?'
        print(f'{r["root"]:<12} {r["verdict"]:<13} blocks {r["blocks"]:>3}  {tag}  '
              f'{r["cases_mode"]}, {r["witness_count"]} witness(es)  '
              f'steps {steps}  path {r.get("visit_trace")}')
        if r['verdict'] == 'FAIL':
            for c in r['cases']:
                if c['verdict'] == 'divergent':
                    print(f'      case {c["case"]}: {c["detail"]}  '
                          f'steps {c["steps interp"]}/{c["steps native"]}  '
                          f'finished {c["finished interp"]}/{c["finished native"]}  '
                          f'errors={c["errors"]} {c["fault"]}')
    good = sum(1 for r in results if r['ok'])
    refused = sum(1 for r in results if r['verdict'] == 'REFUSED')
    notex = sum(1 for r in results if r['verdict'] == 'NOT EXERCISED')
    bad = len(results) - good - refused - notex
    print()
    print(f'{good}/{len(results)} recovered function(s) agree end to end (fuel {fuel}, '
          f'codegen {opts}); {bad} divergent; {notex} not exercised; {refused} refused')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
