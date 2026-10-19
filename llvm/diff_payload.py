"""The write-devirt differential: the code that runs **inside the patched PE** against the interpreter.

`diff.py` compares a listing with itself through two compilations; `diff_whole.py` compares a whole
recovered function compiled as a DLL against the interpreter. This module closes the last gap: the
bytes that were actually appended to the host image are mapped the way the loader maps them, the
payload's own entry points are called through the payload's own header, and the resulting register
file and sealed memory are compared **byte for byte** with a Python walk of the same listing.

Four things make the comparison meaningful rather than decorative:

* **The register seed is read out of the payload**, not recomputed (`tvm_ctx_seed`, published in the
  header), and the witness assignment is the one the header records. If the payload ran a different
  experiment from the interpreter there would be nothing to compare.
* **The memory model is the same function of the address on both sides.** The payload answers an
  unwritten byte with `seed_free(addr)`; the interpreter is given `interp.Mem` with one region
  covering the whole address space and `default=seed_free`. The bound that `diff.py` needs -- so that
  a native fault and an interpreter error are comparable -- is *not* wanted here: the sealed accessors
  cannot fault, so bounding the interpreter would only discard blocks the payload legitimately ran.
* **Steps are compared, not just state.** A transcription that computes the right bytes and takes the
  wrong exit is invisible to a state-only comparison.
* **Every function is reported, including refusals.** A root whose blocks the listing does not settle
  never reaches the payload (`whole.WholeRefusal`), and the report says so per root instead of
  quietly containing twelve of thirteen.

Run:  python llvm/diff_payload.py [--payload=DIR] [--patched=PATH] [--root=HEX] [--json]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'toolchain_spike'))

import guestir as G  # noqa: E402
import interp as I  # noqa: E402
import miniload as M  # noqa: E402
import payload as P  # noqa: E402
import peedit  # noqa: E402

DEFAULT_PAYLOAD = HERE / 'build' / 'write-devirt'
DEFAULT_LISTINGS = ROOT / 'out' / 'hard-cases-listings-final'


def unbounded_mem() -> I.Mem:
    """`interp.Mem` with the payload's memory semantics: every address readable, seed-free default.

    The whole 64-bit space is one region because that is exactly what the sealed accessors model --
    they cannot fault and they never refuse an address. See the module docstring for why the bounded
    form used by `diff.py` would be the wrong experiment here.
    """
    return I.Mem(regions=[(0, 1 << 64)], default=I.seed_free)


def compare(payload_dir: Path = DEFAULT_PAYLOAD, patched: Path | None = None,
            listings: Path = DEFAULT_LISTINGS, only_root: str | None = None) -> dict:
    meta = json.loads((payload_dir / 'payload.json').read_text(encoding='ascii'))
    patched = patched or (payload_dir / 'payload_dll_exe')
    if not Path(patched).exists():
        raise SystemExit(f'no patched image at {patched}; run llvm/peinject.py first')
    data = Path(patched).read_bytes()

    paths = sorted(Path(listings).glob('0x*.txt'))
    mod = P.emit_module(paths)  # regenerates the witness table; the IR itself is not needed here
    table = mod['table']
    by_root = {}
    for p in paths:
        fn = G.parse_listing(p.read_text(encoding='utf-8', errors='replace'), p)
        by_root[f'{fn.root:#x}'] = fn

    image = M.Mapped(data, None, str(patched))
    mapped_at, preferred = image.base, image.pe.image_base
    relocs_applied = image.relocs_applied
    try:
        header = M.neko_header(image)
        results = []
        for i in range(header['nfunc']):
            fn_meta = header['functions'][i]
            root = f"{fn_meta['root']:#x}"
            if only_root and only_root.lower().replace('0x', '') not in root.lower():
                continue
            rec = {'index': i, 'root': root, 'name': fn_meta['name'],
                   'blocks': fn_meta['blocks'], 'statements': fn_meta['statements']}
            try:
                native = M.run_function(image, header, i)
            except OSError as exc:
                rec.update(verdict='NATIVE FAULT', detail=str(exc))
                results.append(rec)
                continue
            fn = by_root.get(root)
            if fn is None:
                rec.update(verdict='NO LISTING', detail=f'{root} is in the payload but not in '
                                                        f'{listings}')
                results.append(rec)
                continue
            ctx = I.Ctx()
            ctx.mem = bytearray(native['seed'])
            mem = unbounded_mem()
            # `strict=False`: the payload has no harness to drive it, so every value it cannot derive
            # -- a register after a native call, the flags after one, a call's result -- comes from the
            # documented default that `interp.default_call*` mirrors. Driving the interpreter in strict
            # mode instead makes it substitute its own fallback for a missing source, and the two
            # readings then differ in exactly the way a differential test exists to catch. (Measured:
            # strict mode produced a one-bit difference in rdx on 0x1401687d5, in a packed flags word.)
            inputs = I.Inputs(undef={name: header['witness_value'] for name in table}, strict=False)
            try:
                rr = I.run_function(fn, ctx, mem, inputs, fuel=header['fuel'])
            except Exception as exc:  # noqa: BLE001 - an interpreter refusal is a result
                rec.update(verdict='INTERPRETER REFUSED', detail=f'{type(exc).__name__}: {exc}')
                results.append(rec)
                continue

            raw = native['steps']
            # The payload encodes fuel exhaustion as `~steps` -- the convention `whole.py` emits, so a
            # budget stop can never be mistaken for a completed run of that many blocks. Decoding it
            # here is what makes "both sides ran out of fuel at the same point" a *match* rather than
            # a difference between 2000 and 0xfffffffffffff82f.
            from execution_status import decode
            native_status = decode(raw)
            native_finished = native_status['finished']
            native_steps = native_status['steps']
            same_steps = (rr.steps == native_steps and bool(rr.finished) == native_finished
                          and bool(rr.exhausted) == native_status['exhausted']
                          and bool(rr.terminal) == native_status['terminal'])
            same_ctx = ctx.snapshot() == native['ctx']
            sealed = native['sealed']
            same_mem = sealed == {a: v for a, v in mem.bytes.items()}
            rec.update({
                'steps_python': rr.steps, 'steps_payload': native_steps, 'return_raw': raw,
                'finished_python': bool(rr.finished), 'finished_payload': native_finished,
                'exhausted_python': bool(rr.exhausted),
                'status_payload': native_status['status'], 'terminal_python': rr.terminal,
                'mem_bytes_payload': len(sealed), 'mem_bytes_python': len(mem.bytes),
                'steps_match': same_steps, 'ctx_match': same_ctx, 'mem_match': same_mem,
                'visited': list(rr.visited[:16]),
            })
            if same_steps and same_ctx and same_mem:
                rec['verdict'] = 'OK'
            else:
                rec['verdict'] = 'DIVERGENT'
                rec['detail'] = first_difference(native, ctx, sealed, mem)
            results.append(rec)
    finally:
        image.free()

    ok = sum(1 for r in results if r['verdict'] == 'OK')
    return {
        'patched': str(patched), 'payload': str(payload_dir),
        'mapped_base': hex(mapped_at), 'preferred_base': hex(preferred),
        'relocs_applied': relocs_applied, 'fuel': header['fuel'],
        'witnesses': len(table), 'witness_value': header['witness_value'],
        'functions': results, 'ok': ok, 'total': len(results),
        'refusals': meta.get('refusals', []),
    }


def first_difference(native: dict, ctx: I.Ctx, sealed: dict, mem: I.Mem) -> str:
    """A named, specific first disagreement -- or an empty string if the three checks agreed."""
    if ctx.snapshot() != native['ctx']:
        exp, got = ctx.snapshot(), native['ctx']
        for i in range(len(exp)):
            if exp[i] != got[i]:
                reg = next((r for r, o in G.CTX_LAYOUT.items() if o <= i < o + 8), '?')
                return f'context differs at byte {i} ({reg}): python {exp[i]:#02x} payload {got[i]:#02x}'
    if sealed != dict(mem.bytes):
        for a in sorted(set(sealed) | set(mem.bytes)):
            if sealed.get(a) != mem.bytes.get(a):
                return (f'sealed memory differs at {a:#x}: python '
                        f'{mem.bytes.get(a)!r} payload {sealed.get(a)!r}')
    return 'steps differ'


def main(argv: list[str]) -> int:
    payload_dir = DEFAULT_PAYLOAD
    patched = None
    listings = DEFAULT_LISTINGS
    only = None
    as_json = False
    for a in argv:
        if a.startswith('--payload='):
            payload_dir = Path(a.split('=', 1)[1])
        elif a.startswith('--patched='):
            patched = Path(a.split('=', 1)[1])
        elif a.startswith('--listings='):
            listings = Path(a.split('=', 1)[1])
        elif a.startswith('--root='):
            only = a.split('=', 1)[1]
        elif a == '--json':
            as_json = True
        elif a in ('-h', '--help'):
            print(__doc__)
            return 2
    rep = compare(payload_dir, patched, listings, only)
    if as_json:
        print(json.dumps(rep, indent=2))
        return 0 if rep['ok'] == rep['total'] else 1
    print(f"payload {rep['payload']}  patched {rep['patched']}")
    print(f"mapped at {rep['mapped_base']} (preferred {rep['preferred_base']}, "
          f"{rep['relocs_applied']} host relocation(s) applied), fuel {rep['fuel']}, "
          f"{rep['witnesses']} witness(es) = {rep['witness_value']}")
    for r in rep['functions']:
        line = (f"  {r['root']:<12} {r['verdict']:<20} blocks {r['blocks']:>3}  "
                f"steps {r.get('steps_python')}/{r.get('steps_payload')}  "
                f"sealed {r.get('mem_bytes_payload')}/{r.get('mem_bytes_python')} B")
        print(line)
        if r['verdict'] != 'OK':
            print(f"      {r.get('detail', '')[:160]}")
    print(f"\n{rep['ok']}/{rep['total']} function(s) agree byte for byte between the payload "
          f"running in the patched image and the interpreter")
    for r in rep['refusals']:
        print(f"  not in the payload: {r['root']} -- {r['reason'][:110]}")
    return 0 if rep['ok'] == rep['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
