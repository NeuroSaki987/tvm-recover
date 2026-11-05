"""Reproducible bounded PE32 cohort audit; never executes the target process.

python -m probe.pe32_sweep --inventory=../out/pe32-delta-inventory-20261104.json --out=../out/pe32-20261104/delta-sweep
"""
import hashlib
import json
from pathlib import Path
import sys
import time

from core.devirt import devirtualize
from core.entries import analyze_entries
from core.peimg import PeImage


def main():
    options = dict(arg[2:].split('=', 1) for arg in sys.argv[1:] if arg.startswith('--'))
    inventory = json.loads(Path(options['inventory']).read_text(encoding='utf-8'))
    out = Path(options['out'])
    out.mkdir(parents=True, exist_ok=True)
    root_limit = int(options.get('roots', 2))
    module_limit = int(options.get('modules', 6))
    steps, blocks = int(options.get('steps', 2000)), int(options.get('blocks', 8))
    seen, results = set(), []
    core_hash = hashlib.sha256()
    for source in sorted((Path(__file__).resolve().parents[1] / 'core').glob('*.py')):
        core_hash.update(source.name.encode())
        core_hash.update(source.read_bytes())
    for row in inventory:
        if row['machine'] != '0x14c':
            continue
        path = Path(row['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        image = PeImage(str(path))
        vm = image.section_by_name('.tvm0')
        entries = analyze_entries(image)
        candidates = [r for r in entries.candidates if r.status != 'rejected']
        validated = sorted({r.target for r in candidates if r.status == 'validated'})
        unproven = sorted({r.target for r in candidates if r.status != 'validated'} - set(validated))
        roots = (validated + unproven)[:root_limit]
        result = dict(path=str(path), sha256=digest, engine_sha256=core_hash.hexdigest(), steps_per_block=steps,
                      block_budget=blocks, candidate_count=len(candidates),
                      entry_statuses={s: sum(r.status == s for r in entries.candidates)
                                      for s in ('validated', 'unproven', 'rejected')}, roots=[])
        for root in roots:
            start = time.monotonic()
            try:
                fn = devirtualize(image, vm, root, steps=steps, blocks=blocks)
                folder = out / (path.stem + '-' + digest[:8])
                folder.mkdir(exist_ok=True)
                (folder / f'{root:#x}.txt').write_text(fn.text, encoding='utf-8')
                state = fn.guest_state.to_dict()
                (folder / f'{root:#x}.guest-state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
                measured = dict(root=hex(root), blocks=fn.blocks, unresolved=fn.unresolved,
                                instructions=fn.instructions, truncated=fn.truncated,
                                guest_effect_blocks=fn.emitted.blocks_with_effect,
                                verifier_errors=state['validation']['verifier_errors'],
                                roundtrip_errors=state['validation']['roundtrip_errors'],
                                stops=[dict(kind=b.kind, detail=b.detail) for b in fn.cfg.blocks.values()])
            except Exception as exc:
                measured = dict(root=hex(root), error=f'{type(exc).__name__}: {exc}')
            measured['seconds'] = round(time.monotonic() - start, 3)
            measured['entry_status'] = 'validated' if root in validated else 'unproven'
            result['roots'].append(measured)
            print(path.name, hex(root), json.dumps(measured), flush=True)
        results.append(result)
        (out / 'measurements.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        if len(results) >= module_limit:
            break
    return int(any('error' in r or r.get('verifier_errors') or r.get('roundtrip_errors')
                   for m in results for r in m['roots']))


if __name__ == '__main__':
    raise SystemExit(main())
