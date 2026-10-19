"""Why each block of a root has no successor.

`core.devirt` prints one line per block and, for a block whose exit was not settled, prints an
arrow saying so. That is enough for a reader to know a root is incomplete and not enough for
anyone to close it: "the dispatcher target was not recovered" is a *symptom*, and the listing
does not carry the stop site, the kind the CFG assigned, or the destination expression the
resolver failed on.

This probe prints exactly that, per block: `kind`, the recovery's own `detail`, the number of
transfers and instructions the block recorded, the native calls it made, and the destination
expression in the form the resolver saw. It is a *measurement*, not a repair: it makes no claim
about which of the reasons is fixable.

    python -m probe.block_exits <module> --roots 0x145bd3fc0,0x145c24c0b
    python -m probe.block_exits <module> --refusals ../out/driver-core3-closure/payload.json

`--refusals` reads a payload.json and takes the refused roots from it, so the set examined is the
set that was actually refused rather than a list copied by hand into a command line.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from core.devirt import DEFAULT_BLOCKS, DEFAULT_STEPS, devirtualize  # noqa: E402
from core.peimg import PeImage  # noqa: E402


def refused_roots(payload_json: str):
    """The roots a payload build refused, with the reason it gave."""
    with open(payload_json, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for r in data.get("refusals") or ():
        out.append(int(r["root"], 16))
    return out


def parse_roots(text: str):
    return [int(x, 0) for x in text.split(",") if x.strip()]


def prove(img: PeImage, fn, node: int, max_values: int, table_limit: int) -> dict:
    """Re-run the bit-vector image proof on one destination with a wider budget.

    The recovery's own call runs with `max_values=16` and a 350 ms deadline. Those are the
    numbers that decide whether a dispatch is called unresolved, purely, so a diagnostic that
    only prints "unresolved" hides which limit fired. This reports the status and, when the
    image is finite, how many of the values are executable code.
    """
    from core.bitproof import finite_values

    proof = finite_values(fn.arena, node, {}, max_values=max_values, table_limit=table_limit,
                          timeout_ms=20000)
    ex = [v for v in proof.values
          if (lambda s: s is not None and s.executable)(img.section_for_va(v))]
    return {"status": proof.status, "values": len(proof.values), "executable": len(ex),
            "tables": proof.tables, "checks": proof.checks,
            "sample": [f"{v:#x}" for v in proof.values[:6]]}


def leaves(fn, node: int, limit: int = 12) -> list[dict]:
    """The genuinely unknown leaves of a destination expression, with how each is spelled.

    A dispatch the resolver cannot settle is settled by *something it does not know*, and the
    question that decides whether it is fixable is which: an `init` register (recoverable, it is the
    entry state), an `opaque` (a value the model deliberately dropped), or a `load` (guest memory,
    whose contents are not in the model at all). The render is the engine's own spelling.
    """
    from core.expr import render

    a = fn.arena
    out: list[dict] = []
    seen: set[int] = set()
    stack = [node]
    while stack and len(out) < limit:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        kind = a.node(cur)[0]
        if kind in ("load", "opaque", "param", "init"):
            out.append({"kind": kind, "node": cur, "text": render(a, cur)[:160]})
            continue
        for child in children_of(a, cur):
            stack.append(child)
    return out


def children_of(a, ref: int) -> list[int]:
    n = a.node(ref)
    kind = n[0]
    if kind in ("trunc", "zext", "sext", "un"):
        return [n[1] if kind.startswith(("trunc", "zext", "sext")) else n[2]]
    if kind == "bin":
        return [n[2], n[3]]
    if kind == "sel":
        return [n[1], n[2], n[3]]
    if kind == "load":
        return [n[1]]
    return []


def describe(img: PeImage, vm, root: int, steps: int, blocks: int,
             prove_limit: int = 0, leaf_limit: int = 0) -> dict:
    """Devirtualise one root and report every block whose exit was not settled."""
    fn = devirtualize(img, vm, root, steps=steps, blocks=blocks)
    rows = []
    for b in fn.blocks_by_label.values():
        if b.succs:
            continue
        row = {
            "label": str(getattr(b, "key", "")),
            "kind": b.kind,
            "detail": getattr(b, "detail", ""),
            "predicate": getattr(b, "predicate", ""),
            "transfers": len(getattr(b, "transfers", ()) or ()),
            "instructions": len(getattr(b, "guest_trace", ()) or ()),
            "calls": [f"{name} @{site:#x}" for site, name in (b.calls or ())],
        }
        node = getattr(b, "pred_node", 0)
        if node and getattr(b, "kind", "") == "unresolved":
            if leaf_limit:
                row["leaves"] = leaves(fn, node, leaf_limit)
            if prove_limit:
                row["proof"] = prove(img, fn, node, prove_limit, prove_limit * 4)
        rows.append(row)
    return {
        "root": root,
        "blocks": fn.blocks,
        "unresolved": fn.unresolved,
        "proofs": (fn.recovery or {}).get("proofs"),
        "rows": rows,
        "text_lines": fn.text.count("\n"),
    }


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    roots: list[int] = []
    steps, blocks = DEFAULT_STEPS, DEFAULT_BLOCKS
    prove_limit = 0
    leaf_limit = 0
    as_json = "--json" in sys.argv
    for x in sys.argv[1:]:
        if x.startswith("--roots="):
            roots.extend(parse_roots(x.split("=", 1)[1]))
        if x.startswith("--refusals="):
            roots.extend(refused_roots(x.split("=", 1)[1]))
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--blocks="):
            blocks = int(x.split("=", 1)[1], 0)
        if x.startswith("--prove="):
            prove_limit = int(x.split("=", 1)[1], 0)
        if x.startswith("--leaves="):
            leaf_limit = int(x.split("=", 1)[1], 0)
    if not args or not roots:
        print(__doc__)
        return 2

    img = PeImage(args[0])
    vm = img.section_by_name(".tvm0")
    records = []
    for root in roots:
        rec = describe(img, vm, root, steps, blocks, prove_limit, leaf_limit)
        records.append(rec)
        if as_json:
            continue
        print(f"{root:#x}  {rec['blocks']} block(s)  {rec['unresolved']} unresolved  "
              f"{rec['text_lines']} line(s)  proofs={rec['proofs']}")
        for r in rec["rows"]:
            print(f"    [{r['kind']}] {r['label']}  detail={r['detail']!r}  "
                  f"transfers={r['transfers']}  insns={r['instructions']}")
            if r["calls"]:
                print(f"        calls: {', '.join(r['calls'])}")
            if r["predicate"]:
                print(f"        destination: {r['predicate'][:220]}")
            for lf in r.get("leaves") or ():
                print(f"        leaf {lf['kind']:<7} {lf['text']}")
            if "proof" in r:
                p = r["proof"]
                print(f"        proof: {p['status']}  values={p['values']}  "
                      f"executable={p['executable']}  tables={p['tables']}  "
                      f"checks={p['checks']}  sample={p['sample']}")
        print()
    if as_json:
        print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
