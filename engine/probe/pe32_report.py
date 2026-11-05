"""Summarize a bounded PE32 cohort without upgrading sampled roots to proof.

The input is the ``measurements.json`` written by :mod:`probe.pe32_sweep`.
The report keeps the measured budget, entry-validation state, state projection,
and stop reasons together so downstream notes cannot quote one without the
others.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from core.peimg import PeImage


def _family(path: str) -> str:
    """Coarse path family for reporting; this is not a VM-layout inference."""
    text = path.replace("/", "\\").casefold()
    name = Path(path).name.casefold()
    if "\\anticheatexpert\\" in text and name.startswith("ace-"):
        return "ACE x86"
    if name in {"tersafe.dll", "tenslx.dll", "teniotpf.dll", "tcj.dll",
                "tss_api.dll"}:
        return "TenProtect/TerSafe"
    if (name.startswith("tas") or "\\tenprotect\\" in text
            or "\\tcls\\tenio\\" in text):
        return "TenProtect/TAS"
    if name in {"wgcore.dll", "wglogin.dll", "common.dll", "tinyget.dll",
                "meta_resource.dll"}:
        return "WeGame shell"
    return "other/game helper"


def _projection(listing: Path) -> str:
    try:
        with listing.open(encoding="utf-8") as source:
            for _ in range(8):
                line = source.readline()
                match = re.match(r"; state projection: (\S+)", line)
                if match:
                    return match.group(1)
    except OSError:
        pass
    return "missing"


def _fingerprints(path: Path) -> dict:
    image = PeImage(str(path))
    layout = [
        [section.name, section.vsize, section.rsize, section.chars]
        for section in image.sections
    ]
    imported_dlls = sorted({dll.casefold() for dll, _symbol
                            in image.imports.values()})
    return {
        "section_layout_sha256": hashlib.sha256(
            json.dumps(layout, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "import_dll_set_sha256": hashlib.sha256(
            json.dumps(imported_dlls, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def summarize(measurements_path: Path, inventory_path: Path | None) -> dict:
    modules = json.loads(measurements_path.read_text(encoding="utf-8"))
    roots = [root for module in modules for root in module["roots"]]
    projections = {"guest_backing": 0, "native_i386": 0, "missing": 0}
    stops: dict[str, int] = {}
    families: dict[str, dict] = {}
    exact_groups: dict[tuple[str, str], list[str]] = {}

    for module in modules:
        path = Path(module["path"])
        family = _family(module["path"])
        bucket = families.setdefault(
            family,
            {"files": 0, "roots": 0, "blocks": 0, "unresolved": 0,
             "instructions": 0, "guest_effect_blocks": 0, "truncated_roots": 0},
        )
        bucket["files"] += 1
        artifact = measurements_path.parent / (path.stem + "-" + module["sha256"][:8])
        fingerprints = _fingerprints(path)
        group_key = (
            fingerprints["section_layout_sha256"],
            fingerprints["import_dll_set_sha256"],
        )
        exact_groups.setdefault(group_key, []).append(module["path"])
        for root in module["roots"]:
            bucket["roots"] += 1
            for field in ("blocks", "unresolved", "instructions",
                          "guest_effect_blocks"):
                bucket[field] += root[field]
            bucket["truncated_roots"] += int(root["truncated"])
            projection = _projection(artifact / f'{root["root"]}.txt')
            projections[projection] = projections.get(projection, 0) + 1
            for stop in root["stops"]:
                detail = stop["detail"]
                if stop["kind"] == "unresolved" and "step budget" in detail:
                    kind = "step_budget"
                elif "native_call_abi" in detail:
                    kind = "native_call_abi"
                elif "dispatcher target" in detail:
                    kind = "dispatcher_target"
                elif "unresolved branch" in detail:
                    kind = "symbolic_or_unrecovered_branch"
                else:
                    kind = stop["kind"]
                stops[kind] = stops.get(kind, 0) + 1

    result = {
        "schema": 1,
        "source_measurements": str(measurements_path),
        "engine_sha256": modules[0]["engine_sha256"] if modules else None,
        "budget": (
            {
                "steps_per_block": modules[0]["steps_per_block"],
                "blocks_per_root": modules[0]["block_budget"],
                "roots_per_module": max(len(module["roots"]) for module in modules),
            }
            if modules else {}
        ),
        "population": {
            "content_distinct_files": len(modules),
            "selected_roots": len(roots),
            "validated_roots": sum(root["entry_status"] == "validated"
                                   for root in roots),
            "unproven_roots": sum(root["entry_status"] != "validated"
                                  for root in roots),
        },
        "results": {
            "blocks": sum(root["blocks"] for root in roots),
            "unresolved_blocks": sum(root["unresolved"] for root in roots),
            "instructions": sum(root["instructions"] for root in roots),
            "guest_effect_blocks": sum(root["guest_effect_blocks"] for root in roots),
            "truncated_roots": sum(bool(root["truncated"]) for root in roots),
            "verifier_errors": sum(len(root["verifier_errors"]) for root in roots),
            "roundtrip_errors": sum(len(root["roundtrip_errors"]) for root in roots),
            "exceptions": sum("error" in root for root in roots),
        },
        "state_projection": projections,
        "stop_reasons": dict(sorted(stops.items())),
        "path_families": dict(sorted(families.items())),
        "exact_binary_fingerprint_groups": [
            sorted(paths) for paths in exact_groups.values()
        ],
        "fingerprint_scope": (
            "Exact section-table and imported-DLL-name grouping only; it does "
            "not establish a shared VM engine or Guest register layout."
        ),
        "claim_scope": (
            "Bounded root sampling of content-distinct PE32 files. Native i386 "
            "execution is tested, but unresolved calls, symbolic transfers, and "
            "truncations remain explicit. Not full-module or full-Guest proof."
        ),
    }
    if inventory_path is not None:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        pe32 = [row for row in inventory if row["machine"] == "0x14c"]
        distinct = {
            row.get("sha256")
            or hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            for row in pe32
        }
        result["inventory"] = {
            "path": str(inventory_path),
            "pe32_paths": len(pe32),
            "content_distinct_pe32_files": len(distinct),
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.measurements, args.inventory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
