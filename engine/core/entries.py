"""VM entry candidates, with anchored instruction-boundary evidence.

Trampoline discovery only. Recovery does not need it -- an analyst supplies a root -- but
anything that wants to work over the *whole module* does, and the reference's own entry census is
the natural domain to compare over.

This lived in `probe/dispatch_compare.py` until round 41, where `core/devirt.py` needed it and
could not import it: `core` is a top-level package, so `..probe` is beyond the top level, and a
capability the engine needs must not live in a probe. The probe now delegates here.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class EntryCandidate:
    site: int
    target: int
    status: str
    reason: str
    provenance: tuple[str, ...] = ()
    covering_instruction: tuple[int, int, str] | None = None


@dataclass(frozen=True)
class EntryAnalysis:
    candidates: tuple[EntryCandidate, ...]
    instructions: int
    anchors: int
    complete: bool


def raw_candidates(img, vm_section: str = ".tvm0") -> list[tuple[int, int]]:
    """Legacy byte census; it deliberately includes E9 bytes inside instructions."""
    vm = img.section_by_name(vm_section)
    if vm is None:
        return []
    out: list[tuple[int, int]] = []
    for sec in img.sections:
        if sec.name == vm_section or not sec.executable:
            continue
        body = img.read_va(sec.va, sec.rsize)
        if not body:
            continue
        i, n = 0, len(body)
        while i + 5 <= n:
            if body[i] != 0xE9:
                i += 1
                continue
            rel = int.from_bytes(body[i + 1 : i + 5], "little", signed=True)
            tgt = sec.va + i + 5 + rel
            if vm.va <= tgt < vm.end:
                out.append((sec.va + i, tgt))
                i += 5
            else:
                i += 1
    return out


def _anchors(img) -> dict[int, set[str]]:
    """PE entry, unwind function starts and non-forwarded export addresses."""
    out: dict[int, set[str]] = {}

    def add(va, reason):
        sec = img.section_for_va(va)
        if sec is not None and sec.executable:
            out.setdefault(va, set()).add(reason)

    add(getattr(img, 'entry_point', 0), 'pe-entry')
    for begin, end in getattr(img, 'pdata', ()):
        if begin < end:
            add(begin, f'unwind-function:{begin:#x}')
    directories = getattr(img, 'directories', ())
    if directories:
        rva, size = directories[0]
        if rva and size >= 40:
            base = img.image_base
            header = img.read_va(base + rva, 40)
            if header is not None and len(header) == 40:
                count = struct.unpack_from('<I', header, 20)[0]
                table_rva = struct.unpack_from('<I', header, 28)[0]
                sec = img.section_for_va(base + table_rva)
                if sec is not None:
                    # Bound malformed table sizes by actual file-backed bytes.
                    available = max(0, sec.va + sec.rsize - base - table_rva)
                    count = min(count, available // 4)
                    table = img.read_va(base + table_rva, count * 4) if count else b''
                    if table is not None:
                        for ordinal, (target_rva,) in enumerate(struct.iter_unpack('<I', table[:len(table) // 4 * 4])):
                            if target_rva and not rva <= target_rva < rva + size:
                                add(base + target_rva, f'export-index:{ordinal}')
    return out


def analyze_entries(img, vm_section: str = '.tvm0',
                    instruction_limit: int = 600_000) -> EntryAnalysis:
    """Classify candidates using forward decoding from PE-owned code anchors.

    Direct branches and calls establish additional boundaries, including intentional
    overlapping code. An interior byte is rejected only after the anchored graph
    has been explored completely and no such incoming instruction establishes it.
    Unproven candidates remain explicit; lack of evidence is not a rejection.
    """
    from capstone import (CS_ARCH_X86, CS_MODE_32, CS_MODE_64, CS_OP_IMM,
                          CS_GRP_CALL, CS_GRP_IRET, CS_GRP_JUMP, CS_GRP_RET, Cs)

    cache = getattr(img, '_entry_analysis_cache', None)
    if cache is None:
        cache = img._entry_analysis_cache = {}
    key = (vm_section, instruction_limit)
    if key in cache:
        return cache[key]
    candidates = raw_candidates(img, vm_section)
    if not candidates:
        result = EntryAnalysis((), 0, 0, True)
        cache[key] = result
        return result
    sites = {site for site, _ in candidates}
    provenance = _anchors(img)
    anchor_count = len(provenance)
    todo = list(provenance)
    visited = set()
    decoded = set()
    interiors = {}
    md = Cs(CS_ARCH_X86, CS_MODE_64 if getattr(img, 'is64', True) else CS_MODE_32)
    md.detail = True

    def enqueue(target, reason):
        provenance.setdefault(target, set()).add(reason)
        if target not in visited:
            todo.append(target)

    while todo and len(visited) < instruction_limit:
        ip = todo.pop()
        if ip in visited:
            continue
        visited.add(ip)
        sec = img.section_for_va(ip)
        if sec is None or not sec.executable:
            continue
        count = min(15, sec.va + sec.rsize - ip)
        body = img.read_va(ip, count) if count > 0 else None
        if not body:
            continue
        insn = next(md.disasm(body, ip, 1), None)
        if insn is None:
            continue
        decoded.add(ip)
        for byte in range(ip + 1, ip + insn.size):
            if byte in sites:
                interiors[byte] = (ip, insn.size, f'{insn.mnemonic} {insn.op_str}'.strip())
        if (insn.group(CS_GRP_JUMP) or insn.group(CS_GRP_CALL)):
            if insn.operands and insn.operands[0].type == CS_OP_IMM:
                target = insn.operands[0].imm
                enqueue(target, f'direct-control-flow:{ip:#x}')
        terminal = (insn.mnemonic in ('jmp', 'ljmp', 'int3', 'int1', 'int',
                                     'ud0', 'ud1', 'ud2', 'hlt')
                    or insn.group(CS_GRP_RET) or insn.group(CS_GRP_IRET))
        if not terminal:
            enqueue(ip + insn.size, f'fallthrough:{ip:#x}')

    complete = not todo
    records = []
    for site, target in candidates:
        cover = interiors.get(site)
        if site in decoded:
            status, reason = 'validated', 'instruction boundary reached from a PE code anchor'
        elif cover is not None and complete:
            status, reason = 'rejected', 'E9 byte is inside an anchored instruction; no proven incoming edge'
        else:
            status = 'unproven'
            reason = ('anchored decoding budget exhausted' if not complete
                      else 'no anchored instruction establishes or excludes this boundary')
        records.append(EntryCandidate(site, target, status, reason,
                                      tuple(sorted(provenance.get(site, ()))), cover))
    result = EntryAnalysis(tuple(records), len(decoded), anchor_count, complete)
    cache[key] = result
    return result


def vm_entries(img, vm_section: str = '.tvm0', *, legacy: bool = False,
               strict: bool = False) -> list[tuple[int, int]]:
    """Entries excluding proved instruction interiors, with an optional strict census.

    Default discovery retains unproven boundaries to avoid silently dropping valid
    entries absent from PE metadata. `strict=True` accepts only positive boundaries;
    `legacy=True` reproduces the original raw byte population for comparisons.
    `analyze_entries` exposes the status and evidence behind every decision.
    """
    if legacy:
        return raw_candidates(img, vm_section)
    records = analyze_entries(img, vm_section).candidates
    return [(r.site, r.target) for r in records
            if r.status == 'validated' or (not strict and r.status == 'unproven')]


def vm_roots(img, vm_section: str = ".tvm0", *, legacy: bool = False,
             strict: bool = False) -> list[int]:
    """The entry addresses, in address order."""
    return sorted({t for _s, t in vm_entries(img, vm_section, legacy=legacy, strict=strict)})
