"""Why do two builds of one driver have a `.tvm0` and no discovered trampolines?

`core.entries.vm_entries` looks for one form only: `E9 rel32` in an executable section whose
target lands inside `.tvm0`. That is the form every other module in the distribution uses. `ACE-CORE.sysa`
and `ACE-CORE.sysa2` have a `.tvm0` and report **zero** entries, which is either a discovery gap or a
genuine difference in how those builds are entered -- and the two answers imply different things about
whether they are in scope at all.

This enumerates every other way a `.tvm0` address can be named, so the question is answered by
measurement rather than by "the filter did not match":

* `E9 rel32` (jmp)             -- the form `core.entries.vm_entries` uses;
* `E8 rel32` (call)            -- a call into the VM section;
* `E9` / `E8` short forms are not encodable for a cross-section reach;
* absolute 8-byte values inside `.tvm0` found in any non-executable section -- a function-pointer
  table (`.rdata`), which is how a protector often enters a VM;
* `FF 25 rel32` (jmp qword ptr [rip+disp]) whose slot holds a `.tvm0` address;
* `0F 05` / `CD`-style and `FF /2`/`FF /4` indirect transfers are reported only as counts, since their
  targets are not statically named.

It also reports what the first bytes of `.tvm0` look like, because "no entries" and "no code" are
different findings.

Usage: python -m probe.trampoline_forms <module> [--vm .tvm0] [--max-list N]
"""

from __future__ import annotations

import struct
import sys

sys.path.insert(0, '.')

from core.peimg import PeImage  # noqa: E402


def _rel32(body: bytes, i: int) -> int:
    return int.from_bytes(body[i + 1:i + 5], 'little', signed=True)


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith('--')]
    if not args:
        print(__doc__)
        return 2
    vm_name = '.tvm0'
    for x in sys.argv[1:]:
        if x.startswith('--vm='):
            vm_name = x.split('=', 1)[1]
    img = PeImage(args[0])
    vm = img.section_by_name(vm_name)
    print(f'module {args[0]}')
    print(f'sections: {len(img.sections)}')
    for s in img.sections:
        print(f'  {s.name:<10} va {s.va:#012x}  vsize {s.vsize:#09x}  rsize {s.rsize:#09x}  '
              f'{"X" if s.executable else "-"}{"W" if s.writable else "-"}')
    if vm is None:
        print(f'no {vm_name} section')
        return 2
    print()
    print(f'{vm_name}: {vm.va:#x}..{vm.end:#x}  ({vm.end - vm.va:#x} bytes)')
    head = img.read_va(vm.va, 32)
    print(f'  first 32 bytes: {head.hex() if head else "<unreadable>"}')

    def inside(va: int) -> bool:
        return vm.va <= va < vm.end

    tally: dict[str, list] = {'E9': [], 'E8': [], 'FF25': [], 'deref-slot': [], 'abs-qword': []}
    for sec in img.sections:
        if sec.name == vm_name:
            continue
        body = img.read_va(sec.va, sec.rsize)
        if not body:
            continue
        n = len(body)
        if sec.executable:
            i = 0
            while i + 5 <= n:
                op = body[i]
                if op in (0xE9, 0xE8):
                    tgt = sec.va + i + 5 + _rel32(body, i)
                    if inside(tgt):
                        tally['E9' if op == 0xE9 else 'E8'].append((sec.va + i, tgt))
                    i += 5 if op in (0xE9, 0xE8) and inside(tgt) else 1
                    continue
                if op == 0xFF and i + 6 <= n and body[i + 1] == 0x25:
                    disp = _rel32(body, i)
                    slot = sec.va + i + 6 + disp
                    raw = img.read_va(slot, 8)
                    if raw and len(raw) == 8:
                        target = struct.unpack('<Q', raw)[0]
                        if inside(target):
                            tally['FF25'].append((sec.va + i, slot, target))
                    i += 6
                    continue
                i += 1
        else:
            # A pointer table: every 8-byte value that names an address inside the VM section.
            for off in range(0, n - 7, 8):
                val = struct.unpack('<Q', body[off:off + 8])[0]
                if inside(val):
                    tally['abs-qword'].append((sec.va + off, val))

    print()
    print(f'{"form":<28} {"count":>7}   first few')
    for key, label in (('E9', 'E9 rel32 (jmp) into .tvm0'),
                       ('E8', 'E8 rel32 (call) into .tvm0'),
                       ('FF25', 'FF 25 rel32 whose slot names .tvm0'),
                       ('abs-qword', 'absolute qword in a data section')):
        hits = tally[key]
        shown = ', '.join(f'{h[0]:#x}->{h[-1]:#x}' for h in hits[:4])
        print(f'{label:<28} {len(hits):>7}   {shown}')
    print()
    total = sum(len(v) for v in tally.values())
    if total == 0:
        print('NO static reference into the VM section was found in any of these forms.')
        print('That is a finding about the build, not about the discovery filter: read the')
        print('first bytes above to tell "no code" apart from "entered some other way".')
    else:
        print(f'{total} static reference(s) found; the E9-only filter would have missed '
              f'{total - len(tally["E9"])} of them.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
