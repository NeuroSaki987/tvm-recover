"""Which context slot holds which guest register? -- measured, and located first.

`FINDINGS.md` records the mapping as unresolved, "ambiguous between two candidate
orderings that the measurements do not separate". The VM's entry sequence reads
like a mapping (`mov [rbp+0x78], r14` and friends, each wrapped in a `not`/`xchg`
dance), but reading a mapping off obfuscated disassembly is exactly how this
workspace has produced confident wrong answers before.

So it is measured. Every general register is given a **distinct marker** before the
VM starts, and every store the VM then performs is watched. A store whose value is
a marker, or the bitwise complement of one, is the VM putting that guest register
somewhere -- and the offset from `rbp` at that moment is the slot.

**Where the image is, unlike which slot is which, is now measured rather than
assumed.** `core/ctxlayout` reads the displacement out of the `lea base,[rbp+D]`
that the handlers emit and then confirms a candidate window inside a live frame by
two signatures that only a real register file satisfies: the `rsp` slot holds the
machine stack base, and the other slots hold `.tvm0` pointers. A frame on this build
does not sit at a fixed offset from the image -- `rbp` drifts by `0xb0` during one
walk -- so the location is re-derived whenever the signature stops holding.

The two measurements answer different questions and this probe reports them
separately:

* **location** -- which `rbp + D` window is the register file (dynamic, per frame);
* **order** -- which of the sixteen 8-byte slots inside that window is `rax` (marker
  stores; unchanged by where the window was found).

Watching the stores rather than reading a window at the end matters, and the first
attempt got it wrong: `rbp` is not constant across the entry sequence (it moves
from `STACK_BASE-0xB28` to `STACK_BASE-0xBD0` within a few hundred instructions),
so a slot read at the final `rbp` is a different address from the slot the same
number named earlier, and the answer comes out plausible and wrong.

The falsifier is that it has to be a *mapping*: if this is real, each register's
marker lands in its own slot and no two registers share one. Scattered or shared
slots would mean the markers are being copied around arbitrarily rather than saved.

Usage: python -m probe.ctx_map <module> <root_va> [--steps N]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core import ctxlayout  # noqa: E402
from core.expr import W64, Arena  # noqa: E402
from core.guestctx import ARCH, FLAGS  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import GP64, Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0
BASE = 0x5A5A_0000_0000_0000
STEP = 0x0000_0001_0000_0000
MASK64 = (1 << 64) - 1


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    steps, prologue, dump_at = 2000, 0, 0
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--prologue="):
            prologue = int(x.split("=", 1)[1], 0)
        if x.startswith("--dump-at="):
            dump_at = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, STACK_BASE)

    # ---- step 0: where is the register file? (dynamic, not a constant) ----
    layout = ctxlayout.for_image(img, vm)
    print("dynamic context location")
    print("-" * 74)
    if layout is not None:
        print(f"  handler idiom : lea base,[rbp+{layout.handler_disp:#x}] then [base+off]")
        print(f"  slots the idiom addresses: {len(layout.slots)} "
              f"(flags at +{layout.flags_off:#x})")
        print(f"  displacement histogram   : "
              f"{ {hex(k): v for k, v in layout.histogram.items()} }")
        print(f"  order evidence           : {layout.order_evidence}")
    else:
        print("  no lea/[base+off] idiom found; the frame will be searched directly")
    # Confirm the window inside a live frame, which is what a run actually needs.
    probe_arena = Arena()
    probe_machine = Machine(probe_arena, STACK_BASE)
    ip_locate = root
    L0 = Lifter(img, probe_machine, probe_arena, vm.va, vm.end)
    located = None
    for i in range(steps):
        rbp0 = probe_arena.as_const(probe_machine.reg("rbp", 64))
        if rbp0 is not None and i % 8 == 0:
            cand = ctxlayout.locate_in_frame(
                probe_arena, probe_machine, rbp0, img, vm.va, vm.end, STACK_BASE)
            if ctxlayout.fits(cand):
                located = cand
                break
        kind, payload = L0.step(ip_locate)
        if kind != "next":
            break
        ip_locate = payload
    if located is not None:
        print(f"  located in frame: rbp{located.off_from_rbp:+#x} -> {located.base:#x}")
        print(f"                    vm_slots={located.vm_slots} "
              f"rsp_signature={located.rsp_ok}")
    else:
        print("  no window in the frame satisfied the signatures")
    print()

    marks: dict[int, str] = {}
    for i, name in enumerate(GP64):
        if name in ("rsp", "rbp"):
            continue
        v = BASE + i * STEP
        m.set_reg(name, a.const(v, W64), 64)
        marks[v] = name
        marks[(~v) & MASK64] = f"~{name}"

    L = Lifter(img, m, a, vm.va, vm.end)
    hits: list[tuple[int, int, int, str, int]] = []
    frame: list[tuple[int, int, int, int]] = []
    orig_store = Machine.store
    step_no = [0]

    def store(self, addr, value, nbytes, site):
        c = self.a.as_const(addr)
        cv = self.a.as_const(value)
        rbp = self.a.as_const(self.reg("rbp", 64))
        if c is not None:
            who = marks.get(cv & MASK64) if cv is not None else None
            if who:
                rsp = self.a.as_const(self.reg("rsp", 64))
                hits.append((site, c, rbp if rbp is not None else -1, who,
                             rsp if rsp is not None else -1))
            # Every store into the VM frame during the prologue, marked or not:
            # the two slots no marker lands in are the ones the marker set cannot
            # cover (rsp and rbp), and this says what they actually hold.
            if (prologue and rbp is not None and step_no[0] <= prologue
                    and rbp <= c < rbp + 0xB28 and cv is not None):
                frame.append((step_no[0], site, c - rbp, cv))
        return orig_store(self, addr, value, nbytes, site)

    Machine.store = store
    ip, stop, at = root, None, 0
    base = None
    try:
        for i in range(steps):
            step_no[0] = i + 1
            kind, payload = L.step(ip)
            at = i + 1
            if base is None:
                # The frame base is `mov rbp, rsp` after the `lea rsp,[rsp-N]`,
                # so it is the first concrete rbp and it is worth capturing
                # rather than recomputing: it moves later.
                b = a.as_const(m.reg("rbp", 64))
                if b is not None:
                    base = b
            if dump_at and i + 1 >= dump_at:
                break
            if kind != "next":
                stop = payload
                break
            ip = payload
    finally:
        Machine.store = orig_store

    print(f"root {root:#x}   walked {at} instructions   "
          f"stop {stop.kind if stop else 'budget'}"
          + (f" at {stop.site:#x}" if stop else ""))
    print(f"stores of a marked guest register: {len(hits)}")
    print()
    # The frame as it stands at a chosen instruction, read at the base captured
    # from `mov rbp, rsp`. This is the measurement that names the slots no marker
    # can reach: `rsp` cannot be marked, and `rbp` is overwritten by the prologue
    # before anything can read it back.
    if base is not None:
        cur = a.as_const(m.reg("rbp", 64))
        for label, b in (("entry frame (first concrete rbp)", base),
                         ("frame at the end of the walk (current rbp)", cur)):
            if b is None:
                continue
            moved = "" if b == base else f"   MOVED by {b - base:+#x}"
            print(f"{label}: {b:#x}{moved}")
            print(f"{'slot':>8} {'value':>20}  marker")
            print("-" * 42)
            for off in range(0, 0x90, 8):
                ref = m.load(a.const(b + off, W64), 8, 0, img.image_u8)
                v = a.as_const(ref)
                if v is None:
                    print(f"{off:#8x} {'(symbolic)':>20}  -")
                    continue
                who = marks.get(v & MASK64, "")
                note = who
                if not note and v == STACK_BASE:
                    note = "<- the machine stack base, i.e. the guest rsp at entry"
                sec = img.section_for_va(v)
                if not note and sec is not None and sec.name == ".tvm0":
                    note = "<- a .tvm0 pointer"
                print(f"{off:#8x} {v:>#20x}  {note}")
            print()

    print(f"{'site':>14} {'address':>18} {'rel rbp':>10} {'rel rsp':>10}  register")
    print("-" * 74)
    for site, addr, rbp, who, rsp in hits[:60]:
        rel_rbp = f"{addr - rbp:+#x}" if rbp >= 0 else "?"
        rel_rsp = f"{addr - rsp:+#x}" if rsp >= 0 else "?"
        print(f"{site:>#14x} {addr:>#18x} {rel_rbp:>10} {rel_rsp:>10}  {who}")

    print()
    # Every store into the frame during the prologue, so the slots no marker can
    # reach -- rsp, which the engine cannot mark, and anything set from a
    # non-register source -- are named rather than assumed.
    if prologue and frame:
        print(f"all stores into the VM frame during the first {prologue} "
              f"instructions: {len(frame)}")
        print(f"{'step':>6} {'site':>14} {'slot':>10} {'value':>20}")
        print("-" * 54)
        for step, site, off, val in frame[:48]:
            print(f"{step:>6} {site:>#14x} rbp{off:+#010x} {val:>#20x}")
        print()

    # The mapping is the claim, so state it as one and check it is a function.
    by_reg: dict[str, set[int]] = collections.defaultdict(set)
    by_slot: dict[int, set[str]] = collections.defaultdict(set)
    for _site, addr, rbp, who, _rsp in hits:
        if rbp < 0:
            continue
        by_reg[who.lstrip("~")].add(addr - rbp)
        by_slot[addr - rbp].add(who.lstrip("~"))
    print(f"guest registers observed being saved: {len(by_reg)}")
    for name in sorted(by_reg):
        offs = ", ".join(f"rbp{o:+#x}" for o in sorted(by_reg[name]))
        print(f"  {name:<4} -> {offs}")
    shared = {o: v for o, v in by_slot.items() if len(v) > 1}
    print()
    if shared:
        print(f"!! {len(shared)} slot(s) receive more than one guest register, so this "
              f"is not a mapping:")
        for o, v in sorted(shared.items())[:10]:
            print(f"   rbp{o:+#x}: {sorted(v)}")
    else:
        print("every slot receives exactly one guest register: consistent with a mapping")

    # ---- does the file sit where the handlers say it does, in the order claimed? ----
    print()
    print("order check against the dynamically located window")
    print("-" * 74)
    if located is None:
        print("  skipped: no window was located in the frame, so there is nothing to")
        print("  compare against. The rbp-relative offsets above are still measurements,")
        print("  they just do not name slots of a located file.")
        return 0
    by_img_slot: dict[int, set[str]] = collections.defaultdict(set)
    for _site, addr, _rbp, who, _rsp in hits:
        d = addr - located.base
        if 0 <= d < ctxlayout.SPAN and d % 8 == 0:
            by_img_slot[d].add(who.lstrip("~"))
    covered = sorted(by_img_slot)
    print(f"  marker stores landing inside the located window: "
          f"{sum(len(v) for v in by_img_slot.values())} over {len(covered)} slot(s)")
    if not covered:
        print("  none. The markers landed somewhere else, so this window is not the")
        print("  register file the entry sequence fills -- do not read names out of it.")
        return 0
    agree = 0
    rows = []
    for off in range(0, ctxlayout.SPAN, 8):
        want = ARCH[off // 8] if off < ctxlayout.FLAGS_OFF else FLAGS
        got = sorted(by_img_slot.get(off, ()))
        if got == [want]:
            agree += 1
            mark = "="
        elif not got:
            mark = "."
        else:
            mark = "!"
        rows.append((off, want, got, mark))
    for off, want, got, mark in rows:
        if mark == ".":
            continue
        print(f"  {off:#06x}  want {want:<5} got {','.join(got) or '-':<12} {mark}")
    print(f"  {agree}/{len(rows)} slots match the architectural order")
    if agree >= 12:
        print("  evidence: measured:architectural  -- this window may be read as ARCH")
    else:
        print("  evidence: measured:non-architectural -- this window is NOT in ARCH order;")
        print("  the located base is a different structure from the one filled at entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
