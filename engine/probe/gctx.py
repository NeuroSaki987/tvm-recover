"""Which structure is the guest register image?

`E-CONTEXT-MAPPING.md` measured a block of sixteen guest registers saved at
`entry_rbp + 0x08`, in an order that is **not** architectural (`rdi` sits between
`rcx` and `rdx`). That was called "the guest register image", and on the strength
of it `probe/guest_effect.py` seeded those slots and watched: 400 of 400 handlers
changed nothing. The conclusion drawn was that the image is a backup.

The census in `probe/gmem.py` then found a **second** sixteen-register block at
`entry_rbp + 0x90`, in **architectural order** (rax, rbx, rcx, rdx, rsp, rbp, rsi,
rdi, r8..r15, rflags). Two blocks, two orders, and the earlier one had been
assumed to be the guest context because it was found first.

So the two are seeded with **different symbol families** and the handlers are
simply watched:

* `e_<reg>` into `entry_rbp + 0x08` -- the older block;
* `v_<reg>` into `entry_rbp + 0x90` -- the architectural block;

and every load that returns one of them is counted, **including loads through a
symbolic address**. That last part is a correction: `gmem` recorded an access only
when the address folded to a constant, so a handler that reads the context through
a register-held base was invisible. Since the reference's own IR prints
`load [rcx+0x18]`, precisely that shape is expected, and dropping it would have
made a live context look dead -- which is how the first block came to be believed
dead in the first place.

Falsifiers, stated before the run:

1. one family must dominate. If both are read at similar rates then neither is
   *the* context and the question is still open -- reported as such rather than
   resolved by preference;
2. a live context must be **written** by handlers, not only read;
3. the layout is a claim, so it is checked as one: each slot of the winning block
   must see **only its own** guest register, giving a permutation, not a smear.

Usage:
  python -m probe.gctx <module> <root_va> [--handlers N] [--steps N]
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, ".")

from core.expr import W64, Arena, render  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0
SNAPSHOT_SITE = -1

# The block measured in `E-CONTEXT-MAPPING.md`, at entry_rbp + 0x08.
ENTRY_BLOCK = {
    0x00: "rflags", 0x08: "rax", 0x10: "rbp", 0x18: "rbx", 0x20: "rcx",
    0x28: "rdi", 0x30: "rdx", 0x38: "rsi", 0x40: "rsp", 0x48: "r8",
    0x50: "r9", 0x58: "r10", 0x60: "r11", 0x68: "r12", 0x70: "r13",
    0x78: "r14", 0x80: "r15",
}
ENTRY_BASE_OFF = 0x00
# The block the census found, in architectural order.
LIVE_BASE_OFF = 0x90
ARCH_ORDER = ("rax", "rbx", "rcx", "rdx", "rsp", "rbp", "rsi", "rdi",
              "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
LIVE_BLOCK = {i * 8: r for i, r in enumerate(ARCH_ORDER)}
LIVE_BLOCK[0x80] = "rflags"
MREGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
         "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    handlers, steps = 300, 120_000
    for x in sys.argv[1:]:
        if x.startswith("--handlers="):
            handlers = int(x.split("=", 1)[1], 0)
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, STACK_BASE)
    L = Lifter(img, m, a, vm.va, vm.end)

    for name in MREGS:
        m.set_reg(name, a.opaque(f"m_{name}", W64), 64)
    guest: dict[int, str] = {}

    def names_of(v: int) -> list[str]:
        return sorted({guest[n] for n in a.leaves(v) if n in guest})

    base: list[int] = []
    seeded = [False]
    written: set[int] = {0x10}
    seed_at: list[int] = []

    # Per family: reads by (slot, symbol) and writes by (slot, symbol).
    rd: dict[str, collections.Counter] = {"e": collections.Counter(),
                                          "v": collections.Counter()}
    wr: dict[str, collections.Counter] = {"e": collections.Counter(),
                                          "v": collections.Counter()}
    sym_addr_reads: dict[str, int] = {"e": 0, "v": 0}
    sym_addr_examples: dict[str, list[str]] = {"e": [], "v": []}

    orig_load = Machine.load
    orig_store = Machine.store

    def fam(nm: str) -> str | None:
        return nm[0] if nm[:1] in ("e", "v") else None

    def load(self, addr, nbytes, site, image_u8):
        v = orig_load(self, addr, nbytes, site, image_u8)
        if site == SNAPSHOT_SITE:
            return v
        for nm in names_of(v):
            f = fam(nm)
            if f is None:
                continue
            c = self.a.as_const(addr)
            slot = (c - base[0]) if (c is not None and base) else None
            rd[f][(slot, nm)] += 1
            if c is None:
                sym_addr_reads[f] += 1
                if len(sym_addr_examples[f]) < 6:
                    sym_addr_examples[f].append(
                        f"{nm} <- {render(self.a, addr, limit=14)[:80]}")
        return v

    def store(self, addr, value, nbytes, site):
        c = self.a.as_const(addr)
        if c is not None and base and not seeded[0] and c - base[0] in ENTRY_BLOCK:
            written.add(c - base[0])
        for nm in names_of(value):
            f = fam(nm)
            if f is None:
                continue
            slot = (c - base[0]) if (c is not None and base) else None
            wr[f][(slot, nm)] += 1
        return orig_store(self, addr, value, nbytes, site)

    Machine.load = load
    Machine.store = store

    def make(prefix: str, name: str) -> int:
        s = a.opaque(f"{prefix}_{name}", W64)
        guest[s] = f"{prefix}_{name}"
        return s

    at, seen, ip, stop = 0, 0, root, None
    try:
        while at < steps and seen < handlers + 1:
            kind, payload = L.step(ip)
            at += 1
            if not base:
                b = a.as_const(m.reg("rbp", 64))
                if b is not None:
                    base.append(b)
            if kind != "next":
                stop = payload
                break
            ip = payload
            if len(L.dispatches) <= seen:
                continue
            seen = len(L.dispatches)
            if not seeded[0] and len(written) >= len(ENTRY_BLOCK):
                seeded[0] = True
                seed_at.append(L.dispatches[-1][1])
                for off, gname in ENTRY_BLOCK.items():
                    if gname == "rsp":
                        continue  # rsp is the machine stack; not markable
                    SNAPSHOT = off + ENTRY_BASE_OFF
                    m.mem_owned = getattr(m, "mem_owned", set())
                    m.mem_owned.add(base[0] + SNAPSHOT)
                    orig_store(m, a.const(base[0] + SNAPSHOT, W64),
                               make("e", gname), 8, SNAPSHOT_SITE)
                for off, gname in LIVE_BLOCK.items():
                    if gname == "rsp":
                        continue
                    m.mem_owned.add(base[0] + LIVE_BASE_OFF + off)
                    orig_store(m, a.const(base[0] + LIVE_BASE_OFF + off, W64),
                               make("v", gname), 8, SNAPSHOT_SITE)
    finally:
        Machine.load = orig_load
        Machine.store = orig_store

    print(f"root {root:#x}   walked {at} instructions   "
          f"stop {stop.kind if stop else 'budget'}")
    print(f"entry frame rbp0: {base[0]:#x}" if base else "entry frame NOT found")
    print(f"both blocks re-seeded at handler {seed_at[0]:#x}" if seed_at else
          "NOT seeded")
    print(f"handler transitions after seeding: {seen}")
    print()

    # ------------------------------------------------------------- falsifier 1
    print("=== which block do handlers actually touch? ===")
    print(f"{'family':<26} {'block':>18} {'reads':>7} {'writes':>7}")
    print("-" * 62)
    for f, label, blk in (("e", "entry save (+0x08)", ENTRY_BASE_OFF),
                          ("v", "live image (+0x90)", LIVE_BASE_OFF)):
        print(f"{'  ' + f + '_*  ' + label:<26} {f'rbp0+{blk:#x}':>18} "
              f"{sum(rd[f].values()):>7} {sum(wr[f].values()):>7}")
    print()
    print(f"reads through a *symbolic* address, which the earlier census dropped:")
    for f in ("e", "v"):
        print(f"  {f}_*: {sym_addr_reads[f]}")
        for ex in sym_addr_examples[f]:
            print(f"      {ex}")
    print()

    # ------------------------------------------------------------- falsifier 3
    print("=== is the winning block a layout, or a smear? ===")
    for f, blk in (("e", ENTRY_BLOCK), ("v", LIVE_BLOCK)):
        if not (rd[f] or wr[f]):
            continue
        print(f"--- family {f}_* ---")
        for slot in sorted(blk):
            syms = sorted({nm for (s, nm) in list(rd[f]) + list(wr[f])
                           if s == blk[slot]})
            if not syms:
                continue
            r = sum(n for (s, nm), n in rd[f].items() if s == blk[slot])
            w = sum(n for (s, nm), n in wr[f].items() if s == blk[slot])
            print(f"  rbp0+0x{blk[slot] + (0x00 if f == 'e' else LIVE_BASE_OFF):03x}"
                  f"  (slot claimed = {blk[slot]:#04x} {blk[slot] and ''})"
                  f"  reads={r:<4} writes={w:<4}  {', '.join(syms)}")
        print()
    print("per-slot detail with the slot offset as the key:")
    for f, off0 in (("e", ENTRY_BASE_OFF), ("v", LIVE_BASE_OFF)):
        for (slot, nm), n in sorted(rd[f].items(), key=lambda kv: (kv[0][0] or 0)):
            print(f"  read  {f}  rbp0+{slot:#x}" if slot is not None
                  else f"  read  {f}  <symbolic addr>", f"-> {nm}  x{n}")
        for (slot, nm), n in sorted(wr[f].items(), key=lambda kv: (kv[0][0] or 0)):
            print(f"  write {f}  rbp0+{slot:#x}" if slot is not None
                  else f"  write {f}  <symbolic addr>", f"-> {nm}  x{n}")
        _ = off0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
