"""Derive the live guest context by measurement, not by a constant.

Three findings meet here and they disagree with what this workspace recorded.

1. `E-CONTEXT-MAPPING.md` found sixteen guest registers saved at `entry_rbp+0x08`
   in a scrambled order, called it the guest register image, and `guest_effect.py`
   seeded it: **400 of 400 handlers changed nothing**. That was read as "the image
   is a backup".
2. `probe/gmem.py` found a second sixteen-register block at `entry_rbp+0x90` in
   **architectural** order, and `probe/gctx.py` seeded both blocks with different
   symbol families: `+0x08` is read-only after the prologue (0 writes into it),
   `+0x90` is written and read. So the answer was "a backup" about the wrong block.
3. The reference's `vmctx` command independently names `rbp+0x330` -- which is
   `entry_rbp+0x90` on this build -- "the guest register image".

This probe does not use any of those constants. It **derives** the live context:

* the machine register file is seeded with `m_<reg>` before the entry sequence,
  so the prologue marks the block it saves into, and the block is then overwritten
  with a second family `e_<reg>` -- one symbol per slot, named by the marker that
  landed there;
* the walk resumes, and every store whose value is **exactly** an `e_<reg>` node
  is a copy *out of* that block. Its destination address is therefore the slot
  that guest register occupies in whatever structure the VM copies it into;
* the destination addresses are collected per register, and the base is **solved
  for** rather than assumed: if the destinations are a context, they sit at
  `base + k*8` for distinct `k`, so the base is `dest - 8*k` for whichever `k`
  makes all sixteen agree.

Falsifiers, stated before the run:

1. the derived destinations must be **one address per register**; a register
   landing at several addresses means the destinations are spill slots and not a
   context;
2. the derived offsets must form a **permutation of 0..0x78 plus a flags slot** --
   sixteen distinct values in 8-byte steps. A context with holes is not a context;
3. the derivation is run on **two functions with different frames**. The live
   context is at a fixed offset from the stack base, so its *absolute* address
   must be identical across roots while its offset from the frame is not -- if
   the addresses differ, the structure is per-function and that is a finding.

Usage:
  python -m probe.guestmap <module> <root_va> [--handlers N] [--steps N]
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
MREGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
         "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
# Every slot the protector is expected to save: the sixteen general registers and
# the flags word. Nothing about their *order* is assumed.
SLOTS = ("rflags",) + MREGS + ("rsp",)


def run(path: str, root: int, handlers: int, steps: int) -> dict:
    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, STACK_BASE)
    L = Lifter(img, m, a, vm.va, vm.end)

    node_of: dict[str, int] = {}
    for name in MREGS:
        s = a.opaque(f"m_{name}", W64)
        node_of[f"m_{name}"] = s
        m.set_reg(name, s, 64)

    base: list[int] = []
    marked: dict[int, str] = {}       # entry-block slot offset -> guest register
    e_node: dict[int, str] = {}       # node id -> e_<reg>
    marked_done = [False]
    writes_to_entry: collections.Counter = collections.Counter()
    copies: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    copy_site: dict[str, int] = {}
    seed_at: list[int] = []

    orig_store = Machine.store
    orig_load = Machine.load
    entry_lo = [None]

    def store(self, addr, value, nbytes, site):
        c = self.a.as_const(addr)
        if c is not None and base and not marked_done[0]:
            marked.setdefault(c - base[0], None)  # first touch of any slot
        if c is not None and base and marked_done[0] and entry_lo[0] is not None \
                and entry_lo[0] <= c - base[0] < entry_lo[0] + 0x90:
            writes_to_entry[c - base[0]] += 1
        # A store whose value is exactly an `e_<reg>` node: a copy out of the block.
        if marked_done[0] and base and c is not None:
            ls = [n for n in self.a.leaves(value) if n in e_node]
            if len(ls) == 1 and self.a.op(value) == "opaque":
                copies[e_node[ls[0]]][c] += 1
                copy_site.setdefault(e_node[ls[0]], site)
        return orig_store(self, addr, value, nbytes, site)

    def load(self, addr, nbytes, site, image_u8):
        return orig_load(self, addr, nbytes, site, image_u8)

    Machine.store = store
    Machine.load = load

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
            if not marked_done[0]:
                # The prologue has saved every guest register into a block whose
                # base is `base[0]`; the marker stores say which register is where.
                came_from: dict[int, str] = {}
                for off in range(0, 0x90, 8):
                    v = orig_load(m, a.const(base[0] + off, W64), 8,
                                  SNAPSHOT_SITE, img.image_u8)
                    nm = self_name(m, a, v, node_of)
                    if nm:
                        came_from[off] = nm
                if len(came_from) < 12:
                    continue
                marked_done[0] = True
                seed_at.append(L.dispatches[-1][1])
                for off, reg in came_from.items():
                    s = a.opaque(f"e_{reg}", W64)
                    e_node[s] = f"e_{reg}"
                    marked[off] = reg
                    node_of[f"e_{reg}"] = s
                    orig_store(m, a.const(base[0] + off, W64), s, 8,
                               SNAPSHOT_SITE)
                entry_lo[0] = min(marked)
    finally:
        Machine.store = orig_store
        Machine.load = orig_load

    return {
        "root": root, "at": at, "stop": stop, "base": base[0] if base else None,
        "marked": marked, "copies": copies, "copy_site": copy_site,
        "seed_at": seed_at[0] if seed_at else None,
        "writes_to_entry": writes_to_entry, "transitions": seen,
    }


def self_name(m, a: Arena, v: int, node_of: dict[str, int]) -> str | None:
    """The marker name a value is, when it is exactly one of them."""
    n = a.node(v)
    if n[0] != "opaque":
        return None
    for name, node in node_of.items():
        if node == v:
            return name[2:]  # strip the `m_`/`e_` prefix
    return None


def report(res: dict) -> None:
    base, marked, copies = res["base"], res["marked"], res["copies"]
    print(f"root {res['root']:#x}   walked {res['at']} instructions   "
          f"stop {res['stop'].kind if res['stop'] else 'budget'}")
    print(f"entry frame rbp0: {base:#x}" if base else "entry frame NOT found")
    print(f"marker block re-seeded at handler {res['seed_at']:#x}"
          if res["seed_at"] else "NOT seeded")
    print(f"handler transitions: {res['transitions']}")
    print()
    print(f"marker block slots found: {len(marked)}")
    for off in sorted(marked):
        print(f"    rbp0+{off:#04x}  {marked[off]}")
    print()
    w = res["writes_to_entry"]
    print(f"writes back into the marker block after seeding: {sum(w.values())}")
    print()
    print("=== where each guest register is copied TO (derived, no constant used) ===")
    print(f"{'guest':<8} {'copies':>7} {'site':>14}  destinations")
    print("-" * 72)
    for reg in sorted(copies):
        dests = copies[reg]
        ds = ", ".join(f"{c:#x}(x{n})" for c, n in dests.most_common(4))
        print(f"{reg:<8} {sum(dests.values()):>7} {res['copy_site'][reg]:>#14x}  {ds}")
    print()

    # ---- derive the base, assuming no constant ---------------------------------
    #
    # The first attempt at this was wrong and the falsifier caught it. It assumed a
    # register is copied to *one* address, so the destination would name its slot.
    # All thirteen registers came back with several destinations, because the VM
    # does two different things with the same move instruction: it writes the
    # context slot **and** it spills to scratch cells (`entry_rbp-0x8` takes all
    # thirteen). Failing that falsifier did not invalidate the data, it invalidated
    # the model -- the one that was written down, not the one in the binary.
    #
    # So the base is *fitted*. For each architectural index `i` and each observed
    # destination `D`, the pair proposes `D - 8*i` as a base. A base is scored by
    # how many guest registers have a destination exactly at `base + 8*index(r)`.
    # A context explains many registers with one parameter; spill scratch explains
    # nothing beyond itself, and has to be reported as leftover.
    idx = {r: i for i, r in enumerate(
        ("rax", "rbx", "rcx", "rdx", "rsp", "rbp", "rsi", "rdi",
         "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"))}
    idx["rflags"] = 0x10

    # `copies` is keyed by the symbol name, which carries the `e_` prefix; the
    # first version of this forgot that, scored nothing, and reported "no copies
    # observed" on a run that had just printed forty of them.
    def arch(reg: str) -> int | None:
        return idx.get(reg[2:] if reg.startswith("e_") else reg)

    proposals: set[int] = set()
    for reg, dests in copies.items():
        i = arch(reg)
        if i is None:
            continue
        for d in dests:
            proposals.add(d - 8 * i)
    scored = []
    for B in proposals:
        hit = sorted(r for r, dests in copies.items()
                     if arch(r) is not None and (B + 8 * arch(r)) in dests)
        scored.append((len(hit), B, hit))
    scored.sort(reverse=True)
    if not scored:
        print("no copies observed: nothing to derive")
        return {}
    print("=== fitting a context base (one parameter, no constant assumed) ===")
    for n, B, hit in scored[:4]:
        print(f"  base {B:#x}   explains {n} of {len(copies)} registers"
              f"{'   <- best' if n == scored[0][0] else ''}")
        print(f"      slots hit: {', '.join(hit)}")
    best_n, best_B, best_hit = scored[0]
    print()
    explained = {best_B + 8 * arch(r) for r in best_hit}
    leftover = sorted({d for dests in copies.values() for d in dests}
                      - explained)
    print(f"FALSIFIER 1 -- a context must explain most registers with one "
          f"parameter: {best_n} of {len(copies)}  "
          f"({'HOLDS' if best_n >= 12 else 'FAILS'})")
    print(f"FALSIFIER 2 -- destinations the context does NOT explain (scratch): "
          f"{len(leftover)}")
    print(f"    {[hex(x) for x in leftover[:14]]}")
    print()
    print(f"derived context base: {best_B:#x}   "
          f"(rbp0 + {best_B - base:#x})   "
          f"(STACK_BASE - {STACK_BASE - best_B:#x})")
    print(f"{'guest':<8} {'slot':<14} {'expected offset':>16}")
    for r in sorted(best_hit, key=lambda x: arch(x)):
        print(f"    {r:<8} {best_B + 8 * arch(r):>#12x}   {8 * arch(r):>#14x}")
    return {"root": res["root"], "base": best_B, "rbp0": base,
            "hits": best_hit, "leftover": len(leftover)}


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    handlers, steps = 300, 120_000
    for x in sys.argv[1:]:
        if x.startswith("--handlers="):
            handlers = int(x.split("=", 1)[1], 0)
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
    path = args[0]
    roots = [int(x, 0) for x in args[1:]]
    got = []
    for r in roots:
        d = report(run(path, r, handlers, steps))
        if d:
            got.append(d)
        print("=" * 78)
    if len(got) > 1:
        # Falsifier 3. The context is a fixed offset from the stack base, so its
        # absolute address must be identical across functions whose frames differ.
        print("=== FALSIFIER 3: is the context per-function or per-module? ===")
        print(f"{'root':>14} {'rbp0':>18} {'derived base':>18} "
              f"{'base-STACK_BASE':>18}")
        for d in got:
            print(f"{d['root']:>#14x} {d['rbp0']:>#18x} {d['base']:>#18x} "
                  f"{d['base'] - STACK_BASE:>#18x}")
        bases = {d["base"] for d in got}
        print(f"distinct derived bases: {len(bases)} -- "
              f"{'per-module, one fixed address' if len(bases) == 1 else 'PER-FUNCTION'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
