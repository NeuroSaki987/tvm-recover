"""What does a VM handler *do to the guest*?

This is the step the engine has been built towards and has never taken. Up to now
the deliverable has been the VM's own handler transitions -- "control went from
handler A to handler B". What a devirtualizer owes an analyst is the guest program:
"this handler adds the guest's `rbx` to its `rax`".

The method is the engine's own machinery pointed at the right question. The
seventeen guest slots in the entry context image (measured in
`evidence/E-CONTEXT-MAPPING.md`) are overwritten with fresh symbolic guest
registers, and after each handler transition the image is read back. A slot whose
value changed now holds an *expression over the old guest registers*, and that
expression is the guest operation.

It works only if the VM's own scaffolding cancels, and that is a risk rather than
a hope: handlers are wrapped in `pushfq`/`popfq`, they load the vip, they walk a
handler table, and they shuffle registers through a `not`/`xchg` dance. If any of
that leaks into a recovered expression, the method has failed and the honest answer
is to say so. Hence explicit falsifiers:

* an **untouched slot must still be the identical symbol node**, not merely an
  equal expression -- otherwise the handler rewrote it with a copy of itself, which
  is not an operation;
* **no recovered value may mention the vip, a `.tvm0` address or a frame pointer**;
* and a handler that changes nothing is reported as changing nothing.

`rsp` is deliberately **not** seeded. The VM emulates the guest stack inside its own
frame, so a symbolic `rsp` would put every guest push at an unknown address and
recover nothing at all; leaving it concrete is what makes the rest work.

Usage:
  python -m probe.guest_effect <module> <root_va> [--handlers N] [--show N]
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

GUEST_SLOTS = {
    0x00: "rflags", 0x08: "rax", 0x10: "rbp", 0x18: "rbx", 0x20: "rcx",
    0x28: "rdi", 0x30: "rdx", 0x38: "rsi", 0x48: "r8", 0x50: "r9",
    0x58: "r10", 0x60: "r11", 0x68: "r12", 0x70: "r13", 0x78: "r14",
    0x80: "r15",
}
RSP_SLOT = 0x40
GUEST_RSP_NAME = "rsp"


def reads_image(a: Arena, m: Machine, img, base: int) -> dict[int, int]:
    return {off: m.load(a.const(base + off, W64), 8, 0, img.image_u8)
            for off in GUEST_SLOTS}


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    handlers, show, steps = 300, 12, 400_000
    seed_image = False
    for x in sys.argv[1:]:
        if x.startswith("--handlers="):
            handlers = int(x.split("=", 1)[1], 0)
        if x.startswith("--show="):
            show = int(x.split("=", 1)[1], 0)
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x == "--seed-image":
            seed_image = True
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, STACK_BASE)
    L = Lifter(img, m, a, vm.va, vm.end)

    # Seed the *machine* registers, before the entry sequence runs.
    #
    # Reading the context image back was tried first and came out flat: 400 of 400
    # handlers changed nothing, with the seeding verified correct. That is a
    # finding rather than a bug -- the image is a **backup**, written by the
    # prologue and not the live state the handlers work on. The guest's registers
    # live in the machine's registers between dispatches, shuffled by the
    # protector, and the image is only the entry snapshot.
    #
    # Seeding the registers instead makes the prologue store symbols into the image
    # on its own, so both views are symbolic and the question becomes the real one:
    # which machine register holds which guest register after a handler?
    MREGS = [r for r in
             ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
              "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")]
    mseed: dict[str, int] = {}
    for name in MREGS:
        s = a.opaque(f"g_{name}", W64)
        mseed[name] = s
        m.set_reg(name, s, 64)

    base: list[int] = []
    seeded = [False]
    # Slot +0x10 is guest rbp, and it is saved by `mov qword ptr [rsp + 0x10], rbp`
    # -- the instruction *before* `mov rbp, rsp` establishes the frame base. It
    # therefore predates the base and can never be attributed to an offset by
    # watching stores, so a completeness test that waits for it never fires. The
    # other fifteen are all written after the base exists.
    written: set[int] = {0x10}
    seed_at: list[int] = []
    orig_store = Machine.store

    def note_store(addr) -> None:
        pass  # superseded by the `store` wrapper below; kept for the base capture

    def seed_now(site: int) -> None:
        """Replace the guest image with fresh symbols.

        Called once every slot of the image has actually been written **and** the
        walk is at a dispatch boundary. Both conditions were got wrong in turn,
        and each mistake produced a plausible wrong answer:

        * keying on a single store (`+0x80`) seeded the image in the middle of the
          prologue, and the stores at `0x140164dc4` and `0x140165f27` came after it
          and overwrote the symbols with the machine's initial registers;
        * keying on the first `L.dispatches` entry seeded at the entry sequence's
          own `jmp rcx`, which is *also* before the prologue's tail, so the same
          eleven registers were overwritten again and appeared as guest effects.
        """
        if seeded[0] or not base:
            return
        seeded[0] = True
        seed_at.append(site)
        for off, name in GUEST_SLOTS.items():
            orig_store(m, a.const(base[0] + off, W64),
                       a.opaque(f"g_{name}", W64), 8, site)

    def store(self, addr, value, nbytes, site):
        if base and not seeded[0]:
            c = self.a.as_const(addr)
            if c is not None:
                off = c - base[0]
                if off in GUEST_SLOTS:
                    written.add(off)
        return orig_store(self, addr, value, nbytes, site)

    Machine.store = store
    results: list[tuple[int, dict[int, int], dict[int, int]]] = []
    ip, stop, at, seen = root, None, 0, 0
    symbols: dict[int, int] = {}
    try:
        while at < steps and len(results) < handlers:
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
            if len(L.dispatches) > seen:
                seen = len(L.dispatches)
                if seed_image and not seeded[0]:
                    if len(written) >= len(GUEST_SLOTS):
                        # Every slot has been written, so the image is complete
                        # and this is the first dispatch after that.
                        seed_now(L.dispatches[-1][0])
                        symbols = reads_image(a, m, img, base[0])
                    continue
                if not seed_image:
                    # Register-seeded mode: a handler boundary is just a
                    # transition, and what matters is read afterwards.
                    results.append((L.dispatches[-1][0], {}, {}))
                    continue
                after = reads_image(a, m, img, base[0])
                results.append((L.dispatches[-1][0], dict(symbols), after))
                symbols = after
    finally:
        Machine.store = orig_store

    print(f"root {root:#x}   walked {at} instructions   "
          f"stop {stop.kind if stop else 'budget'}")
    print(f"entry frame  {base[0]:#x}" if base else "entry frame NOT found")
    print(f"handler transitions observed: {len(results)}")
    print()

    if results and not seeded[0]:
        # Register-seeded mode: read what the machine's registers hold now.
        print("machine registers after each handler, and which guest symbol each "
              "mentions:")
        named: collections.Counter = collections.Counter()
        stable: collections.Counter = collections.Counter()
        for site, _b, _af in results[:show]:
            holds = []
            for name in MREGS:
                v = m.reg(name, 64)
                txt = render(a, v, limit=8)
                if "g_" in txt:
                    holds.append(f"{name}={txt[:40]}")
                    named[name] += 1
            print(f"--- at {site:#x} ---")
            print("    " + ("; ".join(holds) if holds else "(no register mentions "
                                                          "a guest symbol)"))
        for site, _b, _af in results:
            for name in MREGS:
                if "g_" in render(a, m.reg(name, 64), limit=8):
                    stable[name] += 1
        print()
        print(f"guest symbols still visible, by register and handler count: "
              f"{dict(stable.most_common())}")
        return 0

    if not results:
        print(f"the walk ended before any handler transition "
              f"(seeded={seeded[0]}, slots written={len(written)})")
        return 2

    ops: collections.Counter = collections.Counter()
    leaks = 0
    touched = collections.Counter()
    unchanged = 0
    for site, before, after in results:
        changed = {off: after[off] for off in GUEST_SLOTS
                   if off in before and after[off] != before[off]}
        if not changed:
            unchanged += 1
        for off in changed:
            touched[GUEST_SLOTS[off]] += 1
            txt = render(a, changed[off], limit=30)
            if any(t in txt for t in (":", "0x1401", "0x1400")):
                leaks += 1
        ops["changed=%d" % len(changed)] += 1

    print(f"handlers that changed nothing: {unchanged} of {len(results)}")
    print(f"how many guest slots each handler changed: "
          f"{dict(sorted(ops.items()))}")
    print(f"guest registers touched, by handler count: {dict(touched.most_common())}")
    print(f"recovered values mentioning VM bookkeeping (a `:` means a loaded node): "
          f"{leaks}")
    print()
    for site, before, after in results[:show]:
        changed = [(off, after[off]) for off in GUEST_SLOTS
                   if off in before and after[off] != before[off]]
        print(f"--- handler transition at {site:#x} ---")
        if not changed:
            print("    (no guest register changed)")
        for off, v in changed:
            print(f"    {GUEST_SLOTS[off]:<7} = {render(a, v, limit=16)[:118]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
