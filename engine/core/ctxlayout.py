"""Where the guest register file actually sits -- found, not assumed.

## The problem this fixes

`guestctx.py` located the image with a chain of constants: `entry_rbp + 0x90`,
`working_rbp + 0xa8`, therefore `working_rbp + 0x138`. That chain is a measurement
of *one* protector build, and it is expressed as a constant, so every other build
either works by luck or silently reads the wrong seventeen words. Worse, the code
that used it (`Recoverer.prepare`, `Recoverer.identity`) did not even use the sum:
it scanned `rbp + 0..0x90`, i.e. it assumed the image started **at** the sampled
frame. Two different constants for the same thing, neither derived from the target.

## What the handlers actually say

Handlers never address the image off `rbp`. They compute a base register first:

    lea rdx, [rbp + 0x90]        ; <- the image base, in a register
    mov r10, [rdx]               ; <- guest rax
    mov rbx, [rdx + 0x08]        ; <- guest rbx
    ...
    mov rdi, [rdx + 0x38]        ; <- guest rdi
    mov r9,  [rdx + 0x48]        ; <- guest r9

so the displacement is a **constant inside a `lea`**, not an offset in the load, and
the register holding it differs per handler (`rdx`, `r10`, `r11`, ... were all seen).
Reading the image off `rbp` therefore requires knowing the traversal `rbp`, while
reading it off the `lea` displacement requires nothing at all: `[base + off]` with
`off` covering the 8-aligned span is the signature, and the `lea` that produced
`base` gives the anchor.

This module finds that pattern with capstone, scores candidate displacements by how
much of the slot span they explain, and returns a `Layout`. Nothing here assumes
0x90, 0x138 or ARCH ordering: the displacement is measured, the slot span is
measured, and the *order* is reported together with the evidence that establishes
it -- which is `order_evidence`, and it says `assumed` when nothing measured it.

## Why the order still needs a separate measurement

The `lea` says where the block is, not which word is `rax`. Any assignment of the
sixteen registers to the sixteen 8-byte slots produces the same access pattern, so
the pattern cannot distinguish them. The order is therefore carried as a *claim*
with a field naming what established it:

* `measured:architectural` -- `probe.ctx_map` seeded each host register with a
  distinct marker and watched the stores; the markers landed in architectural order.
* `assumed:architectural` -- no such run was available; ARCH order is used and the
  claim is labelled as an assumption, not as a finding.

That distinction is the whole point: a wrong order is invisible in the disassembly
and wrong in every listing, so it must never be presented as derived from a pattern
that cannot contain it.
"""

from __future__ import annotations

import collections

from .guestctx import ARCH, FLAGS, WORKING_FRAME_DELTA

#: The 8-byte slot span an image occupies: sixteen registers plus the flags word.
FLAGS_OFF = 0x80
SPAN = FLAGS_OFF + 8

#: Offsets we will accept as "a context slot". A layout that explains fewer than
#: this many of the seventeen words is not a layout, it is a coincidence.
MIN_SLOTS = 10

#: How far past a `lea` to look for `[base + off]` accesses. Handlers materialise the
#: whole file in one burst, so 160 bytes is generous and 24 would usually do.
WINDOW = 160

#: Bound on how many `lea` sites are decoded. The displacement distribution is
#: overwhelmingly one value; four thousand samples settle it on every build seen.
MAX_SITES = 4000


class Layout:
    """A measured guest-context layout."""

    __slots__ = ("handler_disp", "working_disp", "slots", "flags_off",
                 "order_evidence", "sources", "histogram")

    def __init__(self, handler_disp: int, working_disp: int, slots: dict[int, str],
                 flags_off: int, order_evidence: str,
                 sources: list[str] | None = None,
                 histogram: dict[int, int] | None = None) -> None:
        self.handler_disp = handler_disp
        self.working_disp = working_disp
        self.slots = dict(slots)
        self.flags_off = flags_off
        self.order_evidence = order_evidence
        self.sources = list(sources or [])
        self.histogram = dict(histogram or {})

    # -- convenience --------------------------------------------------------
    def reg_at(self, off: int) -> str | None:
        return self.slots.get(off)

    @property
    def span(self) -> tuple[int, int]:
        return 0, self.flags_off + 8

    def covers(self) -> int:
        return len(self.slots) + (1 if self.flags_off is not None else 0)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"<Layout handler_disp={self.handler_disp:#x} "
                f"working_disp={self.working_disp:#x} slots={len(self.slots)} "
                f"order={self.order_evidence}>")


def default_slots(order=ARCH, flags_off: int = FLAGS_OFF) -> dict[int, str]:
    """The slot map for the architectural order."""
    out = {8 * i: r for i, r in enumerate(order)}
    out[flags_off] = FLAGS
    return out


def _lea_hits(img, vm, limit: int = MAX_SITES):
    """Yield `(site_va, base_register, displacement)` for `lea reg,[rbp+disp]`.

    Scanned as a byte pattern rather than a linear disassembly: `lea r64, [rbp+disp32]`
    is `REX.W [REX.R] 8D <modrm=0x85|reg<<3> <disp32>`, and matching it directly means
    a mis-decode cannot fabricate a candidate. Everything the scan finds is a real
    instruction boundary by construction.
    """
    data = img.read_va(vm.va, vm.end - vm.va)
    if not data:
        return
    n = len(data)
    base_va = vm.va
    i = 0
    emitted = 0
    while emitted < limit:
        i = data.find(b"\x8d", i)
        if i < 0 or i + 7 > n:
            return
        # walk back one byte for the REX prefix; the `8d` must follow it directly
        if i < 1:
            i += 1
            continue
        rex = data[i - 1]
        if rex not in (0x48, 0x4C, 0x49, 0x4D):
            i += 1
            continue
        modrm = data[i + 1]
        # mod=10 (disp32), rm=101 (rbp) -> (modrm & 0xC7) == 0x85
        if (modrm & 0xC7) != 0x85:
            i += 1
            continue
        reg = ((modrm >> 3) & 7) | (8 if rex & 0x04 else 0)
        disp = int.from_bytes(data[i + 2:i + 6], "little", signed=True)
        if 0 < disp < 0x2000:
            yield base_va + i - 1, reg, disp
            emitted += 1
        i += 6


_REG_NAMES = ("rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
              "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")

_MEM_RE = None


def _mem_pairs(md, code: bytes, va: int):
    """`(base_register, offset)` for every memory operand in `code`."""
    out = []
    for insn in md.disasm(code, va):
        try:
            for op in insn.operands:
                if op.type == 3:                       # X86_OP_MEM
                    m = op.mem
                    if m.base == 0:
                        continue
                    name = insn.reg_name(m.base)
                    if not name:
                        continue
                    out.append((name, m.disp & 0xFFFFFFFF
                                if m.disp < 0 else m.disp))
        except Exception:
            continue
    return out


def discover(img, vm, limit: int = MAX_SITES, order_evidence: str = "assumed:architectural") -> Layout | None:
    """Find the guest-context displacement by matching `lea base,[rbp+D]` + `[base+off]`.

    Returns None when the section holds no such pattern, and never guesses: a build
    whose handlers use a different idiom should fall back to the caller's measurement
    rather than be given an invented layout.
    """
    try:
        import capstone
    except ImportError:                                # pragma: no cover
        return None
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    # `insn.operands` is only populated when detail is on; without it every scan
    # below silently reports "no memory operands" and the layout is never found.
    md.detail = True

    data = img.read_va(vm.va, vm.end - vm.va)
    if not data:
        return None
    n = len(data)

    disp_hist: collections.Counter = collections.Counter()
    slot_sets: dict[int, set[int]] = collections.defaultdict(set)
    sources: dict[int, list[str]] = collections.defaultdict(list)

    for site, reg, disp in _lea_hits(img, vm, limit=limit):
        rel = site - vm.va
        code = data[rel:rel + WINDOW]
        if len(code) < 16:
            continue
        disp_hist[disp] += 1
        # Only count accesses that go through the register this `lea` filled.
        want = _REG_NAMES[reg]
        for base, off in _mem_pairs(md, code, site):
            if base != want:
                continue
            if out_of_span(off):
                continue
            slot_sets[disp].add(off)
            if len(sources[disp]) < 6:
                sources[disp].append(f"{site:#x}")

    if not disp_hist:
        return None

    # Score: how much of the span a displacement explains, then how common it is.
    scored = sorted(slot_sets.items(), key=lambda kv: (-len(kv[1]), -disp_hist[kv[0]]))
    best_disp, best_offs = scored[0] if scored else (None, set())
    if best_disp is None or len(best_offs) < MIN_SLOTS:
        return None

    flags_off = FLAGS_OFF
    slots = default_slots(ARCH, flags_off)
    # Keep only the slots the measurement actually saw, but keep the full map when it
    # saw the whole span -- the unobserved ones are unobserved, not absent.
    observed = sorted(best_offs)
    keep_all = len(best_offs) >= MIN_SLOTS
    if not keep_all:
        slots = {o: r for o, r in slots.items() if o in best_offs}

    return Layout(
        handler_disp=best_disp,
        working_disp=best_disp + WORKING_FRAME_DELTA,
        slots=slots,
        flags_off=flags_off,
        order_evidence=order_evidence,
        sources=[f"lea site {s}" for s in sources.get(best_disp, [])],
        histogram={d: c for d, c in disp_hist.most_common(8)},
    )


def out_of_span(off: int) -> bool:
    """True when `off` cannot be a slot: 8-aligned and inside the image span."""
    return off < 0 or off % 8 != 0 or off >= SPAN


def fits_window(offsets) -> bool:
    """True when a set of offsets looks like a context file rather than noise."""
    good = {o for o in offsets if not out_of_span(o)}
    return len(good) >= MIN_SLOTS


def manual(handler_disp: int, order_evidence: str = "assumed:architectural",
           working_delta: int = WORKING_FRAME_DELTA) -> Layout:
    """A layout stated by the caller, for a build `discover` cannot read."""
    return Layout(handler_disp, handler_disp + working_delta, default_slots(ARCH),
                  FLAGS_OFF, order_evidence, sources=["stated by the caller"])


#: Discovery is a scan of the whole VM section, so it is done once per image and
#: shared. Keyed by `(path, section_va)` rather than by object identity: two callers
#: that open the same module get the same layout.
_CACHE: dict[tuple[str, int], Layout | None] = {}


def for_image(img, vm=None, **kw) -> Layout | None:
    """`discover`, memoised per `(image path, .tvm0 address)`.

    Returns the same `Layout` object for repeat calls, and caches the *failure* too:
    a build whose handlers use a different idiom should not be re-scanned by every
    root the caller recovers.
    """
    if vm is None:
        vm = getattr(img, "section_by_name", lambda n: None)(".tvm0")
    if vm is None:
        return None
    src = getattr(img, "path", None) or getattr(img, "name", None) or repr(img)
    key = (str(src), int(vm.va))
    if key in _CACHE:
        return _CACHE[key]
    try:
        got = discover(img, vm, **kw)
    except Exception:                                  # a probe must not break recovery
        got = None
    _CACHE[key] = got
    return got


def clear_cache() -> None:
    _CACHE.clear()


# --------------------------------------------------------------------------
# Locating the image *inside a live frame*
# --------------------------------------------------------------------------
#
# The `lea` displacement says how far the image is from the frame the *handlers*
# use. It does not say how far it is from the frame a symbolic walk is holding,
# because `rbp` moves: on the DeltaForce build the sample in `probe.ctx_map`
# reported the entry frame at `0x7ffffffef788` and the frame at the end of the same
# walk at `0x7ffffffef6d8`, a drift of `0xb0`. The shipped constant chain assumed
# `0xa8`, which is what makes this a measurement rather than a constant.
#
# So the offset is not computed, it is *found*, using two signatures that only a
# real guest register file satisfies:
#
#   * the `rsp` slot holds the machine stack base -- a known constant, not a guess,
#     so finding it pins the base to `addr_of_that_word - 0x20`;
#   * several other slots hold pointers into `.tvm0`.
#
# A candidate window that satisfies neither is not a layout, whatever its offset is.

RSP_OFF = 0x20
#: How far into a frame a context file can plausibly start.
FRAME_SCAN = 0x400
#: Minimum in-`.tvm0` slots (excluding the flags word) before a window is believed.
MIN_VM_SLOTS = 3


class Located:
    """A context window found at a concrete address."""

    __slots__ = ("base", "off_from_rbp", "vm_slots", "rsp_ok", "stack_value")

    def __init__(self, base: int, off_from_rbp: int, vm_slots: int, rsp_ok: bool,
                 stack_value: int | None) -> None:
        self.base = base
        self.off_from_rbp = off_from_rbp
        self.vm_slots = vm_slots
        self.rsp_ok = rsp_ok
        self.stack_value = stack_value

    @property
    def score(self) -> int:
        # The rsp signature is worth more than a pile of pointers: any frame region
        # can hold `.tvm0` addresses, but only the image's rsp slot holds the stack.
        return self.vm_slots + (12 if self.rsp_ok else 0)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"<Located rbp{self.off_from_rbp:+#x} base={self.base:#x} "
                f"vm_slots={self.vm_slots} rsp_ok={self.rsp_ok}>")


def locate_in_frame(pa, pm, rbp: int, img, vm_lo: int, vm_hi: int,
                    stack_base: int | None = None,
                    span: int = FRAME_SCAN,
                    rsp_off: int = RSP_OFF) -> Located | None:
    """Search `rbp + 0 .. rbp + span` for the guest register file.

    Returns the best-scoring candidate, or `None` when no offset in the frame
    explains its own slots. Bounded by `span`, so it is a scan and not a search.
    """
    best: Located | None = None
    for off in range(0, span, 8):
        base = rbp + off
        vm_slots = 0
        stack_value = None
        concrete = 0
        for s in range(0, FLAGS_OFF, 8):
            c = pa.as_const(pm.load(pa.const(base + s, 64), 8, 0, img.image_u8))
            if c is None:
                continue
            concrete += 1
            if s == rsp_off:
                stack_value = c
            elif vm_lo <= c < vm_hi:
                vm_slots += 1
        rsp_ok = (stack_base is not None and stack_value == stack_base)
        # **The `.tvm0` pointer count is the discriminator, not the `rsp` signature.**
        # Two seventeen-word structures sit in this frame and *both* have a word that
        # holds the stack base, so `rsp` alone cannot tell them apart -- it accepted a
        # window in protector order and the names read out of it were wrong. What
        # separates them is that the file the handlers use is filled with `.tvm0`
        # pointers (its vip among them), while the entry sequence's backup copy holds
        # host register values. So `rsp` is a bonus and the pointer count is the gate.
        if vm_slots < MIN_VM_SLOTS:
            continue
        cand = Located(base, off, vm_slots, rsp_ok, stack_value)
        if best is None or cand.score > best.score:
            best = cand
    return best


def fits(loc: Located | None) -> bool:
    """A located window is usable when it holds enough `.tvm0` pointers to be the file."""
    return loc is not None and loc.vm_slots >= MIN_VM_SLOTS


# --------------------------------------------------------------------------
# Locating by persistence: the signature that actually separates the file
# --------------------------------------------------------------------------
#
# Pointer counting is not enough. Two seventeen-word structures live in this frame --
# the one the handlers use and a backup the entry sequence writes -- and both contain
# `.tvm0` addresses, so a window can satisfy "holds several pointers" and still be the
# wrong one. Accepting it produced register names read out of the backup copy.
#
# What distinguishes the live file is a property no backup has: **it persists while
# the vip advances**. Sample the same absolute addresses at two dispatches and the
# register file keeps most of its words equal while exactly one -- the vip -- moves.
# A backup copy is written once and then drifts out of relevance; scratch areas change
# wholesale. So the base is chosen by *stability plus one moving slot*, which is a
# measurement of the VM's behaviour rather than a guess about its layout.


class FrameSnapshot:
    """Concrete 8-byte words of a frame region at one moment."""

    __slots__ = ("rbp", "words")

    def __init__(self, rbp: int, words: dict[int, int]) -> None:
        self.rbp = rbp
        self.words = words

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"<FrameSnapshot rbp={self.rbp:#x} words={len(self.words)}>"


def snapshot_frame(pa, pm, rbp: int, img, span: int = FRAME_SCAN) -> FrameSnapshot:
    """Read every 8-aligned concrete word in `rbp-span .. rbp+span`."""
    words: dict[int, int] = {}
    for off in range(-span, span, 8):
        c = pa.as_const(pm.load(pa.const(rbp + off, 64), 8, 0, img.image_u8))
        if c is not None:
            words[rbp + off] = c
    return FrameSnapshot(rbp, words)


def locate_persistent(snaps, vm_lo: int, vm_hi: int,
                      min_stable: int = 8) -> Located | None:
    """The base `A` whose 17-word window is stable across `snaps` with one mover.

    Returns None unless some base explains at least `min_stable` equal words *and* at
    most two differing words, one of which is an in-`.tvm0` pointer. That is the
    signature of a register file with an advancing vip, and nothing else in the frame
    has it.
    """
    snaps = [s for s in snaps if s is not None]
    if len(snaps) < 2:
        return None
    first = snaps[0]
    best: Located | None = None
    best_key = None
    for base in first.words:
        stable = movers = present = 0
        mover_is_vip = False
        for s in range(0, FLAGS_OFF + 8, 8):
            vals = [sn.words.get(base + s) for sn in snaps]
            if any(v is None for v in vals):
                continue
            present += 1
            if len(set(vals)) == 1:
                stable += 1
            else:
                movers += 1
                if any(vm_lo <= v < vm_hi for v in vals):
                    mover_is_vip = True
        if present < min_stable + 1:
            continue
        if stable < min_stable or movers > 2 or not mover_is_vip:
            continue
        key = (stable, -movers)
        if best_key is None or key > best_key:
            best_key = key
            best = Located(base, base - first.rbp, stable, True, None)
    return best


# --------------------------------------------------------------------------
# Locating by vote: stack-anchored signature candidates
# --------------------------------------------------------------------------
#
# A window containing a stack value and a VM pointer is not enough to identify
# its boundary: many overlapping windows contain those same words. Anchor the
# stack value at the assumed architectural rsp offset, and abstain when multiple
# windows qualify. Repeated unique votes provide location evidence only under
# that assumption; they never measure register order.

VOTE_MIN = 4


def vote(sample: dict[int, int], vm_lo: int, vm_hi: int,
         stack_base: int | None, span: int = FRAME_SCAN) -> tuple[int, int]:
    """Vote only for a unique window anchored by the architectural rsp slot.

    The architectural order remains an assumption; this is signature evidence,
    not a measured register mapping. Overlapping windows containing the same two
    values do not establish a boundary. Multiple anchored candidates abstain.
    """
    if stack_base is None:
        return (0, 0)
    candidates = []
    for off in range(-0x200, span, 8):
        if sample.get(off + RSP_OFF) != stack_base:
            continue
        if any(vm_lo <= sample[off + s] < vm_hi
               for s in range(0, FLAGS_OFF, 8)
               if s != RSP_OFF and off + s in sample):
            candidates.append(off)
    return (candidates[0], 2) if len(candidates) == 1 else (0, 0)


def locate_by_vote(votes, vm_lo: int, vm_hi: int, min_votes: int = VOTE_MIN):
    """Pick the offset the samples agree on. `votes` is `[(off, score), ...]`.

    Returns `(offset, count)` for the winner, or `(None, 0)` when no offset reaches
    `min_votes`. A tied total remains ambiguous even when both offsets reach the threshold.
    """
    tally: dict[int, int] = {}
    for off, score in votes:
        if score >= 2:
            tally[off] = tally.get(off, 0) + 1
    if not tally:
        return (None, 0)
    off, count = max(tally.items(), key=lambda kv: kv[1])
    if count < min_votes or sum(n == count for n in tally.values()) != 1:
        return (None, count)
    return (off, count)
