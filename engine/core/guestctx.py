"""The guest register image: where it is, and what each slot means.

Two seventeen-slot structures sit in the VM's entry frame, adjacent to each other,
and for several rounds this workspace knew about only the first of them.

* At `+0x08`, the sixteen general registers in the order the protector chose --
  `rdi` is between `rcx` and `rdx` -- with the flags word at `+0x00`. The entry
  sequence writes it once. **Handlers do not write it**: seeding it and watching
  gave 400 of 400 transitions unchanged, which is correct and was read as "the
  image is a backup".
* At `+0x90`, the sixteen general registers in **architectural** order with the
  flags word at `+0x80`. Handlers both read and write this one.

`evidence/E-GUEST-CONTEXT-LOCATED.md` records the derivation. Briefly: the entry
sequence marks the first block by saving the machine registers into it; replacing
that block's slots with fresh symbols turns every later copy *out of* it into a
value whose destination can be observed; and fitting a base such that a register
lands at `base + 8 * architectural_index(r)` explains 13/13, 14/14 and 15/15
registers on three functions with three different frames, where the runner-up
base explains 5, 11 and 6. The reference implementation's `vmctx` command names
the same absolute address and the same layout, which is corroboration and not the
derivation.

The offset is expressed from the **working frame**, because that is the frame the
recovery loop samples:

    entry_rbp = working_rbp + 0xa8
    guest context = entry_rbp + 0x90 = working_rbp + 0x138

Both `0xa8` and `0x90` are measured for this build. `GuestContext` is constructed
per root from the sampled frame, so nothing assumes a fixed stack base, and a
caller can override the offset when it meets a build where the measurement does
not hold.
"""

from __future__ import annotations

# The architectural encoding order, which is the order of the image: the fit that
# found it would have found any other order equally well, and it found this one.
ARCH = ("rax", "rbx", "rcx", "rdx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
FLAGS = "rflags"
FLAGS_OFF = 0x80

# The image is at `entry_rbp + 0x90`, and the frame the recovery loop samples is
# `entry_rbp + 0xa8`, so the offset from the working frame is the sum. The working
# frame is the one the engine has, and the entry frame is the one a probe has when
# it walks from the root; deriving one from the other keeps the two from drifting.
GUEST_CTX_FROM_ENTRY_RBP = 0x90
WORKING_FRAME_DELTA = 0xA8
GUEST_CTX_FROM_WORKING_RBP = GUEST_CTX_FROM_ENTRY_RBP + WORKING_FRAME_DELTA

W64 = 64


class GuestContext:
    """Maps an address to the guest register it holds, and back."""

    __slots__ = ("base", "_by_addr", "_by_reg")

    def __init__(self, base: int):
        self.base = base
        self._by_addr: dict[int, str] = {}
        self._by_reg: dict[str, int] = {}
        for i, reg in enumerate(ARCH):
            self._by_addr[base + 8 * i] = reg
            self._by_reg[reg] = base + 8 * i
        self._by_addr[base + FLAGS_OFF] = FLAGS
        self._by_reg[FLAGS] = base + FLAGS_OFF

    @property
    def span(self) -> tuple[int, int]:
        return self.base, self.base + FLAGS_OFF + 8

    def reg_at(self, addr: int) -> str | None:
        """The guest register an address names, or None if it is not a slot."""
        return self._by_addr.get(addr)

    def addr_of(self, reg: str) -> int | None:
        return self._by_reg.get(reg)

    def offset(self, reg: str) -> int | None:
        a = self._by_reg.get(reg)
        return None if a is None else a - self.base

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"<GuestContext base={self.base:#x} slots={len(self._by_addr)}>"


def from_working_frame(working_rbp: int,
                       delta: int = GUEST_CTX_FROM_WORKING_RBP) -> GuestContext:
    """The context for a root whose working frame base has been sampled."""
    return GuestContext(working_rbp + delta)


# How a `GuestCache` was filled. The two answer different questions and disagree,
# so the provenance travels with the object rather than living in whoever built it.
MATERIALISATION = "materialisation layout (context read correlated with the register write in the same instruction)"
HANDLER_TIME = "handler-time cache (context seeded with fresh symbols; host registers holding exactly one symbol at a dispatch boundary)"


class GuestCache:
    """Which host register is holding which guest register -- and *how we know*.

    **The two ways of filling this disagree, and that is deliberate.** The
    disagreement is the main result of the round that added this class, so it is
    written here rather than left as an implementation detail.

    The context block is the guest's register file, but it is not where the guest
    registers spend their time: only 8.5-9.3 % of handlers touch it. The values
    live in the host register file between materialisations, and the assignment is
    chosen per virtualized function, so it has to be derived rather than assumed.

    * `MATERIALISATION`, produced by the census riding along with recovery: when a
      load lands on a context slot, remember the value node it returned, and treat
      a register write *in the same instruction* storing that very node as a
      register loaded from that slot. Node identity is exact, so `mov rsi,
      [ctx+0x10]` is recognised without disassembling anything. Free, but it
      answers "which host register is each slot loaded into".
    * `HANDLER_TIME`, produced by `Recoverer.derive_guest_cache`: seed the context
      with fresh symbols and watch which host register each one reaches. A separate
      bounded walk, and it answers "which host register is the handler actually
      reading guest `rcx` from".

    On `0x14015fc2e` the first says `rsi <- guest rsi`, from a single observation.
    The second says `rsi <- guest rcx`, stable across **107 of 107** dispatch
    boundaries. Both are true and they are about different moments: the VM
    materialises `ctx[R]` into host `R`, and a permutation applied afterwards puts
    guest `rcx` in `rsi` for the handler to use. **The handler-time map is the one
    that matters for semantics.** Check `source` before reading `assignment`.

    The first attempt at the census version looked for a register holding a bare
    `load(ctx+slot)` node and found none at all -- a lesson about this VM rather
    than about the code: once the VM has *written* a slot, the machine's memory
    holds the bytes, and a later read rebuilds the stored expression byte by byte
    instead of interning a fresh load. A bare load node exists only the first time
    a slot is read. Correlating by the returned value works in both cases.

    Ambiguity is recorded rather than resolved: a host register seen carrying two
    different guest registers is **staging**, not cache, and `guest_of` declines to
    name it. `rdx` and `r10` are exactly that, and a version of this that picked
    the most common answer would have silently labelled `rdx` as eleven different
    guest registers in turn.

    **The counts are conditional, and that matters.** A host register is credited on
    the dispatch boundaries where it carried a seeded symbol -- not on all of them.
    Over the whole of `0x14015fc2e` (798 boundaries) the seeded walk gives
    `r12`/`r13`/`r14`/`rdi` 596 each, `r15` 591, `rsi` 170, `rbx` 71. So
    "`rsi` <- guest `rcx`" means **"whenever `rsi` carried a seeded symbol it was
    `rcx`, on 170 of 798 boundaries"**, and not "`rsi` always holds `rcx`". The
    other 628 boundaries have `rsi` doing something else -- the same register is
    scratch between materialisations. `guest_of` answers "is this register's guest
    identity unambiguous", which is the question that has an answer; it does not
    answer "is this register always the guest register", which it is not. `counts()`
    exists so a caller can see the coverage rather than infer it.
    """

    __slots__ = ("_by_host", "_counts", "source", "boundaries", "ctx", "walk_status")

    def __init__(self, source: str = "unspecified") -> None:
        self.source = source
        # How many dispatch boundaries the map was built from. Needed because a count
        # without a denominator says nothing: `rsi` carrying guest `rcx` **170** times is
        # a different claim at 798 boundaries than at 200.
        self.boundaries = 0
        self.walk_status: dict = {}
        # **The guest context this map was derived against.** `derive_guest_cache` samples
        # `rbp` at every dispatch boundary and builds the context the first time it is
        # concrete; `Recoverer.recover` samples it once, in the first block, and when that
        # first sample is symbolic it gets nothing. On `0x1401687d5` that difference made the
        # whole census a silent no-op -- 0 writes, 0 marks, a 15187-instruction function
        # devirtualising to one line -- while the cache derivation on the same root found 460
        # boundaries. Carrying the frame here lets a recovery use the one that was found.
        self.ctx = None
        self._by_host: dict[str, set[str]] = {}
        self._counts: dict[str, dict[str, int]] = {}

    def stability(self, host: str) -> float | None:
        """The fraction of boundaries at which `host` carried *its* guest register.

        `None` when the register is staging and has no single guest. This is the number
        that decides whether a register may be treated as holding its guest register at
        an arbitrary point: `rdi` at 596/798 may, `rsi` at 170/798 may not, and a caller
        that ignores the difference attributes correctly 21 % of the time and wrongly the
        other 79 % -- which is exactly what round 17's seeding did before this existed.
        """
        guest = self.guest_of(host)
        if guest is None or not self.boundaries:
            return None
        return self._counts[host][guest] / self.boundaries

    def note(self, host: str, guest: str) -> None:
        """Record that `host` was just loaded from `guest`'s context slot."""
        self._by_host.setdefault(host, set()).add(guest)
        self._counts.setdefault(host, {})
        self._counts[host][guest] = self._counts[host].get(guest, 0) + 1

    def guest_of(self, host: str) -> str | None:
        """The guest register `host` caches, or None if it stages several."""
        seen = self._by_host.get(host)
        if not seen or len(seen) != 1:
            return None
        return next(iter(seen))

    def host_of(self, guest: str) -> str | None:
        for host, seen in self._by_host.items():
            if seen == {guest}:
                return host
        return None

    @property
    def assignment(self) -> dict[str, str]:
        return {h: self.guest_of(h) for h in self._by_host
                if self.guest_of(h) is not None}

    @property
    def staging(self) -> dict[str, list[str]]:
        return {h: sorted(s) for h, s in self._by_host.items() if len(s) > 1}

    def counts(self, host: str) -> dict[str, int]:
        return dict(self._counts.get(host, {}))

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"<GuestCache {self.source.split(' (')[0]} "
                f"cache={self.assignment} "
                f"staging={ {h: len(v) for h, v in self.staging.items()} }>")
