"""Abstract machine state for the guided evaluator.

Every architectural value is an `expr.Ref`. A value is concrete exactly when its
node is a constant, which is what lets one code path handle "the VM's own
bookkeeping, which always folds" and "the guest's data, which never does"
without a separate concrete emulator.

Memory is tracked by *concrete address only*. An access whose address does not
fold is recorded as a precision loss and returns an interned `load` node: the
interner is what makes two identical symbolic loads cancel in a subtraction,
which is how the protector's self-cancelling dispatch arithmetic collapses.

RSP is the one register that starts concrete. That is not an optimisation: the
VM keeps its own frame, its guest register image and its spill area on the
machine stack, so a concrete RSP is what turns the entire context into
concretely-addressed memory that the evaluator can read like an array.
"""

from __future__ import annotations

import re

from .callsum import UNKNOWN_WORK_BELOW
from .expr import W8, W16, W32, W64, Arena, mask
from .memory import (address_bounds, add_clobber, finite_addresses,
                      guest_only_address, may_alias)

# ------------------------------------------------------------------ registers

GP64 = (
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
)

# Sub-register -> (parent 64-bit register, bit width, bit offset)
SUB: dict[str, tuple[str, int, int]] = {}
for _r in GP64:
    if _r.startswith("r") and _r[1:].isdigit():
        _base = _r
        SUB[_r] = (_base, 64, 0)
        SUB[_r + "d"] = (_base, 32, 0)
        SUB[_r + "w"] = (_base, 16, 0)
        SUB[_r + "b"] = (_base, 8, 0)
    else:
        SUB[_r] = (_r, 64, 0)
SUB.update(
    {
        "eax": ("rax", 32, 0), "ax": ("rax", 16, 0), "al": ("rax", 8, 0), "ah": ("rax", 8, 8),
        "ebx": ("rbx", 32, 0), "bx": ("rbx", 16, 0), "bl": ("rbx", 8, 0), "bh": ("rbx", 8, 8),
        "ecx": ("rcx", 32, 0), "cx": ("rcx", 16, 0), "cl": ("rcx", 8, 0), "ch": ("rcx", 8, 8),
        "edx": ("rdx", 32, 0), "dx": ("rdx", 16, 0), "dl": ("rdx", 8, 0), "dh": ("rdx", 8, 8),
        "esi": ("rsi", 32, 0), "si": ("rsi", 16, 0), "sil": ("rsi", 8, 0),
        "edi": ("rdi", 32, 0), "di": ("rdi", 16, 0), "dil": ("rdi", 8, 0),
        "ebp": ("rbp", 32, 0), "bp": ("rbp", 16, 0), "bpl": ("rbp", 8, 0),
        "esp": ("rsp", 32, 0), "sp": ("rsp", 16, 0), "spl": ("rsp", 8, 0),
        "rip": ("rip", 64, 0), "eip": ("rip", 32, 0), "ip": ("rip", 16, 0),
    }
)

# Flags the lifter models, and their RFLAGS bit positions.
FLAG_BITS = {"cf": 0, "pf": 2, "af": 4, "zf": 6, "sf": 7, "df": 10, "of": 11}

#: Control and debug registers (`cr8`, `cr3`, ...). They are *not* guest state and their value is not
#: derivable from the image, so each is a slot whose first read is an opaque -- the same treatment every
#: other unknown input gets. Anything else stays `UnmodelledRegister`.
_CONTROL_RE = re.compile(r'^(?:cr|dr)\d+$')
FLAGS = tuple(FLAG_BITS)
RFLAGS_FIXED = 0x2

ARG_REGS = ("rcx", "rdx", "r8", "r9")
VOLATILE = ("rax", "rcx", "rdx", "r8", "r9", "r10", "r11")

# The machine's own stack allocation: the window around the stack base this machine was
# constructed with. It is the region the *protector* runs on, and it is what the escape
# assumption in `Machine.store` is allowed to spare -- nothing more.
#
# Measured need, `probe/frameregion.py` on `0x140155dc1`: all 4675 concrete stores land inside
# `[stack_base - 0x9f8, stack_base + 0x26]`, and all 980-1060 tracked bytes that a guest-pointer
# store would otherwise delete lie in that range. The bounds below are 16x the observed depth
# below and 64x the observed reach above, so a deeper frame on another root stays inside, while
# staying 2**47 away from every guest address this protector is seen to use (the image at
# `0x140000000`, guest pointers at `0x9000` scale).
FRAME_WINDOW_BELOW = 0x1_0000
FRAME_WINDOW_ABOVE = 0x1000

# Ceiling on the pending guest-pointer writes a read may consult. Real roots need three; past the
# cap the oldest store falls back to deleting the bytes it may reach -- sound, and counted -- rather
# than growing every later read's expression without limit.
MAX_PENDING_ISOLATED = 64


def index_range_width(arena: "Arena", node: int) -> int:
    """How wide the *range* of an index expression actually is.

    `Arena.width` reports the expression's type width, which is not the same thing and
    was a bug in the first version of `sym_store_windows`: `zext64(trunc16(e))` is 64
    bits wide and takes only 2**16 values, so recording 64 made every window cover the
    whole address space and every failing read match all 113 of them. The instrument's
    own output is what exposed it -- "span 0x100000000" on a store the earlier
    `probe/scratch_slot` had printed as `base + trunc16(...)`.

    A narrowing or widening node bounds the range by what is inside it; anything else
    is bounded by its own width. Deliberately conservative: an unrecognised shape
    returns the type width, which over-states the range rather than under-stating it,
    because under-stating would turn "maybe" into "no".
    """
    n = arena.node(node)
    if n[0] in ("zext", "sext"):
        return arena.width(n[1])
    if n[0] == "trunc":
        return arena.width(node)
    return arena.width(node)


class UnmodelledRegister(KeyError):
    """A register the machine model does not carry.

    Subclasses `KeyError` so every existing `except KeyError` around `canon` keeps working, while the
    lifter can name the case precisely. It exists because a `KeyError: 'cr8'` escaping `step()` killed
    a whole corpus run: the VM's own handler code executes `mov ..., cr8` on `ACE-CORE.sys3`, and an
    unmodelled control register is an *unsupported instruction*, not a crash. The lifter turns this
    into `Stop("unsupported", text=mn)`, which is a named stop the recovery already handles.
    """


def canon(name: str) -> tuple[str, int, int]:
    try:
        return SUB[name]
    except KeyError:
        pass
    # **A control or debug register is a register, not an unsupported instruction.**
    # The VM's own handler code moves `cr8` and `cr3` (`ACE-BOOT.sys`, `ACE-SSC-DRV64.sys`), and this
    # model used to stop there with "unmodelled register", which refused the whole root -- two roots on
    # `ACE-BOOT.sys` and one on `ACE-SSC-DRV64.sys` in the nine-image corpus. What the model can say
    # about such a register is exactly what it says about any other unknown input: nothing. So it is a
    # 64-bit slot whose first read is an opaque, kept per run, and a write replaces it. Nothing about
    # the *guest* register file changes, which is the point: `cr8` is not guest state.
    m = _CONTROL_RE.match(name)
    if m:
        return (name, W64, 0)
    raise UnmodelledRegister(name) from None


class Machine:
    """Guest architectural state with hash-consed values."""

    # Modelled from the callee's pointer arguments. `False` restores the earlier model, for A/B.
    call_memory_model_default = True
    # A `forget` wider than this is recorded, because the region is a *bounding box* over an
    # observed store address and a pointer argument, and those can be far apart: on
    # `ACE-CORE.sys3` root `0x145bbb46a` the box is about 128 TiB. Forgetting more than the callee
    # wrote is sound -- it only costs knowledge -- but a reader must be able to see that a root gave
    # up its memory knowledge wholesale, so the width is counted rather than hidden.
    WIDE_FORGET = 1 << 24
    # Whether forgetting a call's writes moves the memory version. It must, for soundness; the
    # switch exists so the coverage cost can be attributed to the version or to the bytes.
    call_bump_version_default = True

    def __init__(self, arena: Arena, stack_base: int):
        self.a = arena
        self.stack_base = stack_base
        self.regs: dict[str, int] = {r: arena.init_reg(r) for r in GP64}
        self.flags: dict[str, int] = {
            f: (arena.const(0, W8) if f == "df" else arena.opaque(f"init_{f}", W8))
            for f in FLAGS
        }
        self.mem: dict[int, object] = {}
        # Epoch of the newest explicit byte value. Deferred symbolic stores older
        # than this value have already been superseded and must not be replayed.
        self.byte_versions: dict[int, int] = {}
        self.xmm: dict[tuple[int, int], int] = {}
        self.sym_loads: list[tuple[int, int]] = []  # (site, width)
        self.sym_stores: list[tuple[int, int]] = []
        # Counts every store performed. A read records the version it happened at, so two reads of
        # the same location intern to the same node only when no store came between them -- the fix
        # for `load(p); store(p, 5); load(p)` returning one node for both reads and folding their
        # difference to zero. See `Arena.load` and `store`.
        self.mem_version = 0
        self.symbolic_mem: dict[tuple[int, int], int] = {}
        self.clobbered_ranges: list[tuple[int, int]] = []
        # Set by the image owner. Recovery models non-faulting execution;
        # unknown stores cannot change a read-only mapped image section.
        self.readonly = lambda addr, size: False
        self._historical_cache = {}
        self._precision_views = {}
        # (base, index width, site) for every symbolic-address store whose address
        # decomposed as `const + expr`. See `store` and `possible_writers`.
        self.sym_store_windows: list[tuple[int, int, int]] = []
        self.unmapped_loads: list[tuple[int, int]] = []
        self.regs["rsp"] = arena.const(stack_base, W64)
        # The private frame, as the interval the escape assumption may spare. See the constants
        # above for the measurement, and `_isolated_store` for what is done with it.
        self.stack_window: tuple[int, int] | None = (
            (stack_base - FRAME_WINDOW_BELOW, stack_base + FRAME_WINDOW_ABOVE)
            if stack_base else None)
        # **The conditions the recovery is carrying.** Every byte the escape assumption spared is
        # a derivation that holds only while a guest pointer never names the machine's own stack;
        # these lists are what make that auditable, and `refute_isolation` is what discharges it
        # against evidence instead of leaving it as a claim in a docstring.
        self.isolation_conditions: list[tuple[int, int, int]] = []  # (site, nbytes, spared)
        self.isolation_spared: set[int] = set()
        self.isolation_refuted = 0
        self.isolation_log: list[tuple[int, int]] = []
        # **The pending guest-pointer writes, which are what a read consults.** The conditions live
        # here rather than being written into `mem` at store time, so a byte that is cached and a
        # byte that still holds its image value are answered by the same rule -- see `_guard_base`.
        # Part of the machine state, therefore rolled back with it (`core/state.py`).
        self.pending_isolated: list[tuple] = []
        self.guards_built = 0
        self.guards_dropped = 0
        self.guards_applied = 0
        # The census's identity map, assigned **once** when the census is enabled and never
        # toggled. It used to be a `frame_separated` flag set around a single store, which made
        # recording an observation change program behaviour -- see `guest_only_address`.
        self.guest_marks: dict | None = None
        # Whether a native call's memory effects are modelled from its argument pointers. A switch
        # rather than a rule so the A/B is available: `--nocallmem` sets
        # `Machine.call_memory_model_default = False`, which restores the model that clobbered
        # registers and flags and nothing else.
        self.call_memory_model = Machine.call_memory_model_default
        self.call_bump_version = Machine.call_bump_version_default
        self.call_memory_checked = 0
        self.call_memory_forgotten = 0
        # Forgets wider than `Machine.WIDE_FORGET`, and the widest one seen. See `forget`.
        self.wide_forgets = 0
        self.widest_forget = 0
        # **Not `None`.** These counters are summed across roots by `probe.corpus_compare`, which
        # does `total[k] += v`; a None makes the whole comparison die on a root that never widened a
        # forget. Zero is also the truthful value there -- "no wide forget happened".
        self.widest_forget_addr = 0        # Native calls whose observed writes land in the VM's own frame, i.e. transfers into code
        # that behaves like handler code rather than a leaf. See `_outside_window`.
        self.calls_into_window = 0
        # Half-open range of the VM's own scratch area whose *contents* are to be
        # treated as unknowable. Off by default; the recovery driver switches it
        # on only for a block whose precise pass produced a provably impossible
        # result, which is the one situation where the stored bytes cannot be
        # trusted to be the bytes the VM would see.
        self.symbolic_window: tuple[int, int] | None = None
        # Set by `Lifter.__init__` when the `not`/`xchg` deobfuscation is on. The
        # machine only reports writes to it; the decision to undo a swap lives in
        # `mba.SwapTracker`, so this file stays free of protector-specific policy.
        self.swap_tracker = None

    # ------------------------------------------------------------- registers

    def reg(self, name: str, width: int | None = None) -> int:
        parent, w, off = canon(name)
        if parent == "rip":
            raise KeyError("rip is supplied by the instruction pointer, not the register file")
        if parent not in self.regs:
            # First touch of a control/debug register: an unknown input, named so a reader can see it.
            self.regs[parent] = self.a.opaque(f"ctrl_{parent}", W64)
        v = self.regs[parent]
        if width is None or width == 64:
            return v
        if off:
            v = self.a.bin("shr", v, self.a.const(off, W64), W64)
        return self.a.trunc(v, width)

    def set_reg(self, name: str, value: int, width: int | None = None) -> None:
        """Write a register, honouring x86's partial-write rules.

        A 32-bit write zero-extends; 8/16-bit writes preserve the rest; a
        high-byte write (`ah`) replaces bits 8..15. Getting this wrong silently
        corrupts the obfuscator's inverted-register dance, which mixes byte and
        word accesses to the same register on purpose.

        Every write is reported to `swap_tracker` when one is installed, because an
        `xchg` pair can only be undone while neither of its registers has been
        written in between. See `core/mba.SwapTracker`.
        """
        parent, w, off = canon(name)
        if parent == "rip":
            raise KeyError("cannot write rip through the register file")
        if self.swap_tracker is not None:
            self.swap_tracker.on_write(name)
        if parent not in self.regs:
            self.regs[parent] = self.a.opaque(f"ctrl_{parent}", W64)
        if width is None:
            width = w
        if width == 64:
            self.regs[parent] = value
            return
        if width == 32:
            self.regs[parent] = self.a.zext(value, W64)
            return
        keep = mask(W64) ^ (mask(width) << off)
        wide = self.a.zext(value, W64)
        if off:
            wide = self.a.bin("shl", wide, self.a.const(off, W64), W64)
        self.regs[parent] = self.a.bin("or", self.a.bin("and", self.regs[parent], self.a.const(keep, W64), W64), wide, W64)

    # ----------------------------------------------------------------- flags

    def flag(self, f: str) -> int:
        return self.flags[f]

    def set_flag(self, f: str, v: int) -> None:
        self.flags[f] = self.a.trunc(v, W8) if self.a.width(v) != W8 else v

    def set_flags_unknown(self, names=FLAGS) -> None:
        for f in names:
            self.flags[f] = self.a.opaque(f"flag_{f}", W8)

    def pack_flags(self) -> int:
        acc = self.a.const(RFLAGS_FIXED, W64)
        for f, bit in FLAG_BITS.items():
            b = self.a.zext(self.a.bin("and", self.flag(f), self.a.const(1, W8), W8), W64)
            acc = self.a.bin("or", acc, self.a.bin("shl", b, self.a.const(bit, W64), W64), W64)
        return acc

    def unpack_flags(self, value: int) -> None:
        for f, bit in FLAG_BITS.items():
            shifted = self.a.bin("shr", value, self.a.const(bit, W64), W64)
            # Mask each flag back down to its own bit. Without this, `zf` after a
            # `popfq` is bits 6..13 of RFLAGS -- SF and DF leak into it -- and
            # every `jz` on the round-tripped value becomes a branch on four
            # flags at once.
            self.set_flag(f, self.a.trunc(self.a.bin("and", shifted, self.a.const(1, W64), W64), W8))

    # ---------------------------------------------------------------- memory

    def concrete_rsp(self) -> int | None:
        return self.a.as_const(self.regs["rsp"])

    def read_image(self, addr: int, width: int, image_u8) -> int | None:
        """Little-endian value from the loaded image, or None if not mapped."""
        acc = 0
        for i in range(width):
            b = image_u8(addr + i)
            if b is None:
                return None
            acc |= b << (8 * i)
        return acc

    def load(self, addr: int, nbytes: int, site: int, image_u8) -> int:
        """Load `nbytes` bytes from the (possibly symbolic) address `addr`.

        Sizes are in bytes because that is how an instruction encodes them; the
        expression width is derived once, here.
        """
        a = self.a
        if a.is_const(addr):
            return self.load_concrete(a.as_const(addr), nbytes, site, image_u8)

        # Symbolic address: hand back an interned load. Two loads at the same
        # symbolic address must be the *same* node, or `load(A) - load(A)` will
        # not fold and the dispatch arithmetic stays opaque.
        self.sym_loads.append((site, nbytes))
        forwarded = self.symbolic_mem.get((addr, nbytes))
        if forwarded is not None:
            return a.trunc(forwarded, nbytes * 8)
        return a.load(addr, nbytes * 8, self.mem_version)

    def load_concrete(self, addr: int, nbytes: int, site: int, image_u8) -> int:
        a = self.a
        width = nbytes * 8
        win = self.symbolic_window
        if win is not None and win[0] <= addr < win[1] and nbytes < 8:
            # Deliberate loss of precision, requested by the driver for one
            # block. An interned load keeps the value identifiable, so two reads
            # of the same address still cancel -- without that, the dispatch
            # arithmetic built on such a read would not fold at all.
            #
            # Only sub-qword reads. The VM's working area holds both its own
            # link pointers, which are qword entries and are structural, and the
            # values it derives from bytecode and guest state. Symbolising the
            # qword entries too severs the pointer chains the VM walks to reach
            # its own tables, which costs far more than it recovers.
            self.sym_loads.append((site, nbytes))
            key = (self.mem_version, win)
            if key not in self._precision_views:
                self._precision_views[key] = -a.new_memory_version()
            return a.load(a.const(addr, W64), width, self._precision_views[key])

        # Each byte is tagged, because a constant byte and a *reference to a
        # node* are both Python ints. An earlier version of this function told
        # them apart with `isinstance(p, int)`, which meant every unmapped byte
        # -- whose placeholder is a node reference -- was read as the constant
        # `arena_index`. The result was plausible-looking garbage: runs of
        # consecutive small numbers that looked like a memory fill pattern and
        # were in fact node indices. Constants are ("c", value); a symbolic byte
        # is ("s", ref) for a whole byte value, or ("p", ref, index) for byte
        # `index` of a wider value.
        pieces: list[tuple[str, int] | tuple[str, int, int]] = []
        need_symbolic = False
        unmapped = False
        for i in range(nbytes):
            # **One read rule for both populations.** A byte that is cached (`mem`) and a byte that
            # still holds its image value are two branches of the same question -- "what does this
            # byte hold" -- and a guest-pointer store that may have written it has to be applied to
            # both. `_guard_base` returns None when nothing may alias, which keeps the constant path
            # and the `contiguous` fast path below exactly as they were.
            ba = addr + i
            cur = self.mem.get(ba)
            if cur is None:
                clobbered = (not self.readonly(ba, 1)
                             and any(lo <= ba < hi for lo, hi in self.clobbered_ranges))
                b = None if clobbered else image_u8(ba)
                if b is None:
                    unmapped = True
                    break
                piece: tuple = ("c", b)
                base = a.const(b, W8)
            elif isinstance(cur, int):
                piece = ("c", cur)
                base = a.const(cur, W8)
            else:
                src, index = cur
                piece = ("p", src, index)
                base = self._byte_of(src, index)
            guarded = self._guard_base(ba, base)
            if guarded is not None:
                need_symbolic = True
                pieces.append(("p", guarded, 0))
                continue
            if piece[0] == "p":
                need_symbolic = True
            pieces.append(piece)

        if unmapped:
            # Nothing is known about this read, so it becomes *one* symbolic load
            # at the address -- the same node the symbolic-address path builds.
            #
            # Building it from one opaque placeholder per byte instead is what
            # this used to do, and it is wrong in a way that never looks wrong:
            # the eight placeholders are eight distinct nodes, so an eight-term
            # `or`-chain comes out, nothing cancels, and two reads of the very
            # same address produce two different expressions. The reference
            # refuses the whole load for the same reason.
            for i in range(nbytes):
                self.unmapped_loads.append((site, addr + i))
            self.sym_loads.append((site, nbytes))
            return a.load(a.const(addr, W64), width, self.mem_version)

        if not need_symbolic:
            acc = 0
            for i, p in enumerate(pieces):
                acc |= int(p[1]) << (8 * i)
            return a.const(acc, width)

        # A contiguous run of bytes projected from one wider value *is* that
        # value truncated, at any width -- not only at eight bytes. Gating this
        # on `nbytes == 8` meant a 4-byte read of a symbolic word was rebuilt as
        # `b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)`, four shifts and three `or`s
        # where one `trunc` says the same thing. That shape then propagates: it
        # is unrecognisable as a value, so nothing downstream folds, and the
        # difference is invisible because the bytes are all correct.
        first = pieces[0]
        if first[0] == "p" and first[2] == 0:
            contiguous = all(
                p[0] == "p" and p[1] == first[1] and p[2] == i
                for i, p in enumerate(pieces)
            )
            if contiguous:
                return a.trunc(first[1], width)

        acc_ref: int | None = None
        for i, p in enumerate(pieces):
            if p[0] == "c":
                r = a.const(p[1], W8)
            elif p[0] == "s":
                r = p[1]
            else:
                r = self._byte_of(p[1], p[2])
            wide = a.zext(r, width) if width > 8 else r
            if i:
                wide = a.bin("shl", wide, a.const(i * 8, width), width)
            acc_ref = wide if acc_ref is None else a.bin("or", acc_ref, wide, width)
        assert acc_ref is not None
        return acc_ref

    def _byte_of(self, src: int, index: int) -> int:
        a = self.a
        sw = a.width(src)
        if index >= sw // 8:
            return a.const(0, W8)
        v = a.bin("shr", src, a.const(index * 8, sw), sw) if index else src
        return a.trunc(v, W8)

    def store(self, addr: int, value: int, nbytes: int, site: int) -> None:
        a = self.a
        # **Any store may change what a symbolic read returns, so the version moves.**
        # `load(p); store(p, 5); load(p)` previously produced the *same* node for both reads and
        # the subtraction folded to zero -- a wrong constant, silently. A store at a symbolic
        # address may alias anything, so it bumps too. Concrete stores to an address the read
        # cannot reach also bump, which is conservative and costs precision rather than
        # correctness; the alternative is an alias analysis that does not exist here yet.
        parent = self.mem_version
        self.mem_version = a.new_memory_version()
        a.memory_versions[self.mem_version] = (parent, addr, value, nbytes)
        self.symbolic_mem = {
            (old_addr, old_bytes): old_value
            for (old_addr, old_bytes), old_value in self.symbolic_mem.items()
            if not may_alias(a, addr, nbytes, old_addr, old_bytes)
        }
        if not a.is_const(addr):
            self.sym_stores.append((site, nbytes))
            # A guest pointer. See `_isolated_store`: the escape assumption is what makes this a
            # different kind of store, and it is scoped to the machine's own stack window.
            isolated = guest_only_address(a, addr, self.guest_marks)
            split = finite_addresses(a, addr)
            if split is not None:
                cond, zero, one = split
                if isolated:
                    # The address stopped being unbounded: it provably takes two concrete
                    # values, which is the one place this assumption becomes *decidable* rather
                    # than assumed. Testing it here is what turns it into a checked claim.
                    for cand in dict.fromkeys((zero, one)):
                        self.refute_isolation(cand, site)
                for target in dict.fromkeys((zero, one)):
                    old = Machine.load_concrete(self, target, nbytes, site, lambda _: None)
                    if zero == one:
                        merged = a.trunc(value, nbytes * 8)
                    elif target == one:
                        merged = a.select(cond, a.trunc(value, nbytes * 8), old, nbytes * 8)
                    else:
                        merged = a.select(cond, old, a.trunc(value, nbytes * 8), nbytes * 8)
                    constant = a.as_const(merged)
                    for i in range(nbytes):
                        self.mem[target + i] = ((constant >> (8 * i)) & 0xff
                                                if constant is not None else (merged, i))
                        self.byte_versions[target + i] = self.mem_version
                self.symbolic_mem[(addr, nbytes)] = a.trunc(value, nbytes * 8)
                return
            lo, hi = address_bounds(a, addr, nbytes)
            if isolated:
                self._isolated_store(addr, value, nbytes, site, lo, hi)
                return
            self.clobbered_ranges = add_clobber(self.clobbered_ranges, lo, hi)
            for byte_addr in list(self.mem):
                if lo <= byte_addr < hi and not self.readonly(byte_addr, 1):
                    del self.mem[byte_addr]
            self.symbolic_mem[(addr, nbytes)] = a.trunc(value, nbytes * 8)
            # Where could it have landed? A store with a symbolic address is dropped
            # -- the memory is not updated and no value is recorded -- which is
            # **sound** (a later read of that address returns an unmapped symbol
            # rather than a stale byte) and **imprecise in a way that was invisible**:
            # the engine could not tell "nothing ever wrote here" from "a store whose
            # address it could not resolve may have written here", and both were
            # reported as the same unmapped read.
            #
            # Measured on `0x14015fc2e`: **171** stores are dropped this way, against
            # 5999 performed, and the four unresolved blocks all fail on a constant
            # read inside the window one of them covers. So the *base* is recorded
            # when the address decomposes as `const + expr`, together with the width
            # of the index, which bounds how far it could have reached.
            n = a.node(addr)
            if n[0] == "bin" and n[1] == "add":
                for x, y in ((n[2], n[3]), (n[3], n[2])):
                    c = a.as_const(x)
                    if c is not None:
                        self.sym_store_windows.append(
                            (c, index_range_width(a, y), site))
                        break
            return
        base = a.as_const(addr)
        if self.readonly(base, nbytes):
            a.readonly_writes.append((base, base + nbytes))
        cval = a.as_const(value)
        for i in range(nbytes):
            self.byte_versions[base + i] = self.mem_version
            if cval is not None:
                self.mem[base + i] = (cval >> (8 * i)) & 0xFF
            else:
                self.mem[base + i] = (value, i)

    # ------------------------------------------- the escape assumption, made conditional

    def in_stack_window(self, addr: int) -> bool:
        """Is this address inside the machine's own stack allocation?"""
        w = self.stack_window
        return w is not None and w[0] <= addr < w[1]

    def _byte_value(self, cur) -> int:
        """The byte a `mem` entry holds, as a value node."""
        if isinstance(cur, int):
            return self.a.const(cur & 0xFF, W8)
        src, index = cur
        return self._byte_of(src, index)

    def _guard_byte(self, byte_addr: int, cur, addr: int, value: int,
                    nbytes: int) -> tuple[int, int]:
        """One byte, conditioned on where the store went. See `_guard_base`.

        Little-endian and byte-granular: the byte at `addr + k` is bits `8k..8k+7` of `value`, so
        for byte `b` the store writes it exactly when `addr == b - k`, and the expression says so.
        """
        a = self.a
        out = cur
        for k in range(nbytes):
            cond = a.bin("eq", addr, a.const((byte_addr - k) & mask(W64), W64), W64)
            out = a.select(cond, self._byte_of(value, k), out, W8)
        return (out, 0)

    def _guard_base(self, byte_addr: int, base: int) -> int | None:
        """Apply every pending guest-pointer store's condition to one byte's value.

        **The read side is where the conditions belong, and putting them here is what makes a
        cached byte and an image-initial byte behave the same.** The first version wrote the guard
        into `mem` at store time, so only bytes that were *already cached* received a condition --
        and the review reproduced the consequence: a store through `rcx` followed by a read of an
        address that still held its image value returned the image value even when `rcx` provably
        equalled that address. Whether an address had ever been cached decided the answer, which is
        not a property of the program.

        Two bytes are deliberately skipped, and both are stated rather than implied:

        * a byte inside the machine's stack window, because that is what the escape assumption in
          `_isolated_store` spares -- it is applied identically on both sides now;
        * a read-only byte, because a store cannot have written a mapped image section, which is the
          same rule `Machine.store` and `load_concrete` already use for clobbered ranges. Without it
          every constant read out of the image would become a `select` over a guest pointer, and the
          engine would lose far more than it gained.

        Returns None when nothing may alias, which is the common case and keeps the fast path intact.
        """
        if not self.pending_isolated or self.in_stack_window(byte_addr):
            return None
        if self.readonly(byte_addr, 1):
            return None
        a = self.a
        out: int | None = None
        written_at = self.byte_versions.get(byte_addr, 0)
        for s_addr, s_nbytes, s_value, version in self.pending_isolated:
            if version <= written_at:
                continue
            if not may_alias(a, s_addr, s_nbytes, a.const(byte_addr, W64), 1):
                continue
            for k in range(s_nbytes):
                cond = a.bin("eq", s_addr, a.const((byte_addr - k) & mask(W64), W64), W64)
                old = out if out is not None else base
                out = a.select(cond, self._byte_of(s_value, k), old, W8)
        if out is not None:
            self.guards_applied += 1
        return out

    def _isolated_store(self, addr, value: int, nbytes: int, site: int,
                        lo: int, hi: int) -> None:
        """A store through a guest pointer: the frame is spared, everything else is conditioned.

        Two rules live here and only the first is an assumption.

        **The escape assumption.** A guest pointer cannot name the VM's own stack, or the VM would
        corrupt its own saved registers, so such a store does not invalidate the window. Without it
        `0x140155dc1` reproduces 535 of 769 reference transitions, because stores built on `in.rcx`
        delete 980 frame bytes and the VM's saved RSP becomes unknown (`probe/aliasgap.py`). Every
        spared byte is recorded in `isolation_conditions` and can be withdrawn by
        `refute_isolation`.

        **Everything else is conditioned, not dropped and not written now.** The store is recorded as
        a pending write and the condition is applied when a byte is *read* (`_guard_base`), so bytes
        that are cached and bytes that still hold their image value go through one mechanism. The
        first version wrote guards into `mem` only, which made the answer depend on whether an address
        had been cached -- the asymmetry the review reproduced with an image-initial address.
        """
        a = self.a
        spared = [b for b in self.mem
                  if lo <= b < hi and not self.readonly(b, 1) and self.in_stack_window(b)]
        if spared:
            self.isolation_conditions.append((site, nbytes, len(spared)))
            self.isolation_spared.update(spared)
        # The pending list is what the read path consults, and it is bounded. Real roots need three
        # entries; past the cap a store falls back to deleting the bytes it may reach, which is
        # sound and is counted, rather than growing the expression without limit.
        self.pending_isolated.append((addr, nbytes, value, self.mem_version))
        if len(self.pending_isolated) > MAX_PENDING_ISOLATED:
            dropped = self.pending_isolated.pop(0)
            self.guards_dropped += 1
            d_addr, d_nbytes, _v, d_version = dropped
            d_lo, d_hi = address_bounds(a, d_addr, d_nbytes)
            # Losing a write must also invalidate untouched writable image bytes.
            # Newer explicit values remain valid; the private-window exemption
            # is retained under the same allocation assumption as the write.
            for lo, hi in self._outside_window(d_lo, d_hi, count_call=False):
                self.clobbered_ranges = add_clobber(self.clobbered_ranges, lo, hi)
                for byte_addr in list(self.mem):
                    if (lo <= byte_addr < hi and not self.readonly(byte_addr, 1)
                            and self.byte_versions.get(byte_addr, 0) <= d_version):
                        del self.mem[byte_addr]
        self.symbolic_mem[(addr, nbytes)] = a.trunc(value, nbytes * 8)

    def refute_isolation(self, addr_value: int, site: int) -> bool:
        """Withdraw the exemption when a guest-pointer store is *proven* to name the VM's stack.

        The escape assumption is a condition on every derivation that used it, and this is where it
        meets evidence: `finite_addresses` resolving a guest-pointer store to concrete targets is
        exactly the case that can decide it. Withdrawal is deliberately blunt -- the spared bytes
        become unknown rather than unknown-for-this-store -- because the derivations that lean on
        them are not tracked finely enough to repair locally yet. A coarse, honest withdrawal beats
        a precise one that is claimed and not implemented.
        """
        if not self.in_stack_window(addr_value):
            return False
        self.isolation_refuted += 1
        self.isolation_log.append((site, addr_value))
        for byte_addr in list(self.isolation_spared):
            self.mem.pop(byte_addr, None)
        self.isolation_spared.clear()
        parent = self.mem_version
        self.mem_version = self.a.new_memory_version()
        self.a.memory_versions[self.mem_version] = (
            parent, self.a.const(addr_value, W64), None, 0)
        return True

    def isolation_report(self) -> dict:
        """Coverage, unproved conditions and refutations -- the three numbers direction 1 needs."""
        return {
            "conditions": len(self.isolation_conditions),
            "spared_bytes": len(self.isolation_spared),
            "refuted": self.isolation_refuted,
            "guards_built": self.guards_built,
            "guards_applied": self.guards_applied,
            "guards_dropped": self.guards_dropped,
            "pending_isolated": len(self.pending_isolated),
            "calls_checked": self.call_memory_checked,
            "call_writes_forgotten": self.call_memory_forgotten,
            "calls_into_window": self.calls_into_window,
            "wide_forgets": self.wide_forgets,
            "widest_forget": self.widest_forget,
            "widest_forget_at": self.widest_forget_addr,
        }

    def call_write_targets(self, args) -> dict[str, int]:
        """The concrete in-window argument pointers, by register, read **before** the clobber.

        A native callee is given its arguments and nothing else that names this machine's stack, so a
        pointer argument landing in the window is the whole of the reachability question. Measured:
        15 of 26 concrete argument values across the corpus are window pointers, including
        `ADVAPI32.dll!InitializeSecurityDescriptor(rcx=0x7ffffffefce8)`.

        **Where it writes is a separate question, and this used to answer it wrongly.** The region
        was `[ptr, window_top)`, which is aimed *upwards*; `probe/callcost.py` shows the intra-image
        callees on `0x1401687d5` writing 773 stores between `0x7ffffffef2d8` and `0x7ffffffef358`,
        every one of them **below** the pointer argument, while the bytes the rule deleted
        (`0x7ffffffefd50` upward) are the VM's own bookkeeping that the recovery needs. The region is
        now computed in `forget_call_writes` from what the callee was observed to write and, for
        named imports, from `core/callsum.py`.
        """
        w = self.stack_window
        if w is None or not self.call_memory_model:
            return {}
        out: dict[str, int] = {}
        for name, value in args:
            c = self.a.as_const(value)
            if c is not None and w[0] <= c < w[1]:
                out[name] = c
        return out

    def forget_call_writes(self, site: int, pointers, observed=None,
                           described=None) -> None:
        """Forget the memory a native callee may have written, from evidence where there is any.

        `pointers` is `call_write_targets`; `observed` is the `(addr, nbytes)` store list from the
        summarizer, including a failed attempt; `described` is the region list from the import table.
        The rule, in the order it prefers knowledge:

        1. **`described`** -- a named import whose signature says what it writes. An empty list means
           the signature says it writes nothing, which is an answer and not a gap.
        2. **`observed`** -- what the partial interpretation of an intra-image callee actually wrote.
           Unexplored paths may write more, so the span is extended to the pointer argument and no
           further: every observed store on this corpus is below it.
        3. **Neither** -- one page below the pointer, which is the stack direction and is labelled a
           fallback rather than knowledge.
        """
        if not self.call_memory_model:
            return
        self.call_memory_checked += 1
        if described is not None:
            for lo, hi in described:
                if hi > lo:
                    self.call_memory_forgotten += 1
                    self.forget(self.a.const(lo, W64), hi - lo,
                                bump=self.call_bump_version)
            return
        a = self.a
        observed_ranges = []
        for addr_node, nbytes in (observed or ()):
            c = a.as_const(addr_node)
            if c is not None:
                observed_ranges.append((c, c + nbytes))
        for ptr in pointers.values():
            if observed_ranges:
                lo = min(r[0] for r in observed_ranges)
                hi = max(max(r[1] for r in observed_ranges), ptr)
                ranges = [(min(lo, ptr), hi)]
            else:
                ranges = [(ptr - UNKNOWN_WORK_BELOW, ptr)]
            for lo, hi in ranges:
                for piece_lo, piece_hi in self._outside_window(lo, hi):
                    self.call_memory_forgotten += 1
                    self.forget(a.const(piece_lo, W64), piece_hi - piece_lo,
                                bump=self.call_bump_version)

    def _outside_window(self, lo: int, hi: int, count_call: bool = True) -> list[tuple[int, int]]:
        """`[lo, hi)` minus the machine's stack window, and a count when the window was in it.

        **An intra-image callee whose stores land in the VM's frame is not a native leaf.** The
        callee on `0x1401687d5` performs 770 stores between `0x7ffffffef2d8` and `0x7ffffffef358` --
        across the guest register image -- and then stops at a `symbolic indirect transfer`, which is
        the signature of *handler* code, not of a library function that returns. Forgetting the
        window for such a call destroys the only copy of the VM's state the engine has, and the
        root loses two thirds of its reference transitions. The window is therefore kept, the rest of
        the observed region is forgotten, and `calls_into_window` counts how often this happens -- so
        a rule that fires on every call can be told from one that fires on a mis-classified transfer.
        """
        w = self.stack_window
        if w is None or hi <= w[0] or lo >= w[1]:
            return [(lo, hi)]
        if count_call:
            self.calls_into_window += 1
        out = []
        if lo < w[0]:
            out.append((lo, w[0]))
        if hi > w[1]:
            out.append((w[1], hi))
        return out

    def forget(self, addr: int, nbytes: int, bump: bool = True) -> None:
        a = self.a
        if bump:
            parent = self.mem_version
            self.mem_version = a.new_memory_version()
            a.memory_versions[self.mem_version] = (parent, addr, None, nbytes)
        self.symbolic_mem = {
            (p, size): value for (p, size), value in self.symbolic_mem.items()
            if not may_alias(a, addr, nbytes, p, size)
        }
        lo, hi = address_bounds(a, addr, nbytes)
        self.clobbered_ranges = add_clobber(self.clobbered_ranges, lo, hi)
        # **Cost is O(bytes actually modelled), not O(nbytes).**
        #
        # This used to erase `nbytes` addresses one at a time, and the range is a *bounding box*: a
        # summarised callee on `ACE-CORE.sys3` root `0x145bbb46a` produced `nbytes = 140 732 025 750
        # 648` (about 128 TiB) because `forget_call_writes` bounds its region by the minimum and
        # maximum of an observed store address (an image address near `0x1459...`) and a pointer
        # argument (a guest address near `0x7fff...`). The bounding box is *sound* -- forgetting
        # more than the callee wrote only costs knowledge -- but erasing it byte by byte meant
        # 1.4e14 Python iterations, so the whole corpus run stopped advancing: 2h05m at 99 % of a
        # core, with no output and nothing to show it was not merely slow.
        #
        # Filtering the *existing* keys gives the identical result and costs one pass over the
        # modelled memory, which is bounded by the observed stores.
        mem = self.mem
        if a.is_const(addr):
            base = a.as_const(addr)
            # A tight range is cheaper to walk than the dict when the dict is the bigger of the two.
            if nbytes <= len(mem):
                for i in range(nbytes):
                    mem.pop(base + i, None)
            else:
                for byte_addr in [k for k in mem if base <= k < base + nbytes]:
                    del mem[byte_addr]
        else:
            for byte_addr in [k for k in mem if lo <= k < hi]:
                del mem[byte_addr]
        if nbytes > self.WIDE_FORGET:
            # Recorded rather than forbidden: the width is a property of the model's bounding box and
            # a reader needs to be able to see that a root's memory knowledge was given up wholesale.
            self.wide_forgets += 1
            if nbytes > self.widest_forget:
                self.widest_forget = nbytes
                self.widest_forget_addr = a.as_const(addr)

    def historical_constant(self, addr: int, nbytes: int, version: int, image_u8):
        """Read a past state from its persistent history, never today's overlay.

        Only constants are returned to the folding hook. A possibly aliasing
        symbolic write or an unknown initial byte is an explicit refusal.
        """
        a = self.a
        if version < 0:
            return None
        if self.readonly(addr, nbytes) and not any(
                lo < addr + nbytes and addr < hi for lo, hi in a.readonly_writes):
            value = self.read_image(addr, nbytes, image_u8)
            return None if value is None else a.const(value, nbytes * 8)
        key = (addr, nbytes, version, image_u8)
        if key in self._historical_cache:
            return self._historical_cache[key]
        result = self._historical_constant(addr, nbytes, version, image_u8)
        self._historical_cache[key] = result
        return result

    def _historical_constant(self, addr, nbytes, version, image_u8):
        a = self.a
        pieces = {}
        while version:
            record = a.memory_versions.get(version)
            if record is None:
                return None
            version, target, value, size = record
            target_c = a.as_const(target)
            if target_c is None:
                if self.readonly(addr, nbytes):
                    continue
                split = finite_addresses(a, target)
                if split is not None:
                    _, zero, one = split
                    if not any(i not in pieces and
                               (zero <= addr + i < zero + size or one <= addr + i < one + size)
                               for i in range(nbytes)):
                        continue
                lo, hi = address_bounds(a, target, size)
                if any(i not in pieces and lo <= addr + i < hi for i in range(nbytes)):
                    return None
                continue
            for i in range(nbytes):
                offset = addr + i - target_c
                if i in pieces or not 0 <= offset < size:
                    continue
                if value is None:
                    return None
                byte = a.as_const(self._byte_of(value, offset))
                if byte is None:
                    return None
                pieces[i] = byte
            if len(pieces) == nbytes:
                break
        for i in range(nbytes):
            if i not in pieces:
                byte = image_u8(addr + i)
                if byte is None:
                    return None
                pieces[i] = byte
        return a.const(sum(pieces[i] << (8 * i) for i in range(nbytes)), nbytes * 8)

    def possible_writers(self, addr: int) -> list[tuple[int, int]]:
        """Dropped stores whose unresolved address could have landed on `addr`.

        Returns `(site, span)` pairs for every symbolic-address store whose base and
        index width make `addr` reachable. The span is reported because it is what
        the claim rests on: a store indexed by a 16-bit value covers 64 KiB and says
        something useful, while one indexed by a full 64-bit expression covers the
        address space and a match means nothing. Callers should treat a wide span as
        uninformative rather than as evidence.
        """
        out: list[tuple[int, int]] = []
        for base, width, site in self.sym_store_windows:
            span = 1 << min(width, 32)
            if base <= addr < base + span:
                out.append((site, span))
        return out

    # ------------------------------------------------------------ diagnostics

    def symbolic_roots(self) -> list[int]:
        out = list(self.regs.values()) + list(self.flags.values()) + list(self.xmm.values())
        for v in self.mem.values():
            if not isinstance(v, int):
                out.append(v[0])
        return out


def condition(a: Arena, m: Machine, cc: str) -> int:
    """The x86 condition code as a 0/1 expression over the modelled flags.

    Kept in one place because `jcc`, `setcc` and `cmovcc` must agree bit for bit;
    a divergence between them shows up as an unexplained dispatch split.
    """
    zf, sf, of, cf, pf = (m.flag("zf"), m.flag("sf"), m.flag("of"), m.flag("cf"), m.flag("pf"))
    one = a.const(1, W8)
    n = lambda x: a.bin("xor", x, one, W8)  # noqa: E731
    so = a.bin("xor", sf, of, W8)
    if cc == "o":
        return of
    if cc == "no":
        return n(of)
    if cc == "b":
        return cf
    if cc == "ae":
        return n(cf)
    if cc == "e":
        return zf
    if cc == "ne":
        return n(zf)
    if cc == "be":
        return a.bin("or", cf, zf, W8)
    if cc == "a":
        return n(a.bin("or", cf, zf, W8))
    if cc == "s":
        return sf
    if cc == "ns":
        return n(sf)
    if cc == "p":
        return pf
    if cc == "np":
        return n(pf)
    if cc == "l":
        return so
    if cc == "ge":
        return n(so)
    if cc == "le":
        return a.bin("or", zf, so, W8)
    if cc == "g":
        return n(a.bin("or", zf, so, W8))
    raise KeyError(cc)
