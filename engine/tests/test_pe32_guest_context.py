from types import SimpleNamespace

from core import guestctx
from core.expr import Arena
from core.guestctx import GuestCache, HANDLER_TIME
from core.gueststate import capture
from core.lift64 import Lifter
from core.mach import Machine


def setup(code, stack=0x10000):
    image = SimpleNamespace(
        is64=False,
        imports={},
        image_u8=lambda addr: None,
        image_bytes=lambda addr, size: code[addr - 0x1000:addr - 0x1000 + size],
        foldable=lambda addr, size: False,
        section_for_va=lambda addr: SimpleNamespace(
            executable=True, writable=True, end=0x2000),
        is_loader_bound=lambda addr, size: False,
    )
    arena = Arena()
    machine = Machine(arena, stack, bits=32)
    lifter = Lifter(image, machine, arena, 0x1000, 0x2000)
    return arena, machine, lifter


def test_pe32_memory_context_uses_four_byte_architectural_slots():
    ctx = guestctx.GuestContext(0x8000, bits=32)

    assert ctx.slot_bytes == 4
    assert ctx.addr_of("rax") == 0x8000
    assert ctx.addr_of("rsp") == 0x8010
    assert ctx.addr_of("rflags") == 0x8020
    assert ctx.addr_of("r8") is None
    assert ctx.reg_at(0x8002) is None
    assert ctx.reg_at(0x8004) == "rbx"
    assert ctx.span == (0x8000, 0x8024)


def test_pe32_memory_capture_zero_extends_guest_registers_and_flags():
    a, m, _ = setup(b"\xc3")
    ctx = guestctx.GuestContext(0x8000, bits=32)
    values = {
        "rax": 0xffffffff,
        "rbx": 0x11223344,
        "rsp": 0x10000,
        "rflags": 0x246,
    }
    for reg, value in values.items():
        m.store(a.const(ctx.addr_of(reg), 32), a.const(value, 32), 4, 0)

    captured = capture(m, ctx, lambda _addr: None)

    assert a.as_const(captured["rax"]) == 0xffffffff
    assert a.as_const(captured["rbx"]) == 0x11223344
    assert a.as_const(captured["rsp"]) == 0x10000
    assert a.as_const(captured["rflags"]) == 0x246
    assert a.as_const(captured["r8"]) == 0


def test_pe32_register_context_observes_native_x86_homes():
    a, m, _ = setup(b"\xc3")
    ctx = guestctx.RegisterGuestContext32()
    m.set_reg("eax", a.const(0xffffffff, 32), 32)
    m.set_reg("ebx", a.const(0x11223344, 32), 32)
    m.set_reg("esp", a.const(0x10000, 32), 32)

    captured = capture(m, ctx, lambda _addr: None)

    assert ctx.host_of("rax") == "eax"
    assert a.as_const(captured["rax"]) == 0xffffffff
    assert a.as_const(captured["rbx"]) == 0x11223344
    assert a.as_const(captured["rsp"]) == 0x10000
    assert a.as_const(captured["r8"]) == 0


def test_pe32_register_context_does_not_turn_an_unmapped_guest_home_into_zero():
    a, m, _ = setup(b"\xc3")
    ctx = guestctx.RegisterGuestContext32({"rax": "eax"})
    m.set_reg("eax", a.const(0x11223344, 32), 32)

    captured = capture(m, ctx, lambda _addr: None)

    assert a.as_const(captured["rax"]) == 0x11223344
    assert a.as_const(captured["rbx"]) is None
    assert a.op(captured["rbx"]) == "zext"
    assert a.op(a.node(captured["rbx"])[1]) == "opaque"


def test_register_cache_does_not_pick_between_two_hosts_for_one_guest():
    cache = GuestCache(HANDLER_TIME)
    cache.note("ecx", "rbp")
    cache.note("ebp", "rbp")

    assert cache.host_of("rbp") is None


def test_pe32_register_census_attributes_32_bit_guest_updates():
    a, m, lifter = setup(bytes.fromhex("40 01 c3 c3"))
    ctx = guestctx.RegisterGuestContext32()
    m.set_reg("eax", a.init_reg("rax"), 32)
    m.set_reg("ebx", a.init_reg("rbx"), 32)
    lifter.enable_register_guest_census(ctx)

    assert lifter.step(0x1000) == ("next", 0x1001)
    assert lifter.step(0x1001) == ("next", 0x1003)

    assert ("rax", "rax") in {
        (guest, host) for _step, guest, host, _value
        in lifter.guest_reg_writes
    }
    assert ("rbx", "rbx") in {
        (guest, host) for _step, guest, host, _value
        in lifter.guest_reg_writes
    }


def test_pe32_recovery_uses_a_measured_register_backed_cache():
    from core.explore import Recoverer
    from core.mach import canon

    _a, _m, lifter = setup(bytes.fromhex("40 01 c3 c3"))
    ctx = guestctx.RegisterGuestContext32()
    cache = GuestCache(HANDLER_TIME)
    cache.ctx = ctx
    cache.boundaries = 1
    cache.walk_status["confirmed"] = True
    for guest in ctx.regs:
        cache.note(canon(ctx.host_of(guest))[0], guest)

    recoverer = Recoverer(
        lifter.img, 0x1000, 0x2000, step_budget=20, guest_census=True)
    recoverer.handler_cache = cache
    cfg = recoverer.recover(0x1000)

    assert recoverer.guest_ctx.register_backed
    assert any(block.guest_reg_writes for block in cfg.blocks.values())


def test_pe32_derive_cache_seeds_native_registers_without_a_spill_layout():
    from core.explore import Recoverer

    _a, _m, lifter = setup(bytes.fromhex("40 01 c3 c3"))
    recoverer = Recoverer(lifter.img, 0x1000, 0x2000)

    cache = recoverer.derive_guest_cache(0x1000, handlers=2, steps=20)

    assert cache.ctx.register_backed
    assert cache.walk_status["seeded"] is True
    assert "native x86" in cache.walk_status["scope"]


def test_pe32_cache_needs_a_repeated_boundary_before_confirmation():
    from core.explore import Recoverer

    _a, machine, lifter = setup(bytes.fromhex("c3"))
    recoverer = Recoverer(lifter.img, 0x1000, 0x2000)

    def step(_ip):
        lifter.dispatches.append((0x1000, 0x1000))
        return "stop", SimpleNamespace(kind="native_branch")

    lifter.step = step
    recoverer._lifter = lambda _arena, _machine: lifter
    cache = recoverer.derive_guest_cache(0x1000, handlers=2, steps=2)

    assert cache.walk_status["boundaries"] == 1
    assert cache.walk_status["candidate"] is True
    assert cache.walk_status["confirmed"] is False


def test_pe32_register_backing_is_not_labelled_native_only():
    from core.devirt import devirtualize

    _a, _m, lifter = setup(bytes.fromhex("b8 00 10 00 00 ff e0"))
    vm = SimpleNamespace(va=0x1000, end=0x1007)

    result = devirtualize(
        lifter.img, vm, 0x1000, handlers=2, steps=8, blocks=1)

    assert result.ctx.register_backed
    assert result.recovery["state_projection"] == "guest_backing"
    assert "NATIVE_STATE_ONLY" not in result.summary


def test_pe32_unproved_context_is_explicitly_native_only():
    from core.devirt import devirtualize

    _a, _m, lifter = setup(bytes.fromhex("90 c3"))
    vm = SimpleNamespace(va=0x1000, end=0x1002)

    result = devirtualize(
        lifter.img, vm, 0x1000, handlers=1, steps=8, blocks=1)

    assert result.ctx.native_registers
    assert result.recovery["state_projection"] == "native_i386"
    assert "NATIVE_STATE_ONLY" in result.summary
