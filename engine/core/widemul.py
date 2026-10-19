"""Exact high halves of products using the existing <=64-bit expression IR."""


def high_product(arena, x, y, width, signed=False):
    a = arena
    if width not in (8, 16, 32, 64):
        raise ValueError('unsupported multiply width')
    if width < 64:
        wide = width * 2
        xx = a.sext(x, wide) if signed else a.zext(x, wide)
        yy = a.sext(y, wide) if signed else a.zext(y, wide)
        product = a.bin('mul', xx, yy, wide)
        return a.trunc(a.bin('shr', product, a.const(width, wide), wide), width)
    # Base 2^32 schoolbook multiplication. Each intermediate below fits u64,
    # including the cross-term + low carry; no 128-bit Arena width is needed.
    k = a.const(32, 64)
    mask = a.const(0xFFFFFFFF, 64)
    xl, yl = a.bin('and', x, mask, 64), a.bin('and', y, mask, 64)
    xh, yh = a.bin('shr', x, k, 64), a.bin('shr', y, k, 64)
    carry = a.bin('shr', a.bin('mul', xl, yl, 64), k, 64)
    cross = a.bin('add', a.bin('mul', xh, yl, 64), carry, 64)
    cross2 = a.bin('add', a.bin('mul', xl, yh, 64), a.bin('and', cross, mask, 64), 64)
    high = a.bin('add', a.bin('mul', xh, yh, 64),
                 a.bin('add', a.bin('shr', cross, k, 64), a.bin('shr', cross2, k, 64), 64), 64)
    if signed:
        sign_x = a.bin('sar', x, a.const(63, 64), 64)
        sign_y = a.bin('sar', y, a.const(63, 64), 64)
        high = a.bin('sub', high, a.bin('and', sign_x, y, 64), 64)
        high = a.bin('sub', high, a.bin('and', sign_y, x, 64), 64)
    return high


def set_multiply_flags(lifter, low, high, width, signed, site):
    a, m = lifter.a, lifter.m
    expected_high = a.bin('sar', low, a.const(width - 1, width), width) if signed else a.const(0, width)
    overflow = a.bin('ne', high, expected_high, width)
    m.set_flag('cf', overflow)
    m.set_flag('of', overflow)
    # Intel SDM MUL/IMUL: these flags are architecturally undefined. Distinct
    # dynamic instructions must not accidentally share the same unknown value.
    for flag in ('sf', 'zf', 'af', 'pf'):
        ref = a.opaque(f'mul_undefined_{flag}#{lifter.steps}', 8)
        m.set_flag(flag, ref)
        lifter.local_effects.append((ref, dict(kind='undefined_flag', flag=flag,
                                              site=site, step=lifter.steps)))


def write_double_result(lifter, low, high, width):
    a, m = lifter.a, lifter.m
    if width == 8:
        product = a.bin('or', a.zext(low, 16),
                        a.bin('shl', a.zext(high, 16), a.const(8, 16), 16), 16)
        m.set_reg('ax', product, 16)
    else:
        m.set_reg('rax', low, width)
        m.set_reg('rdx', high, width)
