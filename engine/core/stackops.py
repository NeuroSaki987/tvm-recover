"""Stack operations on architectural addresses, whether concrete or symbolic.

Only constant offsets are combined here. An unknown stack address stays unknown;
no scratch allocation or substituted concrete address is introduced.
"""
from .expr import W64

MASK64 = (1 << 64) - 1


def offset(lifter, address: int, delta: int) -> int:
    """Add a byte displacement modulo 2**64, combining nested constant offsets."""
    a = lifter.a
    base = address
    displacement = delta
    while True:
        n = a.node(base)
        if n[0] != 'bin' or n[1] not in ('add', 'sub') or a.width(base) != W64:
            break
        right = a.as_const(n[3])
        if right is not None:
            displacement += right if n[1] == 'add' else -right
            base = n[2]
            continue
        left = a.as_const(n[2])
        if n[1] == 'add' and left is not None:
            displacement += left
            base = n[3]
            continue
        break
    displacement &= MASK64
    return base if displacement == 0 else a.bin('add', base, a.const(displacement, W64), W64)


def adjust(lifter, delta: int) -> int:
    """Adjust RSP in bytes and return its new expression."""
    address = offset(lifter, lifter.m.reg('rsp', W64), delta)
    lifter.m.set_reg('rsp', address, W64)
    return address


def push(lifter, value: int, nbytes: int, site: int) -> None:
    """Push an already evaluated source; callers evaluate it before decrementing RSP."""
    address = adjust(lifter, -nbytes)
    lifter.m.store(address, value, nbytes, site)


def pop(lifter, nbytes: int, site: int) -> int:
    """Load at old RSP and increment it before the caller writes the destination."""
    address = lifter.m.reg('rsp', W64)
    value = lifter.m.load(address, nbytes, site, lifter.img.image_u8)
    adjust(lifter, nbytes)
    return value
