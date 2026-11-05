"""Stack operations on architectural addresses, whether concrete or symbolic.

Only constant offsets are combined here. An unknown stack address stays unknown;
no scratch allocation or substituted concrete address is introduced.
"""
def offset(lifter, address: int, delta: int) -> int:
    """Add modulo the native address width, combining nested constant offsets."""
    a = lifter.a
    width = getattr(lifter, "bits", 64)
    address = a.trunc(address, width)
    base = address
    displacement = delta
    while True:
        n = a.node(base)
        if n[0] != 'bin' or n[1] not in ('add', 'sub') or a.width(base) != width:
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
    displacement &= (1 << width) - 1
    return base if displacement == 0 else a.bin(
        'add', base, a.const(displacement, width), width)


def adjust(lifter, delta: int) -> int:
    """Adjust the native stack pointer in bytes and return its new expression."""
    width = getattr(lifter, 'bits', 64)
    stack_pointer = getattr(lifter, 'sp', 'rsp')
    address = offset(lifter, lifter.m.reg(stack_pointer, width), delta)
    lifter.m.set_reg(stack_pointer, address, width)
    return address


def push(lifter, value: int, nbytes: int, site: int) -> None:
    """Push an already evaluated source; callers evaluate it before decrementing RSP."""
    address = adjust(lifter, -nbytes)
    lifter.m.store(address, value, nbytes, site)


def pop(lifter, nbytes: int, site: int) -> int:
    """Load at the old stack pointer and increment it before the caller writes."""
    width = getattr(lifter, 'bits', 64)
    address = lifter.m.reg(getattr(lifter, 'sp', 'rsp'), width)
    value = lifter.m.load(address, nbytes, site, lifter.img.image_u8)
    adjust(lifter, nbytes)
    return value
