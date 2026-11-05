"""Target architecture at native machine boundaries; expression storage stays canonical."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    bits: int

    @property
    def pointer_bytes(self):
        return self.bits // 8

    @property
    def mask(self):
        return (1 << self.bits) - 1

    @property
    def stack_register(self):
        return 'rsp' if self.bits == 64 else 'esp'

    @classmethod
    def for_image(cls, image):
        machine = getattr(image, 'machine', None)
        bits = 64 if getattr(image, 'is64', True) else 32
        if machine is not None and machine != (0x8664 if bits == 64 else 0x14c):
            raise ValueError(f'unsupported native architecture {machine:#x} for PE{bits}')
        return cls(bits)


class NativeRegisters32:
    """Explicit native-state projection when no PE32 backing layout was proved.

    This deliberately does not apply the x64 protector's measured stack offsets.
    It describes native registers at CFG cuts, not recovered Guest register homes.
    """
    native_registers = True
    bits = 32
    slot_bytes = 4
    base = 0
    span = (0, 0)

    def addr_of(self, reg):
        return None

    def reg_at(self, address):
        return None
