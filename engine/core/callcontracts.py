"""Documented non-returning imports, matched by DLL and exact export name.

An unknown call keeps its normal conservative continuation. A nearby trap byte
or a similar-looking symbol is never evidence that the call cannot return.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NoReturnContract:
    module: str
    symbol: str
    reason: str
    source: str

    @property
    def name(self) -> str:
        return f'{self.module}!{self.symbol}'


_NORETURN_IMPORTS = {
    ('ntoskrnl.exe', symbol): NoReturnContract(
        'ntoskrnl.exe', symbol, 'Windows kernel bugcheck terminates execution',
        'https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/' + page)
    for symbol, page in (
        ('KeBugCheck', 'ntddk/nf-ntddk-kebugcheck'),
        ('KeBugCheckEx', 'wdm/nf-wdm-kebugcheckex'),
    )
}


def import_noreturn(module: str, symbol: str) -> NoReturnContract | None:
    """DLL names ignore case; PE export names are matched exactly."""
    return _NORETURN_IMPORTS.get((module.casefold(), symbol))
