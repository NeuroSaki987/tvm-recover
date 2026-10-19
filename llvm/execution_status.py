"""Safe analysis ABI outcomes; these statuses never resume normal guest flow.

The top two bits distinguish normal completion (00), a modeled terminal
effect (10), and fuel exhaustion (11). Counts occupy at most 62 bits. The
terminal kind/detail remain in the manifest and interpreter descriptor; the
scalar ABI reports that a terminal occurred without executing a host trap.
"""

MASK64 = (1 << 64) - 1
COUNT_MASK = (1 << 62) - 1
TERMINAL_BIT = 1 << 63
TERMINAL_KINDS = frozenset(('noreturn', 'fastfail'))
ENCODING = '00: normal steps; 10: analysis terminal | steps; 11: ~steps fuel exhaustion'


def decode(raw: int) -> dict:
    raw &= MASK64
    tag = raw >> 62
    if tag == 0:
        status, steps = 'normal', raw
    elif tag == 2:
        status, steps = 'terminal', raw & COUNT_MASK
    elif tag == 3:
        status, steps = 'exhausted', (~raw) & MASK64
    else:
        raise ValueError(f'invalid analysis status {raw:#x}')
    return {'status': status, 'steps': steps, 'finished': status == 'normal',
            'exhausted': status == 'exhausted', 'terminal': status == 'terminal'}


def terminal_descriptor(block) -> dict | None:
    if block.term not in TERMINAL_KINDS:
        return None
    return {'kind': block.term, 'detail': block.terminal_detail, 'block': block.index,
            'calls': [{'name': name, 'site': site} for name, site in block.calls]}
