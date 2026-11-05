"""Integer string instructions, with explicit refusal of unbounded REP work."""


def execute(lifter, insn, ip, nxt):
    from .lift64 import Stop
    a, m = lifter.a, lifter.m
    words = insn.mnemonic.split()
    name = words[-1]
    family, suffix = name[:-1], name[-1:]
    if family not in ('lods', 'stos', 'movs', 'cmps', 'scas'):
        return None
    # MOVSD/CMPSD also name SSE instructions; only the implicit string forms
    # use opcodes A4..AF, after legacy prefixes.
    raw = list(insn.bytes)
    while raw and raw[0] in (0x26, 0x2e, 0x36, 0x3e, 0x64, 0x65, 0x66, 0x67, 0xf2, 0xf3):
        raw.pop(0)
    if lifter.bits == 64 and raw and 0x40 <= raw[0] <= 0x4f:
        raw.pop(0)
    if not raw or raw[0] not in range(0xa4, 0xb0):
        return None
    width = {'b': 8, 'w': 16, 'd': 32, 'q': 64}.get(suffix)
    if width is None:
        return None
    aw = insn.addr_size * 8
    src, dst, counter = {16: ('si', 'di', 'cx'), 32: ('esi', 'edi', 'ecx'),
                         64: ('rsi', 'rdi', 'rcx')}[aw]
    repeat = 0xf3 if 0xf3 in insn.prefix else 0xf2 if 0xf2 in insn.prefix else 0
    count = a.as_const(m.reg(counter, aw)) if repeat else 1
    if count == 0:
        return 'next', nxt
    direction = a.as_const(m.flag('df'))
    if count is None or count > 65536 or direction is None:
        return 'stop', Stop('unsupported', ip, text='string operation needs bounded count and known DF')
    delta = -(width // 8) if direction else width // 8
    for _ in range(count):
        source = m.reg(src, aw)
        target = m.reg(dst, aw)
        if family in ('lods', 'movs', 'cmps'):
            segment = 'fs' if 0x64 in insn.prefix else 'gs' if 0x65 in insn.prefix else None
            if segment:
                source = a.bin('add', a.opaque(f'{segment}_base', lifter.bits),
                               a.zext(source, lifter.bits), lifter.bits)
            value = lifter.load_at(source, width // 8, ip)
        else:
            value = m.reg('rax', width)
        if family == 'lods':
            m.set_reg('rax', value, width)
        elif family in ('stos', 'movs'):
            m.store(target, value, width // 8, ip)
        else:
            other = lifter.load_at(target, width // 8, ip)
            lifter._sub_flags(value, other, a.bin('sub', value, other, width), width)
        if family in ('lods', 'movs', 'cmps'):
            m.set_reg(src, a.bin('add', m.reg(src, aw), a.const(delta, aw), aw), aw)
        if family in ('stos', 'movs', 'cmps', 'scas'):
            m.set_reg(dst, a.bin('add', target, a.const(delta, aw), aw), aw)
        if repeat:
            remaining = a.bin('sub', m.reg(counter, aw), a.const(1, aw), aw)
            m.set_reg(counter, remaining, aw)
            if family in ('cmps', 'scas') and a.as_const(remaining) != 0:
                zf = a.as_const(m.flag('zf'))
                if zf is None:
                    # Resume this REP instruction through the CFG with decremented count.
                    pred = m.flag('zf')
                    taken, fall = (ip, nxt) if repeat == 0xf3 else (nxt, ip)
                    return 'stop', Stop('native_branch', ip, pred=pred, taken=taken, not_taken=fall)
                if bool(zf) != (repeat == 0xf3):
                    break
    return 'next', nxt
