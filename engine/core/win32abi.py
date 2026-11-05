"""Win32 native call boundaries; unknown stack cleanup is an explicit hole."""
import re

from capstone import CS_OP_IMM, CS_OP_MEM, CS_OP_REG


# x86 Win32 APIs use stdcall unless their declaration says otherwise.  Keep
# this table deliberately small and evidence-driven: each entry is an API that
# the PE32 cohort actually reached and whose public declaration fixes the
# number of stack arguments.  A Rust-style `extern "C"` or a private export
# must never be inferred from the fact that it lives in a DLL named kernel32.
_STDCALL_API_CONTRACTS = {
    ('kernel32.dll', 'GetLastError'): (
        0,
        'https://learn.microsoft.com/windows/win32/api/errhandlingapi/'
        'nf-errhandlingapi-getlasterror'),
    ('kernel32.dll', 'IsDebuggerPresent'): (
        0,
        'https://learn.microsoft.com/windows/win32/api/debugapi/'
        'nf-debugapi-isdebuggerpresent'),
    ('kernel32.dll', 'SetUnhandledExceptionFilter'): (
        4,
        'https://learn.microsoft.com/windows/win32/api/errhandlingapi/'
        'nf-errhandlingapi-setunhandledexceptionfilter'),
    ('kernel32.dll', 'UnhandledExceptionFilter'): (
        4,
        'https://learn.microsoft.com/windows/win32/api/errhandlingapi/'
        'nf-errhandlingapi-unhandledexceptionfilter'),
    ('kernel32.dll', 'GetCurrentProcess'): (
        0,
        'https://learn.microsoft.com/windows/win32/api/processthreadsapi/'
        'nf-processthreadsapi-getcurrentprocess'),
    ('kernel32.dll', 'TerminateProcess'): (
        8,
        'https://learn.microsoft.com/windows/win32/api/processthreadsapi/'
        'nf-processthreadsapi-terminateprocess'),
    ('kernel32.dll', 'HeapAlloc'): (
        12,
        'https://learn.microsoft.com/windows/win32/api/heapapi/nf-heapapi-heapalloc'),
    ('kernel32.dll', 'MultiByteToWideChar'): (
        24,
        'https://learn.microsoft.com/windows/win32/api/stringapiset/'
        'nf-stringapiset-multibytetowidechar'),
    ('kernelbase.dll', 'HeapAlloc'): (
        12,
        'https://learn.microsoft.com/windows/win32/api/heapapi/nf-heapapi-heapalloc'),
    ('kernelbase.dll', 'MultiByteToWideChar'): (
        24,
        'https://learn.microsoft.com/windows/win32/api/stringapiset/'
        'nf-stringapiset-multibytetowidechar'),
}


def _known_import_cleanup(lifter, target):
    """Return a documented/decorated stack cleanup for an import slot."""
    imported = lifter.imports.get(target)
    if imported is None:
        return None
    dll, symbol = imported
    api_contract = _STDCALL_API_CONTRACTS.get((dll.casefold(), symbol))
    if api_contract is not None:
        return api_contract[0]
    decorated = re.fullmatch(r'(_|@)([^@]+)@(\d+)', symbol)
    if decorated and decorated[1] == '_':
        total = int(decorated[3])
        if total <= 4096 and total % 4 == 0:
            return total
    return None


def _integer(text):
    try:
        return int(text.rstrip('h'), 16) if text.endswith('h') else int(text, 0)
    except ValueError:
        return None


def infer_caller_cleanup(lifter, nxt):
    """Find a straight-line caller-side cleanup after a call.

    This is deliberately narrow evidence: only literal `add/sub esp` and
    `lea esp,[esp+imm]` instructions before a branch, another call, a push or
    a return participate. The result means the caller owns the cleanup, so the
    resumed instruction stream will execute it; the lifter must not apply it a
    second time.
    """
    code = lifter.img.image_bytes(nxt, min(128, max(0, lifter.img.section_for_va(nxt).end - nxt)))
    if not code:
        return None
    delta = 0
    evidence = []
    for count, insn in enumerate(lifter._md.disasm(code, nxt)):
        if count >= 32:
            break
        mnemonic, operands = insn.mnemonic, insn.op_str.replace(' ', '')
        if mnemonic in ('push', 'pushad', 'pushfd', 'call', 'ret', 'retf',
                        'jmp', 'loop', 'loope', 'loopne', 'jecxz', 'jrcxz'):
            break
        if mnemonic.startswith('j') or mnemonic in ('sysenter', 'int', 'int3'):
            break
        if mnemonic == 'mov' and operands.startswith('esp,'):
            break
        if mnemonic in ('add', 'sub') and operands.startswith('esp,'):
            value = _integer(operands.split(',', 1)[1])
            if value is None:
                break
            delta += value if mnemonic == 'add' else -value
            evidence.append((insn.address, mnemonic, value))
        elif mnemonic == 'lea' and operands.startswith('esp,[esp'):
            inc = re.fullmatch(r'esp,\[esp\+([^\]]+)\]', operands)
            dec = re.fullmatch(r'esp,\[esp-([^\]]+)\]', operands)
            value = _integer((inc or dec).group(1)) if (inc or dec) else None
            if value is None:
                break
            delta += value if inc else -value
            evidence.append((insn.address, mnemonic, value if inc else -value))
        elif 'esp' in operands and mnemonic not in ('mov', 'cmp', 'test'):
            # A computed stack pointer has no static cleanup contract.
            break
        if delta:
            if delta < 0 or delta > 4096 or delta % 4:
                return None
            return delta, evidence
    return None


def infer_native_return_cleanup(lifter, entry, max_instructions=4096):
    """Prove one return-cleanup value over a bounded direct native CFG.

    This is deliberately weaker than a function summary: it does not recover a
    return value or side effects. It only visits every reachable direct branch
    and accepts when each terminal is a ``ret`` with the same immediate. Direct
    calls recurse into the same proof. Import-slot calls participate only when
    a documented or decorated stack cleanup is known. Other indirect transfers,
    unreadable instructions and incomplete traversal make the contract unknown.
    An observed caller-side cleanup remains stronger and is preferred before
    this fallback is consulted.
    """
    if lifter.bits != 32 or not lifter.executable(entry):
        return None
    cache = getattr(lifter, 'native_return_contracts', None)
    if cache is None:
        cache = {}
        lifter.native_return_contracts = cache
    if entry in cache:
        # A re-entrant call has no finite proof yet.  ``None`` is a completed
        # refusal; the distinct sentinel prevents an indirect recursion through
        # direct call edges from re-entering this helper forever.
        return None if cache[entry] == 'in_progress' else cache[entry]
    cache[entry] = 'in_progress'
    pending = [(entry, 0, None)]
    visited = set()
    returns = {}
    while pending:
        ip, stack_used, frame_used = pending.pop()
        state = (ip, stack_used, frame_used)
        if state in visited:
            continue
        if len(visited) >= max_instructions:
            cache[entry] = None
            return None
        section = lifter.img.section_for_va(ip)
        if section is None or not section.executable or ip >= section.end:
            cache[entry] = None
            return None
        visited.add(state)
        size, insn, _raw = lifter.decode(ip)
        if not size or insn is None:
            cache[entry] = None
            return None
        mnemonic = insn.mnemonic
        ops = insn.operands
        if stack_used < 0 or stack_used > 4096:
            cache[entry] = None
            return None
        if mnemonic in ('ret', 'repz ret', 'bnd ret', 'retf'):
            if mnemonic == 'retf':
                cache[entry] = None
                return None
            if stack_used:
                cache[entry] = None
                return None
            cleanup = int(insn.op_str, 0) if insn.op_str else 0
            if cleanup < 0 or cleanup > 4096 or cleanup % 4:
                cache[entry] = None
                return None
            returns[ip] = cleanup
            continue
        if mnemonic in ('int', 'int1', 'int3', 'into', 'hlt',
                        'iret', 'iretd', 'sysenter', 'sysexit', 'ud0',
                        'ud1', 'ud2'):
            cache[entry] = None
            return None
        if mnemonic == 'call':
            if len(ops) != 1:
                cache[entry] = None
                return None
            if ops[0].type == CS_OP_IMM:
                target = ops[0].imm & 0xffffffff
                nested = infer_native_return_cleanup(
                    lifter, target, max_instructions=max_instructions)
                if nested is None:
                    cache[entry] = None
                    return None
                cleanup = nested['stack_cleanup']
            elif (ops[0].type == CS_OP_MEM
                  and not ops[0].mem.base and not ops[0].mem.index):
                cleanup = _known_import_cleanup(
                    lifter, ops[0].mem.disp & 0xffffffff)
            else:
                cleanup = None
            if cleanup is None or cleanup > stack_used:
                cache[entry] = None
                return None
            # A completed call includes its return pop.  A callee `ret imm`
            # additionally removes the caller-pushed argument bytes.
            pending.append((ip + size,
                            stack_used - cleanup,
                            frame_used))
            continue
        if mnemonic == 'leave':
            if frame_used is None or frame_used < 4:
                cache[entry] = None
                return None
            pending.append((ip + size, frame_used - 4, None))
            continue
        if mnemonic in ('enter', 'pusha', 'pushad', 'popa', 'popad'):
            cache[entry] = None
            return None
        memory_writers = {
            'adc', 'add', 'and', 'btc', 'btr', 'bts', 'cmpxchg', 'dec',
            'imul', 'inc', 'mov', 'neg', 'not', 'or', 'pop', 'rcl', 'rcr',
            'rol', 'ror', 'sal', 'sar', 'sbb', 'shl', 'shr', 'sub', 'xadd',
            'xchg', 'xor',
        }
        for operand in ops:
            if operand.type != CS_OP_MEM:
                continue
            if lifter._reg_name(insn, operand.mem.base) != 'esp':
                continue
            # `stack_used` is the distance from the current ESP back to the
            # caller-pushed return address. A write there changes the
            # continuation, which a stack-cleanup contract alone cannot model.
            if (mnemonic in memory_writers
                    and (operand.mem.index or operand.mem.disp == stack_used)):
                cache[entry] = None
                return None
        if mnemonic == 'push':
            amount = ops[0].size if len(ops) == 1 and ops[0].size == 2 else 4
            pending.append((ip + size, stack_used + amount, frame_used))
            continue
        if mnemonic == 'pop':
            if stack_used < 4:
                cache[entry] = None
                return None
            pops_ebp = (len(ops) == 1 and ops[0].type == CS_OP_REG
                        and lifter._reg_name(insn, ops[0].reg) == 'ebp')
            pending.append((ip + size, stack_used - 4,
                            None if pops_ebp else frame_used))
            continue
        if mnemonic in ('pushfd', 'pushf'):
            pending.append((ip + size, stack_used + 4, frame_used))
            continue
        if mnemonic in ('popfd', 'popf'):
            if stack_used < 4:
                cache[entry] = None
                return None
            pending.append((ip + size, stack_used - 4, frame_used))
            continue
        if (mnemonic == 'xchg'
                and any(op.type == CS_OP_REG
                        and lifter._reg_name(insn, op.reg) == 'esp'
                        for op in ops)):
            cache[entry] = None
            return None
        if (mnemonic == 'mov' and len(ops) == 2
                and ops[0].type == CS_OP_REG and ops[1].type == CS_OP_REG
                and lifter._reg_name(insn, ops[0].reg) == 'ebp'
                and lifter._reg_name(insn, ops[1].reg) == 'esp'):
            pending.append((ip + size, stack_used, stack_used))
            continue
        if (mnemonic == 'mov' and len(ops) == 2
                and ops[0].type == CS_OP_REG and ops[1].type == CS_OP_REG
                and lifter._reg_name(insn, ops[0].reg) == 'esp'
                and lifter._reg_name(insn, ops[1].reg) == 'ebp'):
            if frame_used is None:
                cache[entry] = None
                return None
            pending.append((ip + size, frame_used, frame_used))
            continue
        writes_active_frame = frame_used is not None and (
            (bool(ops) and ops[0].type == CS_OP_REG
             and lifter._reg_name(insn, ops[0].reg) == 'ebp'
             and mnemonic not in ('cmp', 'test'))
            or (mnemonic == 'xchg'
                and any(op.type == CS_OP_REG
                        and lifter._reg_name(insn, op.reg) == 'ebp'
                        for op in ops)))
        if writes_active_frame:
            cache[entry] = None
            return None
        if (mnemonic in ('add', 'sub') and len(ops) == 2
                and ops[0].type == CS_OP_REG
                and lifter._reg_name(insn, ops[0].reg) == 'esp'
                and ops[1].type == CS_OP_IMM):
            amount = ops[1].imm
            next_stack = stack_used - amount if mnemonic == 'add' else stack_used + amount
            pending.append((ip + size, next_stack, frame_used))
            continue
        if (mnemonic == 'lea' and len(ops) == 2
                and ops[0].type == CS_OP_REG
                and lifter._reg_name(insn, ops[0].reg) == 'esp'
                and ops[1].type == CS_OP_MEM
                and lifter._reg_name(insn, ops[1].mem.base) == 'esp'
                and not ops[1].mem.index):
            pending.append((ip + size, stack_used - ops[1].mem.disp, frame_used))
            continue
        if ops and ops[0].type == CS_OP_REG and \
                lifter._reg_name(insn, ops[0].reg) == 'esp' and \
                mnemonic not in ('cmp', 'test'):
            cache[entry] = None
            return None
        if mnemonic == 'jmp':
            if len(ops) != 1 or ops[0].type != CS_OP_IMM:
                cache[entry] = None
                return None
            pending.append((ops[0].imm & 0xffffffff, stack_used, frame_used))
            continue
        if mnemonic.startswith('j') and mnemonic not in ('jecxz', 'jrcxz'):
            if len(ops) != 1 or ops[0].type != CS_OP_IMM:
                cache[entry] = None
                return None
            pending.extend(((ip + size, stack_used, frame_used),
                            (ops[0].imm & 0xffffffff, stack_used, frame_used)))
            continue
        if mnemonic in ('jecxz', 'jrcxz', 'loop', 'loope', 'loopne'):
            if len(ops) != 1 or ops[0].type != CS_OP_IMM:
                cache[entry] = None
                return None
            pending.extend(((ip + size, stack_used, frame_used),
                            (ops[0].imm & 0xffffffff, stack_used, frame_used)))
            continue
        pending.append((ip + size, stack_used, frame_used))

    values = set(returns.values())
    if len(values) != 1:
        cache[entry] = None
        return None
    result = {
        'stack_cleanup': values.pop(),
        'instructions_visited': len(visited),
        'return_sites': sorted(returns),
        'method': ('bounded direct native CFG; every path preserves the return slot, '
                   'balances its stack, and terminates with one ret immediate'),
    }
    cache[entry] = result
    return result


def call(lifter, insn, ip, nxt, target):
    from .lift64 import Stop
    from .stackops import offset, adjust
    a, m = lifter.a, lifter.m
    lifter.call_count += 1
    serial = lifter.call_count
    imported = lifter.imports.get(target)
    symbol = imported[1] if imported else ''
    decorated = re.fullmatch(r'(_|@)([^@]+)@(\d+)', symbol)
    cleanup = None
    cleanup_evidence = None
    cleanup_mode = None
    api_contract = (_STDCALL_API_CONTRACTS.get((imported[0].casefold(), symbol))
                    if imported else None)
    register_args = []
    if decorated and decorated[1] == '_':
        total = int(decorated[3])
        if total <= 4096 and total % 4 == 0:
            cleanup = total
            cleanup_mode = 'callee'
    if cleanup is None and api_contract is not None:
        cleanup, _documentation = api_contract
        cleanup_mode = 'callee'
    native_return_contract = None
    # Fastcall decoration gives total bytes, but not which parameter types are
    # eligible for ECX/EDX. Do not guess register placement from byte count.
    from .callcontracts import import_noreturn
    terminal = import_noreturn(*imported) if imported else None
    if terminal:
        cleanup = 20 if terminal.symbol == 'KeBugCheckEx' else 4
        cleanup_mode = 'callee'
    # Parameterless timer queries have no caller-visible memory output.
    pure = (imported is not None and imported[0].casefold() in ('kernel32.dll', 'kernelbase.dll')
            and symbol in ('GetTickCount', 'GetTickCount64'))
    if pure:
        cleanup = 0
        cleanup_mode = 'none'
    if cleanup is None:
        inferred = infer_caller_cleanup(lifter, nxt)
        if inferred is not None:
            cleanup = inferred[0]
            cleanup_evidence = inferred[1]
            cleanup_mode = 'caller'
    if cleanup is None and imported is None and target is not None:
        native_return_contract = infer_native_return_cleanup(lifter, target)
        if native_return_contract is not None:
            cleanup = native_return_contract['stack_cleanup']
            cleanup_mode = 'callee'
    args = [(r, m.reg(r, 32)) for r in register_args]
    if cleanup is not None:
        for index in range(cleanup // 4):
            tail_offset = 4 if insn.mnemonic == 'jmp' else 0
            addr = offset(lifter, m.reg('esp', 32), index * 4 + tail_offset)
            args.append((f'arg{index}', lifter.load_at(addr, 4, ip)))
    ret = a.opaque(f'call_ret#{serial}', 32)
    event = dict(kind='call', site=ip, target=target, args=args, ret=ret,
                 import_slot=target if imported else None,
                 abi='win32', stack_cleanup=cleanup,
                 cleanup_mode=cleanup_mode,
                 argument_contract=('decorated export' if decorated else
                                    'documented Win32 stdcall import' if api_contract else
                                    'parameterless export' if pure else
                                    'caller cleanup observed at call continuation'
                                    if cleanup_evidence else
                                    'bounded direct native return contract'
                                    if native_return_contract else 'unknown'))
    if api_contract is not None:
        event['abi_contract'] = {
            'convention': 'stdcall',
            'stack_bytes': api_contract[0],
            'source': api_contract[1],
        }
    if native_return_contract is not None:
        event['native_return_contract'] = {
            **native_return_contract,
            'return_sites': [
                {'site': site}
                for site in native_return_contract['return_sites']],
        }
    if cleanup_evidence:
        event['cleanup_evidence'] = [
            {'site': site, 'instruction': mnemonic, 'delta': value}
            for site, mnemonic, value in cleanup_evidence]
    lifter.events.append(event)
    if terminal:
        event['noreturn'] = True
        event['noreturn_contract'] = terminal.name
        return 'stop', Stop('noreturn', ip, text=f'{terminal.name}: {terminal.reason}',
                            info={'event': event, 'callee': terminal.name,
                                  'reason': terminal.reason, 'source': terminal.source})
    if cleanup is None:
        return 'stop', Stop('native_call_abi', ip,
                            text='Win32 argument count/stack cleanup not established', info={'event': event})
    # Conservatively forget all writable memory when the signature says nothing
    # about writes. Never apply Win64 structure sizes to Win32 pointers.
    if not pure:
        m.forget(a.const(0, 32), 1 << 32)
    for reg in ('eax', 'ecx', 'edx'):
        m.set_reg(reg, a.opaque(f'call{reg}#{serial}', 32), 32)
    for index in range(8):
        for half in range(2):
            m.xmm[(f'xmm{index}', half)] = a.opaque(f'callxmm{index}.{half}#{serial}', 64)
    for flag in ('cf', 'pf', 'af', 'zf', 'sf', 'of'):
        m.set_flag(flag, a.opaque(f'flag_{flag}#call{serial}', 8))
    m.set_reg('eax', ret, 32)
    if symbol == 'GetTickCount64':
        m.set_reg('edx', a.opaque(f'call_ret_high#{serial}', 32), 32)
    if cleanup_mode == 'callee':
        adjust(lifter, cleanup)
    return 'next', nxt
