"""Replay a proved caller prefix to construct a caller-specialized VM entry.

No branch is forced and no missing memory or register is assigned a guessed value.
The recovered function remains conditional on entering the named caller with the
usual symbolic inputs and reaching its named direct call.
"""
from dataclasses import dataclass

from capstone import CS_OP_IMM

from .entries import _anchors
from .expr import Arena
from .lift64 import Lifter
from .mach import Machine
from .stackops import push


class EntryContextError(ValueError):
    def __init__(self, reason: str, site: int, detail: str = ''):
        self.reason, self.site, self.detail = reason, site, detail
        super().__init__(f'{reason} at {site:#x}' + (f': {detail}' if detail else ''))


@dataclass(frozen=True)
class EntryContextSpec:
    root: int
    caller_entry: int
    call_site: int
    max_steps: int = 10_000


@dataclass
class PreparedEntryContext:
    arena: Arena
    machine: Machine
    lifter: Lifter
    entry: int
    contract: dict


@dataclass(frozen=True)
class EntryContextFactory:
    spec: EntryContextSpec

    def __call__(self, img, vm_lo: int, vm_hi: int, stack_base: int) -> PreparedEntryContext:
        spec = self.spec
        if spec.caller_entry not in _anchors(img):
            raise EntryContextError('unvalidated-caller-entry', spec.caller_entry,
                                    'requires a PE entry, unwind function start or export')
        if not vm_lo <= spec.root < vm_hi:
            raise EntryContextError('target-mismatch', spec.root, 'root is outside the VM range')
        a = Arena()
        m = Machine(a, stack_base)
        L = Lifter(img, m, a, vm_lo, vm_hi)
        ip = spec.caller_entry
        prefix = []
        while ip != spec.call_site:
            if len(prefix) >= spec.max_steps:
                raise EntryContextError('prefix-budget', ip, f'{spec.max_steps} instructions')
            event_mark, box_mark = len(L.events), len(L.boxed)
            kind, result = L.step(ip)
            prefix.append(ip)
            if kind != 'next':
                raise EntryContextError('prefix-stop', ip, str(result))
            if any(e.get('kind') == 'call' for e in L.events[event_mark:]):
                raise EntryContextError('opaque-prefix-call', ip,
                                        'callee effects were not fully summarized')
            if len(L.boxed) != box_mark:
                raise EntryContextError('unsupported-prefix-effect', ip,
                                        'boxed instruction cannot establish caller context')
            ip = result

        size, insn, _ = L.decode(ip)
        if (insn is None or insn.mnemonic != 'call' or len(insn.operands) != 1
                or insn.operands[0].type != CS_OP_IMM):
            raise EntryContextError('not-direct-call', ip)
        target = insn.operands[0].imm & ((1 << 64) - 1)
        chain = [target]
        seen = set()
        while target != spec.root:
            if target in seen or len(chain) >= 8:
                raise EntryContextError('target-mismatch', ip, 'cyclic or excessive trampoline chain')
            seen.add(target)
            if not L.executable(target):
                raise EntryContextError('target-mismatch', ip, f'non-executable target {target:#x}')
            _, jump, _ = L.decode(target)
            if (jump is None or jump.mnemonic != 'jmp' or len(jump.operands) != 1
                    or jump.operands[0].type != CS_OP_IMM):
                raise EntryContextError('target-mismatch', ip,
                                        f'{target:#x} is not a direct trampoline to {spec.root:#x}')
            target = jump.operands[0].imm & ((1 << 64) - 1)
            chain.append(target)
        if not L.executable(spec.root):
            raise EntryContextError('target-mismatch', ip, 'root is not executable')

        # Execute the CALL's stack effect rather than the opaque native-call model.
        # The trampoline chain contains only direct JMPs and has no state effect.
        return_address = spec.call_site + size
        push(L, a.const(return_address, 64), 8, spec.call_site)
        contract = {
            'scope': 'caller-specialized',
            'caller_entry': spec.caller_entry,
            'call_site': spec.call_site,
            'target_root': spec.root,
            'target_chain': chain,
            'return_address': return_address,
            'initial_rsp': stack_base,
            'prefix_instructions': prefix,
            'prefix_steps': len(prefix),
            'initial_registers': 'symbolic architectural inputs; no register values supplied',
            'memory': 'caller stores and memory epochs retained; uninitialized reads stay symbolic',
            'precondition': 'execution enters the named caller and follows its deterministic modeled prefix',
            'opaque_prefix_calls': 0,
        }
        # Counters and guest events start at the callee. Every architectural value,
        # overlay, memory epoch, control register and symbolic expression survives.
        child = Lifter(img, m, a, vm_lo, vm_hi)
        return PreparedEntryContext(a, m, child, spec.root, contract)


@dataclass(frozen=True)
class EntryContext:
    """Public recovery hook, replayed afresh for sampling and actual recovery."""
    caller_entry: int
    call_site: int
    target_root: int
    budget: int = 10_000

    def replay(self, img, vm_lo: int, vm_hi: int, stack_base: int):
        prepared = EntryContextFactory(EntryContextSpec(
            self.target_root, self.caller_entry, self.call_site, self.budget))(
                img, vm_lo, vm_hi, stack_base)
        return prepared.arena, prepared.machine, prepared.contract
