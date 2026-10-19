"""Build bounded Guest state transfers for adjacent observed-path segments.

    python -m core.segment MODULE --root 0x140157674 --at 1000 --period 226 --out DIR

This does not close a loop or replace whole-function recovery. The exported
transfers retain RSP specialization, path guards and residual native inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .explore import Recoverer, Block
from .gueststate import capture, build
from .peimg import PeImage
from .replay import TraceStep, replay
from .state import LifterCheckpoint
from .xstate import REGS


def closed_subsystem(rows):
    signatures = [row['closure']['signature'] for row in rows]
    return dict(registers=rows[0]['closure']['registers'],
                same_guarded_transfer=bool(signatures[0]) and len(set(signatures)) == 1,
                same_control_boundary=all(row['entry_ip'] == hex(row['next_ip']) == rows[0]['entry_ip']
                                         for row in rows),
                conditional=True, full_guest_state=False,
                proof='structural transfer and guard identity; no loop invariant established')


def derive_update(arena, before, after, observed_before, observed_after):
    """Use an observed delta only as a proposal; require symbolic identity.

    Equality of original concrete samples alone cannot establish a recurrence.
    The returned rule must reconstruct the complete symbolic output node.
    """
    delta = arena.as_const(arena.bin('sub', observed_after, observed_before, 64))
    if delta is None:
        return None
    for width in (8, 16, 32, 64):
        base = arena.trunc(before, width)
        expected = arena.bin('add', base, arena.const(delta, width), width)
        if width < 64:
            expected = arena.zext(expected, 64)
        if expected == after:
            return dict(kind='modular_add', bits=width, delta=delta & ((1 << width) - 1),
                        zero_extend=width < 64, proof='structural_identity', conditional=True)
    return None


def collect(lifter, root, boundaries, limit):
    """Acquire a finite original-state guide; every outer step is bounded."""
    boundaries = sorted(set(boundaries))
    if not boundaries or boundaries[0] < 0 or limit <= 0:
        raise ValueError('nonnegative boundaries and a positive budget are required')
    records, checkpoints, offsets = [], {}, {}
    ip, steps = root, 0
    while True:
        count = len(lifter.dispatches)
        if count in boundaries and count not in checkpoints:
            checkpoints[count] = (ip, LifterCheckpoint(lifter))
            offsets[count] = len(records)
        if count >= boundaries[-1]:
            return dict(ok=count == boundaries[-1], reason='' if count == boundaries[-1] else 'boundary_overshoot',
                        steps=steps, records=records, checkpoints=checkpoints, offsets=offsets)
        if steps >= limit:
            return dict(ok=False, reason='guide_step_budget', steps=steps)
        mark = len(lifter.dispatches)
        kind, payload = lifter.step(ip)
        steps += 1
        if kind != 'next':
            return dict(ok=False, reason='guide_stop:' + payload.kind, site=payload.site,
                        steps=steps, transfers=count)
        if count >= boundaries[0]:
            records.append(TraceStep(ip, payload, tuple(lifter.dispatches[mark:])))
        ip = payload
        # Never claim a checkpoint at a boundary crossed inside one step.
        if any(count < boundary < len(lifter.dispatches) for boundary in boundaries):
            return dict(ok=False, reason='boundary_overshoot', steps=steps)


def lift_segments(img, root, at=1000, period=226, copies=2, steps=400_000):
    if at < 0 or period <= 0 or copies <= 0 or steps <= 0:
        raise ValueError('invalid segment boundaries or budget')
    vm = img.section_by_name('.tvm0')
    if vm is None:
        raise ValueError('module has no .tvm0 section')
    recovery = Recoverer(img, vm.va, vm.end, guest_census=True)
    arena, machine, lifter = recovery.prepare(root)
    boundaries = [at + i * period for i in range(copies + 1)]
    guide = collect(lifter, root, boundaries, steps)
    report = dict(schema=1, root=hex(root), at=at, period=period, copies=copies,
                  steps_budget=steps, guide={k: v for k, v in guide.items()
                                            if k not in ('records', 'checkpoints', 'offsets')},
                  segments=[], complete=False, loop_proved=False,
                  contract='conditional observed-path transfer; no cache ownership or loop invariant proof')
    if not guide['ok']:
        return report, {}
    context = recovery.guest_ctx
    if context is None:
        report['reason'] = 'no measured Guest context'
        return report, {}
    programs, output_sets = {}, []
    for start, end in zip(boundaries, boundaries[1:]):
        entry_ip, checkpoint = guide['checkpoints'][start]
        checkpoint.restore(lifter)
        native_inputs = capture(machine, context, img.image_u8)
        native_outputs = capture(machine, context, img.image_u8,
                                 snapshot=guide['checkpoints'][end][1].machine)
        # RSP remains concrete because the current lifter requires a concrete
        # native stack. Other slots use the same symbolic inputs across cuts.
        for reg in REGS:
            if reg != 'rsp':
                machine.store(arena.const(context.addr_of(reg), 64),
                              arena.opaque('segment.in.' + reg, 64), 8, -3)
        inputs = capture(machine, context, img.image_u8)
        memory_entry = machine.mem_version
        entry_memory_seed = {addr: value for addr, value in machine.mem.items()
                             if isinstance(value, int) and not context.base <= addr < context.base + 136}
        trace_mark = len(lifter.guest_trace)
        effect_mark = len(lifter.local_effects)
        dispatch_mark = len(lifter.dispatches)
        records = guide['records'][guide['offsets'][start]:guide['offsets'][end]]
        result = replay(lifter, entry_ip, records, limit=steps)
        row = dict(start=start, end=end, entry_ip=hex(entry_ip), **asdict(result))
        row.pop('guards')  # The state artifact contains these expressions.
        report['segments'].append(row)
        if not result.ok:
            continue
        outputs = capture(machine, context, img.image_u8)
        label = f'segment:{start}:{end}'
        block = Block(label, kind='segment', guest_entry=inputs, guest_exit=outputs,
                      memory_entry=memory_entry, memory_exit=machine.mem_version,
                      guest_trace=list(lifter.guest_trace[trace_mark:]),
                      transfers=list(lifter.dispatches[dispatch_mark:]), cost=result.steps)
        block.path_constraints = result.guards
        block.local_effects = dict(lifter.local_effects[effect_mark:])
        block.entry_memory_seed = entry_memory_seed
        program = build(SimpleNamespace(entry=label, blocks={label: block}), arena)
        program.assumptions = dict(context_base=context.base, rsp_specialized=True,
                                   private_stack_window=machine.stack_window,
                                   segment_start=start, segment_end=end,
                                   path='every native instruction and transfer matched the observed guide',
                                   cache='only context slots were abstracted; carried native state is residual',
                                   loop_proved=False,
                                   memory='native memory history with existing isolation/call/readonly assumptions')
        row['state'] = program.validation
        row['closure'] = program.closure(label)
        row['memory_closure'] = program.memory_closure(label)
        row['entry_memory_seed_bytes'] = len(entry_memory_seed)
        row['guards'] = len(result.guards)
        row['relations'] = {}
        for reg in REGS:
            delta = arena.as_const(arena.bin('sub', outputs[reg], inputs[reg], 64))
            row['relations'][reg] = dict(unchanged=outputs[reg] == inputs[reg], constant_delta=delta)
            update = derive_update(arena, inputs[reg], outputs[reg], native_inputs[reg], native_outputs[reg])
            if update is not None:
                row['relations'][reg]['derived_update'] = update
                if update['delta'] > 0:
                    from .guarddomain import derive_bound
                    bound = derive_bound(arena, inputs[reg], outputs[reg], result.guards,
                                         arena.as_const(native_inputs[reg]), update['delta'])
                    if bound is not None:
                        row['relations'][reg]['path_guard_equivalence'] = bound
        row['native_entry_constants'] = {reg: arena.as_const(value) for reg, value in native_inputs.items()}
        row['native_exit_constants'] = {reg: arena.as_const(value) for reg, value in native_outputs.items()}
        output_sets.append(outputs)
        programs[label] = program
    report['complete'] = (len(programs) == copies and all(
        not p.validation['verifier_errors'] and not p.validation['roundtrip_errors']
        for p in programs.values()))
    if len(output_sets) == 2:
        report['identical_output_expressions'] = [r for r in REGS if output_sets[0][r] == output_sets[1][r]]
        # Equality here is only about backing expressions under each segment's
        # own guards and residual native state; it is not a recurrence proof.
    if report['complete']:
        report['closed_subsystem'] = closed_subsystem(report['segments'])
        from .loopproof import prove
        report['loop_proof'] = prove(report['segments'], programs)
        report['loop_proved'] = report['loop_proof']['loop_proved']
        report['loop_proof_scope'] = report['loop_proof']['scope']
        report['full_native_loop_proved'] = False
        if report['loop_proved']:
            report['closed_subsystem']['proof'] = report['loop_proof']['proof']
    return report, programs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', type=Path)
    parser.add_argument('--root', type=lambda s: int(s, 0), required=True)
    parser.add_argument('--at', type=int, default=1000)
    parser.add_argument('--period', type=int, default=226)
    parser.add_argument('--copies', type=int, default=2)
    parser.add_argument('--steps', type=int, default=400_000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.at < 0 or min(args.period, args.copies, args.steps) <= 0:
        parser.error('invalid boundaries or budget')
    report, programs = lift_segments(PeImage(str(args.module)), args.root, args.at, args.period,
                                     args.copies, args.steps)
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    report['engine_sha256'] = source_hash.hexdigest()
    report['module_sha256'] = hashlib.sha256(args.module.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    for label, program in programs.items():
        (args.out / (label.replace(':', '-') + '.guest-state.json')).write_text(
            json.dumps(program.to_dict(), indent=2), encoding='utf-8')
    (args.out / 'segments.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
