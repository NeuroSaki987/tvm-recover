"""A repeated trace is insufficient: require an invariant and a ranking proof."""
import copy
import unittest
from unittest.mock import patch

from core.gueststate import Program
from core.xstate import REGS


def fixture():
    programs, rows = {}, []
    for index in range(2):
        key = str(index)
        p = Program(entry=key)
        params = {r: p.add('param', 64, block=key, reg=r) for r in REGS}
        mem = p.add('memparam', 0, block=key)
        one = p.add('c', 64, value=1)
        bound = p.add('c', 64, value=10)
        counter = p.add('bin', 64, (params['r10'], one), operator='add')
        accepted = p.add('bin', 8, (counter, bound), operator='ule')
        sp = p.add('c', 64, value=0x8000)
        p.blocks[key] = dict(params=params, memory_in=mem, memory_out=mem,
                             outputs=dict(params, r10=counter),
                             guards=[[accepted, p.add('c', 8, value=1)], [params['rsp'], sp]],
                             return_outputs={})
        programs[key] = p
        rows.append(dict(ok=True, entry_ip='0x1000', next_ip=0x1000,
                         start=100 + index * 4, end=104 + index * 4,
                         native_entry_constants=dict(r10=2 + index, rsp=0x8000),
                         relations={'r10': {'derived_update': dict(kind='modular_add', bits=64, delta=1),
                                             'path_guard_equivalence': dict(maximum_output=10)}}))
    return rows, programs


class LoopProofTests(unittest.TestCase):
    def prove(self, rows=None, programs=None, **kwargs):
        from core.loopproof import prove
        r, p = fixture()
        return prove(rows or r, programs or p, **kwargs)

    def test_inductive_projection_has_verified_rank_and_terminal_boundary(self):
        c = self.prove()
        self.assertTrue(c['loop_proved'], c)
        self.assertEqual('serialized_register_subsystem', c['scope'])
        self.assertFalse(c['full_native_loop_proved'])
        self.assertEqual(8, c['maximum_iterations_from_observed_entry'])
        self.assertTrue(all(x == 'unsat' for x in c['obligations'].values()))

    def test_same_boundary_without_guard_is_not_a_proof(self):
        rows, programs = fixture()
        for p in programs.values():
            p.blocks[p.entry]['guards'] = p.blocks[p.entry]['guards'][1:]
        self.assertFalse(self.prove(rows, programs)['loop_proved'])

    def test_specialized_register_must_be_preserved(self):
        rows, programs = fixture()
        for p in programs.values():
            p.blocks[p.entry]['outputs']['rsp'] = p.add('c', 64, value=0x9000)
        self.assertFalse(self.prove(rows, programs)['loop_proved'])

    def test_changed_transfer_or_nonloop_boundary_is_rejected(self):
        rows, programs = fixture()
        rows[1]['next_ip'] = 0x2000
        self.assertFalse(self.prove(rows, programs)['loop_proved'])
        rows, programs = fixture()
        p = programs['1']; p.blocks['1']['outputs']['rax'] = p.add('c', 64, value=123)
        self.assertFalse(self.prove(rows, programs)['loop_proved'])

    def test_metadata_bound_and_update_are_only_candidates(self):
        rows, programs = fixture()
        for r in rows:
            r['relations']['r10']['path_guard_equivalence']['maximum_output'] = 20
        self.assertFalse(self.prove(rows, programs)['loop_proved'])
        rows, programs = fixture()
        for p in programs.values():
            p.blocks[p.entry]['outputs']['r10'] = p.blocks[p.entry]['params']['r10']
        self.assertFalse(self.prove(rows, programs)['loop_proved'])

    def test_observed_entry_must_satisfy_inductive_domain(self):
        rows, programs = fixture()
        rows[0]['native_entry_constants']['r10'] = (1 << 64) - 1
        self.assertFalse(self.prove(rows, programs)['loop_proved'])

    def test_solver_unavailable_or_budget_is_not_success(self):
        with patch.dict('sys.modules', {'z3': None}):
            self.assertFalse(self.prove()['loop_proved'])
        self.assertFalse(self.prove(timeout_ms=0)['loop_proved'])
