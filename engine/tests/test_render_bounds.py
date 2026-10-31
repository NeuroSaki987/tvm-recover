"""Diagnostic text must stay bounded even when a small DAG expands exponentially."""
from types import SimpleNamespace

from core.devirt import branch_condition
from core.expr import Arena, render


def shared_expression(depth):
    arena = Arena()
    ref = arena.opaque('guest_x', 64)
    for _ in range(depth):
        ref = arena.bin('mul', ref, ref, 64)
    return arena, ref


def test_shared_dag_does_not_expand_into_megabytes():
    arena, ref = shared_expression(20)
    before = list(arena.nodes)
    text = render(arena, ref)
    assert len(text) <= 4096
    assert '...' in text
    assert arena.nodes == before


def test_node_limit_bounds_visits_not_just_memo_entries():
    arena, ref = shared_expression(16)

    class CountingNodes:
        reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return original[index]

    original = arena.nodes
    counter = CountingNodes()
    arena.nodes = counter
    text = render(arena, ref, limit=8)
    assert counter.reads <= 8
    assert '...' in text


def test_long_leaf_is_capped_before_assembly():
    arena = Arena()
    ref = arena.opaque('x' * 100_000, 64)
    text = render(arena, ref)
    assert len(text) <= 4096
    assert text.endswith('...')


def test_deep_expression_has_explicit_omission():
    arena, ref = shared_expression(1000)
    text = render(arena, ref)
    assert len(text) <= 4096
    assert '...' in text


def test_small_expression_spelling_is_unchanged():
    arena = Arena()
    x, y = arena.opaque('x', 64), arena.opaque('y', 64)
    ref = arena._add(('bin', 'add', x, y, 64))
    assert render(arena, ref) == '(x add y)'


def test_character_budget_marks_omission_at_the_boundary():
    arena = Arena()
    ref = arena.opaque('abcdefgh', 64)
    assert render(arena, ref, max_chars=8) == 'abcdefgh'
    assert render(arena, ref, max_chars=7) == 'abcd...'
    assert render(arena, ref, max_chars=3) == '...'
    assert render(arena, ref, limit=0) == '...'


def test_standard_node_spellings_remain_readable():
    arena = Arena()
    x = arena.opaque('x', 64)
    cases = [
        (arena.const(0xff, 64), '0xff'),
        (arena.init_reg('rax'), 'in.rax'),
        (arena._add(('param', 2, 'rax', 64)), 'p2.rax'),
        (arena._add(('load', x, 64, 0)), '*x:64'),
        (arena._add(('un', 'not', x, 64)), 'not(x)'),
        (arena._add(('trunc', x, 8)), 'trunc8(x)'),
        (arena._add(('zext', x, 64)), 'zext64(x)'),
        (arena._add(('sext', x, 64)), 'sext64(x)'),
        (arena._add(('sel', x, x, x, 64)), '(x ? x : x)'),
    ]
    for ref, expected in cases:
        assert render(arena, ref) == expected


def test_fallback_branch_comment_caps_existing_predicate_text():
    block = SimpleNamespace(predicate='x' * 100_000, edge_cond=[], succs=['A', 'B'])
    assert len(branch_condition(block)) <= 512
    assert '...' in branch_condition(block)
