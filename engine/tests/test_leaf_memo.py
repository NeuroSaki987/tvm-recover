"""Leaf-query acceleration must preserve source order and bounded storage."""
import random
import unittest

from core.expr import Arena


def uncached_leaves(arena, root):
    """The pre-memo DFS contract, including load nodes and their addresses."""
    result, seen, pending = [], set(), [root]
    while pending:
        ref = pending.pop()
        if ref in seen:
            continue
        seen.add(ref)
        node = arena.nodes[ref]
        kind = node[0]
        if kind in ('init', 'param', 'opaque', 'load'):
            result.append(ref)
        if kind == 'bin':
            pending.extend(node[2:4])
        elif kind == 'un':
            pending.append(node[2])
        elif kind in ('trunc', 'zext', 'sext', 'load'):
            pending.append(node[1])
        elif kind == 'sel':
            pending.extend(node[1:4])
    return result


class CountingNodes(list):
    reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return super().__getitem__(key)


class LeafMemoTests(unittest.TestCase):
    def test_order_load_self_and_shared_sources(self):
        a = Arena()
        x, y, z = [a.init_reg(r) for r in ('rax', 'rbx', 'rcx')]
        shared = a._add(('bin', 'add', x, y, 64))
        loaded = a.load(shared, 64)
        tail = a._add(('bin', 'xor', shared, z, 64))
        root = a._add(('sel', x, tail, loaded, 64))
        self.assertEqual([loaded, y, x, z], a.leaves(root))
        self.assertEqual([loaded, y, x, z], a.leaves(root))

    def test_generated_dags_match_exact_uncached_order(self):
        a, rng = Arena(), random.Random(0x1EAF)
        refs = [a.init_reg(str(i)) for i in range(20)] + [a.const(0, 64)]
        for _ in range(1000):
            x, y, z = (rng.choice(refs) for _ in range(3))
            node = rng.choice([('bin', 'xor', x, y, 64), ('un', 'not', x, 64),
                               ('sel', x, y, z, 64), ('load', x, 64, 0),
                               ('trunc', x, 8), ('zext', x, 64)])
            refs.append(a._add(node))
        for ref in refs:
            self.assertEqual(uncached_leaves(a, ref), a.leaves(ref))

    def test_wide_sources_fall_back_without_losing_or_reordering(self):
        a = Arena()
        refs = [a.opaque('source' + str(i), 64) for i in range(100)]
        root = refs[0]
        for ref in refs[1:]:
            root = a._add(('bin', 'add', root, ref, 64))
        self.assertEqual(list(reversed(refs)), a.leaves(root))
        a.nodes = CountingNodes(a.nodes)
        self.assertEqual(uncached_leaves(a, root), a.leaves(root))
        self.assertGreater(a.nodes.reads, 100)
        memo = getattr(a, '_leaf_memo', {})
        self.assertTrue(memo, 'bounded source summaries should be retained')
        self.assertTrue(all(value is None or len(value) <= 64 for value in memo.values()))

    def test_new_parent_reuses_cached_deep_ancestor(self):
        a = Arena()
        source = a.init_reg('rax')
        root = source
        for _ in range(10000):
            root = a._add(('un', 'not', root, 64))
        self.assertEqual([source], a.leaves(root))
        extra = a.opaque('extra', 64)
        parent = a._add(('bin', 'add', root, extra, 64))
        a.nodes = CountingNodes(a.nodes)
        self.assertEqual([extra, source], a.leaves(parent))
        self.assertLess(a.nodes.reads, 12, 'new parents must not revisit 10,000 cached ancestors')
        a.nodes.reads = 0
        self.assertEqual([extra, source], uncached_leaves(a, parent))
        self.assertGreater(a.nodes.reads, 10000, 'negative control must traverse the ancestor chain')

    def test_results_do_not_cache_mutable_guest_labels_or_returned_lists(self):
        a = Arena()
        x, y = a.init_reg('rax'), a.init_reg('rbx')
        root = a._add(('bin', 'add', x, y, 64))
        labels = {x: 'rax'}
        self.assertEqual({'rax'}, {labels[r] for r in a.leaves(root) if r in labels})
        # A snapshot restore changes the mapping in place, not the immutable DAG.
        labels.clear()
        labels.update({y: 'rdi'})
        self.assertEqual({'rdi'}, {labels[r] for r in a.leaves(root) if r in labels})
        returned = a.leaves(root)
        returned.clear()
        self.assertEqual([y, x], a.leaves(root))


if __name__ == '__main__':
    unittest.main()
