"""Conservative byte-range alias checks for the concrete memory overlay."""

ADDRESS_SPACE = 1 << 64
ADDRESS_MASK = ADDRESS_SPACE - 1


class _IndependenceBudget(Exception):
    pass


def _free_word(arena, root, work_limit=4096):
    """A full-width independent input survives injectively to this expression.

    For fixed other inputs, add/sub/xor and neg/not are permutations of a word.
    If that word never occurs in the other operand, the image has 2**64 values,
    so no boolean split can make *both* arms constant. Shared/cancelling inputs,
    masks, comparisons, casts and loads are deliberately not certified.
    Caches contain only immutable DAG facts, never mutable memory observations.
    """
    from .expr import _children
    anchors = getattr(arena, '_free_word_anchors', None)
    if anchors is None:
        anchors = arena._free_word_anchors = {}
    # Pairwise independence can be quadratic. Keep it local and bounded; only
    # proved per-node anchors survive this optional query.
    dependencies = {}
    remaining = work_limit

    def charge():
        nonlocal remaining
        if remaining <= 0:
            raise _IndependenceBudget
        remaining -= 1

    def depends(ref, word):
        stack = [(ref, False)]
        while stack:
            charge()
            current, ready = stack.pop()
            key = (current, word)
            if key in dependencies:
                continue
            if current == word:
                dependencies[key] = True
                continue
            children = _children(arena.node(current))
            if not ready:
                stack.append((current, True))
                stack.extend((child, False) for child in children if (child, word) not in dependencies)
            else:
                dependencies[key] = any(dependencies[(child, word)] for child in children)
        return dependencies[(ref, word)]

    stack = [(root, False)]
    while stack:
        charge()
        ref, ready = stack.pop()
        if ref in anchors:
            continue
        n = arena.node(ref)
        if arena.width(ref) != 64:
            anchors[ref] = None
            continue
        if n[0] in ('init', 'param', 'opaque'):
            anchors[ref] = ref
            continue
        if n[0] == 'bin' and n[1] in ('add', 'sub', 'xor'):
            children = (n[2], n[3])
        elif n[0] == 'un' and n[1] in ('neg', 'not'):
            children = (n[2],)
        else:
            anchors[ref] = None
            continue
        if not ready:
            stack.append((ref, True))
            stack.extend((child, False) for child in children if child not in anchors)
            continue
        anchor = None
        for index, child in enumerate(children):
            word = anchors[child]
            if word is not None and all(not depends(other, word)
                                        for other in children[:index] + children[index + 1:]):
                anchor = word
                break
        anchors[ref] = anchor
    return anchors[root]


def finite_addresses(arena, addr, *, max_nodes=4096, max_rebuild_work=16384,
                     candidate_limit=32, prune=False, prefer_selectors=True):
    """Prove a two-address access by exhaustive splitting of a boolean term.

    Both arms must fold. Work is bounded because this is an optional precision
    query, performed for every symbolic store. An incomplete query retains the
    symbolic write; it is not evidence of an infinite address domain. The
    native memory epoch/store is always recorded by Machine.store.
    """
    from .expr import boolean_subexpressions
    cache = getattr(arena, '_finite_address_cache', None)
    if cache is None:
        cache = arena._finite_address_cache = {}
        arena.address_search_stats = dict(calls=0, cache_hits=0, free_word=0,
                                         node_budget=0, work_budget=0, rebuild_work=0,
                                         finite=0, exhausted=0, pruned=0,
                                         normalized=0, hint_hits=0)
    stats = arena.address_search_stats
    stats['calls'] += 1
    key = (addr, max_nodes, max_rebuild_work, candidate_limit, prune, prefer_selectors)
    if key in cache:
        stats['cache_hits'] += 1
        return cache[key]
    cache[key] = None  # recursion guard while historical reads consult history
    learned = max_rebuild_work is None
    if learned:
        addr = arena.normalize(addr)
        constant = arena.as_const(addr)
        if constant is not None:
            stats['normalized'] += 1
            cache[key] = (arena.const(1, 8), constant, constant)
            return cache[key]
    order = arena._descendants_ordered(addr)
    if max_nodes is not None and len(order) > max_nodes:
        stats['node_budget'] += 1
        return None
    try:
        independence_limit = 4096 if max_rebuild_work is None else min(4096, max_rebuild_work)
        if _free_word(arena, addr, work_limit=independence_limit) is not None:
            stats['free_word'] += 1
            return None
    except _IndependenceBudget:
        # Failure to establish independence says nothing about the address.
        pass
    # Full load proofs may be large. Normalize once, then reuse *guards* from
    # earlier successful proofs, never their addresses or loaded values. This
    # avoids rediscovering the same flag amid hundreds of nested flag bits.
    # Bounded store queries retain their original candidate/work allocation.
    hints = getattr(arena, '_address_split_hints', ()) if learned else ()

    def remember(result):
        cache[key] = result
        stats['finite'] += 1
        if learned:
            recent = list(getattr(arena, '_address_split_hints', ()))
            if result[0] in recent:
                recent.remove(result[0])
            recent.append(result[0])
            arena._address_split_hints = recent[-32:]
        return result

    attempted = set()
    if hints:
        descendants = set(order)
        for cond in reversed(hints):
            if cond not in descendants:
                continue
            attempted.add(cond)
            stats['rebuild_work'] += 2 * len(order)
            z = arena.as_const(arena.substitute(
                addr, {cond: arena.const(0, arena.width(cond))}, order=order))
            if z is None:
                continue
            o = arena.as_const(arena.substitute(
                addr, {cond: arena.const(1, arena.width(cond))}, order=order))
            if o is not None:
                stats['hint_hits'] += 1
                return remember((cond, z, o))
    # Explicit selects are usually above many flag-bit subexpressions. Try
    # their selectors first, rather than spending the budget inside flag math.
    selectors = ([arena.node(ref)[1] for ref in order
                  if arena.node(ref)[0] == 'sel' and arena.is_bool(arena.node(ref)[1])]
                 if prefer_selectors else [])
    candidates = dict.fromkeys(selectors + boolean_subexpressions(arena, addr, limit=candidate_limit))
    leaves = arena.leaves(addr) if prune else []
    counts = arena.leaf_path_counts(addr) if prune else {}
    work = 0
    for cond in candidates:
        if cond in attempted:
            continue
        if prune and not arena.pin_can_fold(leaves, counts, cond):
            stats['pruned'] += 1
            continue
        cost = 2 * len(order)
        if max_rebuild_work is not None and work + cost > max_rebuild_work:
            stats['work_budget'] += 1
            break
        work += cost
        stats['rebuild_work'] += cost
        zero = arena.substitute(addr, {cond: arena.const(0, arena.width(cond))}, order=order)
        z = arena.as_const(zero)
        if z is None:
            continue
        one = arena.substitute(addr, {cond: arena.const(1, arena.width(cond))}, order=order)
        o = arena.as_const(one)
        if o is not None:
            remember((cond, z, o))
            break
    else:
        stats['exhausted'] += 1
    return cache[key]


def address_bounds(arena, addr: int, nbytes: int) -> tuple[int, int]:
    """A half-open interval containing every byte this access may touch."""
    def bounds(ref, depth=0):
        if depth > 16:
            return 0, ADDRESS_MASK
        n = arena.node(ref)
        c = arena.as_const(ref)
        if c is not None:
            return c, c
        if n[0] == 'zext':
            return bounds(n[1], depth + 1)
        if n[0] == 'bin' and n[1] in ('add', 'sub'):
            lo1, hi1 = bounds(n[2], depth + 1)
            lo2, hi2 = bounds(n[3], depth + 1)
            lo, hi = ((lo1 + lo2, hi1 + hi2) if n[1] == 'add'
                      else (lo1 - hi2, hi1 - lo2))
            limit = (1 << arena.width(ref)) - 1
            return (lo, hi) if 0 <= lo <= hi <= limit else (0, limit)
        if n[0] == 'bin' and n[1] == 'and':
            for child in (n[2], n[3]):
                mask = arena.as_const(child)
                if mask is not None:
                    return 0, mask
        full = (1 << arena.width(ref)) - 1
        return arena.obits(ref), full ^ arena.zbits(ref)

    lo, hi = bounds(addr)
    return (0, ADDRESS_SPACE) if hi + nbytes > ADDRESS_SPACE else (lo, hi + nbytes)


def _base_offset(arena, addr):
    off = 0
    for _ in range(16):
        n = arena.node(addr)
        if n[0] != 'bin' or n[1] not in ('add', 'sub'):
            break
        rhs = arena.as_const(n[3])
        if rhs is not None:
            off += rhs if n[1] == 'add' else -rhs
            addr = n[2]
            continue
        lhs = arena.as_const(n[2])
        if lhs is not None and n[1] == 'add':
            off += lhs
            addr = n[3]
            continue
        break
    return addr, off & ADDRESS_MASK


def guest_only_address(arena, addr, marks: dict | None = None) -> bool:
    """True when every leaf of `addr` is guest state, so the address is built only from guest values.

    This is the escape test the frame separation needs, and it is stated over **leaves rather than
    shape** because shape failed. The blocking store on `0x140155dc1` is
    `bin:add` of two nested `bin` nodes whose single combined leaf is `in.rcx` -- guest pointer
    arithmetic that no offset pattern matches, so an earlier `init + k` shape test returned NO and
    the gap did not close.

    The criterion is the right one because it is exactly the guest/VM split: an address generated
    from guest values lives in the guest's address space, and the machine's own stack is not
    guest-generable -- if guest code could name it, the VM would corrupt its own saved registers.
    The contrasting store on the same root, `load[?] + load[0x7ffffffef430]`, has leaves that are
    **loads**, one of them from the VM's own frame, so it is not guest-only and it keeps its
    conservative treatment.

    `marks` is the census's identity map, and it is an **argument rather than a machine flag**. A
    `load` leaf that the census has justified as *being* a guest register is guest state just as an
    `init` leaf is, so passing the map is what lets the census path and the CFG path agree -- but the
    census used to work by setting `Machine.frame_separated` around a single store, which made
    *recording an observation* change *program behaviour*: enabling the census could change the
    analysis. Evidence is now an input to the test, and nothing about the test depends on when it
    happens to be called.
    """
    leaves = arena.leaves(addr)
    if not leaves:
        return False
    for lf in leaves:
        if arena.node(lf)[0] == "init":
            continue
        if marks is not None and lf in marks:
            continue
        return False
    return True


def may_alias(arena, left: int, left_bytes: int, right: int, right_bytes: int) -> bool:
    """False only when disjointness is proved, including same-base offsets."""
    lb, lo = _base_offset(arena, left)
    rb, ro = _base_offset(arena, right)
    if lb == rb:
        return ((ro - lo) & ADDRESS_MASK) < left_bytes or ((lo - ro) & ADDRESS_MASK) < right_bytes
    llo, lhi = address_bounds(arena, left, left_bytes)
    rlo, rhi = address_bounds(arena, right, right_bytes)
    return llo < rhi and rlo < lhi


def add_clobber(intervals, lo: int, hi: int):
    """Union intervals so repeated unknown writes keep queries bounded."""
    out = []
    for start, end in intervals:
        if end < lo or hi < start:
            out.append((start, end))
        else:
            lo, hi = min(lo, start), max(hi, end)
    out.append((lo, hi))
    return out
