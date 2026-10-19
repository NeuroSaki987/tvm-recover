"""The differential test: our own reading of a block against the code LLVM actually produced.

Three artifacts are compared on the same inputs:

1. **`interp.run_block`** -- a hand-written evaluator over the parsed listing. It shares no code with
   the emitter, so it is an independent reading of the same source.
2. **The emitted LLVM IR**, compiled by llvmlite to COFF, linked by `link.exe` into a PE DLL, and
   called through `ctypes`.
3. **The recorded effect in the listing itself** -- the `; write to ctx @0x...` lines name every slot
   the recovery says the block writes, so a transcription that writes a different set is wrong even
   if it is self-consistent.

The comparison is byte-exact on the whole context block and on the whole guest-memory region, not on
a summary: a difference in one byte of one slot is exactly the failure this exists to catch, and
this project's history is full of summary numbers that agreed while the thing they summarised did
not.

Run:  python llvm/diff.py                    (every self-contained block, both values of every witness)
      python llvm/diff.py --root=14016e8ee --block=1
"""

from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'toolchain_spike'))

import guestir as G  # noqa: E402
import interp as I  # noqa: E402
import lower_llvm as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

def _boot_env() -> None:
    """Load `main/.env` if present. `main/` is not on sys.path for a llvm/ script, so walk up."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'bootenv.py').is_file():
            text = str(parent)
            if text not in sys.path:
                sys.path.insert(0, text)
            try:
                from bootenv import boot
                boot(here)
            except Exception:
                pass
            return


_boot_env()
LISTINGS = ROOT / 'out' / 'memloop-listings'
BUILD = Path(__file__).resolve().parent / 'build'
RUNTIME_C = Path(__file__).resolve().parent / 'runtime' / 'tvm_runtime.c'
PAGE = 0x1000
# The range a 64-bit Windows user-mode process can reserve. On 4-level paging the ceiling is
# 0x7FFFFFFEFFFF; anything above it is kernel space and `VirtualAlloc` refuses with error 87, so a
# guest address there can never be provided by any harness -- which is a fact to report, not to crash on.
USER_LO = 0x10000
USER_HI = 0x800000000000
ALLOC_GRAN = 0x10000

# A synthetic guest address space for the *inputs*, not for the recovered constants.
#
# Several blocks read memory through a guest register whose value the harness chooses (`v429 = in.rcx;
# load64 [v429]`), so with arbitrary register seeds the access lands nowhere and BOTH readings must
# refuse it -- which is honest but tests nothing. Seeding the general registers into a reserved span
# makes the block runnable, and it is a harness choice stated in the open: the comparison is still
# byte-exact, and the region is the same one the engine's own `STACK_BASE` (0x7FFFFFFEFDA0) lives in.
SCRATCH_LO = 0x7FFFFFFE0000
SCRATCH_SIZE = 0x10000

_k32 = ctypes.WinDLL('kernel32', use_last_error=True)
_k32.VirtualAlloc.restype = ctypes.c_void_p
_k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
_k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE = 0x1000, 0x2000, 0x8000
PAGE_EXECUTE_READWRITE = 0x40


def seed_byte(addr: int) -> int:
    """The shared initial-memory function. One definition, in `interp`, used by both readings."""
    return I.seed_free(addr)


def runtime_object() -> Path:
    """Compile the runtime stub once, with the Microsoft compiler and no CRT."""
    out = BUILD / 'tvm_runtime.obj'
    if out.exists() and out.stat().st_mtime > RUNTIME_C.stat().st_mtime:
        return out
    BUILD.mkdir(parents=True, exist_ok=True)
    import subprocess
    cl = 'cl.exe'
    cmd = [cl, '/c', '/nologo', '/O2', '/GS-', '/Zl', '/W4', f'/Fo{out}', str(RUNTIME_C)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f'cl.exe failed ({r.returncode}):\n{r.stdout}\n{r.stderr}')
    return out


_BUILD_SEQ = [0]


def build_module(text: str, tag: str, opt: int = 2) -> Path:
    """Compile and link one module, under a **unique** name.

    `ctypes.WinDLL` keeps the DLL loaded for the life of the process, so re-linking to the same path
    fails with `LNK1104: cannot open file`. That bites as soon as one process builds the same block
    twice, which the region-growth retry and any multi-seed experiment do. A per-build sequence number
    costs a few hundred extra small files in `build/` and removes the whole class.
    """
    import toolchain
    BUILD.mkdir(parents=True, exist_ok=True)
    _BUILD_SEQ[0] += 1
    uniq = f'{tag}_n{_BUILD_SEQ[0]}'
    obj = BUILD / f'{uniq}_O{opt}.obj'
    dll = BUILD / f'{uniq}_O{opt}.dll'
    toolchain.emit_object(text, str(obj), opt=opt)
    exports = tuple(f'/EXPORT:{n}' for n in re.findall(r'^define \S+ @(\w+)\(', text, re.M))
    exports += ('/EXPORT:tvm_set_undef', '/EXPORT:tvm_clear_undef',
                '/EXPORT:tvm_stub_hits', '/EXPORT:tvm_reset_stub_hits')
    toolchain.link_dll([str(obj), str(runtime_object())], str(dll), extra_args=exports)
    return dll


class Region:
    """Memory the emitted code needs, reserved at the *recovered* absolute addresses.

    The recovered addresses are absolute, so the runtime has to provide memory there rather than the
    emitter relocating anything: a relocation would be a transformation of values the recovery
    produced, and choosing which literal is a pointer is exactly the guess this project refuses. If a
    reservation fails, that is reported -- the code is never silently given a different layout.

    Two considerations decide the shape:

    * **Coalesce by page, not by span.** A block can name addresses in two unrelated memory spaces --
      a guest stack address around `0x7ffffffef...` and a host image address around `0x1401...` --
      and a single span covering both is 140 TB, which `VirtualAlloc` refuses with error 487. The
      earlier version of this class did exactly that and reported "cannot reserve", which read like
      an address-space problem rather than the modelling mistake it was.
    * **Seed deterministically**, so the interpreter and the native code read the same initial bytes.
    """

    def __init__(self, addresses):
        self.spans: list[tuple[int, int]] = []
        self.map: dict[int, tuple[int, int]] = {}
        self._seed: dict[int, bytes] = {}
        # **`VirtualAlloc` rounds a requested address down to the 64 KiB allocation granularity, not
        # to the 4 KiB page.** The first version of this class aligned spans to `PAGE`, so a request
        # for `0x140115000` silently came back as `0x140110000`: the reservation succeeded, the seed
        # was written 20 KiB too low, and the differential test then reported a real-looking
        # disagreement at `0x1401150f8` (`interp 0xcb native 0x00`) that was entirely this harness.
        # Spans are granularity-aligned and the returned base is asserted to equal the request.
        gran = ALLOC_GRAN
        pages = sorted({a & ~(gran - 1) for a in addresses})
        run_start = run_end = None
        for p in pages:
            if run_start is None:
                run_start, run_end = p, p + gran
            elif p == run_end:
                run_end = p + gran
            else:
                self.spans.append((run_start, run_end))
                run_start, run_end = p, p + gran
        if run_start is not None:
            self.spans.append((run_start, run_end))
        for lo, hi in self.spans:
            base = _k32.VirtualAlloc(ctypes.c_void_p(lo), hi - lo,
                                     MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE)
            if not base:
                err = ctypes.get_last_error()
                raise RuntimeError(f'cannot reserve guest memory at {lo:#x} '
                                   f'({hi - lo:#x} bytes): Win32 error {err}')
            if base != lo:
                _k32.VirtualFree(ctypes.c_void_p(base), 0, MEM_RELEASE)
                raise RuntimeError(f'VirtualAlloc returned {base:#x} for a request of {lo:#x}; '
                                   f'the recovered addresses would not be honoured')
            self.map[lo] = (base, hi - lo)

    def _span_of(self, a: int):
        for lo, hi in self.spans:
            if lo <= a < hi:
                return lo, self.map[lo][0]
        raise KeyError(f'address {a:#x} is outside every reserved span')

    def seed(self) -> None:
        """Write the shared initial bytes into every reserved span.

        **Built once per span as a `bytes` object and copied with `memmove`.** The first version
        wrote one byte per `ctypes` index, which is fine for a page and ruinous for a span: a block
        that names many guest-stack addresses reserves tens of megabytes, and the differential test
        then appeared to hang -- it was doing millions of individual ctypes stores per witness per
        codegen level. The seed is a pure function of the address, so it can be materialised once.
        """
        for lo, hi in self.spans:
            if lo not in self._seed:
                self._seed[lo] = bytes(seed_byte(a) for a in range(lo, hi))
            base = self.map[lo][0]
            ctypes.memmove(ctypes.c_void_p(base), self._seed[lo], hi - lo)

    def read(self, a: int) -> int:
        lo, base = self._span_of(a)
        return ctypes.cast(base, ctypes.POINTER(ctypes.c_ubyte))[a - base]

    def free(self) -> None:
        for lo, _n in self.map.values():
            _k32.VirtualFree(ctypes.c_void_p(lo), 0, MEM_RELEASE)


def compare_block(fn, block, dll_path: Path, verbose: bool = False, opt: int = 2) -> dict:
    """Run one block through both paths for every value of every witness, and compare.

    The case loop is wrapped in a **region-growth retry**. Fourteen blocks compute a guest address from
    register values, and with any fixed set of register seeds that address falls outside every reserved
    region: the interpreter stops at the unprovided access (it would be a fault) and the compiled code
    faults too, so the case cannot be compared and drops out of the denominator. That is honest but it
    wasted the population -- 14 of 149 blocks, and all 66 sampled cases for one of them.

    Reserving the page the fault named and retrying is a legitimate test rather than a way to make a
    failure disappear, because both readings derive unwritten memory from the same function of the
    address (`interp.seed_free`, and `Region.seed` writing it into the mapped page). After the retry
    they are still compared against each other on identical inputs. What it removes is the harness's
    inability to supply memory, not a disagreement.
    """
    from valueflow import prepare
    fn = prepare(fn)
    block = next(b for b in fn.blocks if b.index == block.index)
    text, manifests = L.emit_function(fn, {block.index})
    man = manifests[0]
    problems = L.hazard_lint(text)
    res = {'root': f'{fn.root:#x}', 'block': block.index, 'label': block.label,
           'term': block.term, 'statements': len(block.stmts), 'uses': man['uses'],
           'hazard': problems, 'cases': [], 'ok': False}
    if problems:
        res['error'] = 'hazard lint failed'
        return res

    dll = build_module(text, f'{fn.root:x}_b{block.index}', opt=opt)
    lib = ctypes.WinDLL(str(dll))
    fname = man['function']
    call = getattr(lib, fname)
    returns = man.get('returns', 'void')
    call.restype = ctypes.c_int if returns == 'i32' else None
    call.argtypes = [ctypes.c_void_p]
    set_undef = lib.tvm_set_undef
    set_undef.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.tvm_clear_undef.argtypes = []
    stub_hits = lib.tvm_stub_hits
    stub_hits.restype = ctypes.c_int

    witnesses = man['undef_used']
    # **Exhaustive when it is affordable, sampled when it is not -- and the result says which.**
    # `0x1401505d5` block 2 reads 386 of them, so an unstated 2^n bound is a coverage claim nobody can
    # check.
    MAX_EXHAUSTIVE = 256
    MAX_SAMPLES = 64
    if len(witnesses) <= 8:
        cases = [{w: (mask >> i) & 1 for i, w in enumerate(witnesses)}
                 for mask in range(1 << len(witnesses))]
        mode = f'exhaustive over 2^{len(witnesses)}'
    else:
        cases = [{w: 0 for w in witnesses}, {w: 1 for w in witnesses}]
        for s in range(MAX_SAMPLES):
            h = (s * 0x9E3779B1 + 0x1234567) & 0x7FFFFFFF
            assign = {}
            for w in witnesses:
                h = (h * 1103515245 + 12345) & 0x7FFFFFFF
                assign[w] = (h >> 16) & 1
            cases.append(assign)
        mode = f'sampled {len(cases)} of 2^{len(witnesses)}'
    res['witness_count'] = len(witnesses)
    res['cases_mode'] = mode
    table = man['undef_table']

    resolve = list(man['guest_addrs'])
    ctx_seed = {}
    for i, (reg, _off) in enumerate(G.CTX_LAYOUT.items()):
        if reg == 'rsp':
            ctx_seed[reg] = 0x7FFFFFFEFDA0
        elif reg == 'rbp':
            ctx_seed[reg] = 0x7FFFFFFEFD40
        elif reg == 'rflags':
            ctx_seed[reg] = 0x202
        else:
            ctx_seed[reg] = SCRATCH_LO + 0x1100 + 0x10 * i

    scratch = [SCRATCH_LO + k for k in range(0, SCRATCH_SIZE, PAGE)]
    extra: list[int] = []
    MAX_GROWTH = 8
    rounds = 0
    while True:
        region = Region(resolve + extra + scratch)
        spans = region.spans
        region.seed()
        res['cases'] = []
        ok = True
        pending: list[int] = []
        for vals in cases:
            region.seed()          # per case, because a block may write memory
            # --- our own reading of the listing
            ctx_a = I.Ctx(dict(ctx_seed))
            mem_a = I.Mem(regions=spans, default=seed_byte)
            for a in resolve:
                for k in range(8):
                    mem_a.bytes[a + k] = seed_byte(a + k)
            rr = I.run_block(block, ctx_a, mem_a, I.Inputs(undef=vals, strict=False))
            # --- the code LLVM produced, on the same initial state
            ctx_b = I.Ctx(dict(ctx_seed))
            buf = ctypes.create_string_buffer(bytes(ctx_b.mem), len(ctx_b.mem))
            lib.tvm_clear_undef()
            for i, name in enumerate(table):
                set_undef(i, vals.get(name, 0))
            lib.tvm_reset_stub_hits()
            fault = ''
            native_succ = None
            try:
                ret = call(ctypes.cast(buf, ctypes.c_void_p))
                if returns == 'i32':
                    native_succ = int(ret)
            except OSError as exc:
                fault = f'native call faulted: {exc}'
            native = bytes(buf.raw[:len(ctx_b.mem)])
            expected = ctx_a.snapshot()
            same_ctx = expected == native
            same_mem = True
            diff_mem = ''
            watched = set(resolve)
            watched.update(mem_a.bytes)
            for a in sorted(watched):
                for k in range(8):
                    want = mem_a.bytes.get(a + k, seed_byte(a + k)) & 0xFF
                    try:
                        gotb = region.read(a + k)
                    except KeyError:
                        continue
                    if want != gotb:
                        same_mem = False
                        diff_mem = (f'first memory difference at {a + k:#x}: '
                                    f'interp {want:#02x} native {gotb:#02x}')
                        break
                if not same_mem:
                    break
            first = ''
            if not same_ctx:
                for i in range(len(expected)):
                    if expected[i] != native[i]:
                        reg = next((r_ for r_, o in G.CTX_LAYOUT.items()
                                    if o <= i < o + 8), '?')
                        first = (f'first context difference at byte {i} ({reg}): '
                                 f'interp {expected[i]:#02x} native {native[i]:#02x}')
                        break
            # **The fork decision is part of the block's effect.** A transcription that computes the
            # right context and then takes the wrong exit is invisible to a memory-only comparison.
            same_succ = (rr.succ == native_succ)
            # **Both sides refusing is neither a pass nor a failure.** It is not exercised *by this
            # reservation*: the outer loop retries it with the faulting page mapped, and only what
            # still cannot run after that is reported as not exercised.
            if rr.faulted and fault:
                if rr.fault_addr is not None:
                    pending.append(rr.fault_addr & ~(PAGE - 1))
                res['cases'].append({'case': len(res['cases']), 'witness': vals,
                                     'verdict': 'not_exercised',
                                     'interp_errors': rr.errors[:1], 'fault': fault,
                                     'fault_addr': None if rr.fault_addr is None
                                                   else hex(rr.fault_addr),
                                     'stub_hits': stub_hits()})
                continue
            cases_ok = same_ctx and same_mem and not rr.errors and not fault and same_succ
            ok = ok and cases_ok
            res['cases'].append({'case': len(res['cases']), 'witness': vals,
                                 'verdict': 'ok' if cases_ok else 'divergent',
                                 'ctx_match': same_ctx, 'mem_match': same_mem,
                                 'succ_match': same_succ, 'interp_succ': rr.succ,
                                 'native_succ': native_succ,
                                 'interp_errors': rr.errors, 'fault': fault,
                                 'detail': first or diff_mem, 'stub_hits': stub_hits()})
        region.free()
        # **Only addresses a process can actually reserve.** A block can compute an address above the
        # user-mode ceiling -- `0x140169e9d` block 9 asks for `0xdbb6916c470000` -- and `VirtualAlloc`
        # answers ERROR_INVALID_PARAMETER (87); no retry can ever map it. Growing towards such an
        # address would turn a reported limitation into a crash, so it is filtered here and counted:
        # those cases stay `not_exercised`, labelled with the address that could not be provided.
        provable = [a for a in pending if USER_LO <= a <= USER_HI - PAGE]
        grown = sorted(set(provable) - set(extra))
        if not grown or rounds >= MAX_GROWTH:
            res['unaddressable'] = sum(
                1 for c in res['cases']
                if c['verdict'] == 'not_exercised' and c.get('fault_addr')
                and not (USER_LO <= (int(c['fault_addr'], 16) & ~(PAGE - 1)) <= USER_HI - PAGE))
            break
        extra.extend(grown)
        rounds += 1
    res['region_growth_rounds'] = rounds
    res.setdefault('unaddressable', 0)

    exercised = [c for c in res['cases'] if c['verdict'] != 'not_exercised']
    unexercised = [c for c in res['cases'] if c['verdict'] == 'not_exercised']
    res['exercised'] = len(exercised)
    res['unexercised'] = len(unexercised)
    # A block with no exercised case has no verdict to give. Reporting it as OK would be a green tick
    # over an empty population, which is the failure this project keeps having to correct.
    res['ok'] = bool(exercised) and all(c['verdict'] == 'ok' for c in exercised)
    res['verdict'] = ('OK' if res['ok'] else
                      'NOT EXERCISED' if not exercised else 'FAIL')
    res['ctx_seed'] = {k: f'{v:#x}' for k, v in ctx_seed.items()}
    return res


def recorded_writes(block) -> set[str]:
    """The slots the listing itself says the block writes, for the third comparison."""
    out = set()
    for st in block.stmts:
        if st.kind in ('ctx-write', 'ctx-partial', 'home') and st.extra and st.extra[0]:
            out.add(st.extra[0])
    return out


def main(argv: list[str]) -> int:
    only_root = only_block = None
    opts = [2]
    listing_dir = LISTINGS
    for a in argv:
        if a.startswith('--root='):
            only_root = a.split('=', 1)[1].lower().replace('0x', '')
        elif a.startswith('--block='):
            only_block = int(a.split('=', 1)[1], 0)
        elif a.startswith('--opt='):
            opts = [int(x) for x in a.split('=', 1)[1].split(',')]
        elif a.startswith('--listings='):
            listing_dir = Path(a.split('=', 1)[1])
    results = []
    for p in sorted(listing_dir.glob('0x*.txt')):
        if only_root and only_root not in p.stem.lower():
            continue
        fn = G.parse_listing(p.read_text(encoding='utf-8', errors='replace'), p)
        for b in fn.blocks:
            if only_block is not None and b.index != only_block:
                continue
            # **Any block whose terminator it states is now in scope**, including a two-way fork:
            # `Block.branches` is true only when the listing printed the condition *and* the 0/1
            # association, and then the emitted function returns which successor is taken.
            if b.term != 'return' and not (b.term == 'goto' and len(b.succs) == 1) \
                    and not b.branches:
                continue
            per_opt = {o: compare_block(fn, b, None, verbose='-v' in argv, opt=o) for o in opts}
            r = per_opt[opts[0]]
            r['per_opt'] = {str(o): v['ok'] for o, v in per_opt.items()}
            # **The levels must agree with each other, not only with us.** A block that passes at one
            # codegen level and fails at another is an LLVM finding, and it must be visible as such
            # rather than averaged into a single verdict.
            r['opt_consistent'] = len(set(r['per_opt'].values())) == 1
            r['ok'] = all(v['ok'] for v in per_opt.values()) and r['opt_consistent']
            r['exercised'] = min(v['exercised'] for v in per_opt.values())
            r['unexercised'] = max(v['unexercised'] for v in per_opt.values())
            results.append(r)
            br = ''
            if r.get('branch'):
                br = f' fork({r["branch"]["cond"]})'
            opttag = '/'.join(f'O{o}={"ok" if v["ok"] else "FAIL"}' for o, v in per_opt.items())
            ux = f'  not-exercised {r["unexercised"]}' if r.get('unexercised') else ''
            print(f'{r["root"]:<12} b{r["block"]:<3} {r.get("verdict", "?"):<13} '
                  f'stmt {r["statements"]:>4}  {opttag}  {r.get("cases_mode", "?")}, '
                  f'{r.get("witness_count", 0)} witness(es){ux}{br}  uses {",".join(r["uses"])}')
            if not r['ok']:
                for oname, rr_ in per_opt.items():
                    for c in rr_['cases']:
                        if c['verdict'] == 'divergent':
                            print(f'      O{oname} case {c["case"]}/{rr_["cases_mode"]}  '
                                  f'{c["detail"]}  '
                                  f'succ interp={c["interp_succ"]} native={c["native_succ"]}  '
                                  f'errors={c["interp_errors"][:2]}  {c["fault"]}  '
                                  f'stub_hits={c["stub_hits"]}')
    good = sum(1 for r in results if r['ok'])
    notex = sum(1 for r in results if not r['ok'] and r.get('verdict') == 'NOT EXERCISED')
    bad = len(results) - good - notex
    print()
    print(f'{good}/{len(results)} blocks agree between our reading and the code LLVM produced '
          f'(codegen levels {opts}); {bad} divergent; {notex} not exercised')
    return 0 if bad == 0 and results else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
