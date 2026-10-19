/* The runtime the emitted code calls into. Deliberately tiny, deliberately visible.
 *
 * Everything the recovery does not know is funnelled through a function in this file, so that the
 * choice is made once, at a boundary a reader can inspect, instead of being made silently by LLVM:
 *
 *   tvm_undef_flag   an x86 flag whose value the ISA leaves architecturally undefined
 *                    (`evidence/E-FLAG-ORIGIN-AND-OF.md`). `tvm_set_undef` lets a harness drive it,
 *                    which is how the differential test can be exhaustive over it instead of
 *                    pretending to know the answer.
 *   tvm_call_reg     a guest register's value after a native call
 *   tvm_call_flags   the x86 flags word after a native call
 *   tvm_native_call  a native call made by the guest, addressed by its recovered site
 *   tvm_udiv/urem    division, which is UB in LLVM on a zero divisor and a fault on x86
 *
 * The stubs return zero and record nothing. That is not a semantic claim: a caller that reaches
 * them without a harness value is told so by `tvm_stub_hits`, which the differential test asserts is
 * zero for the blocks it claims to have modelled.
 *
 * Compiled with the Microsoft compiler and no CRT, so the emitted module needs no library.
 *   cl /c /O2 /GS- /Zl /W4 tvm_runtime.c
 */

typedef unsigned long long u64;
typedef long long i64;

/* One slot per witness, **keyed by the index the module assigns**, not by a single global value.
 *
 * The first version returned one shared value for every architecturally-undefined flag, which is
 * wrong the moment a block reads two of them: `0x1401505d5` block 2 reads about ninety, and the
 * differential test then reported `rflags` differing in bit 4 -- the interpreter had a per-witness
 * value and the runtime had one global. The index table makes the two agree by construction, and the
 * module emits the index-to-name mapping as comments so the harness can be driven from it.
 */
#define TVM_MAX_UNDEF 8192
#ifndef TVM_PAYLOAD
static int g_undef[TVM_MAX_UNDEF];
static int g_stub_hits = 0;

__declspec(dllexport) void tvm_set_undef(int index, int value) {
    if (index >= 0 && index < TVM_MAX_UNDEF) {
        g_undef[index] = value & 1;
    }
}
__declspec(dllexport) void tvm_clear_undef(void) {
    /* `volatile`, because without it the compiler recognises the loop as `memset` and emits a call
     * to it -- and this module is linked with `/NODEFAULTLIB`, so there is no CRT to resolve it
     * against and the link fails with LNK2019. A volatile pointer defeats the idiom. */
    volatile int *p = g_undef;
    int i;
    for (i = 0; i < TVM_MAX_UNDEF; i++) {
        p[i] = 0;
    }
}
__declspec(dllexport) int tvm_stub_hits(void) { return g_stub_hits; }
__declspec(dllexport) void tvm_reset_stub_hits(void) { g_stub_hits = 0; }

/* The value of an architecturally-undefined flag, chosen by the harness. A real runtime would
 * reproduce whatever the host CPU left in the flag from the instruction that produced it. */
int tvm_undef_flag(int which) {
    if (which >= 0 && which < TVM_MAX_UNDEF) {
        return g_undef[which];
    }
    return 0;
}
#define TVM_STUB_HIT() (g_stub_hits++)
#else
/* **The injected payload edition has no mutable globals at all.**
 *
 * `g_undef` and `g_stub_hits` are written at run time, and the payload is appended as a
 * read+execute section (`.neko`), so a writable global would either fault on the first write or
 * force the section to be RWX. Instead:
 *
 *   * the witness values become a read-only table baked at build time -- one chosen assignment,
 *     recorded in the payload header (`llvm/payload.py`), instead of a harness-settable one. The
 *     builder supplies `tvm_undef_flag` from the generated translation unit;
 *   * the stub-hit counter is a diagnostic for the differential harness only, so the payload
 *     edition drops it. The *formulas* below stay single-sourced: this is the same file, and the
 *     differential test compares the two readings against these very functions.
 */
#define TVM_STUB_HIT() ((void)0)
#endif

i64 tvm_call_reg(int call, int reg) {
    /* A documented default, not a claim about the guest. It exists so the differential test can
     * compare two readings of the same block on identical inputs; the formula is mirrored exactly in
     * `llvm/interp.py::default_callreg`. */
    u64 h = (u64)call * 0x9E3779B97F4A7C15ULL + (u64)reg * 0x100ULL;
    return (i64)(h ^ 0x5A5A5A5AULL);
}

i64 tvm_call_flags(int call) {
    TVM_STUB_HIT();
    return (i64)(0x202ULL | ((u64)call & 7ULL));
}

i64 tvm_call_ret(int call) {
    /* The return value of a call some *other* block made. Mirrored in
     * `llvm/interp.py::default_callresult`, which is also what `tvm_native_call` uses, so a block
     * that both makes a call and reads its result agrees with a block that only reads it. */
    TVM_STUB_HIT();
    return (i64)(((u64)call * 0x9E3779B97F4A7C15ULL) ^ 0xC0FFEEULL);
}

i64 tvm_native_call(i64 site, i64 a, i64 b, i64 c, i64 d) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    TVM_STUB_HIT();
    /* Mirrored in `llvm/interp.py::default_callresult`. Injective in `site`, so a transcription that
     * confuses two call sites cannot pass the differential test by coincidence. */
    return (i64)(((u64)site * 0x9E3779B97F4A7C15ULL) ^ 0xC0FFEEULL);
}

/* x86 `div` by zero raises #DE. LLVM's `udiv` is undefined instead, so the check lives here and the
 * IR calls it rather than emitting an instruction that has no defined behaviour. */
i64 tvm_udiv(u64 a, u64 b) {
    if (b == 0) {
        TVM_STUB_HIT();
        return 0;
    }
    return (i64)(a / b);
}

i64 tvm_urem(u64 a, u64 b) {
    if (b == 0) {
        TVM_STUB_HIT();
        return 0;
    }
    return (i64)(a % b);
}
