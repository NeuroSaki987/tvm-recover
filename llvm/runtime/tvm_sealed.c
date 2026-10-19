/* The payload's sealed guest memory: what the injected code reads and writes instead of the
 * absolute addresses the analysis invented.
 *
 * Why this exists. The transcription lowers a recovered guest address `A` to an access at `A`
 * (`llvm/lower_llvm.py`, `mem_mode='direct'`). That is the honest spelling for an artifact compared
 * against an interpreter walking the same listing, but it is not executable inside another process:
 * `A` is an address from the *analysis* run -- a stack frame at `0x7ffffffefda0`, a witness pointer
 * above the user-space limit -- so a load there faults and a store there corrupts whatever lives at
 * that address in the host. Injecting that code into a PE and running it at the entry point would
 * take the host down on the first memory access.
 *
 * `mem_mode='sealed'` therefore routes every guest access through `tvm_ld<bits>`/`tvm_st<bits>`,
 * which address a private table instead. The semantics are **exactly `llvm/interp.py::Mem` with a
 * `default` of `seed_free`**:
 *
 *   read a byte never written -> `tvm_seed_byte(a)`, the same pure function of the address that
 *                                `interp.seed_free` is;
 *   write a byte              -> recorded, readable back at the same address.
 *
 * The address is still an immediate at the call site, so the emitted code still *says* which
 * address the recovery named -- the fact an analyst wants is in the instruction stream, it just
 * does not become a dereference.
 *
 * **All mutable state is the caller's.** `tvm_state_t` (tvm_sealed.h) is passed in: the stub puts it
 * in its stack frame, the differential harness in a buffer it owns. That is what lets `.neko` be a
 * read+execute section instead of RWX or a second section.
 *
 * The table is open addressing over three parallel arrays rather than a packed key, because the key
 * is a full 64-bit address: storing `address + 1` with 0 as "empty" would alias the address
 * `0xffffffffffffffff` -- which the closure report saw a block compute -- with an empty slot.
 */

#include "tvm_gen.h"
#include "tvm_sealed.h"

/* **No <stddef.h>, and no compile-time `offsetof`.** This toolchain is invoked as a bare `cl.exe`
 * with no `INCLUDE` set -- which is how every other object in this project is compiled -- so
 * `offsetof` has no header to come from, and this compiler's C front end rejects
 * `__builtin_offsetof` (C2059; measured, not assumed). The offsets `payload.py` computes in Python
 * are therefore published *and checked at run time*: `tvm_state_layout_ok()` compares every field
 * offset against the generated `TVM_OFF_*` constants and the differential test asserts it returns 1.
 * A layout disagreement then fails a named test with the offending offsets, which is what the static
 * assert was for; without a header it is the only way to get the same guarantee. */
#define TVM_SOFF(f) ((u64)(char*)&(((tvm_state_t*)(void*)0)->f))

u64 tvm_state_off(u32 which) {
    switch (which) {
    case 0: return TVM_SOFF(magic);
    case 1: return TVM_SOFF(slots);
    case 2: return TVM_SOFF(used);
    case 3: return TVM_SOFF(overflow);
    case 4: return TVM_SOFF(loads);
    case 5: return TVM_SOFF(stores);
    case 6: return TVM_SOFF(total_steps);
    case 7: return TVM_SOFF(calls);
    case 8: return TVM_SOFF(result);
    case 9: return TVM_SOFF(ctx);
    case 10: return TVM_SOFF(report);
    case 11: return TVM_SOFF(keys);
    case 12: return TVM_SOFF(vals);
    case 13: return TVM_SOFF(live);
    default: return ~0ULL;
    }
}

u32 tvm_state_layout_ok(void) {
    return (TVM_SOFF(magic) == (u64)TVM_OFF_MAGIC
            && TVM_SOFF(slots) == (u64)TVM_OFF_SLOTS
            && TVM_SOFF(used) == (u64)TVM_OFF_USED
            && TVM_SOFF(overflow) == (u64)TVM_OFF_OVERFLOW
            && TVM_SOFF(loads) == (u64)TVM_OFF_LOADS
            && TVM_SOFF(stores) == (u64)TVM_OFF_STORES
            && TVM_SOFF(total_steps) == (u64)TVM_OFF_STEPS
            && TVM_SOFF(calls) == (u64)TVM_OFF_CALLS
            && TVM_SOFF(result) == (u64)TVM_OFF_RESULT
            && TVM_SOFF(ctx) == (u64)TVM_OFF_CTX
            && TVM_SOFF(report) == (u64)TVM_OFF_REPORT
            && TVM_SOFF(keys) == (u64)TVM_OFF_KEYS
            && TVM_SOFF(vals) == (u64)TVM_OFF_VALS
            && TVM_SOFF(live) == (u64)TVM_OFF_LIVE
            && (u64)sizeof(tvm_state_t) <= (u64)TVM_STATE_BYTES) ? 1u : 0u;
}

/* This one *is* a constant expression, so it stays a compile-time check. */
typedef char tvm_state_slots[TVM_SLOTS == (1 << TVM_SLOT_BITS) ? 1 : -1];

u64 tvm_state_bytes(void) { return (u64)sizeof(tvm_state_t); }
u64 tvm_state_slot_bits(void) { return (u64)TVM_SLOT_BITS; }

/* The shared initial-memory function. Mirrors `interp.seed_free` operation for operation, so the
 * two readings agree on every unwritten byte by construction; one test compares the two over a
 * range of addresses instead of assuming it. */
int tvm_seed_byte(u64 a) {
    return (int)((a * 37u + (a >> 13) * 11u + 5u) & 0xFFu);
}
u8 tvm_seed_byte_of(u64 a) { return (u8)tvm_seed_byte(a); }

static u64 sl_slot(u64 a) {
    return (a * 0x9E3779B97F4A7C15ull) & (u64)(TVM_SLOTS - 1);
}

static u8 sl_read(tvm_state_t* s, u64 a) {
    u64 i = sl_slot(a);
    u32 n;
    for (n = 0; n < TVM_SLOTS; n++) {
        if (!s->live[i]) {
            return (u8)tvm_seed_byte(a);
        }
        if (s->keys[i] == a) {
            return s->vals[i];
        }
        i = (i + 1) & (u64)(TVM_SLOTS - 1);
    }
    s->overflow++;
    return (u8)tvm_seed_byte(a);
}

static void sl_write(tvm_state_t* s, u64 a, u8 v) {
    u64 i = sl_slot(a);
    u32 n;
    for (n = 0; n < TVM_SLOTS; n++) {
        if (!s->live[i]) {
            s->live[i] = 1;
            s->keys[i] = a;
            s->vals[i] = v;
            s->used++;
            return;
        }
        if (s->keys[i] == a) {
            s->vals[i] = v;
            return;
        }
        i = (i + 1) & (u64)(TVM_SLOTS - 1);
    }
    s->overflow++;
}

void tvm_state_clear_mem(tvm_state_t* s) {
    /* `volatile`, for the reason spelled out in `tvm_runtime.c`: without it the compiler recognises
     * the loop as `memset`, and this module is linked `/NODEFAULTLIB`. */
    volatile u8* p = s->live;
    u32 i;
    for (i = 0; i < TVM_SLOTS; i++) {
        p[i] = 0;
    }
    s->used = 0;
    s->slots = TVM_SLOTS;
}

void tvm_state_reset(tvm_state_t* s) {
    /* `volatile` here for a reason measured at `-O1`: without it MSVC recognises this loop as
     * `memset` and the link fails with LNK2019, because the payload has no CRT. `-Od` and `-O2` do
     * not do it, so the idiom has to be defeated at every level, not just the one that was tried
     * first (`llvm/payload_readability.py` builds all three). */
    volatile u64* r = s->result;
    u32 i;
    tvm_state_clear_mem(s);
    /* The counters are *cumulative over the run* and are reset only here, so the report's load and
     * store totals cover every function the payload ran, while `used` describes the last one. */
    s->overflow = 0;
    s->loads = 0;
    s->stores = 0;
    s->total_steps = 0;
    s->calls = 0;
    for (i = 0; i < TVM_NFUNC_MAX; i++) {
        r[i] = 0;
    }
    s->report.magic = 0;
    s->report.version = 0;
    s->magic = TVM_STATE_MAGIC;
}

/* --- the eight accessors the emitted code calls ------------------------------------------------
 *
 * One per width, and the parameter/return types are the **access width**, matching the module's
 * `declare {i8,i16,i32,i64} @tvm_ld<bits>(ptr, i64)` and `declare void @tvm_st<bits>(ptr, i64,
 * {i8,i16,i32,i64})`. That is not cosmetic: LLVM's own verifier then checks that the listing's
 * stated width and the helper agree, where an `i64`-everywhere helper would let a wrong width link
 * and disagree with the interpreter only in the upper bits.
 */
u8 tvm_ld8(tvm_state_t* s, u64 a) {
    s->loads++;
    return sl_read(s, a);
}
void tvm_st8(tvm_state_t* s, u64 a, u8 v) {
    s->stores++;
    sl_write(s, a, v);
}
u16 tvm_ld16(tvm_state_t* s, u64 a) {
    s->loads++;
    return (u16)((u32)sl_read(s, a) | ((u32)sl_read(s, a + 1) << 8));
}
void tvm_st16(tvm_state_t* s, u64 a, u16 v) {
    s->stores++;
    sl_write(s, a, (u8)(v & 0xFF));
    sl_write(s, a + 1, (u8)((v >> 8) & 0xFF));
}
u32 tvm_ld32(tvm_state_t* s, u64 a) {
    u32 v = 0;
    int i;
    s->loads++;
    for (i = 0; i < 4; i++) {
        v |= (u32)sl_read(s, a + (u64)i) << (8 * i);
    }
    return v;
}
void tvm_st32(tvm_state_t* s, u64 a, u32 v) {
    int i;
    s->stores++;
    for (i = 0; i < 4; i++) {
        sl_write(s, a + (u64)i, (u8)((v >> (8 * i)) & 0xFF));
    }
}
u64 tvm_ld64(tvm_state_t* s, u64 a) {
    u64 v = 0;
    int i;
    s->loads++;
    for (i = 0; i < 8; i++) {
        v |= (u64)sl_read(s, a + (u64)i) << (8 * i);
    }
    return v;
}
void tvm_st64(tvm_state_t* s, u64 a, u64 v) {
    int i;
    s->stores++;
    for (i = 0; i < 8; i++) {
        sl_write(s, a + (u64)i, (u8)((v >> (8 * i)) & 0xFF));
    }
}

/* --- the report ------------------------------------------------------------------------------
 *
 * `tvm_payload_main` calls this last, so a partial run leaves `magic` zero rather than leaving a
 * half-filled report that looks complete. The report lives in the caller's state block, so whoever
 * owns the state can read it after the payload returns; the miniload harness works out the stub's
 * dead frame from the restored stack pointer and reads it there (see `llvm/miniload.py`).
 */
void tvm_report_fill(tvm_report_t* r, u32 nfunc, u64 fuel, u64 total_steps, u32 overflow,
                     u64 loads, u64 stores) {
    r->magic = TVM_REPORT_MAGIC;
    r->version = 1;
    r->nfunc = nfunc;
    r->slots = TVM_SLOTS;
    r->overflow = overflow;
    r->fuel = fuel;
    r->total_steps = total_steps;
    r->loads = loads;
    r->stores = stores;
}
