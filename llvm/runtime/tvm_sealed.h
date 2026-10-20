/* The payload's per-run state.
 *
 * **This header is why the injected code needs no writable section.** The `.neko` section the
 * injector appends is read+execute (`0x60000020`); every byte the payload mutates lives in a block
 * of memory the *caller* owns -- the stub's stack frame when the loader runs the entry point, a
 * `ctypes` buffer when `llvm/miniload.py` runs the same code. The image therefore contains no
 * packer-shaped artefact (no RWX section, no second RW section holding state), and the same `.neko`
 * bytes can be injected into any x64 PE without touching its `.data`.
 *
 * The layout is published twice -- `payload.py` computes the offsets and writes them into the
 * payload header, `tvm_sealed.c` static-asserts them against `offsetof` -- so a layout disagreement
 * fails the build instead of making the differential test read plausible-looking garbage.
 */

#ifndef TVM_SEALED_H
#define TVM_SEALED_H

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef long long i64;

/* 17 eight-byte slots, `llvm/guestir.py::CTX_LAYOUT`:
 *   rax rbx rcx rdx rsp rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 rflags */
#define TVM_CTX_BYTES 136

/* Capacity of the sealed byte map: one slot per guest byte written. The state records peak
 * occupancy, so the differential test asserts the capacity was never approached instead of assuming
 * it. 8192 slots = ~81 KiB of state, which the stub probes page by page as it takes it. */
#ifndef TVM_SLOTS
#define TVM_SLOTS 8192
#endif

/* Results are kept for at most this many functions; the builder refuses a corpus that exceeds it
 * rather than truncating the report.
 *
 * **Must equal `llvm/payload.py::NFUNC_MAX`.** The pairing is a layout contract, not a convention:
 * `payload.py` derives `STATE_BYTES` from its copy and hands it to the generated C, this header sizes
 * `result[]`, and `tvm_state_layout_ok()` checks the consequence at run time. It was 256, which the
 * DeltaForce base image outgrew (351 well-formed roots, 265 of them composable); 512 costs 2 048 more
 * bytes of state and all of it is in `result[]`. */
#define TVM_NFUNC_MAX 512

/* Little-endian spellings of the ASCII tags: 'NEKO' and 'NKP1' read back as these values. */
#define TVM_STATE_MAGIC 0x4F4B454EULL
#define TVM_REPORT_MAGIC 0x31504B4EUL

/* What the payload did, so a caller that only ran the entry point can still tell whether the
 * devirtualized code executed. `magic` is written last, so a partial run leaves it zero rather than
 * leaving a half-filled report that looks complete. */
typedef struct tvm_report_block {
    u32 magic;
    u32 version;
    u32 nfunc;
    u32 slots;
    u32 overflow;
    u32 pad0;
    u64 fuel;
    u64 total_steps;
    u64 loads;
    u64 stores;
    u64 pad1;
} tvm_report_t;

typedef struct tvm_state {
    u64 magic;
    u32 slots;
    u32 used;
    u32 overflow;
    u32 pad0;
    u64 loads;
    u64 stores;
    u64 total_steps;
    u32 calls;
    u32 pad1;
    u64 result[TVM_NFUNC_MAX];
    u8 ctx[TVM_CTX_BYTES];
    tvm_report_t report;
    /* The table last, so its size never shifts the fields above. */
    u64 keys[TVM_SLOTS];
    u8 vals[TVM_SLOTS];
    u8 live[TVM_SLOTS];
} tvm_state_t;

#endif /* TVM_SEALED_H */
