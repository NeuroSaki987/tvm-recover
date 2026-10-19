/* A native probe for the entry-point handoff. Test tooling only -- this file is not part of the
 * payload.
 *
 * Why it exists. `llvm/miniload.py` compares the register state handed to the *original* entry point
 * (a) by a plain call and (b) when it is reached through the payload's entry stub. Capturing that
 * state from two separate `ctypes` calls does not work: the libffi thunk behind each call leaves its
 * own value in r13, and that value differs between call sites by an amount that has nothing to do
 * with the payload. Measured, not guessed -- the first version of the harness reported exactly one
 * "divergence", in r13, with a delta of 0xa0 and every other register equal.
 *
 * So both calls are made from **one** function here, with identical locals and identical arguments:
 * the only input that differs is the target address. Whatever the capture then sees is a property of
 * the code under test.
 *
 * **No SDK headers and no CRT.** This toolchain is invoked as a bare `cl.exe` with no `INCLUDE` or
 * `LIB` set -- the same restriction every other object in this project is built under -- so there is
 * no `windows.h`, no `VirtualAlloc`, and no import library to link against. The capture stub
 * therefore lives in its own read/write/execute section, created by `#pragma section` and given the
 * `RWE` characteristics on the link line, and the module needs no import at all.
 */

typedef unsigned long long u64;

#define PAGE_BYTES 0x1000
#define RECORD_OFF 0x100
/* Offset of the lea's disp32: pushfq(1) + 6 pushes(6) + 8 REX.B pushes(16) + mov rax,rsp(3) + 3 */
#define LEA_DISP (1 + 6 + 16 + 3 + 3)

#pragma section(".neko_cap", read, write)
__declspec(allocate(".neko_cap")) static unsigned char g_page[PAGE_BYTES];

/* push rax rcx rdx rbx rsi rdi  /  pop rdi rsi rbx rdx rcx rax -- in push order, and reversed. */
static const unsigned char PUSH_OP[6] = { 0x50, 0x51, 0x52, 0x53, 0x56, 0x57 };
static const unsigned char POP_OP[6] = { 0x5F, 0x5E, 0x5B, 0x5A, 0x59, 0x58 };

static void build_stub(void) {
    unsigned char* s = g_page;
    int i, off = 0;
    s[off++] = 0x9C;                                          /* pushfq */
    for (i = 0; i < 6; i++) {
        s[off++] = PUSH_OP[i];                                /* push rax rcx rdx rbx rsi rdi */
    }
    for (i = 0; i < 8; i++) {
        s[off++] = 0x41;                                      /* REX.B */
        s[off++] = (unsigned char)(0x50 + i);                 /* push r8..r15 */
    }
    s[off++] = 0x48; s[off++] = 0x89; s[off++] = 0xE0;         /* mov rax, rsp */
    s[off++] = 0x48; s[off++] = 0x8D; s[off++] = 0x0D;         /* lea rcx, [rip+disp32] */
    {
        int disp = RECORD_OFF - (LEA_DISP + 4);
        s[off++] = (unsigned char)(disp & 0xFF);
        s[off++] = (unsigned char)((disp >> 8) & 0xFF);
        s[off++] = (unsigned char)((disp >> 16) & 0xFF);
        s[off++] = (unsigned char)((disp >> 24) & 0xFF);
    }
    s[off++] = 0x48; s[off++] = 0x89; s[off++] = 0x01;         /* mov [rcx], rax */
    for (i = 7; i >= 0; i--) {
        s[off++] = 0x41;                                      /* REX.B */
        s[off++] = (unsigned char)(0x58 + i);                 /* pop r15..r8 */
    }
    for (i = 0; i < 6; i++) {
        s[off++] = POP_OP[i];                                 /* pop rdi rsi rbx rdx rcx rax */
    }
    s[off++] = 0x9D;                                          /* popfq */
    s[off++] = 0xC3;                                          /* ret */
    {
        /* olatile, or the compiler recognises the fill as memset -- and this module has no CRT
         * (the same trap the runtime stub documents). The filler is int3 so a stray entry traps. */
        volatile unsigned char* p = g_page + off;
        int n;
        for (n = 0; n < PAGE_BYTES - off; n++) {
            p[n] = 0xCC;
        }
    }
}

static u64* record(void) { return (u64*)(void*)(g_page + RECORD_OFF); }

__declspec(dllexport) int probe_init(void) {
    int i;
    if (g_page[0] == 0x9C) {
        return 0;
    }
    for (i = 0; i < PAGE_BYTES; i++) {
        ((volatile unsigned char*)g_page)[i] = 0xCC;
    }
    build_stub();
    return 0;
}

__declspec(dllexport) u64 probe_stub(void) { return (u64)(void*)g_page; }

/* Call both targets with the same arguments from the same function. `outA`/`outB` receive the
 * fifteen captured registers plus the flags, in the order the stub pushes them:
 *   r15 r14 r13 r12 r11 r10 r9 r8 rdi rsi rbx rdx rcx rax rflags */
__declspec(dllexport) void probe_pair(u64 t0, u64 t1, u64 a0, u64 a1, u64 a2,
                                      u64* outA, u64* outB) {
    typedef u64 (*ep_t)(u64, u64, u64);
    u64* blk;
    int i;

    *record() = 0;
    (*(ep_t)t0)(a0, a1, a2);
    blk = (u64*)(void*)*record();
    for (i = 0; i < 16; i++) {
        outA[i] = blk ? blk[i] : 0;
    }

    *record() = 0;
    (*(ep_t)t1)(a0, a1, a2);
    blk = (u64*)(void*)*record();
    for (i = 0; i < 16; i++) {
        outB[i] = blk ? blk[i] : 0;
    }
}
