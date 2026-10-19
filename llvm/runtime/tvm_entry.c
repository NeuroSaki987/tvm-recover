/* The injected entry point: run the recovered functions, then hand control to the original entry
 * point with the arguments the loader passed in.
 *
 * Four things are deliberate.
 *
 * **It is a C function, not hand-written assembly.** The ABI is then the compiler's problem, and the
 * contract with the loader is exactly the x64 ABI's: the three incoming arguments (RCX, RDX, R8 --
 * `hinstDLL`, `fdwReason`, `lpvReserved` for a DLL's DllMain) are forwarded unchanged, callee-saved
 * registers and XMM6-XMM15 are preserved, and the stack pointer is restored. Flags and the volatile
 * registers are ABI-volatile and are not preserved; `llvm/miniload.py` measures exactly which of them
 * survive at the original entry point instead of claiming all of them do.
 *
 * **The payload's ~81 KiB of state lives in a callee, not here.** `tvm_run_payload` owns the
 * `tvm_state_t`; when it returns, the frame is gone and the original entry point runs on the stack
 * the loader gave it. Keeping the state alive across the tail call also "works" -- measured: the
 * first version was a single function, the compiler kept the frame, and the original entry point ran
 * 82 KiB lower on the stack. That is legal and wasteful; it also means a DllMain that nearly fills
 * its stack would fault earlier, which is not a property a devirtualization pass should add.
 *
 * **The original entry point is called, not jumped to.** A `jmp` would have to leave RSP pointing at
 * the loader's return address with no frame to return through; calling and returning means the
 * original entry point runs under the loader's own caller, and the value it returns (DllMain's BOOL)
 * is what this function returns.
 *
 * **The image base comes from two RVAs the injector patches in, never from a relocation.**
 * `base = (address of this function) - (this function's RVA)`. Both halves are position independent,
 * so `.neko` needs no base relocation and works in a host with no `.reloc` directory at all -- and
 * the claim is checkable, which `llvm/payload.py` does by refusing to build a payload whose linked
 * image contains a single DIR64 fixup.
 */

#include "tvm_gen.h"
#include "tvm_sealed.h"

/* Defined in tvm_gen.c and filled in by the injector. Declared `extern` here so the compiler has to
 * load them: a `const` defined in the same translation unit could be constant-folded to the zero it
 * is built with, and the payload would then jump into its own image. Measured in the finished
 * disassembly: both arrive as `mov rax, [rip+...]`, not as immediates. */
extern const u64 tvm_entry_rva;
extern const u64 tvm_oep_rva;

/* Forward, because the kernel shape computes the image base from this function's own address before
 * the function is defined. */
unsigned long long tvm_payload_entry(unsigned long long a0, unsigned long long a1,
                                     unsigned long long a2);

#ifdef TVM_STATE_SECTION
/* **Kernel-mode shape: the state is not on the stack.**
 *
 * A 82 688-byte state block is unremarkable in a user-mode thread (1 MiB default stack) and impossible
 * in a kernel one: x64 Windows gives a thread 12 KiB of kernel stack by default, 24 KiB on ARM, and a
 * DriverEntry frame that size would corrupt whatever is below it in the first call. So for a driver the
 * state lives in a **separate writable section appended to the image** (`.neko.b`), and the only thing
 * on the stack is this frame: a saved `rbx`, a `sub rsp,0x28`, and the return address.
 *
 * Its address is computed the same way the original entry point's is: from the two patched RVAs, never
 * from a relocation, so `.neko` still needs no base relocation of its own. `tvm_state_rva` is the third
 * quadword the injector fills in. */
extern const u64 tvm_state_rva;

__declspec(noinline) static u32 tvm_run_payload(void) {
    unsigned long long base = (unsigned long long)(void*)&tvm_payload_entry - tvm_entry_rva;
    tvm_state_t* s = (tvm_state_t*)(void*)(base + tvm_state_rva);
    tvm_payload_main(s);
    return s->report.magic == TVM_REPORT_MAGIC ? s->report.nfunc : 0u;
}
#else
/* **Separate and noinline on purpose.** This function is where the 81 KiB frame lives; inlining it
 * into the entry point would put that frame back on the stack across the original entry point's
 * call, which is exactly what the module docstring says not to do. */
__declspec(noinline) static u32 tvm_run_payload(void) {
    tvm_state_t s;
    tvm_payload_main(&s);
    /* The report is written into the state, which dies here; the caller of the payload gets the
     * same information from `tvm_payload_main` directly (the harness does exactly that). Returning
     * the function count keeps the compiler from eliding the call, which is the only thing this
     * value is for. */
    return s.report.magic == TVM_REPORT_MAGIC ? s.report.nfunc : 0u;
}
#endif

unsigned long long tvm_payload_entry(unsigned long long a0, unsigned long long a1,
                                     unsigned long long a2) {
    unsigned long long base;
    unsigned long long target;
    unsigned long long (*ep)(unsigned long long, unsigned long long, unsigned long long);

    tvm_run_payload();

    base = (unsigned long long)(void*)&tvm_payload_entry - tvm_entry_rva;
    target = base + tvm_oep_rva;
    ep = (unsigned long long (*)(unsigned long long, unsigned long long, unsigned long long))
         (void*)target;
    return ep(a0, a1, a2);
}
