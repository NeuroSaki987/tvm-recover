/* The i386 half of the injected entry path.
 *
 * The loader's arguments stay on the native stack and the assembly bridge saves the complete
 * register file before this C function runs. The state lives in this frame, matching the user-mode
 * x64 shape; an optional writable state section is still selected by TVM_STATE_SECTION. C owns the
 * layout and stack-probe call, while tvm_entry32.asm owns the ABI-exact save/restore and tail jump.
 */

#include "tvm_gen.h"
#include "tvm_sealed.h"

extern const u32 tvm_entry_rva;
extern const u32 tvm_oep_rva;

#ifdef TVM_STATE_SECTION
extern const u32 tvm_state_rva;

__declspec(noinline) void tvm_run_payload32(void) {
    u32 base = (u32)(void*)&tvm_run_payload32 - tvm_entry_rva;
    tvm_state_t* s = (tvm_state_t*)(void*)(base + tvm_state_rva);
    tvm_payload_main(s);
}
#else
__declspec(noinline) void tvm_run_payload32(void) {
    tvm_state_t s;
    tvm_payload_main(&s);
}
#endif
