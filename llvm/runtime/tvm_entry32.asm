; i386 entry bridge for the injected payload.
;
; pushfd+pushad preserve every register and the flags while the recovered functions run. The
; original stack (including any loader arguments) and return path are left untouched. The final
; push/ret is a register-preserving tail jump: unlike `jmp eax`, it does not consume one of the
; registers the loader supplied.

.386
.model flat

_TEXT SEGMENT DWORD PUBLIC 'CODE' USE32

EXTERN _tvm_run_payload32:PROC
EXTERN _tvm_entry_rva:DWORD
EXTERN _tvm_oep_rva:DWORD

PUBLIC _tvm_payload_entry
_tvm_payload_entry PROC
    sub     esp, 4
    pushfd
    pushad

    call _tvm_run_payload32

    mov     eax, OFFSET _tvm_payload_entry
    sub     eax, DWORD PTR _tvm_entry_rva
    add     eax, DWORD PTR _tvm_oep_rva
    mov     DWORD PTR [esp+36], eax

    popad
    popfd
    add     esp, 4
    push    DWORD PTR [esp-4]
    ret
_tvm_payload_entry ENDP

_TEXT ENDS
END
