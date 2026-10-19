; **Our own `__chkstk`.** The payload's entry point keeps its state (~81 KiB, `tvm_sealed.h`) on the
; stack, so MSVC emits a stack probe. Measured on this toolchain (the disassembly is in
; `llvm/measurements/write-devirt-chkstk.log`), for a frame past one page it emits
;
;       mov  eax,14028h
;       call __chkstk
;       sub  rsp,rax
;
; i.e. the size arrives in **eax**, `__chkstk` must touch one byte per page from the current stack
; pointer down past the new frame, and rsp must be left alone because the caller subtracts it. The
; CRT's implementation is unavailable here (`/NODEFAULTLIB`), and skipping the probe is not an
; option: a `sub rsp, 8000h` in one step steps over the guard page, so the growth fault the guard
; exists to raise never happens and the write lands in whatever is mapped below.
;
; Registers: rax is preserved (the caller subtracts it), rcx is saved and restored, r11 is scratch
; and volatile. Flags are not preserved, which matches the CRT's own contract.

_TEXT SEGMENT ALIGN(16) 'CODE'

PUBLIC __chkstk
__chkstk PROC
    push    rcx
    lea     r11, [rsp+8]          ; r11 = stack pointer at entry (pointing at the return address)
    mov     rcx, rax              ; bytes the caller is about to reserve
probe:
    sub     r11, 1000h
    or      byte ptr [r11], 0     ; touch the page; a read-modify-write cannot be elided
    sub     rcx, 1000h
    cmp     rcx, 1000h
    jae     probe
    sub     r11, rcx
    or      byte ptr [r11], 0
    pop     rcx
    ret
__chkstk ENDP

_TEXT ENDS
END
