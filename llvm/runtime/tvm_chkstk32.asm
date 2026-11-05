; CRT-free stack probe used by the i386 payload runtime.
;
; MSVC calls __chkstk with the requested byte count in EAX and expects the helper to
; move ESP and copy the return address to the new frame. ECX and EDX are volatile.

.386
.model flat

_TEXT SEGMENT DWORD PUBLIC 'CODE' USE32

PUBLIC __chkstk
__chkstk PROC
    push    ecx
    lea     ecx, DWORD PTR [esp+4]
    sub     ecx, eax
    sbb     edx, edx
    not     edx
    and     ecx, edx
    mov     edx, esp
    and     edx, 0fffff000h
probe_page:
    cmp     ecx, edx
    jae     install_frame
    sub     edx, 1000h
    test    DWORD PTR [edx], edx
    jmp     probe_page
install_frame:
    mov     eax, ecx
    pop     ecx
    xchg    eax, esp
    mov     eax, DWORD PTR [eax]
    mov     DWORD PTR [esp], eax
    ret
__chkstk ENDP

_TEXT ENDS
END
