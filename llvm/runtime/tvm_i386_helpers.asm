; CRT-free i386 integer helpers emitted by MSVC for 64-bit C expressions.
;
; The payload links with /NODEFAULTLIB. These implement exactly the helpers referenced by the
; current runtime, with the documented 32-bit MSVC conventions: stack-passed operands and EDX:EAX
; returns. The division helpers preserve all callee-saved registers.

.386
.model flat

_TEXT SEGMENT DWORD PUBLIC 'CODE' USE32

PUBLIC __allmul
__allmul PROC
    mov     eax, DWORD PTR [esp+4]
    mul     DWORD PTR [esp+12]
    mov     ecx, eax
    mov     eax, DWORD PTR [esp+4]
    imul    eax, DWORD PTR [esp+16]
    add     eax, edx
    mov     edx, DWORD PTR [esp+8]
    imul    edx, DWORD PTR [esp+12]
    add     edx, eax
    mov     eax, ecx
    ret     16
__allmul ENDP

PUBLIC __allshl
__allshl PROC
    and     cl, 63
    cmp     cl, 32
    jae     allshl_high
    shld    edx, eax, cl
    shl     eax, cl
    ret
allshl_high:
    mov     edx, eax
    xor     eax, eax
    sub     cl, 32
    shl     edx, cl
    ret
__allshl ENDP

PUBLIC __aullshr
__aullshr PROC
    and     cl, 63
    cmp     cl, 32
    jae     aullshr_high
    shrd    eax, edx, cl
    shr     edx, cl
    ret
aullshr_high:
    mov     eax, edx
    xor     edx, edx
    sub     cl, 32
    shr     eax, cl
    ret
__aullshr ENDP

PUBLIC __aulldiv
__aulldiv PROC
    push    ebx
    push    esi
    push    edi
    push    ebp
    sub     esp, 8
    mov     DWORD PTR [esp], 0
    mov     DWORD PTR [esp+4], 0
    mov     esi, DWORD PTR [esp+28]
    mov     edi, DWORD PTR [esp+32]
    mov     ebp, DWORD PTR [esp+36]
    mov     ebx, DWORD PTR [esp+40]
    xor     eax, eax
    xor     edx, edx
    test    ebx, ebx
    jne     div_ready
    test    ebp, ebp
    je      div_done
div_ready:
    mov     ecx, 64
div_loop:
    shl     DWORD PTR [esp], 1
    rcl     DWORD PTR [esp+4], 1
    shl     esi, 1
    rcl     edi, 1
    rcl     eax, 1
    rcl     edx, 1
    cmp     edx, ebx
    ja      div_sub
    jb      div_next
    cmp     eax, ebp
    jb      div_next
div_sub:
    sub     eax, ebp
    sbb     edx, ebx
    or      DWORD PTR [esp], 1
div_next:
    dec     ecx
    jne     div_loop
div_done:
    mov     eax, DWORD PTR [esp]
    mov     edx, DWORD PTR [esp+4]
    add     esp, 8
    pop     ebp
    pop     edi
    pop     esi
    pop     ebx
    ret     16
__aulldiv ENDP

PUBLIC __aullrem
__aullrem PROC
    push    ebx
    push    esi
    push    edi
    push    ebp
    mov     esi, DWORD PTR [esp+20]
    mov     edi, DWORD PTR [esp+24]
    mov     ebp, DWORD PTR [esp+28]
    mov     ebx, DWORD PTR [esp+32]
    xor     eax, eax
    xor     edx, edx
    test    ebx, ebx
    jne     rem_ready
    test    ebp, ebp
    je      rem_done
rem_ready:
    mov     ecx, 64
rem_loop:
    shl     esi, 1
    rcl     edi, 1
    rcl     eax, 1
    rcl     edx, 1
    cmp     edx, ebx
    ja      rem_sub
    jb      rem_next
    cmp     eax, ebp
    jb      rem_next
rem_sub:
    sub     eax, ebp
    sbb     edx, ebx
rem_next:
    dec     ecx
    jne     rem_loop
rem_done:
    pop     ebp
    pop     edi
    pop     esi
    pop     ebx
    ret     16
__aullrem ENDP

_TEXT ENDS
END
