//go:build !noasm && arm64

// NEON-accelerated dot product for ARM64
// Works on all ARM64 processors (Apple M1+, Linux ARM64, etc.)
// Uses 128-bit NEON vectors (4 x float32)

#include "textflag.h"

// func dotProduct_neon_asm(a, b unsafe.Pointer, n int64) float32
TEXT ·dotProduct_neon_asm(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0       // R0 = a pointer
	MOVD b+8(FP), R1       // R1 = b pointer
	MOVD n+16(FP), R2      // R2 = n (count)

	// Initialize accumulator to zero
	WORD $0x4f000400       // movi v0.4s, #0

	// Process 4 floats at a time using NEON
	LSR $2, R2, R3         // R3 = n / 4 (vector iterations)
	AND $3, R2, R4         // R4 = n % 4 (scalar remainder)

	CBZ R3, scalarloop     // skip vector loop if < 4 elements

vectorloop:
	// Load 4 floats from each array
	WORD $0x4cdf7804       // ld1 {v4.4s}, [x0], #16
	WORD $0x4cdf7828       // ld1 {v8.4s}, [x1], #16

	// Fused multiply-accumulate: v0 += v4 * v8
	WORD $0x4e28cc80       // fmla v0.4s, v4.4s, v8.4s

	SUBS $1, R3, R3
	BNE vectorloop

	// Horizontal reduction: sum all 4 lanes of v0 to get scalar result
	WORD $0x6e20d400       // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7e30d800       // faddp s0, v0.2s

	// Handle remaining 0-3 elements
	CBZ R4, done

scalarloop:
	WORD $0xbc404404       // ldr s4, [x0], #4
	WORD $0xbc404425       // ldr s5, [x1], #4
	WORD $0x1f050080       // fmadd s0, s4, s5, s0
	SUBS $1, R4, R4
	BNE scalarloop

done:
	FMOVS F0, ret+24(FP)
	RET
