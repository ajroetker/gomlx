//go:build !noasm && arm64

// NEON-accelerated FP16 dot product for ARM64
// Uses FMLAL/FMLAL2 instructions for efficient FP16×FP16→FP32 accumulation
// This avoids explicit FP16→FP32 conversion overhead by using native instructions

#include "textflag.h"

// func dotProductFP16_neon_asm(a, b unsafe.Pointer, n int64) float32
// Computes dot product of two FP16 vectors, accumulating in FP32
// Uses FMLAL for lower 4 elements and FMLAL2 for upper 4 elements of 8-element FP16 vectors
TEXT ·dotProductFP16_neon_asm(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0       // R0 = a pointer (FP16 array)
	MOVD b+8(FP), R1       // R1 = b pointer (FP16 array)
	MOVD n+16(FP), R2      // R2 = n (count of FP16 elements)

	// Initialize FP32 accumulator to zero
	WORD $0x4f000400       // movi v0.4s, #0 (4 x float32 accumulators)

	// Process 8 FP16 elements at a time using FMLAL/FMLAL2
	LSR $3, R2, R3         // R3 = n / 8 (vector iterations)
	AND $7, R2, R4         // R4 = n % 8 (scalar remainder)

	CBZ R3, fp16_scalarloop // skip vector loop if < 8 elements

fp16_vectorloop:
	// Load 8 FP16 values from each array (128 bits = 8 x FP16)
	WORD $0x4c407804       // ld1 {v4.8h}, [x0], #16
	WORD $0x4c407828       // ld1 {v8.8h}, [x1], #16

	// FMLAL: Floating-point fused multiply-add long (lower)
	// v0.4s += v4.4h[0:3] * v8.4h[0:3] (lower 4 FP16 elements → 4 FP32)
	WORD $0x4e88ec80       // fmlal v0.4s, v4.4h, v8.4h

	// FMLAL2: Floating-point fused multiply-add long (upper)
	// v0.4s += v4.4h[4:7] * v8.4h[4:7] (upper 4 FP16 elements → 4 FP32)
	WORD $0x4ec8ec80       // fmlal2 v0.4s, v4.4h, v8.4h

	SUBS $1, R3, R3
	BNE fp16_vectorloop

	// Horizontal reduction: sum all 4 FP32 lanes
	WORD $0x6e20d400       // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7e30d800       // faddp s0, v0.2s

	// Handle remaining 0-7 elements with scalar path
	CBZ R4, fp16_done

fp16_scalarloop:
	// Load single FP16 from each array
	WORD $0x7c402404       // ldr h4, [x0], #2
	WORD $0x7c402428       // ldr h8, [x1], #2

	// Convert FP16 to FP32
	WORD $0x1ee24084       // fcvt s4, h4
	WORD $0x1ee24108       // fcvt s8, h8

	// FMA: s0 = s0 + s4 * s8
	WORD $0x1f080080       // fmadd s0, s4, s8, s0

	SUBS $1, R4, R4
	BNE fp16_scalarloop

fp16_done:
	FMOVS F0, ret+24(FP)
	RET


// func dotProductFP16Group4_neon_asm(a, b unsafe.Pointer, b_stride, n int64) (r0, r1, r2, r3 float32)
// Calculates 4 dot products sharing the same LHS (a) with FP16 inputs.
// This is optimized for the common case of matrix-vector multiplication where
// one row is multiplied against 4 different columns.
TEXT ·dotProductFP16Group4_neon_asm(SB), NOSPLIT, $0-48
	MOVD a+0(FP), R0       // R0 = a pointer (FP16)
	MOVD b+8(FP), R1       // R1 = b0 pointer (FP16)
	MOVD b_stride+16(FP), R2
	MOVD n+24(FP), R3      // R3 = n (count of FP16 elements)

	// Calculate b1, b2, b3 pointers (stride is in elements, 2 bytes each)
	ADD  R2, R2, R2        // R2 = stride * 2 (bytes)
	ADD  R2, R1, R4        // R4 = b1
	ADD  R2, R4, R5        // R5 = b2
	ADD  R2, R5, R6        // R6 = b3

	// Initialize 4 FP32 accumulators to zero
	WORD $0x4f000400       // movi v0.4s, #0 (acc0)
	WORD $0x4f000401       // movi v1.4s, #0 (acc1)
	WORD $0x4f000402       // movi v2.4s, #0 (acc2)
	WORD $0x4f000403       // movi v3.4s, #0 (acc3)

	LSR $3, R3, R7         // R7 = n / 8 (vector iterations)
	AND $7, R3, R8         // R8 = n % 8 (scalar remainder)

	CBZ R7, g4fp16_scalarloop

g4fp16_vectorloop:
	// Load LHS (shared) - 8 FP16 values
	WORD $0x4c407804       // ld1 {v4.8h}, [x0], #16

	// Load RHS vectors - 8 FP16 values each
	WORD $0x4c407825       // ld1 {v5.8h}, [x1], #16  (b0)
	WORD $0x4c407886       // ld1 {v6.8h}, [x4], #16  (b1)
	WORD $0x4c4078a7       // ld1 {v7.8h}, [x5], #16  (b2)
	WORD $0x4c4078c8       // ld1 {v8.8h}, [x6], #16  (b3)

	// FMLAL for lower 4 elements of each pair
	WORD $0x4e85ec80       // fmlal v0.4s, v4.4h, v5.4h
	WORD $0x4e86ec81       // fmlal v1.4s, v4.4h, v6.4h
	WORD $0x4e87ec82       // fmlal v2.4s, v4.4h, v7.4h
	WORD $0x4e88ec83       // fmlal v3.4s, v4.4h, v8.4h

	// FMLAL2 for upper 4 elements of each pair
	WORD $0x4ec5ec80       // fmlal2 v0.4s, v4.4h, v5.4h
	WORD $0x4ec6ec81       // fmlal2 v1.4s, v4.4h, v6.4h
	WORD $0x4ec7ec82       // fmlal2 v2.4s, v4.4h, v7.4h
	WORD $0x4ec8ec83       // fmlal2 v3.4s, v4.4h, v8.4h

	SUBS $1, R7, R7
	BNE g4fp16_vectorloop

	// Reduce all 4 accumulators
	WORD $0x6e20d400       // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7e30d800       // faddp s0, v0.2s
	WORD $0x6e21d421       // faddp v1.4s, v1.4s, v1.4s
	WORD $0x7e30d821       // faddp s1, v1.2s
	WORD $0x6e22d442       // faddp v2.4s, v2.4s, v2.4s
	WORD $0x7e30d842       // faddp s2, v2.2s
	WORD $0x6e23d463       // faddp v3.4s, v3.4s, v3.4s
	WORD $0x7e30d863       // faddp s3, v3.2s

	CBZ R8, g4fp16_done

g4fp16_scalarloop:
	// Load FP16 values
	WORD $0x7c402404       // ldr h4, [x0], #2
	WORD $0x7c402425       // ldr h5, [x1], #2
	WORD $0x7c402486       // ldr h6, [x4], #2
	WORD $0x7c4024a7       // ldr h7, [x5], #2
	WORD $0x7c4024c8       // ldr h8, [x6], #2

	// Convert FP16 to FP32
	WORD $0x1ee24084       // fcvt s4, h4
	WORD $0x1ee240a5       // fcvt s5, h5
	WORD $0x1ee240c6       // fcvt s6, h6
	WORD $0x1ee240e7       // fcvt s7, h7
	WORD $0x1ee24108       // fcvt s8, h8

	// FMA for all 4 outputs
	WORD $0x1f050080       // fmadd s0, s4, s5, s0
	WORD $0x1f060081       // fmadd s1, s4, s6, s1
	WORD $0x1f070082       // fmadd s2, s4, s7, s2
	WORD $0x1f080083       // fmadd s3, s4, s8, s3

	SUBS $1, R8, R8
	BNE g4fp16_scalarloop

g4fp16_done:
	FMOVS F0, ret0+32(FP)
	FMOVS F1, ret1+36(FP)
	FMOVS F2, ret2+40(FP)
	FMOVS F3, ret3+44(FP)
	RET


// func dotProductBF16_neon_asm(a, b unsafe.Pointer, n int64) float32
// Computes dot product of two BF16 vectors, accumulating in FP32
// Uses BFMLAL/BFMLAL2 for BFloat16 fused multiply-add operations (ARMv8.6+)
TEXT ·dotProductBF16_neon_asm(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0       // R0 = a pointer (BF16 array)
	MOVD b+8(FP), R1       // R1 = b pointer (BF16 array)
	MOVD n+16(FP), R2      // R2 = n (count of BF16 elements)

	// Initialize FP32 accumulator to zero
	WORD $0x4f000400       // movi v0.4s, #0

	// Process 8 BF16 elements at a time
	LSR $3, R2, R3         // R3 = n / 8
	AND $7, R2, R4         // R4 = n % 8

	CBZ R3, bf16_scalarloop

bf16_vectorloop:
	// Load 8 BF16 values from each array
	WORD $0x4c407804       // ld1 {v4.8h}, [x0], #16
	WORD $0x4c407828       // ld1 {v8.8h}, [x1], #16

	// BFMLALB: BFloat16 fused multiply-add long (bottom/lower elements)
	WORD $0x6e48fc80       // bfmlalb v0.4s, v4.8h, v8.8h

	// BFMLALT: BFloat16 fused multiply-add long (top/upper elements)
	WORD $0x4e48fc80       // bfmlalt v0.4s, v4.8h, v8.8h

	SUBS $1, R3, R3
	BNE bf16_vectorloop

	// Horizontal reduction
	WORD $0x6e20d400       // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7e30d800       // faddp s0, v0.2s

	CBZ R4, bf16_done

bf16_scalarloop:
	// Load single BF16 values
	WORD $0x7c402404       // ldr h4, [x0], #2
	WORD $0x7c402428       // ldr h8, [x1], #2

	// Convert BF16 to FP32 by shifting left 16 bits
	// BF16 is just the upper 16 bits of FP32
	WORD $0x0e213c84       // sxtl v4.4s, v4.4h (sign extend)
	WORD $0x0e213d08       // sxtl v8.4s, v8.4h

	// Shift left by 16 to convert BF16 to FP32
	WORD $0x4f500484       // shl v4.4s, v4.4s, #16
	WORD $0x4f500508       // shl v8.4s, v8.4s, #16

	// Extract scalar and multiply-accumulate
	WORD $0x0e043c84       // mov s4, v4.s[0]
	WORD $0x0e043d08       // mov s8, v8.s[0]
	WORD $0x1f080080       // fmadd s0, s4, s8, s0

	SUBS $1, R4, R4
	BNE bf16_scalarloop

bf16_done:
	FMOVS F0, ret+24(FP)
	RET
