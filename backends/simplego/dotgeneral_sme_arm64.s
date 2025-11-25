//go:build !noasm && arm64

// Package sme assembly implementations
// SME-accelerated operations for Apple M4 and later
// Vectorized version using SVE in streaming mode

#include "textflag.h"

// func dotProduct_sme_asm(a, b unsafe.Pointer, n int64) float32
// Restored to working version with stack-based result storage
TEXT ·dotProduct_sme_asm(SB), NOSPLIT, $16-28
	MOVD a+0(FP), R0       // R0 = a pointer
	MOVD b+8(FP), R1       // R1 = b pointer
	MOVD n+16(FP), R2      // R2 = n (count)

	// Zero out result location on stack
	MOVW $0, 12(RSP)
	ADD  $12, RSP, R10     // R10 = &result

	// Calculate vector count and remainder
	// vec_count = n / 16, remainder = n % 16
	ADD  $15, R2, R8       // R8 = n + 15
	CMP  $0, R2
	CSEL LT, R8, R2, R8    // if n < 0: R8 = R8 else R8 = R2
	ASR  $4, R8, R9        // R9 = vec_count = (n+15) / 16
	AND  $0xFFFFFFFFFFFFFFF0, R8  // R8 = (n+15) & ~15
	SUB  R8, R2, R8        // R8 = remainder = n - (vec_count * 16)

	// SME streaming mode with SVE vectors
	WORD $0xd503477f       // smstart
	WORD $0x04a0e3e3       // cntw x3 (get vector length in words)
	WORD $0x1e2703e0       // fmov s0, wzr (zero scalar accumulator)
	WORD $0x25b8c002       // mov z2.s, #0 (zero vector accumulator)
	WORD $0x2598e3e0       // ptrue p0.s (all lanes active)

	// Check if we have any full vectors
	CBZ  R9, remainder     // Skip vector loop if vec_count == 0

	// Initialize additional accumulators for ILP-4
	WORD $0x25b8c003       // mov z3.s, #0 (accumulator 2)
	WORD $0x25b8c006       // mov z6.s, #0 (accumulator 3)
	WORD $0x25b8c007       // mov z7.s, #0 (accumulator 4)

	// Check if we have at least 4 vectors for ILP-4 loop
	CMP  $4, R9
	BLT  vectorloop_ilp1   // Use ILP-1 if < 4 vectors

	// ILP-4 vector loop - process 64×float32 per iteration (4 vectors)
	// Uses 4 independent accumulators to hide latency
	ASR  $2, R9, R5        // R5 = vec_count / 4
	AND  $3, R9, R6        // R6 = vec_count % 4 (remainder for ILP-1)
	MOVD R5, R4            // R4 = ILP-4 loop counter

vectorloop_ilp4:
	// Prefetch next cacheline (256 bytes ahead)
	WORD $0xf98a0010       // prfm pldl1strm, [x0, #256]
	WORD $0xf98a0031       // prfm pldl1strm, [x1, #256]

	// Load and accumulate 4 independent vector pairs
	WORD $0xa540a000       // ld1w {z0.s}, p0/z, [x0]
	WORD $0xa540a021       // ld1w {z1.s}, p0/z, [x1]
	ADD  R3<<2, R0, R0     // Advance pointers
	ADD  R3<<2, R1, R1
	WORD $0x65a10002       // fmla z2.s, p0/m, z0.s, z1.s  (acc1 += v0 * v1)

	WORD $0xa540a004       // ld1w {z4.s}, p0/z, [x0]
	WORD $0xa540a025       // ld1w {z5.s}, p0/z, [x1]
	ADD  R3<<2, R0, R0
	ADD  R3<<2, R1, R1
	WORD $0x65a50003       // fmla z3.s, p0/m, z4.s, z5.s  (acc2 += v2 * v3)

	WORD $0xa540a008       // ld1w {z8.s}, p0/z, [x0]
	WORD $0xa540a029       // ld1w {z9.s}, p0/z, [x1]
	ADD  R3<<2, R0, R0
	ADD  R3<<2, R1, R1
	WORD $0x65a90006       // fmla z6.s, p0/m, z8.s, z9.s  (acc3 += v4 * v5)

	WORD $0xa540a00a       // ld1w {z10.s}, p0/z, [x0]
	WORD $0xa540a02b       // ld1w {z11.s}, p0/z, [x1]
	ADD  R3<<2, R0, R0
	ADD  R3<<2, R1, R1
	WORD $0x65ab0007       // fmla z7.s, p0/m, z10.s, z11.s (acc4 += v6 * v7)

	SUBS $1, R4, R4
	BNE  vectorloop_ilp4

	// Continue with remaining vectors (ILP-1)
	MOVD R6, R4            // R4 = remainder vectors
	CBZ  R4, reduce        // Skip if no remainder

vectorloop_ilp1:
	WORD $0xa540a000       // ld1w {z0.s}, p0/z, [x0]
	WORD $0xa540a021       // ld1w {z1.s}, p0/z, [x1]
	WORD $0x65a10002       // fmla z2.s, p0/m, z0.s, z1.s
	ADD  R3<<2, R0, R0
	ADD  R3<<2, R1, R1
	SUBS $1, R4, R4
	BNE  vectorloop_ilp1

reduce:
	// Reduce each ILP accumulator separately to scalar registers
	WORD $0x65802040       // faddv s0, p0, z2.s (reduce z2 → s0)
	WORD $0x65802061       // faddv s1, p0, z3.s (reduce z3 → s1)
	WORD $0x658020c6       // faddv s6, p0, z6.s (reduce z6 → s6)
	WORD $0x658020e7       // faddv s7, p0, z7.s (reduce z7 → s7)

	// Add all scalar results together into s0
	WORD $0x1e212800       // fadd s0, s0, s1 (s0 += s1)
	WORD $0x1e262800       // fadd s0, s0, s6 (s0 += s6)
	WORD $0x1e272800       // fadd s0, s0, s7 (s0 += s7)

remainder:
	// Handle remainder elements (scalar loop)
	CBZ  R8, done          // Skip if no remainder

	MOVD R8, R4            // R4 = remainder count
scalarloop:
	WORD $0xbc404403       // ldr s3, [x0], #4  (load a[i])
	WORD $0xbc404424       // ldr s4, [x1], #4  (load b[i])
	WORD $0x1e240865       // fmul s5, s3, s4   (multiply)
	WORD $0x1e252800       // fadd s0, s0, s5   (accumulate)
	SUBS $1, R4, R4        // decrement
	BNE  scalarloop        // loop

done:
	// Store result BEFORE smstop
	WORD $0xbd000140       // str s0, [x10]
	WORD $0xd503467f       // smstop

	// Load from stack and return
	WORD $0xbd400fe0       // ldr s0, [sp, #0xc]
	FMOVS F0, ret+24(FP)   // Store to return value
	RET

// func dotProductGroup4_sme_asm(a, b unsafe.Pointer, b_stride, n int64) (r0, r1, r2, r3 float32)
// Uses stack-based storage to preserve results across smstop
TEXT ·dotProductGroup4_sme_asm(SB), NOSPLIT, $32-48
	MOVD a+0(FP), R0       // R0 = a pointer
	MOVD b+8(FP), R1       // R1 = b0 pointer
	MOVD b_stride+16(FP), R2
	MOVD n+24(FP), R3      // R3 = n (count)

	// Zero out result locations on stack (offsets 0, 4, 8, 12)
	MOVW $0, 0(RSP)
	MOVW $0, 4(RSP)
	MOVW $0, 8(RSP)
	MOVW $0, 12(RSP)

	// Calculate b1, b2, b3 pointers
	// stride_bytes = stride * 4 (convert elements to bytes)
	ADD  R2, R2, R2        // R2 = stride * 2
	ADD  R2, R2, R2        // R2 = stride * 4 (bytes)
	ADD  R2, R1, R4        // R4 = b1
	ADD  R2, R4, R5        // R5 = b2
	ADD  R2, R5, R6        // R6 = b3

	// Calculate vector count before entering streaming mode
	// Use hardcoded VL=16 (512-bit / 32-bit) like single function
	ADD  $15, R3, R8       // R8 = n + 15
	CMP  $0, R3
	CSEL LT, R8, R3, R8    // if n < 0: R8 = R8 else R8 = R3
	ASR  $4, R8, R9        // R9 = vec_count = n / 16 (rounded up then down)
	AND  $0xFFFFFFFFFFFFFFF0, R8
	SUB  R8, R3, R8        // R8 = remainder = n % 16

	// Enter streaming mode
	WORD $0xd503477f       // smstart
	WORD $0x04a0e3e7       // cntw x7 (R7 = vector length in words)
	WORD $0x2598e3e0       // ptrue p0.s (all lanes active)

	// Initialize accumulators to zero (z0, z1, z2, z3)
	WORD $0x25b8c000       // mov z0.s, #0
	WORD $0x25b8c001       // mov z1.s, #0
	WORD $0x25b8c002       // mov z2.s, #0
	WORD $0x25b8c003       // mov z3.s, #0

	// Also zero the scalar accumulators
	WORD $0x1e2703e0       // fmov s0, wzr
	WORD $0x1e2703e1       // fmov s1, wzr
	WORD $0x1e2703e2       // fmov s2, wzr
	WORD $0x1e2703e3       // fmov s3, wzr

	CBZ R9, g4_scalarloop

g4_vectorloop:
	// Load LHS (z4)
	WORD $0xa540a004       // ld1w {z4.s}, p0/z, [x0]
	ADD  R7<<2, R0, R0     // Advance LHS pointer

	// Load RHS0 (z5) and accumulate
	WORD $0xa540a025       // ld1w {z5.s}, p0/z, [x1]
	ADD  R7<<2, R1, R1
	WORD $0x65a50080       // fmla z0.s, p0/m, z4.s, z5.s (acc0 += lhs * rhs0)

	// Load RHS1 (z5 - reuse) and accumulate
	WORD $0xa540a085       // ld1w {z5.s}, p0/z, [x4]
	ADD  R7<<2, R4, R4
	WORD $0x65a50081       // fmla z1.s, p0/m, z4.s, z5.s (acc1 += lhs * rhs1)

	// Load RHS2 (z5 - reuse) and accumulate
	WORD $0xa540a0a5       // ld1w {z5.s}, p0/z, [x5]
	ADD  R7<<2, R5, R5
	WORD $0x65a50082       // fmla z2.s, p0/m, z4.s, z5.s (acc2 += lhs * rhs2)

	// Load RHS3 (z5 - reuse) and accumulate
	WORD $0xa540a0c5       // ld1w {z5.s}, p0/z, [x6]
	ADD  R7<<2, R6, R6
	WORD $0x65a50083       // fmla z3.s, p0/m, z4.s, z5.s (acc3 += lhs * rhs3)

	SUBS $1, R9, R9
	BNE g4_vectorloop

	// Horizontal reduction of vectors to scalars
	WORD $0x65802000       // faddv s0, p0, z0.s
	WORD $0x65802021       // faddv s1, p0, z1.s
	WORD $0x65802042       // faddv s2, p0, z2.s
	WORD $0x65802063       // faddv s3, p0, z3.s

g4_scalarloop:
	CBZ R8, g4_done
	MOVD R8, R9            // R9 = count

g4_scalar_inner:
	// Load LHS element
	WORD $0xbc404404       // ldr s4, [x0], #4

	// Load RHS0 and accumulate
	WORD $0xbc404425       // ldr s5, [x1], #4
	WORD $0x1f050080       // fmadd s0, s4, s5, s0

	// Load RHS1 and accumulate
	WORD $0xbc404485       // ldr s5, [x4], #4
	WORD $0x1f050081       // fmadd s1, s4, s5, s1

	// Load RHS2 and accumulate
	WORD $0xbc4044a5       // ldr s5, [x5], #4
	WORD $0x1f050082       // fmadd s2, s4, s5, s2

	// Load RHS3 and accumulate
	WORD $0xbc4044c5       // ldr s5, [x6], #4
	WORD $0x1f050083       // fmadd s3, s4, s5, s3

	SUBS $1, R9, R9
	BNE g4_scalar_inner

g4_done:
	// Store results to stack BEFORE smstop
	WORD $0xbd0003e0       // str s0, [sp, #0]
	WORD $0xbd0007e1       // str s1, [sp, #4]
	WORD $0xbd000be2       // str s2, [sp, #8]
	WORD $0xbd000fe3       // str s3, [sp, #12]

	WORD $0xd503467f       // smstop

	// Load from stack after smstop
	WORD $0xbd4003e0       // ldr s0, [sp, #0]
	WORD $0xbd4007e1       // ldr s1, [sp, #4]
	WORD $0xbd400be2       // ldr s2, [sp, #8]
	WORD $0xbd400fe3       // ldr s3, [sp, #12]

	FMOVS F0, ret0+32(FP)
	FMOVS F1, ret1+36(FP)
	FMOVS F2, ret2+40(FP)
	FMOVS F3, ret3+44(FP)
	RET
