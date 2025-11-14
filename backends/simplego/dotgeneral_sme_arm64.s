//go:build !noasm && arm64

// Package sme assembly implementations
// SME-accelerated operations for Apple M4 and later
// Vectorized version using SVE in streaming mode

#include "textflag.h"

// func dotProduct_sme_asm(a, b unsafe.Pointer, n int64) float32
TEXT ·dotProduct_sme_asm(SB), NOSPLIT, $16-32
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
