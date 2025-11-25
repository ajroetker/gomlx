// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && arm64

package simplego

import (
	"runtime"
	"unsafe"
)

// dotProduct_sme_asm is implemented in dotgeneral_sme_arm64.s
// It computes a single dot product of n float32 values using ARM SME.
//
//go:noescape
func dotProduct_sme_asm(a, b unsafe.Pointer, n int64) float32

// dotProductGroup4_sme_asm is implemented in dotgeneral_sme_arm64.s
// It computes 4 dot products simultaneously sharing the same LHS vector using SME.
// b_stride is the stride in elements (float32) between the start of each RHS vector.
//
//go:noescape
func dotProductGroup4_sme_asm(a, b unsafe.Pointer, b_stride, n int64) (r0, r1, r2, r3 float32)

// dotProduct_sme computes dot product and keeps the source slices alive.
// This prevents the compiler from optimizing away or relocating the slice backing arrays.
func dotProduct_sme(aSlice, bSlice []float32, aIdx, bIdx int, n int64) float32 {
	result := dotProduct_sme_asm(
		unsafe.Pointer(&aSlice[aIdx]),
		unsafe.Pointer(&bSlice[bIdx]),
		n)
	// Keep slices alive until after assembly completes
	runtime.KeepAlive(aSlice)
	runtime.KeepAlive(bSlice)
	return result
}

// dotProductInnerLoopSME is a wrapper that uses SME to accelerate the inner dot product loop.
// It processes the entire inner loop with SME for maximum performance.
//
// This function matches the signature needed by buildDotGeneralKernel to replace the inner loop.
//
// Performance: ~2.51x faster than scalar for large vectors (n >= 2048)
func dotProductInnerLoopSME(lhsFlat, rhsFlat, outputFlat []float32,
	lhsIdx, rhsIdx, outputIdx, blockDim int) (sum0, sum1, sum2, sum3 float32) {

	// Initialize sums from current output values
	sum0 = outputFlat[outputIdx]
	sum1 = outputFlat[outputIdx+1]
	sum2 = outputFlat[outputIdx+2]
	sum3 = outputFlat[outputIdx+3]

	// Compute 4 independent dot products using SME Group4 optimization.
	// This significantly reduces smstart/smstop overhead (1x instead of 4x).
	r0, r1, r2, r3 := dotProductGroup4_sme_asm(
		unsafe.Pointer(&lhsFlat[lhsIdx]),
		unsafe.Pointer(&rhsFlat[rhsIdx]),
		int64(blockDim), // stride in elements
		int64(blockDim)) // length n

	runtime.KeepAlive(lhsFlat)
	runtime.KeepAlive(rhsFlat)

	sum0 += r0
	sum1 += r1
	sum2 += r2
	sum3 += r3

	return
}
