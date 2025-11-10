// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && arm64

package simplego

import "unsafe"

// dotProduct_sme is implemented in dotgeneral_sme_arm64.s
// It computes a single dot product of n float32 values using ARM SME.
//
//go:noescape
func dotProduct_sme(a, b unsafe.Pointer, n int64) float32

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

	// Compute 4 independent dot products using SME
	// SME handles variable-length vectors efficiently, so we can process the entire blockDim

	// Compute sum0: dot(lhs, rhs[0])
	if blockDim > 0 {
		sum0 += dotProduct_sme(
			unsafe.Pointer(&lhsFlat[lhsIdx]),
			unsafe.Pointer(&rhsFlat[rhsIdx]),
			int64(blockDim))
	}

	// Compute sum1: dot(lhs, rhs[1])
	rhsIdx1 := rhsIdx + blockDim
	if blockDim > 0 {
		sum1 += dotProduct_sme(
			unsafe.Pointer(&lhsFlat[lhsIdx]),
			unsafe.Pointer(&rhsFlat[rhsIdx1]),
			int64(blockDim))
	}

	// Compute sum2: dot(lhs, rhs[2])
	rhsIdx2 := rhsIdx + 2*blockDim
	if blockDim > 0 {
		sum2 += dotProduct_sme(
			unsafe.Pointer(&lhsFlat[lhsIdx]),
			unsafe.Pointer(&rhsFlat[rhsIdx2]),
			int64(blockDim))
	}

	// Compute sum3: dot(lhs, rhs[3])
	rhsIdx3 := rhsIdx + 3*blockDim
	if blockDim > 0 {
		sum3 += dotProduct_sme(
			unsafe.Pointer(&lhsFlat[lhsIdx]),
			unsafe.Pointer(&rhsFlat[rhsIdx3]),
			int64(blockDim))
	}

	return
}
