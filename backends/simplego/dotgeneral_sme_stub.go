// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build noasm || !arm64

package simplego

import "unsafe"

// dotProduct_sme stub for non-ARM64 platforms
func dotProduct_sme(a, b unsafe.Pointer, n int64) float32 {
	return 0
}

// dotProductInnerLoopSME stub for non-ARM64 platforms
func dotProductInnerLoopSME(lhsFlat, rhsFlat, outputFlat []float32,
	lhsIdx, rhsIdx, outputIdx, blockDim int) (sum0, sum1, sum2, sum3 float32) {
	// Should never be called since hasSME will be false
	panic("SME not available")
}
