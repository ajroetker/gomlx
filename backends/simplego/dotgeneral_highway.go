// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package simplego provides a pure Go ML backend.
// This file integrates go-highway's optimized SIMD matmul for float32/float64.

package simplego

import (
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// highwayMatMulAvailable indicates whether go-highway matmul is available.
// go-highway provides optimized SIMD matmul using "broadcast A, stream B" algorithm
// which avoids expensive horizontal reductions.
const highwayMatMulAvailable = true

// dgUseHighwayMatMul checks whether the go-highway matmul path should be used.
// Returns true for float32/float64 matrices in standard [M,K]×[K,N] order.
func dgUseHighwayMatMul(dtype dtypes.DType, lhsShape, rhsShape shapes.Shape, params *dotGeneralNodeData) bool {
	if !highwayMatMulAvailable {
		return false
	}

	// Only support float32 and float64
	if dtype != dtypes.Float32 && dtype != dtypes.Float64 {
		return false
	}

	// Check if axes are in standard matmul order
	if !isMatMulOrder(lhsShape, rhsShape,
		params.lhsContractingAxes, params.rhsContractingAxes,
		params.lhsBatchAxes, params.rhsBatchAxes) {
		return false
	}

	return true
}

// execDotGeneralHighwayFloat32 executes float32 matrix multiplication using go-highway.
// go-highway uses "broadcast A, stream B" algorithm which:
// - Broadcasts each A[i,p] element to all SIMD lanes
// - Streams B[p, j:j+lanes] across the row
// - Accumulates C[i, j:j+lanes] += A[i,p] * B[p, j:j+lanes]
//
// This avoids the expensive horizontal reduction required by "load rows, reduce" algorithms.
func execDotGeneralHighwayFloat32(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	outputFlat := output.flat.([]float32)

	batchSize := params.batchSize
	m := params.lhsCrossSize       // M
	n := params.rhsCrossSize       // N
	k := params.contractingSize    // K

	lhsBatchStride := m * k // M * K elements per batch
	rhsBatchStride := k * n // K * N elements per batch
	outputBatchStride := m * n // M * N elements per batch

	// Process each batch element using go-highway's optimized matmul
	for batchIdx := range batchSize {
		lhsStart := batchIdx * lhsBatchStride
		rhsStart := batchIdx * rhsBatchStride
		outputStart := batchIdx * outputBatchStride

		lhsBatch := lhsFlat[lhsStart : lhsStart+lhsBatchStride]
		rhsBatch := rhsFlat[rhsStart : rhsStart+rhsBatchStride]
		outputBatch := outputFlat[outputStart : outputStart+outputBatchStride]

		// MatMulAuto automatically selects between streaming and blocked algorithms
		// based on matrix size for optimal cache efficiency
		matmul.MatMulAutoFloat32(lhsBatch, rhsBatch, outputBatch, m, n, k)
	}
}

// execDotGeneralHighwayFloat64 executes float64 matrix multiplication using go-highway.
func execDotGeneralHighwayFloat64(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
	lhsFlat := lhs.flat.([]float64)
	rhsFlat := rhs.flat.([]float64)
	outputFlat := output.flat.([]float64)

	batchSize := params.batchSize
	m := params.lhsCrossSize       // M
	n := params.rhsCrossSize       // N
	k := params.contractingSize    // K

	lhsBatchStride := m * k
	rhsBatchStride := k * n
	outputBatchStride := m * n

	for batchIdx := range batchSize {
		lhsStart := batchIdx * lhsBatchStride
		rhsStart := batchIdx * rhsBatchStride
		outputStart := batchIdx * outputBatchStride

		lhsBatch := lhsFlat[lhsStart : lhsStart+lhsBatchStride]
		rhsBatch := rhsFlat[rhsStart : rhsStart+rhsBatchStride]
		outputBatch := outputFlat[outputStart : outputStart+outputBatchStride]

		matmul.MatMulAutoFloat64(lhsBatch, rhsBatch, outputBatch, m, n, k)
	}
}
