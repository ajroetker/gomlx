package simplego

import (
	"unsafe"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// isStandardMatmul checks if the DotGeneral operation is a standard matrix multiplication
// that doesn't require any transposition or complex axis manipulation.
//
// Standard patterns that can skip normalization:
// 1. Matrix × Matrix: [M, K] × [K, N] → [M, N] (contracting on last axis of lhs, first of rhs)
// 2. Matrix × Vector: [M, K] × [K] → [M] (contracting on last axis of lhs, only axis of rhs)
// 3. Batched MatMul: [B, M, K] × [B, K, N] → [B, M, N] (batch on first axis)
//
// Returns true if we can use the fast path (no transpose needed).
func isStandardMatmul(lhsShape, rhsShape shapes.Shape, lhsContractingAxes, rhsContractingAxes, lhsBatchAxes, rhsBatchAxes []int) bool {
	lhsRank := lhsShape.Rank()
	rhsRank := rhsShape.Rank()

	// Check for standard matrix-matrix multiplication: [M, K] × [K, N]
	if lhsRank == 2 && rhsRank == 2 &&
		len(lhsContractingAxes) == 1 && len(rhsContractingAxes) == 1 &&
		len(lhsBatchAxes) == 0 && len(rhsBatchAxes) == 0 {
		// Contracting: lhs last axis (1) with rhs first axis (0)
		if lhsContractingAxes[0] == 1 && rhsContractingAxes[0] == 0 {
			return true
		}
	}

	// Check for matrix-vector multiplication: [M, K] × [K]
	if lhsRank == 2 && rhsRank == 1 &&
		len(lhsContractingAxes) == 1 && len(rhsContractingAxes) == 1 &&
		len(lhsBatchAxes) == 0 && len(rhsBatchAxes) == 0 {
		// Contracting: lhs last axis (1) with rhs only axis (0)
		if lhsContractingAxes[0] == 1 && rhsContractingAxes[0] == 0 {
			return true
		}
	}

	// Check for batched matrix multiplication: [B, M, K] × [B, K, N]
	if lhsRank == 3 && rhsRank == 3 &&
		len(lhsContractingAxes) == 1 && len(rhsContractingAxes) == 1 &&
		len(lhsBatchAxes) == 1 && len(rhsBatchAxes) == 1 {
		// Batch on first axis (0)
		if lhsBatchAxes[0] == 0 && rhsBatchAxes[0] == 0 {
			// Contracting: lhs axis 2 with rhs axis 1
			if lhsContractingAxes[0] == 2 && rhsContractingAxes[0] == 1 {
				return true
			}
		}
	}

	// Check for multi-batch matmul: [B1, B2, M, K] × [B1, B2, K, N]
	if lhsRank == 4 && rhsRank == 4 &&
		len(lhsContractingAxes) == 1 && len(rhsContractingAxes) == 1 &&
		len(lhsBatchAxes) == 2 && len(rhsBatchAxes) == 2 {
		// Batch on first two axes
		if lhsBatchAxes[0] == 0 && lhsBatchAxes[1] == 1 &&
			rhsBatchAxes[0] == 0 && rhsBatchAxes[1] == 1 {
			// Contracting: lhs axis 3 with rhs axis 2
			if lhsContractingAxes[0] == 3 && rhsContractingAxes[0] == 2 {
				return true
			}
		}
	}

	return false
}

// isMemoryContiguous checks if the tensor layout is already contiguous in memory
// for the given contracting pattern.
func isMemoryContiguous(shape shapes.Shape, contractingAxes, batchAxes []int) bool {
	rank := shape.Rank()
	if rank == 0 {
		return true
	}

	// Build expected order: batch axes first, then cross, then contracting
	expectedOrder := make([]int, 0, rank)
	expectedOrder = append(expectedOrder, batchAxes...)

	// Add cross axes (non-batch, non-contracting)
	isContracting := make(map[int]bool)
	isBatch := make(map[int]bool)
	for _, a := range contractingAxes {
		isContracting[a] = true
	}
	for _, a := range batchAxes {
		isBatch[a] = true
	}

	for i := 0; i < rank; i++ {
		if !isContracting[i] && !isBatch[i] {
			expectedOrder = append(expectedOrder, i)
		}
	}
	expectedOrder = append(expectedOrder, contractingAxes...)

	// Check if expected order is 0, 1, 2, ... (natural order)
	for i, axis := range expectedOrder {
		if axis != i {
			return false
		}
	}
	return true
}

// canUseFastPath determines if we can use the optimized fast path for this DotGeneral operation.
func canUseFastPath(lhs, rhs *Buffer, params *dotGeneralNodeData) bool {
	// Only support float32 fast path for now (most common)
	if lhs.shape.DType != dtypes.Float32 {
		return false
	}

	// Check if it's a standard matmul pattern
	if !isStandardMatmul(lhs.shape, rhs.shape,
		params.lhsContractingAxes, params.rhsContractingAxes,
		params.lhsBatchAxes, params.rhsBatchAxes) {
		return false
	}

	return true
}

// execDotGeneralFastPath executes a standard matrix multiplication without normalization.
// This is a significant optimization for the common case of A × B matrix multiplication.
// Returns true if fast path was used, false if caller should use standard path.
func execDotGeneralFastPath(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) bool {
	if !canUseFastPath(lhs, rhs, params) {
		return false
	}

	// Execute the optimized float32 path
	execDotGeneralFastPathFloat32(backend, lhs, rhs, params, output)
	return true
}

// execDotGeneralFastPathFloat32 is the fast path for float32 matrix multiplication.
// It directly operates on the input data without transposing to normalized form.
func execDotGeneralFastPathFloat32(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	outputFlat := output.flat.([]float32)

	batchSize := params.batchSize
	lhsCrossSize := params.lhsCrossSize   // M
	rhsCrossSize := params.rhsCrossSize   // N
	contractingSize := params.contractingSize // K

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// Use NEON for float32 dot products when available
	useNEON := hasNEON && contractingSize >= 4

	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		for m := 0; m < lhsCrossSize; m++ {
			lhsRowStart := lhsBaseIdx + m*contractingSize
			outputRowStart := outputBaseIdx + m*rhsCrossSize

			// Process 4 columns at a time using NEON Group4 if available
			n := 0
			if useNEON && rhsCrossSize >= 4 {
				for ; n+3 < rhsCrossSize; n += 4 {
					rhsCol0Start := rhsBaseIdx + n*contractingSize
					r0, r1, r2, r3 := dotProductGroup4_neon_asm(
						unsafe.Pointer(&lhsFlat[lhsRowStart]),
						unsafe.Pointer(&rhsFlat[rhsCol0Start]),
						int64(contractingSize),
						int64(contractingSize),
					)
					outputFlat[outputRowStart+n] = r0
					outputFlat[outputRowStart+n+1] = r1
					outputFlat[outputRowStart+n+2] = r2
					outputFlat[outputRowStart+n+3] = r3
				}
			}

			// Handle remaining columns
			for ; n < rhsCrossSize; n++ {
				rhsColStart := rhsBaseIdx + n*contractingSize
				var sum float32

				if useNEON {
					sum = dotProduct_neon_asm(
						unsafe.Pointer(&lhsFlat[lhsRowStart]),
						unsafe.Pointer(&rhsFlat[rhsColStart]),
						int64(contractingSize),
					)
				} else {
					// Scalar fallback with loop unrolling
					k := 0
					for ; k+7 < contractingSize; k += 8 {
						sum += lhsFlat[lhsRowStart+k]*rhsFlat[rhsColStart+k] +
							lhsFlat[lhsRowStart+k+1]*rhsFlat[rhsColStart+k+1] +
							lhsFlat[lhsRowStart+k+2]*rhsFlat[rhsColStart+k+2] +
							lhsFlat[lhsRowStart+k+3]*rhsFlat[rhsColStart+k+3] +
							lhsFlat[lhsRowStart+k+4]*rhsFlat[rhsColStart+k+4] +
							lhsFlat[lhsRowStart+k+5]*rhsFlat[rhsColStart+k+5] +
							lhsFlat[lhsRowStart+k+6]*rhsFlat[rhsColStart+k+6] +
							lhsFlat[lhsRowStart+k+7]*rhsFlat[rhsColStart+k+7]
					}
					for ; k < contractingSize; k++ {
						sum += lhsFlat[lhsRowStart+k] * rhsFlat[rhsColStart+k]
					}
				}

				outputFlat[outputRowStart+n] = sum
			}
		}
	}
}
