// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"math"

	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

func init() {
	simplego.SetNodeExecutor(backends.OpTypeFusedSoftmax, simplego.RegisterPriorityArch, execSoftmaxHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedGelu, simplego.RegisterPriorityArch, execGeluHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedLayerNorm, simplego.RegisterPriorityArch, execLayerNormHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedDense, simplego.RegisterPriorityArch, execDenseActivationHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedScaledDotProductAttention, simplego.RegisterPriorityArch, execSDPAHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedQuantizedDense, simplego.RegisterPriorityArch, execQuantizedDenseHighway)
	simplego.SetMultiOutputsNodeExecutor(backends.OpTypeFusedAttentionQKVProjection, simplego.RegisterPriorityArch, execQKVProjectionHighway)
}

// computeAxisStrides decomposes a shape into (outerSize, axisSize, innerSize) for
// iterating over a single axis. This matches the decomposition in exec_fused_ops.go.
func computeAxisStrides(dims []int, axis int) (outerSize, axisSize, innerSize int) {
	outerSize = 1
	for i := 0; i < axis; i++ {
		outerSize *= dims[i]
	}
	axisSize = dims[axis]
	innerSize = 1
	for i := axis + 1; i < len(dims); i++ {
		innerSize *= dims[i]
	}
	return
}

// rowColDecomposition returns (rows, cols) from a shape by treating the last
// dimension as cols and collapsing all leading dimensions into rows. This
// provides a natural decomposition for parallelizing element-wise operations.
func rowColDecomposition(s shapes.Shape) (rows, cols int) {
	if s.Rank() == 0 {
		return 1, 1
	}
	cols = s.Dimensions[s.Rank()-1]
	rows = s.Size() / cols
	if rows == 0 {
		rows = 1
	}
	return
}

// execSoftmaxHighway implements SIMD-accelerated softmax.
//
// For last-axis softmax (innerSize == 1), each contiguous chunk of axisSize
// elements is passed directly to nn.Softmax for full SIMD acceleration.
//
// For non-last-axis softmax (innerSize > 1), we transpose each outer block
// to make the softmax axis contiguous, apply nn.Softmax, then transpose back.
// Both transpose passes use SIMD (matmul.Transpose2D).
func execSoftmaxHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	axis := simplego.SoftmaxParams(node)
	input := inputs[0]
	output := simplego.FusedOpOutput(backend, node)
	shape := simplego.FusedOpOutputShape(node)

	switch input.DType() {
	case dtypes.Float32:
		softmaxHighway(input.Flat().([]float32), output.Flat().([]float32), axis, shape.Dimensions)
	case dtypes.Float64:
		softmaxHighway(input.Flat().([]float64), output.Flat().([]float64), axis, shape.Dimensions)
	default:
		return nil, errors.Errorf("highway Softmax: unsupported dtype %s", input.DType())
	}
	return output, nil
}

func softmaxHighway[T interface{ ~float32 | ~float64 }](input, output []T, axis int, dims []int) {
	outerSize, axisSize, innerSize := computeAxisStrides(dims, axis)

	if innerSize == 1 {
		// Fast path: softmax along last axis. Each contiguous chunk is one softmax group.
		nn.ParallelSoftmax(hwyPool, input, output, outerSize, axisSize)
		return
	}

	// Non-last axis: transpose to make the softmax axis contiguous,
	// apply softmax, then transpose back.
	blockSize := axisSize * innerSize
	tmp := make([]T, blockSize)

	for outer := 0; outer < outerSize; outer++ {
		off := outer * blockSize
		inBlock := input[off : off+blockSize]
		outBlock := output[off : off+blockSize]

		// Transpose (axisSize × innerSize) → (innerSize × axisSize)
		matmul.Transpose2D(inBlock, axisSize, innerSize, tmp)

		// Softmax over each contiguous row of axisSize elements
		nn.ParallelSoftmax(hwyPool, tmp, tmp, innerSize, axisSize)

		// Transpose back (innerSize × axisSize) → (axisSize × innerSize)
		matmul.Transpose2D(tmp, innerSize, axisSize, outBlock)
	}
}

// execGeluHighway implements SIMD-accelerated GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
// Uses row-parallel execution for large tensors.
func execGeluHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)
	rows, cols := rowColDecomposition(input.Shape())

	switch input.DType() {
	case dtypes.Float32:
		activation.ParallelGELU(hwyPool, input.Flat().([]float32), output.Flat().([]float32), rows, cols)
	case dtypes.Float64:
		activation.ParallelGELU(hwyPool, input.Flat().([]float64), output.Flat().([]float64), rows, cols)
	default:
		return nil, errors.Errorf("highway Gelu: unsupported dtype %s", input.DType())
	}
	return output, nil
}

// execLayerNormHighway implements SIMD-accelerated layer normalization.
//
// For trailing axes (the common case in transformers), delegates to nn.LayerNorm
// which uses SIMD for all three passes (mean, variance, normalize).
//
// For non-trailing axes (rare), falls back to the scalar implementation.
func execLayerNormHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	axes, epsilon := simplego.LayerNormParams(node)
	input := inputs[0]
	output := simplego.FusedOpOutput(backend, node)
	shape := input.Shape()

	var gamma, beta *simplego.Buffer
	if len(inputs) > 1 {
		gamma = inputs[1]
	}
	if len(inputs) > 2 {
		beta = inputs[2]
	}

	// Check if axes are trailing (contiguous from the end).
	rank := len(shape.Dimensions)
	isTrailingAxes := true
	for i, a := range axes {
		if a != rank-len(axes)+i {
			isTrailingAxes = false
			break
		}
	}

	if !isTrailingAxes {
		// Non-trailing axes: fall back to scalar implementation.
		switch input.DType() {
		case dtypes.Float32:
			simplego.LayerNormFloat32Fallback(input, output, gamma, beta, axes, epsilon)
		case dtypes.Float64:
			simplego.LayerNormFloat64Fallback(input, output, gamma, beta, axes, epsilon)
		default:
			return nil, errors.Errorf("highway LayerNorm: unsupported dtype %s", input.DType())
		}
		return output, nil
	}

	// Trailing axes: compute normSize and delegate to SIMD LayerNorm.
	normSize := 1
	for _, a := range axes {
		normSize *= shape.Dimensions[a]
	}

	switch input.DType() {
	case dtypes.Float32:
		var gammaData, betaData []float32
		if gamma != nil {
			gammaData = gamma.Flat().([]float32)
		}
		if beta != nil {
			betaData = beta.Flat().([]float32)
		}
		nn.ParallelLayerNorm(hwyPool, input.Flat().([]float32), output.Flat().([]float32), normSize, gammaData, betaData, float32(epsilon))
	case dtypes.Float64:
		var gammaData, betaData []float64
		if gamma != nil {
			gammaData = gamma.Flat().([]float64)
		}
		if beta != nil {
			betaData = beta.Flat().([]float64)
		}
		nn.ParallelLayerNorm(hwyPool, input.Flat().([]float64), output.Flat().([]float64), normSize, gammaData, betaData, epsilon)
	default:
		return nil, errors.Errorf("highway LayerNorm: unsupported dtype %s", input.DType())
	}
	return output, nil
}

// execQKVProjectionHighway implements SIMD-accelerated fused QKV projection.
// Weight wQKV is [inFeatures, totalOut] (ONNX convention). We transpose to [totalOut, inFeatures]
// for nn.QKVDenseAuto which uses MatMulKLastAuto (K-last / PyTorch convention).
func execQKVProjectionHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) ([]*simplego.Buffer, error) {
	// inputs layout: [dotResult, x, wQKV, biasQ?, biasK?, biasV?]
	// We ignore dotResult (inputs[0]) and redo the fused matmul+split+bias from x and wQKV.
	x := inputs[1]
	wQKV := inputs[2]

	qDim, kvDim, hasBiasQ, hasBiasK, hasBiasV := simplego.QKVProjectionParams(node)
	totalOut := qDim + 2*kvDim

	qBuf, kBuf, vBuf := simplego.QKVProjectionOutputBuffers(backend, node)

	inFeatures := x.Shape().Dimensions[x.Shape().Rank()-1]
	batchSize := x.Shape().Size() / inFeatures

	// Use flag-based indexing (matching the scalar fallback) to correctly
	// handle partially-specified biases.
	var biasQ, biasK, biasV *simplego.Buffer
	biasIdx := 3
	if hasBiasQ {
		biasQ = inputs[biasIdx]
		biasIdx++
	}
	if hasBiasK {
		biasK = inputs[biasIdx]
		biasIdx++
	}
	if hasBiasV {
		biasV = inputs[biasIdx]
	}

	switch x.DType() {
	case dtypes.Float32:
		// Transpose wQKV from [inFeatures, totalOut] to [totalOut, inFeatures].
		wTransposed := make([]float32, inFeatures*totalOut)
		matmul.Transpose2D(wQKV.Flat().([]float32), inFeatures, totalOut, wTransposed)

		var bqData, bkData, bvData []float32
		if biasQ != nil {
			bqData = biasQ.Flat().([]float32)
		}
		if biasK != nil {
			bkData = biasK.Flat().([]float32)
		}
		if biasV != nil {
			bvData = biasV.Flat().([]float32)
		}
		nn.QKVDenseAuto(hwyPool,
			x.Flat().([]float32), wTransposed,
			bqData, bkData, bvData,
			qBuf.Flat().([]float32), kBuf.Flat().([]float32), vBuf.Flat().([]float32),
			batchSize, inFeatures, qDim, kvDim,
		)
	case dtypes.Float64:
		wTransposed := make([]float64, inFeatures*totalOut)
		matmul.Transpose2D(wQKV.Flat().([]float64), inFeatures, totalOut, wTransposed)

		var bqData, bkData, bvData []float64
		if biasQ != nil {
			bqData = biasQ.Flat().([]float64)
		}
		if biasK != nil {
			bkData = biasK.Flat().([]float64)
		}
		if biasV != nil {
			bvData = biasV.Flat().([]float64)
		}
		nn.QKVDenseAuto(hwyPool,
			x.Flat().([]float64), wTransposed,
			bqData, bkData, bvData,
			qBuf.Flat().([]float64), kBuf.Flat().([]float64), vBuf.Flat().([]float64),
			batchSize, inFeatures, qDim, kvDim,
		)
	default:
		return nil, errors.Errorf("highway QKVDense: unsupported dtype %s", x.DType())
	}
	return []*simplego.Buffer{qBuf, kBuf, vBuf}, nil
}

// computeMaskStrides returns (batchStride, headStride) for indexing into a mask
// tensor based on its rank. Dimensions of size 1 are broadcast (stride 0).
func computeMaskStrides(dims []int) (batchStride, headStride int) {
	switch len(dims) {
	case 2:
		return 0, 0
	case 3:
		if dims[0] <= 1 {
			return 0, 0
		}
		return dims[1] * dims[2], 0
	case 4:
		if dims[0] > 1 {
			batchStride = dims[1] * dims[2] * dims[3]
		}
		if dims[1] > 1 {
			headStride = dims[2] * dims[3]
		}
		return batchStride, headStride
	default:
		return 0, 0
	}
}

// execSDPAHighway implements SIMD-accelerated multi-head scaled dot-product attention.
//
// Both BHSD [batch, heads, seq, dim] and BSHD [batch, seq, heads, dim] layouts are
// supported via nn.MultiHeadSDPAStridedAuto. For BHSD the strided API fast-paths to
// the contiguous kernel with zero overhead. For BSHD it gathers each head into a
// contiguous temp buffer, runs the optimized single-head SDPA, and scatters back.
func execSDPAHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	if simplego.SDPAQuantizedMatmuls(node) {
		return execQuantizedSDPAHighway(backend, node, inputs, inputsOwned)
	}
	numHeads, numKVHeads, axesLayout, scale, causal := simplego.SDPAParams(node)

	q := inputs[0]
	k := inputs[1]
	v := inputs[2]
	var mask *simplego.Buffer
	if len(inputs) > 3 {
		mask = inputs[3]
	}

	// For rank-4 BSHD masks [batch, seq, heads, kvLen], transpose to BHSD
	// [batch, heads, seq, kvLen] so per-head mask data is contiguous.
	if axesLayout == backends.AxesLayoutBSHD && mask != nil && mask.Shape().Rank() == 4 {
		mask = simplego.TransposeBuffer(backend, mask, []int{0, 2, 1, 3})
	}

	output := simplego.FusedOpOutput(backend, node)

	// Compute layout-dependent strides.
	dims := q.Shape().Dimensions
	batchSize := dims[0]
	var seqLen, kvLen, headDim int
	var qBatchStride, qHeadStride, qSeqStride int
	var kvBatchStride, kvHeadStride, kvSeqStride int

	if axesLayout == backends.AxesLayoutBSHD {
		// [batch, seq, heads, dim]
		seqLen = dims[1]
		headDim = dims[3]
		kvDims := k.Shape().Dimensions
		kvLen = kvDims[1]
		qSeqStride = numHeads * headDim
		kvSeqStride = numKVHeads * headDim
		qHeadStride = headDim
		kvHeadStride = headDim
		qBatchStride = seqLen * numHeads * headDim
		kvBatchStride = kvLen * numKVHeads * headDim
	} else {
		// BHSD: [batch, heads, seq, dim]
		seqLen = dims[2]
		headDim = dims[3]
		kvDims := k.Shape().Dimensions
		kvLen = kvDims[2]
		qSeqStride = headDim
		kvSeqStride = headDim
		qHeadStride = seqLen * headDim
		kvHeadStride = kvLen * headDim
		qBatchStride = numHeads * seqLen * headDim
		kvBatchStride = numKVHeads * kvLen * headDim
	}

	// Compute mask strides for broadcasting (mask is always in BHSD convention after transpose).
	var maskBatchStride, maskHeadStride int
	if mask != nil {
		maskBatchStride, maskHeadStride = computeMaskStrides(mask.Shape().Dimensions)
	}

	switch q.DType() {
	case dtypes.Float32:
		var maskData []float32
		if mask != nil {
			if mask.Shape().DType == dtypes.Bool {
				maskData = boolToAdditiveMask[float32](mask.Flat().([]bool))
			} else {
				maskData = mask.Flat().([]float32)
			}
		}
		nn.MultiHeadSDPAStridedAuto(hwyPool,
			q.Flat().([]float32), k.Flat().([]float32), v.Flat().([]float32),
			maskData, output.Flat().([]float32),
			batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim,
			qBatchStride, qHeadStride, qSeqStride,
			kvBatchStride, kvHeadStride, kvSeqStride,
			maskBatchStride, maskHeadStride,
			float32(scale), causal,
		)
	case dtypes.Float64:
		var maskData []float64
		if mask != nil {
			if mask.Shape().DType == dtypes.Bool {
				maskData = boolToAdditiveMask[float64](mask.Flat().([]bool))
			} else {
				maskData = mask.Flat().([]float64)
			}
		}
		nn.MultiHeadSDPAStridedAuto(hwyPool,
			q.Flat().([]float64), k.Flat().([]float64), v.Flat().([]float64),
			maskData, output.Flat().([]float64),
			batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim,
			qBatchStride, qHeadStride, qSeqStride,
			kvBatchStride, kvHeadStride, kvSeqStride,
			maskBatchStride, maskHeadStride,
			scale, causal,
		)
	default:
		return nil, errors.Errorf("highway SDPA: unsupported dtype %s", q.DType())
	}
	return output, nil
}

// execQuantizedSDPAHighway implements SIMD-accelerated multi-head quantized SDPA.
// Inputs are float32 Q/K/V; the go-highway kernel internally quantizes to uint8 for
// int8×int8 matmuls (Q@K^T and attn@V), then dequantizes the output back to float32.
func execQuantizedSDPAHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	numHeads, numKVHeads, axesLayout, scale, causal := simplego.SDPAParams(node)

	q := inputs[0]
	k := inputs[1]
	v := inputs[2]
	var mask *simplego.Buffer
	if len(inputs) > 3 {
		mask = inputs[3]
	}

	if axesLayout == backends.AxesLayoutBSHD && mask != nil && mask.Shape().Rank() == 4 {
		mask = simplego.TransposeBuffer(backend, mask, []int{0, 2, 1, 3})
	}

	output := simplego.FusedOpOutput(backend, node)

	dims := q.Shape().Dimensions
	batchSize := dims[0]
	var seqLen, kvLen, headDim int
	var qBatchStride, qHeadStride, qSeqStride int
	var kvBatchStride, kvHeadStride, kvSeqStride int

	if axesLayout == backends.AxesLayoutBSHD {
		seqLen = dims[1]
		headDim = dims[3]
		kvDims := k.Shape().Dimensions
		kvLen = kvDims[1]
		qSeqStride = numHeads * headDim
		kvSeqStride = numKVHeads * headDim
		qHeadStride = headDim
		kvHeadStride = headDim
		qBatchStride = seqLen * numHeads * headDim
		kvBatchStride = kvLen * numKVHeads * headDim
	} else {
		seqLen = dims[2]
		headDim = dims[3]
		kvDims := k.Shape().Dimensions
		kvLen = kvDims[2]
		qSeqStride = headDim
		kvSeqStride = headDim
		qHeadStride = seqLen * headDim
		kvHeadStride = kvLen * headDim
		qBatchStride = numHeads * seqLen * headDim
		kvBatchStride = numKVHeads * kvLen * headDim
	}

	var maskBatchStride, maskHeadStride int
	if mask != nil {
		maskBatchStride, maskHeadStride = computeMaskStrides(mask.Shape().Dimensions)
	}

	switch q.DType() {
	case dtypes.Float32:
		var maskData []float32
		if mask != nil {
			if mask.Shape().DType == dtypes.Bool {
				maskData = boolToAdditiveMask[float32](mask.Flat().([]bool))
			} else {
				maskData = mask.Flat().([]float32)
			}
		}
		nn.MultiHeadQuantizedSDPAStrided(hwyPool,
			q.Flat().([]float32), k.Flat().([]float32), v.Flat().([]float32),
			maskData, output.Flat().([]float32),
			batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim,
			qBatchStride, qHeadStride, qSeqStride,
			kvBatchStride, kvHeadStride, kvSeqStride,
			maskBatchStride, maskHeadStride,
			float32(scale), causal,
		)
	default:
		return nil, errors.Errorf("highway QuantizedSDPA: unsupported dtype %s (only float32 supported)", q.DType())
	}
	return output, nil
}

// boolToAdditiveMask converts a boolean mask to an additive float mask.
// true (attend) → 0, false (mask out) → -inf.
func boolToAdditiveMask[T ~float32 | ~float64](boolMask []bool) []T {
	out := make([]T, len(boolMask))
	negInf := T(math.Inf(-1))
	for i, v := range boolMask {
		if !v {
			out[i] = negInf
		}
	}
	return out
}

// backendToNNActivation converts a backends.ActivationType to the go-highway nn.ActivationType.
func backendToNNActivation(act backends.ActivationType) nn.ActivationType {
	switch act {
	case backends.ActivationNone:
		return nn.ActivationNone
	case backends.ActivationGelu:
		return nn.ActivationGelu
	case backends.ActivationRelu:
		return nn.ActivationRelu
	case backends.ActivationSilu:
		return nn.ActivationSilu
	case backends.ActivationHardSwish:
		return nn.ActivationHardSwish
	case backends.ActivationTanh:
		return nn.ActivationTanh
	default:
		return nn.ActivationNone
	}
}

// backendToMatmulActivation converts a backends.ActivationType to the go-highway matmul.ActivationType.
// Returns (matmulAct, ok). ok is false if the activation has no direct matmul equivalent (e.g. Tanh).
func backendToMatmulActivation(act backends.ActivationType) (matmul.ActivationType, bool) {
	switch act {
	case backends.ActivationNone:
		return matmul.ActNone, true
	case backends.ActivationSilu:
		return matmul.ActSiLU, true
	case backends.ActivationGelu:
		return matmul.ActGELU, true
	case backends.ActivationRelu:
		return matmul.ActReLU, true
	default:
		return matmul.ActNone, false
	}
}

// execQuantizedDenseHighway implements SIMD-accelerated fused quantized dense.
// inputs layout: [x, weights, scales, zeroPoints?, bias?]
func execQuantizedDenseHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	scheme, blockSize, act, hasZeroPoint := simplego.QuantizedDenseParams(node)

	x := inputs[0]
	w := inputs[1]
	s := inputs[2]

	// Determine bias from remaining inputs using hasZeroPoint flag.
	var bias *simplego.Buffer
	nextIdx := 3
	if hasZeroPoint {
		nextIdx++ // skip zeroPoints (not used by highway kernels)
	}
	if nextIdx < len(inputs) {
		bias = inputs[nextIdx]
	}

	if x.DType() != dtypes.Float32 {
		return nil, errors.Errorf("highway QuantizedDense: only float32 input supported, got %s", x.DType())
	}

	output := simplego.FusedOpOutput(backend, node)
	xData := x.Flat().([]float32)
	scalesData := s.Flat().([]float32)
	outData := output.Flat().([]float32)

	K := x.Shape().Dimensions[x.Shape().Rank()-1]
	outShape := simplego.FusedOpOutputShape(node)
	N := outShape.Dimensions[outShape.Rank()-1]
	M := x.Shape().Size() / K

	var biasData []float32
	if bias != nil {
		biasData = bias.Flat().([]float32)
	}

	matmulAct, actSupported := backendToMatmulActivation(act)

	switch scheme {
	case backends.QuantNF4:
		// Highway NF4 kernel expects packed nibbles (2 per byte).
		// After Bitcast, data is unpacked (one nibble per uint8), so re-pack.
		packed := packNibbles(w.Flat().([]uint8))
		if actSupported && matmulAct != matmul.ActNone {
			matmul.ParallelFusedNF4MatMulAct(hwyPool, xData, packed, scalesData, biasData, outData, M, K, N, blockSize, matmulAct)
		} else {
			matmul.ParallelFusedNF4MatMul(hwyPool, xData, packed, scalesData, biasData, outData, M, K, N, blockSize)
			simplego.ApplyActivationFloat32(backend, outData, act)
		}
	case backends.QuantLinear:
		wDType := w.DType()
		switch wDType {
		case dtypes.Int4, dtypes.Uint4:
			// Highway Int4 kernel expects packed nibbles (2 per byte).
			packed := packNibbles(w.Flat().([]uint8))
			if actSupported && matmulAct != matmul.ActNone {
				matmul.ParallelFusedInt4MatMulAct(hwyPool, xData, packed, scalesData, biasData, outData, M, K, N, blockSize, matmulAct)
			} else {
				matmul.ParallelFusedInt4MatMul(hwyPool, xData, packed, scalesData, biasData, outData, M, K, N, blockSize)
				simplego.ApplyActivationFloat32(backend, outData, act)
			}
		case dtypes.Int8:
			weights := w.Flat().([]int8)
			if actSupported && matmulAct != matmul.ActNone {
				matmul.ParallelFusedInt8MatMulAct(hwyPool, xData, weights, scalesData, biasData, outData, M, K, N, blockSize, matmulAct)
			} else {
				matmul.ParallelFusedInt8MatMul(hwyPool, xData, weights, scalesData, biasData, outData, M, K, N, blockSize)
				simplego.ApplyActivationFloat32(backend, outData, act)
			}
		default:
			return nil, errors.Errorf("highway QuantizedDense: QuantLinear unsupported weight dtype %s", wDType)
		}
	default:
		return nil, errors.Errorf("highway QuantizedDense: unknown quantization scheme %d", scheme)
	}

	return output, nil
}

// packNibbles re-packs unpacked nibble data (one value per uint8) into packed form
// (two nibbles per byte, low nibble first). This is needed because the Bitcast
// executor unpacks nibbles, but the go-highway matmul kernels expect packed data.
func packNibbles(unpacked []uint8) []uint8 {
	n := len(unpacked)
	packed := make([]uint8, (n+1)/2)
	for i := 0; i < n-1; i += 2 {
		packed[i/2] = (unpacked[i] & 0x0F) | (unpacked[i+1] << 4)
	}
	if n%2 != 0 {
		packed[n/2] = unpacked[n-1] & 0x0F
	}
	return packed
}

// execDenseActivationHighway implements SIMD-accelerated dense + activation: y = act(x @ W + b).
// Delegates to nn.DenseActivationAuto which fuses the matmul, bias add, and activation.
// inputs layout: [dotResult, x, weight, bias?]
// We ignore dotResult (inputs[0]) and redo the fused matmul+bias+activation from x and weight.
func execDenseActivationHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	x := inputs[1]
	weight := inputs[2]
	var bias *simplego.Buffer
	if len(inputs) > 3 {
		bias = inputs[3]
	}

	output := simplego.FusedOpOutput(backend, node)
	act := simplego.DenseParams(node)

	inFeatures := x.Shape().Dimensions[x.Shape().Rank()-1]
	outFeatures := weight.Shape().Dimensions[1]
	batchSize := x.Shape().Size() / inFeatures

	nnAct := backendToNNActivation(act)

	switch x.DType() {
	case dtypes.Float32:
		var biasData []float32
		if bias != nil {
			biasData = bias.Flat().([]float32)
		}
		// Transpose weight from [in, out] to [out, in] for nn.DenseActivationAuto.
		wTransposed := make([]float32, inFeatures*outFeatures)
		matmul.Transpose2D(weight.Flat().([]float32), inFeatures, outFeatures, wTransposed)
		nn.DenseActivationAuto(hwyPool, x.Flat().([]float32), wTransposed, biasData, output.Flat().([]float32),
			batchSize, inFeatures, outFeatures, nnAct)
	case dtypes.Float64:
		var biasData []float64
		if bias != nil {
			biasData = bias.Flat().([]float64)
		}
		wTransposed := make([]float64, inFeatures*outFeatures)
		matmul.Transpose2D(weight.Flat().([]float64), inFeatures, outFeatures, wTransposed)
		nn.DenseActivationAuto(hwyPool, x.Flat().([]float64), wTransposed, biasData, output.Flat().([]float64),
			batchSize, inFeatures, outFeatures, nnAct)
	default:
		return nil, errors.Errorf("highway DenseActivation: unsupported dtype %s", x.DType())
	}
	return output, nil
}
