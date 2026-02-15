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

// execDenseHighway implements SIMD-accelerated dense layer: y = x @ W + b.
// Weight is [in_features, out_features]. We transpose it to [out, in] for
// compatibility with nn.DenseAuto which expects [out_features, in_features].
func execDenseHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	x := inputs[0]
	weight := inputs[1]
	var bias *simplego.Buffer
	if len(inputs) > 2 {
		bias = inputs[2]
	}

	output := simplego.FusedOpOutput(backend, node)

	inFeatures := x.Shape().Dimensions[x.Shape().Rank()-1]
	outFeatures := weight.Shape().Dimensions[1]
	batchSize := x.Shape().Size() / inFeatures

	switch x.DType() {
	case dtypes.Float32:
		var biasData []float32
		if bias != nil {
			biasData = bias.Flat().([]float32)
		}
		// Transpose weight from [in, out] to [out, in] for nn.DenseAuto.
		wTransposed := make([]float32, inFeatures*outFeatures)
		matmul.Transpose2D(weight.Flat().([]float32), inFeatures, outFeatures, wTransposed)
		nn.DenseAuto(hwyPool, x.Flat().([]float32), wTransposed, biasData, output.Flat().([]float32),
			batchSize, inFeatures, outFeatures)
	case dtypes.Float64:
		var biasData []float64
		if bias != nil {
			biasData = bias.Flat().([]float64)
		}
		wTransposed := make([]float64, inFeatures*outFeatures)
		matmul.Transpose2D(weight.Flat().([]float64), inFeatures, outFeatures, wTransposed)
		nn.DenseAuto(hwyPool, x.Flat().([]float64), wTransposed, biasData, output.Flat().([]float64),
			batchSize, inFeatures, outFeatures)
	default:
		return nil, errors.Errorf("highway Dense: unsupported dtype %s", x.DType())
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

	qDim, kvDim := simplego.QKVProjectionParams(node)
	totalOut := qDim + 2*kvDim

	qBuf, kBuf, vBuf := simplego.QKVProjectionOutputBuffers(backend, node)

	inFeatures := x.Shape().Dimensions[x.Shape().Rank()-1]
	batchSize := x.Shape().Size() / inFeatures

	var biasQ, biasK, biasV *simplego.Buffer
	biasIdx := 3
	if biasIdx < len(inputs) {
		biasQ = inputs[biasIdx]
		biasIdx++
	}
	if biasIdx < len(inputs) {
		biasK = inputs[biasIdx]
		biasIdx++
	}
	if biasIdx < len(inputs) {
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
// inputs layout: [x, packedWeights, scales, bias?]
func execQuantizedDenseHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	quantFormat, groupSize, outFeatures, act := simplego.QuantizedDenseParams(node)

	x := inputs[0]
	w := inputs[1]
	s := inputs[2]
	var bias *simplego.Buffer
	if len(inputs) > 3 {
		bias = inputs[3]
	}

	if x.DType() != dtypes.Float32 {
		return nil, errors.Errorf("highway QuantizedDense: only float32 input supported, got %s", x.DType())
	}

	output := simplego.FusedOpOutput(backend, node)
	xData := x.Flat().([]float32)
	scalesData := s.Flat().([]float32)
	outData := output.Flat().([]float32)

	K := x.Shape().Dimensions[x.Shape().Rank()-1]
	N := outFeatures
	M := x.Shape().Size() / K

	var biasData []float32
	if bias != nil {
		biasData = bias.Flat().([]float32)
	}

	matmulAct, actSupported := backendToMatmulActivation(act)

	switch quantFormat {
	case backends.QuantNF4:
		packed := w.Flat().([]uint8)
		if actSupported && matmulAct != matmul.ActNone {
			matmul.ParallelFusedNF4MatMulAct(hwyPool, xData, packed, scalesData, biasData, outData, M, K, N, groupSize, matmulAct)
		} else {
			matmul.ParallelFusedNF4MatMul(hwyPool, xData, packed, scalesData, biasData, outData, M, K, N, groupSize)
			simplego.ApplyActivationFloat32(backend, outData, act)
		}
	case backends.QuantInt4:
		packed := w.Flat().([]uint8)
		if actSupported && matmulAct != matmul.ActNone {
			matmul.ParallelFusedInt4MatMulAct(hwyPool, xData, packed, scalesData, biasData, outData, M, K, N, groupSize, matmulAct)
		} else {
			matmul.ParallelFusedInt4MatMul(hwyPool, xData, packed, scalesData, biasData, outData, M, K, N, groupSize)
			simplego.ApplyActivationFloat32(backend, outData, act)
		}
	case backends.QuantInt8:
		weights := w.Flat().([]int8)
		if actSupported && matmulAct != matmul.ActNone {
			matmul.ParallelFusedInt8MatMulAct(hwyPool, xData, weights, scalesData, biasData, outData, M, K, N, groupSize, matmulAct)
		} else {
			matmul.ParallelFusedInt8MatMul(hwyPool, xData, weights, scalesData, biasData, outData, M, K, N, groupSize)
			simplego.ApplyActivationFloat32(backend, outData, act)
		}
	default:
		return nil, errors.Errorf("highway QuantizedDense: unknown quant format %d", quantFormat)
	}

	return output, nil
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

	// Convert backends.ActivationType to nn.ActivationType.
	// Both enums use the same iota ordering.
	nnAct := nn.ActivationType(act)

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
