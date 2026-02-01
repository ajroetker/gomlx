// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
)

func init() {
	simplego.SetNodeExecutor(backends.OpTypeFusedSoftmax, simplego.RegisterPriorityArch, execSoftmaxHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedGelu, simplego.RegisterPriorityArch, execGeluHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedLayerNorm, simplego.RegisterPriorityArch, execLayerNormHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedDense, simplego.RegisterPriorityArch, execDenseActivationHighway)
	simplego.SetNodeExecutor(backends.OpTypeFusedMultiHeadSDPA, simplego.RegisterPriorityArch, execMultiHeadSDPAHighway)
	simplego.SetMultiOutputsNodeExecutor(backends.OpTypeFusedQKVDense, simplego.RegisterPriorityArch, execQKVDenseHighway)
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
		for outer := 0; outer < outerSize; outer++ {
			off := outer * axisSize
			nn.Softmax(input[off:off+axisSize], output[off:off+axisSize])
		}
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
		for inner := 0; inner < innerSize; inner++ {
			rowOff := inner * axisSize
			nn.Softmax(tmp[rowOff:rowOff+axisSize], tmp[rowOff:rowOff+axisSize])
		}

		// Transpose back (innerSize × axisSize) → (axisSize × innerSize)
		matmul.Transpose2D(tmp, innerSize, axisSize, outBlock)
	}
}

// execGeluHighway implements SIMD-accelerated GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
func execGeluHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)

	switch input.DType() {
	case dtypes.Float32:
		activation.GELU(input.Flat().([]float32), output.Flat().([]float32))
	case dtypes.Float64:
		activation.GELU(input.Flat().([]float64), output.Flat().([]float64))
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
		nn.LayerNorm(input.Flat().([]float32), output.Flat().([]float32), normSize, gammaData, betaData, float32(epsilon))
	case dtypes.Float64:
		var gammaData, betaData []float64
		if gamma != nil {
			gammaData = gamma.Flat().([]float64)
		}
		if beta != nil {
			betaData = beta.Flat().([]float64)
		}
		nn.LayerNorm(input.Flat().([]float64), output.Flat().([]float64), normSize, gammaData, betaData, epsilon)
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

// execQKVDenseHighway implements SIMD-accelerated fused QKV projection.
// Weight wQKV is [inFeatures, totalOut] (ONNX convention). We transpose to [totalOut, inFeatures]
// for nn.QKVDenseAuto which uses MatMulKLastAuto (K-last / PyTorch convention).
func execQKVDenseHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) ([]*simplego.Buffer, error) {
	x := inputs[0]
	wQKV := inputs[1]

	qDim, kvDim := simplego.QKVDenseParams(node)
	totalOut := qDim + 2*kvDim

	qBuf, kBuf, vBuf := simplego.QKVDenseOutputBuffers(backend, node)

	inFeatures := x.Shape().Dimensions[x.Shape().Rank()-1]
	batchSize := x.Shape().Size() / inFeatures

	var biasQ, biasK, biasV *simplego.Buffer
	biasIdx := 2
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

// execMultiHeadSDPAHighway implements SIMD-accelerated multi-head scaled dot-product attention.
// q: [batch, numHeads, seqLen, headDim], k/v: [batch, numKVHeads, kvLen, headDim]
// mask: optional additive mask of rank 2–4 (broadcasting via strides)
// output: [batch, numHeads, seqLen, headDim]
func execMultiHeadSDPAHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	numHeads, numKVHeads, scale, causal := simplego.MultiHeadSDPAParams(node)
	q := inputs[0]
	k := inputs[1]
	v := inputs[2]
	var mask *simplego.Buffer
	if len(inputs) > 3 {
		mask = inputs[3]
	}
	output := simplego.FusedOpOutput(backend, node)

	batchSize := q.Shape().Dimensions[0]
	seqLen := q.Shape().Dimensions[2]
	kvLen := k.Shape().Dimensions[2]
	headDim := q.Shape().Dimensions[3]

	// Compute mask strides for broadcasting.
	var maskBatchStride, maskHeadStride int
	if mask != nil {
		maskBatchStride, maskHeadStride = computeMaskStrides(mask.Shape().Dimensions)
	}

	switch q.DType() {
	case dtypes.Float32:
		var maskData []float32
		if mask != nil {
			maskData = mask.Flat().([]float32)
		}
		nn.MultiHeadSDPAAuto(hwyPool,
			q.Flat().([]float32), k.Flat().([]float32), v.Flat().([]float32),
			maskData, output.Flat().([]float32),
			batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim,
			maskBatchStride, maskHeadStride,
			float32(scale), causal,
		)
	case dtypes.Float64:
		var maskData []float64
		if mask != nil {
			maskData = mask.Flat().([]float64)
		}
		nn.MultiHeadSDPAAuto(hwyPool,
			q.Flat().([]float64), k.Flat().([]float64), v.Flat().([]float64),
			maskData, output.Flat().([]float64),
			batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim,
			maskBatchStride, maskHeadStride,
			scale, causal,
		)
	default:
		return nil, errors.Errorf("highway MultiHeadSDPA: unsupported dtype %s", q.DType())
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
