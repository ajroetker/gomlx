// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Exported helpers for subpackages (e.g. highway) to implement fused op executors.
// These extract the parameters from opaque node data and allocate output buffers,
// following the same pattern as UnaryOperandAndOutput.

// FusedOpOutput allocates an output buffer for a fused op based on the node's output shape.
func FusedOpOutput(backend *Backend, node *Node) *Buffer {
	buf, err := backend.getBufferForShape(node.shape)
	if err != nil {
		panic(fmt.Sprintf("FusedOpOutput: %v", err))
	}
	return buf
}

// FusedOpOutputForShape allocates an output buffer for a given shape.
func FusedOpOutputForShape(backend *Backend, shape shapes.Shape) *Buffer {
	buf, err := backend.getBufferForShape(shape)
	if err != nil {
		panic(fmt.Sprintf("FusedOpOutputForShape: %v", err))
	}
	return buf
}

// FusedOpOutputShape returns the output shape for a fused op node.
func FusedOpOutputShape(node *Node) shapes.Shape {
	return node.shape
}

// MultiOutputShapes returns the output shapes for a multi-output node.
func MultiOutputShapes(node *Node) []shapes.Shape {
	return node.multiOutputsShapes
}

// SoftmaxParams extracts the axis from a Softmax node.
func SoftmaxParams(node *Node) (axis int) {
	return node.data.(*nodeFusedSoftmax).axis
}

// LayerNormParams extracts axes and epsilon from a LayerNorm node.
func LayerNormParams(node *Node) (axes []int, epsilon float64) {
	data := node.data.(*nodeFusedLayerNorm)
	return data.axes, data.epsilon
}

// DenseParams extracts the activation type from a FusedDense node.
func DenseParams(node *Node) backends.ActivationType {
	return node.data.(*nodeFusedDense).activation
}

// SDPAParams extracts the parameters from a FusedScaledDotProductAttention node.
func SDPAParams(node *Node) (numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool) {
	data := node.data.(*nodeFusedScaledDotProductAttention)
	return data.numHeads, data.numKVHeads, data.axesLayout, data.scale, data.causal
}

// SDPAQuantizedMatmuls returns whether quantized matmuls are requested for an SDPA node.
func SDPAQuantizedMatmuls(node *Node) bool {
	opts := node.data.(*nodeFusedScaledDotProductAttention).options
	return opts != nil && opts.QuantizedMatmuls
}

// QKVProjectionParams extracts the parameters from a FusedAttentionQKVProjection node.
func QKVProjectionParams(node *Node) (qDim, kvDim int, hasBiasQ, hasBiasK, hasBiasV bool) {
	data := node.data.(*nodeFusedAttentionQKVProjection)
	return data.qDim, data.kvDim, data.hasBiasQ, data.hasBiasK, data.hasBiasV
}

// QKVProjectionOutputBuffers allocates the three output buffers (q, k, v) for a QKVProjection node.
func QKVProjectionOutputBuffers(backend *Backend, node *Node) (q, k, v *Buffer) {
	outShapes := node.multiOutputsShapes
	var err error
	q, err = backend.getBufferForShape(outShapes[0])
	if err != nil {
		panic(fmt.Sprintf("QKVProjectionOutputBuffers (q): %v", err))
	}
	k, err = backend.getBufferForShape(outShapes[1])
	if err != nil {
		panic(fmt.Sprintf("QKVProjectionOutputBuffers (k): %v", err))
	}
	v, err = backend.getBufferForShape(outShapes[2])
	if err != nil {
		panic(fmt.Sprintf("QKVProjectionOutputBuffers (v): %v", err))
	}
	return q, k, v
}

// QuantizedDenseParams extracts the parameters from a FusedQuantizedDense node.
func QuantizedDenseParams(node *Node) (scheme backends.QuantizationScheme, blockSize int, activation backends.ActivationType, hasZeroPoint bool) {
	data := node.data.(*nodeFusedQuantizedDense)
	return data.scheme, data.blockSize, data.activation, data.hasZeroPoint
}

// QuantizedDenseGGMLParams extracts the GGML-specific parameters from a FusedQuantizedDense node.
// Only valid when scheme == QuantGGML.
func QuantizedDenseGGMLParams(node *Node) (ggmlType backends.GGMLQuantType, N, K int, activation backends.ActivationType, hasBias bool) {
	data := node.data.(*nodeFusedQuantizedDense)
	return data.ggmlType, data.ggmlN, data.ggmlK, data.activation, data.hasBias
}

// QuantizedGatherGGMLParams extracts the parameters from a FusedQuantizedGather node.
func QuantizedGatherGGMLParams(node *Node) (ggmlType backends.GGMLQuantType, K int) {
	data := node.data.(*nodeFusedQuantizedGather)
	return data.ggmlType, data.ggmlK
}

// TransposeBuffer transposes a buffer according to the given axis permutation.
// Used by the highway subpackage for transposing BSHD masks to BHSD layout.
func TransposeBuffer(backend *Backend, buf *Buffer, permutations []int) *Buffer {
	result, err := transposeBuffer(backend, buf, permutations)
	if err != nil {
		panic(fmt.Sprintf("TransposeBuffer: %v", err))
	}
	return result
}

// ApplyActivationFloat32 applies an activation function to float32 data in-place.
// Used by the highway subpackage for activations not directly supported by go-highway kernels.
func ApplyActivationFloat32(backend *Backend, data []float32, activation backends.ActivationType) {
	fusedDenseApplyActivation(backend, data, activation)
}

// LayerNormFloat32Fallback is the scalar implementation of LayerNorm for float32.
// Used by the highway subpackage for non-trailing axis combinations where SIMD
// acceleration is not applicable.
func LayerNormFloat32Fallback(input, output, gamma, beta *Buffer, axes []int, epsilon float64) {
	layerNorm[float32](input, output, gamma, beta, axes, epsilon)
}

// LayerNormFloat64Fallback is the scalar implementation of LayerNorm for float64.
func LayerNormFloat64Fallback(input, output, gamma, beta *Buffer, axes []int, epsilon float64) {
	layerNorm[float64](input, output, gamma, beta, axes, epsilon)
}
