// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Exported helpers for subpackages (e.g. highway) to implement fused op executors.
// These extract the parameters from opaque node data and allocate output buffers,
// following the same pattern as UnaryOperandAndOutput.

// FusedOpOutput allocates an output buffer for a fused op based on the node's output shape.
func FusedOpOutput(backend *Backend, node *Node) *Buffer {
	return backend.getBufferForShape(node.shape)
}

// FusedOpOutputForShape allocates an output buffer for a given shape.
func FusedOpOutputForShape(backend *Backend, shape shapes.Shape) *Buffer {
	return backend.getBufferForShape(shape)
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

// QKVProjectionParams extracts the parameters from a FusedAttentionQKVProjection node.
func QKVProjectionParams(node *Node) (qDim, kvDim int, hasBiasQ, hasBiasK, hasBiasV bool) {
	data := node.data.(*nodeFusedAttentionQKVProjection)
	return data.qDim, data.kvDim, data.hasBiasQ, data.hasBiasK, data.hasBiasV
}

// QKVProjectionOutputBuffers allocates the three output buffers (q, k, v) for a QKVProjection node.
func QKVProjectionOutputBuffers(backend *Backend, node *Node) (q, k, v *Buffer) {
	outShapes := node.multiOutputsShapes
	return backend.getBufferForShape(outShapes[0]),
		backend.getBufferForShape(outShapes[1]),
		backend.getBufferForShape(outShapes[2])
}

// QuantizedSDPAParams extracts the parameters from a FusedQuantizedScaledDotProductAttention node.
func QuantizedSDPAParams(node *Node) (numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool) {
	data := node.data.(*nodeFusedQuantizedScaledDotProductAttention)
	return data.numHeads, data.numKVHeads, data.axesLayout, data.scale, data.causal
}

// QuantizedDenseParams extracts the parameters from a FusedQuantizedDense node.
func QuantizedDenseParams(node *Node) (quantFormat backends.QuantFormat, groupSize int, outFeatures int, activation backends.ActivationType) {
	data := node.data.(*nodeFusedQuantizedDense)
	return data.quantFormat, data.groupSize, data.outFeatures, data.activation
}

// TransposeBuffer transposes a buffer according to the given axis permutation.
// Used by the highway subpackage for transposing BSHD masks to BHSD layout.
func TransposeBuffer(backend *Backend, buf *Buffer, permutations []int) *Buffer {
	return transposeBuffer(backend, buf, permutations)
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
