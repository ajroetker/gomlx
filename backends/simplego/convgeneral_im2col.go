// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeConvGeneral, priorityArch, execConvGeneralIm2Col)
}

// canUseIm2Col checks whether the convolution can be lowered to im2col + MatMul.
// Requirements:
//   - Highway SIMD matmul is registered and supports the dtype
//   - No batch grouping (rare, not worth the complexity)
//   - No input dilations (transposed conv; rare in inference)
//   - Standard NCHW axis layout: batch=0, channels=1, spatial=[2,3,...],
//     kernel output channels=0, kernel input channels=1, kernel spatial=[2,3,...]
//
// Channel grouping (channelGroupCount > 1) is supported: the convolution is
// split into independent groups, each processed with im2col + MatMul.
func canUseIm2Col(params *convNode, dtype dtypes.DType) bool {
	if !IsHighwayRegistered() || !Highway.HasDTypeSupport(dtype, dtype) {
		return false
	}
	if params.batchGroupCount > 1 {
		return false
	}
	if params.hasInputDilations {
		return false
	}

	// Require standard NCHW layout.
	axes := params.axes
	if axes.InputBatch != 0 || axes.InputChannels != 1 {
		return false
	}
	if axes.KernelOutputChannels != 0 || axes.KernelInputChannels != 1 {
		return false
	}
	if axes.OutputBatch != 0 || axes.OutputChannels != 1 {
		return false
	}
	spatialRank := len(axes.InputSpatial)
	for i := 0; i < spatialRank; i++ {
		if axes.InputSpatial[i] != i+2 || axes.KernelSpatial[i] != i+2 || axes.OutputSpatial[i] != i+2 {
			return false
		}
	}

	return true
}

// execConvGeneralIm2Col performs convolution via im2col + SIMD MatMul.
// Falls back to the generic execConvGeneral when im2col is not applicable.
//
// For grouped convolution (channelGroupCount > 1), the convolution is split
// into G independent groups. Group g processes input channels
// [g*C_in/G, (g+1)*C_in/G) with kernel filters [g*C_out/G, (g+1)*C_out/G)
// to produce output channels [g*C_out/G, (g+1)*C_out/G).
func execConvGeneralIm2Col(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	params := node.data.(*convNode)
	dtype := input.shape.DType

	if !canUseIm2Col(params, dtype) {
		return execConvGeneral(backend, node, inputs, inputsOwned)
	}

	outputShape := node.shape
	inputShape := input.shape
	kernelShape := kernel.shape

	spatialRank := len(params.axes.InputSpatial)
	batchSize := inputShape.Dimensions[0]
	inputChannels := inputShape.Dimensions[1]
	outputChannels := kernelShape.Dimensions[0]

	groups := params.channelGroupCount
	if groups < 1 {
		groups = 1
	}
	inputChannelsPerGroup := inputChannels / groups
	outputChannelsPerGroup := outputChannels / groups

	// Compute output spatial dimensions.
	outputSpatialSize := 1
	for i := 0; i < spatialRank; i++ {
		outputSpatialSize *= outputShape.Dimensions[i+2]
	}

	// patchSize = inputChannelsPerGroup * product(kernelSpatial).
	// Note: kernelShape.Dimensions[1] == inputChannelsPerGroup for grouped conv.
	patchSize := inputChannelsPerGroup
	for i := 0; i < spatialRank; i++ {
		patchSize *= kernelShape.Dimensions[i+2]
	}

	m := batchSize * outputSpatialSize // rows of the im2col matrix
	k := patchSize                     // columns (contracting dimension)

	// Allocate im2col buffer: [M, K]. Reused across groups.
	im2colBuf, err := backend.getBuffer(dtype, m*k)
	if err != nil {
		return nil, errors.Wrapf(err, "failed allocating im2col buffer of size %d", m*k)
	}

	// Allocate per-group matmul output: [M, outputChannelsPerGroup]. Reused across groups.
	matmulOutBuf, err := backend.getBuffer(dtype, m*outputChannelsPerGroup)
	if err != nil {
		backend.putBuffer(im2colBuf)
		return nil, errors.Wrap(err, "failed allocating matmul output buffer")
	}

	// Allocate final output buffer: [batchSize, outputChannels, outputSpatial...].
	output, err := backend.getBufferForShape(outputShape)
	if err != nil {
		backend.putBuffer(im2colBuf)
		backend.putBuffer(matmulOutBuf)
		return nil, errors.Wrapf(err, "failed allocating output buffer shaped %s", outputShape)
	}

	for g := 0; g < groups; g++ {
		inputChannelOffset := g * inputChannelsPerGroup

		// im2col transform for this group's input channels.
		switch dtype {
		case dtypes.Float32:
			im2colTransform(
				input.flat.([]float32), im2colBuf.flat.([]float32),
				inputShape, kernelShape, params,
				batchSize, inputChannels, inputChannelsPerGroup, inputChannelOffset,
				outputSpatialSize, patchSize, spatialRank,
			)
		case dtypes.Float64:
			im2colTransform(
				input.flat.([]float64), im2colBuf.flat.([]float64),
				inputShape, kernelShape, params,
				batchSize, inputChannels, inputChannelsPerGroup, inputChannelOffset,
				outputSpatialSize, patchSize, spatialRank,
			)
		default:
			backend.putBuffer(im2colBuf)
			backend.putBuffer(matmulOutBuf)
			backend.putBuffer(output)
			return execConvGeneral(backend, node, inputs, inputsOwned)
		}

		// Kernel slice for group g: rows [g*outputChannelsPerGroup, (g+1)*outputChannelsPerGroup).
		// Full kernel layout: [outputChannels, inputChannelsPerGroup, kSpatial...].
		// Each filter has patchSize elements, so group g starts at g*outputChannelsPerGroup*patchSize.
		kernelGroupOffset := g * outputChannelsPerGroup * patchSize
		kernelGroupEnd := kernelGroupOffset + outputChannelsPerGroup*patchSize

		var kernelSlice interface{}
		switch dtype {
		case dtypes.Float32:
			kernelSlice = kernel.flat.([]float32)[kernelGroupOffset:kernelGroupEnd]
		case dtypes.Float64:
			kernelSlice = kernel.flat.([]float64)[kernelGroupOffset:kernelGroupEnd]
		}

		// MatMul: im2col [M, K] × kernelGroup^T [outputChannelsPerGroup, K]^T = [M, outputChannelsPerGroup].
		err := Highway.MatMulKLast(dtype, dtype,
			im2colBuf.flat, kernelSlice,
			1, m, outputChannelsPerGroup, k,
			matmulOutBuf.flat)
		if err != nil {
			backend.putBuffer(im2colBuf)
			backend.putBuffer(matmulOutBuf)
			backend.putBuffer(output)
			return nil, errors.Wrapf(err, "im2col matmul failed for group %d", g)
		}

		// Transpose this group's matmul output into the correct position in the NCHW output.
		// matmulOutBuf: [batchSize*outputSpatialSize, outputChannelsPerGroup]
		//   = [batchSize, outputSpatialSize, outputChannelsPerGroup] (row-major, NHWC-like).
		// Output: [batchSize, outputChannels, outputSpatialSize] (NCHW).
		// This group writes to channels [g*outputChannelsPerGroup, (g+1)*outputChannelsPerGroup).
		outputChannelOffset := g * outputChannelsPerGroup
		switch dtype {
		case dtypes.Float32:
			transposeNHWCtoNCHWGroup(
				matmulOutBuf.flat.([]float32), output.flat.([]float32),
				batchSize, outputSpatialSize, outputChannelsPerGroup,
				outputChannels, outputChannelOffset)
		case dtypes.Float64:
			transposeNHWCtoNCHWGroup(
				matmulOutBuf.flat.([]float64), output.flat.([]float64),
				batchSize, outputSpatialSize, outputChannelsPerGroup,
				outputChannels, outputChannelOffset)
		}
	}

	backend.putBuffer(im2colBuf)
	backend.putBuffer(matmulOutBuf)

	return output, nil
}

// im2colTransform performs the im2col transformation for float32 or float64 data.
// Rearranges input patches into a column matrix suitable for matrix multiplication.
//
// Input:  [batchSize, totalInputChannels, inputSpatial...]
// Output: [batchSize * outputSpatialSize, patchSize]
//
// For grouped convolution, only channels [inputChannelOffset, inputChannelOffset+groupInputChannels)
// are extracted per patch. patchSize = groupInputChannels * product(kernelSpatial).
func im2colTransform[T float32 | float64](
	inputFlat []T, im2colFlat []T,
	inputShape, kernelShape shapes.Shape,
	params *convNode,
	batchSize, totalInputChannels, groupInputChannels, inputChannelOffset int,
	outputSpatialSize, patchSize, spatialRank int,
) {
	outputDims := make([]int, spatialRank)
	inputSpatialDims := make([]int, spatialRank)
	kernelSpatialDims := make([]int, spatialRank)
	strides := params.strides
	paddings := params.paddings
	kernelDilations := params.kernelDilations

	for i := 0; i < spatialRank; i++ {
		inputSpatialDims[i] = inputShape.Dimensions[i+2]
		kernelSpatialDims[i] = kernelShape.Dimensions[i+2]
	}

	// Compute output spatial dimensions.
	for i := 0; i < spatialRank; i++ {
		effectiveKernelDim := kernelSpatialDims[i]
		if params.hasKernelDilations && len(kernelDilations) > i {
			effectiveKernelDim = (kernelSpatialDims[i]-1)*kernelDilations[i] + 1
		}
		outputDims[i] = (inputSpatialDims[i] + paddings[i][0] + paddings[i][1] - effectiveKernelDim) / strides[i] + 1
	}

	// Precompute input channel stride: stride to advance one channel in the input.
	inputChannelStride := 1
	for i := 0; i < spatialRank; i++ {
		inputChannelStride *= inputSpatialDims[i]
	}
	// Input batch stride uses total channels (the full input tensor stride).
	inputBatchStride := totalInputChannels * inputChannelStride

	// Pre-allocate scratch buffers reused across patches to avoid
	// per-patch heap allocations in the inner loop.
	inputSpatialStrides := make([]int, spatialRank)
	inputSpatialStrides[spatialRank-1] = 1
	for i := spatialRank - 2; i >= 0; i-- {
		inputSpatialStrides[i] = inputSpatialStrides[i+1] * inputSpatialDims[i+1]
	}
	kernelPos := make([]int, spatialRank)
	kernelTotalSize := 1
	for i := 0; i < spatialRank; i++ {
		kernelTotalSize *= kernelSpatialDims[i]
	}

	// Iterate: for each batch, for each output spatial position, fill patchSize entries.
	im2colIdx := 0
	outputPos := make([]int, spatialRank)

	for b := 0; b < batchSize; b++ {
		batchOffset := b * inputBatchStride

		// Reset output position.
		for i := range outputPos {
			outputPos[i] = 0
		}

		for sp := 0; sp < outputSpatialSize; sp++ {
			// Extract patch for this output position.
			// Patch layout: [groupInputChannels, kernelSpatial...]
			patchIdx := im2colIdx
			for c := 0; c < groupInputChannels; c++ {
				channelOffset := batchOffset + (inputChannelOffset+c)*inputChannelStride

				// Reset kernel position.
				for i := range kernelPos {
					kernelPos[i] = 0
				}

				for ki := 0; ki < kernelTotalSize; ki++ {
					// Compute input position for this kernel element.
					inputOffset := channelOffset
					inBounds := true
					for d := 0; d < spatialRank; d++ {
						kd := kernelPos[d]
						if params.hasKernelDilations && len(kernelDilations) > d {
							kd *= kernelDilations[d]
						}
						inputIdx := outputPos[d]*strides[d] + kd - paddings[d][0]
						if inputIdx < 0 || inputIdx >= inputSpatialDims[d] {
							inBounds = false
							break
						}
						inputOffset += inputIdx * inputSpatialStrides[d]
					}

					if inBounds {
						im2colFlat[patchIdx] = inputFlat[inputOffset]
					} else {
						im2colFlat[patchIdx] = 0
					}
					patchIdx++

					// Advance kernel position (odometer-style).
					for d := spatialRank - 1; d >= 0; d-- {
						kernelPos[d]++
						if kernelPos[d] < kernelSpatialDims[d] {
							break
						}
						kernelPos[d] = 0
					}
				}
			}
			im2colIdx += patchSize

			// Advance output position (odometer-style).
			for d := spatialRank - 1; d >= 0; d-- {
				outputPos[d]++
				if outputPos[d] < outputDims[d] {
					break
				}
				outputPos[d] = 0
			}
		}
	}
}

// transposeNHWCtoNCHWGroup transposes a group's matmul output into the correct
// position in the NCHW output tensor.
//
// src: [batchSize * spatialSize * groupChannels] in [N, HW, groupC] order
// dst: [batchSize * totalChannels * spatialSize] in [N, C, HW] order
//
// The group's channels are written to [channelOffset, channelOffset+groupChannels)
// within each batch element's channel dimension.
func transposeNHWCtoNCHWGroup[T float32 | float64](src, dst []T,
	batchSize, spatialSize, groupChannels, totalChannels, channelOffset int) {
	for b := 0; b < batchSize; b++ {
		srcBatchOffset := b * spatialSize * groupChannels
		dstBatchOffset := b * totalChannels * spatialSize
		for s := 0; s < spatialSize; s++ {
			srcRowOffset := srcBatchOffset + s*groupChannels
			for c := 0; c < groupChannels; c++ {
				dst[dstBatchOffset+(channelOffset+c)*spatialSize+s] = src[srcRowOffset+c]
			}
		}
	}
}
