// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// origBinaryExecs holds references to the executors registered before highway
// overrides them. Used to delegate non-float32/float64 dtypes (e.g. Float16,
// integers) back to the typed/generic implementations.
var origBinaryExecs [4]simplego.NodeExecutor

func init() {
	ops := [4]backends.OpType{backends.OpTypeAdd, backends.OpTypeSub, backends.OpTypeMul, backends.OpTypeDiv}
	fns := [4]simplego.NodeExecutor{execAddHighway, execSubHighway, execMulHighway, execDivHighway}
	for i, op := range ops {
		origBinaryExecs[i] = simplego.GetNodeExecutor(op)
		simplego.SetNodeExecutor(op, simplego.RegisterPriorityArch, fns[i])
	}
}

// execAddHighway executes element-wise addition using SIMD where possible.
func execAddHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	dtype := inputs[0].DType()
	if dtype != dtypes.Float32 && dtype != dtypes.Float64 {
		return origBinaryExecs[0](backend, node, inputs, inputsOwned)
	}
	lhs, rhs, output, lhsIsScalar, rhsIsScalar := simplego.BinaryOperandsAndOutput(backend, inputs, inputsOwned, node.Shape())
	// Add is commutative: put scalar on rhs if possible.
	if lhsIsScalar && !rhsIsScalar {
		lhs, rhs = rhs, lhs
	}
	switch dtype {
	case dtypes.Float32:
		execBinaryFloat32(lhs, rhs, output, vec.AddTo[float32], addScalarFloat32, addFloat32)
	case dtypes.Float64:
		execBinaryFloat64(lhs, rhs, output, vec.AddTo[float64], addScalarFloat64, addFloat64)
	}
	return output, nil
}

// execSubHighway executes element-wise subtraction using SIMD where possible.
func execSubHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	dtype := inputs[0].DType()
	if dtype != dtypes.Float32 && dtype != dtypes.Float64 {
		return origBinaryExecs[1](backend, node, inputs, inputsOwned)
	}
	lhs, rhs, output, _, _ := simplego.BinaryOperandsAndOutput(backend, inputs, inputsOwned, node.Shape())
	// Sub is NOT commutative — don't swap.
	switch dtype {
	case dtypes.Float32:
		execBinaryFloat32(lhs, rhs, output, vec.SubTo[float32], subScalarFloat32, subFloat32)
	case dtypes.Float64:
		execBinaryFloat64(lhs, rhs, output, vec.SubTo[float64], subScalarFloat64, subFloat64)
	}
	return output, nil
}

// execMulHighway executes element-wise multiplication using SIMD where possible.
func execMulHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	dtype := inputs[0].DType()
	if dtype != dtypes.Float32 && dtype != dtypes.Float64 {
		return origBinaryExecs[2](backend, node, inputs, inputsOwned)
	}
	lhs, rhs, output, lhsIsScalar, rhsIsScalar := simplego.BinaryOperandsAndOutput(backend, inputs, inputsOwned, node.Shape())
	// Mul is commutative: put scalar on rhs if possible.
	if lhsIsScalar && !rhsIsScalar {
		lhs, rhs = rhs, lhs
	}
	switch dtype {
	case dtypes.Float32:
		execBinaryFloat32(lhs, rhs, output, vec.MulTo[float32], mulScalarFloat32, mulFloat32)
	case dtypes.Float64:
		execBinaryFloat64(lhs, rhs, output, vec.MulTo[float64], mulScalarFloat64, mulFloat64)
	}
	return output, nil
}

// execDivHighway executes element-wise division using SIMD where possible.
func execDivHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	dtype := inputs[0].DType()
	if dtype != dtypes.Float32 && dtype != dtypes.Float64 {
		return origBinaryExecs[3](backend, node, inputs, inputsOwned)
	}
	lhs, rhs, output, _, _ := simplego.BinaryOperandsAndOutput(backend, inputs, inputsOwned, node.Shape())
	// Div is NOT commutative — don't swap.
	switch dtype {
	case dtypes.Float32:
		execBinaryFloat32(lhs, rhs, output, vec.DivTo[float32], divScalarFloat32, divFloat32)
	case dtypes.Float64:
		execBinaryFloat64(lhs, rhs, output, vec.DivTo[float64], divScalarFloat64, divFloat64)
	}
	return output, nil
}

// Scalar operation helpers for float32.
func addScalarFloat32(lhs []float32, rhs float32, output []float32) {
	copy(output, lhs)
	vec.AddConst(rhs, output)
}

func subScalarFloat32(lhs []float32, rhs float32, output []float32) {
	copy(output, lhs)
	vec.AddConst(-rhs, output)
}

func mulScalarFloat32(lhs []float32, rhs float32, output []float32) {
	copy(output, lhs)
	vec.Scale(rhs, output)
}

func divScalarFloat32(lhs []float32, rhs float32, output []float32) {
	copy(output, lhs)
	vec.Scale(1.0/rhs, output)
}

// Scalar operation helpers for float64.
func addScalarFloat64(lhs []float64, rhs float64, output []float64) {
	copy(output, lhs)
	vec.AddConst(rhs, output)
}

func subScalarFloat64(lhs []float64, rhs float64, output []float64) {
	copy(output, lhs)
	vec.AddConst(-rhs, output)
}

func mulScalarFloat64(lhs []float64, rhs float64, output []float64) {
	copy(output, lhs)
	vec.Scale(rhs, output)
}

func divScalarFloat64(lhs []float64, rhs float64, output []float64) {
	copy(output, lhs)
	vec.Scale(1.0/rhs, output)
}

// Per-element operation functions used for the broadcast fallback.
func addFloat32(a, b float32) float32 { return a + b }
func subFloat32(a, b float32) float32 { return a - b }
func mulFloat32(a, b float32) float32 { return a * b }
func divFloat32(a, b float32) float32 { return a / b }
func addFloat64(a, b float64) float64 { return a + b }
func subFloat64(a, b float64) float64 { return a - b }
func mulFloat64(a, b float64) float64 { return a * b }
func divFloat64(a, b float64) float64 { return a / b }

// execBinaryFloat32 dispatches same-shape, scalar, and broadcast cases for float32.
func execBinaryFloat32(lhs, rhs, output *simplego.Buffer,
	vecOp func([]float32, []float32, []float32),
	scalarOp func([]float32, float32, []float32),
	elemOp func(float32, float32) float32) {
	lhsFlat := lhs.Flat().([]float32)
	rhsFlat := rhs.Flat().([]float32)
	outFlat := output.Flat().([]float32)

	if len(rhsFlat) == 1 {
		scalarOp(lhsFlat, rhsFlat[0], outFlat)
	} else if len(lhsFlat) == 1 {
		// Scalar on LHS (non-commutative ops like Sub/Div).
		c := lhsFlat[0]
		for i, b := range rhsFlat {
			outFlat[i] = elemOp(c, b)
		}
	} else if lhs.Shape().Equal(rhs.Shape()) {
		vecOp(outFlat, lhsFlat, rhsFlat)
	} else {
		broadcastBinaryFloat32(lhsFlat, rhsFlat, outFlat, lhs.Shape(), rhs.Shape(), output.Shape(), vecOp, elemOp)
	}
}

// execBinaryFloat64 dispatches same-shape, scalar, and broadcast cases for float64.
func execBinaryFloat64(lhs, rhs, output *simplego.Buffer,
	vecOp func([]float64, []float64, []float64),
	scalarOp func([]float64, float64, []float64),
	elemOp func(float64, float64) float64) {
	lhsFlat := lhs.Flat().([]float64)
	rhsFlat := rhs.Flat().([]float64)
	outFlat := output.Flat().([]float64)

	if len(rhsFlat) == 1 {
		scalarOp(lhsFlat, rhsFlat[0], outFlat)
	} else if len(lhsFlat) == 1 {
		// Scalar on LHS (non-commutative ops like Sub/Div).
		c := lhsFlat[0]
		for i, b := range rhsFlat {
			outFlat[i] = elemOp(c, b)
		}
	} else if lhs.Shape().Equal(rhs.Shape()) {
		vecOp(outFlat, lhsFlat, rhsFlat)
	} else {
		broadcastBinaryFloat64(lhsFlat, rhsFlat, outFlat, lhs.Shape(), rhs.Shape(), output.Shape(), vecOp, elemOp)
	}
}

// broadcastBinaryFloat32 handles the broadcast case.
// When trailing dimensions match across both operands, it processes contiguous
// blocks with SIMD vecOp instead of per-element iteration.
func broadcastBinaryFloat32(lhs, rhs, output []float32,
	lhsShape, rhsShape, outputShape shapes.Shape,
	vecOp func([]float32, []float32, []float32),
	elemOp func(float32, float32) float32) {

	rank := outputShape.Rank()

	// Find contiguous trailing dimensions that match in all three shapes.
	// These can be processed as a single SIMD block.
	blockSize := 1
	outerRank := rank
	for axis := rank - 1; axis >= 0; axis-- {
		if lhsShape.Dimensions[axis] == outputShape.Dimensions[axis] &&
			rhsShape.Dimensions[axis] == outputShape.Dimensions[axis] {
			blockSize *= outputShape.Dimensions[axis]
			outerRank--
		} else {
			break
		}
	}

	if blockSize > 1 && outerRank > 0 {
		// Fast path: iterate over outer broadcast dims, SIMD over inner contiguous block.
		outerSize := len(output) / blockSize
		lhsOuterShape := shapes.Make(lhsShape.DType, lhsShape.Dimensions[:outerRank]...)
		rhsOuterShape := shapes.Make(rhsShape.DType, rhsShape.Dimensions[:outerRank]...)
		outOuterShape := shapes.Make(outputShape.DType, outputShape.Dimensions[:outerRank]...)

		lhsIter := simplego.NewBroadcastIterator(lhsOuterShape, outOuterShape)
		rhsIter := simplego.NewBroadcastIterator(rhsOuterShape, outOuterShape)

		for i := range outerSize {
			lhsBase := lhsIter.Next() * blockSize
			rhsBase := rhsIter.Next() * blockSize
			outBase := i * blockSize
			vecOp(output[outBase:outBase+blockSize], lhs[lhsBase:lhsBase+blockSize], rhs[rhsBase:rhsBase+blockSize])
		}
		return
	}

	// Fallback: per-element iteration with BroadcastIterator.
	lhsIter := simplego.NewBroadcastIterator(lhsShape, outputShape)
	rhsIter := simplego.NewBroadcastIterator(rhsShape, outputShape)
	for outputIdx := range output {
		lhsIdx := lhsIter.Next()
		rhsIdx := rhsIter.Next()
		output[outputIdx] = elemOp(lhs[lhsIdx], rhs[rhsIdx])
	}
}

// broadcastBinaryFloat64 handles the broadcast case for float64.
func broadcastBinaryFloat64(lhs, rhs, output []float64,
	lhsShape, rhsShape, outputShape shapes.Shape,
	vecOp func([]float64, []float64, []float64),
	elemOp func(float64, float64) float64) {

	rank := outputShape.Rank()

	blockSize := 1
	outerRank := rank
	for axis := rank - 1; axis >= 0; axis-- {
		if lhsShape.Dimensions[axis] == outputShape.Dimensions[axis] &&
			rhsShape.Dimensions[axis] == outputShape.Dimensions[axis] {
			blockSize *= outputShape.Dimensions[axis]
			outerRank--
		} else {
			break
		}
	}

	if blockSize > 1 && outerRank > 0 {
		outerSize := len(output) / blockSize
		lhsOuterShape := shapes.Make(lhsShape.DType, lhsShape.Dimensions[:outerRank]...)
		rhsOuterShape := shapes.Make(rhsShape.DType, rhsShape.Dimensions[:outerRank]...)
		outOuterShape := shapes.Make(outputShape.DType, outputShape.Dimensions[:outerRank]...)

		lhsIter := simplego.NewBroadcastIterator(lhsOuterShape, outOuterShape)
		rhsIter := simplego.NewBroadcastIterator(rhsOuterShape, outOuterShape)

		for i := range outerSize {
			lhsBase := lhsIter.Next() * blockSize
			rhsBase := rhsIter.Next() * blockSize
			outBase := i * blockSize
			vecOp(output[outBase:outBase+blockSize], lhs[lhsBase:lhsBase+blockSize], rhs[rhsBase:rhsBase+blockSize])
		}
		return
	}

	lhsIter := simplego.NewBroadcastIterator(lhsShape, outputShape)
	rhsIter := simplego.NewBroadcastIterator(rhsShape, outputShape)
	for outputIdx := range output {
		lhsIdx := lhsIter.Next()
		rhsIdx := rhsIter.Next()
		output[outputIdx] = elemOp(lhs[lhsIdx], rhs[rhsIdx])
	}
}
