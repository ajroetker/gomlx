//go:build amd64 && goexperiment.simd

package simplego

import (
	"github.com/ajroetker/go-highway/hwy/contrib/algo"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// This file provides SIMD-accelerated implementations of unary math operations
// using go-highway's AVX2/AVX-512 vectorized transforms.
//
// These implementations are only enabled when building with GOEXPERIMENT=simd
// on amd64 architecture. They register with priorityArch to override the
// generic scalar implementations.

func init() {
	// Register SIMD-accelerated unary ops with architecture priority
	setNodeExecutor(backends.OpTypeExp, priorityArch, execExpSIMD)
	setNodeExecutor(backends.OpTypeLog, priorityArch, execLogSIMD)
	setNodeExecutor(backends.OpTypeTanh, priorityArch, execTanhSIMD)
	setNodeExecutor(backends.OpTypeSin, priorityArch, execSinSIMD)
	setNodeExecutor(backends.OpTypeCos, priorityArch, execCosSIMD)
	setNodeExecutor(backends.OpTypeLogistic, priorityArch, execLogisticSIMD)
	setNodeExecutor(backends.OpTypeErf, priorityArch, execErfSIMD)
	setNodeExecutor(backends.OpTypeSqrt, priorityArch, execSqrtSIMD)
}

// execExpSIMD executes Exp using SIMD-accelerated transforms.
func execExpSIMD(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Float32:
		algo.ExpTransform(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		algo.ExpTransform64(input.flat.([]float64), output.flat.([]float64))
	default:
		// Fall back to generic for other types (BFloat16, etc.)
		return execExp(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

// execLogSIMD executes Log using SIMD-accelerated transforms.
func execLogSIMD(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Float32:
		algo.LogTransform(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		algo.LogTransform64(input.flat.([]float64), output.flat.([]float64))
	default:
		return execLog(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

// execTanhSIMD executes Tanh using SIMD-accelerated transforms.
func execTanhSIMD(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Float32:
		algo.TanhTransform(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		algo.TanhTransform64(input.flat.([]float64), output.flat.([]float64))
	default:
		return execTanh(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

// execSinSIMD executes Sin using SIMD-accelerated transforms.
func execSinSIMD(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Float32:
		algo.SinTransform(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		algo.SinTransform64(input.flat.([]float64), output.flat.([]float64))
	default:
		return execSin(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

// execCosSIMD executes Cos using SIMD-accelerated transforms.
func execCosSIMD(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Float32:
		algo.CosTransform(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		algo.CosTransform64(input.flat.([]float64), output.flat.([]float64))
	default:
		return execCos(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

// execLogisticSIMD executes Logistic (sigmoid) using SIMD-accelerated transforms.
func execLogisticSIMD(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Float32:
		algo.SigmoidTransform(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		algo.SigmoidTransform64(input.flat.([]float64), output.flat.([]float64))
	default:
		return execLogistic(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

// execErfSIMD executes Erf using SIMD-accelerated transforms.
func execErfSIMD(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Float32:
		algo.ErfTransform(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		algo.ErfTransform64(input.flat.([]float64), output.flat.([]float64))
	default:
		return execErf(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

// execSqrtSIMD executes Sqrt using SIMD-accelerated transforms.
func execSqrtSIMD(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Float32:
		algo.SqrtTransform(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		algo.SqrtTransform64(input.flat.([]float64), output.flat.([]float64))
	default:
		return execSqrt(backend, node, inputs, inputsOwned)
	}
	return output, nil
}
