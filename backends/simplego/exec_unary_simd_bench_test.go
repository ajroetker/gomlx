package simplego

import (
	"math/rand"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Benchmarks for SIMD-accelerated unary operations.
// Run with: GOEXPERIMENT=simd go1.26rc1 test -bench=. -benchmem ./backends/simplego/...

func benchUnaryOp(b *testing.B, size int, dtype dtypes.DType, opName string, execFn func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)) {
	be := backend.(*Backend)

	shape := shapes.Make(dtype, size)
	input := be.getBuffer(dtype, size)
	input.shape = shape

	// Initialize with random data
	switch dtype {
	case dtypes.Float32:
		data := input.flat.([]float32)
		for i := range data {
			data[i] = rand.Float32()*10 - 5 // Range [-5, 5]
		}
	case dtypes.Float64:
		data := input.flat.([]float64)
		for i := range data {
			data[i] = rand.Float64()*10 - 5 // Range [-5, 5]
		}
	}

	node := &Node{shape: shape}
	inputs := []*Buffer{input}
	inputsOwned := []bool{false}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		output, err := execFn(be, node, inputs, inputsOwned)
		if err != nil {
			b.Fatal(err)
		}
		be.putBuffer(output)
	}
}

// Float32 benchmarks - 1M elements
func BenchmarkExp_Float32_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float32, "Exp", execExp)
}

func BenchmarkLog_Float32_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float32, "Log", execLog)
}

func BenchmarkTanh_Float32_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float32, "Tanh", execTanh)
}

func BenchmarkSin_Float32_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float32, "Sin", execSin)
}

func BenchmarkCos_Float32_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float32, "Cos", execCos)
}

func BenchmarkLogistic_Float32_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float32, "Logistic", execLogistic)
}

func BenchmarkErf_Float32_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float32, "Erf", execErf)
}

func BenchmarkSqrt_Float32_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float32, "Sqrt", execSqrt)
}

// Float64 benchmarks - 1M elements
func BenchmarkExp_Float64_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float64, "Exp", execExp)
}

func BenchmarkLog_Float64_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float64, "Log", execLog)
}

func BenchmarkTanh_Float64_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float64, "Tanh", execTanh)
}

func BenchmarkSin_Float64_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float64, "Sin", execSin)
}

func BenchmarkCos_Float64_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float64, "Cos", execCos)
}

func BenchmarkLogistic_Float64_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float64, "Logistic", execLogistic)
}

func BenchmarkErf_Float64_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float64, "Erf", execErf)
}

func BenchmarkSqrt_Float64_1M(b *testing.B) {
	benchUnaryOp(b, 1_000_000, dtypes.Float64, "Sqrt", execSqrt)
}

// Smaller size benchmarks for comparison - 10K elements
func BenchmarkExp_Float32_10K(b *testing.B) {
	benchUnaryOp(b, 10_000, dtypes.Float32, "Exp", execExp)
}

func BenchmarkTanh_Float32_10K(b *testing.B) {
	benchUnaryOp(b, 10_000, dtypes.Float32, "Tanh", execTanh)
}

func BenchmarkLogistic_Float32_10K(b *testing.B) {
	benchUnaryOp(b, 10_000, dtypes.Float32, "Logistic", execLogistic)
}
