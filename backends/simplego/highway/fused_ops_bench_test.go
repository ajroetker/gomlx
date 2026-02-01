// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// --- Decomposed implementations for baseline comparison ---
// These mirror what the decomposed graph ops would compute, simulating
// separate element-wise ops with intermediate memory allocations.

func decomposedSoftmaxFloat32(input []float32, dims []int, axis int) []float32 {
	outerSize, axisSize, innerSize := computeAxisStrides(dims, axis)
	size := len(input)

	maxVals := make([]float32, outerSize*innerSize)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			maxVal := float32(math.Inf(-1))
			for i := 0; i < axisSize; i++ {
				idx := outer*axisSize*innerSize + i*innerSize + inner
				if input[idx] > maxVal {
					maxVal = input[idx]
				}
			}
			maxVals[outer*innerSize+inner] = maxVal
		}
	}

	shifted := make([]float32, size)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			maxVal := maxVals[outer*innerSize+inner]
			for i := 0; i < axisSize; i++ {
				idx := outer*axisSize*innerSize + i*innerSize + inner
				shifted[idx] = input[idx] - maxVal
			}
		}
	}

	expVals := make([]float32, size)
	for i, v := range shifted {
		expVals[i] = float32(math.Exp(float64(v)))
	}

	sumVals := make([]float32, outerSize*innerSize)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			var sum float32
			for i := 0; i < axisSize; i++ {
				idx := outer*axisSize*innerSize + i*innerSize + inner
				sum += expVals[idx]
			}
			sumVals[outer*innerSize+inner] = sum
		}
	}

	output := make([]float32, size)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			invSum := 1.0 / sumVals[outer*innerSize+inner]
			for i := 0; i < axisSize; i++ {
				idx := outer*axisSize*innerSize + i*innerSize + inner
				output[idx] = expVals[idx] * invSum
			}
		}
	}
	return output
}

func decomposedGeluFloat32(input []float32) []float32 {
	sqrt2Inv := float32(1.0 / math.Sqrt(2.0))
	size := len(input)

	scaled := make([]float32, size)
	for i, x := range input {
		scaled[i] = x * sqrt2Inv
	}

	erfVals := make([]float32, size)
	for i, v := range scaled {
		erfVals[i] = float32(math.Erf(float64(v)))
	}

	onePlusErf := make([]float32, size)
	for i, v := range erfVals {
		onePlusErf[i] = 1.0 + v
	}

	cdf := make([]float32, size)
	for i, v := range onePlusErf {
		cdf[i] = 0.5 * v
	}

	output := make([]float32, size)
	for i, x := range input {
		output[i] = x * cdf[i]
	}
	return output
}

func decomposedLayerNormFloat32(input []float32, normSize int, epsilon float64, gamma, beta []float32) []float32 {
	numGroups := len(input) / normSize
	normSizeF := float32(normSize)
	output := make([]float32, len(input))

	for g := 0; g < numGroups; g++ {
		base := g * normSize

		var sum float32
		for i := 0; i < normSize; i++ {
			sum += input[base+i]
		}
		mean := sum / normSizeF

		diff := make([]float32, normSize)
		for i := 0; i < normSize; i++ {
			diff[i] = input[base+i] - mean
		}

		diffSq := make([]float32, normSize)
		for i, d := range diff {
			diffSq[i] = d * d
		}

		var varSum float32
		for _, v := range diffSq {
			varSum += v
		}
		variance := varSum / normSizeF

		invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))

		for i := 0; i < normSize; i++ {
			normalized := diff[i] * invStd
			if gamma != nil {
				normalized *= gamma[i]
			}
			if beta != nil {
				normalized += beta[i]
			}
			output[base+i] = normalized
		}
	}
	return output
}

func decomposedDenseFloat32(xData, wData, biasData []float32, batchSize, inFeatures, outFeatures int) []float32 {
	matmulOut := make([]float32, batchSize*outFeatures)
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outFeatures; o++ {
			var sum float32
			for i := 0; i < inFeatures; i++ {
				sum += xData[b*inFeatures+i] * wData[i*outFeatures+o]
			}
			matmulOut[b*outFeatures+o] = sum
		}
	}

	output := make([]float32, batchSize*outFeatures)
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outFeatures; o++ {
			output[b*outFeatures+o] = matmulOut[b*outFeatures+o] + biasData[o]
		}
	}
	return output
}

// --- Benchmarks: Highway SIMD Fused vs Decomposed Scalar ---

func BenchmarkFusedSoftmax(b *testing.B) {
	sizes := []struct {
		name string
		dims []int
		axis int
	}{
		{"8x64_axis1", []int{8, 64}, 1},
		{"32x128_axis1", []int{32, 128}, 1},
		{"64x512_axis1", []int{64, 512}, 1},
		{"8x16x64_axis2", []int{8, 16, 64}, 2},
		{"4x8x32x128_axis3", []int{4, 8, 32, 128}, 3},
	}

	for _, sz := range sizes {
		shape := shapes.Make(dtypes.Float32, sz.dims...)
		data := make([]float32, shape.Size())
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}

		b.Run(fmt.Sprintf("Highway/%s", sz.name), func(b *testing.B) {
			output := make([]float32, len(data))
			// Warmup
			softmaxHighway(data, output, sz.axis, sz.dims)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				softmaxHighway(data, output, sz.axis, sz.dims)
			}
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = decomposedSoftmaxFloat32(data, sz.dims, sz.axis)
			}
		})
	}
}

func BenchmarkFusedGelu(b *testing.B) {
	sizes := []struct {
		name string
		n    int
	}{
		{"512", 512},
		{"4096", 4096},
		{"32768", 32 * 1024},
		{"262144", 64 * 4096},
	}

	for _, sz := range sizes {
		data := make([]float32, sz.n)
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}

		b.Run(fmt.Sprintf("Highway/%s", sz.name), func(b *testing.B) {
			output := make([]float32, sz.n)
			// Warmup
			activation.GELU(data, output)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				activation.GELU(data, output)
			}
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = decomposedGeluFloat32(data)
			}
		})
	}
}

func BenchmarkFusedLayerNorm(b *testing.B) {
	sizes := []struct {
		name     string
		batch    int
		normSize int
	}{
		{"8x64", 8, 64},
		{"32x256", 32, 256},
		{"64x768", 64, 768},
		{"128x1024", 128, 1024},
	}

	for _, sz := range sizes {
		total := sz.batch * sz.normSize
		data := make([]float32, total)
		gamma := make([]float32, sz.normSize)
		beta := make([]float32, sz.normSize)
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}
		for i := range gamma {
			gamma[i] = rand.Float32()*2 - 1
			beta[i] = rand.Float32()*2 - 1
		}

		b.Run(fmt.Sprintf("Highway/%s", sz.name), func(b *testing.B) {
			output := make([]float32, total)
			// Warmup
			nn.LayerNorm(data, output, sz.normSize, gamma, beta, float32(1e-5))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nn.LayerNorm(data, output, sz.normSize, gamma, beta, float32(1e-5))
			}
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = decomposedLayerNormFloat32(data, sz.normSize, 1e-5, gamma, beta)
			}
		})
	}
}

func BenchmarkFusedDense(b *testing.B) {
	sizes := []struct {
		name        string
		batch       int
		inFeatures  int
		outFeatures int
	}{
		{"1x64x64", 1, 64, 64},
		{"8x128x256", 8, 128, 256},
		{"32x512x1024", 32, 512, 1024},
		{"11x1024x1024", 11, 1024, 1024},
	}

	for _, sz := range sizes {
		xData := make([]float32, sz.batch*sz.inFeatures)
		wData := make([]float32, sz.inFeatures*sz.outFeatures)
		biasData := make([]float32, sz.outFeatures)
		for i := range xData {
			xData[i] = rand.Float32()*2 - 1
		}
		for i := range wData {
			wData[i] = rand.Float32()*2 - 1
		}
		for i := range biasData {
			biasData[i] = rand.Float32()*2 - 1
		}

		b.Run(fmt.Sprintf("Highway/%s", sz.name), func(b *testing.B) {
			output := make([]float32, sz.batch*sz.outFeatures)
			// Warmup
			nn.DenseAuto(hwyPool, xData, wData, biasData, output, sz.batch, sz.inFeatures, sz.outFeatures)
			flops := float64(2*sz.batch*sz.inFeatures*sz.outFeatures + sz.batch*sz.outFeatures)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nn.DenseAuto(hwyPool, xData, wData, biasData, output, sz.batch, sz.inFeatures, sz.outFeatures)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			flops := float64(2*sz.batch*sz.inFeatures*sz.outFeatures + sz.batch*sz.outFeatures)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = decomposedDenseFloat32(xData, wData, biasData, sz.batch, sz.inFeatures, sz.outFeatures)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// BenchmarkFusedViaGraph benchmarks fused ops through the full graph execution path,
// which is how they're actually used. The highway init() registers executors at
// RegisterPriorityArch, so these will use SIMD implementations.
func BenchmarkFusedViaGraph(b *testing.B) {
	b.Run("Softmax_32x128", func(b *testing.B) {
		shape := shapes.Make(dtypes.Float32, 32, 128)
		data := make([]float32, shape.Size())
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}

		inputBuf, err := backend.BufferFromFlatData(0, data, shape)
		if err != nil {
			b.Fatal(err)
		}

		builder := backend.Builder("bench_softmax")
		mainFn := builder.Main()
		param, _ := mainFn.Parameter("x", shape, nil)
		out, _ := mainFn.FusedSoftmax(param, 1)
		_ = mainFn.Return([]backends.Value{out}, nil)
		exec, _ := builder.Compile()

		// Warmup
		exec.Execute([]backends.Buffer{inputBuf}, nil, 0)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			outputs, _ := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
			backend.BufferFinalize(outputs[0])
		}
	})

	b.Run("GELU_32768", func(b *testing.B) {
		shape := shapes.Make(dtypes.Float32, 32768)
		data := make([]float32, shape.Size())
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}

		inputBuf, err := backend.BufferFromFlatData(0, data, shape)
		if err != nil {
			b.Fatal(err)
		}

		builder := backend.Builder("bench_gelu")
		mainFn := builder.Main()
		param, _ := mainFn.Parameter("x", shape, nil)
		out, _ := mainFn.FusedGelu(param, false)
		_ = mainFn.Return([]backends.Value{out}, nil)
		exec, _ := builder.Compile()

		// Warmup
		exec.Execute([]backends.Buffer{inputBuf}, nil, 0)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			outputs, _ := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
			backend.BufferFinalize(outputs[0])
		}
	})

	b.Run("LayerNorm_64x768", func(b *testing.B) {
		shape := shapes.Make(dtypes.Float32, 64, 768)
		gammaShape := shapes.Make(dtypes.Float32, 768)
		betaShape := shapes.Make(dtypes.Float32, 768)

		data := make([]float32, shape.Size())
		gamma := make([]float32, 768)
		beta := make([]float32, 768)
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}
		for i := range gamma {
			gamma[i] = rand.Float32()*2 - 1
			beta[i] = rand.Float32()*2 - 1
		}

		inputBuf, _ := backend.BufferFromFlatData(0, data, shape)
		gammaBuf, _ := backend.BufferFromFlatData(0, gamma, gammaShape)
		betaBuf, _ := backend.BufferFromFlatData(0, beta, betaShape)

		builder := backend.Builder("bench_layernorm")
		mainFn := builder.Main()
		pX, _ := mainFn.Parameter("x", shape, nil)
		pG, _ := mainFn.Parameter("gamma", gammaShape, nil)
		pB, _ := mainFn.Parameter("beta", betaShape, nil)
		out, _ := mainFn.FusedLayerNorm(pX, []int{1}, 1e-5, pG, pB)
		_ = mainFn.Return([]backends.Value{out}, nil)
		exec, _ := builder.Compile()

		bufs := []backends.Buffer{inputBuf, gammaBuf, betaBuf}

		// Warmup
		exec.Execute(bufs, nil, 0)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			outputs, _ := exec.Execute(bufs, nil, 0)
			backend.BufferFinalize(outputs[0])
		}
	})

	b.Run("Dense_32x512x1024", func(b *testing.B) {
		xShape := shapes.Make(dtypes.Float32, 32, 512)
		wShape := shapes.Make(dtypes.Float32, 512, 1024)
		bShape := shapes.Make(dtypes.Float32, 1024)

		xData := make([]float32, xShape.Size())
		wData := make([]float32, wShape.Size())
		biasData := make([]float32, bShape.Size())
		for i := range xData {
			xData[i] = rand.Float32()*2 - 1
		}
		for i := range wData {
			wData[i] = rand.Float32()*2 - 1
		}
		for i := range biasData {
			biasData[i] = rand.Float32()*2 - 1
		}

		xBuf, _ := backend.BufferFromFlatData(0, xData, xShape)
		wBuf, _ := backend.BufferFromFlatData(0, wData, wShape)
		bBuf, _ := backend.BufferFromFlatData(0, biasData, bShape)

		builder := backend.Builder("bench_dense")
		mainFn := builder.Main()
		pX, _ := mainFn.Parameter("x", xShape, nil)
		pW, _ := mainFn.Parameter("w", wShape, nil)
		pB, _ := mainFn.Parameter("b", bShape, nil)
		out, _ := mainFn.FusedDense(pX, pW, pB, backends.ActivationNone)
		_ = mainFn.Return([]backends.Value{out}, nil)
		exec, _ := builder.Compile()

		bufs := []backends.Buffer{xBuf, wBuf, bBuf}
		flops := float64(2*32*512*1024 + 32*1024)

		// Warmup
		exec.Execute(bufs, nil, 0)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			outputs, _ := exec.Execute(bufs, nil, 0)
			backend.BufferFinalize(outputs[0])
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}
