// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const fusedTol = 1e-5

// execFusedOp builds, compiles and executes a single-output fused op through the backend.
func execFusedOp(t *testing.T, inputShape shapes.Shape, inputData any,
	buildFn func(f backends.Function, param backends.Value) (backends.Value, error),
) any {
	t.Helper()
	builder := backend.Builder("highway_fused_test")
	mainFn := builder.Main()

	param, err := mainFn.Parameter("x", inputShape, nil)
	require.NoError(t, err)

	out, err := buildFn(mainFn, param)
	require.NoError(t, err)

	require.NoError(t, mainFn.Return([]backends.Value{out}, nil))
	exec, err := builder.Compile()
	require.NoError(t, err)

	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	return outputs[0].(*simplego.Buffer).Flat()
}

// execFusedOpMulti builds, compiles and executes with multiple inputs.
func execFusedOpMulti(t *testing.T, inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f backends.Function, params []backends.Value) (backends.Value, error),
) any {
	t.Helper()
	builder := backend.Builder("highway_fused_test_multi")
	mainFn := builder.Main()

	params := make([]backends.Value, len(inputShapes))
	for i, s := range inputShapes {
		p, err := mainFn.Parameter("x"+string(rune('0'+i)), s, nil)
		require.NoError(t, err)
		params[i] = p
	}

	out, err := buildFn(mainFn, params)
	require.NoError(t, err)

	require.NoError(t, mainFn.Return([]backends.Value{out}, nil))
	exec, err := builder.Compile()
	require.NoError(t, err)

	inputBufs := make([]backends.Buffer, len(inputDatas))
	for i, data := range inputDatas {
		buf, err := backend.BufferFromFlatData(0, data, inputShapes[i])
		require.NoError(t, err)
		inputBufs[i] = buf
	}

	outputs, err := exec.Execute(inputBufs, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	return outputs[0].(*simplego.Buffer).Flat()
}

func TestHighwaySoftmax_LastAxis(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	shape := shapes.Make(dtypes.Float32, 2, 2)

	got := execFusedOp(t, shape, input, func(f backends.Function, x backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(x, 1) // softmax along last axis
	}).([]float32)

	// Each row should sum to 1.
	assert.InDelta(t, 1.0, float64(got[0]+got[1]), fusedTol)
	assert.InDelta(t, 1.0, float64(got[2]+got[3]), fusedTol)

	// softmax([1,2]) = [e^1/(e^1+e^2), e^2/(e^1+e^2)]
	e1, e2 := math.Exp(1), math.Exp(2)
	assert.InDelta(t, e1/(e1+e2), got[0], fusedTol)
	assert.InDelta(t, e2/(e1+e2), got[1], fusedTol)
}

func TestHighwaySoftmax_NonLastAxis(t *testing.T) {
	// 2×3 matrix, softmax along axis 0 (column-wise).
	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	shape := shapes.Make(dtypes.Float32, 2, 3)

	got := execFusedOp(t, shape, input, func(f backends.Function, x backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(x, 0)
	}).([]float32)

	// Each column should sum to 1.
	for col := 0; col < 3; col++ {
		colSum := got[col] + got[3+col]
		assert.InDelta(t, 1.0, colSum, fusedTol, "column %d sum", col)
	}
}

func TestHighwaySoftmax_3D(t *testing.T) {
	// Shape [2,3,4], softmax along axis 2 (last).
	n := 24
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(i) * 0.1
	}
	shape := shapes.Make(dtypes.Float32, 2, 3, 4)

	got := execFusedOp(t, shape, input, func(f backends.Function, x backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(x, 2)
	}).([]float32)

	// Each group of 4 should sum to 1.
	for group := 0; group < 6; group++ {
		var sum float32
		for j := 0; j < 4; j++ {
			sum += got[group*4+j]
		}
		assert.InDelta(t, 1.0, sum, fusedTol, "group %d sum", group)
	}
}

func TestHighwaySoftmax_Float64(t *testing.T) {
	input := []float64{1.0, 2.0, 3.0, 4.0}
	shape := shapes.Make(dtypes.Float64, 4)

	got := execFusedOp(t, shape, input, func(f backends.Function, x backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(x, 0)
	}).([]float64)

	var sum float64
	for _, v := range got {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 1e-10)
}

func TestHighwayGelu(t *testing.T) {
	input := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	shape := shapes.Make(dtypes.Float32, 5)

	got := execFusedOp(t, shape, input, func(f backends.Function, x backends.Value) (backends.Value, error) {
		return f.FusedGelu(x, false)
	}).([]float32)

	// GELU(0) = 0
	assert.InDelta(t, 0.0, got[2], fusedTol)

	// GELU is approximately identity for large positive x
	assert.InDelta(t, 2.0*0.5*(1.0+math.Erf(2.0/math.Sqrt(2.0))), got[4], fusedTol)

	// GELU(-x) ≈ -GELU(x) only approximately; check exact formula
	sqrt2Inv := 1.0 / math.Sqrt(2.0)
	for i, x := range input {
		expected := float64(x) * 0.5 * (1.0 + math.Erf(float64(x)*sqrt2Inv))
		assert.InDelta(t, expected, got[i], fusedTol, "GELU(%f)", x)
	}
}

func TestHighwayGelu_Float64(t *testing.T) {
	input := []float64{-2.0, -1.0, 0.0, 1.0, 2.0}
	shape := shapes.Make(dtypes.Float64, 5)

	got := execFusedOp(t, shape, input, func(f backends.Function, x backends.Value) (backends.Value, error) {
		return f.FusedGelu(x, false)
	}).([]float64)

	sqrt2Inv := 1.0 / math.Sqrt(2.0)
	for i, x := range input {
		expected := x * 0.5 * (1.0 + math.Erf(x*sqrt2Inv))
		assert.InDelta(t, expected, got[i], 1e-6, "GELU(%f)", x)
	}
}

func TestHighwayLayerNorm(t *testing.T) {
	// 2×4 matrix, normalize over last axis (normSize=4).
	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	shape := shapes.Make(dtypes.Float32, 2, 4)

	got := execFusedOp(t, shape, input, func(f backends.Function, x backends.Value) (backends.Value, error) {
		return f.FusedLayerNorm(x, []int{1}, 1e-5, nil, nil)
	}).([]float32)

	// Each row should have mean ≈ 0 and variance ≈ 1.
	for row := 0; row < 2; row++ {
		var sum, sqSum float32
		for j := 0; j < 4; j++ {
			v := got[row*4+j]
			sum += v
			sqSum += v * v
		}
		mean := sum / 4.0
		variance := sqSum/4.0 - mean*mean
		assert.InDelta(t, 0.0, mean, 1e-4, "row %d mean", row)
		assert.InDelta(t, 1.0, variance, 1e-4, "row %d variance", row)
	}
}

func TestHighwayLayerNorm_WithGammaBeta(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	gamma := []float32{2.0, 2.0, 2.0, 2.0}
	beta := []float32{1.0, 1.0, 1.0, 1.0}

	xShape := shapes.Make(dtypes.Float32, 1, 4)
	gShape := shapes.Make(dtypes.Float32, 4)
	bShape := shapes.Make(dtypes.Float32, 4)

	got := execFusedOpMulti(t,
		[]shapes.Shape{xShape, gShape, bShape},
		[]any{input, gamma, beta},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedLayerNorm(params[0], []int{1}, 1e-5, params[1], params[2])
		},
	).([]float32)

	// After normalization: y = 2 * normalized + 1
	// Mean of output should be 1 (beta), std should be 2 (gamma).
	var sum float32
	for _, v := range got {
		sum += v
	}
	mean := sum / 4.0
	assert.InDelta(t, 1.0, mean, 1e-4)
}

func TestHighwayDense(t *testing.T) {
	// x: [2, 3], weight: [3, 4] (in=3, out=4), bias: [4]
	// output: [2, 4]
	x := []float32{1, 2, 3, 4, 5, 6}
	// Weight [in, out] = [3, 4]:
	weight := []float32{
		1, 0, 0, 1,
		0, 1, 0, 1,
		0, 0, 1, 1,
	}
	bias := []float32{10, 20, 30, 40}

	xShape := shapes.Make(dtypes.Float32, 2, 3)
	wShape := shapes.Make(dtypes.Float32, 3, 4)
	bShape := shapes.Make(dtypes.Float32, 4)

	got := execFusedOpMulti(t,
		[]shapes.Shape{xShape, wShape, bShape},
		[]any{x, weight, bias},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], backends.ActivationNone)
		},
	).([]float32)

	// Row 0: x=[1,2,3], y = [1+10, 2+20, 3+30, 6+40] = [11, 22, 33, 46]
	// Row 1: x=[4,5,6], y = [4+10, 5+20, 6+30, 15+40] = [14, 25, 36, 55]
	expected := []float32{11, 22, 33, 46, 14, 25, 36, 55}
	for i := range expected {
		assert.InDelta(t, expected[i], got[i], fusedTol, "Dense output[%d]", i)
	}
}

func TestHighwayDense_NoBias(t *testing.T) {
	x := []float32{1, 2, 3}
	weight := []float32{1, 1, 1} // [3, 1]: single output = sum

	xShape := shapes.Make(dtypes.Float32, 1, 3)
	wShape := shapes.Make(dtypes.Float32, 3, 1)

	got := execFusedOpMulti(t,
		[]shapes.Shape{xShape, wShape},
		[]any{x, weight},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedDense(params[0], params[1], nil, backends.ActivationNone)
		},
	).([]float32)

	assert.InDelta(t, 6.0, got[0], fusedTol)
}

func TestHighwayDenseActivation(t *testing.T) {
	x := []float32{1, -1}
	weight := []float32{1, 0, 0, 1} // identity [2,2]
	bias := []float32{0, 0}

	xShape := shapes.Make(dtypes.Float32, 1, 2)
	wShape := shapes.Make(dtypes.Float32, 2, 2)
	bShape := shapes.Make(dtypes.Float32, 2)

	tests := []struct {
		name       string
		activation backends.ActivationType
		check      func(t *testing.T, got []float32)
	}{
		{
			name:       "ReLU",
			activation: backends.ActivationRelu,
			check: func(t *testing.T, got []float32) {
				assert.InDelta(t, 1.0, got[0], fusedTol)  // ReLU(1) = 1
				assert.InDelta(t, 0.0, got[1], fusedTol)  // ReLU(-1) = 0
			},
		},
		{
			name:       "Tanh",
			activation: backends.ActivationTanh,
			check: func(t *testing.T, got []float32) {
				assert.InDelta(t, math.Tanh(1), got[0], fusedTol)
				assert.InDelta(t, math.Tanh(-1), got[1], fusedTol)
			},
		},
		{
			name:       "SiLU",
			activation: backends.ActivationSilu,
			check: func(t *testing.T, got []float32) {
				// SiLU(x) = x / (1 + exp(-x))
				assert.InDelta(t, 1.0/(1.0+math.Exp(-1.0)), got[0], fusedTol)
			},
		},
		{
			name:       "GELU",
			activation: backends.ActivationGelu,
			check: func(t *testing.T, got []float32) {
				sqrt2Inv := 1.0 / math.Sqrt(2.0)
				expected := 0.5 * (1.0 + math.Erf(sqrt2Inv))
				assert.InDelta(t, expected, got[0], fusedTol)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			act := tt.activation
			got := execFusedOpMulti(t,
				[]shapes.Shape{xShape, wShape, bShape},
				[]any{x, weight, bias},
				func(f backends.Function, params []backends.Value) (backends.Value, error) {
					return f.FusedDense(params[0], params[1], params[2], act)
				},
			).([]float32)
			tt.check(t, got)
		})
	}
}
