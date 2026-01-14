// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/require"
)

func TestDynamicShapeExecution(t *testing.T) {
	// Create a simple graph with dynamic batch dimension: y = -x
	builder := backend.Builder("test_dynamic_shape")
	mainFn := builder.Main()

	// Parameter with dynamic "batch" dimension
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 2), nil)
	require.NoError(t, err)

	// Negate (unary op, no broadcasting needed)
	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	// Return
	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	// Compile
	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2
	input1, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 2, 2))
	require.NoError(t, err)
	outputs1, err := exec.Execute([]backends.Buffer{input1}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs1, 1)

	// Verify output shape and values
	outputShape1, err := backend.BufferShape(outputs1[0])
	require.NoError(t, err)
	require.Equal(t, shapes.Make(dtypes.Float32, 2, 2), outputShape1)

	outputData1 := outputs1[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{-1, -2, -3, -4}, outputData1)

	// Execute with batch=3
	input2, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, shapes.Make(dtypes.Float32, 3, 2))
	require.NoError(t, err)
	outputs2, err := exec.Execute([]backends.Buffer{input2}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs2, 1)

	// Verify output shape and values
	outputShape2, err := backend.BufferShape(outputs2[0])
	require.NoError(t, err)
	require.Equal(t, shapes.Make(dtypes.Float32, 3, 2), outputShape2)

	outputData2 := outputs2[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{-1, -2, -3, -4, -5, -6}, outputData2)
}

func TestSpecializationCaching(t *testing.T) {
	// Create a graph with dynamic batch dimension
	builder := backend.Builder("test_specialization_caching")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch"), nil)
	require.NoError(t, err)

	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	simpleGoExec := exec.(*Executable)
	require.True(t, simpleGoExec.hasDynamicAxes)

	// Execute with batch=5
	input1, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5}, shapes.Make(dtypes.Float32, 5))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{input1}, nil, 0)
	require.NoError(t, err)

	// Check that specialization was created
	spec1, ok := simpleGoExec.specializations.Load("batch=5")
	require.True(t, ok)
	require.NotNil(t, spec1)

	// Execute with batch=5 again
	input2, err := backend.BufferFromFlatData(0, []float32{5, 4, 3, 2, 1}, shapes.Make(dtypes.Float32, 5))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{input2}, nil, 0)
	require.NoError(t, err)

	// Check that same specialization was reused
	spec2, ok := simpleGoExec.specializations.Load("batch=5")
	require.True(t, ok)
	require.Same(t, spec1, spec2)

	// Execute with batch=10
	input3, err := backend.BufferFromFlatData(0, make([]float32, 10), shapes.Make(dtypes.Float32, 10))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{input3}, nil, 0)
	require.NoError(t, err)

	// Check that a new specialization was created
	spec3, ok := simpleGoExec.specializations.Load("batch=10")
	require.True(t, ok)
	require.NotSame(t, spec1, spec3)
}

func TestMultipleDynamicAxes(t *testing.T) {
	// Create a graph with multiple dynamic dimensions: [batch, seq, hidden]
	builder := backend.Builder("test_multiple_dynamic_axes")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", "seq", 4), nil)
	require.NoError(t, err)

	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2, seq=3, hidden=4
	input, err := backend.BufferFromFlatData(0, make([]float32, 2*3*4), shapes.Make(dtypes.Float32, 2, 3, 4))
	require.NoError(t, err)
	outputs, err := exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify output shape
	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, shapes.Make(dtypes.Float32, 2, 3, 4), outputShape)

	// Check specialization key
	simpleGoExec := exec.(*Executable)
	_, ok := simpleGoExec.specializations.Load("batch=2,seq=3")
	require.True(t, ok)
}

func TestMixedStaticDynamic(t *testing.T) {
	// Create a graph with one static input and one dynamic input of same rank
	builder := backend.Builder("test_mixed_static_dynamic")
	mainFn := builder.Main()

	// Static input: fixed shape [2, 3]
	a, err := mainFn.Parameter("a", shapes.Make(dtypes.Float32, 2, 3), nil)
	require.NoError(t, err)

	// Dynamic input: dynamic batch dimension but same rank [batch, 3]
	// Note: same rank but first axis is dynamic
	b, err := mainFn.Parameter("b", shapes.MakeDynamic(dtypes.Float32, "batch", 3), nil)
	require.NoError(t, err)

	// Return both separately (no binary op to avoid broadcasting complexity)
	aNeg, err := mainFn.Neg(a)
	require.NoError(t, err)
	bNeg, err := mainFn.Neg(b)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{aNeg, bNeg}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Verify hasDynamicAxes is true (because at least one input has named axes)
	simpleGoExec := exec.(*Executable)
	require.True(t, simpleGoExec.hasDynamicAxes)

	// Execute with a=[2,3] and batch=2 (both are [2,3] at runtime)
	inputA, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, shapes.Make(dtypes.Float32, 2, 3))
	require.NoError(t, err)
	inputB, err := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40, 50, 60}, shapes.Make(dtypes.Float32, 2, 3))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{inputA, inputB}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 2)

	// Verify outputs
	outputShape0, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, shapes.Make(dtypes.Float32, 2, 3), outputShape0)

	outputData0 := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{-1, -2, -3, -4, -5, -6}, outputData0)

	outputShape1, err := backend.BufferShape(outputs[1])
	require.NoError(t, err)
	require.Equal(t, shapes.Make(dtypes.Float32, 2, 3), outputShape1)

	outputData1 := outputs[1].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{-10, -20, -30, -40, -50, -60}, outputData1)
}

func TestShapeSpecialization(t *testing.T) {
	// Test newSpecialization directly
	builder := backend.Builder("test_shape_specialization").(*Builder)
	mainFn := builder.Main()

	// Create parameter with dynamic shape
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 512), nil)
	require.NoError(t, err)

	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	_, err = builder.Compile()
	require.NoError(t, err)

	// Create specialization with batch=32
	bindings := shapes.AxisBindings{"batch": 32}
	spec := newSpecialization(builder, bindings)

	require.NotNil(t, spec)
	require.Equal(t, "batch=32", spec.Key())

	// Get node shapes
	xNode := x.(*Node)
	yNode := y.(*Node)

	xResolvedShape := spec.NodeShape(xNode.builderIdx)
	// Resolved shapes preserve axis names but have concrete dimensions
	require.Equal(t, dtypes.Float32, xResolvedShape.DType)
	require.Equal(t, []int{32, 512}, xResolvedShape.Dimensions)
	require.True(t, xResolvedShape.IsFullyConcrete())

	yResolvedShape := spec.NodeShape(yNode.builderIdx)
	require.Equal(t, dtypes.Float32, yResolvedShape.DType)
	require.Equal(t, []int{32, 512}, yResolvedShape.Dimensions)
	require.True(t, yResolvedShape.IsFullyConcrete())
}

func TestStaticShapeNoSpecialization(t *testing.T) {
	// Verify that static shapes don't create specializations
	builder := backend.Builder("test_static_no_specialization")
	mainFn := builder.Main()

	// Fully static parameter
	x, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 3, 4), nil)
	require.NoError(t, err)

	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	simpleGoExec := exec.(*Executable)
	require.False(t, simpleGoExec.hasDynamicAxes)
	require.Nil(t, simpleGoExec.inputPatterns)

	// Execute
	input, err := backend.BufferFromFlatData(0, make([]float32, 12), shapes.Make(dtypes.Float32, 3, 4))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)

	// Verify no specializations were created (sync.Map has no direct length check,
	// but we can try to load a key and verify it doesn't exist)
	count := 0
	simpleGoExec.specializations.Range(func(_, _ any) bool {
		count++
		return true
	})
	require.Equal(t, 0, count)
}

func TestDynamicShapeError(t *testing.T) {
	// Test that mismatched static dimensions are caught
	builder := backend.Builder("test_dynamic_shape_error")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 512), nil)
	require.NoError(t, err)

	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Try to execute with wrong static dimension (256 instead of 512)
	input, err := backend.BufferFromFlatData(0, make([]float32, 32*256), shapes.Make(dtypes.Float32, 32, 256))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{input}, nil, 0)
	require.Error(t, err)
	require.Contains(t, err.Error(), "mismatch")
}
