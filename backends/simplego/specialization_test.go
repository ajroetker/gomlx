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

func TestShapeSpecificPool(t *testing.T) {
	// Test that ShapeSpecificPool pre-creates pools for unique (dtype, size) combinations
	builder := backend.Builder("test_shape_specific_pool").(*Builder)
	mainFn := builder.Main()

	// Create a graph with various shapes
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	// Add a constant with a different shape
	c, err := mainFn.Constant([]float32{1, 2, 3}, 3)
	require.NoError(t, err)

	_ = c // unused but creates a node with different shape

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	_, err = builder.Compile()
	require.NoError(t, err)

	// Create specialization with batch=8
	bindings := shapes.AxisBindings{"batch": 8}
	spec := newSpecialization(builder, bindings)

	// Verify buffer pool was created
	require.NotNil(t, spec.bufferPool)

	// Check that pools exist for the shapes we expect
	// Shape [8, 4] = 32 elements of Float32
	require.True(t, spec.bufferPool.hasPool(dtypes.Float32, 32), "should have pool for [8,4] = 32 Float32")
	// Shape [3] = 3 elements of Float32 (from constant)
	require.True(t, spec.bufferPool.hasPool(dtypes.Float32, 3), "should have pool for [3] = 3 Float32")

	// Test getBuffer and putBuffer
	buf := spec.bufferPool.getBuffer(dtypes.Float32, 32)
	require.NotNil(t, buf)
	require.True(t, buf.valid)
	require.Equal(t, dtypes.Float32, buf.shape.DType)

	// Put it back
	spec.bufferPool.putBuffer(buf)
	require.False(t, buf.valid)

	// Get it again - should get the same buffer (from pool)
	buf2 := spec.bufferPool.getBuffer(dtypes.Float32, 32)
	require.NotNil(t, buf2)
	require.Same(t, buf, buf2, "should reuse pooled buffer")

	// Verify getBuffer returns nil for unknown sizes
	buf3 := spec.bufferPool.getBuffer(dtypes.Float32, 999)
	require.Nil(t, buf3, "should return nil for unknown size")

	// Verify numPools returns correct count
	require.GreaterOrEqual(t, spec.bufferPool.numPools(), 2)
}

func TestDynamicShapeDotGeneral(t *testing.T) {
	// Test that DotGeneral works correctly with dynamic batch dimension.
	// This verifies:
	// 1. Algorithm selection uses concrete sizes (not DimDynamic)
	// 2. Execution path is re-computed per specialization
	// 3. Results are correct with different batch sizes
	builder := backend.Builder("test_dynamic_dotgeneral")
	mainFn := builder.Main()

	// Create matrices: A [batch, 4, 8] × B [batch, 8, 3] = C [batch, 4, 3]
	// Using batch dimension as both batch axes
	aShape := shapes.MakeDynamic(dtypes.Float32, "batch", 4, 8)
	bShape := shapes.MakeDynamic(dtypes.Float32, "batch", 8, 3)

	a, err := mainFn.Parameter("a", aShape, nil)
	require.NoError(t, err)

	b, err := mainFn.Parameter("b", bShape, nil)
	require.NoError(t, err)

	// DotGeneral: contract last axis of A with second-to-last axis of B, batch over first axis
	// A: [batch, 4, 8] - contracting axis 2 (8), batch axis 0
	// B: [batch, 8, 3] - contracting axis 1 (8), batch axis 0
	c, err := mainFn.DotGeneral(a, []int{2}, []int{0}, b, []int{1}, []int{0})
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{c}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	simpleGoExec := exec.(*Executable)
	require.True(t, simpleGoExec.hasDynamicAxes)

	// Execute with batch=2
	// A: 2×4×8 = 64 elements, B: 2×8×3 = 48 elements
	aData := make([]float32, 2*4*8)
	for i := range aData {
		aData[i] = float32(i + 1)
	}
	bData := make([]float32, 2*8*3)
	for i := range bData {
		bData[i] = float32(i + 1) * 0.1
	}

	inputA, err := backend.BufferFromFlatData(0, aData, shapes.Make(dtypes.Float32, 2, 4, 8))
	require.NoError(t, err)
	inputB, err := backend.BufferFromFlatData(0, bData, shapes.Make(dtypes.Float32, 2, 8, 3))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{inputA, inputB}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify output shape is [2, 4, 3]
	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, dtypes.Float32, outputShape.DType)
	require.Equal(t, []int{2, 4, 3}, outputShape.Dimensions)

	// Verify specialization was created with correct opParams
	spec1, ok := simpleGoExec.specializations.Load("batch=2")
	require.True(t, ok)
	spec := spec1.(*ShapeSpecialization)

	// Find the DotGeneral node and verify opParams
	var dotGeneralIdx int
	for i, node := range simpleGoExec.builder.nodes {
		if node.opType == backends.OpTypeDotGeneral {
			dotGeneralIdx = i
			break
		}
	}
	require.NotNil(t, spec.opParams)
	dgParams := spec.opParams[dotGeneralIdx]
	require.NotNil(t, dgParams, "DotGeneral should have specialized params")
	dgSpecParams := dgParams.(*DotGeneralSpecParams)
	require.Equal(t, 2, dgSpecParams.batchSize)
	require.Equal(t, 4, dgSpecParams.lhsCrossSize)
	require.Equal(t, 3, dgSpecParams.rhsCrossSize)
	require.Equal(t, 8, dgSpecParams.contractingSize)

	// Execute with batch=1 (different specialization)
	inputA2, err := backend.BufferFromFlatData(0, aData[:32], shapes.Make(dtypes.Float32, 1, 4, 8))
	require.NoError(t, err)
	inputB2, err := backend.BufferFromFlatData(0, bData[:24], shapes.Make(dtypes.Float32, 1, 8, 3))
	require.NoError(t, err)

	outputs2, err := exec.Execute([]backends.Buffer{inputA2, inputB2}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs2, 1)

	// Verify output shape is [1, 4, 3]
	outputShape2, err := backend.BufferShape(outputs2[0])
	require.NoError(t, err)
	require.Equal(t, []int{1, 4, 3}, outputShape2.Dimensions)

	// Verify a different specialization was created
	spec2, ok := simpleGoExec.specializations.Load("batch=1")
	require.True(t, ok)
	require.NotSame(t, spec1, spec2)
}

func TestDynamicShapeWithIota(t *testing.T) {
	// Test that Iota works correctly with dynamic shapes.
	// Iota uses node.shape directly for buffer allocation, so this verifies
	// the resolved shape is passed to executors.
	builder := backend.Builder("test_dynamic_iota")
	mainFn := builder.Main()

	// Parameter with dynamic "batch" dimension: [batch, 4]
	// The parameter is needed to establish the dynamic binding
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	// Create an Iota with dynamic shape [batch, 4] along axis 0
	// The iota output will have the same shape as x
	iotaShape := shapes.MakeDynamic(dtypes.Float32, "batch", 4)
	iota, err := mainFn.Iota(iotaShape, 0)
	require.NoError(t, err)

	// Add x and iota (both have shape [batch, 4])
	y, err := mainFn.Add(x, iota)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=3: input is [3, 4]
	// Iota along axis 0 with shape [3, 4] should produce:
	// [[0, 0, 0, 0],
	//  [1, 1, 1, 1],
	//  [2, 2, 2, 2]]
	input, err := backend.BufferFromFlatData(0, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}, shapes.Make(dtypes.Float32, 3, 4))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify output shape is [3, 4]
	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, dtypes.Float32, outputShape.DType)
	require.Equal(t, []int{3, 4}, outputShape.Dimensions)

	// Verify values: x + iota
	// Row 0: [1+0, 2+0, 3+0, 4+0] = [1, 2, 3, 4]
	// Row 1: [5+1, 6+1, 7+1, 8+1] = [6, 7, 8, 9]
	// Row 2: [9+2, 10+2, 11+2, 12+2] = [11, 12, 13, 14]
	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{
		1, 2, 3, 4,
		6, 7, 8, 9,
		11, 12, 13, 14,
	}, outputData)

	// Execute with batch=2
	input2, err := backend.BufferFromFlatData(0, []float32{
		10, 20, 30, 40,
		50, 60, 70, 80,
	}, shapes.Make(dtypes.Float32, 2, 4))
	require.NoError(t, err)

	outputs2, err := exec.Execute([]backends.Buffer{input2}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs2, 1)

	// Verify output shape is [2, 4]
	outputShape2, err := backend.BufferShape(outputs2[0])
	require.NoError(t, err)
	require.Equal(t, dtypes.Float32, outputShape2.DType)
	require.Equal(t, []int{2, 4}, outputShape2.Dimensions)

	// Verify values: x + iota
	// Row 0: [10+0, 20+0, 30+0, 40+0] = [10, 20, 30, 40]
	// Row 1: [50+1, 60+1, 70+1, 80+1] = [51, 61, 71, 81]
	outputData2 := outputs2[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{
		10, 20, 30, 40,
		51, 61, 71, 81,
	}, outputData2)
}

func TestBufferPoolIntegration(t *testing.T) {
	// Test that funcExecBuffers helper methods correctly use the specialization pool.
	// This tests the getBuffer, putBuffer, and cloneBuffer methods on funcExecBuffers.

	be := backend.(*Backend)

	// Create a specialization with known shapes
	builder := backend.Builder("test_buffer_pool_integration").(*Builder)
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	_, err = builder.Compile()
	require.NoError(t, err)

	// Create specialization
	bindings := shapes.AxisBindings{"batch": 8}
	spec := newSpecialization(builder, bindings)

	// Create a funcExecBuffers with the specialization pool
	execBuf := &funcExecBuffers{
		bufferPool: spec.bufferPool,
	}

	// Test getBuffer - should get from specialization pool
	buf := execBuf.getBuffer(be, dtypes.Float32, 32) // 8*4 = 32
	require.NotNil(t, buf)
	require.True(t, buf.valid)

	// Mark buffer for pool return tracking
	buf.shape = shapes.Make(dtypes.Float32, 8, 4)

	// Test putBuffer - should return to specialization pool
	execBuf.putBuffer(be, buf)
	require.False(t, buf.valid) // Should be invalidated

	// Get another buffer - should get the same one back from pool
	buf2 := execBuf.getBuffer(be, dtypes.Float32, 32)
	require.Same(t, buf, buf2, "should reuse pooled buffer")

	// Test getBuffer with unknown size - should fall back to backend pool
	buf3 := execBuf.getBuffer(be, dtypes.Float32, 999)
	require.NotNil(t, buf3)
	buf3.shape = shapes.Make(dtypes.Float32, 999)

	// Test putBuffer for unknown size - should go to backend pool
	execBuf.putBuffer(be, buf3)
	require.False(t, buf3.valid)

	// Test cloneBuffer
	srcBuf := execBuf.getBuffer(be, dtypes.Float32, 32)
	srcBuf.shape = shapes.Make(dtypes.Float32, 8, 4)
	srcFlat := srcBuf.flat.([]float32)
	for i := range srcFlat {
		srcFlat[i] = float32(i)
	}

	clonedBuf := execBuf.cloneBuffer(be, srcBuf)
	require.NotNil(t, clonedBuf)
	require.NotSame(t, srcBuf, clonedBuf)
	require.Equal(t, srcBuf.shape, clonedBuf.shape)

	// Verify the data was copied
	clonedFlat := clonedBuf.flat.([]float32)
	require.Equal(t, srcFlat, clonedFlat)

	// Clean up
	execBuf.putBuffer(be, srcBuf)
	execBuf.putBuffer(be, clonedBuf)

	// Test with nil bufferPool (should fall back to backend pool)
	execBufNoPool := &funcExecBuffers{
		bufferPool: nil,
	}

	buf4 := execBufNoPool.getBuffer(be, dtypes.Float32, 32)
	require.NotNil(t, buf4)
	buf4.shape = shapes.Make(dtypes.Float32, 32)
	execBufNoPool.putBuffer(be, buf4)
}
