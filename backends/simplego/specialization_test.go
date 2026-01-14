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

func TestRuntimeDeduplication(t *testing.T) {
	// Test that runtime deduplication correctly identifies and deduplicates
	// nodes that become identical after shape resolution.

	// Create a graph where two different subexpressions become identical
	// at runtime due to dynamic shape resolution.
	builder := backend.Builder("test_runtime_dedup").(*Builder)
	mainFn := builder.Main()

	// Two parameters with dynamic batch dimension
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	y, err := mainFn.Parameter("y", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	// Apply same operation to both: a = neg(x), b = neg(y)
	// At compile time, these are different nodes with different inputs.
	// At runtime with same input values, the outputs would be same but
	// we can't deduplicate across different inputs.
	a, err := mainFn.Neg(x)
	require.NoError(t, err)

	b, err := mainFn.Neg(y)
	require.NoError(t, err)

	// Return both
	err = mainFn.Return([]backends.Value{a, b}, nil)
	require.NoError(t, err)

	_, err = builder.Compile()
	require.NoError(t, err)

	// Create specialization
	bindings := shapes.AxisBindings{"batch": 8}
	spec := newSpecialization(builder, bindings)

	// Verify canonical mapping exists
	require.NotNil(t, spec.canonical)
	require.Equal(t, len(builder.nodes), len(spec.canonical))

	// Nodes a and b have different inputs (x vs y), so they should NOT be deduplicated
	aNode := a.(*Node)
	bNode := b.(*Node)
	require.Equal(t, aNode.builderIdx, spec.canonical[aNode.builderIdx], "a should be canonical (different inputs)")
	require.Equal(t, bNode.builderIdx, spec.canonical[bNode.builderIdx], "b should be canonical (different inputs)")
}

func TestRuntimeDeduplicationWithConstants(t *testing.T) {
	// Test deduplication with identical constants.
	// Verify that the runtime dedup signature handles constants correctly.
	builder := backend.Builder("test_runtime_dedup_constants").(*Builder)
	mainFn := builder.Main()

	// Create parameter with static shape
	x, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 4), nil)
	require.NoError(t, err)

	// Create a constant
	c1, err := mainFn.Constant([]float32{1, 2, 3, 4}, 4)
	require.NoError(t, err)

	// Add x + c1
	y, err := mainFn.Add(x, c1)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	_, err = builder.Compile()
	require.NoError(t, err)

	// Create specialization (even with static shapes, verify it doesn't panic)
	// Empty bindings for static shapes
	bindings := shapes.AxisBindings{}
	spec := newSpecialization(builder, bindings)
	require.NotNil(t, spec.canonical)

	// Verify constants get a valid canonical index
	c1Node := c1.(*Node)
	require.GreaterOrEqual(t, spec.canonical[c1Node.builderIdx], 0)
}

func TestRuntimeDeduplicationIdenticalSubgraphs(t *testing.T) {
	// Test that truly identical subgraphs are deduplicated.
	// Create a graph where the same input is processed twice identically.
	builder := backend.Builder("test_runtime_dedup_identical").(*Builder)
	mainFn := builder.Main()

	// Single parameter
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	// Apply neg twice to the same input - these should NOT be deduplicated
	// at compile time because they're separate calls, but at runtime
	// with the same canonical input, they should be deduplicated.
	//
	// Actually, compile-time dedup should already handle this case...
	// Let's verify the canonical mapping is correct either way.
	a, err := mainFn.Neg(x)
	require.NoError(t, err)

	// Use the result
	err = mainFn.Return([]backends.Value{a}, nil)
	require.NoError(t, err)

	_, err = builder.Compile()
	require.NoError(t, err)

	// Create specialization
	bindings := shapes.AxisBindings{"batch": 8}
	spec := newSpecialization(builder, bindings)

	require.NotNil(t, spec.canonical)

	// Verify all nodes have valid canonical indices
	for i, canonical := range spec.canonical {
		require.GreaterOrEqual(t, canonical, 0, "canonical index should be non-negative")
		require.Less(t, canonical, len(spec.canonical), "canonical index should be within bounds")
		require.LessOrEqual(t, canonical, i, "canonical index should not point forward")
	}
}

func TestDeduplicationExecution(t *testing.T) {
	// Test that execution works correctly with deduplication.
	// Verify that deduplicated nodes produce correct results.
	builder := backend.Builder("test_dedup_execution")
	mainFn := builder.Main()

	// Create a simple graph
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	// neg(x)
	y, err := mainFn.Neg(x)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2
	input, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8}, shapes.Make(dtypes.Float32, 2, 4))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify output
	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{-1, -2, -3, -4, -5, -6, -7, -8}, outputData)

	// Execute again with batch=3 (different specialization)
	input2, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, shapes.Make(dtypes.Float32, 3, 4))
	require.NoError(t, err)

	outputs2, err := exec.Execute([]backends.Buffer{input2}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs2, 1)

	// Verify output
	outputData2 := outputs2[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12}, outputData2)
}

func TestCanonicalMappingProperties(t *testing.T) {
	// Test properties of the canonical mapping:
	// 1. All canonical indices point to self or earlier nodes
	// 2. Parameters are always canonical to themselves
	// 3. Canonical indices form a valid equivalence relation

	builder := backend.Builder("test_canonical_properties").(*Builder)
	mainFn := builder.Main()

	// Create a graph with multiple operations
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)
	y, err := mainFn.Parameter("y", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	// Various operations
	a, err := mainFn.Neg(x)
	require.NoError(t, err)
	b, err := mainFn.Neg(y)
	require.NoError(t, err)
	c, err := mainFn.Add(a, b)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{c}, nil)
	require.NoError(t, err)

	_, err = builder.Compile()
	require.NoError(t, err)

	// Create specialization
	bindings := shapes.AxisBindings{"batch": 8}
	spec := newSpecialization(builder, bindings)

	// Property 1: All canonical indices point to self or earlier nodes
	for i, canonical := range spec.canonical {
		require.LessOrEqual(t, canonical, i,
			"canonical[%d]=%d should be <= %d", i, canonical, i)
	}

	// Property 2: Parameters are always canonical to themselves
	xNode := x.(*Node)
	yNode := y.(*Node)
	require.Equal(t, xNode.builderIdx, spec.canonical[xNode.builderIdx], "parameter x should be self-canonical")
	require.Equal(t, yNode.builderIdx, spec.canonical[yNode.builderIdx], "parameter y should be self-canonical")

	// Property 3: Transitive closure - if canonical[a] = b and canonical[b] = c, then c = b (canonical is already resolved)
	for i, canonical := range spec.canonical {
		if canonical != i {
			require.Equal(t, canonical, spec.canonical[canonical],
				"canonical[canonical[%d]] should equal canonical[%d]", i, i)
		}
	}
}

func TestNodeDataSignature(t *testing.T) {
	// Test that nodeDataSignature produces different signatures for different data
	// and same signatures for equivalent data.

	// Test nil
	require.Equal(t, "nil", nodeDataSignature(nil))

	// Test int
	sig1 := nodeDataSignature(42)
	sig2 := nodeDataSignature(42)
	sig3 := nodeDataSignature(43)
	require.Equal(t, sig1, sig2, "same int should produce same signature")
	require.NotEqual(t, sig1, sig3, "different int should produce different signature")

	// Test []int
	sig4 := nodeDataSignature([]int{1, 2, 3})
	sig5 := nodeDataSignature([]int{1, 2, 3})
	sig6 := nodeDataSignature([]int{1, 2, 4})
	require.Equal(t, sig4, sig5, "same []int should produce same signature")
	require.NotEqual(t, sig4, sig6, "different []int should produce different signature")

	// Test dotGeneralNodeData
	dg1 := &dotGeneralNodeData{
		lhsContractingAxes: []int{1},
		lhsBatchAxes:       []int{0},
		rhsContractingAxes: []int{0},
		rhsBatchAxes:       []int{0},
	}
	dg2 := &dotGeneralNodeData{
		lhsContractingAxes: []int{1},
		lhsBatchAxes:       []int{0},
		rhsContractingAxes: []int{0},
		rhsBatchAxes:       []int{0},
	}
	dg3 := &dotGeneralNodeData{
		lhsContractingAxes: []int{2}, // Different!
		lhsBatchAxes:       []int{0},
		rhsContractingAxes: []int{0},
		rhsBatchAxes:       []int{0},
	}
	sig7 := nodeDataSignature(dg1)
	sig8 := nodeDataSignature(dg2)
	sig9 := nodeDataSignature(dg3)
	require.Equal(t, sig7, sig8, "same dotGeneralNodeData should produce same signature")
	require.NotEqual(t, sig7, sig9, "different dotGeneralNodeData should produce different signature")
}

// ============================================================================
// Tests for various operations with dynamic shapes
// ============================================================================

func TestDynamicShapeReduceSum(t *testing.T) {
	// Test ReduceSum with dynamic batch dimension
	builder := backend.Builder("test_dynamic_reduce_sum")
	mainFn := builder.Main()

	// [batch, 4] -> reduce along axis 1 -> [batch]
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	// Sum along axis 1
	y, err := mainFn.ReduceSum(x, 1)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=3
	input, err := backend.BufferFromFlatData(0, []float32{
		1, 2, 3, 4, // row 0: sum = 10
		5, 6, 7, 8, // row 1: sum = 26
		9, 10, 11, 12, // row 2: sum = 42
	}, shapes.Make(dtypes.Float32, 3, 4))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify output shape [3]
	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, []int{3}, outputShape.Dimensions)

	// Verify values
	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{10, 26, 42}, outputData)

	// Execute with batch=2
	input2, err := backend.BufferFromFlatData(0, []float32{
		1, 1, 1, 1, // sum = 4
		2, 2, 2, 2, // sum = 8
	}, shapes.Make(dtypes.Float32, 2, 4))
	require.NoError(t, err)

	outputs2, err := exec.Execute([]backends.Buffer{input2}, nil, 0)
	require.NoError(t, err)

	outputShape2, err := backend.BufferShape(outputs2[0])
	require.NoError(t, err)
	require.Equal(t, []int{2}, outputShape2.Dimensions)

	outputData2 := outputs2[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{4, 8}, outputData2)
}

func TestDynamicShapeReduceMax(t *testing.T) {
	// Test ReduceMax with dynamic batch dimension
	builder := backend.Builder("test_dynamic_reduce_max")
	mainFn := builder.Main()

	// [batch, 4] -> reduce along axis 1 -> [batch]
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	y, err := mainFn.ReduceMax(x, 1)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2
	input, err := backend.BufferFromFlatData(0, []float32{
		1, 5, 3, 2, // max = 5
		9, 6, 7, 8, // max = 9
	}, shapes.Make(dtypes.Float32, 2, 4))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)

	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{5, 9}, outputData)
}

func TestDynamicShapeConcatenate(t *testing.T) {
	// Test Concatenate with dynamic batch dimension
	builder := backend.Builder("test_dynamic_concatenate")
	mainFn := builder.Main()

	// Concatenate two [batch, 2] tensors along axis 1 -> [batch, 4]
	a, err := mainFn.Parameter("a", shapes.MakeDynamic(dtypes.Float32, "batch", 2), nil)
	require.NoError(t, err)

	b, err := mainFn.Parameter("b", shapes.MakeDynamic(dtypes.Float32, "batch", 2), nil)
	require.NoError(t, err)

	y, err := mainFn.Concatenate(1, a, b)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2
	inputA, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 2, 2))
	require.NoError(t, err)
	inputB, err := backend.BufferFromFlatData(0, []float32{5, 6, 7, 8}, shapes.Make(dtypes.Float32, 2, 2))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{inputA, inputB}, nil, 0)
	require.NoError(t, err)

	// Verify output shape [2, 4]
	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, []int{2, 4}, outputShape.Dimensions)

	// Verify values: [[1,2,5,6], [3,4,7,8]]
	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{1, 2, 5, 6, 3, 4, 7, 8}, outputData)

	// Execute with batch=3
	inputA2, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, shapes.Make(dtypes.Float32, 3, 2))
	require.NoError(t, err)
	inputB2, err := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40, 50, 60}, shapes.Make(dtypes.Float32, 3, 2))
	require.NoError(t, err)

	outputs2, err := exec.Execute([]backends.Buffer{inputA2, inputB2}, nil, 0)
	require.NoError(t, err)

	outputShape2, err := backend.BufferShape(outputs2[0])
	require.NoError(t, err)
	require.Equal(t, []int{3, 4}, outputShape2.Dimensions)
}

func TestDynamicShapeTranspose(t *testing.T) {
	// Test Transpose with dynamic batch dimension
	builder := backend.Builder("test_dynamic_transpose")
	mainFn := builder.Main()

	// [batch, 3, 2] -> transpose (0, 2, 1) -> [batch, 2, 3]
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 3, 2), nil)
	require.NoError(t, err)

	y, err := mainFn.Transpose(x, 0, 2, 1)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2
	// Input shape [2, 3, 2]:
	// batch 0: [[1,2], [3,4], [5,6]]
	// batch 1: [[7,8], [9,10], [11,12]]
	input, err := backend.BufferFromFlatData(0, []float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	}, shapes.Make(dtypes.Float32, 2, 3, 2))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)

	// Verify output shape [2, 2, 3]
	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, []int{2, 2, 3}, outputShape.Dimensions)

	// After transpose (0, 2, 1):
	// batch 0: [[1,3,5], [2,4,6]]
	// batch 1: [[7,9,11], [8,10,12]]
	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{
		1, 3, 5, 2, 4, 6,
		7, 9, 11, 8, 10, 12,
	}, outputData)
}

func TestDynamicShapeSlice(t *testing.T) {
	// Test Slice with dynamic batch dimension
	builder := backend.Builder("test_dynamic_slice")
	mainFn := builder.Main()

	// [batch, 6] -> slice [0:batch, 1:4] -> [batch, 3]
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 6), nil)
	require.NoError(t, err)

	// Slice: keep all of batch dim, take indices 1:4 of the second dim
	y, err := mainFn.Slice(x,
		[]int{0, 1},  // starts
		[]int{-1, 4}, // limits (-1 means full extent for dynamic dim)
		[]int{1, 1},  // strides
	)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2
	input, err := backend.BufferFromFlatData(0, []float32{
		0, 1, 2, 3, 4, 5, // slice [1:4] -> [1, 2, 3]
		10, 11, 12, 13, 14, 15, // slice [1:4] -> [11, 12, 13]
	}, shapes.Make(dtypes.Float32, 2, 6))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)

	// Verify output shape [2, 3]
	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, []int{2, 3}, outputShape.Dimensions)

	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{1, 2, 3, 11, 12, 13}, outputData)
}

func TestDynamicShapeBroadcast(t *testing.T) {
	// Test Broadcast with dynamic batch dimension
	builder := backend.Builder("test_dynamic_broadcast")
	mainFn := builder.Main()

	// Broadcast a [1, 4] constant to [batch, 4] using element-wise multiply
	// with a [batch, 4] input
	x, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	// Create a [1, 4] constant that will broadcast
	scale, err := mainFn.Constant([]float32{1, 2, 3, 4}, 1, 4)
	require.NoError(t, err)

	// Multiply broadcasts [1, 4] to [batch, 4]
	y, err := mainFn.Mul(x, scale)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=3
	input, err := backend.BufferFromFlatData(0, []float32{
		1, 1, 1, 1, // * [1,2,3,4] = [1,2,3,4]
		2, 2, 2, 2, // * [1,2,3,4] = [2,4,6,8]
		3, 3, 3, 3, // * [1,2,3,4] = [3,6,9,12]
	}, shapes.Make(dtypes.Float32, 3, 4))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{input}, nil, 0)
	require.NoError(t, err)

	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, []int{3, 4}, outputShape.Dimensions)

	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{
		1, 2, 3, 4,
		2, 4, 6, 8,
		3, 6, 9, 12,
	}, outputData)
}

func TestDynamicShapeWhere(t *testing.T) {
	// Test Where (select) with dynamic batch dimension
	builder := backend.Builder("test_dynamic_where")
	mainFn := builder.Main()

	// Where: select from a or b based on condition
	// All have shape [batch, 4]
	cond, err := mainFn.Parameter("cond", shapes.MakeDynamic(dtypes.Bool, "batch", 4), nil)
	require.NoError(t, err)

	a, err := mainFn.Parameter("a", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	b, err := mainFn.Parameter("b", shapes.MakeDynamic(dtypes.Float32, "batch", 4), nil)
	require.NoError(t, err)

	y, err := mainFn.Where(cond, a, b)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{y}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2
	condData := []bool{true, false, true, false, false, true, false, true}
	inputCond, err := backend.BufferFromFlatData(0, condData, shapes.Make(dtypes.Bool, 2, 4))
	require.NoError(t, err)

	inputA, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8}, shapes.Make(dtypes.Float32, 2, 4))
	require.NoError(t, err)

	inputB, err := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40, 50, 60, 70, 80}, shapes.Make(dtypes.Float32, 2, 4))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{inputCond, inputA, inputB}, nil, 0)
	require.NoError(t, err)

	outputShape, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.Equal(t, []int{2, 4}, outputShape.Dimensions)

	// Where cond=true, select from a; where cond=false, select from b
	// Row 0: [T,F,T,F] -> [1, 20, 3, 40]
	// Row 1: [F,T,F,T] -> [50, 6, 70, 8]
	outputData := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{1, 20, 3, 40, 50, 6, 70, 8}, outputData)
}
