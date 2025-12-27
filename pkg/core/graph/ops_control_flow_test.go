package graph_test

import (
	"testing"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	stablehloshapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/xla"
)

// TestWhileCountTo10 tests a simple While loop that counts from 0 to 10
func TestWhileCountTo10(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	require.NotNil(t, backend, "XLA backend required for While loop tests")

	// Test a loop that computes: counter from 0 to 10
	exec := MustNewExec(backend, func(g *Graph) *Node {
		counter := Scalar(g, dtypes.Int32, int32(0))

		fn := g.StableHLOFunction()
		require.NotNil(t, fn, "StableHLOFunction should not be nil for XLA backend")

		// Condition: counter < 10
		condFn := fn.Closure()
		condInput, err := condFn.Input(xla.ShapeToXLA(counter.Shape()))
		require.NoError(t, err)
		limit, err := condFn.ConstantFromScalar(int32(10))
		require.NoError(t, err)
		cond, err := stablehlo.Compare(condInput, limit, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
		require.NoError(t, err)
		err = condFn.Return(cond)
		require.NoError(t, err)

		// Body: counter = counter + 1
		bodyFn := fn.Closure()
		bodyInput, err := bodyFn.Input(xla.ShapeToXLA(counter.Shape()))
		require.NoError(t, err)
		one, err := bodyFn.ConstantFromScalar(int32(1))
		require.NoError(t, err)
		next, err := stablehlo.Add(bodyInput, one)
		require.NoError(t, err)
		err = bodyFn.Return(next)
		require.NoError(t, err)

		// Execute While loop
		results := While(condFn, bodyFn, counter)
		require.Len(t, results, 1, "While should return 1 result")

		return results[0]
	})

	result := exec.MustExec()[0]
	value := result.Value().(int32)
	assert.Equal(t, int32(10), value, "Counter should reach 10")
}

// TestWhileMultipleStates tests While with multiple loop state variables
func TestWhileMultipleStates(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	require.NotNil(t, backend, "XLA backend required for While loop tests")

	// Test a loop that computes: counter from 0 to 5, sum = 1+2+3+4+5 = 15
	exec := MustNewExec(backend, func(g *Graph) *Node {
		counter := Scalar(g, dtypes.Int32, int32(0))
		sum := Scalar(g, dtypes.Int32, int32(0))

		fn := g.StableHLOFunction()

		// Condition: counter < 5
		condFn := fn.Closure()
		condCounter, _ := condFn.Input(xla.ShapeToXLA(counter.Shape()))
		condSum, _ := condFn.Input(xla.ShapeToXLA(sum.Shape()))
		_ = condSum // Not used in condition
		limit, _ := condFn.ConstantFromScalar(int32(5))
		cond, _ := stablehlo.Compare(condCounter, limit, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
		condFn.Return(cond)

		// Body: counter += 1, sum += counter
		bodyFn := fn.Closure()
		bodyCounter, _ := bodyFn.Input(xla.ShapeToXLA(counter.Shape()))
		bodySum, _ := bodyFn.Input(xla.ShapeToXLA(sum.Shape()))
		one, _ := bodyFn.ConstantFromScalar(int32(1))
		nextCounter, _ := stablehlo.Add(bodyCounter, one)
		nextSum, _ := stablehlo.Add(bodySum, nextCounter)
		bodyFn.Return(nextCounter, nextSum)

		results := While(condFn, bodyFn, counter, sum)
		// Return the sum
		return results[1]
	})

	result := exec.MustExec()[0]
	value := result.Value().(int32)
	// sum = 1 + 2 + 3 + 4 + 5 = 15
	assert.Equal(t, int32(15), value, "Sum should be 15")
}

// TestWhileTensorState tests While with tensor (non-scalar) state
func TestWhileTensorState(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	require.NotNil(t, backend, "XLA backend required for While loop tests")

	// Test incrementing a vector [0, 0, 0] to [5, 5, 5]
	exec := MustNewExec(backend, func(g *Graph) *Node {
		vec := Const(g, []int32{0, 0, 0})

		fn := g.StableHLOFunction()

		// Condition: check if first element < 5
		condFn := fn.Closure()
		condVec, _ := condFn.Input(xla.ShapeToXLA(vec.Shape()))
		firstElem, _ := stablehlo.Slice(condVec, []int{0}, []int{1}, []int{1})
		scalar, _ := stablehlo.Reshape(firstElem, stablehloshapes.Make(xla.DTypeToXLA(dtypes.Int32)))
		limit, _ := condFn.ConstantFromScalar(int32(5))
		cond, _ := stablehlo.Compare(scalar, limit, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
		condFn.Return(cond)

		// Body: add [1, 1, 1] to vector
		bodyFn := fn.Closure()
		bodyVec, _ := bodyFn.Input(xla.ShapeToXLA(vec.Shape()))
		ones, _ := bodyFn.ConstantFromFlatAndDimensions([]int32{1, 1, 1}, 3)
		nextVec, _ := stablehlo.Add(bodyVec, ones)
		bodyFn.Return(nextVec)

		results := While(condFn, bodyFn, vec)
		return results[0]
	})

	result := exec.MustExec()[0]
	value := result.Value().([]int32)
	assert.Equal(t, []int32{5, 5, 5}, value, "Vector should be [5, 5, 5]")
}

// TestClosureGraphIf tests the If operation with closure-backed branches
func TestClosureGraphIf(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	require.NotNil(t, backend, "Backend required for If tests")

	t.Run("simple conditional with closure - true", func(t *testing.T) {
		result := MustExecOnce(backend, func(cond *Node) *Node {
			g := cond.Graph()

			// Create parent values
			a := Scalar(g, dtypes.Float32, float32(5.0))
			b := Scalar(g, dtypes.Float32, float32(3.0))

			// Create closure graphs
			thenG := g.NewClosureGraph("then")
			elseG := g.NewClosureGraph("else")

			if thenG == nil || elseG == nil {
				t.Skip("Backend doesn't support closures")
				return nil
			}

			// Import parent values into closures (StableHLO inherits parent scope)
			thenA := thenG.UseParentValue(a)
			thenB := thenG.UseParentValue(b)
			elseA := elseG.UseParentValue(a)
			elseB := elseG.UseParentValue(b)

			// Build operations in closures
			thenResult := Add(thenA, thenB)
			elseResult := Sub(elseA, elseB)

			// Compile closures
			thenG.CompileClosure(thenResult)
			elseG.CompileClosure(elseResult)

			// Use IfClosure
			return IfClosure(cond, thenG, elseG)[0]
		}, true) // cond = true

		// Should return 5+3 = 8
		require.Equal(t, float32(8.0), result.Value().(float32))
	})

	t.Run("simple conditional with closure - false", func(t *testing.T) {
		result := MustExecOnce(backend, func(cond *Node) *Node {
			g := cond.Graph()

			// Create parent values
			a := Scalar(g, dtypes.Float32, float32(10.0))
			b := Scalar(g, dtypes.Float32, float32(4.0))

			// Create closure graphs
			thenG := g.NewClosureGraph("then")
			elseG := g.NewClosureGraph("else")

			if thenG == nil || elseG == nil {
				t.Skip("Backend doesn't support closures")
				return nil
			}

			// Import parent values into closures
			thenA := thenG.UseParentValue(a)
			thenB := thenG.UseParentValue(b)
			elseA := elseG.UseParentValue(a)
			elseB := elseG.UseParentValue(b)

			// Build operations in closures
			thenResult := Add(thenA, thenB)
			elseResult := Sub(elseA, elseB)

			// Compile closures
			thenG.CompileClosure(thenResult)
			elseG.CompileClosure(elseResult)

			// Use IfClosure
			return IfClosure(cond, thenG, elseG)[0]
		}, false) // cond = false

		// Should return 10-4 = 6
		require.Equal(t, float32(6.0), result.Value().(float32))
	})

	t.Run("multiple return values", func(t *testing.T) {
		exec := MustNewExec(backend, func(cond *Node) []*Node {
			g := cond.Graph()

			// Create parent values
			a := Scalar(g, dtypes.Float32, float32(5.0))
			b := Scalar(g, dtypes.Float32, float32(3.0))

			// Create closure graphs
			thenG := g.NewClosureGraph("then")
			elseG := g.NewClosureGraph("else")

			if thenG == nil || elseG == nil {
				t.Skip("Backend doesn't support closures")
				return nil
			}

			// Import parent values into closures
			thenA := thenG.UseParentValue(a)
			thenB := thenG.UseParentValue(b)
			elseA := elseG.UseParentValue(a)
			elseB := elseG.UseParentValue(b)

			// Build operations in closures - return two values
			thenResult1 := Add(thenA, thenB)
			thenResult2 := Mul(thenA, thenB)
			elseResult1 := Sub(elseA, elseB)
			elseResult2 := Div(elseA, elseB)

			// Compile closures
			thenG.CompileClosure(thenResult1, thenResult2)
			elseG.CompileClosure(elseResult1, elseResult2)

			// Use IfClosure
			return IfClosure(cond, thenG, elseG)
		})

		results := exec.MustExec(true) // cond = true

		// Should return (5+3=8, 5*3=15)
		require.Equal(t, float32(8.0), results[0].Value().(float32))
		require.Equal(t, float32(15.0), results[1].Value().(float32))
	})
}
