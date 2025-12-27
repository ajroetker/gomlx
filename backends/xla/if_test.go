package xla_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/stretchr/testify/require"
)

// TestIfOperation tests the If control flow operation using the graph API.
func TestIfOperation(t *testing.T) {
	// Test with predicate = true
	t.Run("PredicateTrue", func(t *testing.T) {
		result, err := graph.ExecOnce(backend, func(predicate *graph.Node) *graph.Node {
			g := predicate.Graph()
			fn := g.StableHLOFunction()
			require.NotNil(t, fn)

			// Create true branch: returns constant 1.0
			trueBranch := fn.Closure()
			trueValue, err := trueBranch.ConstantFromScalar(float32(1.0))
			require.NoError(t, err)
			trueBranch.Return(trueValue)

			// Create false branch: returns constant 2.0
			falseBranch := fn.Closure()
			falseValue, err := falseBranch.ConstantFromScalar(float32(2.0))
			require.NoError(t, err)
			falseBranch.Return(falseValue)

			// Call If operation
			return graph.If(predicate, trueBranch, falseBranch)[0]
		}, true)
		require.NoError(t, err)
		require.Equal(t, float32(1.0), result.Value())
	})

	// Test with predicate = false
	t.Run("PredicateFalse", func(t *testing.T) {
		result, err := graph.ExecOnce(backend, func(predicate *graph.Node) *graph.Node {
			g := predicate.Graph()
			fn := g.StableHLOFunction()
			require.NotNil(t, fn)

			// Create true branch: returns constant 1.0
			trueBranch := fn.Closure()
			trueValue, err := trueBranch.ConstantFromScalar(float32(1.0))
			require.NoError(t, err)
			trueBranch.Return(trueValue)

			// Create false branch: returns constant 2.0
			falseBranch := fn.Closure()
			falseValue, err := falseBranch.ConstantFromScalar(float32(2.0))
			require.NoError(t, err)
			falseBranch.Return(falseValue)

			// Call If operation
			return graph.If(predicate, trueBranch, falseBranch)[0]
		}, false)
		require.NoError(t, err)
		require.Equal(t, float32(2.0), result.Value())
	})
}

// TestIfMultipleOutputs tests the If operation with multiple outputs.
func TestIfMultipleOutputs(t *testing.T) {
	// Test with predicate = true
	t.Run("PredicateTrue", func(t *testing.T) {
		exec := graph.MustNewExec(backend, func(predicate *graph.Node) []*graph.Node {
			g := predicate.Graph()
			fn := g.StableHLOFunction()
			require.NotNil(t, fn)

			// True branch: returns (1.0, 10)
			trueBranch := fn.Closure()
			trueVal1, err := trueBranch.ConstantFromScalar(float32(1.0))
			require.NoError(t, err)
			trueVal2, err := trueBranch.ConstantFromScalar(int32(10))
			require.NoError(t, err)
			trueBranch.Return(trueVal1, trueVal2)

			// False branch: returns (2.0, 20)
			falseBranch := fn.Closure()
			falseVal1, err := falseBranch.ConstantFromScalar(float32(2.0))
			require.NoError(t, err)
			falseVal2, err := falseBranch.ConstantFromScalar(int32(20))
			require.NoError(t, err)
			falseBranch.Return(falseVal1, falseVal2)

			// Call If operation
			return graph.If(predicate, trueBranch, falseBranch)
		})

		results, err := exec.Exec(true)
		require.NoError(t, err)
		require.Len(t, results, 2)
		require.Equal(t, float32(1.0), results[0].Value())
		require.Equal(t, int32(10), results[1].Value())
	})

	// Test with predicate = false
	t.Run("PredicateFalse", func(t *testing.T) {
		exec := graph.MustNewExec(backend, func(predicate *graph.Node) []*graph.Node {
			g := predicate.Graph()
			fn := g.StableHLOFunction()
			require.NotNil(t, fn)

			// True branch: returns (1.0, 10)
			trueBranch := fn.Closure()
			trueVal1, err := trueBranch.ConstantFromScalar(float32(1.0))
			require.NoError(t, err)
			trueVal2, err := trueBranch.ConstantFromScalar(int32(10))
			require.NoError(t, err)
			trueBranch.Return(trueVal1, trueVal2)

			// False branch: returns (2.0, 20)
			falseBranch := fn.Closure()
			falseVal1, err := falseBranch.ConstantFromScalar(float32(2.0))
			require.NoError(t, err)
			falseVal2, err := falseBranch.ConstantFromScalar(int32(20))
			require.NoError(t, err)
			falseBranch.Return(falseVal1, falseVal2)

			// Call If operation
			return graph.If(predicate, trueBranch, falseBranch)
		})

		results, err := exec.Exec(false)
		require.NoError(t, err)
		require.Len(t, results, 2)
		require.Equal(t, float32(2.0), results[0].Value())
		require.Equal(t, int32(20), results[1].Value())
	})
}
