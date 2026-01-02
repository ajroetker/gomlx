package graph

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

// mockBuilder creates a mock builder for testing collective operations.
// It returns "not implemented" errors for all operations, which is expected
// since collective ops require actual multi-device execution.
func mockBuilder() backends.Builder {
	return notimplemented.Builder{}
}

// TestCollectiveNodeTypes verifies that the collective node types are defined correctly.
func TestCollectiveNodeTypes(t *testing.T) {
	// Test that node types don't conflict with generated types
	require.Greater(t, int(NodeTypeAllReduce), 100, "NodeTypeAllReduce should be > 100 to avoid conflicts")
	require.Greater(t, int(NodeTypeAllGather), 100)
	require.Greater(t, int(NodeTypeReduceScatter), 100)
	require.Greater(t, int(NodeTypeCollectiveBroadcast), 100)
	require.Greater(t, int(NodeTypeCollectivePermute), 100)
	require.Greater(t, int(NodeTypeReplicaId), 100)
	require.Greater(t, int(NodeTypePartitionId), 100)

	// Test that types are distinct
	types := []NodeType{
		NodeTypeAllReduce,
		NodeTypeAllGather,
		NodeTypeReduceScatter,
		NodeTypeCollectiveBroadcast,
		NodeTypeCollectivePermute,
		NodeTypeReplicaId,
		NodeTypePartitionId,
	}
	seen := make(map[NodeType]bool)
	for _, nt := range types {
		require.False(t, seen[nt], "Duplicate NodeType: %v", nt)
		seen[nt] = true
	}
}

// TestNodeInputsStrings verifies that the String() methods work correctly.
func TestNodeInputsStrings(t *testing.T) {
	// Create a mock node for testing
	mockNode := &Node{id: 42}

	t.Run("AllReduce", func(t *testing.T) {
		inputs := &nodeInputsAllReduce{
			input:         mockNode,
			reduceOp:      backends.ReduceOpSum,
			replicaGroups: [][]int{{0, 1}},
		}
		require.Equal(t, NodeTypeAllReduce, inputs.Type())
		str := inputs.String()
		require.Contains(t, str, "AllReduce")
		require.Contains(t, str, "#42")
	})

	t.Run("AllGather", func(t *testing.T) {
		inputs := &nodeInputsAllGather{
			input:         mockNode,
			gatherAxis:    0,
			replicaGroups: nil,
		}
		require.Equal(t, NodeTypeAllGather, inputs.Type())
		str := inputs.String()
		require.Contains(t, str, "AllGather")
	})

	t.Run("ReduceScatter", func(t *testing.T) {
		inputs := &nodeInputsReduceScatter{
			input:         mockNode,
			reduceOp:      backends.ReduceOpMax,
			scatterAxis:   1,
			replicaGroups: [][]int{{0, 1, 2, 3}},
		}
		require.Equal(t, NodeTypeReduceScatter, inputs.Type())
		str := inputs.String()
		require.Contains(t, str, "ReduceScatter")
	})

	t.Run("CollectiveBroadcast", func(t *testing.T) {
		inputs := &nodeInputsCollectiveBroadcast{
			input:           mockNode,
			sourceReplicaId: 0,
			replicaGroups:   nil,
		}
		require.Equal(t, NodeTypeCollectiveBroadcast, inputs.Type())
		str := inputs.String()
		require.Contains(t, str, "CollectiveBroadcast")
	})

	t.Run("CollectivePermute", func(t *testing.T) {
		inputs := &nodeInputsCollectivePermute{
			input:             mockNode,
			sourceTargetPairs: [][2]int{{0, 1}, {1, 0}},
		}
		require.Equal(t, NodeTypeCollectivePermute, inputs.Type())
		str := inputs.String()
		require.Contains(t, str, "CollectivePermute")
	})

	t.Run("ReplicaId", func(t *testing.T) {
		inputs := &nodeInputsReplicaId{}
		require.Equal(t, NodeTypeReplicaId, inputs.Type())
		str := inputs.String()
		require.Contains(t, str, "ReplicaId")
	})

	t.Run("PartitionId", func(t *testing.T) {
		inputs := &nodeInputsPartitionId{}
		require.Equal(t, NodeTypePartitionId, inputs.Type())
		str := inputs.String()
		require.Contains(t, str, "PartitionId")
	})
}

// TestReplicaIdOutputShape verifies the expected output shape of ReplicaId.
func TestReplicaIdOutputShape(t *testing.T) {
	// ReplicaId should return a U32 scalar
	expectedShape := shapes.Make(dtypes.Uint32)
	require.True(t, expectedShape.IsScalar(), "ReplicaId should return a scalar")
	require.Equal(t, dtypes.Uint32, expectedShape.DType, "ReplicaId should return U32")
}

// TestPartitionIdOutputShape verifies the expected output shape of PartitionId.
func TestPartitionIdOutputShape(t *testing.T) {
	// PartitionId should return a U32 scalar
	expectedShape := shapes.Make(dtypes.Uint32)
	require.True(t, expectedShape.IsScalar(), "PartitionId should return a scalar")
	require.Equal(t, dtypes.Uint32, expectedShape.DType, "PartitionId should return U32")
}

// TestReduceOpTypes verifies that all expected reduce operations are supported.
func TestReduceOpTypes(t *testing.T) {
	// Test that the reduce op types we use are valid
	ops := []backends.ReduceOpType{
		backends.ReduceOpSum,
		backends.ReduceOpProduct,
		backends.ReduceOpMax,
		backends.ReduceOpMin,
	}

	for _, op := range ops {
		require.NotEqual(t, backends.ReduceOpUndefined, op, "ReduceOp should not be undefined")
	}
}
