// Package graph contains the collective operations for distributed/SPMD execution.
package graph

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// NodeType constants for collective operations.
const (
	NodeTypeAllReduce NodeType = iota + 1000 // Start at 1000 to avoid conflicts with generated types
	NodeTypeAllGather
	NodeTypeReduceScatter
	NodeTypeCollectiveBroadcast
	NodeTypeCollectivePermute
	NodeTypeReplicaId
	NodeTypePartitionId
)

// nodeInputsAllReduce holds the inputs used for the AllReduce operation.
type nodeInputsAllReduce struct {
	input         *Node
	reduceOp      backends.ReduceOpType
	replicaGroups [][]int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsAllReduce) Type() NodeType {
	return NodeTypeAllReduce
}

// String implements the interface NodeInputs.
func (ni *nodeInputsAllReduce) String() string {
	return fmt.Sprintf("AllReduce(input=[#%d], reduceOp=%s, replicaGroups=%v)",
		ni.input.Id(), ni.reduceOp, ni.replicaGroups)
}

// AllReduce performs a reduction operation across all replicas and broadcasts the result back to all replicas.
//
// This is a collective operation that must be called by all replicas simultaneously.
// Each replica contributes its local tensor, and the reduction (sum, max, min, etc.) is computed
// across all replicas. The result is then broadcast to all replicas.
//
// Parameters:
//   - input: The tensor to reduce. Must have the same shape across all replicas.
//   - reduceOp: The reduction operation to apply (backends.ReduceOpSum, ReduceOpMax, ReduceOpMin, ReduceOpProduct).
//   - replicaGroups: Groups of replica IDs that participate in the reduction together.
//     If nil or empty, all replicas form a single group.
//
// Example usage in gradient aggregation:
//
//	gradients := ... // local gradients on this replica
//	aggregatedGradients := AllReduce(gradients, backends.ReduceOpSum, nil) // sum across all replicas
func AllReduce(input *Node, reduceOp backends.ReduceOpType, replicaGroups [][]int) *Node {
	g := validateBuildingGraphFromInputs(input)
	inputs := &nodeInputsAllReduce{
		input:         input,
		reduceOp:      reduceOp,
		replicaGroups: replicaGroups,
	}
	result, err := g.builder.AllReduce(input.outputOps[0], reduceOp, replicaGroups)
	if err != nil {
		panic(err)
	}
	node := &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       inputs,
	}
	g.registerNode(node)
	return node
}

// nodeInputsAllGather holds the inputs used for the AllGather operation.
type nodeInputsAllGather struct {
	input         *Node
	gatherAxis    int
	replicaGroups [][]int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsAllGather) Type() NodeType {
	return NodeTypeAllGather
}

// String implements the interface NodeInputs.
func (ni *nodeInputsAllGather) String() string {
	return fmt.Sprintf("AllGather(input=[#%d], gatherAxis=%d, replicaGroups=%v)",
		ni.input.Id(), ni.gatherAxis, ni.replicaGroups)
}

// AllGather gathers tensors from all replicas and concatenates them along the specified axis.
//
// This is a collective operation that must be called by all replicas simultaneously.
// Each replica contributes its local tensor, and the result on each replica is the concatenation
// of all contributions along the gather axis.
//
// Parameters:
//   - input: The tensor to gather. All replicas must have the same shape.
//   - gatherAxis: The axis along which to concatenate the gathered tensors.
//   - replicaGroups: Groups of replica IDs that participate in the gather together.
//     If nil or empty, all replicas form a single group.
//
// Example: If there are 4 replicas, each with a tensor of shape [2, 3], and gatherAxis=0,
// the result on each replica will be shape [8, 3].
func AllGather(input *Node, gatherAxis int, replicaGroups [][]int) *Node {
	g := validateBuildingGraphFromInputs(input)
	inputs := &nodeInputsAllGather{
		input:         input,
		gatherAxis:    gatherAxis,
		replicaGroups: replicaGroups,
	}
	result, err := g.builder.AllGather(input.outputOps[0], gatherAxis, replicaGroups)
	if err != nil {
		panic(err)
	}
	node := &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       inputs,
	}
	g.registerNode(node)
	return node
}

// nodeInputsReduceScatter holds the inputs used for the ReduceScatter operation.
type nodeInputsReduceScatter struct {
	input         *Node
	reduceOp      backends.ReduceOpType
	scatterAxis   int
	replicaGroups [][]int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsReduceScatter) Type() NodeType {
	return NodeTypeReduceScatter
}

// String implements the interface NodeInputs.
func (ni *nodeInputsReduceScatter) String() string {
	return fmt.Sprintf("ReduceScatter(input=[#%d], reduceOp=%s, scatterAxis=%d, replicaGroups=%v)",
		ni.input.Id(), ni.reduceOp, ni.scatterAxis, ni.replicaGroups)
}

// ReduceScatter performs a reduction across all replicas and scatters the result.
//
// This is the inverse of AllGather followed by a reduction. It reduces the input tensors
// from all replicas and then scatters (splits) the result so each replica gets a portion.
//
// Parameters:
//   - input: The tensor to reduce and scatter. All replicas must have the same shape.
//   - reduceOp: The reduction operation to apply (backends.ReduceOpSum, ReduceOpMax, ReduceOpMin, ReduceOpProduct).
//   - scatterAxis: The axis along which to scatter (split) the reduced result.
//   - replicaGroups: Groups of replica IDs that participate together.
//     If nil or empty, all replicas form a single group.
//
// Example: If there are 4 replicas, each with a tensor of shape [8, 3], scatterAxis=0,
// the result on each replica will be shape [2, 3] containing the reduced sum of that portion.
func ReduceScatter(input *Node, reduceOp backends.ReduceOpType, scatterAxis int, replicaGroups [][]int) *Node {
	g := validateBuildingGraphFromInputs(input)
	inputs := &nodeInputsReduceScatter{
		input:         input,
		reduceOp:      reduceOp,
		scatterAxis:   scatterAxis,
		replicaGroups: replicaGroups,
	}
	result, err := g.builder.ReduceScatter(input.outputOps[0], reduceOp, scatterAxis, replicaGroups)
	if err != nil {
		panic(err)
	}
	node := &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       inputs,
	}
	g.registerNode(node)
	return node
}

// nodeInputsCollectiveBroadcast holds the inputs used for the CollectiveBroadcast operation.
type nodeInputsCollectiveBroadcast struct {
	input           *Node
	sourceReplicaId int
	replicaGroups   [][]int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsCollectiveBroadcast) Type() NodeType {
	return NodeTypeCollectiveBroadcast
}

// String implements the interface NodeInputs.
func (ni *nodeInputsCollectiveBroadcast) String() string {
	return fmt.Sprintf("CollectiveBroadcast(input=[#%d], sourceReplicaId=%d, replicaGroups=%v)",
		ni.input.Id(), ni.sourceReplicaId, ni.replicaGroups)
}

// CollectiveBroadcast broadcasts a tensor from one replica to all other replicas.
//
// This is a collective operation where one replica (the source) sends its tensor to all
// other replicas in the group.
//
// Parameters:
//   - input: The tensor to broadcast. Only the source replica's value is used.
//   - sourceReplicaId: The replica ID that broadcasts its value.
//   - replicaGroups: Groups of replica IDs that participate in the broadcast together.
//     If nil or empty, all replicas form a single group.
func CollectiveBroadcast(input *Node, sourceReplicaId int, replicaGroups [][]int) *Node {
	g := validateBuildingGraphFromInputs(input)
	inputs := &nodeInputsCollectiveBroadcast{
		input:           input,
		sourceReplicaId: sourceReplicaId,
		replicaGroups:   replicaGroups,
	}
	result, err := g.builder.CollectiveBroadcast(input.outputOps[0], sourceReplicaId, replicaGroups)
	if err != nil {
		panic(err)
	}
	node := &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       inputs,
	}
	g.registerNode(node)
	return node
}

// nodeInputsCollectivePermute holds the inputs used for the CollectivePermute operation.
type nodeInputsCollectivePermute struct {
	input             *Node
	sourceTargetPairs [][2]int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsCollectivePermute) Type() NodeType {
	return NodeTypeCollectivePermute
}

// String implements the interface NodeInputs.
func (ni *nodeInputsCollectivePermute) String() string {
	return fmt.Sprintf("CollectivePermute(input=[#%d], sourceTargetPairs=%v)",
		ni.input.Id(), ni.sourceTargetPairs)
}

// CollectivePermute sends data from each replica to a specified target replica.
//
// This is a collective operation that performs point-to-point communication between replicas
// according to a specified source-target mapping.
//
// Parameters:
//   - input: The tensor to send. Each replica sends its own tensor.
//   - sourceTargetPairs: A list of [source, target] pairs specifying which replica
//     sends to which. Each replica can appear as source at most once and as target at most once.
//     Example: [][2]int{{0,1}, {1,2}, {2,0}} creates a circular permutation.
func CollectivePermute(input *Node, sourceTargetPairs [][2]int) *Node {
	g := validateBuildingGraphFromInputs(input)
	inputs := &nodeInputsCollectivePermute{
		input:             input,
		sourceTargetPairs: sourceTargetPairs,
	}
	result, err := g.builder.CollectivePermute(input.outputOps[0], sourceTargetPairs)
	if err != nil {
		panic(err)
	}
	node := &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       inputs,
	}
	g.registerNode(node)
	return node
}

// nodeInputsReplicaId holds the inputs used for the ReplicaId operation.
type nodeInputsReplicaId struct{}

// Type implements the interface NodeInputs.
func (ni *nodeInputsReplicaId) Type() NodeType {
	return NodeTypeReplicaId
}

// String implements the interface NodeInputs.
func (ni *nodeInputsReplicaId) String() string {
	return "ReplicaId()"
}

// ReplicaId returns a scalar tensor containing the replica ID of the current device.
//
// The replica ID is a unique integer assigned to each participating device in the SPMD execution.
// Replica IDs range from 0 to (numReplicas - 1).
//
// This operation is useful for:
//   - Conditional execution based on replica ID
//   - Debugging and logging in distributed settings
//   - Implementing asymmetric computation patterns
//
// Returns: A scalar U32 tensor containing the replica ID.
func ReplicaId(g *Graph) *Node {
	g.AssertBuilding()
	inputs := &nodeInputsReplicaId{}
	result, err := g.builder.ReplicaId()
	if err != nil {
		panic(err)
	}
	node := &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{shapes.Make(dtypes.Uint32)}, // ReplicaId returns U32 scalar
		inputs:       inputs,
	}
	g.registerNode(node)
	return node
}

// nodeInputsPartitionId holds the inputs used for the PartitionId operation.
type nodeInputsPartitionId struct{}

// Type implements the interface NodeInputs.
func (ni *nodeInputsPartitionId) Type() NodeType {
	return NodeTypePartitionId
}

// String implements the interface NodeInputs.
func (ni *nodeInputsPartitionId) String() string {
	return "PartitionId()"
}

// PartitionId returns a scalar tensor containing the partition ID of the current device.
//
// In GSPMD (General SPMD) execution, computation can be partitioned across devices.
// The partition ID identifies which partition the current device belongs to.
// Partition IDs range from 0 to (numPartitions - 1).
//
// Note: This is primarily used with the GSPMD strategy. For SimpleSPMD,
// typically only replica IDs are used.
//
// Returns: A scalar U32 tensor containing the partition ID.
func PartitionId(g *Graph) *Node {
	g.AssertBuilding()
	inputs := &nodeInputsPartitionId{}
	result, err := g.builder.PartitionId()
	if err != nil {
		panic(err)
	}
	node := &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{shapes.Make(dtypes.Uint32)}, // PartitionId returns U32 scalar
		inputs:       inputs,
	}
	g.registerNode(node)
	return node
}

// CrossReplicaSum is a convenience function that performs an AllReduce with sum reduction.
// This is a common pattern in distributed training for gradient aggregation.
//
// Parameters:
//   - input: The tensor to sum across replicas.
//   - replicaGroups: Groups of replica IDs that participate in the reduction together.
//     If nil or empty, all replicas form a single group.
func CrossReplicaSum(input *Node, replicaGroups [][]int) *Node {
	return AllReduce(input, backends.ReduceOpSum, replicaGroups)
}
