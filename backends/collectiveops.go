package backends

// CollectiveOps is an interface for collective operations, that is, operations executed across multiple devices.
//
// These operations are essential for distributed/SPMD execution where computation is split across multiple
// devices (GPUs, TPUs, etc.). Each operation performs synchronized communication across all devices
// in the replica group.
//
// Collective operations are used in scenarios like:
//   - Data parallelism: AllReduce to sum gradients across replicas
//   - Model parallelism: AllGather to collect activations from different shards
//   - Batch normalization: CrossReplicaSum to compute statistics across the batch axis
//
// Note: These operations require the backend to be configured for multi-device execution.
// The execution semantics follow SPMD (Single Program, Multiple Data) model where the same
// computation runs on each device with different data.
type CollectiveOps interface {
	// AllReduce performs a reduction operation across all replicas and broadcasts the result back to all replicas.
	//
	// This is a collective operation that must be called by all replicas simultaneously.
	// Each replica contributes its local tensor, and the reduction (sum, max, min, etc.) is computed
	// across all replicas. The result is then broadcast to all replicas.
	//
	// Parameters:
	//   - input: The tensor to reduce. Must have the same shape across all replicas.
	//   - reduceOp: The reduction operation to apply (ReduceOpSum, ReduceOpMax, ReduceOpMin, ReduceOpProduct).
	//   - replicaGroups: Groups of replica IDs that participate in the reduction together.
	//     If nil or empty, all replicas form a single group.
	//     Example: [[0,1], [2,3]] means replicas 0,1 reduce together and replicas 2,3 reduce together.
	//
	// Returns: A tensor with the same shape as input, containing the reduced result.
	//
	// Example usage in gradient aggregation:
	//
	//	gradients := ... // local gradients on this replica
	//	aggregatedGradients := AllReduce(gradients, ReduceOpSum, nil) // sum across all replicas
	AllReduce(input Op, reduceOp ReduceOpType, replicaGroups [][]int) (Op, error)

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
	// Returns: A tensor where the gather axis dimension is multiplied by the number of replicas
	// in the group.
	//
	// Example: If there are 4 replicas, each with a tensor of shape [2, 3], and gatherAxis=0,
	// the result on each replica will be shape [8, 3].
	AllGather(input Op, gatherAxis int, replicaGroups [][]int) (Op, error)

	// ReduceScatter performs a reduction across all replicas and scatters the result.
	//
	// This is the inverse of AllGather followed by a reduction. It reduces the input tensors
	// from all replicas and then scatters (splits) the result so each replica gets a portion.
	//
	// Parameters:
	//   - input: The tensor to reduce and scatter. All replicas must have the same shape.
	//   - reduceOp: The reduction operation to apply (ReduceOpSum, ReduceOpMax, ReduceOpMin, ReduceOpProduct).
	//   - scatterAxis: The axis along which to scatter (split) the reduced result.
	//   - replicaGroups: Groups of replica IDs that participate together.
	//     If nil or empty, all replicas form a single group.
	//
	// Returns: A tensor where the scatter axis dimension is divided by the number of replicas
	// in the group.
	//
	// Example: If there are 4 replicas, each with a tensor of shape [8, 3], scatterAxis=0,
	// the result on each replica will be shape [2, 3] containing the reduced sum of that portion.
	ReduceScatter(input Op, reduceOp ReduceOpType, scatterAxis int, replicaGroups [][]int) (Op, error)

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
	//
	// Returns: A tensor with the same shape as input, containing the source replica's data
	// on all replicas.
	CollectiveBroadcast(input Op, sourceReplicaId int, replicaGroups [][]int) (Op, error)

	// CollectivePermute sends data from each replica to a specified target replica.
	//
	// This is a collective operation that performs point-to-point communication between replicas
	// according to a specified source-target mapping.
	//
	// Parameters:
	//   - input: The tensor to send. Each replica sends its own tensor.
	//   - sourceTargetPairs: A list of [source, target] pairs specifying which replica
	//     sends to which. Each replica can appear as source at most once and as target at most once.
	//     Example: [[0,1], [1,2], [2,0]] creates a circular permutation.
	//
	// Returns: A tensor with the same shape as input, containing the data received from the
	// source replica (as specified in sourceTargetPairs). If a replica is not a target in any pair,
	// it receives zeros.
	CollectivePermute(input Op, sourceTargetPairs [][2]int) (Op, error)

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
	ReplicaId() (Op, error)

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
	PartitionId() (Op, error)
}
