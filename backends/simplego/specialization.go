// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"reflect"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// ShapeSpecialization holds resolved shapes for a specific axis binding.
// When a graph has dynamic axes (like "batch"), each unique binding creates
// a specialization that caches the concrete shapes for efficient execution.
type ShapeSpecialization struct {
	// bindings are the axis name to value mappings (e.g., {"batch": 32, "seq": 128}).
	bindings shapes.AxisBindings

	// nodeShapes holds the resolved shape for each node in builder.nodes.
	// Indexed by node.builderIdx.
	nodeShapes []shapes.Shape

	// bufferPool provides exact-sized buffer pools for this specialization.
	// Pre-created for all unique (dtype, size) combinations in nodeShapes.
	bufferPool *ShapeSpecificPool

	// opParams holds pre-computed operation parameters for nodes that need them.
	// Indexed by node.builderIdx. Nil for nodes that don't need specialized params.
	// Currently used for DotGeneral to store algorithm selection based on concrete shapes.
	opParams []any

	// canonical maps each node index to the "canonical" node that computes the same result.
	// If canonical[i] == i, this node is the canonical version (first occurrence).
	// If canonical[i] != i, this node's result should be reused from canonical[i].
	// This enables runtime deduplication of nodes that became identical after shape resolution.
	canonical []int
}

// DotGeneralSpecParams holds pre-computed parameters for DotGeneral operations.
// These are computed at specialization time with concrete shapes, allowing
// proper algorithm selection that couldn't happen at build time with dynamic shapes.
type DotGeneralSpecParams struct {
	// execPath is the selected execution strategy based on concrete sizes.
	execPath dotGeneralExecutionPath

	// Concrete sizes computed from resolved shapes.
	batchSize       int
	lhsCrossSize    int
	rhsCrossSize    int
	contractingSize int
}

// poolKey identifies a buffer pool by dtype and size.
type poolKey struct {
	dtype  dtypes.DType
	length int
}

// ShapeSpecificPool provides exact-sized buffer pools for a specialization.
// Unlike the global backend pools that use sync.Map lookups, this provides
// direct access to pools for the known shapes in a specialization.
type ShapeSpecificPool struct {
	pools map[poolKey]*sync.Pool
}

// newShapeSpecificPool creates buffer pools for all unique (dtype, size) combinations
// in the given node shapes.
func newShapeSpecificPool(nodeShapes []shapes.Shape) *ShapeSpecificPool {
	// Collect unique (dtype, size) combinations
	poolKeys := make(map[poolKey]bool)
	for _, shape := range nodeShapes {
		if shape.Ok() && shape.IsFullyConcrete() {
			key := poolKey{dtype: shape.DType, length: shape.Size()}
			poolKeys[key] = true
		}
	}

	// Create pools for each unique key
	pools := make(map[poolKey]*sync.Pool, len(poolKeys))
	for key := range poolKeys {
		dtype := key.dtype
		length := key.length
		pools[key] = &sync.Pool{
			New: func() any {
				return &Buffer{
					flat:  reflect.MakeSlice(reflect.SliceOf(dtype.GoType()), length, length).Interface(),
					shape: shapes.Make(dtype, length),
				}
			},
		}
	}

	return &ShapeSpecificPool{pools: pools}
}

// getBuffer retrieves a buffer from the pool for the given dtype and length.
// Returns nil if no pool exists for this combination.
func (p *ShapeSpecificPool) getBuffer(dtype dtypes.DType, length int) *Buffer {
	if p == nil {
		return nil
	}
	key := poolKey{dtype: dtype, length: length}
	pool := p.pools[key]
	if pool == nil {
		return nil
	}
	buf := pool.Get().(*Buffer)
	buf.valid = true
	return buf
}

// putBuffer returns a buffer to the pool.
// Does nothing if the pool doesn't exist for this buffer's size.
func (p *ShapeSpecificPool) putBuffer(buffer *Buffer) {
	if p == nil || buffer == nil || !buffer.shape.Ok() {
		return
	}
	key := poolKey{dtype: buffer.shape.DType, length: buffer.shape.Size()}
	pool := p.pools[key]
	if pool != nil {
		buffer.valid = false
		pool.Put(buffer)
	}
}

// hasPool returns true if a pool exists for the given dtype and length.
func (p *ShapeSpecificPool) hasPool(dtype dtypes.DType, length int) bool {
	if p == nil {
		return false
	}
	key := poolKey{dtype: dtype, length: length}
	_, ok := p.pools[key]
	return ok
}

// numPools returns the number of unique pools in this ShapeSpecificPool.
func (p *ShapeSpecificPool) numPools() int {
	if p == nil {
		return 0
	}
	return len(p.pools)
}

// Key returns the cache key for this specialization.
func (s *ShapeSpecialization) Key() string {
	return s.bindings.Key()
}

// newSpecialization creates a ShapeSpecialization for the given builder and axis bindings.
// It iterates through all nodes in the builder and resolves each node's shape
// using the provided bindings, then pre-creates buffer pools for all unique sizes.
func newSpecialization(builder *Builder, bindings shapes.AxisBindings) *ShapeSpecialization {
	spec := &ShapeSpecialization{
		bindings:   bindings.Clone(),
		nodeShapes: make([]shapes.Shape, len(builder.nodes)),
	}

	for i, node := range builder.nodes {
		spec.nodeShapes[i] = node.shape.Resolve(bindings)
	}

	// Create buffer pools for all unique (dtype, size) combinations
	spec.bufferPool = newShapeSpecificPool(spec.nodeShapes)

	// Pre-compute operation-specific parameters (e.g., DotGeneral algorithm selection)
	spec.computeOpParams(builder)

	// Compute runtime deduplication mapping
	spec.computeDeduplication(builder)

	return spec
}

// computeOpParams pre-computes operation parameters for nodes that need them.
// Currently handles DotGeneral operations to select the optimal algorithm
// based on concrete shapes rather than symbolic ones.
func (s *ShapeSpecialization) computeOpParams(builder *Builder) {
	s.opParams = make([]any, len(builder.nodes))
	for nodeIdx, node := range builder.nodes {
		switch node.opType {
		case backends.OpTypeDotGeneral:
			s.opParams[nodeIdx] = s.computeDotGeneralParams(builder, node)
		}
	}
}

// computeDotGeneralParams computes specialized parameters for a DotGeneral operation.
// It recalculates sizes from resolved shapes and selects the execution path accordingly.
func (s *ShapeSpecialization) computeDotGeneralParams(builder *Builder, node *Node) *DotGeneralSpecParams {
	if len(node.inputs) < 2 {
		return nil
	}

	params := node.data.(*dotGeneralNodeData)

	// Get resolved shapes for inputs
	lhsIdx := node.inputs[0].builderIdx
	rhsIdx := node.inputs[1].builderIdx
	lhsShape := s.nodeShapes[lhsIdx]
	rhsShape := s.nodeShapes[rhsIdx]

	// Recalculate sizes with concrete shapes
	batchSize, lhsCrossSize, contractingSize, _ := dgFindSizes(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes)
	_, rhsCrossSize, _, _ := dgFindSizes(rhsShape, params.rhsContractingAxes, params.rhsBatchAxes)

	// Select execution path with concrete sizes
	execPath := dgSelectExecPathWithSizes(builder.backend, lhsShape, rhsShape, params,
		batchSize, lhsCrossSize, rhsCrossSize, contractingSize)

	return &DotGeneralSpecParams{
		execPath:        execPath,
		batchSize:       batchSize,
		lhsCrossSize:    lhsCrossSize,
		rhsCrossSize:    rhsCrossSize,
		contractingSize: contractingSize,
	}
}

// NodeShape returns the resolved shape for the node at the given builder index.
func (s *ShapeSpecialization) NodeShape(builderIdx int) shapes.Shape {
	if builderIdx < 0 || builderIdx >= len(s.nodeShapes) {
		return shapes.Invalid()
	}
	return s.nodeShapes[builderIdx]
}

// computeDeduplication computes the canonical mapping for runtime deduplication.
// Nodes with identical signatures (opType, canonical inputs, data, concrete shape)
// are deduplicated at specialization time. The first occurrence becomes canonical,
// and subsequent identical nodes map to it.
func (s *ShapeSpecialization) computeDeduplication(builder *Builder) {
	numNodes := len(builder.nodes)
	s.canonical = make([]int, numNodes)

	// Map from signature to canonical node index
	signatures := make(map[string]int)

	// Process nodes in topological order (they're already in DAG order)
	for nodeIdx, node := range builder.nodes {
		// Parameters can't be deduplicated - they're inputs
		if node.opType == backends.OpTypeParameter {
			s.canonical[nodeIdx] = nodeIdx
			continue
		}

		// Multi-output select nodes: deduplicate based on parent's canonical + output index
		if node.isNodeSelectOutput {
			parentIdx := node.inputs[0].builderIdx
			canonicalParent := s.canonical[parentIdx]
			sig := fmt.Sprintf("select:%d:%d", canonicalParent, node.selectOutputIdx)
			if existingIdx, found := signatures[sig]; found {
				s.canonical[nodeIdx] = existingIdx
			} else {
				s.canonical[nodeIdx] = nodeIdx
				signatures[sig] = nodeIdx
			}
			continue
		}

		// Compute signature for regular nodes
		sig := s.computeNodeSignature(builder, node, nodeIdx)

		if existingIdx, found := signatures[sig]; found {
			// This node duplicates an existing one
			s.canonical[nodeIdx] = existingIdx
		} else {
			// This is the canonical version
			s.canonical[nodeIdx] = nodeIdx
			signatures[sig] = nodeIdx
		}
	}
}

// computeNodeSignature computes a unique signature for a node based on:
// - opType
// - Canonical indices of inputs (using canonical[] recursively)
// - Node data (operation parameters)
// - Concrete shape from nodeShapes[]
func (s *ShapeSpecialization) computeNodeSignature(builder *Builder, node *Node, nodeIdx int) string {
	// Start with opType
	sig := fmt.Sprintf("%s:", node.opType)

	// Add canonical input indices
	sig += "inputs["
	for i, input := range node.inputs {
		if i > 0 {
			sig += ","
		}
		sig += fmt.Sprintf("%d", s.canonical[input.builderIdx])
	}
	sig += "]"

	// Add concrete shape
	sig += fmt.Sprintf(":shape[%s]", s.nodeShapes[nodeIdx])

	// Add data signature
	sig += fmt.Sprintf(":data[%s]", nodeDataSignature(node.data))

	return sig
}

// nodeDataSignature creates a string representation of node data for deduplication.
// Uses the same comparison logic as compile-time deduplication where possible.
func nodeDataSignature(data any) string {
	if data == nil {
		return "nil"
	}

	// Use type name as prefix
	typ := reflect.TypeOf(data)
	prefix := typ.String()

	// Handle known types with meaningful representation
	switch d := data.(type) {
	case int:
		return fmt.Sprintf("%s:%d", prefix, d)
	case []int:
		return fmt.Sprintf("%s:%v", prefix, d)
	case *Buffer:
		// For constants, use shape and data hash
		return fmt.Sprintf("%s:%s:%v", prefix, d.shape, bufferHash(d))
	case *dotGeneralNodeData:
		return fmt.Sprintf("%s:lhsC%v:lhsB%v:rhsC%v:rhsB%v",
			prefix,
			d.lhsContractingAxes, d.lhsBatchAxes,
			d.rhsContractingAxes, d.rhsBatchAxes)
	case *convNode:
		return fmt.Sprintf("%s:axes%v:strides%v:pad%v:inDil%v:kDil%v:chGrp%d:batchGrp%d",
			prefix, d.axes, d.strides, d.paddings, d.inputDilations, d.kernelDilations,
			d.channelGroupCount, d.batchGroupCount)
	case nodeDataComparable:
		// For types implementing nodeDataComparable, use reflection to get key fields
		return fmt.Sprintf("%s:%v", prefix, reflect.ValueOf(data).Elem())
	default:
		// For other types, use %v which will show struct fields
		return fmt.Sprintf("%s:%v", prefix, data)
	}
}

// bufferHash creates a simple hash of buffer contents for deduplication.
// This is used for constants to detect identical values.
func bufferHash(b *Buffer) uint64 {
	if b == nil || b.flat == nil {
		return 0
	}
	// Simple FNV-like hash over the bytes
	h := uint64(14695981039346656037)
	bytes := b.mutableBytes()
	for _, byte := range bytes {
		h ^= uint64(byte)
		h *= 1099511628211
	}
	return h
}
