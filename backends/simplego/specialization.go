// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
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
