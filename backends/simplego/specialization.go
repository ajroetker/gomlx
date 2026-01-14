// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"reflect"
	"sync"

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

	return spec
}

// NodeShape returns the resolved shape for the node at the given builder index.
func (s *ShapeSpecialization) NodeShape(builderIdx int) shapes.Shape {
	if builderIdx < 0 || builderIdx >= len(s.nodeShapes) {
		return shapes.Invalid()
	}
	return s.nodeShapes[builderIdx]
}
