// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
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
}

// Key returns the cache key for this specialization.
func (s *ShapeSpecialization) Key() string {
	return s.bindings.Key()
}

// newSpecialization creates a ShapeSpecialization for the given builder and axis bindings.
// It iterates through all nodes in the builder and resolves each node's shape
// using the provided bindings.
func newSpecialization(builder *Builder, bindings shapes.AxisBindings) *ShapeSpecialization {
	spec := &ShapeSpecialization{
		bindings:   bindings.Clone(),
		nodeShapes: make([]shapes.Shape, len(builder.nodes)),
	}

	for i, node := range builder.nodes {
		spec.nodeShapes[i] = node.shape.Resolve(bindings)
	}

	return spec
}

// NodeShape returns the resolved shape for the node at the given builder index.
func (s *ShapeSpecialization) NodeShape(builderIdx int) shapes.Shape {
	if builderIdx < 0 || builderIdx >= len(s.nodeShapes) {
		return shapes.Invalid()
	}
	return s.nodeShapes[builderIdx]
}
