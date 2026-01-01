package graph

import (
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// backend_dynamic_bounds.go contains backend wrappers for dynamic operations with bounds.
// These are separate from gen_backend_ops.go to avoid being overwritten by code generation.

// backendDynamicReshapeWithBoundsAndShape uses static reshape with the given output dimensions.
// Dynamic reshape is not supported - this function will panic if the operand has symbolic
// dimensions or if the output dimensions cannot be determined at compile time.
func backendDynamicReshapeWithBoundsAndShape(operand *Node, outputShapeTensor *Node, bounds []int, outputDims []int) (node *Node) {
	// Check if operand has symbolic dimensions
	if operand.Shape().HasSymbolicDim() {
		panic("DynamicReshape is not supported: operand has symbolic dimensions. Use static Reshape with concrete dimensions instead.")
	}

	// Check if all output dimensions are concrete (non-negative)
	for _, d := range outputDims {
		if d < 0 {
			panic("DynamicReshape is not supported: output dimensions must be concrete (non-negative)")
		}
	}

	// Check if bounds differ from outputDims (bounded dynamic)
	if len(bounds) == len(outputDims) {
		for i := range bounds {
			if outputDims[i] > 0 && bounds[i] > 0 && outputDims[i] != bounds[i] {
				panic("DynamicReshape is not supported: bounded dynamic dimensions (bounds != outputDims) are not supported")
			}
		}
	}

	// Verify sizes match
	outputSize := 1
	for _, d := range outputDims {
		outputSize *= d
	}
	operandSize := operand.Shape().Size()
	if operandSize < 0 {
		operandSize = -operandSize
	}
	if operandSize > 0 && operandSize != outputSize {
		panic(fmt.Sprintf("DynamicReshape: size mismatch - operand size %d != output size %d", operandSize, outputSize))
	}

	// Use static Reshape
	g := validateBuildingGraphFromInputs(operand)
	ni := &nodeInputsReshape{
		x:          operand,
		dimensions: slices.Clone(outputDims),
	}

	result, err := g.builder.Reshape(operand.outputOps[0], outputDims...)
	if err != nil {
		panic(err)
	}
	outputShape := shapes.Make(operand.DType(), outputDims...)
	node = &Node{
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{outputShape},
		graph:        g,
		inputs:       ni,
		inputNodes:   []*Node{operand},
	}
	g.registerNode(node)
	return
}

// nodeInputsDynamicBroadcastInDimWithBounds holds the inputs for DynamicBroadcastInDim with bounds.
type nodeInputsDynamicBroadcastInDimWithBounds struct {
	operand             *Node
	outputDimensions    *Node
	broadcastDimensions []int
	bounds              []int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsDynamicBroadcastInDimWithBounds) Type() NodeType {
	return NodeTypeDynamicBroadcastInDim
}

// InputNodes implements the interface NodeInputs.
func (ni *nodeInputsDynamicBroadcastInDimWithBounds) InputNodes() []*Node {
	return []*Node{ni.operand, ni.outputDimensions}
}

// String implements the interface NodeInputs.
func (ni *nodeInputsDynamicBroadcastInDimWithBounds) String() string {
	return fmt.Sprintf("%s(operand=[#%d], outputDimensions=[#%d], broadcastDimensions=%v, bounds=%v)",
		ni.Type(),
		ni.operand.Id(),
		ni.outputDimensions.Id(),
		ni.broadcastDimensions,
		ni.bounds,
	)
}

// backendDynamicBroadcastInDimWithBoundsAndShape broadcasts operand using DynamicBroadcastInDim
// and explicitly sets the output shape for GoMLX shape propagation.
func backendDynamicBroadcastInDimWithBoundsAndShape(operand *Node, outputDimensions *Node, broadcastDimensions []int, bounds []int, outputDims []int) (node *Node) {
	inputNodes := []*Node{operand, outputDimensions}
	g := validateBuildingGraphFromInputs(inputNodes...)
	inputs := &nodeInputsDynamicBroadcastInDimWithBounds{
		operand:             operand,
		outputDimensions:    outputDimensions,
		broadcastDimensions: broadcastDimensions,
		bounds:              bounds,
	}
	result, err := g.builder.DynamicBroadcastInDim(operand.outputOps[0], outputDimensions.outputOps[0], broadcastDimensions)
	if err != nil {
		panic(err)
	}
	// Get the shape from the XLA backend result, which tracks logical dimensions
	// when they differ from physical (bounds) dimensions.
	outputShape, err := g.builder.OpShape(result)
	if err != nil {
		// Fallback to computing shape from outputDims if OpShape fails
		hasDynamic := false
		for _, d := range outputDims {
			if d < 0 {
				hasDynamic = true
				break
			}
		}
		if hasDynamic {
			outputShape = shapes.MakeDynamic(operand.DType(), outputDims...)
		} else {
			outputShape = shapes.Make(operand.DType(), outputDims...)
		}
	}

	// Check if any dimension uses bounded dynamic (logical != physical)
	// If so, mark output as symbolic so downstream ops use dynamic variants
	needsSymbolic := false
	for i, d := range outputDims {
		if d > 0 && i < len(bounds) && d != bounds[i] {
			needsSymbolic = true
			break
		}
	}

	if needsSymbolic {
		symbolicDims := make([]int, len(outputDims))
		for i, d := range outputDims {
			if d > 0 && i < len(bounds) && d != bounds[i] {
				symbolicDims[i] = -1
			} else {
				symbolicDims[i] = d
			}
		}
		outputShape = shapes.MakeDynamic(operand.DType(), symbolicDims...)
	}

	node = &Node{
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{outputShape},
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return
}
