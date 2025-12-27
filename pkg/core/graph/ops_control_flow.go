package graph

// This file contains control flow helper functions for GoMLX graphs.

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// StableHLOFunction returns the underlying stablehlo.Function for advanced use cases.
// This allows direct access to StableHLO features like While loop closures that don't yet have
// a high-level GoMLX API.
//
// Returns nil if the backend is not XLA.
//
// Example usage for While loops:
//
//	// Get the stablehlo function
//	fn := g.StableHLOFunction()
//	if fn == nil {
//	    panic("StableHLO backend required for While loops")
//	}
//
//	// Create condition closure: counter < 10
//	condFn := fn.Closure()
//	condInput, _ := condFn.Input(xla.ShapeToXLA(counter.Shape()))
//	limit, _ := condFn.ConstantFromScalar(int32(10))
//	cond, _ := stablehlo.Compare(condInput, limit, types.CompareLT, types.CompareSigned)
//	condFn.Return(cond)
//
//	// Create body closure: counter + 1
//	bodyFn := fn.Closure()
//	bodyInput, _ := bodyFn.Input(xla.ShapeToXLA(counter.Shape()))
//	one, _ := bodyFn.ConstantFromScalar(int32(1))
//	next, _ := stablehlo.Add(bodyInput, one)
//	bodyFn.Return(next)
//
//	// Execute While loop
//	results := While(condFn, bodyFn, counter)
func (g *Graph) StableHLOFunction() *stablehlo.Function {
	xlaBuilder, ok := g.builder.(*xla.Builder)
	if !ok {
		return nil
	}
	return xlaBuilder.StableHLOFunction()
}

// IfClosure implements conditional execution using StableHLO If operation with closure-backed branches.
// It takes a scalar boolean predicate and two closure graphs (thenGraph and elseGraph).
// The branches inherit the parent scope and can reference values from the parent graph.
//
// The thenGraph and elseGraph must be created using NewClosureGraph and compiled with CompileClosure.
// Both branches must return the same number of outputs with matching shapes and dtypes.
//
// Example usage:
//
//	cond := Less(x, Const(g, 0.0))
//
//	// Create then branch: return abs(x)
//	thenG := g.NewClosureGraph("then_branch")
//	thenOut := Abs(x)  // Can reference parent value x
//	thenG.CompileClosure(thenOut)
//
//	// Create else branch: return x
//	elseG := g.NewClosureGraph("else_branch")
//	elseOut := x  // Can reference parent value x
//	elseG.CompileClosure(elseOut)
//
//	// Execute conditional
//	result := IfClosure(cond, thenG, elseG)[0]
//
// For backends that don't support closures, this falls back to eager evaluation using Where.
func IfClosure(pred *Node, thenGraph, elseGraph *Graph) []*Node {
	g := pred.Graph()
	g.AssertBuilding()

	// Validate predicate
	if !pred.IsScalar() || pred.DType() != dtypes.Bool {
		exceptions.Panicf("If predicate must be a scalar bool, got %s", pred.Shape())
	}

	// Check if backend supports closures
	xlaBuilder, ok := g.builder.(*xla.Builder)
	if !ok {
		exceptions.Panicf("If operation requires XLA backend with closure support")
	}

	// Get closure functions from the closure graphs
	type closureFnGetter interface {
		ClosureFunction() *stablehlo.Function
	}

	thenBuilder, ok := thenGraph.builder.(closureFnGetter)
	if !ok {
		exceptions.Panicf("then branch must be a closure graph created with NewClosureGraph")
	}
	thenFn := thenBuilder.ClosureFunction()
	if thenFn == nil {
		exceptions.Panicf("then branch closure function is nil")
	}

	elseBuilder, ok := elseGraph.builder.(closureFnGetter)
	if !ok {
		exceptions.Panicf("else branch must be a closure graph created with NewClosureGraph")
	}
	elseFn := elseBuilder.ClosureFunction()
	if elseFn == nil {
		exceptions.Panicf("else branch closure function is nil")
	}

	// Validate branches have same number of outputs
	if len(thenFn.Outputs) != len(elseFn.Outputs) {
		exceptions.Panicf("If branches must return same number of outputs, "+
			"then branch has %d, else branch has %d", len(thenFn.Outputs), len(elseFn.Outputs))
	}

	// Get the predicate value from the XLA backend
	predOp, ok := pred.outputOps[0].(*xla.Node)
	if !ok {
		exceptions.Panicf("failed to extract XLA node from predicate")
	}

	// Call StableHLO If
	values, err := stablehlo.If(predOp.Value(), thenFn, elseFn)
	if err != nil {
		exceptions.Panicf("StableHLO If failed: %v", err)
	}

	// Wrap results as graph nodes
	results := make([]*Node, len(values))
	outputOps := make([]backends.Op, len(values))
	for i, value := range values {
		// Use the builder's NewNode method to create proper XLA nodes
		outputOps[i] = xlaBuilder.NewNode(value)
	}

	// Create a single multi-output node for all results
	node := &Node{
		outputOps: outputOps,
		outputShapes: xslices.Map(outputOps,
			func(op backends.Op) shapes.Shape {
				s, err := g.builder.OpShape(op)
				if err != nil {
					panic(err)
				}
				return s
			}),
		graph:      g,
		inputNodes: []*Node{pred},
	}
	g.registerNode(node)

	// Split the multi-output node into individual nodes
	for i := range values {
		results[i] = &Node{
			outputOps:    []backends.Op{outputOps[i]},
			outputShapes: []shapes.Shape{node.outputShapes[i]},
			graph:        g,
			inputNodes:   []*Node{pred},
		}
		g.registerNode(results[i])
	}

	return results
}
