// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"sync"

	"github.com/pkg/errors"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

var _ backends.Executable = (*Executable)(nil)

// Executable holds a frozen Builder. It assumes the graph in Builder is valid and has been properly
// checked that all the shapes and data types are valid.
//
// If any inconsistencies are found, please fix in the Builder, so Executable can be written without the need
// of any duplicate checks.
type Executable struct {
	backend *Backend

	// builder must have Builder.compiled set to true, so it is no longer active.
	builder *Builder

	// mainFn is the compiled main function.
	mainFn *FunctionExecutable

	// Dynamic shape support: when inputs have named axes (e.g., "batch"),
	// we create specializations for each unique binding at execution time.

	// hasDynamicAxes is true if any input parameter has named axes.
	hasDynamicAxes bool

	// inputPatterns stores the pattern shapes (with named axes) for each input.
	// Used to extract bindings from concrete input shapes.
	inputPatterns []shapes.Shape

	// specializations caches ShapeSpecialization by bindings key.
	// Key: bindings.Key() (e.g., "batch=32,seq=128")
	// Value: *ShapeSpecialization
	specializations sync.Map
}

// Compile time check.
var _ backends.Executable = (*Executable)(nil)

// Finalize immediately frees resources associated with the executable.
//
// TODO: Race-condition where calling Finalize will make execution crash, if finalized while executing.
//
//	Make Finalize wait for all the current executions to exit, before finalizing.
//	And add a latch indicating Finalize has been called, to tell the executions to exit immediately
//	without finishing. Finally, remove the `e.builder == nil` checks, that won't be necessary anymore,
//	since e.builder will never be set to nil while there is an execution alive.
func (e *Executable) Finalize() {
	e.builder.Finalize()
	e.builder = nil
}

// Inputs returns the list of parameters names and shapes, in order created by the Builder.Parameter calls.
func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	numInputs := len(e.builder.inputs)
	if numInputs == 0 {
		return
	}
	names = make([]string, numInputs)
	inputShapes = make([]shapes.Shape, numInputs)
	for ii, node := range e.builder.inputs {
		parameter := e.builder.inputs[ii].data.(*nodeParameter)
		names[ii] = parameter.name
		inputShapes[ii] = node.shape
	}
	return
}

// Outputs returns the output shapes of the computation, in order given to the Builder.Compile call.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	numOutputs := len(e.builder.outputs)
	if numOutputs == 0 {
		return
	}
	outputShapes = make([]shapes.Shape, numOutputs)
	for ii, node := range e.builder.outputs {
		outputShapes[ii] = node.shape
	}
	return outputShapes
}

// newExecutable creates an Executable ready to run the graph built with builder.
// The main function must have been compiled (via Return() and then any
// duplicate output handling in Builder.Compile()).
func newExecutable(builder *Builder, mainFn *FunctionExecutable) *Executable {
	e := &Executable{
		backend: builder.backend,
		builder: builder,
		mainFn:  mainFn,
	}

	// Check if any input has named axes (dynamic shapes).
	for _, inputNode := range builder.inputs {
		if inputNode.shape.HasNamedAxes() {
			e.hasDynamicAxes = true
			break
		}
	}

	// Store input patterns for binding extraction if we have dynamic axes.
	if e.hasDynamicAxes {
		e.inputPatterns = make([]shapes.Shape, len(builder.inputs))
		for i, inputNode := range builder.inputs {
			e.inputPatterns[i] = inputNode.shape.Clone()
		}
	}

	return e
}

// nodeExecutor for the given operation type.
//
// It is given the buffers for its inputs, and a reserved buffer where to store its output, already
// with the shape pre-calculated.
type nodeExecutor func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error)

// nodeMultiOutputExecutor is a version of a node executor when it returns multiple outputs.
type nodeMultiOutputExecutor func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error)

var (
	// nodeExecutors should be populated during initialization (`init` functions) for the ops implemented.
	// For the nodes not implemented, leave it as nil, and it will return an error.
	//
	// nodeExecutors should be populated with a priority (see setNodeExecutor), which can conctorl whether
	// to overwrite a nodeExecutors configuration independent of the order of settting.
	nodeExecutors         [backends.OpTypeLast]nodeExecutor
	nodeExecutorsPriority [backends.OpTypeLast]registerPriority

	// multiOutputsNodeExecutors should be populated during initialization for the multi-output ops
	// implemented. E.g.: RNGBitGenerator.
	multiOutputsNodeExecutors [backends.OpTypeLast]nodeMultiOutputExecutor
)

// registerPriority defines the priority of a node executor. Highest priority takes precedence.
// Anything with priority < 0 is ignored.
type registerPriority int

const (
	priorityGeneric registerPriority = 0
	priorityTyped   registerPriority = 1   // Specialized typed implementation.
	priorityArch    registerPriority = 10  // Specialized architecture implementation.
	priorityUser    registerPriority = 100 // Custom user overrides.
)

// setNodeExecutor sets the node executor for the given operation type with the specified priority.
// If the priority is lower than the current priority for the operation type, the executor is ignored.
func setNodeExecutor(opType backends.OpType, priority registerPriority, executor nodeExecutor) {
	if priority < nodeExecutorsPriority[opType] {
		// We have soemthing registered with higher priority, ignore.
		return
	}
	nodeExecutorsPriority[opType] = priority
	nodeExecutors[opType] = executor
}

type opsExecutionType int

const (
	opsExecutionDynamic opsExecutionType = iota
	opsExecutionParallel
	opsExecutionSequential
)

// Execute the executable on the default device (0).
// The number and shapes of the inputs must match those returned by Inputs.
//
// For graphs with dynamic shapes (named axes like "batch"), the input shapes
// must match the pattern (dtype, rank, static dimensions) and bindings are
// extracted at execution time.
//
// The inputs marked in `donate` will become invalid after use.
// This is useful if the input buffer is no longer needed or if updating a variable
// so its Buffer space can be reused as an output Buffer.
//
// Donated buffers are no longer valid after the call.
// If donate is nil, it is assumed to be false for all buffers, and no buffer is donated.
func (e *Executable) Execute(inputs []backends.Buffer, donate []bool, _ backends.DeviceNum) ([]backends.Buffer, error) {
	// Keep the live executions count.
	e.backend.numLiveExecutions.Add(1)
	defer e.backend.numLiveExecutions.Add(-1)

	// Check inputs length
	if len(inputs) != len(e.builder.inputs) {
		return nil, errors.Errorf("Execute: expected %d inputs, got %d", len(e.builder.inputs), len(inputs))
	}

	// donate defaults to false for all buffers.
	if len(donate) == 0 {
		donate = make([]bool, len(inputs))
	}

	// Check input shapes and convert to *Buffer
	bufInputs := make([]*Buffer, len(inputs))
	var bindings shapes.AxisBindings

	for ii, input := range inputs {
		if input == nil {
			return nil, errors.Errorf("Execute: input buffer #%d is nil!?", ii)
		}
		inputBuffer, ok := input.(*Buffer)
		if !ok {
			return nil, errors.Errorf("Execute: input buffer #%d is not from SimpleGo backend", ii)
		}
		if !inputBuffer.valid {
			return nil, errors.Errorf(
				"Execute: input buffer (%p) #%d is not valid, likely it is being used after being isFinalized",
				inputBuffer, ii)
		}
		if inputBuffer.flat == nil {
			return nil, errors.Errorf("Execute: input buffer #%d flat data is set to nil (!?)", ii)
		}

		nodeInput := e.builder.inputs[ii]
		paramName := nodeInput.data.(*nodeParameter).name

		if e.hasDynamicAxes {
			// For dynamic shapes: extract bindings from concrete input shape
			inputBindings, err := shapes.ExtractBindings(e.inputPatterns[ii], inputBuffer.shape)
			if err != nil {
				return nil, errors.Errorf("Execute: parameter %q (input #%d) for %q: shape mismatch: %v",
					paramName, ii, e.builder.name, err)
			}
			if bindings == nil {
				bindings = inputBindings
			} else if err := bindings.Merge(inputBindings); err != nil {
				return nil, errors.Errorf("Execute: conflicting axis bindings for parameter %q (input #%d): %v",
					paramName, ii, err)
			}
		} else {
			// For static shapes: require exact match
			if !inputBuffer.shape.Equal(nodeInput.shape) {
				return nil, errors.Errorf("Execute: parameter %q (input #%d) for %q: expected shape %s, got %s",
					paramName, ii, e.builder.name, nodeInput.shape, inputBuffer.shape)
			}
		}
		bufInputs[ii] = inputBuffer
	}

	// Get or create specialization if we have dynamic axes
	var spec *ShapeSpecialization
	if e.hasDynamicAxes {
		spec = e.getOrCreateSpecialization(bindings)
	}

	// Delegate to FunctionExecutable with specialization
	outputs, err := e.mainFn.Execute(e.backend, bufInputs, donate, spec)
	if err != nil {
		return nil, err
	}

	// Convert outputs to backends.Buffer
	result := make([]backends.Buffer, len(outputs))
	for i, out := range outputs {
		result[i] = out
	}
	return result, nil
}

// getOrCreateSpecialization returns a cached specialization for the given bindings,
// creating one if it doesn't exist.
func (e *Executable) getOrCreateSpecialization(bindings shapes.AxisBindings) *ShapeSpecialization {
	key := bindings.Key()

	// Try to load existing specialization
	if cached, ok := e.specializations.Load(key); ok {
		return cached.(*ShapeSpecialization)
	}

	// Create new specialization
	spec := newSpecialization(e.builder, bindings)

	// Store and return (LoadOrStore handles race condition)
	actual, _ := e.specializations.LoadOrStore(key, spec)
	return actual.(*ShapeSpecialization)
}
