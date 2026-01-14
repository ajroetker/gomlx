// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build gliner_model

package gliner

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

func TestLoadModel(t *testing.T) {
	model, err := LoadModel("model")
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	t.Logf("Model loaded with %d tensors", len(model.weights.Tensors))

	// Check config.
	if model.Config.VocabSize != 128004 {
		t.Errorf("expected vocab size 128004, got %d", model.Config.VocabSize)
	}
	if model.Config.HiddenSize != 768 {
		t.Errorf("expected hidden size 768, got %d", model.Config.HiddenSize)
	}
}

func TestLoadWeights(t *testing.T) {
	model, err := LoadModel("model")
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	// Create a context and load weights.
	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("LoadWeightsIntoContext failed: %v", err)
	}

	// Check that some key variables exist.
	embVar := ctx.GetVariableByScopeAndName("/gliner/token_rep/embeddings", "word_embeddings")
	if embVar == nil {
		t.Error("word_embeddings variable not found")
	} else {
		t.Logf("word_embeddings shape: %s", embVar.Shape())
		if embVar.Shape().Dimensions[0] != 128004 || embVar.Shape().Dimensions[1] != 768 {
			t.Errorf("unexpected word_embeddings shape: %s", embVar.Shape())
		}
	}

	// Check LSTM weights.
	lstmVar := ctx.GetVariableByScopeAndName("/gliner/rnn/forward", "weight_ih")
	if lstmVar == nil {
		t.Error("LSTM weight_ih variable not found")
	} else {
		t.Logf("LSTM weight_ih shape: %s", lstmVar.Shape())
	}
}

func TestBuildGraph(t *testing.T) {
	model, err := LoadModel("model")
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	// Create context and load weights.
	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("LoadWeightsIntoContext failed: %v", err)
	}

	// Get backend.
	backend := graphtest.BuildTestBackend()

	// Build a simple graph to test.
	// We'll create minimal inputs to verify the graph builds.
	batchSize := 1
	seqLen := 8
	numTypes := 2
	typeSeqLen := 4

	g := graph.NewGraph(backend, "gliner_test")

	inputIDs := graph.Parameter(g, "input_ids", shapes.Make(dtypes.Int32, batchSize, seqLen))
	attentionMask := graph.Parameter(g, "attention_mask", shapes.Make(dtypes.Float32, batchSize, seqLen))
	entityTypeIDs := graph.Parameter(g, "entity_type_ids", shapes.Make(dtypes.Int32, batchSize, numTypes, typeSeqLen))
	entityTypeMask := graph.Parameter(g, "entity_type_mask", shapes.Make(dtypes.Float32, batchSize, numTypes, typeSeqLen))

	// Build the model graph.
	scores := model.BuildGraph(ctx, inputIDs, attentionMask, entityTypeIDs, entityTypeMask)

	// Check output shape.
	expectedShape := shapes.Make(dtypes.Float32, batchSize, seqLen, model.Config.MaxWidth, numTypes)
	if !scores.Shape().Equal(expectedShape) {
		t.Errorf("expected output shape %s, got %s", expectedShape, scores.Shape())
	}

	t.Logf("Graph built successfully with output shape: %s", scores.Shape())
}
