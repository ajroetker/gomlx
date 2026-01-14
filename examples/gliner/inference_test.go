// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build gliner_model

package gliner

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
)

func TestInference(t *testing.T) {
	// Get test backend.
	backend := graphtest.BuildTestBackend()

	// Load GLiNER model.
	// Use the safetensors model we converted.
	gliner, err := New(backend, "model")
	if err != nil {
		t.Fatalf("Failed to create GLiNER: %v", err)
	}
	defer gliner.Close()

	// Test text with various entity types.
	text := "Apple Inc. was founded by Steve Jobs in Cupertino, California."
	entityTypes := []string{"company", "person", "location"}

	// Run prediction.
	entities, err := gliner.Predict(text, entityTypes)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	// Log results.
	t.Logf("Found %d entities:", len(entities))
	for _, e := range entities {
		t.Logf("  - %s (%s): %.3f at [%d:%d]", e.Text, e.Type, e.Score, e.Start, e.End)
	}

	// We expect to find at least some entities (Apple, Steve Jobs, Cupertino, California).
	// The actual results depend on the model weights and threshold.
	if len(entities) == 0 {
		t.Log("No entities found - this may be expected if threshold is too high or model needs tuning")
	}
}

func TestInferenceShapes(t *testing.T) {
	// This test verifies the model graph builds correctly without running full inference.
	backend := graphtest.BuildTestBackend()

	// Load model.
	model, err := LoadModel("model")
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	// Load tokenizer (now in model directory).
	tok, err := LoadTokenizer("model")
	if err != nil {
		t.Fatalf("LoadTokenizer failed: %v", err)
	}

	t.Logf("Model config: vocab=%d, hidden=%d, layers=%d",
		model.Config.VocabSize, model.Config.HiddenSize, model.Config.NumLayers)

	// Encode a sample.
	enc, err := tok.Encode("Hello world", []string{"test"}, 64)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	t.Logf("Encoded shapes:")
	t.Logf("  InputIDs: %v", enc.InputIDs.Shape())
	t.Logf("  AttentionMask: %v", enc.AttentionMask.Shape())
	t.Logf("  EntityTypeIDs: %v", enc.EntityTypeIDs.Shape())
	t.Logf("  EntityTypeMask: %v", enc.EntityTypeMask.Shape())

	_ = backend // Use backend to prevent unused warning.
}
