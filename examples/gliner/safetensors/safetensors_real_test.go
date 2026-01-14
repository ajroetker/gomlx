// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build gliner_model

package safetensors

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// TestRealModel tests parsing the actual GLiNER model.
// Run with: go test -tags=gliner_model -v ./examples/gliner/safetensors/
func TestRealModel(t *testing.T) {
	// Find model path relative to this test file.
	modelPath := filepath.Join("..", "model", "model.safetensors")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s - run convert_to_safetensors.py first", modelPath)
	}

	f, err := Open(modelPath)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Print summary.
	t.Logf("Loaded model with %d tensors", len(f.Tensors))

	// Check expected tensors exist.
	expectedTensors := []string{
		"token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight",
		"token_rep_layer.bert_layer.model.encoder.layer.0.attention.self.query_proj.weight",
		"span_rep_layer.span_rep_layer.out_project.0.weight",
		"prompt_rep_layer.0.weight",
		"rnn.lstm.weight_ih_l0",
	}

	for _, name := range expectedTensors {
		info, ok := f.Get(name)
		if !ok {
			t.Errorf("expected tensor %q not found", name)
			continue
		}
		t.Logf("  %s: %s", name, info.Shape)
	}

	// Test loading a tensor.
	embeddingName := "token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight"
	tensor, err := f.ToTensor(embeddingName)
	if err != nil {
		t.Fatalf("ToTensor failed: %v", err)
	}

	// Check shape: [128004, 768]
	if tensor.Shape().Rank() != 2 {
		t.Errorf("expected rank 2, got %d", tensor.Shape().Rank())
	}
	if tensor.Shape().Dimensions[0] != 128004 {
		t.Errorf("expected vocab size 128004, got %d", tensor.Shape().Dimensions[0])
	}
	if tensor.Shape().Dimensions[1] != 768 {
		t.Errorf("expected hidden size 768, got %d", tensor.Shape().Dimensions[1])
	}
	if tensor.Shape().DType != dtypes.Float32 {
		t.Errorf("expected Float32, got %v", tensor.Shape().DType)
	}

	t.Logf("Successfully loaded embedding tensor with shape %s", tensor.Shape())
}
