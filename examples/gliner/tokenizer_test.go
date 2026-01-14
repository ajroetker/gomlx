// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build gliner_model

package gliner

import (
	"testing"
)

func TestLoadTokenizer(t *testing.T) {
	// Load tokenizer from model directory.
	tok, err := LoadTokenizer("model")
	if err != nil {
		t.Fatalf("LoadTokenizer failed: %v", err)
	}

	t.Log("Tokenizer loaded successfully")

	// Test encoding.
	text := "Apple Inc. is a technology company based in Cupertino, California."
	entityTypes := []string{"company", "location", "person"}

	enc, err := tok.Encode(text, entityTypes, 128)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	// Check shapes.
	if enc.InputIDs.Shape().Dimensions[0] != 1 {
		t.Errorf("expected batch size 1, got %d", enc.InputIDs.Shape().Dimensions[0])
	}
	if enc.InputIDs.Shape().Dimensions[1] != 128 {
		t.Errorf("expected seq len 128, got %d", enc.InputIDs.Shape().Dimensions[1])
	}

	t.Logf("InputIDs shape: %v", enc.InputIDs.Shape())
	t.Logf("AttentionMask shape: %v", enc.AttentionMask.Shape())
	t.Logf("EntityTypeIDs shape: %v", enc.EntityTypeIDs.Shape())
	t.Logf("EntityTypeMask shape: %v", enc.EntityTypeMask.Shape())
	t.Logf("TextOffset: %d", enc.TextOffset)
	t.Logf("EntityTypes: %v", enc.EntityTypes)

	// Verify entity types tensor has correct number of types.
	if enc.EntityTypeIDs.Shape().Dimensions[1] != len(entityTypes) {
		t.Errorf("expected %d entity types, got %d", len(entityTypes), enc.EntityTypeIDs.Shape().Dimensions[1])
	}
}

func TestTokenizerSpecialTokens(t *testing.T) {
	// Verify our special token constants.
	if TokenCLS != 1 {
		t.Errorf("TokenCLS should be 1, got %d", TokenCLS)
	}
	if TokenSEP != 2 {
		t.Errorf("TokenSEP should be 2, got %d", TokenSEP)
	}
	if TokenENT != 128002 {
		t.Errorf("TokenENT should be 128002, got %d", TokenENT)
	}
	if TokenENTS != 128003 {
		t.Errorf("TokenENTS should be 128003, got %d", TokenENTS)
	}
}
