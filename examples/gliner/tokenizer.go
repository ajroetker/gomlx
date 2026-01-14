// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gliner

import (
	"path/filepath"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// Special token IDs for GLiNER.
const (
	TokenPAD  = 0      // [PAD] token
	TokenCLS  = 1      // [CLS] token
	TokenSEP  = 2      // [SEP] token
	TokenUNK  = 3      // [UNK] token
	TokenMASK = 128000 // [MASK] token
	TokenENT  = 128002 // <<ENT>> entity type marker
	TokenENTS = 128003 // <<SEP>> separator between entity types and text
)

// Tokenizer wraps the HuggingFace tokenizer for GLiNER.
type Tokenizer struct {
	tok *tokenizer.Tokenizer
}

// LoadTokenizer loads the tokenizer from the model directory.
func LoadTokenizer(modelDir string) (*Tokenizer, error) {
	tokenizerPath := filepath.Join(modelDir, "tokenizer.json")
	tok, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load tokenizer from %s", tokenizerPath)
	}

	return &Tokenizer{tok: tok}, nil
}

// EncodedInput represents the encoded inputs for GLiNER.
type EncodedInput struct {
	// InputIDs are the token IDs [batch, seq_len].
	InputIDs *tensors.Tensor

	// AttentionMask is the attention mask [batch, seq_len].
	AttentionMask *tensors.Tensor

	// EntityTypeIDs are the token IDs for entity types [batch, num_types, type_seq_len].
	EntityTypeIDs *tensors.Tensor

	// EntityTypeMask is the attention mask for entity types [batch, num_types, type_seq_len].
	EntityTypeMask *tensors.Tensor

	// TextOffset is the position where the text starts in InputIDs (after entity markers).
	TextOffset int

	// EntityTypes stores the original entity type names in order.
	EntityTypes []string
}

// Encode tokenizes text with entity types for GLiNER.
//
// The input format is:
//
//	[CLS] <<ENT>> entity1 <<ENT>> entity2 ... <<SEP>> text tokens [SEP]
//
// This follows the GLiNER input convention where entity types are prepended to the text.
func (t *Tokenizer) Encode(text string, entityTypes []string, maxSeqLen int) (*EncodedInput, error) {
	// Tokenize the main text (without special tokens, we'll add them manually).
	textEnc, err := t.tok.EncodeSingle(text, false)
	if err != nil {
		return nil, errors.Wrap(err, "failed to encode text")
	}

	// Tokenize each entity type.
	var entityTypeEncodings [][]int
	maxTypeLen := 0
	for _, et := range entityTypes {
		etEnc, err := t.tok.EncodeSingle(et, false)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to encode entity type %q", et)
		}
		entityTypeEncodings = append(entityTypeEncodings, etEnc.GetIds())
		if len(etEnc.GetIds()) > maxTypeLen {
			maxTypeLen = len(etEnc.GetIds())
		}
	}

	// Build the input sequence:
	// [CLS] <<ENT>> entity1_tokens <<ENT>> entity2_tokens ... <<SEP>> text_tokens [SEP]
	inputIDs := make([]int32, 0, maxSeqLen)
	inputIDs = append(inputIDs, TokenCLS)

	// Add entity markers and tokens.
	for _, etEnc := range entityTypeEncodings {
		inputIDs = append(inputIDs, TokenENT)
		for _, tid := range etEnc {
			inputIDs = append(inputIDs, int32(tid))
		}
	}

	// Add separator and text.
	inputIDs = append(inputIDs, TokenENTS)
	textOffset := len(inputIDs)
	for _, tid := range textEnc.GetIds() {
		inputIDs = append(inputIDs, int32(tid))
	}
	inputIDs = append(inputIDs, TokenSEP)

	// Truncate if necessary.
	if len(inputIDs) > maxSeqLen {
		inputIDs = inputIDs[:maxSeqLen]
		// Ensure we end with SEP.
		inputIDs[maxSeqLen-1] = TokenSEP
	}

	// Pad to maxSeqLen.
	attentionMask := make([]float32, maxSeqLen)
	for i := 0; i < len(inputIDs); i++ {
		attentionMask[i] = 1.0
	}
	for len(inputIDs) < maxSeqLen {
		inputIDs = append(inputIDs, TokenPAD)
	}

	// Build entity type tensors.
	// Shape: [1, num_types, max_type_len]
	numTypes := len(entityTypes)
	entityTypeIDs := make([]int32, numTypes*maxTypeLen)
	entityTypeMask := make([]float32, numTypes*maxTypeLen)

	for i, etEnc := range entityTypeEncodings {
		for j, tid := range etEnc {
			entityTypeIDs[i*maxTypeLen+j] = int32(tid)
			entityTypeMask[i*maxTypeLen+j] = 1.0
		}
		// Rest is already zero (padding).
	}

	// Create tensors.
	return &EncodedInput{
		InputIDs:       tensors.FromFlatDataAndDimensions(inputIDs, 1, maxSeqLen),
		AttentionMask:  tensors.FromFlatDataAndDimensions(attentionMask, 1, maxSeqLen),
		EntityTypeIDs:  tensors.FromFlatDataAndDimensions(entityTypeIDs, 1, numTypes, maxTypeLen),
		EntityTypeMask: tensors.FromFlatDataAndDimensions(entityTypeMask, 1, numTypes, maxTypeLen),
		TextOffset:     textOffset,
		EntityTypes:    entityTypes,
	}, nil
}
