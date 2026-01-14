// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gliner

import (
	"sort"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// Entity represents a detected named entity.
type Entity struct {
	// Text is the entity text span.
	Text string

	// Type is the entity type (e.g., "person", "location").
	Type string

	// Score is the confidence score.
	Score float32

	// Start is the character offset where the entity starts.
	Start int

	// End is the character offset where the entity ends.
	End int
}

// GLiNER is the main inference interface for named entity recognition.
type GLiNER struct {
	model     *Model
	tokenizer *Tokenizer
	ctx       *context.Context
	backend   backends.Backend
	exec      *context.Exec
}

// New creates a new GLiNER instance from a model directory.
//
// The modelDir should contain:
//   - model.safetensors or model/ directory with weights
//   - tokenizer.json
//   - config.json (optional, uses defaults if not present)
func New(backend backends.Backend, modelDir string) (*GLiNER, error) {
	// Load model.
	model, err := LoadModel(modelDir)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load model")
	}

	// Load tokenizer.
	tokenizer, err := LoadTokenizer(modelDir)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load tokenizer")
	}

	// Create context and load weights.
	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		return nil, errors.Wrap(err, "failed to load weights")
	}

	return &GLiNER{
		model:     model,
		tokenizer: tokenizer,
		ctx:       ctx,
		backend:   backend,
	}, nil
}

// Predict runs named entity recognition on the input text.
//
// entityTypes specifies the types of entities to detect (e.g., ["person", "location", "organization"]).
// Returns a list of detected entities sorted by position.
func (g *GLiNER) Predict(text string, entityTypes []string) ([]Entity, error) {
	// Encode input.
	maxSeqLen := 512
	enc, err := g.tokenizer.Encode(text, entityTypes, maxSeqLen)
	if err != nil {
		return nil, errors.Wrap(err, "failed to encode input")
	}

	// Create or reuse exec.
	if g.exec == nil {
		g.exec = context.MustNewExec(g.backend, g.ctx,
			func(ctx *context.Context, inputIDs, attentionMask, entityTypeIDs, entityTypeMask *Node) *Node {
				return g.model.BuildGraph(ctx, inputIDs, attentionMask, entityTypeIDs, entityTypeMask)
			})
	}

	// Run inference.
	results := g.exec.MustExec(enc.InputIDs, enc.AttentionMask, enc.EntityTypeIDs, enc.EntityTypeMask)
	if len(results) == 0 {
		return nil, errors.New("model returned no results")
	}

	// Parse results.
	// Output shape: [batch, seq_len, max_width, num_types]
	scores := results[0]
	entities := g.decodeEntities(text, scores, enc, g.model.Config.Threshold)

	// Sort by position.
	sort.Slice(entities, func(i, j int) bool {
		return entities[i].Start < entities[j].Start
	})

	return entities, nil
}

// decodeEntities converts model output scores to entity spans.
func (g *GLiNER) decodeEntities(text string, scores *tensors.Tensor, enc *EncodedInput, threshold float64) []Entity {
	// Get score data.
	// Shape: [1, seq_len, max_width, num_types]
	shape := scores.Shape()
	seqLen := shape.Dimensions[1]
	maxWidth := shape.Dimensions[2]
	numTypes := shape.Dimensions[3]

	var entities []Entity

	// Access flattened scores via callback.
	tensors.MustConstFlatData(scores, func(scoreData []float32) {
		// Scan all positions and widths.
		for pos := enc.TextOffset; pos < seqLen; pos++ {
			for width := 0; width < maxWidth; width++ {
				endPos := pos + width
				if endPos >= seqLen {
					continue
				}

				for typeIdx := 0; typeIdx < numTypes; typeIdx++ {
					// Index into flattened array: [batch=0, pos, width, typeIdx]
					idx := pos*maxWidth*numTypes + width*numTypes + typeIdx
					score := scoreData[idx]

					if float64(score) > threshold {
						// Get character positions from tokens.
						// For now, use approximate positions based on token positions.
						// A full implementation would use the tokenizer's offset mapping.
						entities = append(entities, Entity{
							Text:  text, // TODO: extract actual span text
							Type:  enc.EntityTypes[typeIdx],
							Score: score,
							Start: pos - enc.TextOffset,
							End:   endPos - enc.TextOffset,
						})
					}
				}
			}
		}
	})

	return entities
}

// Close releases resources.
func (g *GLiNER) Close() {
	// Nothing to clean up currently.
}
