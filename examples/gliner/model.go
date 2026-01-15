// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package gliner provides a GLiNER (Generalist and Lightweight Named Entity Recognition) model
// implementation in GoMLX, with support for loading pre-trained weights from safetensors format.
//
// GLiNER uses a DeBERTa-v3 encoder with bidirectional LSTM and span-based entity prediction.
// It can recognize arbitrary entity types specified at inference time.
//
// Reference: https://arxiv.org/abs/2311.08526
// Model: https://huggingface.co/urchade/gliner_small-v2.1
package gliner

import (
	"fmt"
	"path/filepath"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/pkg/errors"

	"github.com/gomlx/gomlx/examples/gliner/safetensors"
)

// Config holds the model configuration.
type Config struct {
	// Model architecture parameters (from gliner_config.json and model structure).
	VocabSize      int // 128004 for DeBERTa-v3
	HiddenSize     int // 768 - DeBERTa hidden dimension
	NumLayers      int // 6 - number of transformer layers
	NumHeads       int // 12 - number of attention heads
	IntermediateFF int // 3072 - feed-forward intermediate size

	// GLiNER specific.
	ProjectionDim int // 512 - projection dimension after DeBERTa
	MaxWidth      int // 12 - maximum span width to consider

	// Inference parameters.
	Threshold float64 // 0.5 - threshold for entity detection
}

// DefaultConfig returns the default configuration for GLiNER small.
func DefaultConfig() *Config {
	return &Config{
		VocabSize:      128004,
		HiddenSize:     768,
		NumLayers:      6,
		NumHeads:       12,
		IntermediateFF: 3072,
		ProjectionDim:  512,
		MaxWidth:       12,
		Threshold:      0.5,
	}
}

// Model represents a GLiNER model with loaded weights.
type Model struct {
	Config    *Config
	WeightsDir string
	weights   *safetensors.File
}

// LoadModel loads a GLiNER model from a directory containing model.safetensors.
func LoadModel(weightsDir string) (*Model, error) {
	safetensorsPath := filepath.Join(weightsDir, "model.safetensors")
	weights, err := safetensors.Open(safetensorsPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load weights from %s", safetensorsPath)
	}

	return &Model{
		Config:     DefaultConfig(),
		WeightsDir: weightsDir,
		weights:    weights,
	}, nil
}

// LoadWeightsIntoContext loads all model weights into the given context.
// This should be called once before building the graph.
func (m *Model) LoadWeightsIntoContext(ctx *context.Context) error {
	ctx = ctx.In("gliner")

	// Load token representation layer (DeBERTa) and projection.
	tokenRepCtx := ctx.In("token_rep")
	if err := m.loadDeBERTaWeights(tokenRepCtx); err != nil {
		return err
	}

	// Load projection layer (nested under token_rep).
	if err := m.loadProjectionWeights(tokenRepCtx.In("projection")); err != nil {
		return err
	}

	// Load BiLSTM weights.
	if err := m.loadLSTMWeights(ctx.In("rnn")); err != nil {
		return err
	}

	// Load span representation layer.
	if err := m.loadSpanRepWeights(ctx.In("span_rep")); err != nil {
		return err
	}

	// Load prompt representation layer.
	if err := m.loadPromptRepWeights(ctx.In("prompt_rep")); err != nil {
		return err
	}

	return nil
}

// loadTensorAsVariable loads a tensor from safetensors and creates a variable in context.
func (m *Model) loadTensorAsVariable(ctx *context.Context, safetensorsName, varName string) error {
	tensor, err := m.weights.ToTensor(safetensorsName)
	if err != nil {
		return errors.Wrapf(err, "failed to load tensor %q", safetensorsName)
	}
	ctx.VariableWithValue(varName, tensor)
	return nil
}

// loadDeBERTaWeights loads the DeBERTa encoder weights.
func (m *Model) loadDeBERTaWeights(ctx *context.Context) error {
	// Embeddings - use "embeddings" to match layers.Embedding variable name.
	embCtx := ctx.In("embeddings")
	if err := m.loadTensorAsVariable(embCtx, "token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight", "embeddings"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(embCtx.In("layer_norm"), "token_rep_layer.bert_layer.model.embeddings.LayerNorm.weight", "gain"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(embCtx.In("layer_norm"), "token_rep_layer.bert_layer.model.embeddings.LayerNorm.bias", "offset"); err != nil {
		return err
	}

	// Relative position embeddings.
	if err := m.loadTensorAsVariable(ctx.In("rel_embeddings"), "token_rep_layer.bert_layer.model.encoder.rel_embeddings.weight", "embeddings"); err != nil {
		return err
	}

	// Encoder layers.
	for layer := 0; layer < m.Config.NumLayers; layer++ {
		layerCtx := ctx.In("encoder").In("layer").In(itoa(layer))
		prefix := "token_rep_layer.bert_layer.model.encoder.layer." + itoa(layer)

		// Self-attention.
		attnCtx := layerCtx.In("attention")
		if err := m.loadTensorAsVariable(attnCtx.In("query"), prefix+".attention.self.query_proj.weight", "weights"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(attnCtx.In("query"), prefix+".attention.self.query_proj.bias", "biases"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(attnCtx.In("key"), prefix+".attention.self.key_proj.weight", "weights"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(attnCtx.In("key"), prefix+".attention.self.key_proj.bias", "biases"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(attnCtx.In("value"), prefix+".attention.self.value_proj.weight", "weights"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(attnCtx.In("value"), prefix+".attention.self.value_proj.bias", "biases"); err != nil {
			return err
		}

		// Attention output.
		if err := m.loadTensorAsVariable(attnCtx.In("output").In("dense"), prefix+".attention.output.dense.weight", "weights"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(attnCtx.In("output").In("dense"), prefix+".attention.output.dense.bias", "biases"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(attnCtx.In("output").In("layer_norm"), prefix+".attention.output.LayerNorm.weight", "gain"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(attnCtx.In("output").In("layer_norm"), prefix+".attention.output.LayerNorm.bias", "offset"); err != nil {
			return err
		}

		// Feed-forward.
		ffCtx := layerCtx.In("ff")
		if err := m.loadTensorAsVariable(ffCtx.In("intermediate").In("dense"), prefix+".intermediate.dense.weight", "weights"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(ffCtx.In("intermediate").In("dense"), prefix+".intermediate.dense.bias", "biases"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(ffCtx.In("output").In("dense"), prefix+".output.dense.weight", "weights"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(ffCtx.In("output").In("dense"), prefix+".output.dense.bias", "biases"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(ffCtx.In("output").In("layer_norm"), prefix+".output.LayerNorm.weight", "gain"); err != nil {
			return err
		}
		if err := m.loadTensorAsVariable(ffCtx.In("output").In("layer_norm"), prefix+".output.LayerNorm.bias", "offset"); err != nil {
			return err
		}
	}

	// Final encoder LayerNorm.
	if err := m.loadTensorAsVariable(ctx.In("encoder").In("final_layer_norm"), "token_rep_layer.bert_layer.model.encoder.LayerNorm.weight", "gain"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(ctx.In("encoder").In("final_layer_norm"), "token_rep_layer.bert_layer.model.encoder.LayerNorm.bias", "offset"); err != nil {
		return err
	}

	return nil
}

// loadProjectionWeights loads the projection layer weights.
func (m *Model) loadProjectionWeights(ctx *context.Context) error {
	if err := m.loadTensorAsVariable(ctx, "token_rep_layer.projection.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(ctx, "token_rep_layer.projection.bias", "biases"); err != nil {
		return err
	}
	return nil
}

// loadLSTMWeights loads the bidirectional LSTM weights.
func (m *Model) loadLSTMWeights(ctx *context.Context) error {
	// Forward direction.
	fwdCtx := ctx.In("forward")
	if err := m.loadTensorAsVariable(fwdCtx, "rnn.lstm.weight_ih_l0", "weight_ih"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(fwdCtx, "rnn.lstm.weight_hh_l0", "weight_hh"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(fwdCtx, "rnn.lstm.bias_ih_l0", "bias_ih"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(fwdCtx, "rnn.lstm.bias_hh_l0", "bias_hh"); err != nil {
		return err
	}

	// Reverse direction.
	revCtx := ctx.In("reverse")
	if err := m.loadTensorAsVariable(revCtx, "rnn.lstm.weight_ih_l0_reverse", "weight_ih"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(revCtx, "rnn.lstm.weight_hh_l0_reverse", "weight_hh"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(revCtx, "rnn.lstm.bias_ih_l0_reverse", "bias_ih"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(revCtx, "rnn.lstm.bias_hh_l0_reverse", "bias_hh"); err != nil {
		return err
	}

	return nil
}

// loadSpanRepWeights loads the span representation layer weights.
func (m *Model) loadSpanRepWeights(ctx *context.Context) error {
	// project_start MLP.
	startCtx := ctx.In("project_start")
	if err := m.loadTensorAsVariable(startCtx.In("0"), "span_rep_layer.span_rep_layer.project_start.0.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(startCtx.In("0"), "span_rep_layer.span_rep_layer.project_start.0.bias", "biases"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(startCtx.In("3"), "span_rep_layer.span_rep_layer.project_start.3.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(startCtx.In("3"), "span_rep_layer.span_rep_layer.project_start.3.bias", "biases"); err != nil {
		return err
	}

	// project_end MLP.
	endCtx := ctx.In("project_end")
	if err := m.loadTensorAsVariable(endCtx.In("0"), "span_rep_layer.span_rep_layer.project_end.0.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(endCtx.In("0"), "span_rep_layer.span_rep_layer.project_end.0.bias", "biases"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(endCtx.In("3"), "span_rep_layer.span_rep_layer.project_end.3.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(endCtx.In("3"), "span_rep_layer.span_rep_layer.project_end.3.bias", "biases"); err != nil {
		return err
	}

	// out_project MLP.
	outCtx := ctx.In("out_project")
	if err := m.loadTensorAsVariable(outCtx.In("0"), "span_rep_layer.span_rep_layer.out_project.0.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(outCtx.In("0"), "span_rep_layer.span_rep_layer.out_project.0.bias", "biases"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(outCtx.In("3"), "span_rep_layer.span_rep_layer.out_project.3.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(outCtx.In("3"), "span_rep_layer.span_rep_layer.out_project.3.bias", "biases"); err != nil {
		return err
	}

	return nil
}

// loadPromptRepWeights loads the prompt representation layer weights.
func (m *Model) loadPromptRepWeights(ctx *context.Context) error {
	if err := m.loadTensorAsVariable(ctx.In("0"), "prompt_rep_layer.0.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(ctx.In("0"), "prompt_rep_layer.0.bias", "biases"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(ctx.In("3"), "prompt_rep_layer.3.weight", "weights"); err != nil {
		return err
	}
	if err := m.loadTensorAsVariable(ctx.In("3"), "prompt_rep_layer.3.bias", "biases"); err != nil {
		return err
	}
	return nil
}

// itoa converts int to string (simple helper).
func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}

// BuildGraph builds the GLiNER forward pass graph.
//
// The input sequence format is: [CLS] <<ENT>> type1 <<ENT>> type2 ... <<SEP>> text [SEP]
// Entity type representations are extracted from positions of <<ENT>> tokens (128002).
//
// Parameters:
//   - ctx: context with loaded weights
//   - inputIDs: token IDs, shape [batch, seq_len]
//   - attentionMask: attention mask, shape [batch, seq_len]
//   - entityTypeIDs: entity type token IDs, shape [batch, num_types, type_seq_len]
//   - entityTypeMask: entity type attention mask, shape [batch, num_types, type_seq_len]
//
// Returns:
//   - spanScores: shape [batch, seq_len, max_width, num_types] - scores for each span and entity type
func (m *Model) BuildGraph(ctx *context.Context, inputIDs, attentionMask, entityTypeIDs, entityTypeMask *Node) *Node {
	ctx = ctx.In("gliner").Checked(false)
	g := inputIDs.Graph()

	// 1. Token representation (DeBERTa encoder + projection).
	// This processes the entire sequence including entity type markers.
	tokenRep := m.buildTokenRepresentation(ctx.In("token_rep"), inputIDs, attentionMask)
	// tokenRep shape: [batch, seq_len, projection_dim]

	// 2. Extract entity type embeddings from <<ENT>> token positions in the encoder output.
	numTypes := entityTypeIDs.Shape().Dimensions[1]
	batchSize := inputIDs.Shape().Dimensions[0]
	entityRep := m.extractEntityTypeEmbeddings(tokenRep, inputIDs, numTypes)
	// entityRep shape: [batch, num_types, projection_dim]

	// 3. Apply prompt representation MLP to entity type embeddings.
	// Flatten to 2D for MLP: [batch * num_types, projection_dim]
	projDim := entityRep.Shape().Dimensions[2]
	entityRep = Reshape(entityRep, batchSize*numTypes, projDim)
	entityRep = applyMLP(ctx.In("prompt_rep"), entityRep, g)
	// Reshape back to 3D: [batch, num_types, projection_dim]
	entityRep = Reshape(entityRep, batchSize, numTypes, projDim)

	// 4. BiLSTM on token representations.
	tokenRep = m.buildBiLSTM(ctx.In("rnn"), tokenRep)
	// tokenRep shape: [batch, seq_len, projection_dim]

	// 5. Span representation.
	spanRep := m.buildSpanRepresentation(ctx.In("span_rep"), tokenRep)
	// spanRep shape: [batch, seq_len, max_width, projection_dim]

	// 6. Score spans against entity types.
	scores := m.buildSpanScoring(g, spanRep, entityRep)
	// scores shape: [batch, seq_len, max_width, num_types]

	return scores
}

// extractEntityTypeEmbeddings extracts embeddings at positions of <<ENT>> tokens.
// The <<ENT>> token (128002) marks entity type positions in the input sequence.
func (m *Model) extractEntityTypeEmbeddings(tokenRep, inputIDs *Node, numTypes int) *Node {
	// In GLiNER, entity types are marked with <<ENT>> tokens.
	// Input format: [CLS] <<ENT>> type1 <<ENT>> type2 ... <<SEP>> text
	// With single-token type names, <<ENT>> tokens are at positions 1, 3, 5, ...
	// We extract embeddings at these positions.

	batchSize := tokenRep.Shape().Dimensions[0]
	projDim := tokenRep.Shape().Dimensions[2]

	entityEmbeddings := make([]*Node, numTypes)
	for i := 0; i < numTypes; i++ {
		// Position of i-th entity type token (after <<ENT>>)
		// Format: [CLS]=0, [<<ENT>>=1, type1=2], [<<ENT>>=3, type2=4], ...
		// We use the entity type token (not <<ENT>>) for differentiated embeddings.
		pos := 2 + i*2

		// Extract embedding at this position: Slice gives [batch, 1, projDim]
		emb := Slice(tokenRep, AxisRange(), AxisRange(pos, pos+1), AxisRange())
		// Reshape to remove the size-1 dimension: [batch, projDim]
		emb = Reshape(emb, batchSize, projDim)
		entityEmbeddings[i] = emb
	}

	// Stack along new dimension: [batch, num_types, projDim]
	// First insert dimension for each: [batch, 1, projDim]
	for i := range entityEmbeddings {
		entityEmbeddings[i] = InsertAxes(entityEmbeddings[i], 1)
	}

	// Concatenate along type dimension: [batch, num_types, projDim]
	if len(entityEmbeddings) == 1 {
		return entityEmbeddings[0]
	}
	return Concatenate(entityEmbeddings, 1)
}

// buildTokenRepresentation builds the token representation from input IDs.
func (m *Model) buildTokenRepresentation(ctx *context.Context, inputIDs, attentionMask *Node) *Node {
	g := inputIDs.Graph()

	// Get embeddings.
	embCtx := ctx.In("embeddings")
	embeddings := layers.Embedding(embCtx, inputIDs, dtypes.Float32, m.Config.VocabSize, m.Config.HiddenSize)

	// Ensure 3D output: [batch, seq, hidden].
	// layers.Embedding may return 2D when seq_len=1 due to how it handles trailing dimensions.
	if embeddings.Shape().Rank() == 2 {
		// Insert sequence dimension: [batch, hidden] -> [batch, 1, hidden]
		embeddings = InsertAxes(embeddings, 1)
	}

	// Apply LayerNorm to embeddings.
	embeddings = applyLayerNorm(embCtx.In("layer_norm"), embeddings, g)

	// Run through DeBERTa encoder layers.
	hidden := embeddings
	for layer := 0; layer < m.Config.NumLayers; layer++ {
		hidden = m.buildEncoderLayer(ctx.In("encoder").In("layer").In(itoa(layer)), hidden, attentionMask, layer)
	}

	// Final encoder LayerNorm.
	hidden = applyLayerNorm(ctx.In("encoder").In("final_layer_norm"), hidden, g)

	// Project to smaller dimension.
	hidden = applyDenseWithBias(ctx.In("projection"), hidden, g)

	return hidden
}

// buildEncoderLayer builds a single DeBERTa encoder layer.
func (m *Model) buildEncoderLayer(ctx *context.Context, hidden, attentionMask *Node, layerIdx int) *Node {
	g := hidden.Graph()

	// Self-attention with disentangled attention (simplified - using standard attention for now).
	attnCtx := ctx.In("attention")
	residual := hidden

	// Q, K, V projections.
	query := applyDenseWithBias(attnCtx.In("query"), hidden, g)
	key := applyDenseWithBias(attnCtx.In("key"), hidden, g)
	value := applyDenseWithBias(attnCtx.In("value"), hidden, g)

	// Reshape for multi-head attention: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	headDim := m.Config.HiddenSize / m.Config.NumHeads

	query = Reshape(query, batchSize, seqLen, m.Config.NumHeads, headDim)
	key = Reshape(key, batchSize, seqLen, m.Config.NumHeads, headDim)
	value = Reshape(value, batchSize, seqLen, m.Config.NumHeads, headDim)

	// Transpose to [batch, heads, seq, head_dim].
	query = Transpose(query, 1, 2)
	key = Transpose(key, 1, 2)
	value = Transpose(value, 1, 2)

	// Scaled dot-product attention.
	scores := Einsum("bhqd,bhkd->bhqk", query, key)
	scale := ConstAs(scores, 1.0/float64(headDim))
	scores = Mul(scores, Sqrt(scale))

	// Apply attention mask if provided.
	if attentionMask != nil {
		// Expand mask: [batch, seq] -> [batch, 1, 1, seq]
		// InsertAxes(x, 1, 1) inserts two axes at position 1 (before seq dimension).
		mask := InsertAxes(attentionMask, 1, 1)
		// Broadcast to match scores shape: [batch, heads, query_seq, key_seq]
		mask = BroadcastToDims(mask, scores.Shape().Dimensions...)
		// Convert mask to large negative for masked positions.
		negInf := ConstAs(scores, -1e9)
		zeroMask := Equal(mask, ScalarZero(g, mask.DType()))
		scores = Where(zeroMask, negInf, scores)
	}

	attnWeights := Softmax(scores, -1)
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, value)

	// Transpose back: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
	attnOutput = Transpose(attnOutput, 1, 2)
	// Reshape: [batch, seq, heads, head_dim] -> [batch, seq, hidden]
	attnOutput = Reshape(attnOutput, batchSize, seqLen, m.Config.HiddenSize)

	// Output projection and residual.
	attnOutput = applyDenseWithBias(attnCtx.In("output").In("dense"), attnOutput, g)
	hidden = Add(residual, attnOutput)
	hidden = applyLayerNorm(attnCtx.In("output").In("layer_norm"), hidden, g)

	// Feed-forward network.
	ffCtx := ctx.In("ff")
	residual = hidden
	hidden = applyDenseWithBias(ffCtx.In("intermediate").In("dense"), hidden, g)
	hidden = activations.Gelu(hidden)
	hidden = applyDenseWithBias(ffCtx.In("output").In("dense"), hidden, g)
	hidden = Add(residual, hidden)
	hidden = applyLayerNorm(ffCtx.In("output").In("layer_norm"), hidden, g)

	return hidden
}

// buildBiLSTM applies bidirectional LSTM to the token representations.
func (m *Model) buildBiLSTM(ctx *context.Context, x *Node) *Node {
	g := x.Graph()

	// Get LSTM weights.
	fwdCtx := ctx.In("forward")
	revCtx := ctx.In("reverse")

	// Forward LSTM.
	fwdOutput := lstmForward(fwdCtx, x, g)

	// Reverse LSTM (reverse input, apply LSTM, reverse output).
	xReversed := Reverse(x, 1)
	revOutput := lstmForward(revCtx, xReversed, g)
	revOutput = Reverse(revOutput, 1)

	// Concatenate forward and reverse outputs.
	combined := Concatenate([]*Node{fwdOutput, revOutput}, -1)

	return combined
}

// lstmForward runs LSTM in the forward direction.
func lstmForward(ctx *context.Context, x *Node, g *Graph) *Node {
	// Get weights.
	weightIH := ctx.GetVariableByScopeAndName(ctx.Scope(), "weight_ih").ValueGraph(g)
	weightHH := ctx.GetVariableByScopeAndName(ctx.Scope(), "weight_hh").ValueGraph(g)
	biasIH := ctx.GetVariableByScopeAndName(ctx.Scope(), "bias_ih").ValueGraph(g)
	biasHH := ctx.GetVariableByScopeAndName(ctx.Scope(), "bias_hh").ValueGraph(g)

	batchSize := x.Shape().Dimensions[0]
	seqLen := x.Shape().Dimensions[1]
	inputSize := x.Shape().Dimensions[2]
	hiddenSize := weightHH.Shape().Dimensions[1] // weight_hh is [4*hidden, hidden]

	// Initialize hidden state and cell state.
	h := Zeros(g, shapes.Make(x.DType(), batchSize, hiddenSize))
	c := Zeros(g, shapes.Make(x.DType(), batchSize, hiddenSize))

	// Process each timestep.
	outputs := make([]*Node, seqLen)
	for t := 0; t < seqLen; t++ {
		// Get input at timestep t: [batch, input_size]
		xt := Slice(x, AxisRange(), AxisElem(t), AxisRange())
		xt = Reshape(xt, batchSize, inputSize)

		// gates = x @ W_ih.T + h @ W_hh.T + b_ih + b_hh
		// Reshape biases for broadcasting: [4*hidden] -> [1, 4*hidden]
		biasIHReshaped := Reshape(biasIH, 1, biasIH.Shape().Dimensions[0])
		biasHHReshaped := Reshape(biasHH, 1, biasHH.Shape().Dimensions[0])
		gates := Add(
			Add(Dot(xt, Transpose(weightIH, 0, 1)), biasIHReshaped),
			Add(Dot(h, Transpose(weightHH, 0, 1)), biasHHReshaped),
		)

		// Split gates into i, f, g, o (each of size hiddenSize).
		i := Sigmoid(Slice(gates, AxisRange(), AxisRange(0, hiddenSize)))
		f := Sigmoid(Slice(gates, AxisRange(), AxisRange(hiddenSize, 2*hiddenSize)))
		gGate := Tanh(Slice(gates, AxisRange(), AxisRange(2*hiddenSize, 3*hiddenSize)))
		o := Sigmoid(Slice(gates, AxisRange(), AxisRange(3*hiddenSize, 4*hiddenSize)))

		// Update cell and hidden state.
		c = Add(Mul(f, c), Mul(i, gGate))
		h = Mul(o, Tanh(c))

		outputs[t] = InsertAxes(h, 1) // Add sequence dimension back.
	}

	// Concatenate outputs along sequence dimension.
	return Concatenate(outputs, 1)
}

// buildEntityTypeRepresentation builds representations for entity types.
// Uses a simpler path than the full encoder: embeddings → mean pooling → prompt_rep MLP.
func (m *Model) buildEntityTypeRepresentation(ctx *context.Context, entityTypeIDs, entityTypeMask *Node) *Node {
	g := entityTypeIDs.Graph()

	// Shape: [batch, num_types, type_seq_len]
	batchSize := entityTypeIDs.Shape().Dimensions[0]
	numTypes := entityTypeIDs.Shape().Dimensions[1]
	typeSeqLen := entityTypeIDs.Shape().Dimensions[2]

	// Flatten batch and num_types for embedding lookup.
	flatIDs := Reshape(entityTypeIDs, batchSize*numTypes, typeSeqLen)
	flatMask := Reshape(entityTypeMask, batchSize*numTypes, typeSeqLen)

	// Get embeddings directly (same embedding table as main encoder).
	embCtx := ctx.In("token_rep").In("embeddings")
	embeddings := layers.Embedding(embCtx, flatIDs, dtypes.Float32, m.Config.VocabSize, m.Config.HiddenSize)
	// embeddings shape: [batch*num_types, type_seq_len, hidden_size]

	// Ensure 3D output (handle seq_len=1 case).
	if embeddings.Shape().Rank() == 2 {
		embeddings = InsertAxes(embeddings, 1)
	}

	// Project embeddings from hidden_size (768) to projection_dim (512).
	projected := applyDenseWithBias(ctx.In("token_rep").In("projection"), embeddings, g)
	// projected shape: [batch*num_types, type_seq_len, projection_dim]

	// Mean pooling over sequence (using mask).
	maskExpanded := InsertAxes(flatMask, -1)
	maskExpanded = ConvertDType(maskExpanded, projected.DType())
	maskedProj := Mul(projected, maskExpanded)
	pooled := ReduceSum(maskedProj, 1)
	maskSum := ReduceSum(maskExpanded, 1)
	pooled = Div(pooled, Add(maskSum, ConstAs(maskSum, 1e-9)))
	// pooled shape: [batch*num_types, projection_dim]

	// Apply prompt representation MLP.
	typeRep := applyMLP(ctx.In("prompt_rep"), pooled, g)
	// typeRep shape: [batch*num_types, projection_dim]

	// Reshape back to [batch, num_types, projection_dim].
	projDim := typeRep.Shape().Dimensions[1]
	typeRep = Reshape(typeRep, batchSize, numTypes, projDim)

	return typeRep
}

// buildSpanRepresentation builds span representations from token representations.
func (m *Model) buildSpanRepresentation(ctx *context.Context, tokenRep *Node) *Node {
	g := tokenRep.Graph()

	batchSize := tokenRep.Shape().Dimensions[0]
	seqLen := tokenRep.Shape().Dimensions[1]
	projDim := tokenRep.Shape().Dimensions[2]

	// Project start and end positions.
	startRep := applyMLP(ctx.In("project_start"), tokenRep, g)
	endRep := applyMLP(ctx.In("project_end"), tokenRep, g)
	// Both: [batch, seq_len, projection_dim]

	// Create span representations for all spans up to max_width.
	// For each starting position i, we create spans (i, i), (i, i+1), ..., (i, i+max_width-1).
	spanReps := make([]*Node, m.Config.MaxWidth)

	for width := 0; width < m.Config.MaxWidth; width++ {
		// Start representations: [batch, seq_len, proj_dim]
		start := startRep

		// End representations: shift by width positions.
		// We need end positions at i+width for start position i.
		if width == 0 {
			// Same position.
			spanReps[width] = applyMLP(ctx.In("out_project"), Concatenate([]*Node{start, endRep}, -1), g)
		} else {
			// Shift end representations.
			// Pad at the end and slice from the beginning.
			padding := Zeros(g, shapes.Make(endRep.DType(), batchSize, width, projDim))
			shiftedEnd := Concatenate([]*Node{Slice(endRep, AxisRange(), AxisRange(width, seqLen), AxisRange()), padding}, 1)
			combined := Concatenate([]*Node{start, shiftedEnd}, -1)
			spanReps[width] = applyMLP(ctx.In("out_project"), combined, g)
		}
	}

	// Stack span representations: [batch, seq_len, max_width, projection_dim]
	for i := range spanReps {
		spanReps[i] = InsertAxes(spanReps[i], 2)
	}
	return Concatenate(spanReps, 2)
}

// buildSpanScoring computes scores between spans and entity types.
func (m *Model) buildSpanScoring(g *Graph, spanRep, entityRep *Node) *Node {
	// spanRep: [batch, seq_len, max_width, projection_dim]
	// entityRep: [batch, num_types, projection_dim]

	// Dot product between span representations and entity representations.
	// scores[b, i, w, t] = spanRep[b, i, w, :] . entityRep[b, t, :]
	scores := Einsum("biwd,btd->biwt", spanRep, entityRep)

	// Apply sigmoid to convert logits to probabilities.
	scores = Sigmoid(scores)

	return scores
}

// Helper functions for applying layers with pre-loaded weights.

// applyLayerNorm applies layer normalization using pre-loaded weights.
func applyLayerNorm(ctx *context.Context, x *Node, _ *Graph) *Node {
	g := x.Graph()
	gainVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "gain")
	if gainVar == nil {
		panic(fmt.Sprintf("missing variable 'gain' in scope %q", ctx.Scope()))
	}
	gain := gainVar.ValueGraph(g)
	offsetVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "offset")
	if offsetVar == nil {
		panic(fmt.Sprintf("missing variable 'offset' in scope %q", ctx.Scope()))
	}
	offset := offsetVar.ValueGraph(g)

	// Normalize.
	mean := ReduceAndKeep(x, ReduceMean, -1)
	normalized := Sub(x, mean)
	variance := ReduceAndKeep(Square(normalized), ReduceMean, -1)
	epsilon := ConstAs(x, 1e-7)
	normalized = Div(normalized, Sqrt(Add(variance, epsilon)))

	// Reshape gain and offset to broadcast with x.
	// gain/offset shape: [hidden_size] -> [1, 1, ..., hidden_size] to match x's rank.
	xRank := x.Shape().Rank()
	broadcastShape := make([]int, xRank)
	for i := range broadcastShape {
		broadcastShape[i] = 1
	}
	broadcastShape[xRank-1] = gain.Shape().Dimensions[0]

	gain = Reshape(gain, broadcastShape...)
	offset = Reshape(offset, broadcastShape...)

	// Apply gain and offset.
	normalized = Mul(normalized, gain)
	normalized = Add(normalized, offset)

	return normalized
}

// applyDenseWithBias applies a dense layer using pre-loaded weights.
func applyDenseWithBias(ctx *context.Context, x *Node, _ *Graph) *Node {
	g := x.Graph()
	weightsVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights")
	if weightsVar == nil {
		panic(fmt.Sprintf("missing variable 'weights' in scope %q", ctx.Scope()))
	}
	weights := weightsVar.ValueGraph(g)
	biasesVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "biases")
	if biasesVar == nil {
		panic(fmt.Sprintf("missing variable 'biases' in scope %q", ctx.Scope()))
	}
	biases := biasesVar.ValueGraph(g)

	// Linear transformation: x @ weights.T + bias.
	// weights shape: [out_features, in_features] (PyTorch convention)
	// Use Einsum to handle arbitrary input ranks (2D or 3D).
	// For 3D input [batch, seq, in]: "bsi,oi->bso"
	// For 2D input [batch, in]: "bi,oi->bo"
	var output *Node
	if x.Shape().Rank() == 3 {
		output = Einsum("bsi,oi->bso", x, weights)
		// Reshape biases for broadcasting: [out] -> [1, 1, out]
		biases = Reshape(biases, 1, 1, biases.Shape().Dimensions[0])
	} else {
		output = Einsum("bi,oi->bo", x, weights)
		// Reshape biases for broadcasting: [out] -> [1, out]
		biases = Reshape(biases, 1, biases.Shape().Dimensions[0])
	}
	output = Add(output, biases)

	return output
}

// applyMLP applies a 2-layer MLP with ReLU activation.
func applyMLP(ctx *context.Context, x *Node, g *Graph) *Node {
	// First layer.
	x = applyDenseWithBias(ctx.In("0"), x, g)
	x = activations.Relu(x)

	// Second layer.
	x = applyDenseWithBias(ctx.In("3"), x, g)

	return x
}
