// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// gemma3n demonstrates multimodal (text + image) generation using GoMLX's
// serving engine with the Gemma 3n E2B model.
//
// It downloads the onnx-community/gemma-3n-E2B-it-ONNX model from HuggingFace,
// loads separate ONNX models for vision encoding, token embedding, and text
// decoding, and uses the serving engine's multimodal AuxInputs support to
// stream generated tokens.
//
// Usage:
//
//	go run gemma3n.go
//	go run gemma3n.go --prompt="Describe this image" --image=photo.jpg
//	go run gemma3n.go --prompt="What is Go?" (text-only)
//	go run gemma3n.go --max-tokens=200
package main

import (
	stdctx "context"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	gomlxImages "github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/ajroetker/huggingface-gomlx/kvcache"
	"github.com/ajroetker/huggingface-gomlx/serving"
	"github.com/gomlx/onnx-gomlx/onnx"
	"golang.org/x/image/draw"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
)

const (
	modelRepo = "onnx-community/gemma-3n-E2B-it-ONNX"

	// Number of image feature tokens produced by the vision encoder.
	numImageTokens = 256
)

var (
	flagPrompt    = flag.String("prompt", "What do you see in this image?", "User message for the chat prompt.")
	flagImage     = flag.String("image", "", "Path to an image file. If empty, runs in text-only mode.")
	flagImageSize = flag.Int("image-size", 768, "Target image size for vision encoder (256, 512, or 768).")
	flagMaxTokens       = flag.Int("max-tokens", 200, "Maximum number of tokens to generate.")
	flagMaxSeqLen       = flag.Int("max-seq-len", 512, "Maximum total sequence length (prompt + generated tokens).")
	flagBackend         = flag.String("backend", "", "Backend to use (default: auto-detect).")
	flagCompaction      = flag.Bool("compaction", false, "Enable KV cache compaction after prefill.")
	flagCompactionRatio = flag.Int("compaction-ratio", 2, "Compaction ratio (e.g., 2 = halve cache, 4 = quarter).")
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagBackend != "" {
		if err := os.Setenv("GOMLX_BACKEND", *flagBackend); err != nil {
			klog.Warningf("Failed to set backend: %v", err)
		}
	}

	// Download and cache model files from HuggingFace.
	fmt.Printf("Downloading model: %s\n", modelRepo)
	repo := hub.New(modelRepo).WithProgressBar(true)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to get repo info: %+v", err)
	}

	// Download the ONNX model files.
	embedPath := mustDownload(repo, "onnx/embed_tokens.onnx")
	visionPath := mustDownload(repo, "onnx/vision_encoder.onnx")
	decoderPath := mustDownload(repo, "onnx/decoder_model_merged.onnx")
	// Also download external data files (including split files).
	for _, dataFile := range []string{
		"onnx/embed_tokens.onnx_data",
		"onnx/embed_tokens.onnx_data_1",
		"onnx/vision_encoder.onnx_data",
		"onnx/decoder_model_merged.onnx_data",
		"onnx/decoder_model_merged.onnx_data_1",
		"onnx/decoder_model_merged.onnx_data_2",
		"onnx/decoder_model_merged.onnx_data_3",
		"onnx/decoder_model_merged.onnx_data_4",
	} {
		tryDownload(repo, dataFile)
	}
	fmt.Println("Model files downloaded.")

	// Load tokenizer.
	// The ONNX community repo has tokenizer.json but not tokenizer.model,
	// so try the standard tokenizer first, then fall back to HF tokenizer.
	tok, err := tokenizers.New(repo)
	if err != nil {
		config, configErr := tokenizers.GetConfig(repo)
		if configErr != nil {
			klog.Fatalf("Failed to create tokenizer: %+v (and config: %+v)", err, configErr)
		}
		tok, err = hftokenizer.New(config, repo)
		if err != nil {
			klog.Fatalf("Failed to create tokenizer: %+v", err)
		}
	}

	// Load ONNX models.
	embedModel, err := onnx.ReadFile(embedPath)
	if err != nil {
		klog.Fatalf("Failed to load embed model: %+v", err)
	}
	defer embedModel.Close()

	visionModel, err := onnx.ReadFile(visionPath)
	if err != nil {
		klog.Fatalf("Failed to load vision model: %+v", err)
	}
	defer visionModel.Close()

	decoderModel, err := onnx.ReadFile(decoderPath)
	if err != nil {
		klog.Fatalf("Failed to load decoder model: %+v", err)
	}
	defer decoderModel.Close()

	// Print model structures.
	printModelInfo("Embed tokens", embedModel)
	printModelInfo("Vision encoder", visionModel)
	printModelInfo("Decoder", decoderModel)

	// Parse decoder KV cache structure.
	kv := parseKVStructure(decoderModel)
	if !kv.hasOutputs() {
		klog.Fatalf("Decoder model does not support KV cache outputs.")
	}
	fmt.Printf("Decoder KV cache: %d layers, %d heads, dim=%d\n\n", kv.numLayers, kv.kvHeads, kv.headDim)

	// Load model weights into context.
	ctx := context.New()
	if err := embedModel.VariablesToContext(ctx); err != nil {
		klog.Fatalf("Failed to load embed model variables: %+v", err)
	}
	if err := visionModel.VariablesToContext(ctx); err != nil {
		klog.Fatalf("Failed to load vision model variables: %+v", err)
	}
	if err := decoderModel.VariablesToContext(ctx); err != nil {
		klog.Fatalf("Failed to load decoder model variables: %+v", err)
	}

	// Initialize backend.
	backend := backends.MustNew()
	fmt.Printf("Backend: %s\n\n", backend.Name())

	// Discover image_token_id from the tokenizer.
	imageTokenID := int32(-1)
	if encoded := tok.Encode("<image_soft_token>"); len(encoded) == 1 {
		imageTokenID = int32(encoded[0])
	}
	if imageTokenID < 0 {
		if encoded := tok.Encode("<img>"); len(encoded) == 1 {
			imageTokenID = int32(encoded[0])
		}
	}
	fmt.Printf("Image token ID: %d\n", imageTokenID)

	// Look up EOS and end-of-turn tokens.
	eosID, err := tok.SpecialTokenID(api.TokEndOfSentence)
	if err != nil {
		eosID = 1
	}
	endOfTurnID := -1
	if encoded := tok.Encode("<end_of_turn>"); len(encoded) == 1 {
		endOfTurnID = encoded[0]
	}

	// Prepare image features if an image was provided.
	var imageFeatures *tensors.Tensor
	hasImage := *flagImage != ""
	if hasImage {
		fmt.Printf("Loading image: %s\n", *flagImage)
		imgTensor, imgErr := loadAndPreprocessImage(*flagImage, *flagImageSize)
		if imgErr != nil {
			klog.Fatalf("Failed to load image: %v", imgErr)
		}
		fmt.Printf("Image tensor shape: %v\n", imgTensor.Shape())

		fmt.Println("Running vision encoder...")
		var visionErr error
		imageFeatures, visionErr = runVisionEncoder(backend, ctx, visionModel, imgTensor)
		if visionErr != nil {
			klog.Fatalf("Vision encoder failed: %v", visionErr)
		}
		fmt.Printf("Image features shape: %v\n\n", imageFeatures.Shape())
	}

	// Tokenize prompt.
	prompt := formatChatPrompt(*flagPrompt, hasImage)
	promptTokens := tokenizePrompt(tok, prompt, hasImage, imageTokenID)
	fmt.Printf("Prompt: %q\n", *flagPrompt)
	fmt.Printf("Tokenized to %d tokens (including %d image tokens)\n", len(promptTokens), countImageTokens(promptTokens, imageTokenID))
	fmt.Printf("Token IDs: %v\n\n", promptTokens)

	maxSeqLen := *flagMaxSeqLen
	if len(promptTokens) >= maxSeqLen {
		klog.Fatalf("Prompt too long (%d tokens), max is %d", len(promptTokens), maxSeqLen)
	}

	// Pre-create KV cache variables.
	cacheShape := shapes.Make(kv.kvDType, 1, kv.kvHeads, maxSeqLen, kv.headDim)
	for i := range kv.numLayers {
		layerCtx := ctx.In(fmt.Sprintf("kv_layer_%d", i))
		attention.KVCacheGetVars(layerCtx, cacheShape)
	}

	// Reset KV cache variables before engine use.
	attention.KVCacheReset(ctx)

	// Build the ModelFn (decoder-only, receives pre-computed embeddings via AuxInputs).
	modelFn := makeModelFn(decoderModel, kv, maxSeqLen)

	// Create the serving engine.
	tokWrapper := &servingTokenizer{tok: tok, eosID: eosID, endOfTurnID: endOfTurnID}
	engineCfg := serving.Config{
		MaxSeqLen:    maxSeqLen,
		MaxBatchSize: 1,
	}
	if *flagCompaction {
		ratio := *flagCompactionRatio
		if ratio < 2 {
			ratio = 2
		}
		targetLen := len(promptTokens) / ratio
		if targetLen < 16 {
			targetLen = 16
		}
		engineCfg.Compaction = &kvcache.CompactionConfig{
			TargetLen:     targetLen,
			NumRefQueries: 64,
			MinSeqLen:     32,
		}
		fmt.Printf("Compaction enabled: %dx (target %d tokens from %d)\n", ratio, targetLen, len(promptTokens))
	}
	eng := serving.NewEngine(backend, ctx, modelFn, tokWrapper, engineCfg, kv.kvHeads, kv.headDim, kv.kvDType)
	defer eng.Stop()

	// Set EmbedFn: implements embedding natively in GoMLX using Gather on
	// the weight matrices loaded from the ONNX embed_tokens model. This avoids
	// the NonZero op in the ONNX graph which is incompatible with static
	// graph compilation.
	embedFn := makeEmbedFn(backend, ctx, imageTokenID, imageFeatures, kv.numLayers, kv.headDim)
	eng.SetEmbedFn(embedFn)

	// AuxData is nil since embedding + image merging happen inside EmbedFn.
	var auxData *serving.AuxData

	// Generate.
	fmt.Println("Generating...")
	fmt.Println("---")
	startTime := time.Now()
	n := generateWithEngine(eng, promptTokens, *flagMaxTokens, auxData)
	dur := time.Since(startTime)
	fmt.Println("\n---")
	if n > 0 {
		tokensPerSec := float64(n) / dur.Seconds()
		fmt.Printf("Generated %d tokens in %.2fs (%.1f tokens/s)\n", n, dur.Seconds(), tokensPerSec)
	}
}

// generateWithEngine submits a prompt to the serving engine and streams output.
func generateWithEngine(eng *serving.Engine, promptTokens []int32, maxTokens int, auxData *serving.AuxData) int {
	outputCh, errCh, err := eng.Submit(
		stdctx.Background(),
		promptTokens,
		serving.RequestOptions{MaxNewTokens: maxTokens},
		auxData,
	)
	if err != nil {
		klog.Errorf("Submit failed: %v", err)
		return 0
	}

	tokensGenerated := 0
	for delta := range outputCh {
		if delta.EOSReached {
			break
		}
		fmt.Print(delta.Token)
		tokensGenerated++
	}
	for range outputCh {
	}
	if err := <-errCh; err != nil {
		klog.Errorf("Generation error: %v", err)
	}
	return tokensGenerated
}

// makeEmbedFn creates an EmbedFn that implements token embedding natively in GoMLX.
//
// The embed_tokens ONNX model uses NonZero (data-dependent shapes) for special
// token routing, which is incompatible with GoMLX's static graph compilation.
// Instead, we implement the embedding as native GoMLX Gather operations on the
// weight matrices loaded from the ONNX model.
//
// The computation mirrors the ONNX embed_tokens graph:
//   - inputs_embeds = Gather(embed_tokens.weight, input_ids) * sqrt(2048) → [batch, seq, 2048]
//   - clamped_ids = Where(input_ids < 262144, input_ids, 0)
//   - per_layer_inputs = Gather(embed_tokens_per_layer.weight, clamped_ids) * 16.0
//     → [batch, seq, 7680] → reshape [batch, seq, 30, 256]
//
// For multimodal prefill, image features are merged into the embeddings at
// image placeholder token positions.
func makeEmbedFn(
	backend backends.Backend,
	modelCtx *context.Context,
	imageTokenID int32,
	imageFeatures *tensors.Tensor, // nil for text-only
	numLayers int,
	perLayerDim int,
) serving.EmbedFn {
	// Cache compiled executors by input shape to avoid recompilation.
	var cachedExec *context.Exec

	return func(tokens *tensors.Tensor, auxData *serving.AuxData) (*serving.AuxData, error) {
		if cachedExec == nil {
			var err error
			cachedExec, err = context.NewExec(backend, modelCtx.Reuse(),
				func(ctx *context.Context, tokenIDs *Node) []*Node {
					g := tokenIDs.Graph()

					// Look up embedding weights from context (loaded from ONNX model).
					embedWeight := ctx.GetVariableByScopeAndName("/ONNX", "model.embed_tokens.weight").ValueGraph(g)
					perLayerWeight := ctx.GetVariableByScopeAndName("/ONNX", "model.embed_tokens_per_layer.weight").ValueGraph(g)

					// Gather embeddings: [batch, seq] → [batch, seq, hiddenDim]
					// GoMLX Gather uses the last dimension of indices as the number of
					// indexed dimensions (N). We need N=1 to index only dim-0 of the
					// weight table, so add a trailing axis: [batch, seq] → [batch, seq, 1].
					ids := InsertAxes(ConvertDType(tokenIDs, dtypes.Int32), -1)
					inputsEmbeds := Gather(embedWeight, ids)
					// Gemma models scale embeddings by sqrt(hidden_dim).
					inputsEmbeds = MulScalar(inputsEmbeds, 45.25) // sqrt(2048) ≈ 45.254834

					// Per-layer embeddings: clamp token IDs to [0, 262144) since the
					// per-layer weight table has fewer entries than the full vocab.
					clampedIDs := ConvertDType(tokenIDs, dtypes.Int32)
					maxID := Scalar(g, dtypes.Int32, 262144)
					clampedIDs = Where(LessThan(clampedIDs, maxID), clampedIDs, ZerosLike(clampedIDs))
					clampedIDs = InsertAxes(clampedIDs, -1)

					// Gather per-layer inputs, scale, and reshape.
					perLayerFlat := Gather(perLayerWeight, clampedIDs)
					perLayerFlat = MulScalar(perLayerFlat, 16.0) // per-layer embedding scale
					batchSize := perLayerFlat.Shape().Dimensions[0]
					seqLen := perLayerFlat.Shape().Dimensions[1]
					perLayerInputs := Reshape(perLayerFlat, batchSize, seqLen, numLayers, perLayerDim)

					return []*Node{inputsEmbeds, perLayerInputs}
				},
			)
			if err != nil {
				return nil, fmt.Errorf("compile native embed: %w", err)
			}
		}

		results, err := cachedExec.Exec(tokens)
		if err != nil {
			return nil, fmt.Errorf("run native embed: %w", err)
		}

		result := &serving.AuxData{
			InputsEmbeds:   results[0],
			PerLayerInputs: results[1],
		}

		// Merge image features into embeddings at image token positions (prefill only).
		if imageFeatures != nil && imageTokenID >= 0 {
			result.InputsEmbeds = mergeImageFeatures(
				result.InputsEmbeds, imageFeatures, tokens, imageTokenID,
			)
		}
		if auxData != nil && auxData.ImageFeatures != nil && imageTokenID >= 0 {
			result.InputsEmbeds = mergeImageFeatures(
				result.InputsEmbeds, auxData.ImageFeatures, tokens, imageTokenID,
			)
		}

		return result, nil
	}
}

// mergeImageFeatures replaces embeddings at image token positions with vision features.
func mergeImageFeatures(embeds, imageFeats, tokens *tensors.Tensor, imageTokenID int32) *tensors.Tensor {
	tokenVals := tokens.Value()
	var tokenIDs []int32
	switch v := tokenVals.(type) {
	case [][]int32:
		tokenIDs = v[0] // batch=1
	case []int32:
		tokenIDs = v
	}

	// For single tokens during decode, there's nothing to merge.
	if len(tokenIDs) <= 1 {
		return embeds
	}

	embedShape := embeds.Shape()
	if embedShape.Rank() != 3 {
		return embeds
	}
	seqLen := embedShape.Dimensions[1]
	hiddenDim := embedShape.Dimensions[2]

	// Get flat float32 data (copy so we can mutate).
	var embedFlat []float32
	embeds.ConstFlatData(func(flat any) {
		src := flat.([]float32)
		embedFlat = make([]float32, len(src))
		copy(embedFlat, src)
	})

	var imgFlat []float32
	imageFeats.ConstFlatData(func(flat any) {
		imgFlat = flat.([]float32)
	})
	numImgTokens := len(imgFlat) / hiddenDim

	// Replace embeddings at image token positions.
	imgIdx := 0
	for pos := range seqLen {
		if pos < len(tokenIDs) && tokenIDs[pos] == imageTokenID && imgIdx < numImgTokens {
			copy(embedFlat[pos*hiddenDim:(pos+1)*hiddenDim], imgFlat[imgIdx*hiddenDim:(imgIdx+1)*hiddenDim])
			imgIdx++
		}
	}

	result := tensors.FromShape(embedShape)
	result.MutableFlatData(func(flat any) {
		copy(flat.([]float32), embedFlat)
	})
	return result
}

// makeModelFn wraps the decoder ONNX model into a ModelFn for the serving engine.
// It expects pre-computed embeddings via aux.InputsEmbeds (provided by EmbedFn).
func makeModelFn(
	decoderModel *onnx.Model,
	kv *kvStructure,
	maxSeqLen int,
) decode.ModelFn {
	// Prepare empty KV constants for prefill.
	emptyKV := make(map[string]any)
	for i := range kv.numLayers {
		emptyKV[kv.inputKeyNames[i]] = tensors.FromShape(shapes.Make(kv.kvDType, 1, kv.kvHeads, 0, kv.headDim))
		emptyKV[kv.inputValueNames[i]] = tensors.FromShape(shapes.Make(kv.kvDType, 1, kv.kvHeads, 0, kv.headDim))
	}

	cacheShape := shapes.Make(kv.kvDType, 1, kv.kvHeads, maxSeqLen, kv.headDim)

	// Check which inputs the decoder expects.
	decoderInputNames, _ := decoderModel.Inputs()
	decoderInputSet := make(map[string]bool, len(decoderInputNames))
	for _, name := range decoderInputNames {
		decoderInputSet[name] = true
	}
	hasPositionIDs := decoderInputSet["position_ids"]
	hasAttentionMask := decoderInputSet["attention_mask"]
	usesInputsEmbeds := decoderInputSet["inputs_embeds"]
	usesPerLayerInputs := decoderInputSet["per_layer_inputs"]

	return func(ctx *context.Context, tokens *Node, positions *Node, _ attention.KVCacheAccessor, aux *decode.AuxInputs) *Node {
		g := tokens.Graph()
		inputsEmbeds := aux.InputsEmbeds
		seqLen := inputsEmbeds.Shape().Dimensions[1]

		// Build decoder inputs.
		decoderInputs := make(map[string]*Node)
		if usesInputsEmbeds {
			decoderInputs["inputs_embeds"] = inputsEmbeds
		}
		if usesPerLayerInputs && aux.PerLayerInputs != nil {
			decoderInputs["per_layer_inputs"] = aux.PerLayerInputs
		}

		// Initialize KV cache variables.
		keyVars := make([]*context.Variable, kv.numLayers)
		valVars := make([]*context.Variable, kv.numLayers)
		keyCaches := make([]*Node, kv.numLayers)
		valCaches := make([]*Node, kv.numLayers)
		for i := range kv.numLayers {
			layerCtx := ctx.In(fmt.Sprintf("kv_layer_%d", i))
			keyVars[i], valVars[i] = attention.KVCacheGetVars(layerCtx, cacheShape)
			keyCaches[i] = keyVars[i].ValueGraph(g)
			valCaches[i] = valVars[i].ValueGraph(g)
		}

		posI64 := ConvertDType(positions, dtypes.Int64)
		if seqLen > 1 {
			// --- Prefill ---
			decoderModel.WithInputsAsConstants(emptyKV)
			decoderModel.WithPaddedKVCache(nil)

			if hasAttentionMask {
				decoderInputs["attention_mask"] = Ones(g, shapes.Make(dtypes.Int64, 1, seqLen))
			}
			if hasPositionIDs {
				posOffset := Reshape(posI64, 1, 1)
				decoderInputs["position_ids"] = Add(Iota(g, shapes.Make(dtypes.Int64, 1, seqLen), 1), posOffset)
			}
		} else {
			// --- Decode ---
			posI32 := Reshape(Slice(ConvertDType(positions, dtypes.Int32), AxisElem(0)))
			decoderModel.WithInputsAsConstants(nil)
			decoderModel.WithPaddedKVCache(posI32)
			for i := range kv.numLayers {
				decoderInputs[kv.inputKeyNames[i]] = keyCaches[i]
				decoderInputs[kv.inputValueNames[i]] = valCaches[i]
			}

			if hasAttentionMask {
				// With padded KV cache, present output has same shape as past
				// input (maxSeqLen), so attention_mask length = maxSeqLen.
				idx := Iota(g, shapes.Make(dtypes.Int64, 1, maxSeqLen), 1)
				posExpanded := Reshape(posI64, 1, 1)
				// Valid positions: 0..pos-1 (past tokens) and pos (current token).
				decoderInputs["attention_mask"] = Where(
					LessOrEqual(idx, posExpanded),
					OnesLike(idx), ZerosLike(idx))
			}
			if hasPositionIDs {
				decoderInputs["position_ids"] = Reshape(posI64, 1, 1)
			}
		}

		// Run decoder.
		allOutputs := decoderModel.CallGraph(ctx, g, decoderInputs)
		logits := allOutputs[kv.logitsIndex]

		// Update KV cache.
		zero := Const(g, int32(0))
		for i := range kv.numLayers {
			presentKey := allOutputs[kv.outputKeyIndices[i]]
			presentVal := allOutputs[kv.outputValueIndices[i]]

			if seqLen > 1 {
				// Prefill: present output is [batch, heads, prefillLen, dim].
				// Write it at position 0 in the padded cache buffer.
				keyCaches[i] = DynamicUpdateSlice(keyCaches[i], presentKey, []*Node{zero, zero, zero, zero})
				valCaches[i] = DynamicUpdateSlice(valCaches[i], presentVal, []*Node{zero, zero, zero, zero})
			} else {
				// Decode: with WithPaddedKVCache, the model's internal Concat
				// was replaced with DynamicUpdateSlice, so presentKey/Val
				// already has the new token written at the correct position.
				// The output shape equals the cache shape (maxSeqLen).
				keyCaches[i] = presentKey
				valCaches[i] = presentVal
			}

			keyVars[i].SetValueGraph(keyCaches[i])
			valVars[i].SetValueGraph(valCaches[i])
		}

		return logits
	}
}

// runVisionEncoder runs the vision encoder ONNX model on pre-processed pixel
// values and returns image features.
func runVisionEncoder(backend backends.Backend, ctx *context.Context, visionModel *onnx.Model, pixelValues *tensors.Tensor) (*tensors.Tensor, error) {
	exec, err := context.NewExec(backend, ctx.Reuse(),
		func(ctx *context.Context, pixels *Node) *Node {
			outputs := visionModel.CallGraph(ctx, pixels.Graph(), map[string]*Node{
				"pixel_values": pixels,
			})
			return outputs[0] // image_features: [1, numImageTokens, hiddenSize]
		},
	)
	if err != nil {
		return nil, fmt.Errorf("compile vision encoder: %w", err)
	}
	results, err := exec.Exec(pixelValues)
	if err != nil {
		return nil, fmt.Errorf("run vision encoder: %w", err)
	}
	return results[0], nil
}

// loadAndPreprocessImage loads an image from a file, resizes it to targetSize,
// normalizes with ImageNet mean/std, and returns a [1, 3, H, W] float32 tensor.
func loadAndPreprocessImage(path string, targetSize int) (*tensors.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}

	// Resize to targetSize x targetSize.
	resized := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Convert to [H, W, 3] float32 tensor with values in [0, 1].
	// Gemma 3n's SiglipImageProcessor uses rescale_factor=1/255 and
	// do_normalize=False, so pixel values should be in [0, 1] with no
	// mean/std normalization applied.
	hwcTensor := gomlxImages.ToTensor(dtypes.Float32).Single(resized)
	// hwcTensor shape: [targetSize, targetSize, 3]

	hwcData := hwcTensor.Value().([][][]float32)
	h := len(hwcData)
	w := len(hwcData[0])

	// Build channels-first [1, 3, H, W] tensor.
	result := make([][][][]float32, 1)
	result[0] = make([][][]float32, 3)
	for c := range 3 {
		result[0][c] = make([][]float32, h)
		for y := range h {
			result[0][c][y] = make([]float32, w)
			for x := range w {
				result[0][c][y][x] = hwcData[y][x][c]
			}
		}
	}

	return tensors.FromValue(result), nil
}

// formatChatPrompt wraps the user message in Gemma3's chat template.
// If hasImage is true, it includes the image placeholder.
func formatChatPrompt(userMessage string, hasImage bool) string {
	if hasImage {
		return fmt.Sprintf("<start_of_turn>user\n<image_soft_token>%s<end_of_turn>\n<start_of_turn>model\n", userMessage)
	}
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", userMessage)
}

// gemma3nSpecialTokens maps special token strings to their IDs.
// The HF tokenizer fallback doesn't handle these as atomic tokens,
// so we split them out before encoding regular text.
var gemma3nSpecialTokens = map[string]int32{
	"<start_of_turn>":    105,
	"<end_of_turn>":      106,
	"<image_soft_token>": 262145,
	"\n":                 107,
}

// tokenizePrompt encodes the prompt, prepending BOS. Special tokens in the
// prompt are split out and looked up directly rather than going through the
// tokenizer's subword pipeline. If the prompt contains an image token, it
// expands it to numImageTokens placeholder tokens.
func tokenizePrompt(tok api.Tokenizer, prompt string, hasImage bool, imageTokenID int32) []int32 {
	bosID, err := tok.SpecialTokenID(api.TokBeginningOfSentence)
	if err != nil {
		bosID = 2
	}

	// Split prompt on special token boundaries and encode each segment.
	tokens := make([]int32, 0, 64)
	tokens = append(tokens, int32(bosID))
	remaining := prompt
	for len(remaining) > 0 {
		// Find the earliest special token in remaining.
		bestPos := len(remaining)
		bestToken := ""
		for st := range gemma3nSpecialTokens {
			idx := strings.Index(remaining, st)
			if idx >= 0 && idx < bestPos {
				bestPos = idx
				bestToken = st
			}
		}

		// Encode text before the special token.
		if bestPos > 0 {
			segment := remaining[:bestPos]
			for _, t := range tok.Encode(segment) {
				tokens = append(tokens, int32(t))
			}
		}

		if bestToken == "" {
			break // No more special tokens.
		}

		// Insert the special token ID.
		stID := gemma3nSpecialTokens[bestToken]
		tokens = append(tokens, stID)
		// Expand image tokens.
		if hasImage && imageTokenID >= 0 && stID == imageTokenID {
			for range numImageTokens - 1 {
				tokens = append(tokens, imageTokenID)
			}
		}
		remaining = remaining[bestPos+len(bestToken):]
	}
	return tokens
}

// countImageTokens counts the number of image placeholder tokens in a sequence.
func countImageTokens(tokens []int32, imageTokenID int32) int {
	if imageTokenID < 0 {
		return 0
	}
	n := 0
	for _, t := range tokens {
		if t == imageTokenID {
			n++
		}
	}
	return n
}

// servingTokenizer wraps api.Tokenizer for the serving engine.
type servingTokenizer struct {
	tok         api.Tokenizer
	eosID       int
	endOfTurnID int
}

func (t *servingTokenizer) Decode(tokenID int32) (string, error) {
	return t.tok.Decode([]int{int(tokenID)}), nil
}

func (t *servingTokenizer) IsEOS(tokenID int32) bool {
	id := int(tokenID)
	return id == t.eosID || id == t.endOfTurnID
}

func (t *servingTokenizer) Reset() {}

// kvStructure holds the KV cache layout parsed from the ONNX model.
type kvStructure struct {
	numLayers          int
	inputKeyNames      []string
	inputValueNames    []string
	outputKeyIndices   []int
	outputValueIndices []int
	logitsIndex        int
	kvHeads            int
	headDim            int
	kvDType            dtypes.DType
}

func (kv *kvStructure) hasOutputs() bool {
	return kv.numLayers > 0 && len(kv.outputKeyIndices) == kv.numLayers
}

// parseKVStructure inspects the ONNX model's inputs and outputs to identify
// the KV cache layout.
func parseKVStructure(model *onnx.Model) *kvStructure {
	inputNames, inputShapes := model.Inputs()
	outputNames, _ := model.Outputs()

	kv := &kvStructure{logitsIndex: 0, kvDType: dtypes.Float32}

	layerKeys := make(map[int]string)
	layerValues := make(map[int]string)
	for i, name := range inputNames {
		var layerIdx int
		if n, _ := fmt.Sscanf(name, "past_key_values.%d.key", &layerIdx); n == 1 && name == fmt.Sprintf("past_key_values.%d.key", layerIdx) {
			layerKeys[layerIdx] = name
			dims := inputShapes[i].Dimensions
			kv.kvHeads = dims[1]
			kv.headDim = dims[3]
			kv.kvDType = inputShapes[i].DType
		}
		if n, _ := fmt.Sscanf(name, "past_key_values.%d.value", &layerIdx); n == 1 && name == fmt.Sprintf("past_key_values.%d.value", layerIdx) {
			layerValues[layerIdx] = name
		}
	}

	kv.numLayers = len(layerKeys)
	if kv.numLayers == 0 {
		return kv
	}

	kv.inputKeyNames = make([]string, kv.numLayers)
	kv.inputValueNames = make([]string, kv.numLayers)
	for i := range kv.numLayers {
		kv.inputKeyNames[i] = layerKeys[i]
		kv.inputValueNames[i] = layerValues[i]
	}

	kv.outputKeyIndices = make([]int, kv.numLayers)
	kv.outputValueIndices = make([]int, kv.numLayers)
	foundKeys := 0
	foundValues := 0
	for i, name := range outputNames {
		if name == "logits" {
			kv.logitsIndex = i
		}
		var layerIdx int
		if n, _ := fmt.Sscanf(name, "present.%d.key", &layerIdx); n == 1 && name == fmt.Sprintf("present.%d.key", layerIdx) && layerIdx < kv.numLayers {
			kv.outputKeyIndices[layerIdx] = i
			foundKeys++
		} else if n, _ := fmt.Sscanf(name, "present_key_values.%d.key", &layerIdx); n == 1 && name == fmt.Sprintf("present_key_values.%d.key", layerIdx) && layerIdx < kv.numLayers {
			kv.outputKeyIndices[layerIdx] = i
			foundKeys++
		}
		if n, _ := fmt.Sscanf(name, "present.%d.value", &layerIdx); n == 1 && name == fmt.Sprintf("present.%d.value", layerIdx) && layerIdx < kv.numLayers {
			kv.outputValueIndices[layerIdx] = i
			foundValues++
		} else if n, _ := fmt.Sscanf(name, "present_key_values.%d.value", &layerIdx); n == 1 && name == fmt.Sprintf("present_key_values.%d.value", layerIdx) && layerIdx < kv.numLayers {
			kv.outputValueIndices[layerIdx] = i
			foundValues++
		}
	}

	if foundKeys != kv.numLayers || foundValues != kv.numLayers {
		kv.outputKeyIndices = nil
		kv.outputValueIndices = nil
	}

	return kv
}

// printModelInfo prints the inputs and outputs of an ONNX model.
func printTopToken(label string, logitsTensor *tensors.Tensor, tok tokenizers.Tokenizer) {
	lv := logitsTensor.Value().([][][]float32)[0][0]
	mi, mv := 0, lv[0]
	for i, v := range lv {
		if v > mv {
			mv = v
			mi = i
		}
	}
	fmt.Printf("%-26s: %d (%q), logit: %.4f\n", label, mi, tok.Decode([]int{mi}), mv)
}

func printModelInfo(name string, model *onnx.Model) {
	inputNames, inputShapes := model.Inputs()
	outputNames, _ := model.Outputs()
	fmt.Printf("%s model inputs (%d):\n", name, len(inputNames))
	for i, n := range inputNames {
		fmt.Printf("  %s: %v\n", n, inputShapes[i])
	}
	fmt.Printf("%s model outputs: %v\n\n", name, outputNames)
}

// mustDownload downloads a file from the repo, fatally logging on error.
func mustDownload(repo *hub.Repo, file string) string {
	path, err := repo.DownloadFile(file)
	if err != nil {
		klog.Fatalf("Failed to download %s: %+v", file, err)
	}
	return path
}

// tryDownload attempts to download a file, ignoring errors (for optional data files).
func tryDownload(repo *hub.Repo, file string) {
	if _, err := repo.DownloadFile(file); err != nil {
		// Check if file exists in the repo; external data files may not exist.
		if !strings.Contains(err.Error(), "404") {
			klog.V(1).Infof("Optional download %s: %v", file, err)
		}
	}
}
