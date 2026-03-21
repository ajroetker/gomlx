// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// gemma3 demonstrates ONNX-based text generation using GoMLX's serving engine.
//
// It downloads the onnx-community/gemma-3-270m-it-ONNX model from HuggingFace,
// wraps it in a ModelFn, and uses the serving engine for autoregressive
// generation with KV cache management.
//
// Usage:
//
//	go run gemma3.go
//	go run gemma3.go --prompt="What is Go?"
//	go run gemma3.go --max-tokens=50
//	go run gemma3.go --prompts-file=prompts.txt --warmup=2
//	go run gemma3.go --compaction --compaction-ratio=2
//	go run gemma3.go --compaction --compaction-ratio=4 --max-seq-len=1024
package main

import (
	"bufio"
	stdctx "context"
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/ajroetker/huggingface-gomlx/kvcache"
	"github.com/ajroetker/huggingface-gomlx/serving"
	"github.com/gomlx/onnx-gomlx/onnx"
	onnxparser "github.com/gomlx/onnx-gomlx/onnx/parser"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
)

const (
	// HuggingFace repository for the Gemma3 270M ONNX model.
	modelRepo = "onnx-community/gemma-3-270m-it-ONNX"
)

var (
	flagPrompt      = flag.String("prompt", "Write a short poem about the sea.", "User message for the chat prompt.")
	flagPromptsFile = flag.String("prompts-file", "", "Path to text file with one prompt per line (overrides --prompt).")
	flagWarmup      = flag.Int("warmup", 1, "Number of warmup rounds before measurement (only with --prompts-file).")
	flagMaxTokens   = flag.Int("max-tokens", 100, "Maximum number of tokens to generate.")
	flagMaxSeqLen   = flag.Int("max-seq-len", 256, "Maximum total sequence length (prompt + generated tokens).")
	flagFP16             = flag.Bool("fp16", false, "Use fp16 (float16) model variant (570MB instead of 1.14GB).")
	flagBackend          = flag.String("backend", "", "Backend to use (default: auto-detect).")
	flagCPUProfile       = flag.String("cpuprofile", "", "Write CPU profile to file.")
	flagCompaction       = flag.Bool("compaction", false, "Enable KV cache compaction after prefill.")
	flagCompactionRatio  = flag.Int("compaction-ratio", 2, "Compaction ratio: cache is compressed to 1/ratio of prompt length.")
	flagNumRefQueries    = flag.Int("num-ref-queries", 64, "Number of reference queries for compaction scoring.")
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagCPUProfile != "" {
		f, err := os.Create(*flagCPUProfile)
		if err != nil {
			klog.Fatalf("Failed to create CPU profile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			klog.Fatalf("Failed to start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *flagBackend != "" {
		if err := os.Setenv("GOMLX_BACKEND", *flagBackend); err != nil {
			klog.Warningf("Failed to set backend: %v", err)
		}
	}

	// Determine ONNX model file path based on precision.
	modelFile := "onnx/model.onnx"
	if *flagFP16 {
		modelFile = "onnx/model_fp16.onnx"
	}

	// Download and cache model files from HuggingFace.
	// The ONNX model uses external data storage: model.onnx (graph) + model.onnx_data (weights).
	modelDataFile := modelFile + "_data"
	fmt.Printf("Downloading model: %s (%s)\n", modelRepo, modelFile)
	repo := hub.New(modelRepo).WithProgressBar(true)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to get repo info: %+v", err)
	}
	onnxPath, err := repo.DownloadFile(modelFile)
	if err != nil {
		klog.Fatalf("Failed to download %s: %+v", modelFile, err)
	}
	if _, err := repo.DownloadFile(modelDataFile); err != nil {
		klog.Fatalf("Failed to download %s: %+v", modelDataFile, err)
	}
	fmt.Printf("Model downloaded: %s\n\n", onnxPath)

	// Load tokenizer.
	tok, err := tokenizers.New(repo)
	if err != nil {
		klog.Fatalf("Failed to create tokenizer: %+v", err)
	}

	// Load ONNX model.
	model, err := onnxparser.ParseFile(onnxPath)
	if err != nil {
		klog.Fatalf("Failed to load ONNX model: %+v", err)
	}
	defer model.Close()

	inputNames, inputShapes := model.Inputs()
	outputNames, _ := model.Outputs()
	fmt.Printf("Model inputs (%d):\n", len(inputNames))
	for i, name := range inputNames {
		fmt.Printf("  %s: %v\n", name, inputShapes[i])
	}
	fmt.Printf("Model outputs: %v\n\n", outputNames)

	// Load model weights into context.
	ctx := context.New()
	if err := model.VariablesToContext(ctx); err != nil {
		klog.Fatalf("Failed to load model variables: %+v", err)
	}

	// Initialize backend.
	backend := backends.MustNew()
	fmt.Printf("Backend: %s\n\n", backend.Name())

	// Load prompts.
	prompts := loadPrompts()

	// Discover which inputs the model expects.
	inputSet := make(map[string]bool, len(inputNames))
	for _, name := range inputNames {
		inputSet[name] = true
	}
	hasPositionIDs := inputSet["position_ids"]
	hasAttentionMask := inputSet["attention_mask"]

	eosID, err := tok.SpecialTokenID(api.TokEndOfSentence)
	if err != nil {
		eosID = 1
	}
	// Look up <end_of_turn> token ID once to avoid per-token string decoding.
	endOfTurnID := -1
	if encoded := tok.Encode("<end_of_turn>"); len(encoded) == 1 {
		endOfTurnID = encoded[0]
	}

	maxSeqLen := *flagMaxSeqLen
	maxTokens := *flagMaxTokens

	// Parse KV cache structure from model inputs/outputs.
	kv := parseKVStructure(model)

	if !kv.hasOutputs() {
		klog.Fatalf("Model does not support KV cache outputs; the serving engine requires KV cache support.")
	}

	fmt.Printf("Using KV cache: %d layers, %d heads, dim=%d\n\n", kv.numLayers, kv.kvHeads, kv.headDim)

	// Pre-create KV cache variables on the original context.
	// The serving engine uses ctx.Reuse() for executor compilation, which
	// prevents creating new variables. We must create them here first.
	cacheShape := shapes.Make(kv.kvDType, 1, kv.kvHeads, maxSeqLen, kv.headDim)
	for i := range kv.numLayers {
		layerCtx := ctx.In(fmt.Sprintf("kv_layer_%d", i))
		attention.KVCacheGetVars(layerCtx, cacheShape)
	}

	// Wrap the ONNX model as a ModelFn for the serving engine.
	modelFn := makeModelFn(model, kv, maxSeqLen, hasAttentionMask, hasPositionIDs)

	// Create the serving engine.
	tokWrapper := &servingTokenizer{tok: tok, eosID: eosID, endOfTurnID: endOfTurnID}
	config := serving.Config{
		MaxSeqLen:    maxSeqLen,
		MaxBatchSize: 1, // ONNX model with explicit KV I/O supports batch=1
	}
	if *flagCompaction {
		// Compaction target length is determined per-request based on actual prompt
		// length divided by the ratio. We set a default here; the engine compacts
		// when position > TargetLen after prefill.
		targetLen := maxSeqLen / *flagCompactionRatio
		fmt.Printf("KV cache compaction enabled: ratio=%d, target=%d tokens, ref_queries=%d\n",
			*flagCompactionRatio, targetLen, *flagNumRefQueries)
		config.Compaction = &kvcache.CompactionConfig{
			TargetLen:     targetLen,
			NumRefQueries: *flagNumRefQueries,
		}
	}
	eng := serving.NewEngine(backend, ctx, modelFn, tokWrapper, config, kv.kvHeads, kv.headDim, kv.kvDType)
	defer eng.Stop()

	// Generation loop.
	benchmarkMode := *flagPromptsFile != ""
	warmupRounds := 0
	if benchmarkMode {
		warmupRounds = *flagWarmup
	}
	totalRounds := warmupRounds + 1

	for round := range totalRounds {
		isWarmup := round < warmupRounds
		if isWarmup {
			fmt.Printf("=== Warmup round %d/%d ===\n", round+1, warmupRounds)
		} else if benchmarkMode {
			fmt.Println("=== Measurement round ===")
		}

		var totalTokens int
		var totalDuration time.Duration

		for _, prompt := range prompts {
			promptTokens := tokenizePrompt(tok, formatChatPrompt(prompt))
			if len(promptTokens) >= maxSeqLen {
				fmt.Printf("Warning: prompt %q too long (%d tokens), skipping\n", prompt, len(promptTokens))
				continue
			}

			verbose := !isWarmup
			if verbose {
				fmt.Printf("Prompt: %q\n", prompt)
				fmt.Printf("Tokenized to %d tokens\n\n", len(promptTokens))
				fmt.Println("Generating...")
				fmt.Println("---")
			}

			startTime := time.Now()
			n := generateWithEngine(eng, promptTokens, maxTokens, verbose)
			dur := time.Since(startTime)

			if verbose {
				fmt.Println("\n---")
				if n > 0 {
					tokensPerSec := float64(n) / dur.Seconds()
					fmt.Printf("Generated %d tokens in %.2fs (%.1f tokens/s)\n\n", n, dur.Seconds(), tokensPerSec)
				}
			}

			totalTokens += n
			totalDuration += dur
		}

		if benchmarkMode && !isWarmup && totalTokens > 0 {
			fmt.Printf("\nAverage: %.1f tokens/s (%d tokens in %.2fs)\n",
				float64(totalTokens)/totalDuration.Seconds(), totalTokens, totalDuration.Seconds())
		}
	}
}

// generateWithEngine submits a prompt to the serving engine and streams output.
func generateWithEngine(eng *serving.Engine, promptTokens []int32, maxTokens int, verbose bool) int {
	outputCh, errCh, err := eng.Submit(
		stdctx.Background(),
		promptTokens,
		serving.RequestOptions{MaxNewTokens: maxTokens},
		nil,
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
		if verbose {
			fmt.Print(delta.Token)
		}
		tokensGenerated++
	}
	// Drain any remaining items so the engine can close outputCh and errCh.
	for range outputCh {
	}

	if err := <-errCh; err != nil {
		klog.Errorf("Generation error: %v", err)
	}

	return tokensGenerated
}

// servingTokenizer wraps api.Tokenizer to implement the serving.Tokenizer interface.
type servingTokenizer struct {
	tok         api.Tokenizer
	eosID       int
	endOfTurnID int // cached token ID for <end_of_turn>, -1 if unknown
}

func (t *servingTokenizer) Decode(tokenID int32) (string, error) {
	return t.tok.Decode([]int{int(tokenID)}), nil
}

func (t *servingTokenizer) IsEOS(tokenID int32) bool {
	id := int(tokenID)
	return id == t.eosID || id == t.endOfTurnID
}

func (t *servingTokenizer) Reset() {}

// makeModelFn wraps an ONNX model with explicit KV I/O into a ModelFn for
// the serving engine.
//
// The ONNX model expects past KV as inputs (past_key_values.{i}.key/value) and
// returns present KV as outputs (present.{i}.key/value). This wrapper bridges
// that to the ModelFn pattern by:
//   - Prefill (seqLen > 1): feeding empty KV constants, writing full present KV
//     to context variables
//   - Decode (seqLen == 1): feeding the full padded KV cache with a dynamic
//     attention mask, extracting the new token's KV and writing at position
//
// Positions are tensor parameters, enabling O(1) compiled executors for decode.
// The KVCacheAccessor parameter is unused since the ONNX model manages its own
// KV cache through explicit input/output tensors.
func makeModelFn(
	model *onnx.Model, kv *kvStructure,
	maxSeqLen int, hasAttentionMask, hasPositionIDs bool,
) decode.ModelFn {
	// Prepare empty KV constants for prefill (no past).
	emptyKV := make(map[string]any)
	for i := range kv.numLayers {
		emptyKV[kv.inputKeyNames[i]] = tensors.FromShape(shapes.Make(kv.kvDType, 1, kv.kvHeads, 0, kv.headDim))
		emptyKV[kv.inputValueNames[i]] = tensors.FromShape(shapes.Make(kv.kvDType, 1, kv.kvHeads, 0, kv.headDim))
	}

	// Fixed cache shape for context variables: [1, kvHeads, maxSeqLen, headDim].
	cacheShape := shapes.Make(kv.kvDType, 1, kv.kvHeads, maxSeqLen, kv.headDim)

	return func(ctx *context.Context, newTokens *Node, positions *Node, _ attention.KVCacheAccessor, aux *decode.AuxInputs) *Node {
		g := newTokens.Graph()
		seqLen := newTokens.Shape().Dimensions[1]

		// Convert input_ids to int64 (ONNX models typically expect int64).
		inputs := map[string]*Node{
			"input_ids": ConvertDType(newTokens, dtypes.Int64),
		}

		// Initialize cache variables for all layers.
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

		// positions carries the absolute sequence position (for RoPE/position_ids).
		// cacheWritePos carries the cache write position (for mask and KV writes).
		// After compaction, these differ; otherwise they are the same.
		cacheWritePos := positions
		if aux != nil && aux.CacheWritePositions != nil {
			cacheWritePos = aux.CacheWritePositions
		}

		posI64 := ConvertDType(positions, dtypes.Int64)               // [batchSize=1] — for RoPE
		cacheWritePosI64 := ConvertDType(cacheWritePos, dtypes.Int64) // [batchSize=1] — for mask/writes

		if seqLen > 1 {
			// --- Prefill path ---
			// Feed empty KV constants (no past).
			model.WithInputsAsConstants(emptyKV)

			if hasAttentionMask {
				inputs["attention_mask"] = Ones(g, shapes.Make(dtypes.Int64, 1, seqLen))
			}
			if hasPositionIDs {
				// Sequential positions from offset (normally 0 for fresh requests).
				posOffset := Reshape(posI64, 1, 1)
				inputs["position_ids"] = Add(Iota(g, shapes.Make(dtypes.Int64, 1, seqLen), 1), posOffset)
			}
		} else {
			// --- Decode path (seqLen == 1) ---
			// Feed the full padded KV cache. Invalid positions are masked out
			// by the attention mask, enabling O(1) compilation (shapes are fixed
			// regardless of position).
			model.WithInputsAsConstants(nil)
			for i := range kv.numLayers {
				inputs[kv.inputKeyNames[i]] = keyCaches[i]
				inputs[kv.inputValueNames[i]] = valCaches[i]
			}

			if hasAttentionMask {
				// Mask: [1, maxSeqLen + 1] — past positions [0, cacheWritePos) + current token.
				// The ONNX model sees past_seq_len=maxSeqLen (padded), so total
				// attention length is maxSeqLen + 1 (past + current).
				// After compaction, cacheWritePos < absPosition, so positions beyond
				// cacheWritePos (which were zeroed by compaction) are correctly masked out.
				totalLen := maxSeqLen + 1
				idx := Iota(g, shapes.Make(dtypes.Int64, 1, totalLen), 1)
				cwpExpanded := Reshape(cacheWritePosI64, 1, 1)
				pastValid := LessThan(idx, cwpExpanded)
				currentValid := Equal(idx, ConstAs(idx, int64(maxSeqLen)))
				validMask := Or(pastValid, currentValid)
				inputs["attention_mask"] = Where(validMask, OnesLike(idx), ZerosLike(idx))
			}
			if hasPositionIDs {
				inputs["position_ids"] = Reshape(posI64, 1, 1)
			}
		}

		// Run ONNX model.
		allOutputs := model.CallGraph(ctx, g, inputs)
		logits := allOutputs[kv.logitsIndex]

		// Store updated KV in context variables.
		zero := Const(g, int32(0))
		for i := range kv.numLayers {
			presentKey := allOutputs[kv.outputKeyIndices[i]]
			presentVal := allOutputs[kv.outputValueIndices[i]]

			if seqLen > 1 {
				// Prefill: write all prompt KV at the start of the cache.
				keyCaches[i] = DynamicUpdateSlice(keyCaches[i], presentKey, []*Node{zero, zero, zero, zero})
				valCaches[i] = DynamicUpdateSlice(valCaches[i], presentVal, []*Node{zero, zero, zero, zero})
			} else {
				// Decode: present output is [1, heads, maxSeqLen+1, dim].
				// New token's KV is at the last position (index maxSeqLen).
				newKey := Slice(presentKey, AxisRange(), AxisRange(), AxisRange(maxSeqLen, maxSeqLen+1), AxisRange())
				newVal := Slice(presentVal, AxisRange(), AxisRange(), AxisRange(maxSeqLen, maxSeqLen+1), AxisRange())

				// Use cacheWritePos for the KV cache write position.
				cwpI32 := Reshape(Slice(ConvertDType(cacheWritePos, dtypes.Int32), AxisElem(0)))
				keyCaches[i] = DynamicUpdateSlice(keyCaches[i], newKey, []*Node{zero, zero, cwpI32, zero})
				valCaches[i] = DynamicUpdateSlice(valCaches[i], newVal, []*Node{zero, zero, cwpI32, zero})
			}

			keyVars[i].SetValueGraph(keyCaches[i])
			valVars[i].SetValueGraph(valCaches[i])
		}

		return logits
	}
}

// loadPrompts returns prompts from either --prompts-file or --prompt.
func loadPrompts() []string {
	if *flagPromptsFile != "" {
		f, err := os.Open(*flagPromptsFile)
		if err != nil {
			klog.Fatalf("Failed to open prompts file: %v", err)
		}
		defer f.Close()

		var prompts []string
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line != "" {
				prompts = append(prompts, line)
			}
		}
		if err := scanner.Err(); err != nil {
			klog.Fatalf("Error reading prompts file: %v", err)
		}
		if len(prompts) == 0 {
			klog.Fatalf("No prompts found in file: %s", *flagPromptsFile)
		}
		fmt.Printf("Loaded %d prompts from %s\n", len(prompts), *flagPromptsFile)
		return prompts
	}
	return []string{*flagPrompt}
}

// formatChatPrompt wraps the user message in Gemma3's chat template.
func formatChatPrompt(userMessage string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", userMessage)
}

// tokenizePrompt encodes the prompt, prepending the BOS token.
func tokenizePrompt(tok api.Tokenizer, prompt string) []int32 {
	bosID, err := tok.SpecialTokenID(api.TokBeginningOfSentence)
	if err != nil {
		bosID = 2 // Gemma default BOS
	}
	encoded := tok.Encode(prompt)
	tokens := make([]int32, 0, len(encoded)+1)
	tokens = append(tokens, int32(bosID))
	for _, t := range encoded {
		tokens = append(tokens, int32(t))
	}
	return tokens
}

// kvStructure holds the KV cache layout parsed from the ONNX model.
type kvStructure struct {
	numLayers       int
	inputKeyNames   []string // past_key_values.{i}.key, ordered by layer
	inputValueNames []string // past_key_values.{i}.value, ordered by layer
	// outputKeyIndices are indices into the model's output list for the present key tensors.
	// The model returns present KV (past KV concatenated with the newly computed token's KV).
	outputKeyIndices []int
	// outputValueIndices are indices into the model's output list for the present value tensors.
	outputValueIndices []int
	logitsIndex        int          // index of logits in model.Outputs()
	kvHeads            int          // number of KV attention heads
	headDim            int          // head dimension
	kvDType            dtypes.DType // DType for KV tensors
}

// hasOutputs returns true if the model has KV cache outputs (present_key_values).
func (kv *kvStructure) hasOutputs() bool {
	return kv.numLayers > 0 && len(kv.outputKeyIndices) == kv.numLayers
}

// parseKVStructure inspects the ONNX model's inputs and outputs to identify
// the KV cache layout: layer count, input/output names, and shapes.
func parseKVStructure(model onnx.Model) *kvStructure {
	inputNames, inputShapes := model.Inputs()
	outputNames, _ := model.Outputs()

	kv := &kvStructure{logitsIndex: 0, kvDType: dtypes.Float32} // defaults

	// Find KV input names and shapes.
	// Note: fmt.Sscanf returns n=1 once the integer is parsed, even if the trailing
	// literal doesn't match. We reconstruct and compare to ensure an exact match.
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

	// Build ordered lists of KV input names.
	kv.inputKeyNames = make([]string, kv.numLayers)
	kv.inputValueNames = make([]string, kv.numLayers)
	for i := range kv.numLayers {
		kv.inputKeyNames[i] = layerKeys[i]
		kv.inputValueNames[i] = layerValues[i]
	}

	// Find KV output indices and logits index.
	kv.outputKeyIndices = make([]int, kv.numLayers)
	kv.outputValueIndices = make([]int, kv.numLayers)
	foundKeys := 0
	foundValues := 0
	for i, name := range outputNames {
		if name == "logits" {
			kv.logitsIndex = i
		}
		var layerIdx int
		// Try "present.{i}.key" pattern (HuggingFace Optimum), then "present_key_values.{i}.key".
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
		// Model has KV inputs but not matching outputs; can't use KV cache.
		kv.outputKeyIndices = nil
		kv.outputValueIndices = nil
	}

	return kv
}
