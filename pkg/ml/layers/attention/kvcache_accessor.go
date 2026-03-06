// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// KVCacheAccessor abstracts KV cache storage for model functions.
// It is constructed by the serving engine and passed to the model function
// each step, allowing the engine to own the KV cache layout (flat or paged)
// without the model needing to know the details.
//
// The context passed to WriteRead should be scoped to the current layer
// (e.g., ctx.In("layer_0").In("attention")). The accessor uses
// ctx.In("kv_cache") internally for variable storage.
type KVCacheAccessor interface {
	// WriteRead stores new key/value projections and returns the full cached
	// keys and values for attention computation.
	//
	// newKey, newValue shape: [batchSize, numKVHeads, newSeqLen, headDim]
	// Returns: [batchSize, numKVHeads, cacheSeqLen, headDim]
	WriteRead(ctx *context.Context, g *Graph, newKey, newValue *Node) (cachedKeys, cachedValues *Node)

	// Mask returns a boolean attention mask for unfilled cache positions.
	// True = attend, False = mask out.
	// Shape: [batchSize, 1, querySeqLen, keySeqLen]
	Mask(g *Graph, querySeqLen int) *Node

	// KeySeqLen returns the key sequence length used for attention computation.
	// For flat caches, this is maxSeqLen. For paged caches, this is numBlocks * blockSize.
	KeySeqLen() int
}

// FlatKVCacheAccessor implements KVCacheAccessor using a flat (dense) circular
// KV cache. Each batch element has an independent position, supporting
// continuous batching.
type FlatKVCacheAccessor struct {
	// CacheShape is [batchSize, numKVHeads, maxSeqLen, headDim].
	CacheShape shapes.Shape

	// Positions is a [batchSize] int32 tensor with per-element absolute positions.
	Positions *Node
}

// WriteRead implements KVCacheAccessor.
func (a *FlatKVCacheAccessor) WriteRead(ctx *context.Context, g *Graph, newKey, newValue *Node) (cachedKeys, cachedValues *Node) {
	BatchedKVCacheUpdate(ctx, g, a.CacheShape, a.Positions, newKey, newValue)
	return getKVCache(ctx, g, a.CacheShape)
}

// Mask implements KVCacheAccessor.
func (a *FlatKVCacheAccessor) Mask(g *Graph, querySeqLen int) *Node {
	return createBatchedKVCacheAttentionMask(g, a.CacheShape, a.Positions, querySeqLen, a.CacheShape.Dimensions[2])
}

// KeySeqLen implements KVCacheAccessor.
func (a *FlatKVCacheAccessor) KeySeqLen() int {
	return a.CacheShape.Dimensions[2]
}

// NewFlatKVCacheAccessor creates a FlatKVCacheAccessor.
//
// Parameters:
//   - batchSize: number of requests in the batch
//   - numKVHeads: number of key/value attention heads
//   - maxSeqLen: maximum sequence length (cache capacity)
//   - headDim: dimension of each attention head
//   - dtype: data type for cached entries
//   - positions: [batchSize] int32 tensor with per-element positions
func NewFlatKVCacheAccessor(batchSize, numKVHeads, maxSeqLen, headDim int, dtype dtypes.DType, positions *Node) *FlatKVCacheAccessor {
	return &FlatKVCacheAccessor{
		CacheShape: shapes.Make(dtype, batchSize, numKVHeads, maxSeqLen, headDim),
		Positions:  positions,
	}
}

// PagedKVCacheAccessor implements KVCacheAccessor using a paged (block-based)
// KV cache. Physical blocks are allocated by a BlockManager and mapped to
// logical positions via page tables.
type PagedKVCacheAccessor struct {
	// Config is the paged cache configuration.
	Config PagedKVCacheConfig

	// PageTables is a [batchSize, maxBlocksPerRequest] int32 tensor mapping
	// logical block indices to physical block indices.
	PageTables *Node

	// Positions is a [batchSize] int32 tensor with per-element positions.
	Positions *Node

	// ReadNumBlocks is the number of blocks to read per request (compile-time constant).
	ReadNumBlocks int
}

// WriteRead implements KVCacheAccessor.
func (a *PagedKVCacheAccessor) WriteRead(ctx *context.Context, g *Graph, newKey, newValue *Node) (cachedKeys, cachedValues *Node) {
	PagedKVCacheWriteBatched(ctx, g, a.Config, a.PageTables, a.Positions, newKey, newValue)

	batchSize := newKey.Shape().Dimensions[0]
	seqLen := a.ReadNumBlocks * a.Config.BlockSize

	// Read each batch element's blocks and stack.
	allKeys := make([]*Node, batchSize)
	allValues := make([]*Node, batchSize)
	for b := range batchSize {
		batchPT := Squeeze(Slice(a.PageTables, AxisElem(b), AxisRange()), 0)
		k, v := PagedKVCacheRead(ctx, g, a.Config, batchPT, a.ReadNumBlocks)
		allKeys[b] = k   // [1, numKVHeads, seqLen, headDim]
		allValues[b] = v  // [1, numKVHeads, seqLen, headDim]
	}

	cachedKeys = Concatenate(allKeys, 0)   // [batchSize, numKVHeads, seqLen, headDim]
	cachedValues = Concatenate(allValues, 0)

	_ = seqLen
	return
}

// Mask implements KVCacheAccessor.
func (a *PagedKVCacheAccessor) Mask(g *Graph, querySeqLen int) *Node {
	keySeqLen := a.KeySeqLen()
	batchSize := a.PageTables.Shape().Dimensions[0]

	posI32 := ConvertDType(a.Positions, dtypes.Int32)

	// effectivePositions = min(positions, keySeqLen)
	effectivePositions := MinScalar(posI32, keySeqLen)

	// Key indices: [keySeqLen]
	keyPositions := Iota(g, shapes.Make(dtypes.Int32, keySeqLen), 0)

	// Compare: [batchSize, keySeqLen]
	effectivePositions = ExpandDims(effectivePositions, -1) // [batchSize, 1]
	keyPositions = ExpandDims(keyPositions, 0)               // [1, keySeqLen]
	mask := LessThan(keyPositions, effectivePositions)        // [batchSize, keySeqLen]

	// Reshape to [batchSize, 1, querySeqLen, keySeqLen]
	mask = ExpandDims(mask, 1)
	mask = ExpandDims(mask, 2)
	mask = BroadcastToShape(mask, shapes.Make(dtypes.Bool, batchSize, 1, querySeqLen, keySeqLen))
	return mask
}

// KeySeqLen implements KVCacheAccessor.
func (a *PagedKVCacheAccessor) KeySeqLen() int {
	return a.ReadNumBlocks * a.Config.BlockSize
}
