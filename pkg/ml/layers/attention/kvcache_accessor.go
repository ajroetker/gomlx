// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
)

// FlatKVCacheAccessor implements KVCacheAccessor using a flat (dense) circular
// KV cache. Each batch element has an independent position, supporting
// continuous batching.
//
// It also implements BiasProvider: when a bias variable exists in the context,
// Bias() returns it as a graph node for additive attention logit bias.
type FlatKVCacheAccessor struct {
	// CacheShape is [batchSize, numKVHeads, maxSeqLen, headDim].
	CacheShape shapes.Shape

	// Positions is a [batchSize] int32 tensor with per-element absolute positions.
	Positions *Node

	// newSeqLen is set by WriteRead to the number of tokens just written.
	// Used by Mask to include the newly written positions.
	newSeqLen int

	// biasNode is lazily populated by Bias() on first call.
	biasNode *Node
	biasCtx  *context.Context
}

// WriteRead implements KVCacheAccessor.
func (a *FlatKVCacheAccessor) WriteRead(ctx *context.Context, g *Graph, newKey, newValue *Node) (cachedKeys, cachedValues *Node) {
	a.biasCtx = ctx
	a.newSeqLen = newKey.Shape().Dimensions[2]
	BatchedKVCacheUpdate(ctx, g, a.CacheShape, a.Positions, newKey, newValue)
	return GetKVCache(ctx, g, a.CacheShape)
}

// Mask implements KVCacheAccessor.
// The mask includes all positions written during WriteRead: positions[b]..positions[b]+newSeqLen-1.
func (a *FlatKVCacheAccessor) Mask(g *Graph, querySeqLen int) *Node {
	// Positions is the starting write position. After WriteRead, valid cache entries
	// span 0..positions+newSeqLen-1, so the mask boundary is positions+newSeqLen.
	maskPositions := AddScalar(a.Positions, int32(a.newSeqLen))
	return CreateBatchedKVCacheAttentionMask(g, a.CacheShape, maskPositions, querySeqLen, a.CacheShape.Dimensions[2])
}

// KeySeqLen implements KVCacheAccessor.
func (a *FlatKVCacheAccessor) KeySeqLen() int {
	return a.CacheShape.Dimensions[2]
}

// Bias implements BiasProvider.
// Returns the bias variable as a graph node if it exists, or nil otherwise.
// Shape: [batchSize, numKVHeads, maxSeqLen].
func (a *FlatKVCacheAccessor) Bias(g *Graph) *Node {
	if a.biasCtx == nil {
		return nil
	}
	biasShape := shapes.Make(a.CacheShape.DType, a.CacheShape.Dimensions[0], a.CacheShape.Dimensions[1], a.CacheShape.Dimensions[2])
	biasCtx := a.biasCtx.In(KVCacheScopeName).Reuse().Checked(false)

	// Check if the bias variable exists by looking for it with zero initializer.
	biasCtx = biasCtx.WithInitializer(initializers.Zero)
	biasVar := biasCtx.VariableWithShape(kvCacheBiasName, biasShape)
	return biasVar.ValueGraph(g)
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

