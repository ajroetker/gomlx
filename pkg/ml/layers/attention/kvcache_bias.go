// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"fmt"
	"strings"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
)

const (
	kvCacheBiasName       = "bias"
	kvCacheRefQueriesName = "ref_queries"
)

// KVCacheGetBiasVar returns the bias variable for the KV cache at the given context scope.
// The variable is created on first access with zero-initialization.
//
// Parameters:
//   - ctx: Context for storing/retrieving the bias variable (should be the same
//     scope used for KVCacheGetVars)
//   - biasShape: Shape [batchSize, numKVHeads, maxSeqLen]
func KVCacheGetBiasVar(ctx *context.Context, biasShape shapes.Shape) *context.Variable {
	ctx = ctx.In(KVCacheScopeName).WithInitializer(initializers.Zero)
	return ctx.VariableWithShape(kvCacheBiasName, biasShape)
}

// KVCacheResetBias clears all KV cache bias variables under the given context scope.
func KVCacheResetBias(ctx *context.Context) {
	biasSuffix := fmt.Sprintf("%s%s%s%s", context.ScopeSeparator, KVCacheScopeName, context.ScopeSeparator, kvCacheBiasName)

	for v := range ctx.IterVariablesInScope() {
		if strings.HasSuffix(v.ScopeAndName(), biasSuffix) {
			v.SetValue(tensors.FromShape(v.Shape()))
		}
	}
}

// KVCacheGetRefQueriesVar returns the reference queries variable for compaction.
// The variable stores the last N projected queries (post-RoPE) captured during
// prefill, shaped [numKVHeads, numRefQueries, headDim].
// Returns nil if the variable doesn't exist at this scope.
func KVCacheGetRefQueriesVar(ctx *context.Context, refShape shapes.Shape) *context.Variable {
	ctx = ctx.In(KVCacheScopeName).WithInitializer(initializers.Zero)
	return ctx.VariableWithShape(kvCacheRefQueriesName, refShape)
}

// KVCacheResetRefQueries clears all reference query variables under the given context scope.
func KVCacheResetRefQueries(ctx *context.Context) {
	refSuffix := fmt.Sprintf("%s%s%s%s", context.ScopeSeparator, KVCacheScopeName, context.ScopeSeparator, kvCacheRefQueriesName)

	for v := range ctx.IterVariablesInScope() {
		if strings.HasSuffix(v.ScopeAndName(), refSuffix) {
			v.SetValue(tensors.FromShape(v.Shape()))
		}
	}
}

// BiasProvider is an optional interface that KVCacheAccessor implementations
// may implement to provide per-key additive attention logit biases.
// These biases are used by KV cache compaction to preserve attention mass
// for dropped keys.
type BiasProvider interface {
	// Bias returns additive attention logit biases as a graph node, or nil
	// if no bias is active. Shape: [batchSize, numKVHeads, keySeqLen].
	Bias(g *Graph) *Node
}
