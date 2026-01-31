// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

func init() {
	// Register VJPs for fused ops so they can be used during training.
	VJPRegistration[NodeTypeFusedSoftmax] = vjpForSingleOutput(softmaxVJP)
	VJPRegistration[NodeTypeFusedGelu] = vjpForSingleOutput(geluVJP)
	VJPRegistration[NodeTypeFusedLayerNorm] = vjpForSingleOutput(layerNormVJP)
	VJPRegistration[NodeTypeFusedDense] = vjpForSingleOutput(denseVJP)
	VJPRegistration[NodeTypeFusedMultiHeadSDPA] = vjpForSingleOutput(multiHeadSDPAVJP)
	VJPRegistration[NodeTypeFusedQKVDense] = qkvDenseVJP
}

// softmaxVJP computes the VJP for fused Softmax.
//
// Given s = softmax(x, axis):
//
//	ds/dx · v = s * (v - ReduceAndKeep(v * s, axis))
func softmaxVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsFusedSoftmax)
	s := node // node is the softmax output
	vs := Mul(v, s)
	sumVS := ReduceAndKeep(vs, ReduceSum, params.axis)
	return []*Node{Mul(s, Sub(v, sumVS))}
}

// geluVJP computes the VJP for fused Gelu (exact mode).
//
// Given gelu(x) = x * Φ(x), where Φ(x) = 0.5 * (1 + erf(x / √2)):
//
//	dgelu/dx = Φ(x) + x * φ(x)
//
// where φ(x) = (1/√(2π)) * exp(-x²/2) is the standard normal PDF.
//
//	VJP = v * (Φ(x) + x * φ(x))
func geluVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsFusedGelu)
	x := params.x

	// Φ(x) = 0.5 * (1 + erf(x / √2))
	cdf := MulScalar(AddScalar(Erf(DivScalar(x, math.Sqrt2)), 1), 0.5)

	// φ(x) = (1/√(2π)) * exp(-x²/2)
	pdf := MulScalar(Exp(MulScalar(Mul(x, x), -0.5)), 1.0/math.Sqrt(2.0*math.Pi))

	// dgelu/dx = Φ(x) + x * φ(x)
	grad := Add(cdf, Mul(x, pdf))
	return []*Node{Mul(v, grad)}
}

// layerNormVJP computes the VJP for fused LayerNorm.
//
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Gradients:
//
//	dy/dx: via chain rule through normalization
//	dy/dgamma: sum(v * xhat) over batch dims
//	dy/dbeta: sum(v) over batch dims
//
// where xhat = (x - mean) / sqrt(var + eps).
func layerNormVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsFusedLayerNorm)
	x := params.x
	axes := params.axes
	epsilon := params.epsilon

	// Recompute forward pass intermediates.
	mean := ReduceAndKeep(x, ReduceMean, axes...)
	xCentered := Sub(x, mean)
	variance := ReduceAndKeep(Mul(xCentered, xCentered), ReduceMean, axes...)
	invStd := Rsqrt(AddScalar(variance, epsilon))
	xhat := Mul(xCentered, invStd)

	// Apply gamma scaling to upstream gradient if present.
	var vScaled *Node
	if params.gamma != nil {
		vScaled = Mul(v, params.gamma)
	} else {
		vScaled = v
	}

	// Gradient w.r.t. x:
	// dx = invStd * (vScaled - mean(vScaled) - xhat * mean(vScaled * xhat))
	meanVScaled := ReduceAndKeep(vScaled, ReduceMean, axes...)
	meanVScaledXhat := ReduceAndKeep(Mul(vScaled, xhat), ReduceMean, axes...)
	dx := Mul(invStd, Sub(Sub(vScaled, meanVScaled), Mul(xhat, meanVScaledXhat)))

	results := []*Node{dx}

	if params.gamma != nil {
		// dy/dgamma = sum(v * xhat) over non-normalizing (batch) dimensions.
		dgamma := reduceToBroadcastShape(Mul(v, xhat), params.gamma, axes, x)
		results = append(results, dgamma)
	}

	if params.beta != nil {
		// dy/dbeta = sum(v) over non-normalizing (batch) dimensions.
		dbeta := reduceToBroadcastShape(v, params.beta, axes, x)
		results = append(results, dbeta)
	}

	return results
}

// reduceToBroadcastShape reduces a gradient to match a parameter's shape.
// The parameter was broadcast from normAxes dimensions to x's full shape,
// so we sum over all non-normalizing (batch) dimensions.
func reduceToBroadcastShape(grad, param *Node, normAxes []int, x *Node) *Node {
	normSet := make(map[int]bool, len(normAxes))
	for _, a := range normAxes {
		normSet[a] = true
	}
	var batchAxes []int
	for i := 0; i < x.Rank(); i++ {
		if !normSet[i] {
			batchAxes = append(batchAxes, i)
		}
	}
	if len(batchAxes) > 0 {
		grad = ReduceSum(grad, batchAxes...)
	}
	return ReshapeWithShape(grad, param.Shape())
}

// denseVJP computes the VJP for fused Dense.
//
// Dense: y = activation(x @ W + bias)
// where x: [..., in_features], W: [in_features, out_features]
//
// Chain rule: backprop through activation first (if any), then through dense.
func denseVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsFusedDense)
	x := params.x
	weight := params.weight
	xRank := x.Rank()
	lastAxis := xRank - 1

	// If activation is present, backprop through it first.
	if params.activation != backends.ActivationNone {
		// Recompute pre-activation: z = x @ W + bias.
		z := DotGeneral(x, []int{lastAxis}, []int{}, weight, []int{0}, []int{})
		if params.bias != nil {
			z = Add(z, params.bias)
		}

		// Compute v * activation'(z).
		switch params.activation {
		case backends.ActivationRelu:
			zero := ScalarZero(z.Graph(), z.DType())
			v = Where(GreaterThan(z, zero), v, zero)
		case backends.ActivationGelu:
			cdf := MulScalar(AddScalar(Erf(DivScalar(z, math.Sqrt2)), 1), 0.5)
			pdf := MulScalar(Exp(MulScalar(Mul(z, z), -0.5)), 1.0/math.Sqrt(2.0*math.Pi))
			v = Mul(v, Add(cdf, Mul(z, pdf)))
		case backends.ActivationSilu:
			sig := Logistic(z)
			one := ScalarOne(z.Graph(), z.DType())
			v = Mul(v, Mul(sig, Add(one, Mul(z, Sub(one, sig)))))
		case backends.ActivationTanh:
			t := Tanh(z)
			one := ScalarOne(z.Graph(), z.DType())
			v = Mul(v, Sub(one, Mul(t, t)))
		default:
			Panicf("denseVJP: unsupported activation type %s", params.activation)
		}
	}

	// dx = v @ W^T
	dx := DotGeneral(v, []int{lastAxis}, []int{}, weight, []int{1}, []int{})

	// dW: contract batch dims of v and x, keep in from x and out from v.
	var batchDimsV, batchDimsX []int
	for i := 0; i < xRank-1; i++ {
		batchDimsV = append(batchDimsV, i)
		batchDimsX = append(batchDimsX, i)
	}
	dweight := DotGeneral(x, batchDimsX, []int{}, v, batchDimsV, []int{})

	results := []*Node{dx, dweight}

	if params.bias != nil {
		if len(batchDimsV) > 0 {
			dbias := ReduceSum(v, batchDimsV...)
			results = append(results, dbias)
		} else {
			results = append(results, v)
		}
	}

	return results
}

// multiHeadSDPAVJP computes the VJP for fused MultiHeadSDPA.
//
// Given output = softmax(Q @ K^T * scale + mask) @ V:
//
//	dV = S^T @ dOut              (for each head)
//	dS = dOut @ V^T
//	dLogits = S * (dS - rowsum(dS * S))   (softmax backward)
//	dQ = dLogits * scale @ K
//	dK = (dLogits * scale)^T @ Q
//
// where S = softmax(Q @ K^T * scale + mask).
//
// This decomposes the fused op into primitives for gradient computation.
func multiHeadSDPAVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsFusedMultiHeadSDPA)
	q := params.q
	k := params.k
	vInput := params.v

	// Recompute attention scores: S = softmax(Q @ K^T * scale + mask)
	// q: [batch, numHeads, seqLen, headDim], k: [batch, numKVHeads, kvLen, headDim]
	// scores: [batch, numHeads, seqLen, kvLen]

	// For GQA, we need to expand K/V heads to match Q heads.
	numHeads := params.numHeads
	numKVHeads := params.numKVHeads
	headsPerKV := numHeads / numKVHeads

	kExpanded := k
	vExpanded := vInput
	if headsPerKV > 1 {
		// Repeat K and V heads: [batch, numKVHeads, ...] -> [batch, numHeads, ...]
		// Reshape to insert repeat dim, then broadcast.
		kShape := k.Shape()
		batch := kShape.Dimensions[0]
		kvLen := kShape.Dimensions[2]
		headDim := kShape.Dimensions[3]

		kReshaped := Reshape(k, batch, numKVHeads, 1, kvLen, headDim)
		kBroadcast := BroadcastToDims(kReshaped, batch, numKVHeads, headsPerKV, kvLen, headDim)
		kExpanded = Reshape(kBroadcast, batch, numHeads, kvLen, headDim)

		vReshaped := Reshape(vInput, batch, numKVHeads, 1, kvLen, headDim)
		vBroadcast := BroadcastToDims(vReshaped, batch, numKVHeads, headsPerKV, kvLen, headDim)
		vExpanded = Reshape(vBroadcast, batch, numHeads, kvLen, headDim)
	}

	// scores = Q @ K^T: [batch, numHeads, seqLen, headDim] x [batch, numHeads, headDim, kvLen]
	// -> [batch, numHeads, seqLen, kvLen]
	// Use DotGeneral: batch axes = {0, 1}, contracting: q's axis 3 with k's axis 3
	scores := DotGeneral(q, []int{3}, []int{0, 1}, kExpanded, []int{3}, []int{0, 1})
	scores = MulScalar(scores, params.scale)

	if params.mask != nil {
		scores = Add(scores, params.mask)
	}
	if params.causal {
		// Apply causal mask: positions where col > row get -Inf.
		seqLen := q.Shape().Dimensions[2]
		kvLen := k.Shape().Dimensions[2]
		g := q.Graph()
		iotaShape := shapes.Make(dtypes.Int32, seqLen, kvLen)
		rowIdx := Iota(g, iotaShape, 0)
		colIdx := Iota(g, iotaShape, 1)
		negInf := Scalar(g, q.DType(), math.Inf(-1))
		causalMask := Where(GreaterThan(colIdx, rowIdx),
			negInf,
			ScalarZero(g, q.DType()))
		scores = Add(scores, causalMask)
	}

	// S = softmax(scores, axis=-1)
	S := FusedSoftmax(scores, -1)

	// dV = S^T @ dOut: [batch, numHeads, kvLen, seqLen] x [batch, numHeads, seqLen, headDim]
	// -> [batch, numHeads, kvLen, headDim]
	dV := DotGeneral(S, []int{2}, []int{0, 1}, v, []int{2}, []int{0, 1})

	// dS = dOut @ V^T: [batch, numHeads, seqLen, headDim] x [batch, numHeads, headDim, kvLen]
	// -> [batch, numHeads, seqLen, kvLen]
	dS := DotGeneral(v, []int{3}, []int{0, 1}, vExpanded, []int{3}, []int{0, 1})

	// Softmax backward: dLogits = S * (dS - rowsum(dS * S))
	dLogits := Mul(S, Sub(dS, ReduceAndKeep(Mul(dS, S), ReduceSum, 3)))

	// Scale gradients
	dLogitsScaled := MulScalar(dLogits, params.scale)

	// dQ = dLogitsScaled @ K: [batch, numHeads, seqLen, kvLen] x [batch, numHeads, kvLen, headDim]
	// -> [batch, numHeads, seqLen, headDim]
	dQ := DotGeneral(dLogitsScaled, []int{3}, []int{0, 1}, kExpanded, []int{2}, []int{0, 1})

	// dK = dLogitsScaled^T @ Q: [batch, numHeads, kvLen, seqLen] x [batch, numHeads, seqLen, headDim]
	// -> [batch, numHeads, kvLen, headDim]
	dK := DotGeneral(dLogitsScaled, []int{2}, []int{0, 1}, q, []int{2}, []int{0, 1})

	// For GQA, reduce dK and dV back to [batch, numKVHeads, kvLen, headDim]
	if headsPerKV > 1 {
		batch := k.Shape().Dimensions[0]
		kvLen := k.Shape().Dimensions[2]
		headDim := k.Shape().Dimensions[3]

		dKReshaped := Reshape(dK, batch, numKVHeads, headsPerKV, kvLen, headDim)
		dK = ReduceSum(dKReshaped, 2) // Sum over repeat dim
		dVReshaped := Reshape(dV, batch, numKVHeads, headsPerKV, kvLen, headDim)
		dV = ReduceSum(dVReshaped, 2)
	}

	results := []*Node{dQ, dK, dV}
	if params.mask != nil {
		// dMask = sum(dLogitsScaled) over batch and heads dims
		dMask := ReduceSum(dLogitsScaled, 0, 1)
		results = append(results, dMask)
	}

	return results
}

// qkvDenseVJP computes the VJP for fused QKVDense (multi-output).
//
// QKVDense: q = x @ wQ^T + biasQ, k = x @ wK^T + biasK, v = x @ wV^T + biasV
// where wQKV = [wQ; wK; wV] stacked, wQ is [qDim, inFeatures], etc.
//
// Gradients:
//
//	dx = dq @ wQ + dk @ wK + dv @ wV    (sum of all projections)
//	dwQ = dq^T @ x                       (-> [qDim, inFeatures])
//	dwK = dk^T @ x                       (-> [kvDim, inFeatures])
//	dwV = dv^T @ x                       (-> [kvDim, inFeatures])
//	dwQKV = concat([dwQ, dwK, dwV], axis=0)
//	dbiasQ = sum(dq, batch dims)
//	dbiasK = sum(dk, batch dims)
//	dbiasV = sum(dv, batch dims)
func qkvDenseVJP(node *Node, vjps []*Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsFusedQKVDense)
	x := params.x
	wQKV := params.wQKV
	xRank := x.Rank()
	lastAxis := xRank - 1

	dq := vjps[0]
	dk := vjps[1]
	dv := vjps[2]

	qDim := params.qDim
	kvDim := params.kvDim
	inFeatures := x.Shape().Dimensions[lastAxis]

	// wQKV is [qDim+2*kvDim, inFeatures]
	// Extract wQ, wK, wV slices using backendSlice.
	wQ := backendSlice(wQKV, []int{0, 0}, []int{qDim, inFeatures}, nil)
	wK := backendSlice(wQKV, []int{qDim, 0}, []int{qDim + kvDim, inFeatures}, nil)
	wV := backendSlice(wQKV, []int{qDim + kvDim, 0}, []int{qDim + 2*kvDim, inFeatures}, nil)

	// dx = dq @ wQ + dk @ wK + dv @ wV
	// dq: [..., qDim], wQ: [qDim, inFeatures] -> contract on qDim -> [..., inFeatures]
	dx := DotGeneral(dq, []int{lastAxis}, []int{}, wQ, []int{0}, []int{})
	dx = Add(dx, DotGeneral(dk, []int{lastAxis}, []int{}, wK, []int{0}, []int{}))
	dx = Add(dx, DotGeneral(dv, []int{lastAxis}, []int{}, wV, []int{0}, []int{}))

	// dwQKV: concatenate dwQ, dwK, dwV
	var batchDims []int
	for i := 0; i < lastAxis; i++ {
		batchDims = append(batchDims, i)
	}
	// dwQ = x^T @ dq: contract batch dims -> [inFeatures, qDim], then transpose to [qDim, inFeatures]
	dwQ := DotGeneral(dq, batchDims, []int{}, x, batchDims, []int{})
	dwK := DotGeneral(dk, batchDims, []int{}, x, batchDims, []int{})
	dwV := DotGeneral(dv, batchDims, []int{}, x, batchDims, []int{})
	dwQKV := backendConcatenate(0, dwQ, dwK, dwV)

	results := []*Node{dx, dwQKV}

	if params.biasQ != nil {
		var dbiasQ *Node
		if len(batchDims) > 0 {
			dbiasQ = ReduceSum(dq, batchDims...)
		} else {
			dbiasQ = dq
		}
		results = append(results, dbiasQ)
	}
	if params.biasK != nil {
		var dbiasK *Node
		if len(batchDims) > 0 {
			dbiasK = ReduceSum(dk, batchDims...)
		} else {
			dbiasK = dk
		}
		results = append(results, dbiasK)
	}
	if params.biasV != nil {
		var dbiasV *Node
		if len(batchDims) > 0 {
			dbiasV = ReduceSum(dv, batchDims...)
		} else {
			dbiasV = dv
		}
		results = append(results, dbiasV)
	}

	return results
}
