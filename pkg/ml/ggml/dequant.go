// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package ggml provides graph-level decomposed dequantization for GGML block formats.
//
// These functions express GGML dequantization (Q4_0, Q8_0, IQ4_NL, Q4_K, Q6_K) as standard graph
// primitives (Bitcast, BitwiseAnd, ShiftRightLogical, Slice, ConvertDType, etc.) so that
// any backend — including XLA for GPU — can execute them. The simplego backend uses faster
// fused SIMD implementations when available; these decomposed versions serve as the
// automatic fallback via InternalFusedOpCaller.
package ggml

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// CanDecompose returns true for GGML quantization types that have a graph-level
// decomposed implementation (Q4_0, Q8_0, IQ4_NL, Q4_K, Q6_K).
func CanDecompose(t backends.GGMLQuantType) bool {
	switch t {
	case backends.GGMLQ4_0, backends.GGMLQ8_0, backends.GGMLIQ4NL, backends.GGMLQ4_K, backends.GGMLQ6_K:
		return true
	default:
		return false
	}
}

// Dequant dequantizes GGML block-format weights to [N, K] Float32 using graph primitives.
//
// weights must be [N, bytesPerRow] Uint8 in native GGML block layout.
// N is the number of output rows (output features for dense, vocab size for gather).
func Dequant(weights *Node, ggmlType backends.GGMLQuantType, N int) *Node {
	bytesPerRow := weights.Shape().Dimensions[1]
	bpb := ggmlType.BytesPerBlock()
	numBlocks := bytesPerRow / bpb

	switch ggmlType {
	case backends.GGMLQ8_0:
		return dequantQ8_0(weights, N, numBlocks)
	case backends.GGMLQ4_0:
		return dequantQ4_0(weights, N, numBlocks)
	case backends.GGMLIQ4NL:
		return dequantIQ4NL(weights, N, numBlocks)
	case backends.GGMLQ6_K:
		return dequantQ6_K(weights, N, numBlocks)
	case backends.GGMLQ4_K:
		return dequantQ4_K(weights, N, numBlocks)
	default:
		panic("ggml.Dequant: unsupported type " + ggmlType.String())
	}
}

// dequantQ8_0 dequantizes Q8_0 blocks: 2-byte fp16 scale + 32 int8 quants per block.
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantQ8_0(weights *Node, N, numBlocks int) *Node {
	// Reshape to [N, numBlocks, 34] (34 bytes per Q8_0 block).
	w := Reshape(weights, N, numBlocks, 34)

	// Extract scale: first 2 bytes → fp16 → float32 → [N, numBlocks, 1].
	scaleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2))
	scale := Bitcast(scaleBytes, dtypes.Float16)
	scale = ConvertDType(scale, dtypes.Float32)
	scale = Reshape(scale, N, numBlocks, 1) // ensure rank-3 for broadcast

	// Extract quants: bytes 2:34 → int8 → float32 → [N, numBlocks, 32].
	quantBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(2, 34))
	quants := Bitcast(quantBytes, dtypes.Int8)
	quants = ConvertDType(quants, dtypes.Float32)

	// Dequantize: output = scale * quants
	result := Mul(quants, scale)

	K := numBlocks * 32
	return Reshape(result, N, K)
}

// extractNibbleBlock extracts the fp16 scale and 32 combined nibble values from
// an 18-byte block layout (shared by Q4_0 and IQ4_NL).
// Returns scale [N, numBlocks, 1] Float32 and combined [N, numBlocks, 32] Uint8.
func extractNibbleBlock(weights *Node, N, numBlocks int) (scale, combined *Node) {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 18)

	scaleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2))
	scale = Bitcast(scaleBytes, dtypes.Float16)
	scale = ConvertDType(scale, dtypes.Float32)
	scale = Reshape(scale, N, numBlocks, 1)

	nibbleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(2, 18))
	mask := Scalar(g, dtypes.Uint8, uint8(0x0F))
	lo := BitwiseAnd(nibbleBytes, mask)
	hi := BitwiseShiftRightLogicalScalar(nibbleBytes, uint8(4))
	combined = Concatenate([]*Node{lo, hi}, 2)
	return scale, combined
}

// dequantQ4_0 dequantizes Q4_0 blocks: 2-byte fp16 scale + 16 packed nibble bytes per block.
// Each byte holds two 4-bit values: low nibble → first 16 values, high nibble → last 16.
// Dequant: output[i] = scale * (nibble - 8).
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantQ4_0(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	scale, combined := extractNibbleBlock(weights, N, numBlocks)

	combinedF := ConvertDType(combined, dtypes.Float32)
	eight := Scalar(g, dtypes.Float32, float32(8.0))
	combinedF = Sub(combinedF, eight)

	result := Mul(combinedF, scale)
	K := numBlocks * 32
	return Reshape(result, N, K)
}

// dequantIQ4NL dequantizes IQ4_NL blocks: same layout as Q4_0 (2-byte fp16 scale + 16 packed
// nibble bytes), but nibble values are indices into a non-linear lookup table instead of
// linear (nibble - 8).
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantIQ4NL(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	scale, combined := extractNibbleBlock(weights, N, numBlocks)

	indices := ConvertDType(combined, dtypes.Int32)
	totalElements := N * numBlocks * 32
	indicesFlat := Reshape(indices, totalElements, 1)

	lut := Const(g, backends.IQ4NLLookupTable[:])
	looked := Gather(lut, indicesFlat)
	looked = Reshape(looked, N, numBlocks, 32)

	result := Mul(looked, scale)
	K := numBlocks * 32
	return Reshape(result, N, K)
}

// dequantQ6_K dequantizes Q6_K blocks: 210 bytes per block, 256 values per block.
// Block layout: ql[128] + qh[64] + sc[16] + d[2].
// Each value is a 6-bit quant: low 4 bits from ql (nibble-packed), high 2 bits from qh (bit-pair packed).
// output[i] = d * int8(sc[i/16]) * (q6[i] - 32).
// Returns [N, K] Float32 where K = numBlocks * 256.
func dequantQ6_K(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 210)

	// Extract block components.
	ql := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 128))   // [N, numBlocks, 128]
	qh := Slice(w, AxisRange(), AxisRange(), AxisRange(128, 192)) // [N, numBlocks, 64]
	sc := Slice(w, AxisRange(), AxisRange(), AxisRange(192, 208)) // [N, numBlocks, 16]
	dBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(208, 210))
	d := ConvertDType(Bitcast(dBytes, dtypes.Float16), dtypes.Float32)
	d = Reshape(d, N, numBlocks, 1, 1) // for broadcasting over [16, 16]

	// Convert sc (uint8 representing int8) to float32.
	scFloat := ConvertDType(Bitcast(sc, dtypes.Int8), dtypes.Float32) // [N, numBlocks, 16]
	scFloat = Reshape(scFloat, N, numBlocks, 16, 1)                  // broadcast over 16 values per sub-block

	// Split ql into 4 chunks of 32 bytes.
	qlA := Slice(ql, AxisRange(), AxisRange(), AxisRange(0, 32))
	qlB := Slice(ql, AxisRange(), AxisRange(), AxisRange(32, 64))
	qlC := Slice(ql, AxisRange(), AxisRange(), AxisRange(64, 96))
	qlD := Slice(ql, AxisRange(), AxisRange(), AxisRange(96, 128))

	mask4 := Scalar(g, dtypes.Uint8, uint8(0x0F))

	// Extract low and high nibbles from each ql chunk.
	qlALo := BitwiseAnd(qlA, mask4)
	qlBLo := BitwiseAnd(qlB, mask4)
	qlCLo := BitwiseAnd(qlC, mask4)
	qlDLo := BitwiseAnd(qlD, mask4)
	qlAHi := BitwiseShiftRightLogicalScalar(qlA, uint8(4))
	qlBHi := BitwiseShiftRightLogicalScalar(qlB, uint8(4))
	qlCHi := BitwiseShiftRightLogicalScalar(qlC, uint8(4))
	qlDHi := BitwiseShiftRightLogicalScalar(qlD, uint8(4))

	// Assemble low 4 bits in output order (256 values):
	// First half (0..127): groups from ql[0:64] with qh[0:32]
	//   values 0..31:   qlA low nibbles
	//   values 32..63:  qlB low nibbles
	//   values 64..95:  qlA high nibbles
	//   values 96..127: qlB high nibbles
	// Second half (128..255): groups from ql[64:128] with qh[32:64]
	//   values 128..159: qlC low nibbles
	//   values 160..191: qlD low nibbles
	//   values 192..223: qlC high nibbles
	//   values 224..255: qlD high nibbles
	low4 := Concatenate([]*Node{
		qlALo, qlBLo, qlAHi, qlBHi,
		qlCLo, qlDLo, qlCHi, qlDHi,
	}, 2) // [N, numBlocks, 256]

	// Split qh into two halves of 32 bytes.
	qh0 := Slice(qh, AxisRange(), AxisRange(), AxisRange(0, 32))
	qh1 := Slice(qh, AxisRange(), AxisRange(), AxisRange(32, 64))

	mask2 := Scalar(g, dtypes.Uint8, uint8(3))

	// Extract 2-bit groups from each qh half.
	// qh byte packs 4 groups of 2 bits for 4 groups of 32 values.
	qh0G0 := BitwiseAnd(qh0, mask2)
	qh0G1 := BitwiseAnd(BitwiseShiftRightLogicalScalar(qh0, uint8(2)), mask2)
	qh0G2 := BitwiseAnd(BitwiseShiftRightLogicalScalar(qh0, uint8(4)), mask2)
	qh0G3 := BitwiseShiftRightLogicalScalar(qh0, uint8(6))

	qh1G0 := BitwiseAnd(qh1, mask2)
	qh1G1 := BitwiseAnd(BitwiseShiftRightLogicalScalar(qh1, uint8(2)), mask2)
	qh1G2 := BitwiseAnd(BitwiseShiftRightLogicalScalar(qh1, uint8(4)), mask2)
	qh1G3 := BitwiseShiftRightLogicalScalar(qh1, uint8(6))

	high2 := Concatenate([]*Node{
		qh0G0, qh0G1, qh0G2, qh0G3,
		qh1G0, qh1G1, qh1G2, qh1G3,
	}, 2) // [N, numBlocks, 256]

	// Combine: q6 = low4 | (high2 << 4), then subtract 32 and convert to float.
	q6 := BitwiseOr(low4, BitwiseShiftLeftScalar(high2, uint8(4)))
	q6Int := ConvertDType(q6, dtypes.Int32)
	q6Centered := Sub(q6Int, Scalar(g, dtypes.Int32, int32(32)))
	q6Float := ConvertDType(q6Centered, dtypes.Float32)
	q6Float = Reshape(q6Float, N, numBlocks, 16, 16)

	// result = d * sc * (q6 - 32)
	result := Mul(Mul(d, scFloat), q6Float) // [N, numBlocks, 16, 16]
	K := numBlocks * 256
	return Reshape(result, N, K)
}

// dequantQ4_K dequantizes Q4_K blocks: 144 bytes per block, 256 values per block.
// Block layout: d[2] + dmin[2] + scales[12] + qs[128].
// 8 sub-blocks of 32 values, each with a 6-bit scale and min packed in 12 bytes.
// output[i] = d * sc[i/32] * q4[i] - dmin * mn[i/32].
// Returns [N, K] Float32 where K = numBlocks * 256.
func dequantQ4_K(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 144)

	// Extract block components.
	dBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2))
	dminBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(2, 4))
	scalesRaw := Slice(w, AxisRange(), AxisRange(), AxisRange(4, 16)) // [N, numBlocks, 12]
	qs := Slice(w, AxisRange(), AxisRange(), AxisRange(16, 144))      // [N, numBlocks, 128]

	d := ConvertDType(Bitcast(dBytes, dtypes.Float16), dtypes.Float32)
	d = Reshape(d, N, numBlocks, 1, 1)
	dmin := ConvertDType(Bitcast(dminBytes, dtypes.Float16), dtypes.Float32)
	dmin = Reshape(dmin, N, numBlocks, 1, 1)

	// Unpack 8 (scale, min) pairs from 12-byte packed format.
	// scales[0..3] → s03, scales[4..7] → s47, scales[8..11] → s811
	s03 := Slice(scalesRaw, AxisRange(), AxisRange(), AxisRange(0, 4))   // [N, numBlocks, 4]
	s47 := Slice(scalesRaw, AxisRange(), AxisRange(), AxisRange(4, 8))   // [N, numBlocks, 4]
	s811 := Slice(scalesRaw, AxisRange(), AxisRange(), AxisRange(8, 12)) // [N, numBlocks, 4]

	mask6 := Scalar(g, dtypes.Uint8, uint8(63))
	mask4 := Scalar(g, dtypes.Uint8, uint8(0x0F))

	// scs[0..3] = s[0..3] & 63, mns[0..3] = s[4..7] & 63
	scsLo := BitwiseAnd(s03, mask6)
	mnsLo := BitwiseAnd(s47, mask6)

	// scs[4..7] = (s[8..11] & 0xF) | ((s[0..3] >> 6) << 4)
	scsHi := BitwiseOr(
		BitwiseAnd(s811, mask4),
		BitwiseShiftLeftScalar(BitwiseShiftRightLogicalScalar(s03, uint8(6)), uint8(4)),
	)
	// mns[4..7] = (s[8..11] >> 4) | ((s[4..7] >> 6) << 4)
	mnsHi := BitwiseOr(
		BitwiseShiftRightLogicalScalar(s811, uint8(4)),
		BitwiseShiftLeftScalar(BitwiseShiftRightLogicalScalar(s47, uint8(6)), uint8(4)),
	)

	scsAll := ConvertDType(Concatenate([]*Node{scsLo, scsHi}, 2), dtypes.Float32) // [N, numBlocks, 8]
	mnsAll := ConvertDType(Concatenate([]*Node{mnsLo, mnsHi}, 2), dtypes.Float32) // [N, numBlocks, 8]
	scsAll = Reshape(scsAll, N, numBlocks, 8, 1)                                  // broadcast over 32 values
	mnsAll = Reshape(mnsAll, N, numBlocks, 8, 1)

	// Extract nibbles from qs: 4 chunks of 32 bytes, each producing two sub-blocks.
	// Chunk c → low nibbles = sub-block 2c, high nibbles = sub-block 2c+1.
	nibbleMask := Scalar(g, dtypes.Uint8, uint8(0x0F))
	var subBlocks [8]*Node
	for c := range 4 {
		chunk := Slice(qs, AxisRange(), AxisRange(), AxisRange(c*32, (c+1)*32))
		subBlocks[2*c] = BitwiseAnd(chunk, nibbleMask)
		subBlocks[2*c+1] = BitwiseShiftRightLogicalScalar(chunk, uint8(4))
	}
	q4All := ConvertDType(Concatenate(subBlocks[:], 2), dtypes.Float32) // [N, numBlocks, 256]
	q4All = Reshape(q4All, N, numBlocks, 8, 32)

	// result = d * scs * q4 - dmin * mns
	result := Sub(
		Mul(Mul(d, scsAll), q4All),
		Mul(dmin, mnsAll),
	) // [N, numBlocks, 8, 32]

	K := numBlocks * 256
	return Reshape(result, N, K)
}
