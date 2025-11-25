// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && arm64

package simplego

import (
	"fmt"
	"testing"
)

// TestSMEDetection tests SME feature detection
func TestSMEDetection(t *testing.T) {
	if hasSME {
		t.Log("SME (Scalable Matrix Extension) detected - using 512-bit vectors")
	} else {
		t.Log("SME not available - using scalar fallback")
	}
}

// TestDotProductSME tests the SME dot product implementation with progressive sizes
func TestDotProductSME(t *testing.T) {
	if !hasSME {
		t.Skip("SME not available on this system")
	}

	// Test with progressively larger sizes
	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}

	for _, size := range sizes {
		a := make([]float32, size)
		b := make([]float32, size)

		for i := 0; i < size; i++ {
			a[i] = 1.0
			b[i] = 2.0
		}

		var expected float32
		for i := 0; i < size; i++ {
			expected += a[i] * b[i]
		}

		result := dotProduct_sme(a, b, 0, 0, int64(size))

		if result != expected {
			t.Errorf("Size %d failed: got %f, expected %f", size, result, expected)
			return
		}
	}
}

// BenchmarkDotProductSME benchmarks the SME implementation
func BenchmarkDotProductSME(b *testing.B) {
	if !hasSME {
		b.Skip("SME not available on this system")
	}

	sizes := []int{64, 512, 2048, 8192}

	for _, size := range sizes {
		a := make([]float32, size)
		c := make([]float32, size)
		for i := 0; i < size; i++ {
			a[i] = float32(i)
			c[i] = 2.0
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = dotProduct_sme(a, c, 0, 0, int64(size))
			}
		})
	}
}

// BenchmarkDotProductScalar benchmarks the scalar implementation for comparison
func BenchmarkDotProductScalar(b *testing.B) {
	sizes := []int{64, 512, 2048, 8192}

	for _, size := range sizes {
		a := make([]float32, size)
		c := make([]float32, size)
		for i := 0; i < size; i++ {
			a[i] = float32(i)
			c[i] = 2.0
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var sum float32
				for j := 0; j < size; j++ {
					sum += a[j] * c[j]
				}
				_ = sum
			}
		})
	}
}
