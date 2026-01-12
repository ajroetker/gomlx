// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// BenchmarkDotGeneralPaths compares different DotGeneral execution paths.
func BenchmarkDotGeneralPaths(b *testing.B) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		b.Skip("Test requires Go backend")
	}

	sizes := []struct {
		name    string
		m, k, n int
	}{
		{"64x64", 64, 64, 64},
		{"128x128", 128, 128, 128},
		{"256x256", 256, 256, 256},
		{"512x512", 512, 512, 512},
	}

	paths := []struct {
		name string
		path dotGeneralExecutionPath
	}{
		{"normalized", normalizedPath},
		{"blocked", blockedPath},
		{"smallMatMul", smallMatMulPath},
		{"highway", highwayMatMulPath},
	}

	for _, size := range sizes {
		lhsData := xslices.Iota(float32(1), size.m*size.k)
		rhsData := xslices.Iota(float32(1), size.k*size.n)
		lhs := tensors.FromFlatDataAndDimensions(lhsData, size.m, size.k)
		rhs := tensors.FromFlatDataAndDimensions(rhsData, size.k, size.n)

		for _, p := range paths {
			b.Run(size.name+"/"+p.name, func(b *testing.B) {
				goBackend.dotGeneralForceExecutionPath = p.path
				defer func() {
					goBackend.dotGeneralForceExecutionPath = autoSelectPath
				}()

				// Calculate GFLOPS
				flops := float64(2*size.m*size.n*size.k) / 1e9

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = graph.MustExecOnce(backend, func(l, r *graph.Node) *graph.Node {
						return graph.MatMul(l, r)
					}, lhs, rhs)
				}
				b.StopTimer()

				elapsed := b.Elapsed().Seconds()
				gflops := flops * float64(b.N) / elapsed
				b.ReportMetric(gflops, "GFLOPS")
			})
		}
	}
}

// BenchmarkDotGeneralPathsLarge benchmarks larger matrices where blocking matters more.
func BenchmarkDotGeneralPathsLarge(b *testing.B) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		b.Skip("Test requires Go backend")
	}

	size := 1024
	lhsData := xslices.Iota(float32(1), size*size)
	rhsData := xslices.Iota(float32(1), size*size)
	lhs := tensors.FromFlatDataAndDimensions(lhsData, size, size)
	rhs := tensors.FromFlatDataAndDimensions(rhsData, size, size)

	paths := []struct {
		name string
		path dotGeneralExecutionPath
	}{
		{"blocked", blockedPath},
		{"highway", highwayMatMulPath},
	}

	for _, p := range paths {
		b.Run(fmt.Sprintf("%dx%d/%s", size, size, p.name), func(b *testing.B) {
			goBackend.dotGeneralForceExecutionPath = p.path
			defer func() {
				goBackend.dotGeneralForceExecutionPath = autoSelectPath
			}()

			flops := float64(2*size*size*size) / 1e9

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = graph.MustExecOnce(backend, func(l, r *graph.Node) *graph.Node {
					return graph.MatMul(l, r)
				}, lhs, rhs)
			}
			b.StopTimer()

			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
