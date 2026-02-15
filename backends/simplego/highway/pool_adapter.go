// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"sync"
	"sync/atomic"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
	"github.com/gomlx/gomlx/internal/workerspool"
)

// poolAdapter wraps a workerspool.Pool to implement the workerpool.Executor
// interface from go-highway. This allows a single pool (gomlx's) to drive
// both graph-level and intra-op parallelism, eliminating the ~2x GOMAXPROCS
// thread overhead from maintaining two independent pools.
type poolAdapter struct {
	pool *workerspool.Pool
}

// Compile-time check.
var _ workerpool.Executor = (*poolAdapter)(nil)

// NewPoolAdapter creates a workerpool.Executor backed by the given workerspool.Pool.
func NewPoolAdapter(pool *workerspool.Pool) workerpool.Executor {
	return &poolAdapter{pool: pool}
}

func (a *poolAdapter) NumWorkers() int {
	return a.pool.AdjustedMaxParallelism()
}

func (a *poolAdapter) ParallelFor(n int, fn func(start, end int)) {
	numWorkers := a.NumWorkers()
	if n <= 1 || numWorkers <= 1 {
		fn(0, n)
		return
	}
	workers := min(n, numWorkers)
	chunkSize := n / workers
	remainder := n % workers

	var wg sync.WaitGroup
	wg.Add(workers)
	start := 0
	for i := range workers {
		end := start + chunkSize
		if i < remainder {
			end++
		}
		s, e := start, end
		a.pool.WaitToStart(func() {
			defer wg.Done()
			fn(s, e)
		})
		start = end
	}
	wg.Wait()
}

func (a *poolAdapter) ParallelForAtomic(n int, fn func(i int)) {
	numWorkers := a.NumWorkers()
	if n <= 1 || numWorkers <= 1 {
		for i := range n {
			fn(i)
		}
		return
	}
	workers := min(n, numWorkers)
	var idx atomic.Int32
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		a.pool.WaitToStart(func() {
			defer wg.Done()
			for {
				i := int(idx.Add(1)) - 1
				if i >= n {
					return
				}
				fn(i)
			}
		})
	}
	wg.Wait()
}

func (a *poolAdapter) ParallelForAtomicBatched(n int, batchSize int, fn func(start, end int)) {
	numWorkers := a.NumWorkers()
	if n <= 1 || numWorkers <= 1 {
		fn(0, n)
		return
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	workers := min(n, numWorkers)
	var idx atomic.Int32
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		a.pool.WaitToStart(func() {
			defer wg.Done()
			for {
				start := int(idx.Add(int32(batchSize))) - batchSize
				if start >= n {
					return
				}
				end := min(start+batchSize, n)
				fn(start, end)
			}
		})
	}
	wg.Wait()
}
