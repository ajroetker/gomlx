// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build noasm || !darwin || !arm64

package simplego

// hasSME indicates whether SME SIMD optimizations are available.
// SME is only available on Apple M4+ (darwin/arm64), so it's always false on other platforms.
const hasSME = false
