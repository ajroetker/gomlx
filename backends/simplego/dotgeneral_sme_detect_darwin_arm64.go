// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && darwin && arm64

package simplego

import (
	"sync"
	"syscall"
)

var (
	smeDetected     bool
	smeDetectedOnce sync.Once
)

// detectSME checks for SME support on macOS (Apple M4 and later)
func detectSME() bool {
	smeDetectedOnce.Do(func() {
		// Check for SME support via sysctl
		// hw.optional.arm.FEAT_SME returns binary value: \x01 means available
		// We check for both integer 1 (raw byte 0x01) and string "1" (ASCII 0x31) just in case.
		hasSME, err := syscall.Sysctl("hw.optional.arm.FEAT_SME")
		if err == nil && len(hasSME) > 0 {
			if hasSME[0] == 1 || hasSME[0] == '1' {
				smeDetected = true
			}
		}
	})
	return smeDetected
}

// hasSME indicates whether SME SIMD optimizations are available.
// SME provides 512-bit vectors and is available on Apple M4 and later.
var hasSME = detectSME()
