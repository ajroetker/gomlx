//go:build !arm64

package simplego

// detectFP16NEON returns false on non-ARM64 platforms.
func detectFP16NEON() bool {
	return false
}

// detectBF16NEON returns false on non-ARM64 platforms.
func detectBF16NEON() bool {
	return false
}
