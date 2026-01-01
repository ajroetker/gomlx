package simplego

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCapabilities_Clone(t *testing.T) {
	cloned := Capabilities.Clone()

	// Verify fields are cloned
	assert.Equal(t, len(Capabilities.Operations), len(cloned.Operations),
		"Clone should copy Operations map")
	assert.Equal(t, len(Capabilities.DTypes), len(cloned.DTypes),
		"Clone should copy DTypes map")
}
