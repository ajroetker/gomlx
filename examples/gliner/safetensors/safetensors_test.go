// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// createTestSafetensors creates a minimal safetensors byte buffer for testing.
func createTestSafetensors(t *testing.T) []byte {
	t.Helper()

	// Create test tensor data: 2x3 float32 tensor with values 1-6.
	tensorData := make([]byte, 6*4) // 6 float32s = 24 bytes
	for i := 0; i < 6; i++ {
		binary.LittleEndian.PutUint32(tensorData[i*4:], math.Float32bits(float32(i+1)))
	}

	// Create header.
	header := map[string]any{
		"__metadata__": map[string]string{
			"format": "test",
		},
		"test_tensor": map[string]any{
			"dtype":        "F32",
			"shape":        []int{2, 3},
			"data_offsets": []int{0, 24},
		},
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("failed to marshal header: %v", err)
	}

	// Assemble file: 8 bytes header size + header + data.
	result := make([]byte, 8+len(headerBytes)+len(tensorData))
	binary.LittleEndian.PutUint64(result[:8], uint64(len(headerBytes)))
	copy(result[8:], headerBytes)
	copy(result[8+len(headerBytes):], tensorData)

	return result
}

func TestParse(t *testing.T) {
	data := createTestSafetensors(t)

	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Check metadata.
	if f.Metadata["format"] != "test" {
		t.Errorf("expected metadata format='test', got %q", f.Metadata["format"])
	}

	// Check tensor count.
	if len(f.Tensors) != 1 {
		t.Errorf("expected 1 tensor, got %d", len(f.Tensors))
	}

	// Check tensor info.
	info, ok := f.Get("test_tensor")
	if !ok {
		t.Fatal("tensor 'test_tensor' not found")
	}
	if info.DType != dtypes.Float32 {
		t.Errorf("expected dtype Float32, got %v", info.DType)
	}
	if info.Shape.Rank() != 2 || info.Shape.Dimensions[0] != 2 || info.Shape.Dimensions[1] != 3 {
		t.Errorf("expected shape [2, 3], got %v", info.Shape)
	}
}

func TestToTensor(t *testing.T) {
	data := createTestSafetensors(t)

	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	tensor, err := f.ToTensor("test_tensor")
	if err != nil {
		t.Fatalf("ToTensor failed: %v", err)
	}

	// Check shape.
	if tensor.Shape().Rank() != 2 {
		t.Errorf("expected rank 2, got %d", tensor.Shape().Rank())
	}
	if tensor.Shape().Dimensions[0] != 2 || tensor.Shape().Dimensions[1] != 3 {
		t.Errorf("expected shape [2, 3], got %v", tensor.Shape())
	}

	// Check values.
	values := tensor.Value().([][]float32)
	expected := [][]float32{{1, 2, 3}, {4, 5, 6}}
	for i := range expected {
		for j := range expected[i] {
			if values[i][j] != expected[i][j] {
				t.Errorf("value[%d][%d] = %f, expected %f", i, j, values[i][j], expected[i][j])
			}
		}
	}
}

func TestNames(t *testing.T) {
	data := createTestSafetensors(t)

	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	names := f.Names()
	if len(names) != 1 || names[0] != "test_tensor" {
		t.Errorf("expected names=['test_tensor'], got %v", names)
	}
}

func TestMultipleTensors(t *testing.T) {
	// Create header with two tensors.
	header := map[string]any{
		"tensor_a": map[string]any{
			"dtype":        "F32",
			"shape":        []int{2},
			"data_offsets": []int{0, 8},
		},
		"tensor_b": map[string]any{
			"dtype":        "I64",
			"shape":        []int{3},
			"data_offsets": []int{8, 32},
		},
	}
	headerBytes, _ := json.Marshal(header)

	// Create data: 2 float32s + 3 int64s = 8 + 24 = 32 bytes.
	tensorData := make([]byte, 32)
	binary.LittleEndian.PutUint32(tensorData[0:], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(tensorData[4:], math.Float32bits(2.0))
	binary.LittleEndian.PutUint64(tensorData[8:], 10)
	binary.LittleEndian.PutUint64(tensorData[16:], 20)
	binary.LittleEndian.PutUint64(tensorData[24:], 30)

	// Assemble file.
	result := make([]byte, 8+len(headerBytes)+len(tensorData))
	binary.LittleEndian.PutUint64(result[:8], uint64(len(headerBytes)))
	copy(result[8:], headerBytes)
	copy(result[8+len(headerBytes):], tensorData)

	f, err := Parse(result)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if len(f.Tensors) != 2 {
		t.Errorf("expected 2 tensors, got %d", len(f.Tensors))
	}

	// Check tensor_a.
	tensorA, err := f.ToTensor("tensor_a")
	if err != nil {
		t.Fatalf("ToTensor(tensor_a) failed: %v", err)
	}
	valuesA := tensorA.Value().([]float32)
	if valuesA[0] != 1.0 || valuesA[1] != 2.0 {
		t.Errorf("tensor_a values = %v, expected [1, 2]", valuesA)
	}

	// Check tensor_b.
	tensorB, err := f.ToTensor("tensor_b")
	if err != nil {
		t.Fatalf("ToTensor(tensor_b) failed: %v", err)
	}
	valuesB := tensorB.Value().([]int64)
	if valuesB[0] != 10 || valuesB[1] != 20 || valuesB[2] != 30 {
		t.Errorf("tensor_b values = %v, expected [10, 20, 30]", valuesB)
	}
}

func TestDTypes(t *testing.T) {
	tests := []struct {
		dtype    string
		expected dtypes.DType
	}{
		{"F64", dtypes.Float64},
		{"F32", dtypes.Float32},
		{"F16", dtypes.Float16},
		{"BF16", dtypes.BFloat16},
		{"I64", dtypes.Int64},
		{"I32", dtypes.Int32},
		{"I16", dtypes.Int16},
		{"I8", dtypes.Int8},
		{"U64", dtypes.Uint64},
		{"U32", dtypes.Uint32},
		{"U16", dtypes.Uint16},
		{"U8", dtypes.Uint8},
		{"BOOL", dtypes.Bool},
	}

	for _, tt := range tests {
		t.Run(tt.dtype, func(t *testing.T) {
			got, err := parseDType(tt.dtype)
			if err != nil {
				t.Fatalf("parseDType(%q) failed: %v", tt.dtype, err)
			}
			if got != tt.expected {
				t.Errorf("parseDType(%q) = %v, expected %v", tt.dtype, got, tt.expected)
			}
		})
	}
}

func TestInvalidDType(t *testing.T) {
	_, err := parseDType("INVALID")
	if err == nil {
		t.Error("expected error for invalid dtype")
	}
}

func TestTensorNotFound(t *testing.T) {
	data := createTestSafetensors(t)
	f, _ := Parse(data)

	_, err := f.ToTensor("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent tensor")
	}
}
