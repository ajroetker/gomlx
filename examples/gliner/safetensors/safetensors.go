// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package safetensors provides a parser for the SafeTensors file format.
//
// SafeTensors is a simple, safe format for storing tensors developed by HuggingFace.
// Format specification: https://huggingface.co/docs/safetensors/
//
// File structure:
//   - 8 bytes: header size N (little-endian uint64)
//   - N bytes: JSON header with tensor metadata
//   - Remaining bytes: raw tensor data (contiguous, little-endian)
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// File represents a parsed safetensors file.
type File struct {
	// Metadata contains optional file-level metadata (e.g., {"format": "pt"}).
	Metadata map[string]string

	// Tensors contains all tensors in the file, keyed by name.
	Tensors map[string]*TensorInfo

	// data holds the raw tensor data buffer.
	data []byte
}

// TensorInfo contains metadata about a tensor.
type TensorInfo struct {
	Name   string
	DType  dtypes.DType
	Shape  shapes.Shape
	offset uint64 // start offset in data buffer
	length uint64 // length in bytes
}

// headerEntry is used for JSON unmarshaling of the header.
type headerEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Open reads and parses a safetensors file from disk.
func Open(path string) (*File, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read safetensors file %q", path)
	}
	return Parse(data)
}

// Parse parses safetensors data from a byte buffer.
func Parse(data []byte) (*File, error) {
	if len(data) < 8 {
		return nil, errors.New("safetensors: file too small, missing header size")
	}

	// Read header size (first 8 bytes, little-endian uint64).
	headerSize := binary.LittleEndian.Uint64(data[:8])
	if headerSize > uint64(len(data)-8) {
		return nil, errors.Errorf("safetensors: header size %d exceeds file size %d", headerSize, len(data)-8)
	}

	// Parse JSON header.
	headerBytes := data[8 : 8+headerSize]
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, errors.Wrapf(err, "safetensors: failed to parse JSON header")
	}

	f := &File{
		Metadata: make(map[string]string),
		Tensors:  make(map[string]*TensorInfo),
		data:     data[8+headerSize:],
	}

	// Process header entries.
	for name, raw := range rawHeader {
		if name == "__metadata__" {
			// Parse metadata.
			if err := json.Unmarshal(raw, &f.Metadata); err != nil {
				return nil, errors.Wrapf(err, "safetensors: failed to parse __metadata__")
			}
			continue
		}

		// Parse tensor info.
		var entry headerEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			return nil, errors.Wrapf(err, "safetensors: failed to parse tensor %q", name)
		}

		dtype, err := parseDType(entry.DType)
		if err != nil {
			return nil, errors.Wrapf(err, "safetensors: tensor %q", name)
		}

		f.Tensors[name] = &TensorInfo{
			Name:   name,
			DType:  dtype,
			Shape:  shapes.Make(dtype, entry.Shape...),
			offset: uint64(entry.DataOffsets[0]),
			length: uint64(entry.DataOffsets[1] - entry.DataOffsets[0]),
		}
	}

	return f, nil
}

// Names returns all tensor names in sorted order.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.Tensors))
	for name := range f.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Get returns information about a tensor by name.
func (f *File) Get(name string) (*TensorInfo, bool) {
	info, ok := f.Tensors[name]
	return info, ok
}

// Data returns the raw bytes for a tensor.
func (f *File) Data(name string) ([]byte, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, errors.Errorf("safetensors: tensor %q not found", name)
	}

	end := info.offset + info.length
	if end > uint64(len(f.data)) {
		return nil, errors.Errorf("safetensors: tensor %q data out of bounds", name)
	}

	return f.data[info.offset:end], nil
}

// ToTensor converts a tensor to a GoMLX tensor.
func (f *File) ToTensor(name string) (*tensors.Tensor, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, errors.Errorf("safetensors: tensor %q not found", name)
	}

	data, err := f.Data(name)
	if err != nil {
		return nil, err
	}

	// Create tensor with the correct shape.
	t := tensors.FromShape(info.Shape)

	// Copy data into tensor.
	accessErr := t.MutableBytes(func(tensorBytes []byte) {
		if len(data) != len(tensorBytes) {
			err = errors.Errorf("safetensors: tensor %q data size mismatch: got %d bytes, expected %d",
				name, len(data), len(tensorBytes))
			return
		}
		copy(tensorBytes, data)
	})
	if accessErr != nil {
		return nil, accessErr
	}
	if err != nil {
		return nil, err
	}

	return t, nil
}

// parseDType converts a safetensors dtype string to a GoMLX DType.
func parseDType(s string) (dtypes.DType, error) {
	switch s {
	case "F64":
		return dtypes.Float64, nil
	case "F32":
		return dtypes.Float32, nil
	case "F16":
		return dtypes.Float16, nil
	case "BF16":
		return dtypes.BFloat16, nil
	case "I64":
		return dtypes.Int64, nil
	case "I32":
		return dtypes.Int32, nil
	case "I16":
		return dtypes.Int16, nil
	case "I8":
		return dtypes.Int8, nil
	case "U64":
		return dtypes.Uint64, nil
	case "U32":
		return dtypes.Uint32, nil
	case "U16":
		return dtypes.Uint16, nil
	case "U8":
		return dtypes.Uint8, nil
	case "BOOL":
		return dtypes.Bool, nil
	default:
		return dtypes.InvalidDType, fmt.Errorf("unknown dtype %q", s)
	}
}

// String returns a summary of the file contents.
func (f *File) String() string {
	return fmt.Sprintf("SafeTensors{tensors: %d, metadata: %v}", len(f.Tensors), f.Metadata)
}

// Summary returns a detailed summary of all tensors.
func (f *File) Summary() string {
	var s string
	s += fmt.Sprintf("SafeTensors file with %d tensors:\n", len(f.Tensors))
	if len(f.Metadata) > 0 {
		s += fmt.Sprintf("  Metadata: %v\n", f.Metadata)
	}
	for _, name := range f.Names() {
		info := f.Tensors[name]
		s += fmt.Sprintf("  %s: %s\n", name, info.Shape)
	}
	return s
}
