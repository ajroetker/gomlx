#!/usr/bin/env python3
"""Convert GLiNER pytorch_model.bin to safetensors format.

Usage:
    pip install torch safetensors
    python convert_to_safetensors.py

This downloads the GLiNER small model and converts it to safetensors format.
"""

import torch
from safetensors.torch import save_file
from pathlib import Path
import urllib.request
import os

MODEL_URL = "https://huggingface.co/urchade/gliner_small-v2.1/resolve/main/pytorch_model.bin"
OUTPUT_DIR = Path(__file__).parent / "model"


def download_if_missing(url: str, path: Path) -> None:
    """Download file if it doesn't exist."""
    if path.exists():
        print(f"Already exists: {path}")
        return

    print(f"Downloading {url}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    print(f"Downloaded to {path}")


def convert_to_safetensors(pytorch_path: Path, safetensors_path: Path) -> None:
    """Convert pytorch_model.bin to safetensors format."""
    print(f"Loading {pytorch_path}...")
    state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)

    # Print summary
    print(f"\nFound {len(state_dict)} tensors:")
    total_params = 0
    for name, tensor in sorted(state_dict.items()):
        params = tensor.numel()
        total_params += params
        print(f"  {name}: {list(tensor.shape)} ({tensor.dtype})")

    print(f"\nTotal parameters: {total_params:,}")

    # Save as safetensors
    print(f"\nSaving to {safetensors_path}...")
    save_file(state_dict, safetensors_path)

    # Compare sizes
    pytorch_size = pytorch_path.stat().st_size
    safetensors_size = safetensors_path.stat().st_size
    print(f"\nFile sizes:")
    print(f"  pytorch_model.bin: {pytorch_size:,} bytes")
    print(f"  model.safetensors: {safetensors_size:,} bytes")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pytorch_path = OUTPUT_DIR / "pytorch_model.bin"
    safetensors_path = OUTPUT_DIR / "model.safetensors"

    download_if_missing(MODEL_URL, pytorch_path)
    convert_to_safetensors(pytorch_path, safetensors_path)

    print("\nDone! You can now use model.safetensors with the Go safetensors parser.")


if __name__ == "__main__":
    main()
