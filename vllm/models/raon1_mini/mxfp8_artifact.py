"""Versioned, immutable preconverted weights for the Raon TP=1 MXFP8 recipe."""
import hashlib
import json
import os
from pathlib import Path

FORMAT = "raon-mxfp8-mode4-v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(artifact, model):
    root, model = Path(artifact), Path(model)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != FORMAT:
        raise ValueError("Unsupported Raon MXFP8 artifact format")
    for name, expected in manifest["source_sha256"].items():
        if sha256(model / name) != expected:
            raise ValueError(f"MXFP8 artifact belongs to a different checkpoint: {name}")
    for entry in manifest["entries"].values():
        if sha256(root / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Corrupt MXFP8 weights: {entry['file']}")
    return manifest


def load(key, shapes, device):
    """Missing or incompatible tensors are errors, never online quantization fallbacks."""
    import torch
    from safetensors.torch import load_file

    root = Path(os.environ["RAON_MXFP8_WEIGHTS"])
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != FORMAT:
        raise ValueError("Unsupported Raon MXFP8 artifact format")
    entry = manifest["entries"][key]
    if entry["source_shapes"] != [list(shape) for shape in shapes]:
        raise ValueError(f"MXFP8 source shape mismatch: {key}")
    tensors = load_file(str(root / entry["file"]), device=str(device))
    for name, spec in entry["tensors"].items():
        tensor = tensors[name]
        if list(tensor.shape) != spec["shape"] or str(tensor.dtype) != spec["dtype"]:
            raise ValueError(f"MXFP8 tensor format mismatch: {key}/{name}")
    print(f"[mxfp8-artifact] LOADED {key}; weight quantization skipped", flush=True)
    return tensors
