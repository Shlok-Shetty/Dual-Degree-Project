"""Pick the compute backend for the LLMs: CUDA, Apple MPS, or CPU.

    CUDA -> 4-bit NF4 parser (bf16 compute), fp16 verbalizer  (original setup)
    MPS  -> fp16, no quantization (bitsandbytes is CUDA-only)
    CPU  -> bf16, no quantization (slow, ~9 GB RAM for both models; fp32
            would need ~18 GB)

Set COMPARISON_SEARCH_DEVICE=cuda|mps|cpu to force a backend.
"""
import os

import torch


def pick_device() -> str:
    forced = os.environ.get("COMPARISON_SEARCH_DEVICE", "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def parser_dtype(device: str):
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.bfloat16)


def verbalizer_dtype(device: str):
    return {"cuda": torch.float16, "mps": torch.float16}.get(device, torch.bfloat16)


def describe(device: str) -> str:
    return {
        "cuda": "CUDA GPU",
        "mps": "Apple GPU (MPS, fp16)",
        "cpu": "CPU (bf16, slow)",
    }[device]
