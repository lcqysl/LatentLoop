"""Thin adapter for FlashAttention's upstream SwiGLU; no local CUDA/Triton kernel."""
from functools import lru_cache

import torch
from torch.nn import functional as F


@lru_cache(maxsize=1)
def flash_swiglu():
    try:
        from flash_attn.ops.activations import swiglu
        return swiglu
    except (ImportError, OSError):
        return None


def swiglu(gate, up, *, require_fused=False):
    eligible = (gate.is_cuda and gate.numel() > 0 and gate.shape == up.shape
                and gate.device == up.device and gate.dtype == up.dtype
                and gate.dtype in (torch.float16, torch.bfloat16, torch.float32))
    fused = flash_swiglu() if eligible else None
    if require_fused and fused is None:
        raise RuntimeError("flash SwiGLU requires working flash-attn, CUDA/NVRTC, and fp32/fp16/bf16 inputs")
    if fused is not None:
        return fused(gate, up)
    return F.silu(gate) * up
