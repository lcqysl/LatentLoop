# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Copyright (c) 2022, Tri Dao.
# Adapted from Dao-AILab/flash-attention's layer normalization extension.

"""RMSNorm only: optional fused CUDA kernel, with a stable PyTorch reference."""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def rms_norm_extension():
    try:
        import dropout_layer_norm
        return dropout_layer_norm
    except (ImportError, OSError):
        return None


def _aligned(x):
    x = x.contiguous()
    return x if x.data_ptr() % 16 == 0 else x.clone()


class _FusedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x, weight = _aligned(x), _aligned(weight)
        matrix = x.reshape(-1, weight.numel())
        out, saved, mask, mean, inv_std = rms_norm_extension().dropout_add_ln_fwd(
            matrix, None, weight, None, None, None, None, None,
            0.0, eps, 1.0, 0, None, False, True,
        )
        ctx.save_for_backward(matrix if saved is None else saved, weight, mask, mean, inv_std)
        ctx.shape = x.shape
        return out.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        x, weight, mask, mean, inv_std = ctx.saved_tensors
        dx, _, dw, *_ = rms_norm_extension().dropout_add_ln_bwd(
            _aligned(grad).reshape_as(x), None, x, None, mask, mean, inv_std,
            weight, None, None, None, None, 0.0, 1.0, 0, False, True,
        )
        return dx.view(ctx.shape), dw, None


class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        if dim != -1:
            raise ValueError("Transformer RMSNorm normalizes the last dimension")
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x):
        # Match the fused kernel's output dtype, including bf16-mixed training.
        # Autocast may promote the native RMSNorm result to FP32. Attention
        # requires the same BF16/FP16 output contract as the fused extension.
        return F.rms_norm(x, (self.weight.numel(),), self.weight.to(x.dtype), self.eps).to(x.dtype)


class FusedRMSNorm(RMSNorm):
    def forward(self, x):
        if (x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and self.weight.numel() % 8 == 0 and self.weight.numel() <= 8192
                and rms_norm_extension() is not None):
            # FSDP can pair BF16 weights with an FP32 residual stream. The upstream
            # extension requires FP32 scale weights for FP32 inputs in that case.
            weight = self.weight.float() if x.dtype == torch.float32 else self.weight
            return _FusedRMSNormFn.apply(x, weight, self.eps)
        # The extension rejects non-multiples of 8 (e.g. heads 122/132/60).
        # Normalize the actual dimension; padding would change the statistic.
        return super().forward(x)
