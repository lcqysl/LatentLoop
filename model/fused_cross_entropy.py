# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Copyright (c) 2023, Tri Dao.

"""Token cross entropy with optional fused CUDA forward/backward."""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def cross_entropy_extension():
    try:
        import xentropy_cuda_lib
        return xentropy_cuda_lib
    except (ImportError, OSError):
        return None


class SoftmaxCrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing, ignored_index, inplace_backward):
        logits, labels = logits.contiguous(), labels.contiguous()
        losses, lse = cross_entropy_extension().forward(logits, labels, smoothing)
        losses.masked_fill_(labels == ignored_index, 0)
        ctx.save_for_backward(logits, lse, labels)
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits, lse, labels = ctx.saved_tensors
        grad_loss = grad_loss.contiguous().masked_fill(labels == ctx.ignored_index, 0)
        grad_logits = cross_entropy_extension().backward(
            grad_loss, logits, lse, labels, ctx.smoothing, ctx.inplace_backward, logits.size(-1)
        )
        return grad_logits, None, None, None, None


class FusedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean", label_smoothing=0.0, inplace_backward=True):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean, sum, or none")
        if not 0 <= label_smoothing <= 1:
            raise ValueError("label_smoothing must be in [0, 1]")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward

    def forward(self, input, target):
        input, target = input.reshape(-1, input.size(-1)), target.reshape(-1)
        if not input.is_cuda or cross_entropy_extension() is None:
            return F.cross_entropy(input.float(), target, ignore_index=self.ignore_index,
                                   reduction=self.reduction, label_smoothing=self.label_smoothing)
        losses = SoftmaxCrossEntropyLossFn.apply(
            input, target, self.label_smoothing, self.ignore_index, self.inplace_backward,
        )
        if self.reduction == "mean":
            return losses.sum() / (target != self.ignore_index).sum()
        return losses.sum() if self.reduction == "sum" else losses
