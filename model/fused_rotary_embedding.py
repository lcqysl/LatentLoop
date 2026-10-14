# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Copyright (c) 2023, Tri Dao.

"""RoPE with optional CUDA acceleration; no dependency on any recurrent model."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def rotary_extension():
    try:
        import rotary_emb
        return rotary_emb
    except (ImportError, OSError):
        return None


def apply_rotary_reference(x, cos, sin, interleaved=False):
    length, rotary_dim = x.size(1), 2 * cos.size(-1)
    cos, sin = cos[:length, None, :], sin[:length, None, :]
    rotated, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    a, b = (rotated[..., ::2], rotated[..., 1::2]) if interleaved else rotated.chunk(2, dim=-1)
    first, second = a * cos - b * sin, a * sin + b * cos
    rotated = torch.stack((first, second), dim=-1).flatten(-2) if interleaved else torch.cat((first, second), dim=-1)
    return torch.cat((rotated, tail), dim=-1)


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        length, rotary_dim = x.size(1), 2 * cos.size(-1)
        out = x if inplace else torch.empty_like(x)
        src, dst = x[..., :rotary_dim], out[..., :rotary_dim]
        a, b = (src[..., ::2], src[..., 1::2]) if interleaved else src.chunk(2, dim=-1)
        oa, ob = (dst[..., ::2], dst[..., 1::2]) if interleaved else dst.chunk(2, dim=-1)
        rotary_extension().apply_rotary(a, b, cos[:length, None, :], sin[:length, None, :], oa, ob, False)
        if not inplace:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        else:
            ctx.mark_dirty(x)
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        return out

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        length, rotary_dim = grad.size(1), 2 * cos.size(-1)
        dx = torch.empty_like(grad)
        src, dst = grad[..., :rotary_dim], dx[..., :rotary_dim]
        a, b = (src[..., ::2], src[..., 1::2]) if ctx.interleaved else src.chunk(2, dim=-1)
        oa, ob = (dst[..., ::2], dst[..., 1::2]) if ctx.interleaved else dst.chunk(2, dim=-1)
        rotary_extension().apply_rotary(a, b, cos[:length, None, :], sin[:length, None, :], oa, ob, True)
        dx[..., rotary_dim:].copy_(grad[..., rotary_dim:])
        return dx, None, None, None, None


def apply_rotary_emb_func(x, cos, sin, interleaved=False, inplace=False):
    if cos.shape != sin.shape or cos.size(0) < x.size(1) or 2 * cos.size(-1) > x.size(-1):
        raise ValueError("RoPE cache shape does not match the input")
    if x.is_cuda and rotary_extension() is not None:
        return ApplyRotaryEmb.apply(x, cos, sin, interleaved, inplace)
    out = apply_rotary_reference(x, cos, sin, interleaved)
    return x.copy_(out) if inplace else out
