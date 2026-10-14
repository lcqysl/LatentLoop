# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Copyright Lightning AI. Licensed under the Apache License 2.0.
# Modified for LatentLoop: MoE architectures, loop visits, and latent sequence updates.
"""MoE Base, MoE Loop, and LatentLoop with separate caches for each layer visit."""
import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model.config import Config
from model.latent import ScaledDecoder, latent_correction
from model.fused_rotary_embedding import apply_rotary_emb_func
from model.moe import SparseMoE, load_balancing_loss

RoPECache = Tuple[Tensor, Tensor]
KVCache = Tuple[Tensor, Tensor]


@lru_cache(maxsize=1)
def flash_attention():
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func
    except (ImportError, OSError):
        return None


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
        ))
        self.layer_order = config.layer_order
        self.coarse_execution_span = config.coarse_execution_span
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.kv_caches = []
        self._cache_length = self._cache_capacity = 0
        self._coarse_pending = self._coarse_update = None
        self.apply(self._init_weights)
        self.coarse_decoder = ScaledDecoder(config.n_embd) if config.architecture == "latentloop" else None
        if config.architecture == "latentloop":
            # One physical owner for the shared MoE and both attention blocks.
            for i in range(config.loop_start, config.loop_end):
                block = self.transformer.h[i]
                block.norm_1_repeat = config.norm_class(config.n_embd, eps=config.norm_eps)
                block.norm_2_repeat = config.norm_class(config.n_embd, eps=config.norm_eps)
                block.attn_repeat = CausalSelfAttention(config)
                block.attn_repeat.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def reset_cache(self):
        self.kv_caches = []
        self._cache_length = self._cache_capacity = 0
        self._coarse_pending = self._coarse_update = None

    def _apply(self, fn, recurse=True):
        self.reset_cache()
        self.rope_cos = self.rope_sin = None
        return super()._apply(fn, recurse=recurse)
    def forward(self, idx: Tensor, max_seq_length: Optional[int] = None,
                input_pos: Optional[Tensor] = None, *, return_aux_loss: bool = False):
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx must have shape [batch, nonempty sequence]")
        batch, length = idx.shape
        capacity = self.config.block_size if max_seq_length is None else max_seq_length
        if not 0 < capacity <= self.config.block_size or length > capacity:
            raise ValueError("sequence/cache length must be within config.block_size")
        mask = None
        if input_pos is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("KV caching is inference-only; use torch.no_grad() or inference_mode()")
            if input_pos.ndim != 1 or input_pos.numel() != length or input_pos.dtype != torch.long:
                raise ValueError("input_pos must be a 1-D int64 tensor with one position per token")
            input_pos = input_pos.to(idx.device)
            expected = torch.arange(self._cache_length, self._cache_length + length, device=idx.device)
            if not torch.equal(input_pos, expected) or self._cache_length + length > capacity:
                raise ValueError("KV cache positions must be contiguous from zero and fit the cache; reset_cache() first")
            if not self.kv_caches:
                # Shared weights do not imply shared activations: each visit needs its own KV.
                self.kv_caches = [None] * len(self.layer_order)
                self._cache_capacity = capacity
            elif capacity != self._cache_capacity:
                raise ValueError("cache capacity changed; call reset_cache() first")
            mask = torch.arange(capacity, device=idx.device)[None, :] <= input_pos[:, None]
        if self.config.rotary_dim:
            rope_dtype = (torch.get_autocast_dtype(idx.device.type) if torch.is_autocast_enabled(idx.device.type)
                          else self.transformer.wte.weight.dtype)
            if (self.rope_cos is None or self.rope_cos.device != idx.device
                    or self.rope_cos.size(0) < capacity or self.rope_cos.dtype != rope_dtype):
                self.rope_cos, self.rope_sin = build_rope_cache(
                    capacity, self.config.rotary_dim, rope_dtype, idx.device,
                    self.config.rope_base, 1,
                )
            rope = (self.rope_cos[:length], self.rope_sin[:length]) if input_pos is None else (
                self.rope_cos.index_select(0, input_pos), self.rope_sin.index_select(0, input_pos)
            )
        else:
            rope = None
        x = self.transformer.wte(idx)
        router_stats = []
        visits = [0] * self.config.n_layer
        for i, layer_index in enumerate(self.layer_order):
            first_visit = visits[layer_index] == 0
            visits[layer_index] += 1
            if self.coarse_execution_span is not None and (
                    self.coarse_execution_span[0] <= i < self.coarse_execution_span[1]):
                if i == self.coarse_execution_span[0]:
                    x, stats = self._coarse_pass(x, input_pos, capacity, return_aux_loss)
                    router_stats.extend(stats)
                continue
            block = self.transformer.h[layer_index]
            cache = None if input_pos is None else self.kv_caches[i]
            if return_aux_loss:
                x, cache, stats = block(x, rope, mask, input_pos, cache, capacity,
                                       return_router_stats=True, first_visit=first_visit)
                router_stats.append(stats)  # Average aux over all executions, including repeat visits.
            else:
                x, cache = block(x, rope, mask, input_pos, cache, capacity, first_visit=first_visit)
            if input_pos is not None:
                self.kv_caches[i] = cache
        if input_pos is not None:
            self._cache_length += length
        logits = self.lm_head(self.transformer.ln_f(x))
        if return_aux_loss:
            aux = load_balancing_loss(router_stats) if router_stats else logits.new_zeros((), dtype=torch.float32)
            return logits, aux
        return logits

    def _coarse_pass(self, x, input_pos, capacity, return_aux_loss):
        cached = input_pos is not None
        coarse_capacity = capacity // 2
        stats = []

        def run_core(z, positions):
            ends = positions * 2 + 1
            rope = (self.rope_cos.index_select(0, ends), self.rope_sin.index_select(0, ends))
            mask = (torch.arange(coarse_capacity, device=x.device)[None, :] <= positions[:, None]
                    if cached else None)
            for execution in range(*self.coarse_execution_span):
                block = self.transformer.h[self.layer_order[execution]]
                cache = self.kv_caches[execution] if cached else None
                args = (z, rope, mask, positions if cached else None, cache, coarse_capacity)
                if return_aux_loss:
                    z, cache, entry = block(*args, return_router_stats=True, first_visit=True)
                    stats.append(entry)
                else:
                    z, cache = block(*args, first_visit=True)
                if cached:
                    self.kv_caches[execution] = cache
            return z

        result, pending, update = latent_correction(
            x, run_core, self.coarse_decoder,
            offset=self._cache_length if cached else 0,
            pending=self._coarse_pending if cached else None,
            last_update=self._coarse_update if cached else None,
        )
        if cached:
            self._coarse_pending, self._coarse_update = pending, update
        return result, stats


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = SparseMoE(config)
        self.attn_repeat = self.norm_1_repeat = self.norm_2_repeat = None

    def forward(self, x, rope, mask=None, input_pos=None, kv_cache=None, max_seq_length=None,
                *, return_router_stats=False, first_visit=False):
        repeated = self.attn_repeat is not None and not first_visit
        attention = self.attn_repeat if repeated else self.attn
        norm_1 = self.norm_1_repeat if repeated else self.norm_1
        norm_2 = self.norm_2_repeat if repeated else self.norm_2
        h, kv_cache = attention(norm_1(x), rope, mask, input_pos, kv_cache, max_seq_length)
        x = torch.add(x, h, alpha=1.0)
        if return_router_stats:
            h, stats = self.mlp(norm_2(x), return_router_stats=True)
            return torch.add(x, h, alpha=1.0), kv_cache, stats
        return torch.add(x, self.mlp(norm_2(x)), alpha=1.0), kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_query_groups) * config.head_size,
                              bias=False)
        self.proj = nn.Linear(config.n_head * config.head_size, config.n_embd, bias=False)
        self.q_norm = config.norm_class(config.head_size, eps=config.norm_eps)
        self.k_norm = config.norm_class(config.head_size, eps=config.norm_eps)

    def forward(self, x, rope, mask=None, input_pos=None, kv_cache=None, max_seq_length=None):
        batch, length, width = x.shape
        groups, heads, dim = self.config.n_query_groups, self.config.n_head, self.config.head_size
        projected = self.attn(x)
        # Checkpoint layout: [KV group, queries-per-group + key + value, head_dim].
        qkv = projected.view(batch, length, groups, heads // groups + 2, dim)
        q, k, v = qkv.split((heads // groups, 1, 1), dim=-2)
        q = q.reshape(batch, length, heads, dim)
        k, v = k.reshape(batch, length, groups, dim), v.reshape(batch, length, groups, dim)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            cos, sin = (value.to(q.dtype) for value in rope)
            q = apply_rotary_emb_func(q, cos, sin, inplace=False)
            k = apply_rotary_emb_func(k, cos, sin, inplace=False)
        if input_pos is not None:
            shape = (batch, max_seq_length, groups, dim)
            if kv_cache is None:
                kv_cache = (k.new_zeros(shape), v.new_zeros(shape))
            elif kv_cache[0].shape != shape or kv_cache[0].device != k.device or kv_cache[0].dtype != k.dtype:
                raise ValueError("cache batch/device/dtype changed; call reset_cache() first")
            kv_cache[0].index_copy_(1, input_pos, k)
            kv_cache[1].index_copy_(1, input_pos, v)
            k, v = kv_cache
        y = self.scaled_dot_product_attention(q, k, v, mask, input_pos)
        y = y.reshape(batch, length, heads * dim)
        return self.proj(y), kv_cache

    def scaled_dot_product_attention(self, q, k, v, mask=None, input_pos=None):
        backend = self.config.attention_backend
        flash = flash_attention() if backend != "sdpa" and q.is_cuda else None
        eligible = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
        if backend == "flash" and (flash is None or not eligible):
            raise RuntimeError("flash backend requires working flash-attn and CUDA fp16/bf16")
        # Cached decoding uses an explicit absolute-position mask; training takes the fast path.
        if flash is not None and eligible and mask is None:
            return flash(q, k, v, causal=True, softmax_scale=1 / math.sqrt(q.size(-1)))
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None, enable_gqa=q.size(1) != k.size(1),
        ).transpose(1, 2)


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device,
                     base: float = 10000, condense_ratio: float = 1) -> RoPECache:
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device, dtype=torch.float32) / n_elem))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32) / condense_ratio
    angles = torch.outer(positions, theta)
    return angles.cos().to(dtype), angles.sin().to(dtype)
