"""Dropless, normalized Top-K SwiGLU MoE with native PyTorch grouped GEMMs.

Routing follows Qwen3-MoE (no shared experts). Each rank owns all experts; this
module does not implement expert parallelism. The torch backend is a readable
reference, while grouped_mm requires CUDA BF16 and never silently falls back.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.config import Config
from model.fused_swiglu import swiglu


def load_balancing_loss(router_stats):
    """Qwen/HF auxiliary loss over all layers in the local, unpadded microbatch.

    Stats are (mean softmax probability, selected-token count / token count).
    Averaging each across layers equals concatenating their router logits, but
    avoids retaining concatenated logits/one-hot masks. Uniform routing has
    loss top_k (not 1), matching Hugging Face's Qwen3-MoE normalization.
    """
    probabilities, frequencies = zip(*router_stats)
    probabilities = torch.stack(probabilities).mean(0)
    frequencies = torch.stack(frequencies).mean(0)
    return probabilities.numel() * (probabilities * frequencies).sum()


class SparseMoE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.width = config.moe_intermediate_size
        self.backend = config.moe_backend
        self.activation_backend = config.mlp_backend
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        expert_hidden = config.n_embd
        # [expert, output, input]; fused gate/up projection, separate down projection.
        self.gate_up = nn.Parameter(torch.empty(self.num_experts, 2 * self.width, expert_hidden))
        self.down = nn.Parameter(torch.empty(self.num_experts, expert_hidden, self.width))
        std = config.initializer_range or math.sqrt(2 / (5 * config.n_embd))
        nn.init.normal_(self.gate.weight, std=std)
        nn.init.normal_(self.gate_up, std=std)
        nn.init.normal_(self.down, std=(config.initializer_range or 1 / math.sqrt(config.n_embd) / config.n_layer))

    def _activation(self, projected):
        gate, up = projected.chunk(2, dim=-1)
        if self.activation_backend == "torch" or gate.numel() == 0:
            return F.silu(gate) * up
        return swiglu(gate, up, require_fused=self.activation_backend == "flash")

    def _experts(self, sorted_x, counts):
        gu, down = self.gate_up, self.down
        if self.backend == "torch":
            # Reference only: tolist synchronizes on CUDA. Empty experts remain in
            # the graph and receive zero gradients, keeping DDP's parameter set static.
            chunks = sorted_x.split(counts.tolist())
            return torch.cat([F.linear(self._activation(F.linear(x, gu)), down)
                              for x, gu, down in zip(chunks, gu, down)], dim=0)
        if not sorted_x.is_cuda or sorted_x.dtype != torch.bfloat16:
            raise RuntimeError("grouped_mm MoE requires CUDA BF16 (use bf16-mixed training or moe_backend='torch')")
        grouped_mm = getattr(F, 'grouped_mm', getattr(torch, '_grouped_mm', None))
        if grouped_mm is None:
            raise RuntimeError("This PyTorch lacks torch.nn.functional.grouped_mm; upgrade PyTorch or select the reference backend explicitly")
        offsets = counts.cumsum(0, dtype=torch.int32)
        projected = grouped_mm(sorted_x, gu.to(sorted_x.dtype).contiguous().transpose(-1, -2), offs=offsets)
        hidden = self._activation(projected)
        return grouped_mm(hidden, down.to(hidden.dtype).contiguous().transpose(-1, -2), offs=offsets)

    def forward(self, x, *, return_router_stats=False):
        """x: [..., hidden]; return the same shape, optionally with local routing stats."""
        if x.ndim < 2 or x.numel() == 0:
            raise ValueError("MoE input must contain at least one token and a hidden dimension")
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        # 1. Route each token to K experts; selected weights sum to one.
        logits = self.gate(flat)
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, selected = torch.topk(probabilities, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        # 2. Sort the N*K assignments by expert. order // K recovers token IDs.
        order = selected.flatten().argsort(stable=True)
        counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.int32)
        counts.scatter_add_(0, selected.flatten(), torch.ones_like(selected.flatten(), dtype=torch.int32))
        # Cast before gathering: the residual stream can be FP32 under mixed precision.
        expert_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        expert_x = flat
        sorted_x = expert_x.to(expert_dtype).index_select(0, order // self.top_k)
        sorted_y = self._experts(sorted_x, counts)
        # 3. Undo the permutation and combine the K expert outputs for each token.
        routed_y = torch.empty_like(sorted_y).index_copy(0, order, sorted_y)
        routed_y = routed_y.view(flat.size(0), self.top_k, expert_x.size(-1))
        # Match Qwen's residual dtype for probability weighting and accumulation.
        y = (routed_y.to(x.dtype) * weights.to(x.dtype).unsqueeze(-1)).sum(1)
        y = y.view(shape)
        if return_router_stats:
            return y, (probabilities.mean(0), counts.float() / flat.size(0))
        return y
