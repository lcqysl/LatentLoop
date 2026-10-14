"""Configuration for the three architectures in the paper."""
from dataclasses import dataclass
import math


@dataclass
class Config:
    name: str = "latentloop"
    architecture: str = "latentloop"
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 16
    n_embd: int = 1296
    n_head: int = 8
    n_query_groups: int = 2
    attention_head_dim: int = 140
    num_experts: int = 75
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 648
    router_aux_loss_coef: float = 0.001
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    initializer_range: float = 0.02
    attention_backend: str = "sdpa"
    moe_backend: str = "torch"
    mlp_backend: str = "torch"

    def __post_init__(self):
        if self.architecture not in ("moe_base", "moe_loop", "latentloop"):
            raise ValueError("architecture must be moe_base, moe_loop, or latentloop")
        for key in ("block_size", "vocab_size", "n_layer", "n_embd", "n_head",
                    "n_query_groups", "attention_head_dim", "num_experts",
                    "num_experts_per_tok", "moe_intermediate_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.n_layer % 4 or self.n_head % self.n_query_groups:
            raise ValueError("layers must be divisible by four; KV heads must divide query heads")
        if self.attention_head_dim % 2 or self.num_experts_per_tok > self.num_experts:
            raise ValueError("head dimension must be even; active experts must not exceed stored experts")
        if self.architecture == "latentloop" and self.block_size % 2:
            raise ValueError("LatentLoop context length must be divisible by two")
        if self.attention_backend not in ("sdpa", "flash"):
            raise ValueError("attention_backend must be sdpa or flash")
        if self.moe_backend not in ("torch", "grouped_mm"):
            raise ValueError("moe_backend must be torch or grouped_mm")
        if self.mlp_backend not in ("torch", "flash"):
            raise ValueError("mlp_backend must be torch or flash")
        if self.moe_backend == "grouped_mm" and (self.n_embd % 8 or self.moe_intermediate_size % 8):
            raise ValueError("grouped_mm requires model and expert widths divisible by eight")
        if (not all(math.isfinite(v) for v in (self.norm_eps, self.rope_base, self.initializer_range, self.router_aux_loss_coef))
                or min(self.norm_eps, self.rope_base, self.initializer_range) <= 0 or self.router_aux_loss_coef < 0):
            raise ValueError("invalid normalization, initialization, or router loss scale")

    @property
    def padded_vocab_size(self):
        return (self.vocab_size + 63) // 64 * 64

    @property
    def head_size(self):
        return self.attention_head_dim

    @property
    def rotary_dim(self):
        return self.head_size

    @property
    def norm_class(self):
        from model.rmsnorm import FusedRMSNorm
        return FusedRMSNorm

    @property
    def loop_start(self):
        return self.n_layer // 4

    @property
    def loop_end(self):
        return 3 * self.n_layer // 4

    @property
    def layer_order(self):
        if self.architecture == "moe_base":
            return list(range(self.n_layer))
        return (list(range(self.loop_end)) + list(range(self.loop_start, self.loop_end))
                + list(range(self.loop_end, self.n_layer)))

    @property
    def coarse_execution_span(self):
        return (self.loop_start, self.loop_end) if self.architecture == "latentloop" else None
