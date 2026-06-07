"""nanoBitNet: the core model in one readable file.

This is intentionally close to Karpathy-style educational code: minimal classes,
plain PyTorch, and comments only where the low-bit mechanics are easy to miss.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(x: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward uses quantized, backward sees x."""

    return x + (quantized - x).detach()


def round_clip(x: torch.Tensor, qmin: int | float, qmax: int | float) -> torch.Tensor:
    """Round and clip, with STE gradients through the input."""

    q = torch.clamp(torch.round(x), qmin, qmax)
    return _ste(x, q)


def quantize_weight_absmean(weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """BitNet b1.58-style ternary weight quantization.

    Paper equation:
        W_f = RoundClip(W / beta, -1, 1)
        beta = mean(abs(W))

    We return the dequantized weight beta * W_f so this can drop into
    torch.nn.functional.linear while keeping the implementation simple.
    """

    beta = weight.abs().mean().clamp_min(eps)
    ternary = round_clip(weight / beta, -1, 1)
    return ternary * beta


def ternary_codes(weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return the visible {-1, 0, +1} codes for inspection and tests."""

    beta = weight.abs().mean().clamp_min(eps)
    return torch.clamp(torch.round(weight / beta), -1, 1)


def quantize_activation_absmax(
    x: torch.Tensor,
    bits: int = 8,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Absmax activation quantization, dequantized back to x's dtype.

    BitNet keeps activations at higher precision than weights; the paper uses
    8-bit activations in the main recipe. We quantize per token by taking the
    max over the final feature dimension.
    """

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    gamma = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    q = round_clip(x * qmax / gamma, qmin, qmax)
    return q * gamma / qmax


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class BitLinear(nn.Module):
    """A readable BitLinear layer.

    The latent parameter is full precision. Each forward pass quantizes it to
    ternary values and quantizes normalized activations to an int8-like grid.
    Bias is omitted to match the low-bit projection spirit and keep the math clean.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_bits: int = 8,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.pre_norm = pre_norm
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False) if pre_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x_q = quantize_activation_absmax(x, bits=self.activation_bits)
        w_q = quantize_weight_absmean(self.weight)
        return F.linear(x_q, w_q, None)

    @torch.no_grad()
    def ternary_histogram(self) -> dict[str, int]:
        q = ternary_codes(self.weight)
        return {
            "-1": int((q == -1).sum().item()),
            "0": int((q == 0).sum().item()),
            "+1": int((q == 1).sum().item()),
        }


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    linear_kind: str = "bitlinear"  # "bitlinear" or "linear"
    activation_bits: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


def make_linear(config: GPTConfig, in_features: int, out_features: int) -> nn.Module:
    if config.linear_kind == "bitlinear":
        return BitLinear(in_features, out_features, activation_bits=config.activation_bits)
    if config.linear_kind == "linear":
        return nn.Linear(in_features, out_features, bias=False)
    raise ValueError(f"unknown linear_kind: {config.linear_kind}")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = make_linear(config, config.n_embd, 3 * config.n_embd)
        self.proj = make_linear(config, config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = make_linear(config, config.n_embd, 4 * config.n_embd)
        self.act = SquaredReLU()
        self.proj = make_linear(config, 4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def bitlinear_histograms(self) -> list[dict[str, int]]:
        return [m.ternary_histogram() for m in self.modules() if isinstance(m, BitLinear)]


def ideal_weight_bits_per_param(linear_kind: str) -> float:
    if linear_kind == "bitlinear":
        return math.log2(3)
    if linear_kind == "linear":
        return 32.0
    raise ValueError(f"unknown linear_kind: {linear_kind}")
