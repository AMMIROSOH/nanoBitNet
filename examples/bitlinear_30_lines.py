"""The viral snippet: BitLinear in roughly 30 lines.

This file is for explanation, not for training speed. The full version lives in
nanobitnet.py and includes histograms, config integration, and tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ste(x, q):
    return x + (q - x).detach()


def round_clip(x, lo, hi):
    return ste(x, torch.clamp(torch.round(x), lo, hi))


def ternary_absmean(w, eps=1e-5):
    beta = w.abs().mean().clamp_min(eps)
    return round_clip(w / beta, -1, 1) * beta


def int8_absmax(x, eps=1e-5):
    qmax, qmin = 127, -128
    gamma = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    q = round_clip(x * qmax / gamma, qmin, qmax)
    return q * gamma / qmax


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        x = int8_absmax(self.norm(x))
        w = ternary_absmean(self.weight)
        return F.linear(x, w, None)


if __name__ == "__main__":
    layer = BitLinear(16, 32)
    y = layer(torch.randn(4, 8, 16))
    print(y.shape)
