from __future__ import annotations

import argparse
import time

import torch

from nanobitnet import GPT, GPTConfig, ideal_weight_bits_per_param


def run_one(model: GPT, idx: torch.Tensor, steps: int, device: str) -> float:
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            model(idx)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / steps


def format_mb(params: int, bits_per_param: float) -> str:
    return f"{params * bits_per_param / 8 / 1e6:.2f} MB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare tiny FP and BitNet-style models.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=65)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    idx = torch.randint(args.vocab_size, (args.batch_size, args.block_size), device=args.device)
    rows = []

    for name, kind, dtype in [
        ("fp32 tiny GPT", "linear", torch.float32),
        ("fp16 tiny GPT", "linear", torch.float16),
        ("BitNet-style ternary GPT", "bitlinear", torch.float32),
    ]:
        config = GPTConfig(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            n_layer=4,
            n_head=4,
            n_embd=128,
            linear_kind=kind,
        )
        model = GPT(config).to(args.device)
        if dtype == torch.float16:
            model = model.half()
        params = model.parameter_count()
        try:
            ms = run_one(model, idx, args.steps, args.device)
            ms_text = f"{ms:.2f}"
        except RuntimeError as exc:
            ms_text = f"unsupported ({exc.__class__.__name__})"
        bits = 16.0 if dtype == torch.float16 else ideal_weight_bits_per_param(kind)
        rows.append((name, params, bits, format_mb(params, bits), ms_text))

    print("| model | params | ideal bits/weight | ideal weight storage | forward ms |")
    print("|---|---:|---:|---:|---:|")
    for name, params, bits, storage, ms_text in rows:
        print(f"| {name} | {params:,} | {bits:.2f} | {storage} | {ms_text} |")
    print()
    print("Note: training checkpoints store latent weights in PyTorch tensors.")
    print("The 1.58-bit number is the ideal packed ternary forward representation.")
    print("This educational PyTorch path is not expected to beat FP32 CPU BLAS.")
    print("Real BitNet speedups require packed formats and optimized bitnet.cpp kernels.")


if __name__ == "__main__":
    main()
