from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nanobitnet import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a trained nanoBitNet checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=Path("out/nanobitnet.pt"))
    p.add_argument("--prompt", default="The future of AI is")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}. Run train.py first.")

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()}
    unk = next(iter(stoi.values()))
    idx = torch.tensor([[stoi.get(ch, unk) for ch in args.prompt]], dtype=torch.long, device=args.device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
