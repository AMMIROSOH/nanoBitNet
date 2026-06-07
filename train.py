from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import torch

from nanobitnet import GPT, GPTConfig


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def read_text(dataset: str, data_dir: Path) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "tinyshakespeare":
        path = data_dir / "tinyshakespeare.txt"
        if not path.exists():
            print(f"downloading {TINY_SHAKESPEARE_URL}")
            text = requests.get(TINY_SHAKESPEARE_URL, timeout=30).text
            path.write_text(text, encoding="utf-8")
        return path.read_text(encoding="utf-8")

    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"dataset must be 'tinyshakespeare' or a text file path: {dataset}")
    return path.read_text(encoding="utf-8")


class CharDataset:
    def __init__(self, text: str, split: str = "train") -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.data = data[:n] if split == "train" else data[n:]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def get_batch(self, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - block_size, (batch_size,))
        x = torch.stack([self.data[i : i + block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, train_data: CharDataset, val_data: CharDataset, args) -> dict[str, float]:
    model.eval()
    out = {}
    for split, dataset in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            xb, yb = dataset.get_batch(args.batch_size, args.block_size, args.device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny BitNet-style character model.")
    p.add_argument("--dataset", default="tinyshakespeare")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--linear-kind", choices=["bitlinear", "linear"], default="bitlinear")
    p.add_argument("--activation-bits", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    text = read_text(args.dataset, args.data_dir)
    train_data = CharDataset(text, "train")
    val_data = CharDataset(text, "val")

    config = GPTConfig(
        vocab_size=train_data.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        linear_kind=args.linear_kind,
        activation_bits=args.activation_bits,
    )
    model = GPT(config).to(args.device)
    print(f"params: {model.parameter_count()/1e6:.2f}M | vocab: {config.vocab_size}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            losses = estimate_loss(model, train_data, val_data, args)
            print(
                f"iter {step:04d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = train_data.get_batch(args.batch_size, args.block_size, args.device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    ckpt = {
        "model": model.state_dict(),
        "config": config.to_dict(),
        "stoi": train_data.stoi,
        "itos": train_data.itos,
    }
    ckpt_path = args.out_dir / "nanobitnet.pt"
    torch.save(ckpt, ckpt_path)
    (args.out_dir / "vocab.json").write_text(json.dumps(train_data.stoi, indent=2), encoding="utf-8")
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
