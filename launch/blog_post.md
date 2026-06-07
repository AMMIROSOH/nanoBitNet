# I Implemented BitNet b1.58 From Scratch

BitNet b1.58 is one of those ideas that sounds impossible until you look closely:
use three possible weight values, `{-1, 0, +1}`, and train the model so those
low-bit weights are part of the optimization process rather than a post-training
afterthought.

The point of `nanoBitNet` is to make that idea inspectable.

Microsoft's BitNet project is the right place to look for optimized inference.
This repo is different. It is the readable implementation: a tiny GPT-style
language model, a compact `BitLinear` layer, a training script, a sampler,
benchmarks, and paper notes.

## The Core Trick

The BitNet b1.58 forward pass keeps a latent full-precision weight matrix, then
maps it into ternary values:

```text
W_f = RoundClip(W / mean(abs(W)), -1, 1)
```

Because rounding is not differentiable, training uses a straight-through
estimator. The forward pass sees quantized values. The backward pass updates the
latent full-precision weights.

Activations are quantized too, usually with an 8-bit absmax recipe:

```text
x_q = RoundClip(x * 127 / max(abs(x)), -128, 127)
```

In this repo, the code mirrors those equations directly.

## Why 1.58 Bits?

Binary has two values, so it carries one bit. Ternary has three values, so the
information content is:

```text
log2(3) = 1.58496...
```

That is where "1.58-bit" comes from. It does not mean PyTorch magically stores
the training checkpoint at 1.58 bits per parameter. During training, the latent
weights and optimizer states are still high precision. The low-bit representation
is the quantized forward/inference view.

## What I Learned

The interesting part is not just the memory number. It is the training contract:
you are not taking a normal trained model and crushing it afterward. You are
training the model while it repeatedly sees the low-bit constraint.

That makes `BitLinear` the right unit of study.

## Try It

```bash
pip install -r requirements.txt
python train.py --dataset tinyshakespeare
python sample.py --prompt "The future of AI is"
python bench.py
```

The goal is simple: understand 1.58-bit LLMs in one afternoon.
