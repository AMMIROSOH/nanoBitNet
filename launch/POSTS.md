# Launch Posts

## X / LinkedIn

I built `nanoBitNet`: a minimal PyTorch implementation of BitNet b1.58.

It is not optimized inference like `bitnet.cpp`; it is the readable version.

The repo includes:

- BitLinear
- ternary weight quantization `{-1, 0, +1}`
- int8-like activation quantization
- tiny character-level training
- sampling
- benchmarks
- paper-validated notes

Goal: understand 1.58-bit LLMs in one afternoon.

## Hacker News

Title:

```text
Show HN: nanoBitNet - BitNet b1.58 from scratch in PyTorch
```

Text:

```text
I built nanoBitNet, a small educational implementation of the core BitNet b1.58
idea in PyTorch.

The repo is not an optimized inference framework. Microsoft already has the
official BitNet project for that. This is the readable version: BitLinear,
absmean ternary weights, absmax activations, STE, tiny training, sampling,
benchmarks, and paper notes.

I wrote it for people who want to understand why 1.58-bit means {-1, 0, +1}
and how the training-time latent weights differ from the quantized forward pass.
```

## Reddit

Title:

```text
I implemented BitNet b1.58 from scratch in PyTorch for learning
```

Text:

```text
I made nanoBitNet, a small educational implementation of BitNet b1.58.

It trains a tiny character-level GPT-style model and replaces the attention/MLP
projections with BitLinear:

- latent FP32 weights
- absmean ternary quantization to {-1, 0, +1}
- absmax int8-like activation quantization
- straight-through estimator for training

This is not meant to beat bitnet.cpp or provide packed kernels. The goal is to
make the paper mechanics easy to read, test, and modify.

I included paper notes, diagrams, a benchmark script, and a 30-line BitLinear
example.
```
