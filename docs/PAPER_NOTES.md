# Paper Notes

These notes are the implementation contract for `nanoBitNet`. They separate
paper-faithful ideas from educational simplifications.

## Validated Facts

1. BitNet replaces Transformer matrix multiplications with `BitLinear`.
   The JMLR BitNet paper describes `BitLinear` as a drop-in replacement for
   `nn.Linear` when training low-bit weights from scratch.

2. BitNet b1.58 uses ternary weights.
   The b1.58 paper states that every weight is ternary: `{-1, 0, +1}`.
   The information content is `log2(3) ~= 1.585` bits per weight.

3. Ternary weights use absmean quantization.
   The JMLR paper gives the b1.58 recipe as:

   ```text
   W_f = RoundClip(W / beta, -1, 1)
   beta = mean(abs(W))
   ```

4. Activations use absmax quantization.
   The JMLR paper uses b-bit activation quantization, with `b = 8` in the main
   recipe, by scaling with the absolute maximum of the activation.

5. STE is needed for training.
   Rounding and clipping are non-differentiable, so the paper uses a
   straight-through estimator during backpropagation.

6. Training keeps high-precision latent weights.
   The low-bit values are used in the forward pass; optimizer states, gradients,
   and latent parameters stay high precision for stability.

## What This Repo Implements

- Ternary absmean weight quantization in `quantize_weight_absmean`.
- Absmax activation quantization in `quantize_activation_absmax`.
- STE in `_ste` and `round_clip`.
- A `BitLinear` module that can replace ordinary projection layers.
- A tiny GPT-style language model whose attention and MLP projections use
  `BitLinear`.

## Educational Simplifications

- No packed ternary storage format. PyTorch tensors store the latent weights.
- No custom ternary GEMM kernels. The forward pass uses `torch.nn.functional.linear`.
- No model-parallel group quantization.
- No production tokenizer. Training is character-level to keep the repo readable.
- No claim that this is an official BitNet reproduction or optimized inference stack.

## Primary Sources

- JMLR BitNet paper: https://jmlr.org/papers/v26/24-2050.html
- BitNet b1.58 paper: https://arxiv.org/abs/2402.17764
- BitNet b1.58 2B4T technical report: https://arxiv.org/abs/2504.12285
- Microsoft BitNet official repo: https://github.com/microsoft/BitNet
- PyTorch `linear`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
