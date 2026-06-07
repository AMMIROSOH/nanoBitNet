import torch

from nanobitnet import BitLinear, GPT, GPTConfig, quantize_activation_absmax, ternary_codes


def test_ternary_codes_are_only_minus_zero_plus_one():
    weight = torch.tensor([[-3.0, -0.2, 0.0, 0.2, 3.0]])
    q = ternary_codes(weight)
    assert set(q.flatten().tolist()).issubset({-1.0, 0.0, 1.0})


def test_activation_quantization_preserves_shape_and_gradients():
    x = torch.randn(2, 3, 5, requires_grad=True)
    y = quantize_activation_absmax(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_bitlinear_forward_backward():
    layer = BitLinear(8, 4)
    x = torch.randn(2, 3, 8)
    y = layer(x)
    assert y.shape == (2, 3, 4)
    y.square().mean().backward()
    assert layer.weight.grad is not None


def test_gpt_forward_and_generate():
    config = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(config)
    idx = torch.randint(0, 16, (2, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 8, 16)
    assert loss is not None
    out = model.generate(idx[:, :2], max_new_tokens=3)
    assert out.shape == (2, 5)
