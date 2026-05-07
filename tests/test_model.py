
"""Unit tests for model components."""

import pytest
import torch
import torch.nn as nn

from config import GPTConfig
from model import GPT, MLP, Block, CausalSelfAttention, LayerNorm


class TestLayerNorm:
    def test_output_shape_matches_input(self) -> None:
        ln = LayerNorm(ndim=384)
        x = torch.randn(2, 16, 384)  # (B, T, C)
        out = ln(x)
        assert out.shape == x.shape

    def test_normalized_statistics(self) -> None:
        # With default weight=1, bias=0, the output along the last dim
        # should have mean ~0 and variance ~1.
        ln = LayerNorm(ndim=64)
        x = torch.randn(4, 10, 64) * 5.0 + 3.0  # arbitrary scale & shift
        out = ln(x)
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)

    def test_matches_torch_layernorm(self) -> None:
        # Numerical agreement with the reference implementation.
        torch.manual_seed(0)
        ndim = 128
        ours = LayerNorm(ndim, bias=True)
        ref = nn.LayerNorm(ndim, elementwise_affine=True)
        # Copy the reference's learned weights into ours so we compare math, not init.
        ours.weight.data.copy_(ref.weight.data)
        ours.bias.data.copy_(ref.bias.data)

        x = torch.randn(3, 7, ndim)
        assert torch.allclose(ours(x), ref(x), atol=1e-6)

    def test_bias_false_has_no_bias_param(self) -> None:
        ln = LayerNorm(ndim=32, bias=False)
        assert ln.bias is None
        # Forward still works.
        x = torch.randn(1, 4, 32)
        assert ln(x).shape == x.shape

    def test_gradients_flow(self) -> None:
        ln = LayerNorm(ndim=16)
        x = torch.randn(2, 5, 16, requires_grad=True)
        loss = ln(x).sum()
        loss.backward()
        assert x.grad is not None and x.grad.shape == x.shape
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None


@pytest.fixture
def small_config() -> GPTConfig:
    # Small but realistic: 4 heads, head_size=16, short context. Dropout=0 for determinism.
    return GPTConfig(n_embd=64, n_head=4, block_size=16, dropout=0.0)


class TestCausalSelfAttention:
    def test_output_shape(self, small_config: GPTConfig) -> None:
        attn = CausalSelfAttention(small_config)
        x = torch.randn(2, 8, small_config.n_embd)  # (B, T, C)
        assert attn(x).shape == x.shape

    def test_mask_is_lower_triangular(self, small_config: GPTConfig) -> None:
        attn = CausalSelfAttention(small_config)
        mask = attn.mask.squeeze()  # (block_size, block_size)
        expected = torch.tril(torch.ones(small_config.block_size, small_config.block_size))
        assert torch.equal(mask, expected)
        # Mask is a buffer, not a parameter.
        assert "mask" in dict(attn.named_buffers())
        assert "mask" not in dict(attn.named_parameters())

    def test_causal_property(self, small_config: GPTConfig) -> None:
        """Output at position t must not depend on inputs at positions > t."""
        torch.manual_seed(0)
        attn = CausalSelfAttention(small_config).eval()
        B, T, C = 1, 8, small_config.n_embd

        x1 = torch.randn(B, T, C)
        x2 = x1.clone()
        # Perturb only positions [t+1:]
        t = 3
        x2[:, t + 1 :, :] = torch.randn(B, T - t - 1, C)

        with torch.no_grad():
            y1 = attn(x1)
            y2 = attn(x2)

        # Outputs at [0:t+1] must be identical — no leakage from future tokens.
        assert torch.allclose(y1[:, : t + 1, :], y2[:, : t + 1, :], atol=1e-6)
        # Sanity: outputs at [t+1:] *do* differ (otherwise the test is vacuous).
        assert not torch.allclose(y1[:, t + 1 :, :], y2[:, t + 1 :, :])

    def test_attention_weights_sum_to_one_and_are_causal(self, small_config: GPTConfig) -> None:
        """Probe softmax output directly: rows sum to 1, upper triangle is exactly 0."""
        torch.manual_seed(0)
        attn = CausalSelfAttention(small_config).eval()
        T = 8
        x = torch.randn(1, T, small_config.n_embd)

        # Recompute scores manually to inspect post-softmax weights.
        with torch.no_grad():
            q, k, _ = attn.c_attn(x).split(attn.n_embd, dim=2)
            B = x.size(0)
            q = q.view(B, T, attn.n_head, attn.head_size).transpose(1, 2)
            k = k.view(B, T, attn.n_head, attn.head_size).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (attn.head_size ** 0.5)
            scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1)  # (B, n_head, T, T)

        # Each row sums to 1.
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, attn.n_head, T), atol=1e-6)
        # Strict upper triangle is exactly 0.
        upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
        assert (weights[..., upper] == 0).all()

    def test_short_sequence_below_block_size(self, small_config: GPTConfig) -> None:
        attn = CausalSelfAttention(small_config)
        x = torch.randn(2, 5, small_config.n_embd)  # T=5 < block_size=16
        assert attn(x).shape == x.shape

    def test_sequence_longer_than_block_size_raises(self, small_config: GPTConfig) -> None:
        attn = CausalSelfAttention(small_config)
        x = torch.randn(1, small_config.block_size + 1, small_config.n_embd)
        with pytest.raises(AssertionError, match="exceeds block_size"):
            attn(x)

    def test_gradients_flow(self, small_config: GPTConfig) -> None:
        attn = CausalSelfAttention(small_config)
        x = torch.randn(2, 8, small_config.n_embd, requires_grad=True)
        loss = attn(x).sum()
        loss.backward()
        assert x.grad is not None
        for name, p in attn.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"


class TestMLP:
    def test_output_shape(self, small_config: GPTConfig) -> None:
        mlp = MLP(small_config)
        x = torch.randn(2, 8, small_config.n_embd)  # (B, T, C)
        assert mlp(x).shape == x.shape

    def test_hidden_dim_is_4x(self, small_config: GPTConfig) -> None:
        # GPT-style 4x expansion ratio.
        mlp = MLP(small_config)
        assert mlp.c_fc.in_features == small_config.n_embd
        assert mlp.c_fc.out_features == 4 * small_config.n_embd
        assert mlp.c_proj.in_features == 4 * small_config.n_embd
        assert mlp.c_proj.out_features == small_config.n_embd

    def test_position_wise_independence(self, small_config: GPTConfig) -> None:
        """MLP applies the same transform per token; perturbing position t must not
        change the output at any other position."""
        mlp = MLP(small_config).eval()
        B, T, C = 1, 6, small_config.n_embd
        x1 = torch.randn(B, T, C)
        x2 = x1.clone()
        t = 2
        x2[:, t, :] = torch.randn(C)
        with torch.no_grad():
            y1 = mlp(x1)
            y2 = mlp(x2)
        # All positions except t should be identical.
        mask = torch.ones(T, dtype=torch.bool)
        mask[t] = False
        assert torch.allclose(y1[:, mask, :], y2[:, mask, :], atol=1e-6)
        # Sanity: position t itself differs.
        assert not torch.allclose(y1[:, t, :], y2[:, t, :])

    def test_gradients_flow(self, small_config: GPTConfig) -> None:
        mlp = MLP(small_config)
        x = torch.randn(2, 8, small_config.n_embd, requires_grad=True)
        loss = mlp(x).sum()
        loss.backward()
        assert x.grad is not None
        for name, p in mlp.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"


class TestBlock:
    def test_output_shape(self, small_config: GPTConfig) -> None:
        block = Block(small_config)
        x = torch.randn(2, 8, small_config.n_embd)  # (B, T, C)
        assert block(x).shape == x.shape

    def test_has_pre_ln_submodules(self, small_config: GPTConfig) -> None:
        # Pre-LN structure: two LayerNorms, one attention, one MLP.
        block = Block(small_config)
        assert isinstance(block.ln_1, LayerNorm)
        assert isinstance(block.ln_2, LayerNorm)
        assert isinstance(block.attn, CausalSelfAttention)
        assert isinstance(block.mlp, MLP)

    def test_residual_connection(self, small_config: GPTConfig) -> None:
        """If attn and mlp output ~0, the block is the identity (residual short-circuits).

        We zero out the output projections of both sub-blocks so attn(x)=0 and mlp(x)=0.
        Then Block(x) must equal x exactly (only the residual paths remain).
        """
        block = Block(small_config).eval()
        with torch.no_grad():
            block.attn.c_proj.weight.zero_()
            block.attn.c_proj.bias.zero_()
            block.mlp.c_proj.weight.zero_()
            block.mlp.c_proj.bias.zero_()

        x = torch.randn(2, 8, small_config.n_embd)
        with torch.no_grad():
            y = block(x)
        assert torch.allclose(y, x, atol=1e-6)

    def test_causal_property(self, small_config: GPTConfig) -> None:
        """Block must inherit the causal property from its attention layer."""
        torch.manual_seed(0)
        block = Block(small_config).eval()
        B, T, C = 1, 8, small_config.n_embd

        x1 = torch.randn(B, T, C)
        x2 = x1.clone()
        t = 3
        x2[:, t + 1 :, :] = torch.randn(B, T - t - 1, C)

        with torch.no_grad():
            y1 = block(x1)
            y2 = block(x2)

        # Outputs at [0:t+1] must be identical — no leakage from future tokens.
        assert torch.allclose(y1[:, : t + 1, :], y2[:, : t + 1, :], atol=1e-6)
        assert not torch.allclose(y1[:, t + 1 :, :], y2[:, t + 1 :, :])

    def test_gradients_flow(self, small_config: GPTConfig) -> None:
        block = Block(small_config)
        x = torch.randn(2, 8, small_config.n_embd, requires_grad=True)
        loss = block(x).sum()
        loss.backward()
        assert x.grad is not None
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"


@pytest.fixture
def tiny_gpt_config() -> GPTConfig:
    # Toy config so the full model is cheap to instantiate in tests.
    # vocab_size=128 keeps the embedding table small but still realistic.
    return GPTConfig(
        vocab_size=128, block_size=16, n_layer=2, n_head=4, n_embd=64, dropout=0.0
    )


class TestGPT:
    def test_forward_logits_shape(self, tiny_gpt_config: GPTConfig) -> None:
        model = GPT(tiny_gpt_config)
        idx = torch.randint(0, tiny_gpt_config.vocab_size, (2, 8))
        logits, loss = model(idx)
        assert logits.shape == (2, 8, tiny_gpt_config.vocab_size)
        assert loss is None

    def test_forward_with_targets_returns_scalar_loss(self, tiny_gpt_config: GPTConfig) -> None:
        model = GPT(tiny_gpt_config)
        idx = torch.randint(0, tiny_gpt_config.vocab_size, (2, 8))
        targets = torch.randint(0, tiny_gpt_config.vocab_size, (2, 8))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 8, tiny_gpt_config.vocab_size)
        assert loss is not None and loss.ndim == 0
        # Untrained model on uniform targets should give loss near ln(V).
        # Allow generous slack since init has randomness.
        expected = torch.log(torch.tensor(float(tiny_gpt_config.vocab_size)))
        assert abs(loss.item() - expected.item()) < 1.0

    def test_weight_tying(self, tiny_gpt_config: GPTConfig) -> None:
        # lm_head.weight must be the *same* tensor object as wte.weight.
        model = GPT(tiny_gpt_config)
        assert model.lm_head.weight is model.wte.weight

    def test_sequence_longer_than_block_size_raises(self, tiny_gpt_config: GPTConfig) -> None:
        model = GPT(tiny_gpt_config)
        idx = torch.randint(0, tiny_gpt_config.vocab_size, (1, tiny_gpt_config.block_size + 1))
        with pytest.raises(AssertionError, match="exceeds block_size"):
            model(idx)

    def test_causal_property_end_to_end(self, tiny_gpt_config: GPTConfig) -> None:
        """Logits at position t must not depend on input tokens at positions > t."""
        torch.manual_seed(0)
        model = GPT(tiny_gpt_config).eval()
        T = 8
        idx1 = torch.randint(0, tiny_gpt_config.vocab_size, (1, T))
        idx2 = idx1.clone()
        t = 3
        # Replace tokens at positions [t+1:] with arbitrary other ids.
        idx2[:, t + 1 :] = (idx1[:, t + 1 :] + 7) % tiny_gpt_config.vocab_size

        with torch.no_grad():
            logits1, _ = model(idx1)
            logits2, _ = model(idx2)

        assert torch.allclose(logits1[:, : t + 1, :], logits2[:, : t + 1, :], atol=1e-6)
        assert not torch.allclose(logits1[:, t + 1 :, :], logits2[:, t + 1 :, :])

    def test_gradients_flow(self, tiny_gpt_config: GPTConfig) -> None:
        model = GPT(tiny_gpt_config)
        idx = torch.randint(0, tiny_gpt_config.vocab_size, (2, 8))
        targets = torch.randint(0, tiny_gpt_config.vocab_size, (2, 8))
        _, loss = model(idx, targets)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"

    def test_full_config_param_count_in_expected_range(self) -> None:
        """Sanity-check the full-size model is in the documented ballpark.

        With the project's default GPTConfig (vocab=50257, n_embd=384, n_layer=6,
        block=256) and weight tying, the model should land around 10–35M parameters.
        The exact target documented in CLAUDE.md is ~15M; with vocab_size=50257 the
        token embedding alone is ~19M, so the realistic total is ~30M.
        """
        model = GPT(GPTConfig())
        n = model.num_parameters()
        assert 10_000_000 < n < 40_000_000, f"unexpected param count: {n:,}"
