"""Integration tests for the training loop primitives.

We don't run train.py end-to-end here (that needs the real corpus). Instead
we exercise the train_step function on a tiny model + synthetic data and
check the basic invariants required by workflow step 8: loss decreases,
no NaN, gradients flow into every parameter.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import GPTConfig
from model import GPT
from train import estimate_loss, get_lr, train_step
from utils import TOKEN_DTYPE, get_batch


@pytest.fixture
def tiny_config() -> GPTConfig:
    # Small enough to train quickly on CPU; large enough that loss can move.
    return GPTConfig(
        vocab_size=128, block_size=16, n_layer=2, n_head=4, n_embd=64, dropout=0.0,
        batch_size=8, learning_rate=1e-3,
    )


@pytest.fixture
def fake_corpus(tiny_config: GPTConfig) -> np.ndarray:
    # Learnable signal: tokens cycle 0,1,2,...,V-1,0,1,... so each next-token is
    # perfectly predictable from the previous one. A purely random corpus has no
    # structure to learn (loss is bounded below by ln(V)), which would make
    # "loss decreases" a vacuous assertion.
    return (np.arange(4000) % tiny_config.vocab_size).astype(TOKEN_DTYPE)


def test_train_step_returns_finite_scalar(tiny_config, fake_corpus) -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config)
    opt = torch.optim.AdamW(model.parameters(), lr=tiny_config.learning_rate)
    x, y = get_batch(fake_corpus, tiny_config.block_size, tiny_config.batch_size)
    loss = train_step(model, opt, x, y, grad_clip=tiny_config.grad_clip)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_loss_decreases_over_50_steps(tiny_config, fake_corpus) -> None:
    """Step 8 acceptance criterion: loss drops cleanly with no NaN."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = GPT(tiny_config)
    opt = torch.optim.AdamW(model.parameters(), lr=tiny_config.learning_rate)

    losses: list[float] = []
    for _ in range(50):
        x, y = get_batch(fake_corpus, tiny_config.block_size, tiny_config.batch_size)
        loss = train_step(model, opt, x, y, grad_clip=tiny_config.grad_clip)
        assert np.isfinite(loss), f"non-finite loss: {loss}"
        losses.append(loss)

    # Average of last 10 must beat average of first 10 — robust to per-step noise.
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < early - 0.1, f"loss not decreasing: early={early:.3f}, late={late:.3f}"


def test_gradients_flow_for_all_parameters(tiny_config, fake_corpus) -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config)
    opt = torch.optim.AdamW(model.parameters(), lr=tiny_config.learning_rate)
    x, y = get_batch(fake_corpus, tiny_config.block_size, tiny_config.batch_size)
    train_step(model, opt, x, y, grad_clip=tiny_config.grad_clip)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} missing gradient"


def test_estimate_loss_returns_train_and_val(tiny_config, fake_corpus) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    model = GPT(tiny_config)
    # Use the same corpus for both splits — we're checking shape, not generalization.
    out = estimate_loss(
        model, fake_corpus, fake_corpus,
        block_size=tiny_config.block_size,
        batch_size=tiny_config.batch_size,
        device="cpu",
        eval_iters=3,
    )
    assert set(out.keys()) == {"train", "val"}
    for v in out.values():
        assert np.isfinite(v)


def test_estimate_loss_does_not_leave_model_in_eval_mode(tiny_config, fake_corpus) -> None:
    # Important: estimate_loss flips to eval() but must restore train() so the next
    # training step still has dropout etc. enabled.
    model = GPT(tiny_config)
    model.train()
    estimate_loss(
        model, fake_corpus, fake_corpus,
        block_size=tiny_config.block_size,
        batch_size=tiny_config.batch_size,
        device="cpu",
        eval_iters=2,
    )
    assert model.training is True


class TestGetLr:
    PEAK = 3e-4
    MIN = 3e-5
    WARM = 100
    MAX = 1000

    def test_warmup_grows_monotonically_to_peak(self) -> None:
        prev = -1.0
        for s in range(self.WARM):
            lr = get_lr(s, self.WARM, self.MAX, self.PEAK, self.MIN)
            assert lr > prev, f"warmup not monotonic at step {s}: {lr} <= {prev}"
            prev = lr
        # Last warmup step (warmup_iters - 1) reaches peak exactly.
        assert get_lr(self.WARM - 1, self.WARM, self.MAX, self.PEAK, self.MIN) == pytest.approx(
            self.PEAK
        )

    def test_first_warmup_step_is_small_but_positive(self) -> None:
        lr0 = get_lr(0, self.WARM, self.MAX, self.PEAK, self.MIN)
        # peak * 1/warm  ==  PEAK / 100 == 3e-6
        assert lr0 == pytest.approx(self.PEAK / self.WARM)
        assert lr0 > 0

    def test_at_warmup_end_we_are_at_peak(self) -> None:
        # Step exactly equal to warmup_iters: cosine ratio = 0 → peak.
        assert get_lr(self.WARM, self.WARM, self.MAX, self.PEAK, self.MIN) == pytest.approx(
            self.PEAK
        )

    def test_cosine_decay_reaches_min_at_max_iters(self) -> None:
        assert get_lr(self.MAX, self.WARM, self.MAX, self.PEAK, self.MIN) == pytest.approx(
            self.MIN
        )

    def test_post_max_iters_clamped_at_min(self) -> None:
        # Schedule should not go below min_lr if we accidentally over-run.
        assert get_lr(self.MAX + 50, self.WARM, self.MAX, self.PEAK, self.MIN) == self.MIN

    def test_decay_is_monotonically_decreasing(self) -> None:
        prev = float("inf")
        for s in range(self.WARM, self.MAX + 1, 50):
            lr = get_lr(s, self.WARM, self.MAX, self.PEAK, self.MIN)
            assert lr <= prev, f"decay not monotonic at step {s}: {lr} > {prev}"
            prev = lr

    def test_midpoint_between_min_and_peak(self) -> None:
        # Halfway through cosine decay: cos(π/2) = 0 → coeff=0.5 → midway value.
        mid = (self.WARM + self.MAX) // 2
        lr = get_lr(mid, self.WARM, self.MAX, self.PEAK, self.MIN)
        expected = self.MIN + 0.5 * (self.PEAK - self.MIN)
        assert lr == pytest.approx(expected, rel=1e-3)
