"""Unit tests for the autoregressive sampler in generate.py.

We use a tiny untrained model — these tests check the *sampler mechanics*
(shape, determinism, masking, context cropping), not generation quality.
"""

from __future__ import annotations

import pytest
import torch

from config import GPTConfig
from generate import _apply_top_k, _apply_top_p, generate
from model import GPT


@pytest.fixture
def tiny_gpt_config() -> GPTConfig:
    # Mirrors the test_model.py shape so tests stay fast on CPU.
    return GPTConfig(
        vocab_size=128, block_size=16, n_layer=2, n_head=4, n_embd=64, dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_gpt_config: GPTConfig) -> GPT:
    torch.manual_seed(0)
    model = GPT(tiny_gpt_config)
    model.eval()
    return model


def _prompt(B: int = 1, T: int = 4, vocab: int = 128) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randint(0, vocab, (B, T), dtype=torch.long)


class TestGenerateShape:
    def test_appends_exactly_max_new_tokens(self, tiny_model: GPT) -> None:
        idx = _prompt(B=2, T=4)
        out = generate(tiny_model, idx, max_new_tokens=7, temperature=1.0)
        assert out.shape == (2, 4 + 7)

    def test_output_ids_are_in_vocab_range(
        self, tiny_model: GPT, tiny_gpt_config: GPTConfig
    ) -> None:
        idx = _prompt(B=1, T=3)
        out = generate(tiny_model, idx, max_new_tokens=10, temperature=1.0)
        assert out.dtype == torch.long
        assert out.min().item() >= 0
        assert out.max().item() < tiny_gpt_config.vocab_size

    def test_prompt_prefix_is_preserved(self, tiny_model: GPT) -> None:
        idx = _prompt(B=1, T=5)
        out = generate(tiny_model, idx, max_new_tokens=4, temperature=1.0)
        assert torch.equal(out[:, :5], idx)


class TestGenerateModes:
    def test_greedy_is_deterministic(self, tiny_model: GPT) -> None:
        # temperature=0 → argmax; two runs must produce identical sequences.
        idx = _prompt()
        a = generate(tiny_model, idx, max_new_tokens=8, temperature=0.0)
        b = generate(tiny_model, idx, max_new_tokens=8, temperature=0.0)
        assert torch.equal(a, b)

    def test_sampling_is_stochastic(self, tiny_model: GPT) -> None:
        # With temperature>0 and no top_k, two different RNG seeds should
        # produce different continuations (extremely high probability for
        # 16 sampled tokens out of vocab=128).
        idx = _prompt()
        torch.manual_seed(1)
        a = generate(tiny_model, idx, max_new_tokens=16, temperature=1.0)
        torch.manual_seed(2)
        b = generate(tiny_model, idx, max_new_tokens=16, temperature=1.0)
        assert not torch.equal(a, b)

    def test_top_k_1_matches_greedy(self, tiny_model: GPT) -> None:
        # top_k=1 collapses the distribution to a single token → equivalent
        # to argmax up to temperature scaling (which is irrelevant when
        # only one logit survives).
        idx = _prompt()
        greedy = generate(tiny_model, idx, max_new_tokens=8, temperature=0.0)
        torch.manual_seed(0)
        topk1 = generate(tiny_model, idx, max_new_tokens=8, temperature=1.0, top_k=1)
        assert torch.equal(greedy, topk1)


class TestContextCropping:
    def test_runs_past_block_size(
        self, tiny_model: GPT, tiny_gpt_config: GPTConfig
    ) -> None:
        # Prompt longer than block_size: generate must crop the input
        # window each step rather than crash with a position-embedding OOB.
        block = tiny_gpt_config.block_size
        long_prompt = torch.randint(0, tiny_gpt_config.vocab_size, (1, block + 5), dtype=torch.long)
        out = generate(tiny_model, long_prompt, max_new_tokens=4, temperature=0.0)
        assert out.shape == (1, block + 5 + 4)


class TestTopKMask:
    def test_keeps_only_k_finite_logits(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
        masked = _apply_top_k(logits, top_k=2)
        # Only the two largest survive; the rest are -inf.
        finite = torch.isfinite(masked)
        assert finite.sum().item() == 2
        # The surviving positions are exactly the top-2 indices (1 and 3).
        assert finite[0, 1].item() and finite[0, 3].item()

    def test_top_k_larger_than_vocab_is_a_noop(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        masked = _apply_top_k(logits, top_k=99)
        assert torch.equal(masked, logits)


class TestTopPMask:
    def test_always_keeps_at_least_the_argmax(self) -> None:
        # Even with top_p smaller than every individual probability, the
        # shift-by-one in _apply_top_p must keep the most likely token —
        # otherwise softmax over all -inf would NaN out multinomial.
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
        masked = _apply_top_p(logits, top_p=1e-9)
        # Position 0 has overwhelming probability and must survive.
        assert torch.isfinite(masked[0, 0])
        # And softmax must produce a usable distribution (no NaN).
        probs = torch.softmax(masked, dim=-1)
        assert torch.isfinite(probs).all()
        assert pytest.approx(probs.sum().item(), rel=1e-5) == 1.0

    def test_top_p_1_keeps_everything(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        masked = _apply_top_p(logits, top_p=1.0)
        assert torch.isfinite(masked).all()
