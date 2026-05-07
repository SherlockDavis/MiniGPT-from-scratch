"""Hyperparameter configuration for MiniGPT."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 10000
    warmup_iters: int = 500
    grad_clip: float = 1.0
    # Effective batch = batch_size * gradient_accumulation_steps. Default 1
    # leaves behavior identical to step 8/9; raise to fit a larger effective
    # batch on small GPUs.
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        assert self.gradient_accumulation_steps >= 1, (
            f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
        )
