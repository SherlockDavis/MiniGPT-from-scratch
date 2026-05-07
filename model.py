"""GPT model components, built from scratch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, with optional bias."""

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., ndim) — normalize over the last dimension only
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False matches torch.nn.LayerNorm (population variance, /N)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            return self.weight * x_hat + self.bias
        return self.weight * x_hat


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (lower-triangular) mask.

    Hand-written: no nn.MultiheadAttention, no F.scaled_dot_product_attention.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # Combined Q/K/V projection: (C) -> (3C). One matmul is faster than three.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: 1 on/below diagonal, 0 above. Buffer (moves with .to(device), not a Parameter).
        # Shape (1, 1, block_size, block_size) so it broadcasts over (B, n_head, T, T).
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        assert T <= self.mask.shape[-1], f"sequence length {T} exceeds block_size {self.mask.shape[-1]}"

        # (B, T, C) -> (B, T, 3C) -> 3 * (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Split heads: (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product scores: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # Apply causal mask BEFORE softmax: positions above diagonal -> -inf -> 0 after softmax.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted values: (B, n_head, T, head_size)
        y = att @ v

        # Merge heads: (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward block: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, 4C) -> (B, T, C)
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block: Pre-LN residual around attention, then around MLP.

        x = x + Attention(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2-style decoder-only transformer.

    Inputs are token id sequences of shape (B, T). The model returns
    next-token logits of shape (B, T, vocab_size); when `targets` are
    provided it also returns a scalar cross-entropy loss (target = input
    shifted left by one, supplied by the caller).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)        # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)        # learned position embedding (GPT-2 style)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share parameters between input embedding and output projection.
        # Standard GPT-2 trick — saves ~vocab_size * n_embd parameters and tends to help.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 §2.3): std = 0.02 / sqrt(2 * n_layer).
        # Counteracts variance growth through the residual stream as depth increases.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # wte is tied to lm_head, so it is counted once in `parameters()`.
            # Position embedding is unique; subtract it to get the "non-embedding" count.
            n -= self.wpe.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (B, T) int64 token ids
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(T, dtype=torch.long, device=idx.device)  # (T,)
        tok_emb = self.wte(idx)        # (B, T, C)
        pos_emb = self.wpe(pos)        # (T, C) — broadcasts over batch
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)        # (B, T, vocab_size)

        loss: torch.Tensor | None = None
        if targets is not None:
            # Caller supplies targets already shifted (y = x rolled left by one).
            # Flatten to (B*T, V) and (B*T,) for cross_entropy.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss
