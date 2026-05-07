"""Autoregressive sampling from a trained MiniGPT checkpoint.

Supports four decoding modes via the same `generate` function:
  - greedy        (temperature == 0)
  - temperature   (temperature > 0, no top_k / top_p)
  - top-k         (only the k highest-logit tokens are kept)
  - top-p / nucleus  (smallest set whose cumulative prob >= p)

The sampler lives here rather than as a method on `GPT` to keep `model.py`
under the 200-line budget set in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import sys

import torch

from config import GPTConfig
from model import GPT
from utils import get_tokenizer


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask out everything below the k-th highest logit (per row)."""
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Mask out tokens outside the smallest set whose cumulative prob >= top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Tokens whose *cumulative* prob already exceeds top_p are excluded from the
    # nucleus. Shift right by one so we keep the token that crossed the
    # threshold (otherwise top_p<min_prob would mask everything).
    sorted_to_remove = cumulative_probs > top_p
    sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
    sorted_to_remove[..., 0] = False

    # Map the sorted-position mask back to the original vocab order.
    to_remove = sorted_to_remove.scatter(-1, sorted_indices, sorted_to_remove)
    return logits.masked_fill(to_remove, float("-inf"))


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Autoregressively generate `max_new_tokens` tokens after `idx`.

    Args:
        model:          a trained GPT (any train/eval mode — caller's choice).
        idx:            (B, T) int64 prompt token ids on the same device as `model`.
        max_new_tokens: number of new tokens to append to each row.
        temperature:    0.0 → greedy argmax. >0 → divide logits before softmax.
        top_k:          if set, keep only the k highest-logit tokens.
        top_p:          if set, keep the smallest set with cumulative prob >= p.

    Returns:
        (B, T + max_new_tokens) int64 tensor — original prompt concatenated with samples.
    """
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # Crop context: the model only knows positions [0, block_size).
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, V) — distribution for the next token

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = _apply_top_k(logits, top_k)
            if top_p is not None:
                logits = _apply_top_p(logits, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        idx = torch.cat([idx, next_token], dim=1)
    return idx


def load_checkpoint(path: str, device: str) -> GPT:
    """Load a checkpoint saved by `train.save_checkpoint`."""
    # weights_only=False because the checkpoint stores a GPTConfig dataclass too.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config: GPTConfig = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a MiniGPT checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to ckpt_step*.pt")
    p.add_argument("--prompt", default="Once upon a time", help="Text prompt to seed generation.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="0.0 = greedy; higher = more diverse.",
    )
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=None, help="Set torch RNG seed for reproducibility.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = load_checkpoint(args.checkpoint, args.device)
    tokenizer = get_tokenizer()

    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    text = tokenizer.decode(out[0].tolist())
    # User-facing program output, not debug logging — print is correct here.
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
