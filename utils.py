"""Data utilities: tokenizer wrappers, dataset preparation, and batch sampling."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import GPT2Tokenizer

# GPT-2 vocab is 50257; uint16 holds 0..65535 so it fits comfortably.
# Storing tokens as uint16 (2 B) instead of int64 (8 B) cuts the on-disk
# corpus size by 4x and makes memmap I/O proportionally faster.
TOKEN_DTYPE = np.uint16

# TinyStoriesV2 separates stories with this literal string.
STORY_SEPARATOR = "<|endoftext|>"

# GPT-2 reserves id 50256 for the end-of-text token; both the fast and slow
# tokenizers agree on this. Hard-coded so the fast path doesn't need to
# round-trip through `tokenizer.token_to_id("<|endoftext|>")`.
GPT2_EOT_ID = 50256


def get_tokenizer() -> GPT2Tokenizer:
    """Load the slow (Python) GPT-2 BPE tokenizer (vocab_size=50257).

    Convenient for one-off encode/decode (e.g. `generate.py`) but ~10-50x
    slower than the Rust impl. Use `get_fast_tokenizer` for bulk corpus prep.
    Per CLAUDE.md only `GPT2Tokenizer` is permitted from `transformers`.
    """
    return GPT2Tokenizer.from_pretrained("gpt2")


def get_fast_tokenizer() -> Tokenizer:
    """Load the fast (Rust) GPT-2 BPE tokenizer (vocab_size=50257).

    Same vocab + merges as `GPT2Tokenizer`, but the BPE loop runs in Rust
    and exposes parallel `encode_batch`. This is the only practical way to
    tokenize the ~2GB TinyStories train file in minutes instead of hours.
    """
    return Tokenizer.from_pretrained("gpt2")


def prepare_data(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    tokenizer: GPT2Tokenizer | Tokenizer | None = None,
    chunk_size: int = 1024,
    show_progress: bool = False,
) -> int:
    """Encode a TinyStories-style text file into a flat uint16 token stream on disk.

    Stories are split on the literal `<|endoftext|>` marker. Each non-empty story
    is BPE-encoded and a single EOT token id is appended after it, so the model
    sees an explicit story boundary. The id stream is written as raw uint16
    (load it later via `load_tokens`).

    `tokenizer` may be:
      - None             → loads the fast Rust tokenizer (default; recommended)
      - Tokenizer        → fast path: parallel `encode_batch` over `chunk_size` stories
      - GPT2Tokenizer    → slow path: story-by-story (kept for tests / parity checks)

    Returns the total number of tokens written.
    """
    if tokenizer is None:
        tokenizer = get_fast_tokenizer()
    # Distinguish the two tokenizer flavors by capability rather than isinstance,
    # so any future fast-tokenizer wrapper that exposes encode_batch also works.
    is_fast = hasattr(tokenizer, "encode_batch")
    eot_id = GPT2_EOT_ID if is_fast else tokenizer.eos_token_id

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    stories = [s for s in (story.strip() for story in text.split(STORY_SEPARATOR)) if s]
    del text  # free ~2x corpus size before allocating the encoded buffer

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0

    chunk_starts = range(0, len(stories), chunk_size)
    if show_progress:
        chunk_starts = tqdm(
            chunk_starts, desc=f"Tokenizing {Path(input_path).name}", unit="batch"
        )

    # Write tokens chunk-by-chunk so we never materialize the full encoded
    # corpus in Python list form (would be ~14 GB for the 2 GB train file).
    with open(output_path, "wb") as fout:
        for start in chunk_starts:
            batch = stories[start : start + chunk_size]
            buf: list[int] = []
            if is_fast:
                for enc in tokenizer.encode_batch(batch):
                    buf.extend(enc.ids)
                    buf.append(eot_id)
            else:
                for story in batch:
                    buf.extend(tokenizer.encode(story))
                    buf.append(eot_id)
            arr = np.asarray(buf, dtype=TOKEN_DTYPE)
            arr.tofile(fout)
            total += arr.size
    return total


def load_tokens(path: str | os.PathLike) -> np.ndarray:
    """Memory-map a token file produced by `prepare_data` (read-only)."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of contiguous (x, y) pairs from a token array.

    For each example we draw a random start `i` in [0, len(data) - block_size - 1)
    and return:
        x = data[i     : i + block_size    ]   shape (B, T)
        y = data[i + 1 : i + 1 + block_size]   (x shifted left by one)
    Both are int64 tensors moved to `device` (nn.Embedding requires long indices).

    Sampling is with replacement — fine on the TinyStories scale and matches the
    nanoGPT idiom of step-based training without an explicit epoch.
    """
    assert len(data) > block_size + 1, (
        f"data length {len(data)} too short for block_size {block_size}"
    )
    high = len(data) - block_size - 1
    starts = np.random.randint(0, high, size=batch_size)

    x = torch.from_numpy(
        np.stack([data[i : i + block_size].astype(np.int64) for i in starts])
    )
    y = torch.from_numpy(
        np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in starts])
    )

    if str(device).startswith("cuda"):
        # pin_memory + non_blocking is a small win on CUDA; no-op on CPU.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
