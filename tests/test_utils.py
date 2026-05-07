"""Unit tests for data utilities."""

from pathlib import Path

import numpy as np
import pytest
import torch

from utils import (
    GPT2_EOT_ID,
    STORY_SEPARATOR,
    TOKEN_DTYPE,
    get_batch,
    get_fast_tokenizer,
    get_tokenizer,
    load_tokens,
    prepare_data,
)


# Cache the tokenizers across tests — loading is the slow part.
@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


@pytest.fixture(scope="module")
def fast_tokenizer():
    return get_fast_tokenizer()


class TestTokenizer:
    def test_vocab_size_is_50257(self, tokenizer) -> None:
        assert tokenizer.vocab_size == 50257

    def test_eos_token_id_is_50256(self, tokenizer) -> None:
        # GPT-2 reserves id 50256 for `<|endoftext|>`.
        assert tokenizer.eos_token_id == 50256

    def test_roundtrip_is_lossless(self, tokenizer) -> None:
        text = "Once upon a time, there was a little girl who loved cookies."
        assert tokenizer.decode(tokenizer.encode(text)) == text


class TestPrepareData:
    def test_writes_uint16_file(self, tmp_path: Path, tokenizer) -> None:
        inp = tmp_path / "raw.txt"
        out = tmp_path / "out.bin"
        inp.write_text(
            f"Hello world.{STORY_SEPARATOR}Goodbye world.{STORY_SEPARATOR}",
            encoding="utf-8",
        )

        n = prepare_data(inp, out, tokenizer=tokenizer)
        assert out.exists()
        assert n > 0

        arr = load_tokens(out)
        assert arr.dtype == TOKEN_DTYPE
        assert arr.size == n

    def test_eot_appended_per_story(self, tmp_path: Path, tokenizer) -> None:
        inp = tmp_path / "raw.txt"
        out = tmp_path / "out.bin"
        inp.write_text(
            f"Story one.{STORY_SEPARATOR}Story two.{STORY_SEPARATOR}Story three.{STORY_SEPARATOR}",
            encoding="utf-8",
        )

        prepare_data(inp, out, tokenizer=tokenizer)
        arr = load_tokens(out)
        # Exactly one EOT per non-empty story.
        assert int((arr == tokenizer.eos_token_id).sum()) == 3

    def test_skips_empty_stories(self, tmp_path: Path, tokenizer) -> None:
        inp = tmp_path / "raw.txt"
        out = tmp_path / "out.bin"
        # Adjacent separators and surrounding whitespace should not yield extra EOTs.
        inp.write_text(
            f"{STORY_SEPARATOR}{STORY_SEPARATOR}  {STORY_SEPARATOR}Real story.{STORY_SEPARATOR}",
            encoding="utf-8",
        )

        prepare_data(inp, out, tokenizer=tokenizer)
        arr = load_tokens(out)
        assert int((arr == tokenizer.eos_token_id).sum()) == 1

    def test_creates_output_parent_dir(self, tmp_path: Path, tokenizer) -> None:
        inp = tmp_path / "raw.txt"
        out = tmp_path / "nested" / "subdir" / "out.bin"
        inp.write_text(f"A story.{STORY_SEPARATOR}", encoding="utf-8")

        prepare_data(inp, out, tokenizer=tokenizer)
        assert out.exists()


class TestFastTokenizer:
    def test_vocab_size_is_50257(self, fast_tokenizer) -> None:
        # `tokenizers.Tokenizer.get_vocab_size()` reports 50257 for gpt2 —
        # same as the slow tokenizer.
        assert fast_tokenizer.get_vocab_size() == 50257

    def test_eot_id_constant_matches_tokenizer(self, fast_tokenizer) -> None:
        # The hard-coded GPT2_EOT_ID must match the actual token id, otherwise
        # prepare_data writes a wrong EOT marker.
        assert fast_tokenizer.token_to_id("<|endoftext|>") == GPT2_EOT_ID

    def test_fast_and_slow_agree_on_sample(self, tokenizer, fast_tokenizer) -> None:
        # Same BPE vocab + merges → identical ids. If this drifts, prepare_data's
        # output would no longer be interchangeable between paths.
        text = "Once upon a time, there was a little girl who loved cookies."
        slow_ids = tokenizer.encode(text)
        fast_ids = fast_tokenizer.encode(text).ids
        assert slow_ids == fast_ids

    def test_prepare_data_fast_path_writes_correct_eot(
        self, tmp_path: Path, fast_tokenizer
    ) -> None:
        inp = tmp_path / "raw.txt"
        out = tmp_path / "out.bin"
        inp.write_text(
            f"Story one.{STORY_SEPARATOR}Story two.{STORY_SEPARATOR}",
            encoding="utf-8",
        )
        n = prepare_data(inp, out, tokenizer=fast_tokenizer)
        arr = load_tokens(out)
        assert arr.size == n
        assert int((arr == GPT2_EOT_ID).sum()) == 2

    def test_fast_and_slow_paths_produce_identical_output(
        self, tmp_path: Path, tokenizer, fast_tokenizer
    ) -> None:
        # End-to-end parity: the same corpus through the two paths must yield
        # byte-identical .bin files (up to chunking, which doesn't change tokens).
        inp = tmp_path / "raw.txt"
        inp.write_text(
            f"The cat sat on the mat.{STORY_SEPARATOR}"
            f"A small dog barked twice.{STORY_SEPARATOR}"
            f"Snow fell on the rooftops.{STORY_SEPARATOR}",
            encoding="utf-8",
        )
        out_slow = tmp_path / "slow.bin"
        out_fast = tmp_path / "fast.bin"
        prepare_data(inp, out_slow, tokenizer=tokenizer)
        prepare_data(inp, out_fast, tokenizer=fast_tokenizer)
        assert np.array_equal(load_tokens(out_slow), load_tokens(out_fast))

    def test_default_tokenizer_uses_fast_path(self, tmp_path: Path) -> None:
        # Default (tokenizer=None) loads the fast tokenizer internally — this
        # is what train.py relies on.
        inp = tmp_path / "raw.txt"
        out = tmp_path / "out.bin"
        inp.write_text(f"Hello world.{STORY_SEPARATOR}", encoding="utf-8")
        n = prepare_data(inp, out)  # no tokenizer arg
        assert n > 0
        arr = load_tokens(out)
        assert arr[-1] == GPT2_EOT_ID


@pytest.fixture
def fake_tokens() -> np.ndarray:
    # Deterministic small array: arange makes it easy to assert contiguity.
    return np.arange(1000, dtype=TOKEN_DTYPE)


class TestGetBatch:
    def test_shapes_and_dtype(self, fake_tokens: np.ndarray) -> None:
        np.random.seed(0)
        x, y = get_batch(fake_tokens, block_size=16, batch_size=4, device="cpu")
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

    def test_y_is_x_shifted_left_by_one(self, fake_tokens: np.ndarray) -> None:
        np.random.seed(0)
        x, y = get_batch(fake_tokens, block_size=16, batch_size=8, device="cpu")
        # The shift property is the whole point of next-token prediction targets.
        assert torch.equal(y[:, :-1], x[:, 1:])

    def test_each_row_is_contiguous_slice(self, fake_tokens: np.ndarray) -> None:
        # fake_tokens is arange, so any contiguous window has step-1 differences.
        np.random.seed(0)
        x, _ = get_batch(fake_tokens, block_size=8, batch_size=4, device="cpu")
        diffs = x[:, 1:] - x[:, :-1]
        assert torch.equal(diffs, torch.ones_like(diffs))

    def test_indices_in_range(self, fake_tokens: np.ndarray) -> None:
        np.random.seed(0)
        x, y = get_batch(fake_tokens, block_size=32, batch_size=16, device="cpu")
        assert int(x.min()) >= 0
        assert int(x.max()) < len(fake_tokens)
        assert int(y.min()) >= 0
        assert int(y.max()) < len(fake_tokens)

    def test_data_too_short_raises(self, fake_tokens: np.ndarray) -> None:
        with pytest.raises(AssertionError, match="too short"):
            get_batch(fake_tokens[:5], block_size=16, batch_size=4)

    def test_works_with_memmap(self, tmp_path: Path, tokenizer) -> None:
        # End-to-end: prepare_data -> load_tokens -> get_batch should produce
        # tensors with the correct shift property even on a real memmap.
        inp = tmp_path / "raw.txt"
        out = tmp_path / "out.bin"
        long_story = "The cat sat on the mat. " * 200
        inp.write_text(f"{long_story}{STORY_SEPARATOR}", encoding="utf-8")

        prepare_data(inp, out, tokenizer=tokenizer)
        data = load_tokens(out)

        np.random.seed(0)
        x, y = get_batch(data, block_size=16, batch_size=4, device="cpu")
        assert x.shape == (4, 16)
        assert torch.equal(y[:, :-1], x[:, 1:])
