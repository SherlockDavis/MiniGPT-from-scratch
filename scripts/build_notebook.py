"""Build MiniGPT.ipynb (a Colab-friendly notebook) from a structured cell list.

Run: `python scripts/build_notebook.py`

The notebook clones the repo into Colab, installs deps, runs a smoke test,
then optionally runs full training and generation. Each section is a small
cell so the user can re-run individual steps without re-executing the whole
notebook.

Re-run this script any time the workflow changes — the .ipynb is generated,
not hand-edited (so don't edit MiniGPT.ipynb directly).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path("MiniGPT.ipynb")
REPO_URL = "https://github.com/SherlockDavis/MiniGPT-from-scratch.git"


def md(text: str) -> dict:
    """A markdown cell. Source split into lines, each terminated by \\n except the last."""
    lines = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(text: str) -> dict:
    """A code cell."""
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


CELLS = [
    md(
        """\
# 🧠 MiniGPT-from-scratch — Colab Notebook

Train a ~30M-parameter GPT from scratch on TinyStories, on Colab's free T4 GPU, in ~1 hour.

**Repo**: https://github.com/SherlockDavis/MiniGPT-from-scratch

This notebook walks through:
1. Verify GPU
2. Clone the repo and install dependencies
3. Download the TinyStories dataset
4. Smoke test (100 training steps, ~1 min)
5. (Optional) Full training (10000 steps, ~83 min)
6. Generate text from a trained checkpoint

> ⚠️ **Before running**: go to **Runtime → Change runtime type → Hardware accelerator → T4 GPU**, otherwise training will be unusably slow on CPU.
"""
    ),
    md(
        """\
## 1. Verify GPU

`nvidia-smi` should print info for a Tesla T4 (16 GB). If it errors, you're not on a GPU runtime.
"""
    ),
    code("!nvidia-smi"),
    md(
        """\
## 2. Clone repo + install dependencies

The clone is shallow (`--depth=1`) — we only need the latest commit, not the full history.
"""
    ),
    code(
        f"""\
!git clone --depth=1 {REPO_URL}
%cd MiniGPT-from-scratch
"""
    ),
    code("!pip install -q -r requirements.txt"),
    md(
        """\
## 3. Download TinyStories

The script downloads `TinyStoriesV2-GPT4-train.txt` (~2 GB) and `TinyStoriesV2-GPT4-valid.txt` (~20 MB) into `data/`. Takes 2–5 min on Colab's network.
"""
    ),
    code("!bash scripts/download_data.sh"),
    md(
        """\
## 4. Smoke test (100 steps, ~1 min)

The first run also tokenizes the corpus into `data/train.bin` / `data/val.bin` — that one-time prep takes ~3 min on a fast tokenizer. Subsequent runs reuse the cached `.bin` files.

Watch for:
- `loss` should drop from ~11 (random init) toward ~7 within 100 steps
- No `Non-finite loss` errors
- Step time around 0.5 s on T4
"""
    ),
    code("!python train.py --max-iters 100 --no-wandb"),
    md(
        """\
## 5. Full training (~83 min on T4)

This runs the canonical 10000-step training with FP16 AMP + cosine LR schedule. Final val PPL should be around 4.7.

> 💡 If you don't want to wait, **skip this cell** and use the pre-trained checkpoint by downloading it from the repo's GitHub Release (or just rely on the smoke test for a much weaker model).
>
> 💡 To use Weights & Biases, drop `--no-wandb` and run `wandb login` first (you'll need a free account at https://wandb.ai).
"""
    ),
    code("!python train.py --no-wandb"),
    md(
        """\
## 6. Generate text

After training, the latest checkpoint is at `checkpoints/ckpt_step10000.pt`. Try the four sampling modes:
"""
    ),
    code(
        """\
# Greedy — deterministic, "safest" output
!python generate.py --checkpoint checkpoints/ckpt_step10000.pt \\
    --prompt "Once upon a time" --max-new-tokens 200 \\
    --temperature 0.0 --seed 42
"""
    ),
    code(
        """\
# Temperature 0.8 — recommended for narrative
!python generate.py --checkpoint checkpoints/ckpt_step10000.pt \\
    --prompt "Once upon a time" --max-new-tokens 200 \\
    --temperature 0.8 --seed 42
"""
    ),
    code(
        """\
# Top-p 0.9 nucleus sampling — adaptive cutoff
!python generate.py --checkpoint checkpoints/ckpt_step10000.pt \\
    --prompt "Once upon a time" --max-new-tokens 200 \\
    --temperature 1.0 --top-p 0.9 --seed 42
"""
    ),
    code(
        """\
# Try your own prompt
!python generate.py --checkpoint checkpoints/ckpt_step10000.pt \\
    --prompt "The little dragon" --max-new-tokens 200 \\
    --temperature 0.8 --top-p 0.9 --seed 7
"""
    ),
    md(
        """\
## 7. (Optional) Save the checkpoint to Google Drive

Colab session storage is wiped when the runtime disconnects. To keep your trained checkpoint, mount Drive and copy it over:
"""
    ),
    code(
        """\
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/drive/MyDrive/MiniGPT
!cp checkpoints/ckpt_step10000.pt /content/drive/MyDrive/MiniGPT/
!ls -lh /content/drive/MyDrive/MiniGPT/
"""
    ),
    md(
        """\
## What's next?

- Read the technical blog series in [`blog/`](https://github.com/SherlockDavis/MiniGPT-from-scratch/tree/main/blog) — three parts covering architecture, training, and generation.
- Try editing `config.py` to scale the model up (`n_layer=12, n_embd=768` → 124M, GPT-2 small).
- Add a `repetition_penalty` to `generate.py` to fix the model's mild repeat tendency.

If this was useful, **⭐ the repo on [GitHub](https://github.com/SherlockDavis/MiniGPT-from-scratch)**!
"""
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "colab": {
            "provenance": [],
            "name": "MiniGPT.ipynb",
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "gpuClass": "standard",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def main() -> None:
    OUT_PATH.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
