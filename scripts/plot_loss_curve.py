"""Parse step11.log and generate a publication-ready loss curve PNG.

Output: assets/loss_curve.png

The script is intentionally self-contained — re-runnable any time the training
log is regenerated. It plots:
  - per-step train loss (light blue, with light EMA smoothing)
  - eval train loss (blue dots, every eval_interval steps)
  - eval val loss   (orange dots)
  - lr schedule on a secondary y-axis (dashed, gray)
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("step11.log")
OUT_PATH = Path("assets/loss_curve.png")

STEP_RE = re.compile(r"Step (\d+) \| loss ([\d.]+) \| lr ([\d.eE+\-]+)")
EVAL_RE = re.compile(r"Step (\d+) \| eval \| train ([\d.]+) \| val ([\d.]+)")


def parse_log(path: Path):
    steps, losses, lrs = [], [], []
    eval_steps, eval_train, eval_val = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if (m := EVAL_RE.search(line)):
            eval_steps.append(int(m.group(1)))
            eval_train.append(float(m.group(2)))
            eval_val.append(float(m.group(3)))
            continue  # eval line also contains "Step N" so check it first
        if (m := STEP_RE.search(line)):
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            lrs.append(float(m.group(3)))
    return (
        np.array(steps),
        np.array(losses),
        np.array(lrs),
        np.array(eval_steps),
        np.array(eval_train),
        np.array(eval_val),
    )


def ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def main() -> None:
    steps, losses, lrs, eval_steps, eval_train, eval_val = parse_log(LOG_PATH)
    print(f"Parsed {len(steps)} train points, {len(eval_steps)} eval points")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    fig.patch.set_facecolor("white")

    # Raw train loss (light, noisy)
    ax.plot(steps, losses, color="#9bbcd9", linewidth=0.8, alpha=0.5, label="train loss (per step)")
    # Smoothed train loss (the visually dominant line)
    ax.plot(steps, ema(losses, alpha=0.02), color="#1f6dad", linewidth=2.0, label="train loss (smoothed)")
    # Eval points
    ax.scatter(eval_steps, eval_train, color="#1f6dad", s=22, zorder=5, label="eval train loss")
    ax.scatter(eval_steps, eval_val, color="#d97706", s=22, marker="s", zorder=5, label="eval val loss")
    # Final val loss annotation
    final_val = eval_val[-1]
    ax.annotate(
        f"val loss = {final_val:.3f}\nPPL ≈ {np.exp(final_val):.2f}",
        xy=(eval_steps[-1], final_val),
        xytext=(eval_steps[-1] - 1500, final_val + 1.2),
        fontsize=10,
        color="#d97706",
        arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.0),
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("MiniGPT on TinyStories — 30M params, T4, 83 min", fontsize=13, fontweight="bold")
    ax.set_ylim(1.0, 11.5)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)

    # LR on secondary axis
    ax2 = ax.twinx()
    ax2.plot(steps, lrs, color="#888888", linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.set_ylabel("Learning rate", color="#666666")
    ax2.tick_params(axis="y", colors="#666666")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
