"""Minimal training loop for MiniGPT.

This is workflow step 8: a baseline AdamW loop with constant LR, no mixed
precision, no scheduler, no wandb. Step 9 layers in wandb; step 10 adds
AMP + gradient accumulation + cosine LR schedule. Keeping this file
minimal makes it easy to confirm that the *base* training signal is
healthy before stacking optimizations on top.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from config import GPTConfig
from model import GPT
from utils import get_batch, get_fast_tokenizer, load_tokens, prepare_data

logger = logging.getLogger("minigpt")

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")

TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"
TRAIN_TXT = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VAL_TXT = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_data_prepared() -> tuple[np.ndarray, np.ndarray]:
    """Make sure train.bin and val.bin exist, encoding from .txt on first run."""
    tokenizer = None  # lazy: only load if we actually need to encode
    for bin_path, txt_path, name in [
        (TRAIN_BIN, TRAIN_TXT, "train"),
        (VAL_BIN, VAL_TXT, "val"),
    ]:
        if bin_path.exists():
            continue
        if not txt_path.exists():
            raise FileNotFoundError(
                f"Neither {bin_path} nor {txt_path} found. "
                f"Run `bash scripts/download_data.sh` first."
            )
        if tokenizer is None:
            # Fast Rust tokenizer: ~10-50x faster than GPT2Tokenizer on the
            # 2 GB train file (minutes vs. hours).
            logger.info("Loading fast GPT-2 tokenizer for one-time data prep...")
            tokenizer = get_fast_tokenizer()
        logger.info("Encoding %s corpus -> %s (this can take a while)", name, bin_path)
        n = prepare_data(txt_path, bin_path, tokenizer=tokenizer, show_progress=True)
        logger.info("  wrote %s tokens", f"{n:,}")
    return load_tokens(TRAIN_BIN), load_tokens(VAL_BIN)


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float,
) -> float:
    """One forward/backward/step (single batch, fp32). Returns the scalar loss.

    Kept as the simplest possible training primitive for unit tests. The full
    `train()` loop inlines AMP + gradient accumulation directly because they
    require coordinating multiple micro-batches and a GradScaler.
    """
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def get_lr(
    step: int,
    warmup_iters: int,
    max_iters: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup then cosine decay (GPT-2 style).

    - Steps [0, warmup_iters): linear ramp from peak_lr/warmup_iters to peak_lr
    - Steps [warmup_iters, max_iters]: cosine decay from peak_lr down to min_lr
    - Steps > max_iters: clamped at min_lr (covers any over-runs of the schedule)
    """
    if step < warmup_iters:
        return peak_lr * (step + 1) / warmup_iters
    if step >= max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str,
    eval_iters: int,
) -> dict[str, float]:
    """Average loss over `eval_iters` random batches of each split (model in eval mode)."""
    model.eval()
    out: dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device=device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
        },
        path,
    )
    logger.info("Saved %s", path)


def train(
    config: GPTConfig,
    device: str,
    max_iters_override: int | None = None,
    eval_interval: int = 250,
    eval_iters: int = 20,
    log_interval: int = 10,
    save_interval: int = 1000,
    seed: int = 1337,
    use_wandb: bool = True,
    wandb_project: str = "minigpt",
    wandb_run_name: str | None = None,
) -> None:
    setup_logging()
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Device: %s", device)
    train_data, val_data = ensure_data_prepared()
    logger.info(
        "Train tokens: %s | Val tokens: %s", f"{len(train_data):,}", f"{len(val_data):,}"
    )

    model = GPT(config).to(device)
    logger.info("Model: %.2fM parameters", model.num_parameters() / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    max_iters = max_iters_override if max_iters_override is not None else config.max_iters
    min_lr = config.learning_rate * 0.1  # GPT-style cosine floor (10% of peak)
    grad_accum_steps = config.gradient_accumulation_steps

    # AMP only on CUDA — bf16-on-CPU autocast is slower than fp32 on most chips.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("AMP enabled (fp16 autocast + GradScaler)")
    if grad_accum_steps > 1:
        logger.info(
            "Gradient accumulation: %d micro-batches per step (effective bs=%d)",
            grad_accum_steps,
            config.batch_size * grad_accum_steps,
        )

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={**dataclasses.asdict(config), "max_iters_run": max_iters, "device": device},
        )

    t_start = time.time()
    t_window = t_start
    model.train()

    try:
        for step in range(max_iters):
            # 1) Set lr for this step (single param group — cosine schedule).
            lr = get_lr(step, config.warmup_iters, max_iters, config.learning_rate, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # 2) Gradient accumulation: forward/backward over `grad_accum_steps`
            #    micro-batches, then a single optimizer.step(). Loss is divided
            #    so the accumulated gradient matches a single big-batch update.
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(grad_accum_steps):
                x, y = get_batch(
                    train_data, config.block_size, config.batch_size, device=device
                )
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    _, loss = model(x, y)
                    loss = loss / grad_accum_steps
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += loss.item()

            # 3) Unscale (under AMP) for honest grad-clip math, clip, then step.
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            loss_val = loss_accum  # already the average over micro-batches

            if not np.isfinite(loss_val):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss_val}")

            if use_wandb:
                wandb.log({"train/loss": loss_val, "train/lr": lr}, step=step)

            if step % log_interval == 0 or step == max_iters - 1:
                now = time.time()
                dt = now - t_window
                t_window = now
                logger.info(
                    "Step %d | loss %.4f | lr %.2e | time %.1fs", step, loss_val, lr, dt
                )

            if step > 0 and step % eval_interval == 0:
                losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    config.block_size,
                    config.batch_size,
                    device,
                    eval_iters,
                )
                logger.info(
                    "Step %d | eval | train %.4f | val %.4f",
                    step,
                    losses["train"],
                    losses["val"],
                )
                if use_wandb:
                    wandb.log(
                        {"eval/train_loss": losses["train"], "eval/val_loss": losses["val"]},
                        step=step,
                    )

            if step > 0 and step % save_interval == 0:
                save_checkpoint(
                    CKPT_DIR / f"ckpt_step{step}.pt", model, optimizer, config, step
                )

        save_checkpoint(
            CKPT_DIR / f"ckpt_step{max_iters}.pt", model, optimizer, config, max_iters
        )
        logger.info("Done. Total time: %.1f min", (time.time() - t_start) / 60)
    finally:
        if use_wandb:
            wandb.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MiniGPT on TinyStories.")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu (default: cuda if available)",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override config.max_iters — useful for smoke tests (e.g. --max-iters 100).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config.batch_size — drop to fit smaller VRAM (e.g. 16 on 6GB).",
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override config.gradient_accumulation_steps — combine with --batch-size to "
        "preserve the effective batch (e.g. --batch-size 16 --grad-accum-steps 4 = bs 64).",
    )
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    p.add_argument("--wandb-project", default="minigpt", help="W&B project name.")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Build a config and apply runtime overrides without touching config.py defaults.
    cfg = GPTConfig()
    if args.batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=args.batch_size)
    if args.grad_accum_steps is not None:
        cfg = dataclasses.replace(cfg, gradient_accumulation_steps=args.grad_accum_steps)
    train(
        config=cfg,
        device=args.device,
        max_iters_override=args.max_iters,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
