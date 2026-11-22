"""Training loop: AdamW, linear warmup + cosine decay, gradient clipping, mixed precision, best-checkpointing."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .bpe import ByteBPE
from .config import PRESETS, GPTConfig
from .generate import generate_text
from .model import GPT, load_checkpoint, save_checkpoint

SAMPLE_PROMPTS = ["\n", "ROMEO:", "First Citizen:"]


@dataclass
class TrainConfig:
    data_dir: str = "data/shakespeare"
    out_dir: str = "out/gqa"
    # model
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.2
    # optimisation
    batch_size: int = 64
    grad_accum: int = 1
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 50
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # system
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    seed: int = 1337
    sample_tokens: int = 300
    log_interval: int = 50


def make_config(preset: str = "small", **overrides: Any) -> TrainConfig:
    """Start from a preset ('tiny', 'small', 'gpt2') and apply any non-None overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset '{preset}'. Choose from {sorted(PRESETS)}")
    values = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
    return TrainConfig(**values)


def get_lr(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.lr - cfg.min_lr)


def _make_scaler(enabled: bool, device_type: str):
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _load_bin(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def plot_losses(history: list[dict[str, float]], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    its = [h["iter"] for h in history]
    ax.plot(its, [h["train_loss"] for h in history], label="train")
    ax.plot(its, [h["val_loss"] for h in history], label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def train(cfg: TrainConfig, verbose: bool = True) -> dict[str, Any]:
    """Train and return a summary. Writes ckpt.pt, tokenizer.json, log.csv, loss_curve.png, samples.txt, summary.json."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if device_type == "cuda":
        dtype = cfg.dtype if cfg.dtype != "auto" else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    else:
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = contextlib.nullcontext() if dtype == "float32" else torch.autocast(device_type=device_type, dtype=ptdtype)
    scaler = _make_scaler(dtype == "float16", device_type)

    data_dir, out_dir = Path(cfg.data_dir), Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "meta.json").read_text())
    tok = ByteBPE.load(data_dir / "tokenizer.json")
    shutil.copy(data_dir / "tokenizer.json", out_dir / "tokenizer.json")  # keep the output folder self-contained
    train_data, val_data = _load_bin(data_dir / "train.bin"), _load_bin(data_dir / "val.bin")
    for name, d in (("train", train_data), ("val", val_data)):
        if len(d) <= cfg.block_size + 1:
            raise ValueError(f"{name} split has {len(d)} tokens, need more than block_size+1 = {cfg.block_size + 1}")

    model_cfg = GPTConfig(block_size=cfg.block_size, vocab_size=meta["vocab_size"], n_layer=cfg.n_layer,
                          n_head=cfg.n_head, n_kv_head=cfg.n_kv_head, n_embd=cfg.n_embd, dropout=cfg.dropout)
    raw_model = GPT(model_cfg).to(device)
    model = torch.compile(raw_model) if cfg.compile else raw_model
    optimizer = raw_model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
    n_params = raw_model.num_params()
    if verbose:
        print(f"device={device} dtype={dtype} params={n_params / 1e6:.2f}M "
              f"heads={cfg.n_head} kv_heads={cfg.n_kv_head} (group size {model_cfg.group_size}) "
              f"vocab={meta['vocab_size']} train_tokens={len(train_data):,} val_tokens={len(val_data):,}")

    gen = torch.Generator().manual_seed(cfg.seed)

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,), generator=gen).tolist()
        x = torch.stack([data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x, y = get_batch(split)
                with ctx:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    history: list[dict[str, float]] = []
    best_val = float("inf")
    start = time.time()
    model.train()
    for it in range(cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss()
            history.append({"iter": it, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr,
                            "seconds": time.time() - start})
            if verbose:
                print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} "
                      f"| {time.time() - start:6.1f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out_dir / "ckpt.pt", raw_model, iter=it, val_loss=best_val, train_loss=losses["train"],
                                train_config=asdict(cfg))
        if it == cfg.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch("train")
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if verbose and cfg.log_interval and it % cfg.log_interval == 0 and it % cfg.eval_interval != 0:
            print(f"iter {it:5d} | batch loss {step_loss:.4f}")

    elapsed = time.time() - start
    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plotted = plot_losses(history, out_dir / "loss_curve.png")

    # Samples come from the best checkpoint (lowest validation loss), not necessarily the last step.
    best_model, ckpt = load_checkpoint(out_dir / "ckpt.pt", device)
    samples = []
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        samples.append(f"### prompt={prompt!r} temperature=0.8 top_k=50\n" +
                       generate_text(best_model, tok, prompt, cfg.sample_tokens, 0.8, 50, device, seed=cfg.seed + i))
    (out_dir / "samples.txt").write_text("\n\n".join(samples) + "\n", encoding="utf-8")

    summary = {
        "params": n_params, "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head,
        "kv_cache_bytes_per_token_fp16": model_cfg.kv_cache_bytes_per_token(2),
        "best_val_loss": best_val, "best_iter": ckpt["iter"], "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"], "seconds": round(elapsed, 1), "device": device, "dtype": dtype,
        "vocab_size": meta["vocab_size"], "loss_curve": str(out_dir / "loss_curve.png") if plotted else None,
        "samples": str(out_dir / "samples.txt"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train a GPT-2 style model with grouped-query attention")
    p.add_argument("--preset", default="small", choices=sorted(PRESETS))
    p.add_argument("--data_dir", default="data/shakespeare")
    p.add_argument("--out_dir", default="out/gqa")
    for f in TrainConfig.__dataclass_fields__.values():
        if f.name in ("data_dir", "out_dir"):
            continue
        kind = {"int": int, "float": float, "str": str, "bool": None}[f.type if isinstance(f.type, str) else f.type.__name__]
        if kind is None:
            p.add_argument(f"--{f.name}", action="store_true", default=None)
        else:
            p.add_argument(f"--{f.name}", type=kind, default=None)
    args = vars(p.parse_args(argv))
    preset = args.pop("preset")
    train(make_config(preset, **args))


if __name__ == "__main__":
    main()
