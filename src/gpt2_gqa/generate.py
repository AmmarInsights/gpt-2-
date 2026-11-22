"""Text generation from a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from .bpe import ByteBPE
from .model import GPT, load_checkpoint


def generate_text(model: GPT, tok: ByteBPE, prompt: str = "\n", max_new_tokens: int = 300,
                  temperature: float = 0.8, top_k: Optional[int] = 50, device: str = "cpu",
                  seed: Optional[int] = None) -> str:
    """Sample a continuation of ``prompt``. temperature=0 gives greedy decoding."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = tok.encode(prompt, allow_special=False)
    if not ids:
        raise ValueError("prompt must not be empty")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    return tok.decode(y[0].tolist())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate text with a trained model")
    p.add_argument("--out_dir", default="out/gqa", help="Folder with ckpt.pt and tokenizer.json")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out_dir)
    model, _ = load_checkpoint(out / "ckpt.pt", device)
    tok = ByteBPE.load(out / "tokenizer.json")
    for i in range(args.num_samples):
        print(f"--- sample {i + 1} " + "-" * 50)
        print(generate_text(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device,
                            seed=args.seed + i))


if __name__ == "__main__":
    main()
