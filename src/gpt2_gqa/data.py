"""Dataset preparation: download Tiny Shakespeare, train the BPE tokenizer, write token files."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .bpe import END_OF_TEXT, ByteBPE

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(dest: "str | Path" = "data/shakespeare/input.txt") -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SHAKESPEARE_URL, headers={"User-Agent": "gpt2-gqa"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    if dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"Downloaded file looks too small ({dest.stat().st_size} bytes): {dest}")
    return dest


def prepare(text_path: "str | Path", out_dir: "str | Path", vocab_size: int = 1024,
            val_fraction: float = 0.1, verbose: bool = True) -> dict[str, Any]:
    """Split the text, train BPE on the *training* part only (no leakage), and write train.bin / val.bin."""
    text = Path(text_path).read_text(encoding="utf-8")
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    cut = int(len(text) * (1 - val_fraction))
    nl = text.find("\n", cut)  # cut at a line boundary
    cut = nl + 1 if nl != -1 else cut
    train_text, val_text = text[:cut], text[cut:]
    if not val_text.strip():
        raise ValueError("validation split is empty; use a larger file or a smaller val_fraction")

    tok = ByteBPE().train(train_text, vocab_size, verbose=verbose)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")

    meta: dict[str, Any] = {"vocab_size": tok.vocab_size, "end_of_text_id": tok.special_tokens[END_OF_TEXT]}
    for name, part in (("train", train_text), ("val", val_text)):
        ids = tok.encode(part, allow_special=False)
        assert max(ids) < 2**16, "uint16 storage needs vocab_size <= 65536"
        np.asarray(ids, dtype=np.uint16).tofile(out / f"{name}.bin")
        meta[f"{name}_tokens"] = len(ids)
        meta[f"{name}_chars"] = len(part)
        meta[f"{name}_bytes_per_token"] = round(len(part.encode("utf-8")) / len(ids), 3)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(json.dumps(meta, indent=2))
    return meta


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Download Tiny Shakespeare, train the BPE tokenizer, write train.bin / val.bin")
    p.add_argument("--input", default=None, help="Path to a text file (default: download Tiny Shakespeare)")
    p.add_argument("--out_dir", default="data/shakespeare")
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--val_fraction", type=float, default=0.1)
    args = p.parse_args(argv)
    src = Path(args.input) if args.input else download_shakespeare(Path(args.out_dir) / "input.txt")
    prepare(src, args.out_dir, args.vocab_size, args.val_fraction)


if __name__ == "__main__":
    main()
