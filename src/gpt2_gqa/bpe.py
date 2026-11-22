"""Byte-level BPE tokenizer (GPT-2 style), written from scratch in pure Python.

Why byte-level: the base vocabulary is the 256 possible bytes, so *any* string can be encoded and decoded
without unknown tokens. Training then learns merges of frequent byte pairs.

Training is incremental (pair counts are updated only for words that change), so it takes seconds on a
~1 MB corpus such as Tiny Shakespeare.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Optional

# GPT-2 style pre-tokenisation using only the stdlib `re` (letters, digits, other symbols, whitespace).
# Words never merge across these boundaries, which keeps tokens meaningful ("dog", " dog", "!" ...).
SPLIT_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:(?![^\W\d_]|\d)\S)+|\s+(?!\S)|\s+"
_SPLIT = re.compile(SPLIT_PATTERN)

END_OF_TEXT = "<|endoftext|>"


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPE:
    def __init__(self, merges: Optional[list[tuple[int, int]]] = None,
                 special_tokens: Optional[Iterable[str]] = (END_OF_TEXT,)):
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]  # type: ignore[misc]
        self._special_list = list(special_tokens or [])
        self._rebuild()

    # ------------------------------------------------------------------ setup
    def _rebuild(self) -> None:
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self._vocab[256 + i] = self._vocab[a] + self._vocab[b]
        base = 256 + len(self.merges)
        self.special_tokens = {tok: base + j for j, tok in enumerate(self._special_list)}
        self._special_by_id = {i: tok for tok, i in self.special_tokens.items()}
        self._cache: dict[str, list[int]] = {}
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            self._special_re: Optional[re.Pattern[str]] = re.compile(pattern)
        else:
            self._special_re = None

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    # --------------------------------------------------------------- training
    def train(self, text: str, vocab_size: int, min_frequency: int = 2, verbose: bool = False) -> "ByteBPE":
        """Learn merges from ``text`` until the vocabulary (bytes + merges + special tokens) has ``vocab_size`` entries.

        Ties are broken deterministically (highest count, then largest pair), so training is reproducible.
        """
        n_special = len(self._special_list)
        if vocab_size < 256 + n_special:
            raise ValueError(f"vocab_size must be at least {256 + n_special}")
        n_merges = vocab_size - 256 - n_special

        # Count unique pre-tokenised words; BPE then only has to work on unique words.
        word_freq: Counter[str] = Counter()
        for piece in self._split_special(text):
            if piece not in self.special_tokens:
                word_freq.update(_SPLIT.findall(piece))
        words = [list(w.encode("utf-8")) for w in word_freq]
        freqs = list(word_freq.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            for p in zip(ids, ids[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)

        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=itemgetter(1, 0))[0]
            count = pair_counts[best]
            if count < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            for wi in list(where.pop(best, ())):
                ids, f = words[wi], freqs[wi]
                merged = _merge(ids, best, new_id)
                if len(merged) == len(ids):  # stale index entry
                    continue
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                for p in zip(merged, merged[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                words[wi] = merged
            pair_counts.pop(best, None)
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges}: {best} -> {new_id} (count {count})")
        self.merges = merges
        self._rebuild()
        return self

    # ---------------------------------------------------------------- encoding
    def _split_special(self, text: str) -> list[str]:
        return [p for p in self._special_re.split(text) if p] if self._special_re else [text]

    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            ranked = [(self._ranks[p], p) for p in zip(ids, ids[1:]) if p in self._ranks]
            if not ranked:
                break
            rank, pair = min(ranked)
            ids = _merge(ids, pair, 256 + rank)
        self._cache[word] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Text -> token ids. Special tokens in the text map to their ids when ``allow_special`` is true."""
        out: list[int] = []
        for piece in self._split_special(text):
            if allow_special and piece in self.special_tokens:
                out.append(self.special_tokens[piece])
                continue
            words = _SPLIT.findall(piece)
            if "".join(words) != piece:  # cannot happen with SPLIT_PATTERN; guards against silent data loss
                raise ValueError("pre-tokenisation dropped characters")
            for w in words:
                out.extend(self._encode_word(w))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self._vocab:
                parts.append(self._vocab[i])
            elif i in self._special_by_id:
                parts.append(self._special_by_id[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {i} (vocab size {self.vocab_size})")
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_str(self, token_id: int) -> str:
        """Human-readable form of one token (for debugging)."""
        if token_id in self._special_by_id:
            return self._special_by_id[token_id]
        return self._vocab[token_id].decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence
    def save(self, path: "str | Path") -> None:
        data = {"version": 1, "pattern": SPLIT_PATTERN, "merges": [list(m) for m in self.merges],
                "special_tokens": self._special_list}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: "str | Path") -> "ByteBPE":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("pattern") != SPLIT_PATTERN:
            raise ValueError("tokenizer file was made with a different pre-tokenisation pattern")
        return cls([tuple(m) for m in data["merges"]], data.get("special_tokens", []))
