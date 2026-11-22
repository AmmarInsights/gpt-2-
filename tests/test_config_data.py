import json

import numpy as np
import pytest

from gpt2_gqa.config import PRESETS, GPTConfig
from gpt2_gqa.data import prepare


def test_config_validation():
    with pytest.raises(ValueError, match="n_embd"):
        GPTConfig(n_embd=100, n_head=6)
    with pytest.raises(ValueError, match="n_kv_head"):
        GPTConfig(n_head=6, n_kv_head=4)
    assert GPTConfig(n_head=6, n_kv_head=2).group_size == 3


def test_presets_are_valid_configs():
    model_keys = {"n_layer", "n_head", "n_kv_head", "n_embd", "block_size", "dropout"}
    for name, p in PRESETS.items():
        GPTConfig(**{k: v for k, v in p.items() if k in model_keys})


def test_gpt2_124m_shape_matches_known_parameter_count():
    # Standard multi-head GPT-2 small with its 50257-token vocabulary has 124,439,808 parameters.
    cfg = GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_kv_head=12, n_embd=768, bias=True)
    assert cfg.num_params() == 124_439_808


def test_gqa_has_fewer_params_and_smaller_kv_cache():
    mha = GPTConfig(n_head=12, n_kv_head=12, n_embd=768)
    gqa = GPTConfig(n_head=12, n_kv_head=4, n_embd=768)
    mqa = GPTConfig(n_head=12, n_kv_head=1, n_embd=768)
    assert mqa.num_params() < gqa.num_params() < mha.num_params()
    assert mha.kv_cache_bytes_per_token() == 3 * gqa.kv_cache_bytes_per_token() == 12 * mqa.kv_cache_bytes_per_token()


def test_prepare_writes_aligned_files(tmp_path):
    text = "".join(f"Line {i} the quick brown fox jumps over the lazy dog\n" for i in range(600))
    src = tmp_path / "in.txt"
    src.write_text(text)
    meta = prepare(src, tmp_path / "out", vocab_size=300, verbose=False)
    train = np.fromfile(tmp_path / "out" / "train.bin", dtype=np.uint16)
    val = np.fromfile(tmp_path / "out" / "val.bin", dtype=np.uint16)
    assert len(train) == meta["train_tokens"] and len(val) == meta["val_tokens"]
    assert 0.05 < len(val) / (len(train) + len(val)) < 0.15
    assert train.max() < meta["vocab_size"] and val.max() < meta["vocab_size"]
    from gpt2_gqa.bpe import ByteBPE
    tok = ByteBPE.load(tmp_path / "out" / "tokenizer.json")
    assert tok.decode(train.tolist()) + tok.decode(val.tolist()) == text
    assert json.loads((tmp_path / "out" / "meta.json").read_text())["vocab_size"] == tok.vocab_size
