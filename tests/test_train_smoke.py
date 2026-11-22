"""End-to-end smoke test: prepare data -> train a few steps on CPU -> generate. Needs PyTorch."""
import csv
import math

import pytest

torch = pytest.importorskip("torch")

from gpt2_gqa.bpe import ByteBPE  # noqa: E402
from gpt2_gqa.data import prepare  # noqa: E402
from gpt2_gqa.generate import generate_text, main as generate_main  # noqa: E402
from gpt2_gqa.model import load_checkpoint  # noqa: E402
from gpt2_gqa.train import get_lr, make_config, train  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    text = "".join(f"ROMEO: Line {i % 7} the quick brown fox jumps over the lazy dog\n" for i in range(800))
    (root / "in.txt").write_text(text)
    prepare(root / "in.txt", root / "data", vocab_size=300, verbose=False)
    cfg = make_config("tiny", data_dir=str(root / "data"), out_dir=str(root / "out"), max_iters=60, eval_interval=30,
                      eval_iters=4, batch_size=8, block_size=32, n_layer=1, n_embd=32, n_head=4, n_kv_head=2,
                      sample_tokens=20, device="cpu", log_interval=0)
    return root, train(cfg, verbose=False)


def test_training_improves_and_writes_artifacts(trained):
    root, summary = trained
    out = root / "out"
    for name in ("ckpt.pt", "tokenizer.json", "log.csv", "samples.txt", "summary.json"):
        assert (out / name).exists(), name
    rows = list(csv.DictReader(open(out / "log.csv")))
    assert float(rows[-1]["val_loss"]) < float(rows[0]["val_loss"])
    assert math.isfinite(summary["best_val_loss"]) and summary["n_kv_head"] == 2
    assert (out / "samples.txt").read_text().strip()


def test_generation_from_checkpoint(trained, capsys):
    root, _ = trained
    out = root / "out"
    model, _ = load_checkpoint(out / "ckpt.pt")
    tok = ByteBPE.load(out / "tokenizer.json")
    a = generate_text(model, tok, "ROMEO:", 10, temperature=0.8, top_k=20, seed=1)
    assert a.startswith("ROMEO:") and a == generate_text(model, tok, "ROMEO:", 10, temperature=0.8, top_k=20, seed=1)
    generate_main(["--out_dir", str(out), "--max_new_tokens", "5", "--device", "cpu"])
    assert "sample 1" in capsys.readouterr().out


def test_lr_schedule_warms_up_then_decays_to_min():
    cfg = make_config("tiny", max_iters=100, warmup_iters=10, lr=1e-3, min_lr=1e-4)
    assert get_lr(0, cfg) < get_lr(9, cfg) <= cfg.lr
    assert get_lr(10, cfg) == pytest.approx(cfg.lr)
    assert get_lr(100, cfg) == pytest.approx(cfg.min_lr) and get_lr(500, cfg) == cfg.min_lr
    assert make_config("small", n_kv_head=3, n_head=6).n_kv_head == 3
    with pytest.raises(ValueError):
        make_config("nope")
