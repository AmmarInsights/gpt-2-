"""Model tests. They need PyTorch (skipped automatically when it is not installed)."""
import math

import pytest

torch = pytest.importorskip("torch")

from gpt2_gqa.config import GPTConfig  # noqa: E402
from gpt2_gqa.model import GPT, GroupedQueryAttention, load_checkpoint, save_checkpoint  # noqa: E402


def cfg(**kw):
    base = dict(block_size=32, vocab_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


@pytest.mark.parametrize("kv", [4, 2, 1])
@pytest.mark.parametrize("bias", [True, False])
def test_param_count_matches_analytic_formula(kv, bias):
    c = cfg(n_kv_head=kv, bias=bias)
    assert GPT(c).num_params() == c.num_params()


def test_gqa_has_fewer_params_than_mha():
    assert GPT(cfg(n_kv_head=1)).num_params() < GPT(cfg(n_kv_head=2)).num_params() < GPT(cfg(n_kv_head=4)).num_params()


def test_shapes_and_initial_loss_is_near_uniform():
    torch.manual_seed(0)
    model = GPT(cfg())
    idx = torch.randint(0, 64, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, 64)
    assert abs(loss.item() - math.log(64)) < 0.5
    assert model(idx)[0].shape == (3, 1, 64)
    assert model(idx, return_all=True)[0].shape == (3, 16, 64)


def test_weights_are_tied():
    model = GPT(cfg())
    assert model.lm_head.weight is model.transformer.wte.weight


def test_sequence_longer_than_block_size_is_rejected():
    with pytest.raises(ValueError):
        GPT(cfg())(torch.zeros(1, 33, dtype=torch.long))


def test_model_is_causal():
    torch.manual_seed(0)
    model = GPT(cfg()).eval()
    a = torch.randint(0, 64, (2, 20))
    b = a.clone()
    b[:, 10:] = (b[:, 10:] + 1) % 64
    la, _ = model(a, return_all=True)
    lb, _ = model(b, return_all=True)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)
    assert not torch.allclose(la[:, 10:], lb[:, 10:], atol=1e-5)


def reference_attention(attn: GroupedQueryAttention, x: torch.Tensor) -> torch.Tensor:
    """Slow, explicit per-head attention. Query head h uses key/value head h // group_size."""
    B, T, _ = x.shape
    H, Hkv, D = attn.n_head, attn.n_kv_head, attn.head_dim
    q = attn.q_proj(x).view(B, T, H, D)
    kv = attn.kv_proj(x)
    k = kv[..., : Hkv * D].reshape(B, T, Hkv, D)
    v = kv[..., Hkv * D:].reshape(B, T, Hkv, D)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    out = torch.zeros(B, T, H, D)
    for h in range(H):
        g = h // (H // Hkv)
        scores = torch.einsum("btd,bsd->bts", q[:, :, h], k[:, :, g]) / math.sqrt(D)
        scores = scores.masked_fill(~mask, float("-inf"))
        out[:, :, h] = torch.einsum("bts,bsd->btd", scores.softmax(-1), v[:, :, g])
    return attn.c_proj(out.reshape(B, T, H * D))


@pytest.mark.parametrize("kv", [8, 4, 2, 1])  # MHA, GQA (two group sizes), MQA
def test_grouped_query_attention_matches_reference(kv):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg(n_head=8, n_kv_head=kv, n_embd=64)).eval()
    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        assert torch.allclose(attn(x), reference_attention(attn, x), atol=1e-5)


def test_heads_in_a_group_share_keys_and_values():
    attn = GroupedQueryAttention(cfg(n_head=4, n_kv_head=2)).eval()
    k = torch.arange(2.0).view(1, 2, 1, 1).expand(1, 2, 3, 4)  # two distinct kv heads
    expanded = k.repeat_interleave(attn.group_size, dim=1)
    assert [expanded[0, h, 0, 0].item() for h in range(4)] == [0.0, 0.0, 1.0, 1.0]


def test_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    model = GPT(cfg())
    idx = torch.randint(0, 64, (4, 17))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = model(x, y)[1].item()
    for _ in range(200):
        opt.zero_grad()
        model(x, y)[1].backward()
        opt.step()
    assert model(x, y)[1].item() < 0.6 * first


def test_generate_shapes_greedy_and_top_k():
    torch.manual_seed(0)
    model = GPT(cfg()).eval()
    prompt = torch.randint(0, 64, (1, 5))
    out = model.generate(prompt, 10, temperature=1.0, top_k=5)
    assert out.shape == (1, 15) and out.min() >= 0 and out.max() < 64
    assert torch.equal(out[:, :5], prompt)
    g1, g2 = model.generate(prompt, 8, temperature=0), model.generate(prompt, 8, temperature=0)
    assert torch.equal(g1, g2)
    assert torch.equal(model.generate(prompt, 8, temperature=1.0, top_k=1), g1)


def test_generate_beyond_block_size_and_restores_mode():
    model = GPT(cfg())
    model.train()
    out = model.generate(torch.randint(0, 64, (1, 30)), 10, temperature=0)
    assert out.shape == (1, 40) and model.training


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = GPT(cfg()).eval()
    save_checkpoint(tmp_path / "c.pt", model, iter=7, val_loss=1.5)
    loaded, ckpt = load_checkpoint(tmp_path / "c.pt")
    x = torch.randint(0, 64, (2, 9))
    assert ckpt["iter"] == 7 and loaded.config == model.config
    assert torch.allclose(model(x, return_all=True)[0], loaded(x, return_all=True)[0])
