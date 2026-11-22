# gpt2-gqa

A GPT-2 style language model with **grouped-query attention (GQA)** and a **byte-level BPE tokenizer written from scratch**, trained on Tiny Shakespeare. Runs on a free Google Colab GPU.

It follows the recipe from Andrej Karpathy's *Let's reproduce GPT-2 (124M)* code-along (pre-LayerNorm blocks, GELU(tanh) MLP, learned position embeddings, tied embeddings, scaled residual init, fused scaled-dot-product attention, AdamW with warmup + cosine decay, gradient clipping, mixed precision), with two changes the task asked for: GQA instead of multi-head attention, and our own BPE tokenizer instead of `tiktoken`.

```
text ──► ByteBPE (256 bytes + learned merges) ──► token ids ──► GPT ──► logits
                                                                │
        token emb + position emb ─► [ LN ─► GQA ─► + ─► LN ─► MLP ─► + ] x N ─► LN ─► tied LM head
```

## Grouped-query attention in one picture

Standard attention gives every query head its own key and value head. GQA keeps all the query heads but shares each key/value head between a *group* of query heads:

```
MHA  (n_head=6, n_kv_head=6)   q q q q q q      GQA (n_head=6, n_kv_head=2)   q q q | q q q      MQA (n_kv_head=1)  q q q q q q
                               | | | | | |                                    \ | /   \ | /                         \ \ \ | / /
                               k k k k k k                                      k       k                               k
```

`n_kv_head` is one config value: `= n_head` is plain GPT-2 attention, `1 < n_kv_head < n_head` is GQA, `= 1` is multi-query attention. Query head `h` reads key/value head `h // (n_head // n_kv_head)`.

Why it matters: fewer key/value projections (fewer parameters) and, more importantly at inference, a KV cache that is `n_head / n_kv_head` times smaller. This repo's `generate` recomputes attention each step (as in the video) and does **not** implement a KV cache, so the cache saving is shown analytically below rather than measured.

| Config (vocab 1024, 6 layers, 384 dims) | Parameters | KV cache per token (fp16) | At 256 tokens |
|---|---|---|---|
| MHA, 6 query / 6 kv heads | 11.14M | 9,216 B | 2,304 KB |
| **GQA, 6 query / 2 kv heads (default)** | **9.96M** | **3,072 B** | **768 KB** |
| MQA, 6 query / 1 kv head | 9.66M | 1,536 B | 384 KB |

These numbers come from `GPTConfig.num_params()` and `kv_cache_bytes_per_token()`. The parameter formula is checked against the known GPT-2 124M count (124,439,808) in the tests.

## Quickstart

**Colab (recommended):** open `notebooks/colab_gpt2_gqa.ipynb`, choose a GPU runtime, run all cells. It runs the test suite, downloads Tiny Shakespeare, trains, plots the loss curve and prints generated text.

**Local:**

```bash
pip install -e ".[dev]"          # needs PyTorch 2.x
pytest -q                        # verify the model before training
python -m gpt2_gqa.data          # download Tiny Shakespeare, train BPE, write token files
python -m gpt2_gqa.train --preset small                       # GQA, 6 heads / 2 kv heads
python -m gpt2_gqa.train --preset small --n_kv_head 6 --out_dir out/mha   # baseline: standard MHA
python -m gpt2_gqa.generate --out_dir out/gqa --prompt "ROMEO:" --num_samples 3
```

No GPU? `--preset tiny` is a CPU smoke test.

| Preset | Layers | Heads (q / kv) | Dim | Context | Purpose |
|---|---|---|---|---|---|
| `tiny` | 2 | 4 / 2 | 128 | 128 | CPU smoke test |
| `small` | 6 | 6 / 2 | 384 | 256 | default for a free Colab GPU |
| `gpt2` | 12 | 12 / 4 | 768 | 1024 | GPT-2 124M shape with GQA (needs far more data than Shakespeare) |

Any value can be overridden on the command line (`--n_layer 8 --lr 6e-4 --batch_size 32 ...`) or via `make_config(...)` in Python.

## The BPE tokenizer (`bpe.py`)

- **Byte-level:** base vocabulary is the 256 bytes, so every string round-trips exactly (emoji, CJK, control characters) with no unknown token.
- **GPT-2 style pre-tokenisation** with the stdlib `re` (letters, digits, symbols, whitespace), so merges never cross word boundaries.
- **Fast training:** works on unique words and updates pair counts incrementally. A test checks it produces exactly the same merges as a slow naive reference.
- Deterministic tie-breaking, `<|endoftext|>` special token, JSON save/load.
- The tokenizer is trained on the **training split only**, so validation text never influences the vocabulary.

```python
from gpt2_gqa import ByteBPE
tok = ByteBPE().train(open("input.txt").read(), vocab_size=1024)
ids = tok.encode("To be, or not to be"); print(tok.decode(ids))
```

## Training recipe (`train.py`)

AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), linear warmup then cosine decay to `min_lr`, gradient clipping at 1.0, bf16 autocast (fp16 + GradScaler on GPUs without bf16), optional gradient accumulation and `torch.compile`, best-checkpoint saving on validation loss, and samples generated from the best checkpoint. Each run writes `ckpt.pt`, `tokenizer.json`, `log.csv`, `loss_curve.png`, `samples.txt` and `summary.json` to its output folder.

## Results

Not recorded yet. Run the notebook (or the commands above), then fill this table from `out/*/summary.json` and paste a sample from `out/gqa/samples.txt`:

| Run | Parameters | Best val loss | Time |
|---|---|---|---|
| GQA (6 q / 2 kv heads) | | | |
| MHA baseline (6 / 6) | | | |

Tiny Shakespeare is about 1 MB, so a model this size overfits: expect validation loss to bottom out and then rise, which is why dropout 0.2 is the default and the best checkpoint (not the last) is used for samples. Generated text will look Shakespeare-like in form (speaker names, line breaks, archaic words) but will not be coherent.

## Tests

`pytest -q` covers:

- **Tokenizer (no PyTorch needed):** round trips including unicode and fuzzed strings, equality with a naive reference trainer, special tokens, save/load, determinism.
- **Config and data:** validation, analytic parameter count against GPT-2 124M, files written by `prepare` decode back to the original text.
- **Model (needs PyTorch):** parameter count equals the formula for MHA/GQA/MQA, weight tying, causality, GQA output equals an independent per-head reference implementation, can overfit a tiny batch, generation (greedy, top-k, longer than context), checkpoint round trip.
- **End to end (needs PyTorch):** prepare data, train on CPU, loss decreases, artifacts written, generate from the checkpoint.

## Known limits

- No KV cache in `generate`, so GQA's inference-memory benefit is analytic here.
- Positions are GPT-2's learned embeddings (not RoPE), so context is fixed at `block_size`.
- Single GPU, no distributed training.

## Credits

Architecture and training recipe follow Andrej Karpathy's *Let's reproduce GPT-2 (124M)* video and the `nanoGPT` / `build-nanogpt` repositories; the tokenizer follows the ideas in his *Let's build the GPT Tokenizer* video. Grouped-query attention is from Ainslie et al., 2023. All code here is written from scratch. Dataset: Tiny Shakespeare from `karpathy/char-rnn`.

## License

MIT
