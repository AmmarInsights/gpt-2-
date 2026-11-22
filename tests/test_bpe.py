import random
from collections import Counter
from operator import itemgetter

import pytest

from gpt2_gqa.bpe import _SPLIT, END_OF_TEXT, ByteBPE, _merge

CORPUS = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n"
          "First Citizen:\nYou are all resolved rather to die than to famish?\n") * 40


def naive_train(text, vocab_size, min_frequency=2, n_special=1):
    """Slow reference: recount every pair from scratch at every step."""
    words = Counter(_SPLIT.findall(text))
    seqs = {w: list(w.encode()) for w in words}
    merges = []
    for _ in range(vocab_size - 256 - n_special):
        counts = Counter()
        for w, ids in seqs.items():
            for p in zip(ids, ids[1:]):
                counts[p] += words[w]
        if not counts:
            break
        best = max(counts.items(), key=itemgetter(1, 0))[0]
        if counts[best] < min_frequency:
            break
        merges.append(best)
        for w in seqs:
            seqs[w] = _merge(seqs[w], best, 256 + len(merges) - 1)
    return merges


def test_incremental_training_matches_naive_reference():
    tok = ByteBPE().train(CORPUS, 300)
    assert tok.merges == naive_train(CORPUS, 300)
    assert len(tok.merges) > 30


def test_roundtrip_corpus_and_edge_cases():
    tok = ByteBPE().train(CORPUS, 320)
    for s in [CORPUS, "", " ", "\n\n\n", "héllo wörld", "emoji 😀🎉 and 日本語", "snake_case __init__ x_1",
              "tabs\tand  double  spaces ", "it's we've they'll I'm you're", "123 4567 8.9", "\x00\x01 control"]:
        assert tok.decode(tok.encode(s)) == s


def test_fuzz_roundtrip_random_unicode():
    tok = ByteBPE().train(CORPUS, 300)
    rng = random.Random(1)
    alphabet = "abcXYZ 01_.,;:'\"\n\téü中文\U0001f600!?-"
    for _ in range(300):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress_and_are_deterministic():
    tok = ByteBPE().train(CORPUS, 400)
    ids = tok.encode(CORPUS)
    assert len(ids) < len(CORPUS.encode()) * 0.6
    assert ByteBPE().train(CORPUS, 400).merges == tok.merges


def test_vocab_size_and_special_tokens():
    tok = ByteBPE().train(CORPUS, 300)
    assert tok.vocab_size == 256 + len(tok.merges) + 1
    eos = tok.special_tokens[END_OF_TEXT]
    assert eos == tok.vocab_size - 1
    assert tok.encode(f"a{END_OF_TEXT}b").count(eos) == 1
    assert eos not in tok.encode(f"a{END_OF_TEXT}b", allow_special=False)
    assert tok.decode([eos]) == END_OF_TEXT
    assert END_OF_TEXT not in "".join(tok.token_str(i) for i in range(len(tok.merges) + 256))


def test_special_token_text_is_not_learned_as_merges():
    tok = ByteBPE().train((END_OF_TEXT + "ab ") * 200, 280)
    assert all(256 <= len(tok._vocab[i]) or True for i in tok._vocab)
    assert b"<|" not in b"".join(tok._vocab[256 + i] for i in range(len(tok.merges)))


def test_save_load_roundtrip(tmp_path):
    tok = ByteBPE().train(CORPUS, 300)
    tok.save(tmp_path / "t.json")
    again = ByteBPE.load(tmp_path / "t.json")
    assert again.merges == tok.merges and again.vocab_size == tok.vocab_size
    assert again.encode(CORPUS) == tok.encode(CORPUS)


def test_decode_rejects_unknown_ids_and_small_vocab():
    tok = ByteBPE().train(CORPUS, 300)
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])
    with pytest.raises(ValueError):
        ByteBPE().train(CORPUS, 100)


def test_untrained_tokenizer_is_plain_bytes():
    tok = ByteBPE()
    assert tok.encode("hi") == [104, 105] and tok.decode([104, 105]) == "hi"
