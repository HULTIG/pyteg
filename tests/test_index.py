import numpy as np

from pyteg import CooccurrenceIndex


def brute_joint(docs, x, y):
    return sum(1 for d in docs if x in d and y in d)


def test_counts_match_brute_force(random_docs):
    idx = CooccurrenceIndex.build(random_docs)
    assert idx.n_docs == len(random_docs)
    words = ["w0", "w1", "w5", "w17", "w39", "missing"]
    m = idx.joint_matrix(words, words)
    for i, x in enumerate(words):
        assert idx.freq(x) == sum(1 for d in random_docs if x in d)
        for j, y in enumerate(words):
            assert m[i, j] == brute_joint(random_docs, x, y)


def test_save_and_load_roundtrip(tmp_path, random_docs):
    idx = CooccurrenceIndex.build(random_docs, meta={"corpus": "random"})
    path = tmp_path / "idx.npz"
    idx.save(path)
    again = CooccurrenceIndex.load(path)
    assert again.vocab == idx.vocab
    assert again.n_docs == idx.n_docs
    assert again.meta["corpus"] == "random"
    np.testing.assert_array_equal(again.df, idx.df)
    words = idx.vocab[:10]
    np.testing.assert_array_equal(again.joint_matrix(words, words), idx.joint_matrix(words, words))


def test_prune_keeps_document_count(random_docs):
    idx = CooccurrenceIndex.build(random_docs)
    pruned = idx.prune(20)
    assert pruned.n_docs == idx.n_docs
    assert all(pruned.freq(w) >= 20 for w in pruned.vocab)
    assert all(pruned.freq(w) == idx.freq(w) for w in pruned.vocab)


def test_vocabulary_restriction(random_docs):
    idx = CooccurrenceIndex.build(random_docs, vocabulary={"w0", "w3"})
    assert set(idx.vocab) <= {"w0", "w3"}
    assert idx.n_docs == len(random_docs)


def test_context_vector_ppmi(fruit_index):
    ctx = fruit_index.context_vector("strawberry", size=10, min_df=1, weighting="ppmi")
    # strawberry co-occurs with i, ate, some, fruit, today; 'today' is in every document (PMI 0).
    assert "today" not in ctx
    n, n_s = 100, 10
    expected = np.log(10 * n / (n_s * 40))  # with 'ate'
    assert ctx["ate"] == np.float64(expected)
    assert set(ctx) == {"i", "ate", "some", "fruit"}


def test_from_corpus_file_with_mwus(tmp_path):
    from pyteg import MWUMerger

    corpus = tmp_path / "c.txt"
    corpus.write_text("I live in New York.\nNew York is big.\nYork is old.\n", encoding="utf-8")
    idx = CooccurrenceIndex.from_corpus_file(corpus, merger=MWUMerger([("new", "york")]))
    assert idx.freq("new_york") == 2
    assert idx.freq("york") == 1
    assert idx.meta["mwu"] is True


def test_context_vector_counts(fruit_index):
    ctx = fruit_index.context_vector("strawberry", size=3, min_df=1, exclude={"i"})
    # every strawberry document holds i, ate, some, fruit and today; 'i' is excluded
    assert len(ctx) == 3 and "i" not in ctx
    assert all(v == 1.0 for v in ctx.values())
    try:
        fruit_index.context_vector("strawberry", weighting="tfidf")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown weighting should fail")
