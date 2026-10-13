"""Tests for the helpers of the reproduction pipeline (no downloads needed)."""

import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE))

step2 = pytest.importorskip("step2_prepare_corpora")
step5 = pytest.importorskip("step5_prepare_benchmarks")

DATA_NOUN = """  1 This software and database is being provided to you, the LICENSEE, by
  2 Princeton University under the following license.
00001740 03 n 01 entity 0 001 ~ 00001930 n 0000 | that which is perceived
00001930 03 n 01 fruit 0 002 @ 00001740 n 0000 ~ 00002000 n 0000 | the ripened reproductive body
00002000 13 n 02 strawberry 0 Fragaria_ananassa 0 002 @ 00001930 n 0000 @i 00001740 n 0000 | sweet red berry
00003000 13 n 01 berry 0 001 @ 00002000 n 0000 | an odd loop to test both-way filtering
"""

OMW = """# OpenWN-PT\tpor\turl\tCC BY-SA
00001930-n\tlemma\tfruta
00002000-n\tlemma\tmorango
00002000-n\tpor:lemma\tmorangueiro
00002000-v\tlemma\tnada
"""


def test_windows():
    tokens = [str(i) for i in range(250)]
    chunks = list(step2.windows(tokens, 100, 20))
    assert [len(c) for c in chunks] == [100, 100, 50]
    assert list(step2.windows(tokens[:110], 100, 20)) == [tokens[:100]]


def test_read_floresta(tmp_path):
    (tmp_path / "Floresta_7.4.ptb").write_text(
        "#1 CP1-1 Um revivalismo refrescante\n(UTT+np (H+n revivalismo))\n"
        "#2 CP1-2 O 7_e_Meio é um ex-libris.\n#3 CP2-1 Outra frase.\n",
        encoding="iso-8859-15",
    )
    docs = list(step2.read_floresta(tmp_path))
    assert docs == [("floresta/CP1", "Um revivalismo refrescante O 7 e Meio é um ex-libris."), ("floresta/CP2", "Outra frase.")]


def test_parse_wordnet_and_omw(tmp_path):
    (tmp_path / "data.noun").write_text(DATA_NOUN, encoding="utf-8")
    lemmas, hypernyms = step5.parse_wordnet_nouns(tmp_path / "data.noun")
    assert lemmas["00002000-n"] == ["strawberry", "fragaria_ananassa"]
    assert hypernyms["00002000-n"] == ["00001930-n"]  # the instance pointer (@i) is ignored
    assert hypernyms["00001740-n"] == []
    (tmp_path / "wn-data-por.tab").write_text(OMW, encoding="utf-8")
    pt = step5.parse_omw(tmp_path / "wn-data-por.tab")
    assert pt == {"00001930-n": ["fruta"], "00002000-n": ["morango", "morangueiro"]}


def test_build_hypernym_pairs(tmp_path):
    from pyteg import CooccurrenceIndex

    (tmp_path / "data.noun").write_text(DATA_NOUN, encoding="utf-8")
    lemmas, hypernyms = step5.parse_wordnet_nouns(tmp_path / "data.noun")
    docs = [["strawberry", "fruit"]] * 5 + [["fruit"]] * 20 + [["berry", "strawberry"]] * 3 + [["entity"]] * 3
    index = CooccurrenceIndex.build(docs)
    items, counts = step5.build_hypernym_pairs(lemmas, hypernyms, index, set(), min_df=3, n_pairs=10, seed=1)
    pairs = {(it["hypo"], it["hyper"]) for it in items}
    assert ("strawberry", "fruit") in pairs
    # berry -> strawberry and strawberry -> berry? only one direction exists, so it stays
    assert ("berry", "strawberry") in pairs
    assert ("fruit", "entity") not in pairs  # they never co-occur
    assert counts["sampled"] == len(items)


def test_write_corpus(tmp_path):
    docs = [("a", "One two three four five six."), ("b", "Seven eight")]
    info = step2.write_corpus(iter(docs), tmp_path / "c.txt", size=4, min_size=2)
    assert (tmp_path / "c.txt").read_text(encoding="utf-8").splitlines() == [
        "one two three four",
        "five six",
        "seven eight",
    ]
    assert info == {"source_documents": 2, "windows": 3, "tokens": 8}


def test_read_wikipedia_uses_the_pinned_snapshot(monkeypatch):
    import types

    calls = {}

    def fake_load_dataset(name, config, split, streaming):
        calls.update(name=name, config=config, split=split, streaming=streaming)
        return iter([{"id": "1", "text": "Lisboa é a capital."}, {"id": "2", "text": "Segundo."}, {"id": "3", "text": "x"}])

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    docs = list(step2.read_wikipedia("pt", max_articles=2))
    assert docs == [("wikipedia/1", "Lisboa é a capital."), ("wikipedia/2", "Segundo.")]
    assert calls == {"name": "wikimedia/wikipedia", "config": "20231101.pt", "split": "train", "streaming": True}
