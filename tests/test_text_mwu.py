import pytest

from pyteg import MWUMerger, Preprocessor, extract_mwus, load_mwus, load_stopwords, save_mwus, tokenize


def test_tokenize():
    assert tokenize("Mexico City's air isn't clean, is it?") == ["mexico", "city's", "air", "isn't", "clean", "is", "it"]
    assert tokenize("O João comeu pão-de-ló às 10h.") == ["o", "joão", "comeu", "pão-de-ló", "às", "10h"]
    assert tokenize("Don’t") == ["don't"]


def test_stopwords():
    en, pt = load_stopwords("en"), load_stopwords("portuguese")
    assert {"the", "of", "a"} <= en and "fruit" not in en
    assert {"de", "não", "que"} <= pt
    with pytest.raises(ValueError):
        load_stopwords("klingon")


def test_preprocessor_levels():
    merger = MWUMerger([("new", "york")])
    text = "The mayor of New York spoke."
    assert Preprocessor("all_words")(text) == ["the", "mayor", "of", "new", "york", "spoke"]
    assert Preprocessor("no_stopwords")(text) == ["mayor", "new", "york", "spoke"]
    assert Preprocessor("mwu", merger=merger)(text) == ["the", "mayor", "of", "new_york", "spoke"]
    with pytest.raises(ValueError):
        Preprocessor("lemmas")


def test_merger_prefers_longest_unit():
    m = MWUMerger([("new", "york"), ("new", "york", "city"), ("city", "hall")])
    assert m.merge("new york city hall".split()) == ["new_york_city", "hall"]
    assert m.merge("new york".split()) == ["new_york"]
    assert m.merge("york new".split()) == ["york", "new"]
    assert len(m) == 3


def test_load_and_save(tmp_path):
    p = tmp_path / "u.txt"
    save_mwus([(("new", "york"), 0.5), (("hot", "dog"), 0.25)], p)
    assert load_mwus(p) == [("new", "york"), ("hot", "dog")]
    p.write_text("# comment\nNew York\nsingle\n\nhot dog\textra\n", encoding="utf-8")
    assert load_mwus(p) == [("new", "york"), ("hot", "dog")]


def test_extractor_finds_obvious_units():
    docs = []
    for i in range(30):
        docs.append(f"yesterday in new york the stock market fell number{i}".split())
        docs.append(f"a hot dog costs more in new york than in city{i}".split())
        docs.append(f"the stock market and the weather word{i}".split())
    units = dict(extract_mwus(docs, max_n=3, min_freq=5, stopwords=load_stopwords("en")))
    assert ("new", "york") in units
    assert ("stock", "market") in units
    assert all(u[0] not in load_stopwords("en") and u[-1] not in load_stopwords("en") for u in units)


def test_extractor_rejects_bad_arguments():
    with pytest.raises(ValueError):
        extract_mwus([["a", "b"]], max_n=1)
    assert extract_mwus([]) == []
