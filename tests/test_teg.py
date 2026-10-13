import pytest

from pyteg import (
    CooccurrenceIndex,
    GeneralityClassifier,
    MEASURES,
    MWUMerger,
    ais_s,
    attributional_generality,
    bidirectional_scores,
    index_association,
    word_generality,
)


def test_specific_text_entails_general_hypothesis(fruit_index):
    clf = GeneralityClassifier(fruit_index, measure="P")
    s = clf.score("I ate a strawberry.", "I ate fruit.")
    assert s.entails and s.direction == "T->H"
    assert clf.score("I ate fruit.", "I ate a strawberry.").direction == "H->T"


def test_all_measures_agree_except_published_conviction(fruit_index):
    clf = GeneralityClassifier(fruit_index)
    scores = clf.score_all("I ate a strawberry.", "I ate fruit.")
    for name, s in scores.items():
        expected = MEASURES[name].orientation > 0
        assert s.entails == expected, name


def test_bidirectional_matches_ais_s(fruit_index):
    t = ["i", "ate", "strawberry", "banana"]
    h = ["ate", "fruit", "today"]
    for name in MEASURES:
        s = bidirectional_scores(fruit_index, t, h, measures=[name])[name]
        assoc = index_association(fruit_index, name)
        assert s.forward == pytest.approx(ais_s(t, h, assoc))
        assert s.backward == pytest.approx(ais_s(h, t, assoc))


def test_no_usable_pairs_means_no_decision(fruit_index):
    s = GeneralityClassifier(fruit_index).score("zzz qqq", "xxx")
    assert s.pairs == 0 and not s.entails and s.direction == "none"


def test_word_level_conditional_probability_is_a_frequency_test(random_docs):
    idx = CooccurrenceIndex.build(random_docs)
    for x, y in [("w0", "w9"), ("w3", "w2"), ("w20", "w21"), ("w1", "w30")]:
        fwd, bwd = word_generality(idx, x, y, "P")
        if idx.joint(x, y):
            assert (fwd > bwd) == (idx.freq(x) > idx.freq(y))


def test_word_generality(fruit_index):
    fwd, bwd = word_generality(fruit_index, "fruit", "strawberry", "BB")
    assert fwd == pytest.approx(1.0) and bwd == pytest.approx(10 / 50)
    assert word_generality(fruit_index, "fruit", "nope") == (0.0, 0.0)


def test_attributional_generality_runs(fruit_index):
    s = attributional_generality(fruit_index, "strawberry", "fruit", "P", size=10, min_df=1)
    assert s.pairs >= 0
    assert s.direction in {"T->H", "H->T", "none"}


def test_mwu_level_warns_without_mwu_index(fruit_index):
    with pytest.warns(UserWarning):
        GeneralityClassifier(fruit_index, level="mwu", merger=MWUMerger([("ate", "some")]))


def test_mwu_level_requires_merger(fruit_index):
    with pytest.raises(ValueError):
        GeneralityClassifier(fruit_index, level="mwu")


def test_no_stopwords_level(fruit_index):
    clf = GeneralityClassifier(fruit_index, measure="P", level="no_stopwords")
    assert clf.preprocess("I ate some fruit") == ["ate", "fruit"]


def test_normalisation_first_uses_lengths(fruit_index):
    t, h = ["i", "ate", "strawberry"], ["fruit"]
    a = bidirectional_scores(fruit_index, t, h, measures=["P"], normalize="none")["conditional_probability"]
    b = bidirectional_scores(fruit_index, t, h, measures=["P"], normalize="first")["conditional_probability"]
    assert b.forward == pytest.approx(a.forward / 9)
    assert b.backward == pytest.approx(a.backward / 1)
