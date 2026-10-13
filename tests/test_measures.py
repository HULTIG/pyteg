import math

import numpy as np
import pytest

from pyteg.measures import MEASURES, Contingency, get_measure


@pytest.fixture
def table():
    # a = 10, n_x = 20, n_y = 40, n = 100
    return Contingency.from_counts(10, 20, 40, 100)


def test_cells(table):
    assert (float(table.a), float(table.b), float(table.c), float(table.d)) == (10, 10, 30, 50)
    assert float(table.n) == 100 and float(table.n_x) == 20 and float(table.n_y) == 40


def test_values_by_hand(table):
    expected = {
        "conditional_probability": 10 / 40,
        "added_value": 10 / 40 - 0.2,
        "braun_blanquet": 10 / (10 + 30),
        "certainty_factor": (0.25 - 0.2) / 0.8,
        "conviction": 0.2 * 0.6 / (10.5 / 100),
        "conviction_yx": 0.4 * 0.8 / (30.5 / 100),
        "gini_index": 0.4 * (0.25**2 + 0.75**2) + 0.6 * ((1 / 6) ** 2 + (5 / 6) ** 2) - 0.2**2 - 0.8**2,
        "j_measure": 0.1 * math.log(0.25 / 0.2) + 0.3 * math.log(0.75 / 0.8),
        "laplace": 11 / 42,
    }
    for name, value in expected.items():
        assert float(MEASURES[name](table)) == pytest.approx(value), name


def test_braun_blanquet_equals_conditional_probability_with_exact_counts():
    rng = np.random.default_rng(0)
    n = 1000
    n_x = rng.integers(1, 500, 200)
    n_y = rng.integers(1, 500, 200)
    n_xy = np.minimum(n_x, n_y) // 2
    t = Contingency.from_counts(n_xy, n_x, n_y, n)
    np.testing.assert_allclose(MEASURES["braun_blanquet"](t), MEASURES["conditional_probability"](t))


def test_orientation():
    # x appears whenever y does, but not the other way round: x is the general word.
    t = Contingency.from_counts(n_xy=10, n_x=50, n_y=10, n=200)
    for m in MEASURES.values():
        forward, backward = float(m(t)), float(m(t.swap()))
        if m.orientation > 0:
            assert forward > backward, m.name
        else:
            assert forward < backward, m.name


def test_vectorised_shapes():
    t = Contingency.from_counts(np.array([[1, 2], [3, 0]]), np.array([[5, 5], [6, 6]]), np.array([[4, 7], [4, 7]]), 50)
    for m in MEASURES.values():
        assert np.asarray(m(t)).shape == (2, 2)


def test_zero_counts_do_not_crash():
    t = Contingency.from_counts(0, 0, 0, 10)
    for m in MEASURES.values():
        assert np.isfinite(m(t)).all(), m.name


def test_invalid_counts():
    with pytest.raises(ValueError):
        Contingency.from_counts(5, 3, 10, 100)
    with pytest.raises(ValueError):
        Contingency.from_counts(0, 60, 60, 100)


def test_lookup():
    assert get_measure("BB").name == "braun_blanquet"
    assert get_measure("braun_blanket").name == "braun_blanquet"
    assert get_measure("p").name == "conditional_probability"
    assert get_measure(" JM ").name == "j_measure"
    with pytest.raises(KeyError):
        get_measure("cosine")


def test_thesis_measures_are_flagged():
    assert sum(m.in_thesis for m in MEASURES.values()) == 8
