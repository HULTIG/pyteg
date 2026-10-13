import itertools

import numpy as np
import pytest

from pyteg import ais_s, index_association, infosimba
from pyteg.measures import get_measure


def naive_ais_s(index, xi, xj, measure, normalize):
    m = get_measure(measure)
    total, weight = 0.0, 0.0
    xi, xj = list(dict.fromkeys(xi)), list(dict.fromkeys(xj))
    for a, b in itertools.product(xi, xj):
        if a == b or index.freq(a) == 0 or index.freq(b) == 0:
            continue
        table, _ = index.contingency([a], [b])
        total += float(np.asarray(m(table))[0, 0])
        weight += 1.0
    if normalize == "none":
        return total
    if normalize == "first":
        return total / len(xi) ** 2
    return total / weight if weight else 0.0


@pytest.mark.parametrize("normalize", ["pairs", "none", "first"])
@pytest.mark.parametrize("measure", ["P", "AV", "GI", "JM", "CO"])
def test_ais_s_matches_naive_loop(fruit_index, measure, normalize):
    t = ["i", "ate", "strawberry", "today", "unknown"]
    h = ["i", "ate", "fruit"]
    assoc = index_association(fruit_index, measure)
    assert ais_s(t, h, assoc, normalize=normalize) == pytest.approx(naive_ais_s(fruit_index, t, h, measure, normalize))
    assert ais_s(h, t, assoc, normalize=normalize) == pytest.approx(naive_ais_s(fruit_index, h, t, measure, normalize))


def test_infosimba_is_one_for_identical_vectors(fruit_index):
    sim = index_association(fruit_index, "P")
    v = ["fruit", "banana", "today"]
    assert infosimba(v, v, sim) == pytest.approx(1.0)


def test_weighted_vectors(fruit_index):
    assoc = index_association(fruit_index, "P")
    a = ais_s({"strawberry": 2.0, "ate": 1.0}, {"fruit": 1.0}, assoc, normalize="none")
    p_sf = 10 / 50
    p_af = 40 / 50
    assert a == pytest.approx(2.0 * p_sf + 1.0 * p_af)


def test_empty_vectors(fruit_index):
    assoc = index_association(fruit_index, "P")
    assert ais_s([], ["fruit"], assoc) == 0.0
    with pytest.raises(ValueError):
        ais_s(["a"], ["b"], assoc, normalize="bogus")
