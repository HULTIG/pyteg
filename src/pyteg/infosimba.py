"""InfoSimba and its asymmetric versions, AIS and AISs.

InfoSimba (Dias, Alves & Lopes, 2007) compares two vectors of words by
looking at how similar *every* word of one is to *every* word of the other,
instead of counting exact matches only. Pais (2013) made it asymmetric by
replacing the symmetric similarity with an asymmetric association measure.

A vector is either a sequence of words (each with weight 1, repeats ignored)
or a mapping ``{word: weight}``.

The association is passed in as a function ``assoc(xs, ys)`` that returns a
pair ``(values, valid)``: a matrix with ``AS(xs[k] || ys[l])`` in cell
``[k, l]`` and a boolean matrix of the same shape saying which cells are
usable (for instance, both words are present in the corpus).
:func:`index_association` builds such a function from a co-occurrence index.
"""

from __future__ import annotations

from typing import Callable, List, Mapping, Sequence, Tuple, Union

import numpy as np

from .measures import Measure, get_measure

__all__ = [
    "Vector",
    "Association",
    "as_weighted",
    "index_association",
    "infosimba",
    "ais",
    "ais_s",
    "NORMALIZATIONS",
]

Vector = Union[Sequence[str], Mapping[str, float]]
Association = Callable[[Sequence[str], Sequence[str]], Tuple[np.ndarray, np.ndarray]]

#: Ways of scaling AISs. See :func:`ais_s`.
NORMALIZATIONS = ("pairs", "none", "first")


def as_weighted(vector: Vector) -> Tuple[List[str], np.ndarray]:
    """Return the words of ``vector`` and their weights as an array."""
    if isinstance(vector, Mapping):
        words = list(vector.keys())
        weights = np.array([float(vector[w]) for w in words], dtype=float)
    else:
        seen = dict.fromkeys(vector)
        words = list(seen)
        weights = np.ones(len(words), dtype=float)
    return words, weights


def index_association(index, measure: Union[str, Measure], smoothing: float = 0.0) -> Association:
    """Association function backed by a :class:`~pyteg.index.CooccurrenceIndex`."""
    m = get_measure(measure)

    def assoc(xs: Sequence[str], ys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        table, known = index.contingency(xs, ys, smoothing=smoothing)
        return np.asarray(m(table), dtype=float), known

    return assoc


def infosimba(xi: Vector, xj: Vector, sim: Association) -> float:
    """Symmetric InfoSimba with similarity ``sim`` (Dias et al., 2007).

    ``IS = A / (B + C - A)`` where ``A`` is the weighted similarity between the
    two vectors and ``B`` and ``C`` the weighted similarity of each vector
    with itself.
    """
    wi, ai = as_weighted(xi)
    wj, aj = as_weighted(xj)
    if not wi or not wj:
        return 0.0

    def block(x_words, x_w, y_words, y_w):
        values, valid = sim(x_words, y_words)
        return float(np.sum(np.outer(x_w, y_w) * np.where(valid, values, 0.0)))

    a = block(wi, ai, wj, aj)
    b = block(wi, ai, wi, ai)
    c = block(wj, aj, wj, aj)
    den = b + c - a
    return a / den if den else 0.0


def ais(xi: Vector, xj: Vector, assoc: Association) -> float:
    """Asymmetric InfoSimba, the full formula (Pais, 2013).

    ``AIS(Xi || Xj) = A / (B + C - A)`` with every similarity replaced by the
    asymmetric association ``AS(. || .)``. The denominator mixes both
    directions; the thesis drops it in practice and uses :func:`ais_s`.
    """
    return infosimba(xi, xj, assoc)


def ais_s(
    xi: Vector,
    xj: Vector,
    assoc: Association,
    normalize: str = "pairs",
    skip_identical: bool = True,
) -> float:
    """Simplified Asymmetric InfoSimba, ``AISs(Xi || Xj)``.

    ``sum_k sum_l X_ik * X_jl * AS(W_ik || W_jl)`` over pairs of different
    words (``W_ik != W_jl``) whose association is defined.

    ``normalize`` controls the scaling:

    * ``"pairs"`` (default): divide by the sum of the pair weights, which
      gives a weighted mean. Both directions share the same pairs, so the
      decision is exactly the one taken with the raw sum.
    * ``"none"``: the raw sum, as in Pais, Dias & Moraliyski (2014).
    * ``"first"``: divide by ``p^2`` with ``p`` the length of ``Xi``, our
      reading of Eq. 15 in Pais & Dias (2020). This one *can* change
      decisions when the two vectors have different lengths.
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}")
    wi, ai = as_weighted(xi)
    wj, aj = as_weighted(xj)
    if not wi or not wj:
        return 0.0
    values, valid = assoc(wi, wj)
    mask = np.asarray(valid, dtype=bool)
    if skip_identical:
        same = np.array([[a == b for b in wj] for a in wi], dtype=bool)
        mask = mask & ~same
    weights = np.outer(ai, aj) * mask
    total = float(np.sum(weights * np.where(mask, values, 0.0)))
    if normalize == "none":
        return total
    if normalize == "first":
        return total / (len(wi) ** 2)
    norm = float(np.sum(weights))
    return total / norm if norm else 0.0
