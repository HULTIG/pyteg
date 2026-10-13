"""Asymmetric association measures (AAMs).

Every measure is written as ``AS(x || y)``, following the notation of the
thesis and of Pais & Dias (2020). Read it as "how strongly the presence of
``y`` points to ``x``". With the conditional probability, for example,
``AS(x || y) = P(x | y)``: if ``x`` shows up almost every time ``y`` does,
the value is high, and ``x`` behaves as the more general term of the two.

All counts are document counts. ``n_xy`` is the number of documents that
contain both words, ``n_x`` and ``n_y`` the number of documents that contain
each word, and ``n`` the number of documents in the collection. This mirrors
the Web "hits" used in the original experiments.

The functions are vectorised: they accept plain floats or NumPy arrays of the
same shape and return the same kind of object. Undefined values (a division
by zero) come back as ``0.0``, which in practice means "no evidence".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

__all__ = [
    "Contingency",
    "Measure",
    "MEASURES",
    "get_measure",
    "conditional_probability",
    "added_value",
    "braun_blanquet",
    "certainty_factor",
    "conviction",
    "conviction_yx",
    "gini_index",
    "j_measure",
    "laplace",
]


def _div(num: ArrayLike, den: ArrayLike) -> np.ndarray:
    """Element-wise division that returns 0 where the denominator is 0."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.zeros(np.broadcast(num, den).shape, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _xlogy_ratio(p: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Compute ``p * log(ratio)`` with the usual convention ``0 log 0 = 0``."""
    out = np.zeros(np.broadcast(p, ratio).shape, dtype=float)
    mask = (p > 0) & (ratio > 0)
    np.multiply(p, np.log(ratio, where=mask, out=np.ones_like(out)), out=out, where=mask)
    return out


@dataclass(frozen=True)
class Contingency:
    """A 2x2 table of document counts for the pair ``(x, y)``.

    The four cells are::

                   y        not y
        x          a          b
        not x      c          d

    Build it with :meth:`from_counts`, which takes the joint and marginal
    counts and optionally adds a pseudo-count to every cell.
    """

    a: np.ndarray  # x and y
    b: np.ndarray  # x, not y
    c: np.ndarray  # y, not x
    d: np.ndarray  # neither

    @classmethod
    def from_counts(
        cls,
        n_xy: ArrayLike,
        n_x: ArrayLike,
        n_y: ArrayLike,
        n: ArrayLike,
        smoothing: float = 0.0,
    ) -> "Contingency":
        n_xy = np.asarray(n_xy, dtype=float)
        n_x = np.asarray(n_x, dtype=float)
        n_y = np.asarray(n_y, dtype=float)
        n = np.asarray(n, dtype=float)
        if np.any(n_xy > np.minimum(n_x, n_y)):
            raise ValueError("joint count larger than one of the marginal counts")
        if np.any(n_x + n_y - n_xy > n):
            raise ValueError("marginal counts do not fit in the collection size")
        a = n_xy + smoothing
        b = n_x - n_xy + smoothing
        c = n_y - n_xy + smoothing
        d = n - n_x - n_y + n_xy + smoothing
        return cls(a, b, c, d)

    # Margins -----------------------------------------------------------
    @property
    def n(self) -> np.ndarray:
        return self.a + self.b + self.c + self.d

    @property
    def n_x(self) -> np.ndarray:
        return self.a + self.b

    @property
    def n_y(self) -> np.ndarray:
        return self.a + self.c

    # Probabilities -------------------------------------------------------
    @property
    def p_x(self) -> np.ndarray:
        return _div(self.n_x, self.n)

    @property
    def p_y(self) -> np.ndarray:
        return _div(self.n_y, self.n)

    @property
    def p_not_x(self) -> np.ndarray:
        return _div(self.c + self.d, self.n)

    @property
    def p_not_y(self) -> np.ndarray:
        return _div(self.b + self.d, self.n)

    def swap(self) -> "Contingency":
        """Return the table for ``(y, x)``."""
        return Contingency(self.a, self.c, self.b, self.d)


# ---------------------------------------------------------------------------
# The eight measures used in the thesis (plus one orientation-consistent
# variant of Conviction). Formulas follow Pais, Dias & Moraliyski (2014) and
# Pais & Dias (2020); see docs/method.md for the discussion of each one.
# ---------------------------------------------------------------------------


def conditional_probability(t: Contingency) -> np.ndarray:
    """``P(x | y) = P(x, y) / P(y)``."""
    return _div(t.a, t.n_y)


def added_value(t: Contingency) -> np.ndarray:
    """``AV(x || y) = P(x | y) - P(x)``."""
    return conditional_probability(t) - t.p_x


def braun_blanquet(t: Contingency) -> np.ndarray:
    """``BB(x || y) = f(x, y) / (f(x, y) + f(not x, y))``, as published.

    With exact document counts ``f(x, y) + f(not x, y)`` is simply ``f(y)``,
    so this formula gives the same value as :func:`conditional_probability`.
    The two only differ in the thesis because Web hit counts are estimates
    that do not add up. We keep the published definition on purpose and
    report the coincidence in the documentation.
    """
    return _div(t.a, t.a + t.c)


def certainty_factor(t: Contingency) -> np.ndarray:
    """``CF(x || y) = (P(x | y) - P(x)) / (1 - P(x))``."""
    return _div(conditional_probability(t) - t.p_x, 1.0 - t.p_x)


def _conviction(p_x, p_not_y, p_x_not_y):
    return _div(p_x * p_not_y, p_x_not_y)


def conviction(t: Contingency) -> np.ndarray:
    """``CO(x || y) = P(x) P(not y) / P(x, not y)``, as published.

    Unlike the other measures, this one is large when ``x`` implies ``y``,
    that is, when ``y`` is the more general term. A pseudo-count of 0.5 is
    added to the ``(x, not y)`` cell so that the value stays finite when
    ``x`` never occurs without ``y``.
    """
    p_x_not_y = _div(t.b + 0.5, t.n)
    return _conviction(t.p_x, t.p_not_y, p_x_not_y)


def conviction_yx(t: Contingency) -> np.ndarray:
    """Conviction with the same orientation as the other measures.

    ``CO'(x || y) = P(y) P(not x) / P(y, not x)``. This variant is not in the
    thesis; we add it so that Conviction can be compared on equal terms.
    """
    p_y_not_x = _div(t.c + 0.5, t.n)
    return _conviction(t.p_y, t.p_not_x, p_y_not_x)


def gini_index(t: Contingency) -> np.ndarray:
    """Gini index of the rule ``y -> x`` (Tan, Kumar & Srivastava, 2004).

    ``P(y)[P(x|y)^2 + P(not x|y)^2] + P(not y)[P(x|not y)^2 + P(not x|not y)^2]
    - P(x)^2 - P(not x)^2``
    """
    n_not_y = t.b + t.d
    p_x_y = _div(t.a, t.n_y)
    p_nx_y = _div(t.c, t.n_y)
    p_x_ny = _div(t.b, n_not_y)
    p_nx_ny = _div(t.d, n_not_y)
    return (
        t.p_y * (p_x_y**2 + p_nx_y**2)
        + t.p_not_y * (p_x_ny**2 + p_nx_ny**2)
        - t.p_x**2
        - t.p_not_x**2
    )


def j_measure(t: Contingency) -> np.ndarray:
    """``JM(x || y) = P(x,y) log(P(x|y)/P(x)) + P(not x,y) log(P(not x|y)/P(not x))``.

    Natural logarithm. The base does not change any decision taken by the
    package, since it only rescales the values.
    """
    p_xy = _div(t.a, t.n)
    p_nxy = _div(t.c, t.n)
    ratio_x = _div(_div(t.a, t.n_y), t.p_x)
    ratio_nx = _div(_div(t.c, t.n_y), t.p_not_x)
    return _xlogy_ratio(p_xy, ratio_x) + _xlogy_ratio(p_nxy, ratio_nx)


def laplace(t: Contingency) -> np.ndarray:
    """``LP(x || y) = (N P(x, y) + 1) / (N P(y) + 2)``."""
    return _div(t.a + 1.0, t.n_y + 2.0)


@dataclass(frozen=True)
class Measure:
    """A named asymmetric association measure.

    ``orientation`` is +1 when a high ``AS(x || y)`` means that ``x`` is the
    more general word, and -1 when it means that ``y`` is.
    """

    name: str
    short: str
    func: Callable[[Contingency], np.ndarray]
    orientation: int = 1
    in_thesis: bool = True

    def __call__(self, table: Contingency) -> np.ndarray:
        return self.func(table)


MEASURES: Dict[str, Measure] = {
    m.name: m
    for m in [
        Measure("added_value", "AV", added_value),
        Measure("braun_blanquet", "BB", braun_blanquet),
        Measure("certainty_factor", "CF", certainty_factor),
        Measure("conditional_probability", "P", conditional_probability),
        Measure("conviction", "CO", conviction, orientation=-1),
        Measure("gini_index", "GI", gini_index),
        Measure("j_measure", "JM", j_measure),
        Measure("laplace", "LP", laplace),
        Measure("conviction_yx", "CO'", conviction_yx, in_thesis=False),
    ]
}

_ALIASES = {m.short.lower(): m.name for m in MEASURES.values()}
_ALIASES.update({"bb": "braun_blanquet", "braun_blanket": "braun_blanquet", "p": "conditional_probability"})


def get_measure(name: Union[str, Measure]) -> Measure:
    """Look a measure up by name (``"braun_blanquet"``) or short code (``"BB"``)."""
    if isinstance(name, Measure):
        return name
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return MEASURES[key]
    except KeyError:
        known = ", ".join(sorted(MEASURES))
        raise KeyError(f"unknown measure {name!r}; choose one of: {known}") from None
