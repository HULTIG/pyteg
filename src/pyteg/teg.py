"""Recognising textual entailment by generality.

The decision rule is the one of the thesis:

    T entails H by generality  if and only if  AISs(T || H) < AISs(H || T)

With ``AS(x || y) = P(x | y)``, ``AISs(T || H)`` averages how often the words
of T show up when the words of H do. If H is the more general sentence, its
words are frequent companions of the words of T but not the other way round,
so the left side is the smaller one. No threshold is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .index import CooccurrenceIndex
from .infosimba import NORMALIZATIONS, Vector, as_weighted
from .measures import MEASURES, Measure, get_measure
from .text import Preprocessor

__all__ = [
    "GeneralityScore",
    "GeneralityClassifier",
    "bidirectional_scores",
    "word_generality",
    "attributional_generality",
]


@dataclass(frozen=True)
class GeneralityScore:
    """Both directions of AISs for one pair, and the decision they imply."""

    forward: float  # AISs(T || H)
    backward: float  # AISs(H || T)
    pairs: int  # word pairs that took part in the sums

    @property
    def entails(self) -> bool:
        """``True`` when T entails H by generality (H is the more general one)."""
        return self.pairs > 0 and self.forward < self.backward

    @property
    def direction(self) -> str:
        """``"T->H"``, ``"H->T"`` or ``"none"`` (tie or no usable word pair)."""
        if self.pairs == 0 or self.forward == self.backward:
            return "none"
        return "T->H" if self.forward < self.backward else "H->T"

    @property
    def margin(self) -> float:
        """``backward - forward``; positive values favour T entailing H."""
        return self.backward - self.forward


def bidirectional_scores(
    index: CooccurrenceIndex,
    t: Vector,
    h: Vector,
    measures: Iterable[Union[str, Measure]] = tuple(MEASURES),
    normalize: str = "pairs",
    smoothing: float = 0.0,
    skip_identical: bool = True,
) -> Dict[str, GeneralityScore]:
    """AISs in both directions for several measures at once.

    The co-occurrence counts are fetched a single time and shared by all the
    measures, which is what makes it cheap to compare them.
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}")
    measures = [get_measure(m) for m in measures]
    tw, ta = as_weighted(t)
    hw, ha = as_weighted(h)
    if not tw or not hw:
        return {m.name: GeneralityScore(0.0, 0.0, 0) for m in measures}

    table, known = index.contingency(tw, hw, smoothing=smoothing)
    mask = known.copy()
    if skip_identical:
        same = np.array([[a == b for b in hw] for a in tw], dtype=bool)
        mask &= ~same
    weights = np.outer(ta, ha) * mask
    n_pairs = int(mask.sum())
    swapped = table.swap()

    out: Dict[str, GeneralityScore] = {}
    for m in measures:
        fwd = np.where(mask, m(table), 0.0)  # AS(t_k || h_l)
        bwd = np.where(mask, m(swapped), 0.0)  # AS(h_l || t_k)
        f = float(np.sum(weights * fwd))
        b = float(np.sum(weights * bwd))
        if normalize == "pairs":
            norm = float(np.sum(weights))
            f, b = (f / norm, b / norm) if norm else (0.0, 0.0)
        elif normalize == "first":
            f, b = f / len(tw) ** 2, b / len(hw) ** 2
        out[m.name] = GeneralityScore(f, b, n_pairs)
    return out


class GeneralityClassifier:
    """Decide whether a text T entails a hypothesis H by generality.

    Parameters
    ----------
    index:
        Co-occurrence counts. For the ``"mwu"`` level it must have been built
        from a corpus merged with the same multiword units.
    measure:
        Name or short code of the asymmetric association measure
        (see :data:`pyteg.measures.MEASURES`).
    level:
        Pre-processing level: ``"all_words"``, ``"no_stopwords"`` or ``"mwu"``.
    lang:
        Language of the stop-word list (``"en"`` or ``"pt"``).
    merger:
        A :class:`pyteg.mwu.MWUMerger`, needed for the ``"mwu"`` level.
    normalize, smoothing:
        Passed on to :func:`bidirectional_scores`.

    Example
    -------
    >>> clf = GeneralityClassifier(index, measure="BB")  # doctest: +SKIP
    >>> clf.score("I ate a strawberry.", "I ate fruit.").entails  # doctest: +SKIP
    True
    """

    def __init__(
        self,
        index: CooccurrenceIndex,
        measure: Union[str, Measure] = "braun_blanquet",
        level: str = "all_words",
        lang: str = "en",
        merger=None,
        normalize: str = "pairs",
        smoothing: float = 0.0,
    ) -> None:
        self.index = index
        self.measure = get_measure(measure)
        self.preprocess = Preprocessor(level=level, lang=lang, merger=merger)
        self.normalize = normalize
        self.smoothing = smoothing
        if level == "mwu" and not index.meta.get("mwu", False):
            import warnings

            warnings.warn(
                "the 'mwu' level is used with an index that was not built from an MWU-merged corpus; "
                "merged tokens will be treated as unknown words",
                stacklevel=2,
            )

    @property
    def level(self) -> str:
        return self.preprocess.level

    def score(self, t: Union[str, Sequence[str]], h: Union[str, Sequence[str]]) -> GeneralityScore:
        return self.score_all(t, h, measures=[self.measure])[self.measure.name]

    def score_all(
        self,
        t: Union[str, Sequence[str]],
        h: Union[str, Sequence[str]],
        measures: Iterable[Union[str, Measure]] = tuple(MEASURES),
    ) -> Dict[str, GeneralityScore]:
        return bidirectional_scores(
            self.index,
            self.preprocess(t),
            self.preprocess(h),
            measures=measures,
            normalize=self.normalize,
            smoothing=self.smoothing,
        )

    def entails(self, t, h) -> bool:
        return self.score(t, h).entails

    def predict(self, pairs: Iterable[Tuple[str, str]]) -> List[bool]:
        return [self.score(t, h).entails for t, h in pairs]


# ---------------------------------------------------------------------------
# Word level
# ---------------------------------------------------------------------------


def word_generality(
    index: CooccurrenceIndex, x: str, y: str, measure: Union[str, Measure] = "braun_blanquet", smoothing: float = 0.0
) -> Tuple[float, float]:
    """``(AS(x || y), AS(y || x))`` for two words.

    For a measure with orientation +1, ``x`` is taken as the more general word
    when the first value is the larger one.

    Note that for the conditional probability (and Braun-Blanquet as
    published) this comparison is the same as comparing document
    frequencies, since ``P(x|y) / P(y|x) = n(x) / n(y)``.
    """
    m = get_measure(measure)
    table, known = index.contingency([x], [y], smoothing=smoothing)
    if not known[0, 0]:
        return 0.0, 0.0
    return float(np.asarray(m(table))[0, 0]), float(np.asarray(m(table.swap()))[0, 0])


def attributional_generality(
    index: CooccurrenceIndex,
    x: str,
    y: str,
    measure: Union[str, Measure] = "braun_blanquet",
    size: int = 50,
    min_df: int = 5,
    normalize: str = "pairs",
    cache: Optional[Dict[str, Dict[str, float]]] = None,
) -> GeneralityScore:
    """Compare the distributional profiles of two words with AISs.

    Each word is represented by its ``size`` strongest context words (PPMI
    weights). The score follows the sentence rule: with ``x`` as "T" and ``y``
    as "H", ``entails`` means that ``y`` is the more general word.
    """
    cache = {} if cache is None else cache

    def profile(w: str) -> Dict[str, float]:
        if w not in cache:
            cache[w] = index.context_vector(w, size=size, min_df=min_df)
        return cache[w]

    cx = {k: v for k, v in profile(x).items() if k not in (x, y)}
    cy = {k: v for k, v in profile(y).items() if k not in (x, y)}
    m = get_measure(measure)
    return bidirectional_scores(index, cx, cy, measures=[m], normalize=normalize)[m.name]
