"""Evaluation helpers: binary classification metrics and direction accuracy.

The thesis reports accuracy and the precision of each class, plus their
unweighted and weighted averages. :func:`binary_report` returns all of them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Sequence

__all__ = ["BinaryReport", "binary_report", "direction_accuracy"]


def _safe(num: float, den: float) -> float:
    return num / den if den else 0.0


@dataclass(frozen=True)
class BinaryReport:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return _safe(self.tp + self.tn, self.n)

    @property
    def precision_pos(self) -> float:
        return _safe(self.tp, self.tp + self.fp)

    @property
    def precision_neg(self) -> float:
        return _safe(self.tn, self.tn + self.fn)

    @property
    def recall_pos(self) -> float:
        return _safe(self.tp, self.tp + self.fn)

    @property
    def recall_neg(self) -> float:
        return _safe(self.tn, self.tn + self.fp)

    @property
    def f1_pos(self) -> float:
        p, r = self.precision_pos, self.recall_pos
        return _safe(2 * p * r, p + r)

    @property
    def precision_macro(self) -> float:
        """Unweighted mean of the two class precisions."""
        return (self.precision_pos + self.precision_neg) / 2

    @property
    def precision_weighted(self) -> float:
        """Mean of the class precisions weighted by the true class sizes."""
        pos, neg = self.tp + self.fn, self.tn + self.fp
        return _safe(self.precision_pos * pos + self.precision_neg * neg, pos + neg)

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = dict(asdict(self))
        for name in (
            "accuracy",
            "precision_pos",
            "precision_neg",
            "recall_pos",
            "recall_neg",
            "f1_pos",
            "precision_macro",
            "precision_weighted",
        ):
            out[name] = getattr(self, name)
        return out


def binary_report(gold: Sequence[bool], predicted: Sequence[bool]) -> BinaryReport:
    """Confusion counts (and derived metrics) for two lists of booleans."""
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    tp = fp = fn = tn = 0
    for g, p in zip(gold, predicted):
        if g and p:
            tp += 1
        elif p:
            fp += 1
        elif g:
            fn += 1
        else:
            tn += 1
    return BinaryReport(tp, fp, fn, tn)


def direction_accuracy(margins: Iterable[float]) -> Dict[str, float]:
    """Summarise directional decisions.

    ``margins`` holds, for each item, a value that is positive when the
    method picks the gold direction, negative when it picks the opposite one
    and zero when it cannot decide. Ties count as errors in ``accuracy``;
    ``decided`` is the share of items with a non-zero margin.
    """
    margins = list(margins)
    n = len(margins)
    right = sum(1 for m in margins if m > 0)
    wrong = sum(1 for m in margins if m < 0)
    return {
        "n": n,
        "accuracy": _safe(right, n),
        "decided": _safe(right + wrong, n),
        "accuracy_decided": _safe(right, right + wrong),
    }
