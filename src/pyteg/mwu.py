"""Multiword units: merging them into tokens, and a small built-in extractor.

The thesis used SENTA (Dias, Guilloré & Lopes, 1999) to find multiword units.
SENTA now lives on as a separate package (pysenta). pyteg does not depend on
it: any list of units, one per line, can be loaded with :func:`load_mwus`.

For people who just want to run the pipeline end to end, :func:`extract_mwus`
implements the LocalMaxs algorithm with the SCP_f glue (Silva & Lopes, 1999)
over contiguous n-grams. It is a reasonable stand-in, not a replacement for
SENTA, and the reproduction report says which extractor produced each result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

__all__ = ["MWUMerger", "load_mwus", "save_mwus", "extract_mwus", "JOINER"]

#: Character used to glue the words of a unit into one token.
JOINER = "_"

Unit = Tuple[str, ...]


def load_mwus(path: Union[str, Path]) -> List[Unit]:
    """Read multiword units from a text file.

    One unit per line, words separated by spaces. Empty lines and lines that
    start with ``#`` are ignored. Anything after a tab is ignored too, so a
    file with ``unit<TAB>score`` lines works as is.
    """
    units: List[Unit] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.split("\t", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            words = tuple(line.lower().split())
            if len(words) >= 2:
                units.append(words)
    return units


def save_mwus(units: Iterable[Union[Unit, Tuple[Unit, float]]], path: Union[str, Path]) -> None:
    """Write units (optionally with a score) to ``path``, one per line."""
    with open(path, "w", encoding="utf-8") as handle:
        for item in units:
            if item and isinstance(item[0], tuple):
                unit, score = item  # type: ignore[misc]
                handle.write(" ".join(unit) + f"\t{score:.6g}\n")
            else:
                handle.write(" ".join(item) + "\n")  # type: ignore[arg-type]


class MWUMerger:
    """Replace known multiword units in a token list by single tokens.

    Matching is greedy and left to right, always taking the longest unit that
    starts at the current position.

    >>> MWUMerger([("new", "york"), ("new", "york", "city")]).merge("i love new york city".split())
    ['i', 'love', 'new_york_city']
    """

    def __init__(self, units: Iterable[Sequence[str]], joiner: str = JOINER) -> None:
        self.joiner = joiner
        self.units: Set[Unit] = {tuple(w.lower() for w in u) for u in units if len(u) >= 2}
        self.max_len = max((len(u) for u in self.units), default=0)
        self._first: Set[str] = {u[0] for u in self.units}

    def __len__(self) -> int:
        return len(self.units)

    def merge(self, tokens: Sequence[str]) -> List[str]:
        out: List[str] = []
        i, n = 0, len(tokens)
        while i < n:
            if tokens[i] in self._first:
                for size in range(min(self.max_len, n - i), 1, -1):
                    candidate = tuple(tokens[i : i + size])
                    if candidate in self.units:
                        out.append(self.joiner.join(candidate))
                        i += size
                        break
                else:
                    out.append(tokens[i])
                    i += 1
            else:
                out.append(tokens[i])
                i += 1
        return out

    __call__ = merge


# ---------------------------------------------------------------------------
# LocalMaxs + SCP_f
# ---------------------------------------------------------------------------


def _ngrams(doc: Sequence[int], n: int) -> Iterator[Tuple[int, ...]]:
    for i in range(len(doc) - n + 1):
        yield tuple(doc[i : i + n])


def extract_mwus(
    documents: Iterable[Sequence[str]],
    max_n: int = 3,
    min_freq: int = 5,
    stopwords: Optional[Iterable[str]] = None,
    alphabetic_only: bool = True,
) -> List[Tuple[Unit, float]]:
    """Find multiword units with LocalMaxs and the SCP_f glue.

    Parameters
    ----------
    documents:
        Tokenised documents. n-grams never cross document boundaries.
    max_n:
        Longest unit to return. Counts are collected up to ``max_n + 1``
        because LocalMaxs looks one word further.
    min_freq:
        n-grams seen fewer times than this are ignored.
    stopwords:
        If given, units that start or end with one of these words are dropped
        (stop words are still allowed inside a unit, as in *bank of america*).
    alphabetic_only:
        Drop units containing tokens without any letter (numbers, codes).

    Returns
    -------
    A list of ``(unit, glue)`` pairs, sorted by decreasing glue.
    """
    if max_n < 2:
        raise ValueError("max_n must be at least 2")
    stop = frozenset(stopwords or ())

    # Map words to integers once; it makes the n-gram dictionaries far smaller.
    vocab: Dict[str, int] = {}
    docs: List[List[int]] = []
    for doc in documents:
        docs.append([vocab.setdefault(w, len(vocab)) for w in doc])
    words = [""] * len(vocab)
    for w, i in vocab.items():
        words[i] = w

    total = sum(len(d) for d in docs)
    if total == 0:
        return []

    # Frequent n-grams, order by order. An n-gram can only be frequent if its
    # prefix and suffix of length n-1 are frequent, so we only count those.
    freq: Dict[int, Counter] = {1: Counter()}
    for d in docs:
        freq[1].update((w,) for w in d)
    freq[1] = Counter({g: c for g, c in freq[1].items() if c >= min_freq})
    for n in range(2, max_n + 2):
        prev = freq[n - 1]
        counts: Counter = Counter()
        for d in docs:
            for g in _ngrams(d, n):
                if g[:-1] in prev and g[1:] in prev:
                    counts[g] += 1
        freq[n] = Counter({g: c for g, c in counts.items() if c >= min_freq})
        if not freq[n]:
            break

    def prob(g: Tuple[int, ...]) -> float:
        return freq[len(g)].get(g, 0) / total

    # SCP_f glue for every frequent n-gram with n >= 2.
    glue: Dict[Tuple[int, ...], float] = {}
    for n, counter in freq.items():
        if n < 2:
            continue
        for g in counter:
            avp = sum(prob(g[:i]) * prob(g[i:]) for i in range(1, n)) / (n - 1)
            if avp > 0:
                glue[g] = prob(g) ** 2 / avp

    # Highest glue among the (n+1)-grams that contain each n-gram.
    sup_max: Dict[Tuple[int, ...], float] = defaultdict(float)
    for g, value in glue.items():
        for sub in (g[:-1], g[1:]):
            if len(sub) >= 2 and value > sup_max[sub]:
                sup_max[sub] = value

    units: List[Tuple[Unit, float]] = []
    for g, value in glue.items():
        n = len(g)
        if n > max_n:
            continue
        if not value > sup_max.get(g, 0.0):
            continue
        if n >= 3 and value < max(glue.get(g[:-1], 0.0), glue.get(g[1:], 0.0)):
            continue
        unit = tuple(words[i] for i in g)
        if stop and (unit[0] in stop or unit[-1] in stop):
            continue
        if alphabetic_only and not all(any(ch.isalpha() for ch in w) for w in unit):
            continue
        units.append((unit, value))

    units.sort(key=lambda item: (-item[1], item[0]))
    return units
