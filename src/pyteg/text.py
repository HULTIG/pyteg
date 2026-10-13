"""Tokenisation, stop words and the three pre-processing levels of the thesis.

The tokenizer is deliberately plain: lower-cased runs of letters and digits,
with internal hyphens and apostrophes kept. It has no language-specific rules,
which is in the spirit of the original method and keeps the corpus counts and
the sentence vectors consistent with each other.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from typing import FrozenSet, Iterable, List, Optional, Sequence

__all__ = [
    "LEVELS",
    "tokenize",
    "load_stopwords",
    "Preprocessor",
]

#: The three pre-processing levels compared in the thesis.
LEVELS = ("all_words", "no_stopwords", "mwu")

_TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*", re.UNICODE)

_LANG_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "pt": "pt",
    "por": "pt",
    "portuguese": "pt",
    "português": "pt",
}


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Split ``text`` into word tokens.

    >>> tokenize("Mexico City's air isn't clean, is it?")
    ['mexico', "city's", 'air', "isn't", 'clean', 'is', 'it']
    """
    if lowercase:
        text = text.lower()
    return [m.group(0).replace("’", "'") for m in _TOKEN_RE.finditer(text)]


@lru_cache(maxsize=None)
def load_stopwords(lang: str) -> FrozenSet[str]:
    """Return the bundled Snowball stop-word list for ``"en"`` or ``"pt"``."""
    code = _LANG_ALIASES.get(lang.strip().lower())
    if code is None:
        raise ValueError(f"no bundled stop-word list for language {lang!r} (use 'en' or 'pt')")
    text = resources.files("pyteg.resources").joinpath(f"stopwords_{code}.txt").read_text("utf-8")
    words = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
    return frozenset(words)


class Preprocessor:
    """Turn a sentence into the token list used by the method.

    ``level`` is one of:

    * ``"all_words"``: every token is kept.
    * ``"no_stopwords"``: tokens in the stop-word list are dropped.
    * ``"mwu"``: multiword units are merged into single tokens (for example
      ``new york`` becomes ``new_york``). Stop words are kept, as in the
      thesis, where units such as *with a* appear in the worked example.

    The ``"mwu"`` level needs an :class:`pyteg.mwu.MWUMerger`. Remember that
    the co-occurrence index must be built from a corpus merged with the same
    units, otherwise the merged tokens will not be found in it.
    """

    def __init__(
        self,
        level: str = "all_words",
        lang: str = "en",
        stopwords: Optional[Iterable[str]] = None,
        merger=None,
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        if level == "mwu" and merger is None:
            raise ValueError("the 'mwu' level needs an MWUMerger (see pyteg.mwu)")
        self.level = level
        self.lang = lang
        self.stopwords = frozenset(stopwords) if stopwords is not None else load_stopwords(lang)
        self.merger = merger

    def tokens(self, text_or_tokens) -> List[str]:
        tokens: Sequence[str]
        if isinstance(text_or_tokens, str):
            tokens = tokenize(text_or_tokens)
        else:
            tokens = list(text_or_tokens)
        if self.level == "no_stopwords":
            return [t for t in tokens if t not in self.stopwords]
        if self.level == "mwu":
            return self.merger.merge(tokens)
        return list(tokens)

    __call__ = tokens

    def __repr__(self) -> str:
        return f"Preprocessor(level={self.level!r}, lang={self.lang!r})"
