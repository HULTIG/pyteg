"""pyteg: textual entailment by generality with asymmetric InfoSimba.

A modern, reproducible implementation of the method proposed in

    Sebastião Pais (2013). Asymmetric Distributional Similarity Measures to
    Recognize Textual Entailment by Generality. PhD thesis, Mines ParisTech
    and Universidade da Beira Interior.

Quick start::

    from pyteg import CooccurrenceIndex, GeneralityClassifier

    index = CooccurrenceIndex.load("data/index/en_words.npz")
    clf = GeneralityClassifier(index, measure="BB")
    clf.score("Mexico City has a very bad pollution problem ...",
              "Poor air circulation ... aggravates pollution.")
"""

from .data import Pair, load_jsonl, load_rte_xml, load_tsv, save_jsonl
from .evaluate import binary_report, direction_accuracy
from .index import CooccurrenceIndex
from .infosimba import ais, ais_s, index_association, infosimba
from .measures import MEASURES, Contingency, Measure, get_measure
from .mwu import MWUMerger, extract_mwus, load_mwus, save_mwus
from .teg import (
    GeneralityClassifier,
    GeneralityScore,
    attributional_generality,
    bidirectional_scores,
    word_generality,
)
from .text import LEVELS, Preprocessor, load_stopwords, tokenize

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CooccurrenceIndex",
    "Contingency",
    "GeneralityClassifier",
    "GeneralityScore",
    "LEVELS",
    "MEASURES",
    "MWUMerger",
    "Measure",
    "Pair",
    "Preprocessor",
    "ais",
    "ais_s",
    "attributional_generality",
    "bidirectional_scores",
    "binary_report",
    "direction_accuracy",
    "extract_mwus",
    "get_measure",
    "index_association",
    "infosimba",
    "load_jsonl",
    "load_mwus",
    "load_rte_xml",
    "load_stopwords",
    "load_tsv",
    "save_jsonl",
    "save_mwus",
    "tokenize",
    "word_generality",
]
