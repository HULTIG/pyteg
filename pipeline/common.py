"""Shared settings for the reproduction pipeline.

All paths are relative to the repository root, so the scripts can be run from
anywhere. Everything the pipeline downloads or produces goes under ``data/``
(ignored by git), except the final tables, which go under ``results/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "corpus"
MWU = DATA / "mwu"
INDEX = DATA / "index"
BENCH = DATA / "bench"
RESULTS = ROOT / "results"
SOURCES = Path(__file__).resolve().parent / "sources.json"

LANGS = ("en", "pt")

# Fixed choices of the reference run. Change them on the command line, not here,
# so that results/REPORT.md always describes what was actually run.
WINDOW = 100  # tokens per pseudo-document in the counts corpus
MIN_WINDOW = 20  # a trailing window shorter than this is dropped
MWU_MAX_N = 3
MWU_MIN_FREQ = 10
SEED = 2013


def load_sources() -> dict:
    return json.loads(SOURCES.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for d in (RAW, CORPUS, MWU, INDEX, BENCH, RESULTS):
        d.mkdir(parents=True, exist_ok=True)


def index_path(lang: str, level: str) -> Path:
    """The index used by a pre-processing level ('mwu' has its own)."""
    kind = "mwu" if level == "mwu" else "words"
    return INDEX / f"{lang}_{kind}.npz"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class Timer:
    """Tiny context manager that prints how long a step took."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"[{self.label}] started", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"[{self.label}] done in {time.perf_counter() - self.start:.1f}s", flush=True)
        return False
