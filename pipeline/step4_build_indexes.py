"""Step 4. Build the co-occurrence indexes.

Usage:  python pipeline/step4_build_indexes.py

Two indexes per language:

* data/index/<lang>_words.npz, from the plain corpus. It serves the
  "all_words" and "no_stopwords" levels (stop words are removed from the
  sentences, not from the counts).
* data/index/<lang>_mwu.npz, from the corpus after merging the units of
  step 3. It serves the "mwu" level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS, INDEX, MWU, Timer, ensure_dirs, write_json  # noqa: E402

from pyteg import CooccurrenceIndex, MWUMerger, load_mwus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", nargs="+", default=["en", "pt"])
    parser.add_argument("--min-df", type=int, default=1)
    args = parser.parse_args()

    ensure_dirs()
    stats = {}
    for lang in args.langs:
        corpus = CORPUS / f"{lang}.txt"
        for kind in ("words", "mwu"):
            merger = MWUMerger(load_mwus(MWU / f"{lang}.txt")) if kind == "mwu" else None
            out = INDEX / f"{lang}_{kind}.npz"
            with Timer(f"index {lang}/{kind}"):
                index = CooccurrenceIndex.from_corpus_file(
                    corpus, min_df=args.min_df, merger=merger, meta={"lang": lang, "kind": kind}
                )
                index.save(out)
            stats[f"{lang}_{kind}"] = {
                "documents": index.n_docs,
                "vocabulary": len(index),
                "mwu_units": len(merger) if merger else 0,
                "file_mb": round(out.stat().st_size / 1e6, 1),
            }
            print(f"{index} -> {out}")
    write_json(INDEX / "stats.json", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
