"""Step 3. Find the multiword units used by the third pre-processing level.

Usage:
    python pipeline/step3_extract_mwus.py                      # built-in extractor
    python pipeline/step3_extract_mwus.py --from-file en=units_en.txt pt=units_pt.txt

By default units are extracted from the counts corpus with LocalMaxs and the
SCP_f glue (pyteg.mwu.extract_mwus). The thesis used SENTA instead; to follow
it more closely, export the units found by SENTA/pysenta to a text file (one
unit per line, words separated by spaces) and pass it with --from-file.

Output: data/mwu/<lang>.txt and data/mwu/stats.json.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS, MWU, MWU_MAX_N, MWU_MIN_FREQ, Timer, ensure_dirs, write_json  # noqa: E402

from pyteg.mwu import extract_mwus, load_mwus, save_mwus  # noqa: E402
from pyteg.text import load_stopwords  # noqa: E402


def read_corpus(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            yield line.split()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", nargs="+", default=["en", "pt"])
    parser.add_argument("--max-n", type=int, default=MWU_MAX_N)
    parser.add_argument("--min-freq", type=int, default=MWU_MIN_FREQ)
    parser.add_argument(
        "--from-file",
        nargs="+",
        default=[],
        metavar="LANG=PATH",
        help="use an existing list of units instead of extracting them",
    )
    args = parser.parse_args()

    ensure_dirs()
    external = dict(item.split("=", 1) for item in args.from_file)
    stats = {}
    for lang in args.langs:
        out = MWU / f"{lang}.txt"
        if lang in external:
            shutil.copyfile(external[lang], out)
            units = load_mwus(out)
            stats[lang] = {"extractor": f"external file {Path(external[lang]).name}", "units": len(units)}
        else:
            with Timer(f"mwu {lang}"):
                units = extract_mwus(
                    read_corpus(CORPUS / f"{lang}.txt"),
                    max_n=args.max_n,
                    min_freq=args.min_freq,
                    stopwords=load_stopwords(lang),
                )
            save_mwus(units, out)
            by_len = {}
            for unit, _ in units:
                by_len[len(unit)] = by_len.get(len(unit), 0) + 1
            stats[lang] = {
                "extractor": "pyteg LocalMaxs + SCP_f",
                "max_n": args.max_n,
                "min_freq": args.min_freq,
                "units": len(units),
                "by_length": {str(k): v for k, v in sorted(by_len.items())},
                "examples": [" ".join(u) for u, _ in units[:15]],
            }
        print(f"{lang}: {stats[lang]['units']:,} units -> {out}")
    write_json(MWU / "stats.json", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
