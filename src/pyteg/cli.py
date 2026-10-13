"""Command-line interface: ``pyteg mwu | index | info | score``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .index import CooccurrenceIndex
from .measures import MEASURES
from .mwu import MWUMerger, extract_mwus, load_mwus, save_mwus
from .teg import GeneralityClassifier
from .text import LEVELS, load_stopwords, tokenize


def _read_docs(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            yield tokenize(line)


def cmd_mwu(args: argparse.Namespace) -> int:
    stop = load_stopwords(args.lang) if args.lang else None
    units = extract_mwus(list(_read_docs(args.corpus)), max_n=args.max_n, min_freq=args.min_freq, stopwords=stop)
    save_mwus(units, args.out)
    print(f"wrote {len(units):,} multiword units to {args.out}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    merger = MWUMerger(load_mwus(args.mwu)) if args.mwu else None
    index = CooccurrenceIndex.from_corpus_file(args.corpus, min_df=args.min_df, merger=merger)
    index.save(args.out)
    print(f"{index} saved to {args.out}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    index = CooccurrenceIndex.load(args.index)
    print(index)
    print(json.dumps(index.meta, indent=2, ensure_ascii=False))
    print("most frequent words:", ", ".join(f"{w} ({c})" for w, c in index.top_words(args.top)))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    index = CooccurrenceIndex.load(args.index)
    merger = MWUMerger(load_mwus(args.mwu)) if args.mwu else None
    clf = GeneralityClassifier(index, measure=args.measure, level=args.level, lang=args.lang, merger=merger)
    measures = list(MEASURES) if args.all_measures else [clf.measure.name]
    scores = clf.score_all(args.t, args.h, measures=measures)
    print(f"T: {args.t}\nH: {args.h}\nlevel: {args.level}\n")
    print(f"{'measure':<26}{'AISs(T||H)':>12}{'AISs(H||T)':>12}  decision")
    for name, s in scores.items():
        verdict = "T entails H by generality" if s.entails else f"no ({s.direction})"
        print(f"{name:<26}{s.forward:>12.4f}{s.backward:>12.4f}  {verdict}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyteg", description="Textual entailment by generality with asymmetric InfoSimba.")
    parser.add_argument("--version", action="version", version=f"pyteg {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mwu", help="extract multiword units with LocalMaxs + SCP_f")
    p.add_argument("--corpus", required=True, help="text file, one document per line")
    p.add_argument("--out", required=True)
    p.add_argument("--max-n", type=int, default=3)
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--lang", default="en", help="stop-word list used to trim unit edges ('' to keep all)")
    p.set_defaults(func=cmd_mwu)

    p = sub.add_parser("index", help="build a co-occurrence index from a corpus file")
    p.add_argument("--corpus", required=True, help="text file, one document per line")
    p.add_argument("--out", required=True, help="output .npz file")
    p.add_argument("--mwu", help="merge these multiword units before counting")
    p.add_argument("--min-df", type=int, default=1)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("info", help="describe an index")
    p.add_argument("--index", required=True)
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("score", help="score one text-hypothesis pair")
    p.add_argument("--index", required=True)
    p.add_argument("--t", required=True, help="the text")
    p.add_argument("--h", required=True, help="the hypothesis")
    p.add_argument("--measure", default="braun_blanquet", help=f"one of: {', '.join(MEASURES)}")
    p.add_argument("--level", default="all_words", choices=LEVELS)
    p.add_argument("--lang", default="en")
    p.add_argument("--mwu", help="multiword units file (needed for --level mwu)")
    p.add_argument("--all-measures", action="store_true", help="report every measure")
    p.set_defaults(func=cmd_score)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score" and args.level == "mwu" and not args.mwu:
        parser.error("--level mwu needs --mwu with the multiword units used to build the index")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
