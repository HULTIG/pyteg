"""Step 6. Run the experiments.

Usage:  python pipeline/step6_run_experiments.py [--only e1 e2 e3] [--normalize pairs]

E1  Entailment on RTE-1, RTE-2 and RTE-3 test sets (the thesis' first
    experiment). A pair is labelled "entailment" when T entails H by
    generality. Accuracy and class precisions, per challenge and averaged,
    for every measure and pre-processing level.

E2  Direction on the positive RTE test pairs. For pairs where T is known to
    entail H, how often does the method put the generality on the H side?
    Compared with two simple baselines: "the shorter sentence is the more
    general one" and "the sentence with more frequent words is the more
    general one".

E3  Direction on WordNet hypernym pairs, in English and Portuguese. Which of
    the two words is the more general? Each measure is used word-to-word and
    through the attributional AISs, which compares context vectors (two
    profile weightings, "count" and "ppmi"), and compared with the plain
    frequency baseline. Results are also given for the "hard" subset,
    where the hyponym is the more frequent word.

Outputs go to results/ (aggregated CSV files and run.json) and
data/results/ (per-item scores, useful for error analysis).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCH, DATA, MWU, RESULTS, Timer, ensure_dirs, index_path, write_json  # noqa: E402

import pyteg  # noqa: E402
from pyteg import (  # noqa: E402
    MEASURES,
    CooccurrenceIndex,
    MWUMerger,
    Preprocessor,
    attributional_generality,
    bidirectional_scores,
    binary_report,
    direction_accuracy,
    load_jsonl,
    load_mwus,
    load_stopwords,
    word_generality,
)
from pyteg.text import LEVELS  # noqa: E402

TEST_SETS = ["rte1_test", "rte2_test", "rte3_test"]
MEASURE_NAMES = list(MEASURES)


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def write_gz_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preprocessors(lang: str) -> Dict[str, Preprocessor]:
    merger = MWUMerger(load_mwus(MWU / f"{lang}.txt"))
    return {level: Preprocessor(level, lang=lang, merger=merger if level == "mwu" else None) for level in LEVELS}


def load_indexes(lang: str) -> Dict[str, CooccurrenceIndex]:
    words = CooccurrenceIndex.load(index_path(lang, "all_words"))
    mwu = CooccurrenceIndex.load(index_path(lang, "mwu"))
    return {"all_words": words, "no_stopwords": words, "mwu": mwu}


def mean_log_df(index: CooccurrenceIndex, tokens: List[str]) -> float:
    dfs = [index.freq(t) for t in dict.fromkeys(tokens)]
    dfs = [d for d in dfs if d > 0]
    return float(np.mean(np.log(dfs))) if dfs else 0.0


# ---------------------------------------------------------------------------
# E1 and E2 share the same scores, so they are computed together.
# ---------------------------------------------------------------------------


def score_rte(normalize: str) -> List[dict]:
    pairs = [p for p in load_jsonl(BENCH / "rte.jsonl") if p.source in TEST_SETS]
    indexes = load_indexes("en")
    preps = preprocessors("en")
    rows = []
    for level in LEVELS:
        index, prep = indexes[level], preps[level]
        with Timer(f"rte scores ({level})"):
            for p in pairs:
                t, h = prep(p.t), prep(p.h)
                scores = bidirectional_scores(index, t, h, measures=MEASURE_NAMES, normalize=normalize)
                row = {
                    "id": p.id,
                    "challenge": p.source.split("_")[0],
                    "gold": int(bool(p.label)),
                    "level": level,
                    "len_t": len(t),
                    "len_h": len(h),
                    "logdf_t": mean_log_df(index, t),
                    "logdf_h": mean_log_df(index, h),
                }
                for name, s in scores.items():
                    row[f"{name}_fwd"] = s.forward
                    row[f"{name}_bwd"] = s.backward
                    row["pairs"] = s.pairs
                rows.append(row)
    return rows


def e1(rows: List[dict]) -> List[dict]:
    out = []
    challenges = sorted({r["challenge"] for r in rows})
    for level in LEVELS:
        for method in ["always_yes"] + MEASURE_NAMES:
            per = []
            for ch in challenges + ["pooled"]:
                sel = [r for r in rows if r["level"] == level and (ch == "pooled" or r["challenge"] == ch)]
                gold = [bool(r["gold"]) for r in sel]
                if method == "always_yes":
                    pred = [True] * len(sel)
                else:
                    pred = [r["pairs"] > 0 and r[f"{method}_fwd"] < r[f"{method}_bwd"] for r in sel]
                rep = binary_report(gold, pred).as_dict()
                row = {"level": level, "method": method, "challenge": ch, "n": len(sel)}
                row.update({k: rep[k] for k in ("accuracy", "precision_pos", "precision_neg", "recall_pos", "precision_macro", "precision_weighted")})
                row["predicted_yes"] = sum(pred) / len(pred) if pred else 0.0
                out.append(row)
                if ch != "pooled":
                    per.append(row)
            avg = {"level": level, "method": method, "challenge": "average", "n": sum(r["n"] for r in per)}
            for k in ("accuracy", "precision_pos", "precision_neg", "recall_pos", "precision_macro", "precision_weighted", "predicted_yes"):
                avg[k] = float(np.mean([r[k] for r in per]))
            out.append(avg)
    return out


def e2(rows: List[dict]) -> List[dict]:
    out = []
    positives = [r for r in rows if r["gold"] == 1]
    for level in LEVELS:
        sel = [r for r in positives if r["level"] == level]
        methods = {
            "baseline_shorter_h": [r["len_t"] - r["len_h"] for r in sel],
            "baseline_frequency": [r["logdf_h"] - r["logdf_t"] for r in sel],
        }
        for name in MEASURE_NAMES:
            methods[name] = [(r[f"{name}_bwd"] - r[f"{name}_fwd"]) if r["pairs"] > 0 else 0.0 for r in sel]
        for method, margins in methods.items():
            row = {"level": level, "method": method}
            row.update(direction_accuracy(margins))
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------


PROFILES = ("count", "ppmi")


def e3(context_size: int, normalize: str) -> Tuple[List[dict], List[dict]]:
    summary, items_out = [], []
    for lang in ("en", "pt"):
        index = CooccurrenceIndex.load(index_path(lang, "all_words"))
        stop = load_stopwords(lang)
        with open(BENCH / f"hyper_{lang}.jsonl", encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle]
        caches: Dict[str, Dict[str, Dict[str, float]]] = {w: {} for w in PROFILES}
        margins: Dict[str, List[float]] = {"baseline_frequency": []}
        with Timer(f"hypernyms {lang}"):
            for it in items:
                hypo, hyper = it["hypo"], it["hyper"]
                rec = {"lang": lang, "hypo": hypo, "hyper": hyper, "df_hypo": it["df_hypo"], "df_hyper": it["df_hyper"]}
                margins["baseline_frequency"].append(float(it["df_hyper"] - it["df_hypo"]))
                for name in MEASURE_NAMES:
                    as_hyper, as_hypo = word_generality(index, hyper, hypo, name)
                    m = as_hyper - as_hypo
                    margins.setdefault(f"word:{name}", []).append(m)
                    rec[f"word:{name}"] = m
                    for weighting in PROFILES:
                        s = attributional_generality(
                            index,
                            hypo,
                            hyper,
                            name,
                            size=context_size,
                            normalize=normalize,
                            weighting=weighting,
                            exclude=stop,
                            cache=caches[weighting],
                        )
                        m = s.margin if s.pairs > 0 else 0.0
                        key = f"attr_{weighting}:{name}"
                        margins.setdefault(key, []).append(m)
                        rec[key] = m
                items_out.append(rec)
        hard = [i for i, it in enumerate(items) if it["df_hypo"] > it["df_hyper"]]
        for method, values in margins.items():
            for subset, idx in (("all", range(len(items))), ("hyponym_more_frequent", hard)):
                row = {"lang": lang, "method": method, "subset": subset}
                row.update(direction_accuracy([values[i] for i in idx]))
                summary.append(row)
    return summary, items_out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=("e1", "e2", "e3"), default=["e1", "e2", "e3"])
    parser.add_argument("--normalize", choices=("pairs", "none", "first"), default="pairs")
    parser.add_argument("--context-size", type=int, default=50, help="context words per profile in E3")
    args = parser.parse_args()

    ensure_dirs()
    started = time.time()
    if "e1" in args.only or "e2" in args.only:
        rows = score_rte(args.normalize)
        write_gz_csv(DATA / "results" / "rte_pair_scores.csv.gz", rows)
        if "e1" in args.only:
            write_csv(RESULTS / "e1_rte_entailment.csv", e1(rows))
        if "e2" in args.only:
            write_csv(RESULTS / "e2_rte_direction.csv", e2(rows))
    if "e3" in args.only:
        summary, items = e3(args.context_size, args.normalize)
        write_csv(RESULTS / "e3_hypernym_direction.csv", summary)
        write_gz_csv(DATA / "results" / "hypernym_item_scores.csv.gz", items)

    write_json(
        RESULTS / "run.json",
        {
            "experiments": args.only,
            "normalize": args.normalize,
            "context_size": args.context_size,
            "pyteg_version": pyteg.__version__,
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "finished": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "seconds": round(time.time() - started, 1),
        },
    )
    print(f"results written to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
