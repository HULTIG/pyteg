"""Step 7. Write results/REPORT.md from the CSV files of step 6.

Usage:  python pipeline/step7_write_report.py

The report only states numbers. What they mean is discussed in
docs/findings.md, which refers to the reference run.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCH, CORPUS, INDEX, MWU, RESULTS  # noqa: E402

from pyteg.measures import MEASURES  # noqa: E402

LEVELS = ["all_words", "no_stopwords", "mwu"]
LEVEL_NAMES = {"all_words": "All words", "no_stopwords": "Without stop words", "mwu": "With MWUs"}


def load_csv(name: str) -> List[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


DISPLAY = {
    "added_value": "Added Value",
    "braun_blanquet": "Braun-Blanquet",
    "certainty_factor": "Certainty Factor",
    "conditional_probability": "Conditional probability",
    "conviction": "Conviction",
    "gini_index": "Gini index",
    "j_measure": "J-measure",
    "laplace": "Laplace",
    "conviction_yx": "Conviction, y to x",
}


def ci95(p: float, n: int) -> float:
    """Half-width of a normal-approximation 95% interval for a proportion."""
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def label(method: str) -> str:
    if method in MEASURES:
        m = MEASURES[method]
        return f"{DISPLAY.get(method, method)} ({m.short})" + ("" if m.in_thesis else " *")
    return {
        "always_yes": "Baseline: always entailment",
        "baseline_shorter_h": "Baseline: shorter sentence is more general",
        "baseline_frequency": "Baseline: more frequent words are more general",
    }.get(method, method)


def f(x: str) -> str:
    return f"{float(x):.3f}"


def table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def section_setup() -> str:
    run = load_json(RESULTS / "run.json")
    corpus = load_json(CORPUS / "stats.json")
    index = load_json(INDEX / "stats.json")
    mwu = load_json(MWU / "stats.json")
    bench = load_json(BENCH / "stats.json")
    out = ["## Setup\n"]
    out.append(
        f"pyteg {run.get('pyteg_version', '?')} (commit `{run.get('git_commit', '?')[:10]}`), "
        f"Python {run.get('python', '?')}, NumPy {run.get('numpy', '?')}, "
        f"AISs normalisation `{run.get('normalize', '?')}`, run finished {run.get('finished', '?')} "
        f"in {run.get('seconds', '?')} s.\n"
    )
    if corpus:
        rows = []
        for lang, info in corpus.get("languages", {}).items():
            parts = ", ".join(f"{k} ({v:,} docs)" for k, v in info.get("parts", {}).items())
            rows.append([lang, parts or corpus.get("source", ""), f"{info['tokens']:,}", f"{info['windows']:,}"])
        out.append(f"Counts corpora (source `{corpus.get('source')}`, windows of {corpus.get('window')} tokens):\n")
        out.append(table(["Language", "Parts", "Tokens", "Windows (documents)"], rows) + "\n")
    if index:
        rows = [[k, f"{v['vocabulary']:,}", f"{v['documents']:,}", f"{v['mwu_units']:,}"] for k, v in index.items()]
        out.append(table(["Index", "Vocabulary", "Documents", "MWUs merged"], rows) + "\n")
    if mwu:
        out.append(
            "Multiword units: "
            + "; ".join(f"{lang}: {v['units']:,} units ({v['extractor']})" for lang, v in mwu.items())
            + ".\n"
        )
    if bench:
        rte = bench.get("rte", {})
        tests = {k: v for k, v in rte.items() if k.endswith("_test")}
        if tests:
            out.append(
                "RTE test sets: "
                + ", ".join(f"{k} {v['pairs']} pairs ({v['entailment']} positive)" for k, v in tests.items())
                + ".\n"
            )
        for lang in ("en", "pt"):
            h = bench.get(f"hyper_{lang}")
            if h:
                out.append(
                    f"Hypernym pairs ({lang}): {h['sampled']:,} sampled from {h['cooccurring']:,} usable pairs; "
                    f"the hyponym is the more frequent word in {h['hyponym_more_frequent']:,} of them.\n"
                )
    return "\n".join(out)


def section_e1() -> str:
    rows = load_csv("e1_rte_entailment.csv")
    if not rows:
        return ""
    ref = load_json(Path(__file__).resolve().parent / "thesis_reference.json").get("e1_accuracy_average", {})
    methods = ["always_yes"] + list(MEASURES)
    get: Dict[tuple, dict] = {(r["level"], r["method"], r["challenge"]): r for r in rows}
    body = []
    for m in methods:
        line = [label(m)]
        for level in LEVELS:
            r = get.get((level, m, "average"))
            thesis = ref.get(level, {}).get(m)
            line.append(f(r["accuracy"]) if r else "")
            line.append(f"{thesis:.2f}" if thesis is not None else "")
        body.append(line)
    header = ["Method"]
    for level in LEVELS:
        header += [f"{LEVEL_NAMES[level]}", "thesis"]
    out = [
        "## E1. Entailment on RTE-1 to RTE-3 (test sets)\n",
        "A pair is labelled *entailment* when the method finds that T entails H by generality. "
        "Accuracy averaged over the three challenges. The *thesis* columns are the averages over RTE-1 to RTE-5 "
        "with Web counts reported in Pais, Dias & Moraliyski (2014); they are not computed on the same test sets.\n",
        table(header, body),
        "\n\\* not in the thesis: Conviction with the same orientation as the other measures.\n",
    ]
    body = []
    for m in methods:
        r = get.get(("all_words", m, "average"))
        if r:
            body.append([label(m), f(r["accuracy"]), f(r["precision_pos"]), f(r["precision_neg"]), f(r["recall_pos"]), f(r["predicted_yes"])])
    out.append("Details for the *all words* level (averages over the three challenges):\n")
    out.append(table(["Method", "Accuracy", "Precision (entailment)", "Precision (no entailment)", "Recall (entailment)", "Share predicted entailment"], body))
    body = []
    for m in methods:
        line = [label(m)]
        for ch in ("rte1", "rte2", "rte3", "pooled"):
            r = get.get(("all_words", m, ch))
            line.append(f(r["accuracy"]) if r else "")
        body.append(line)
    out.append("\nAccuracy per challenge (all words):\n")
    out.append(table(["Method", "RTE-1", "RTE-2", "RTE-3", "Pooled"], body))
    pooled = get.get(("all_words", "braun_blanquet", "pooled"))
    if pooled:
        n = int(pooled["n"])
        out.append(
            f"\nWith {n:,} pooled pairs, the 95% confidence interval around an accuracy of 0.5 is about "
            f"±{ci95(0.5, n):.3f}; per challenge (800 pairs) it is about ±{ci95(0.5, 800):.3f}."
        )
    return "\n".join(out) + "\n"


def section_e2() -> str:
    rows = load_csv("e2_rte_direction.csv")
    if not rows:
        return ""
    get = {(r["level"], r["method"]): r for r in rows}
    methods = ["baseline_shorter_h", "baseline_frequency"] + list(MEASURES)
    body = []
    for m in methods:
        line = [label(m)]
        for level in LEVELS:
            r = get.get((level, m))
            line.append(f(r["accuracy"]) if r else "")
        body.append(line)
    n = rows[0]["n"]
    return "\n".join(
        [
            "## E2. Direction on the positive RTE test pairs\n",
            f"Share of the {n} positive test pairs for which the method puts the generality on the H side "
            "(AISs(T||H) < AISs(H||T)). Ties count as errors.\n",
            table(["Method"] + [LEVEL_NAMES[l] for l in LEVELS], body),
            f"\nWith {int(n):,} pairs, the 95% confidence interval around 0.6 is about ±{ci95(0.6, int(n)):.3f}.",
        ]
    ) + "\n"


def section_e3() -> str:
    rows = load_csv("e3_hypernym_direction.csv")
    if not rows:
        return ""
    get = {(r["lang"], r["method"], r["subset"]): r for r in rows}
    out = [
        "## E3. Direction on WordNet hypernym pairs\n",
        "Share of pairs where the hypernym is recognised as the more general word. *Word* compares "
        "AS(hypernym||hyponym) with AS(hyponym||hypernym). *Profiles* compares the context vectors of the two "
        "words with AISs, with count or PPMI weights. *Hard* is the subset where the hyponym is the more "
        "frequent word.\n",
    ]
    for lang in ("en", "pt"):
        n_all = get.get((lang, "baseline_frequency", "all"), {}).get("n", "?")
        n_hard = get.get((lang, "baseline_frequency", "hyponym_more_frequent"), {}).get("n", "?")
        body = []
        r_all = get.get((lang, "baseline_frequency", "all"))
        r_hard = get.get((lang, "baseline_frequency", "hyponym_more_frequent"))
        if r_all:
            body.append([label("baseline_frequency"), f(r_all["accuracy"]), "", "", f(r_hard["accuracy"]), "", ""])
        for m in MEASURES:
            line = [label(m)]
            for subset in ("all", "hyponym_more_frequent"):
                for kind in ("word", "attr_count", "attr_ppmi"):
                    r = get.get((lang, f"{kind}:{m}", subset))
                    line.append(f(r["accuracy"]) if r else "")
            body.append(line)
        out.append(f"### {'English' if lang == 'en' else 'Portuguese'} ({n_all} pairs, {n_hard} hard)\n")
        if n_all != "?":
            out.append(
                f"95% confidence interval around 0.6: about ±{ci95(0.6, int(n_all)):.3f} on all pairs and "
                f"±{ci95(0.6, int(n_hard)):.3f} on the hard subset.\n"
            )
        out.append(
            table(
                ["Method", "Word", "Profiles (count)", "Profiles (PPMI)", "Hard: word", "Hard: count", "Hard: PPMI"],
                body,
            )
            + "\n"
        )
    return "\n".join(out)


def main() -> int:
    parts = [
        "# Reproduction report\n",
        "Generated by `pipeline/step7_write_report.py` from the files in `results/`. "
        "See `docs/findings.md` for a discussion of these numbers.\n",
        section_setup(),
        section_e1(),
        section_e2(),
        section_e3(),
    ]
    report = "\n".join(p for p in parts if p)
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {RESULTS / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
