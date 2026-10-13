"""Step 5. Prepare the evaluation data.

Usage:  python pipeline/step5_prepare_benchmarks.py [--pairs 2000]

Outputs
-------
data/bench/rte.jsonl
    Every pair of RTE-1, RTE-2 and RTE-3 (development and test sets), with
    the gold two-way label. RTE-4 and RTE-5 were distributed by NIST under a
    user agreement and are not freely downloadable, so they are not included.

data/bench/hyper_en.jsonl, data/bench/hyper_pt.jsonl
    Word pairs (hyponym, hypernym) taken from direct noun hypernymy links in
    WordNet 3.0. Portuguese lemmas come from OpenWN-PT through the Open
    Multilingual Wordnet, attached to the same synsets. A pair is kept when
    both words are single alphabetic tokens of three or more letters, neither
    is a stop word, each appears in at least --min-df windows of the counts
    corpus and the two co-occur at least once. Pairs that also appear in the
    opposite direction are dropped. A fixed-seed sample of --pairs pairs is
    written per language.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCH, INDEX, RAW, SEED, Timer, ensure_dirs, write_json  # noqa: E402

from pyteg import CooccurrenceIndex, load_rte_xml, load_stopwords, save_jsonl, tokenize  # noqa: E402

RTE_FILES = ["rte1_dev", "rte1_test", "rte2_dev", "rte2_test", "rte3_dev", "rte3_test"]


# ---------------------------------------------------------------------------
# WordNet
# ---------------------------------------------------------------------------


def parse_wordnet_nouns(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Read data.noun and return (lemmas by synset, hypernyms by synset).

    Synset keys look like ``02084071-n``, the format used by the OMW files.
    Only plain hypernym pointers (``@``) are followed, not instance ones.
    """
    lemmas: Dict[str, List[str]] = {}
    hypernyms: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("  "):  # licence header
                continue
            fields = line.split(" | ", 1)[0].split()
            offset = fields[0]
            n_words = int(fields[3], 16)
            words = [fields[4 + 2 * i].lower() for i in range(n_words)]
            pos = 4 + 2 * n_words
            n_ptrs = int(fields[pos])
            hyper = []
            for i in range(n_ptrs):
                symbol, target, target_pos = fields[pos + 1 + 4 * i : pos + 4 + 4 * i]
                if symbol == "@" and target_pos == "n":
                    hyper.append(f"{target}-n")
            key = f"{offset}-n"
            lemmas[key] = words
            hypernyms[key] = hyper
    return lemmas, hypernyms


def parse_omw(path: Path) -> Dict[str, List[str]]:
    """Read an OMW ``wn-data-xxx.tab`` file: synset -> lemmas (nouns only)."""
    lemmas: Dict[str, List[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].endswith("-n") and parts[1].endswith("lemma"):
                lemma = parts[2].strip().lower()
                if lemma not in lemmas[parts[0]]:
                    lemmas[parts[0]].append(lemma)
    return dict(lemmas)


def candidate_pairs(lemmas: Dict[str, List[str]], hypernyms: Dict[str, List[str]]) -> Iterator[Tuple[str, str, str, str]]:
    for synset, parents in hypernyms.items():
        for parent in parents:
            for hypo in lemmas.get(synset, []):
                for hyper in lemmas.get(parent, []):
                    yield hypo, hyper, synset, parent


def build_hypernym_pairs(
    lemmas: Dict[str, List[str]],
    hypernyms: Dict[str, List[str]],
    index: CooccurrenceIndex,
    stopwords: Set[str],
    min_df: int,
    n_pairs: int,
    seed: int,
) -> Tuple[List[dict], dict]:
    def usable(word: str) -> bool:
        return (
            word.isalpha()
            and len(word) >= 3
            and word not in stopwords
            and tokenize(word) == [word]
            and index.freq(word) >= min_df
        )

    seen: Dict[Tuple[str, str], Tuple[str, str]] = {}
    counts = {"wordnet_links": 0, "usable_words": 0}
    for hypo, hyper, s1, s2 in candidate_pairs(lemmas, hypernyms):
        counts["wordnet_links"] += 1
        if hypo == hyper or not (usable(hypo) and usable(hyper)):
            continue
        counts["usable_words"] += 1
        seen.setdefault((hypo, hyper), (s1, s2))

    keys = sorted(seen)
    both_ways = {k for k in keys if (k[1], k[0]) in seen}
    keys = [k for k in keys if k not in both_ways]
    joint = {}
    for hypo, hyper in keys:
        joint[(hypo, hyper)] = index.joint(hypo, hyper)
    keys = [k for k in keys if joint[k] > 0]
    counts.update({"distinct_pairs": len(seen), "dropped_both_directions": len(both_ways), "cooccurring": len(keys)})

    rng = random.Random(seed)
    rng.shuffle(keys)
    keys = sorted(keys[:n_pairs])
    items = [
        {
            "hypo": hypo,
            "hyper": hyper,
            "synset": seen[(hypo, hyper)][0],
            "hyper_synset": seen[(hypo, hyper)][1],
            "df_hypo": index.freq(hypo),
            "df_hyper": index.freq(hyper),
            "joint": joint[(hypo, hyper)],
        }
        for hypo, hyper in keys
    ]
    counts["sampled"] = len(items)
    counts["hyponym_more_frequent"] = sum(1 for it in items if it["df_hypo"] > it["df_hyper"])
    return items, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=2000, help="hypernym pairs per language")
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    ensure_dirs()
    stats = {"seed": args.seed, "min_df": args.min_df}

    with Timer("rte"):
        pairs = []
        per_file = {}
        for name in RTE_FILES:
            loaded = load_rte_xml(RAW / "rte" / f"{name}.xml", source=name)
            per_file[name] = {"pairs": len(loaded), "entailment": sum(1 for p in loaded if p.label)}
            pairs.extend(loaded)
        save_jsonl(pairs, BENCH / "rte.jsonl")
        stats["rte"] = per_file
    print(f"rte: {len(pairs):,} pairs -> {BENCH / 'rte.jsonl'}")

    with Timer("wordnet"):
        lemmas_en, hypernyms = parse_wordnet_nouns(RAW / "wordnet" / "data.noun")
        lemmas_pt = parse_omw(RAW / "omw-1.4" / "por" / "wn-data-por.tab")
        for lang, lemmas in (("en", lemmas_en), ("pt", lemmas_pt)):
            index = CooccurrenceIndex.load(INDEX / f"{lang}_words.npz")
            items, counts = build_hypernym_pairs(
                lemmas, hypernyms, index, set(load_stopwords(lang)), args.min_df, args.pairs, args.seed
            )
            out = BENCH / f"hyper_{lang}.jsonl"
            import json

            with open(out, "w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats[f"hyper_{lang}"] = counts
            print(f"hyper_{lang}: {len(items):,} pairs -> {out}  {counts}")
    write_json(BENCH / "stats.json", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
