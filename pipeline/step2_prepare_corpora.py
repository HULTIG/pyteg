"""Step 2. Turn the raw corpora into the counts corpora.

Usage:  python pipeline/step2_prepare_corpora.py [--window 100] [--source nltk|wikipedia]

Output: data/corpus/en.txt and data/corpus/pt.txt, one pseudo-document per
line. A pseudo-document is a window of ``--window`` consecutive tokens taken
from a single source document, so no window mixes two texts. Windows replace
the Web pages of the original experiments: the counts are "in how many
windows does this word (or pair of words) appear".

Sources
-------
nltk (default, used for the reference results)
    English: Brown, Reuters-21578 (ApteMod) and the NLTK Gutenberg selection.
    Portuguese: Mac-Morpho and Floresta Sintá(c)tica.
wikipedia (optional, larger)
    The ``wikimedia/wikipedia`` dump on the Hugging Face Hub (``20231101``
    snapshot). Needs ``pip install datasets`` and network access to the Hub.
    Use ``--max-articles`` to bound the size.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS, MIN_WINDOW, RAW, WINDOW, Timer, ensure_dirs, write_json  # noqa: E402

from pyteg.text import tokenize  # noqa: E402


# ---------------------------------------------------------------------------
# Readers: each yields (document_id, text) for one source document.
# ---------------------------------------------------------------------------


def read_brown(root: Path) -> Iterator[Tuple[str, str]]:
    for path in sorted(root.glob("c[a-r][0-9][0-9]")):
        words = []
        for token in path.read_text(encoding="utf-8").split():
            word = token.rsplit("/", 1)[0]
            words.append(word)
        yield f"brown/{path.name}", " ".join(words)


def read_reuters(root: Path) -> Iterator[Tuple[str, str]]:
    for split in ("training", "test"):
        for path in sorted((root / split).iterdir(), key=lambda p: int(p.name)):
            yield f"reuters/{split}/{path.name}", path.read_text(encoding="latin-1")


def read_gutenberg(root: Path) -> Iterator[Tuple[str, str]]:
    for path in sorted(root.glob("*.txt")):
        yield f"gutenberg/{path.name}", path.read_text(encoding="latin-1")


def read_mac_morpho(root: Path) -> Iterator[Tuple[str, str]]:
    for path in sorted(root.glob("*.txt")):
        words = []
        for line in path.read_text(encoding="latin-1").splitlines():
            line = line.strip()
            if not line:
                continue
            word = line.rsplit("_", 1)[0]
            words.append(word)
        yield f"mac_morpho/{path.name}", " ".join(words)


_FLORESTA_SENT = re.compile(r"^#\d+\s+(\S+)-\d+\s+(.*)$")


def read_floresta(root: Path) -> Iterator[Tuple[str, str]]:
    """Use the raw sentence lines (``#12 CP3-4 text``), grouped by extract."""
    path = root / "Floresta_7.4.ptb"
    current, sentences = None, []
    for line in path.read_text(encoding="iso-8859-15").splitlines():
        match = _FLORESTA_SENT.match(line)
        if not match:
            continue
        extract, text = match.groups()
        if extract != current and sentences:
            yield f"floresta/{current}", " ".join(sentences)
            sentences = []
        current = extract
        sentences.append(text.replace("_", " "))
    if sentences:
        yield f"floresta/{current}", " ".join(sentences)


NLTK_SOURCES = {
    "en": [("brown", read_brown), ("reuters", read_reuters), ("gutenberg", read_gutenberg)],
    "pt": [("mac_morpho", read_mac_morpho), ("floresta", read_floresta)],
}


def read_wikipedia(lang: str, max_articles: int) -> Iterator[Tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as err:  # pragma: no cover - optional dependency
        raise SystemExit("the wikipedia source needs `pip install datasets`") from err
    stream = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    for i, row in enumerate(stream):
        if i >= max_articles:
            break
        yield f"wikipedia/{row['id']}", row["text"]


# ---------------------------------------------------------------------------


def windows(tokens: List[str], size: int, min_size: int) -> Iterator[List[str]]:
    for start in range(0, len(tokens), size):
        chunk = tokens[start : start + size]
        if len(chunk) >= min_size:
            yield chunk


def write_corpus(docs: Iterable[Tuple[str, str]], out: Path, size: int, min_size: int) -> dict:
    n_docs = n_windows = n_tokens = 0
    with open(out, "w", encoding="utf-8") as handle:
        for _, text in docs:
            n_docs += 1
            for chunk in windows(tokenize(text), size, min_size):
                handle.write(" ".join(chunk) + "\n")
                n_windows += 1
                n_tokens += len(chunk)
    return {"source_documents": n_docs, "windows": n_windows, "tokens": n_tokens}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--min-window", type=int, default=MIN_WINDOW)
    parser.add_argument("--source", choices=("nltk", "wikipedia"), default="nltk")
    parser.add_argument("--max-articles", type=int, default=200_000, help="wikipedia source only")
    parser.add_argument("--langs", nargs="+", default=["en", "pt"])
    args = parser.parse_args()

    ensure_dirs()
    stats = {"source": args.source, "window": args.window, "min_window": args.min_window, "languages": {}}
    for lang in args.langs:
        out = CORPUS / f"{lang}.txt"
        with Timer(f"corpus {lang}"):
            if args.source == "nltk":
                parts = {}

                def chained():
                    for name, reader in NLTK_SOURCES[lang]:
                        root = RAW / name
                        if not root.exists():
                            raise SystemExit(f"{root} not found; run step1_download.py first")
                        count = 0
                        for item in reader(root):
                            count += 1
                            yield item
                        parts[name] = count

                info = write_corpus(chained(), out, args.window, args.min_window)
                info["parts"] = parts
            else:
                info = write_corpus(read_wikipedia(lang, args.max_articles), out, args.window, args.min_window)
                info["max_articles"] = args.max_articles
        stats["languages"][lang] = info
        print(f"{lang}: {info['windows']:,} windows, {info['tokens']:,} tokens -> {out}")
    write_json(CORPUS / "stats.json", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
