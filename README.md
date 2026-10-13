# pyteg

**Textual entailment by generality with asymmetric InfoSimba.**

pyteg is a Python implementation of the method proposed in Sebastião Pais's
PhD thesis (Mines ParisTech and Universidade da Beira Interior, 2013). It
decides whether a text entails a hypothesis *because the hypothesis is more
general*, the way "I ate a strawberry" entails "I ate fruit" but not the other
way round. The method needs no training data, no linguistic resources and no
threshold: only word co-occurrence counts.

The original experiments relied on Web search hit counts, which can no longer
be obtained. This package replaces them with counts over a fixed corpus and
comes with a pipeline that rebuilds every result from public data in about
six minutes.

## The idea in one example

Take two words. If *fruit* shows up in most of the documents that mention
*strawberry*, but *strawberry* shows up in only a few of the documents that
mention *fruit*, then *fruit* is the more general of the two. An asymmetric
association measure captures this: AS(fruit‖strawberry) is high and
AS(strawberry‖fruit) is low.

For sentences, the simplified Asymmetric InfoSimba (AISs) adds up this
asymmetry over every pair of words, one from each sentence. T entails H by
generality when

```
AISs(T || H) < AISs(H || T)
```

that is, when the words of H are, on the whole, the more general ones. Eight
association measures from the thesis can play the role of AS, and the
sentences can be used as they are, without stop words, or with multiword
units merged into single tokens. [docs/method.md](docs/method.md) gives the
formulas and every implementation choice.

## Installation

pyteg needs Python 3.9 or newer, NumPy and SciPy.

```bash
git clone https://github.com/HULTIG/pyteg.git
cd pyteg
pip install -e .
```

## Quick start

You need a co-occurrence index. The reproduction pipeline builds one for
English and one for Portuguese (steps 1, 2 and 4 in
[REPRODUCING.md](REPRODUCING.md)), or you can build your own from any text
file with one document per line:

```bash
pyteg index --corpus my_corpus.txt --out my_index.npz
```

Then, in Python:

```python
from pyteg import CooccurrenceIndex, GeneralityClassifier

index = CooccurrenceIndex.load("data/index/en_words.npz")
clf = GeneralityClassifier(index, measure="BB", level="all_words", lang="en")

score = clf.score(
    "Mexico City has a very bad pollution problem because the mountains "
    "around the city act as walls and block in dust and smog.",
    "Poor air circulation out of the mountain-walled Mexico City aggravates pollution.",
)
print(score.forward, score.backward, score.entails)
# 0.2097... 0.2135... True
```

The same from the command line, with every measure at once:

```bash
pyteg score --index data/index/en_words.npz --all-measures \
  --t "Mexico City has a very bad pollution problem because the mountains around the city act as walls and block in dust and smog." \
  --h "Poor air circulation out of the mountain-walled Mexico City aggravates pollution."
```

Other entry points you may find useful:

* `word_generality(index, x, y, measure)` compares two words directly.
* `attributional_generality(index, x, y, measure)` compares the context
  profiles of two words.
* `bidirectional_scores(index, t_tokens, h_tokens, measures)` returns both
  directions for many measures at once, sharing the counts.
* `extract_mwus` and `MWUMerger` find and merge multiword units.
* `load_rte_xml` reads the RTE and ASSIN data formats.

## Reproducing the experiments

```bash
pip install -e ".[test]"
python pipeline/run_all.py
```

Seven scripts, run in order, download pinned data (checked with SHA-256),
build English and Portuguese corpora, extract multiword units, build the
indexes, prepare the benchmarks, run three experiments and write
[results/REPORT.md](results/REPORT.md). [REPRODUCING.md](REPRODUCING.md) walks
through each step, what it produces and how to compare your run with ours.

The three experiments are:

1. **Entailment on RTE-1 to RTE-3**, as in the thesis.
2. **Direction on the positive RTE pairs**: given a true entailment, does the
   method put the generality on the hypothesis side?
3. **Direction on WordNet hypernym pairs**, in English and Portuguese: which
   of two words is the more general?

## What the reference run shows

In short: on RTE as a whole the method stays close to chance (accuracy 0.48
to 0.53), which matches the 0.51 to 0.55 reported in the thesis with Web
counts. On the positive pairs, Braun-Blanquet and the conditional probability
put the generality on the hypothesis side 60% of the time, well above chance
but far below a baseline that just picks the shorter sentence (98%). At word
level, the conditional probability turns out to be a frequency comparison in
disguise, and the context profiles add little to it. The discussion, with all
the caveats, is in [docs/findings.md](docs/findings.md).

## How this differs from the thesis

* Web hit counts are replaced by document counts over a fixed corpus, cut into
  windows of 100 tokens. Exact numbers therefore cannot match.
* RTE-4 and RTE-5 are not freely available, so the RTE experiments use RTE-1
  to RTE-3.
* The crowd-annotated corpus of entailment by generality built for the thesis
  is not available. Experiments 2 and 3 are proxies built from public data,
  not a replacement for it.
* Multiword units come from a built-in LocalMaxs extractor rather than SENTA,
  although SENTA output can be plugged in.
* Where the published formulas are ambiguous or inconsistent (Braun-Blanquet,
  Conviction, the normalisation of AISs), the choice is explained in
  [docs/method.md](docs/method.md#8-interpretation-choices).

## Repository layout

```
src/pyteg/         the package
  measures.py      asymmetric association measures
  index.py         co-occurrence counts
  infosimba.py     InfoSimba, AIS and AISs
  teg.py           decision rule for sentences and words
  text.py          tokenisation, stop words, pre-processing levels
  mwu.py           multiword units
  data.py          RTE / ASSIN / JSONL / TSV loaders
  evaluate.py      metrics
  cli.py           the pyteg command
pipeline/          the seven reproduction steps
tests/             unit tests (python -m pytest)
docs/              method notes and discussion of the results
results/           reference results and REPORT.md
```

## Citing

If you use this package, please cite the thesis:

> Sebastião Pais. 2013. *Asymmetric Distributional Similarity Measures to
> Recognize Textual Entailment by Generality*. PhD thesis, École nationale
> supérieure des mines de Paris and Universidade da Beira Interior.
> [pastel-00962176](https://pastel.hal.science/pastel-00962176).

Related publications:

* Sebastião Pais and Gaël Dias. 2020. Asymmetric Attributional Word
  Similarity Measures to Detect the Relations of Textual Generality.
  *Computers* 9(4):81. https://doi.org/10.3390/computers9040081
* Sebastião Pais, Gaël Dias, João Cordeiro and Rumen Moraliyski. 2014.
  Unsupervised and Language Independent Method to Recognize Textual
  Entailment by Generality. In *Proceedings of CLIB 2014*, pages 82–90.
* Sebastião Pais, Gaël Dias and Rumen Moraliyski. 2014. Recognize the
  Generality Relation between Sentences Using Asymmetric Association
  Measures. In *Proceedings of CLIB 2014*.
* Gaël Dias, Sebastião Pais, Katarzyna Wegrzyn-Wolska and Robert Mahl. 2011.
  Recognizing Textual Entailment by Generality Using Informative Asymmetric
  Measures and Multiword Unit Identification to Summarize Ephemeral Clusters.
  In *Proceedings of WI-IAT 2011*.

[CITATION.cff](CITATION.cff) has the same information in a machine-readable
form.

## Licence

The code is released under the MIT licence. The Snowball stop-word lists
bundled in `src/pyteg/resources/` are BSD-licensed.

The pipeline downloads data that this repository does not redistribute, and
each dataset keeps its own terms: the PASCAL RTE data, the Brown, Reuters-21578
and Gutenberg corpora as packaged by NLTK, Mac-Morpho and Floresta
Sintá(c)tica, Princeton WordNet 3.0 (WordNet licence) and OpenWN-PT (CC BY-SA).
Please check them before reusing the data for anything else.
