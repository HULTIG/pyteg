# Reproducing the experiments

Everything in `results/` can be rebuilt from scratch on an ordinary laptop. The
whole run takes about six minutes, downloads about 55 MB and needs about 350 MB
of disk space. Nothing has to be configured by hand, and every random choice
uses a fixed seed, so a second run gives the same tables byte for byte.

## 0. Before you start

You need Python 3.9 or newer and git.

```bash
git clone https://github.com/HULTIG/pyteg.git
cd pyteg
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e ".[test]"
python -m pytest                 # about one second; everything should pass
```

The pipeline lives in `pipeline/`, one script per step. You can run them all at
once:

```bash
python pipeline/run_all.py
```

or one at a time, as described below. `make reproduce` does the same thing if
you have make. Each step writes a small `stats.json` next to its output, and
step 7 collects them into the report.

## 1. Download the data

```bash
python pipeline/step1_download.py
```

Eight archives are fetched from the
[nltk_data](https://github.com/nltk/nltk_data) repository, pinned to commit
`550b6625`, and checked against the SHA-256 values in `pipeline/sources.json`.
If a checksum does not match, the step stops. The archives are unpacked into
`data/raw/`.

| Archive | Used for |
|---|---|
| rte | RTE-1, RTE-2 and RTE-3 development and test sets |
| brown, reuters, gutenberg | English counts corpus |
| mac_morpho, floresta | Portuguese counts corpus |
| wordnet | WordNet 3.0 noun hierarchy |
| omw-1.4 | Portuguese lemmas (OpenWN-PT) for the WordNet synsets |

RTE-4 and RTE-5, also used in the thesis, were distributed by NIST under a
user agreement and cannot be downloaded freely. They are not part of this
pipeline, but `pyteg.load_rte_xml` reads their format if you have a copy.

## 2. Build the counts corpora

```bash
python pipeline/step2_prepare_corpora.py
```

Each source text is tokenised and cut into windows of 100 tokens. A window
never mixes two texts, and a trailing window shorter than 20 tokens is dropped.
The output is `data/corpus/en.txt` and `data/corpus/pt.txt`, one window per
line. These windows play the role that Web pages played in the thesis.

Expected size: 50,220 English windows (4.6 M tokens) and 13,056 Portuguese
windows (1.2 M tokens).

## 3. Extract the multiword units

```bash
python pipeline/step3_extract_mwus.py
```

Units of two or three words are extracted from the counts corpora with
LocalMaxs and the SCP_f glue, keeping n-grams seen at least 10 times and
dropping units that start or end with a stop word. Expect 7,851 English and
1,308 Portuguese units in `data/mwu/`.

The thesis used SENTA. If you have SENTA or pysenta output, write it as one unit
per line and use it instead:

```bash
python pipeline/step3_extract_mwus.py --from-file en=my_units_en.txt pt=my_units_pt.txt
```

## 4. Build the co-occurrence indexes

```bash
python pipeline/step4_build_indexes.py
```

Two indexes per language go to `data/index/`: `*_words.npz` for the *all
words* and *without stop words* levels, and `*_mwu.npz`, built after merging
the units, for the *MWU* level.

You can look inside an index with the command-line tool:

```bash
pyteg info --index data/index/en_words.npz
```

## 5. Prepare the benchmarks

```bash
python pipeline/step5_prepare_benchmarks.py
```

This writes three files to `data/bench/`:

* `rte.jsonl`: the 4,167 RTE pairs with their gold labels.
* `hyper_en.jsonl` and `hyper_pt.jsonl`: 2,000 (hyponym, hypernym) pairs
  per language, taken from direct WordNet hypernymy links. Both words must be
  single alphabetic tokens, appear in at least 20 windows and co-occur at
  least once. Pairs found in both directions are removed. The sample is drawn
  with seed 2013.

## 6. Run the experiments

```bash
python pipeline/step6_run_experiments.py
```

This is the only step that takes a few minutes. It writes:

* `results/e1_rte_entailment.csv`: entailment on the RTE test sets, per
  challenge, measure and pre-processing level.
* `results/e2_rte_direction.csv`: direction on the positive RTE test pairs.
* `results/e3_hypernym_direction.csv`: direction on the hypernym pairs.
* `results/run.json`: versions, commit and settings of the run.
* `data/results/*.csv.gz`: the score of every pair, for error analysis.

Options worth knowing:

* `--only e1 e2` runs a subset of the experiments.
* `--normalize none|pairs|first` changes how AISs is scaled (see
  [docs/method.md](docs/method.md#8-interpretation-choices), point 5).
* `--context-size 50` sets the size of the word profiles in E3.

## 7. Write the report

```bash
python pipeline/step7_write_report.py
```

This turns the CSV files into `results/REPORT.md`. The numbers are discussed in
[docs/findings.md](docs/findings.md).

## Checking your run against ours

The committed tables in `results/` come from the reference run. After running
the pipeline, this command should print nothing:

```bash
git diff --stat -- results/*.csv
```

`results/REPORT.md` and `results/run.json` will differ in the date, the run
time and possibly the commit, which is expected. The GitHub workflow
`reproduce.yml` does exactly this check once a month.

Different NumPy or SciPy versions should not change any count. If a table does
differ, the per-item files in `data/results/` are the place to start looking.

## Using Wikipedia instead

The NLTK corpora are small. To count over Wikipedia, install `datasets` and
rebuild from step 2:

```bash
pip install datasets
python pipeline/step2_prepare_corpora.py --source wikipedia --max-articles 200000
python pipeline/run_all.py --from-step 3
```

This streams the `20231101` snapshot of `wikimedia/wikipedia` from the Hugging
Face Hub, in English and in Portuguese. It needs network access to the Hub and
takes much longer. The reference results do not use this option, and this
path was not run for the 0.1.0 release (the tests only cover it with a
mocked dataset), so treat it as a starting point. If you report numbers from
it, say so and give the value of `--max-articles`.

## Starting over

```bash
make clean-data      # or: rm -rf data
```

removes everything downloaded or derived. The `results/` folder is left alone.
