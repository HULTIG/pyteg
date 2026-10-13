# What the reference run shows

These notes discuss the numbers in [`results/REPORT.md`](../results/REPORT.md),
produced by `python pipeline/run_all.py` at commit `d53c38a`. The counts come
from about 4.6 million English tokens (Brown, Reuters-21578, Gutenberg) and
1.2 million Portuguese tokens (Mac-Morpho, Floresta), cut into windows of 100
tokens. That is a small corpus next to the Web, and every conclusion below
should be read with that in mind.

## The short version

The thesis found that the method does not work as a general entailment
detector on RTE. With fixed corpus counts instead of Web hits, that result
holds. The method is meant for something narrower, telling apart entailment
by generality from other entailment, and the corpus that would test this
directly is lost. The two proxies we can build from public data say that the
directional signal is real but weak, and that a good part of it is word
frequency in disguise.

## E1. RTE as a whole

Accuracy stays between 0.48 and 0.53 on the RTE-1 to RTE-3 test sets, against
0.504 for always answering "entailment". The thesis reported 0.51 to 0.55 on
RTE-1 to RTE-5 with Web counts. The ranking of the measures is also similar:
Braun-Blanquet, the conditional probability and Laplace on top (0.525 with all
words; Braun-Blanquet and the conditional probability reach 0.527 with
multiword units), Conviction at the bottom. With 2,400
pairs the 95% interval around 0.5 is about ±0.02, so the best measures are
just above chance and the rest are not distinguishable from it.

This is the expected outcome, and the thesis says as much: half of the RTE
pairs are not entailments at all, and the rule only asks which side is more
general. It was never designed to say whether one sentence follows from the
other.

## E2. Direction on the positive RTE pairs

The more relevant question is whether, given a true entailment, the method
puts the generality on the hypothesis side. For Braun-Blanquet and the
conditional probability it does so in 60.1% of the 1,210 positive test pairs
(60.8% with multiword units), clearly above 50% (±0.028).

Two baselines put that number in context.

* *The shorter sentence is the more general one* is right 98.4% of the time.
  In RTE the hypothesis is almost always much shorter than the text, so the
  direction is close to given away by length. AISs never looks at length (the
  sums in both directions run over the same word pairs), so this baseline is
  not something the method is secretly using, but it shows how easy this
  particular test is for anything that does.
* *The sentence with the more frequent words is the more general one* is
  right only 48.6% of the time, so the measures are not simply averaging word
  frequencies.

Removing stop words turns the result around: Braun-Blanquet and the
conditional probability fall to 38.8%. A large share of the signal therefore
travels through function words. We have not looked into why, and it is a
good first thing to examine in the per-pair scores
(`data/results/rte_pair_scores.csv.gz`).

## E3. Which word is the more general?

WordNet gives thousands of (hyponym, hypernym) pairs where the answer is known.
We kept pairs in which both words are reasonably frequent in the corpus and
co-occur at least once: 2,000 per language.

**Word against word, the conditional probability is a frequency test.** As
shown in [method.md](method.md#9-a-property-worth-knowing), P(x|y) > P(y|x)
holds exactly when x is the more frequent word. The numbers confirm it: P,
Braun-Blanquet, Gini, J-measure and Laplace all make the same decisions as
the frequency baseline, 58.5% in English and 58.9% in Portuguese, and score 0
on the *hard* subset where the hyponym is the more frequent word (41% of the
pairs). Conviction as published looks good on the hard subset (74%) only
because it points the other way, and it loses overall (44%).

**Profiles help a little in Portuguese, not in English.** Comparing the
context vectors of the two words, weighted by co-occurrence, gives 58.1% in
English, the same as frequency, and 61.7% in Portuguese, about three points
above it (the 95% interval is about ±0.021). On the hard subset these profiles
reach only 28%, so they are still mostly driven by frequency. PPMI-weighted
profiles stay close to 50% everywhere.

A note on process: the first version of the pipeline used PPMI profiles only.
After seeing them sit at chance on a 600-pair slice of the English benchmark,
we added count-weighted profiles and made them the default. Both are
reported, and the PPMI numbers should be kept in view when judging the count
numbers.

## What this means for future work

* The idea of entailment by generality stands on its own, and RTE confirms
  that it needs its own evaluation data. Rebuilding a TEG corpus, even a few
  hundred pairs annotated by hand, would make it possible to test the method
  on the task it was built for.
* Co-occurrence-based asymmetry mostly measures frequency when the corpus is
  small. Larger counts (the pipeline can use Wikipedia, see
  [REPRODUCING.md](../REPRODUCING.md#using-wikipedia-instead)) are the obvious
  next check.
* Conditional probabilities do not have to come from counts. Estimating
  P(x|y) with a language model, or comparing against current NLI models on
  the direction task, would show whether the asymmetric view adds anything to
  them.
* Any evaluation of direction should report the length and frequency
  baselines next to the method. Without them, the 60% of E2 and the 58% of E3
  look better than they are.
