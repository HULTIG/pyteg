# The method, and how this implementation reads it

This page describes what pyteg computes, step by step, and lists every place
where the original description left room for interpretation. When the thesis
and the later papers disagree, or when a detail was never written down, the
choice made here is stated together with the reason for it.

The sources are the thesis (Pais, 2013), the two CLIB 2014 papers (Pais, Dias
& Moraliyski; Pais, Dias, Cordeiro & Moraliyski) and the journal version
(Pais & Dias, 2020). Full references are in the [README](../README.md#citing).

## 1. Entailment by generality

A text T entails a hypothesis H *by generality* when H says roughly the same
thing as T, only in more general terms. The usual example is that
*strawberry* implies *fruit* but *fruit* does not imply *strawberry*. A
sentence-level example from RTE-1 (pair 754):

> T: Mexico City has a very bad pollution problem because the mountains around
> the city act as walls and block in dust and smog.
>
> H: Poor air circulation out of the mountain-walled Mexico City aggravates
> pollution.

The method assumes two things: the two sentences share many related words,
and the words of H tend to be more general than the words of T. It does not
try to decide whether T entails H in the first place. In the thesis the input
is a pair already known to be an entailment, and the question is whether the
entailment goes from the specific to the general.

## 2. Counting

Every measure is built from four numbers for a word pair *(x, y)*: the number
of documents containing *x*, the number containing *y*, the number containing
both, and the size of the collection *N*.

The original experiments took these numbers from a Web search engine (hit
counts). That is no longer possible, and hit counts were never reproducible
anyway: they are estimates that change from one day to the next. pyteg counts
documents in a fixed corpus instead (`CooccurrenceIndex`). In the
reproduction pipeline a document is a window of 100 consecutive tokens taken
from one source text, which is close to the size of a paragraph.

Words that are not in the index are left out of every sum, in both
directions, so they cannot tilt a decision.

## 3. Asymmetric association measures

The notation AS(x‖y) reads "how strongly *y* points to *x*". With the
conditional probability, AS(x‖y) = P(x | y): if *x* is present in almost
every document where *y* is, *x* behaves as the more general word.

| Code | Measure | AS(x‖y) |
|---|---|---|
| P  | Conditional probability | P(x\|y) |
| AV | Added Value | P(x\|y) − P(x) |
| BB | Braun-Blanquet (as published) | f(x,y) / (f(x,y) + f(¬x,y)) |
| CF | Certainty Factor | (P(x\|y) − P(x)) / (1 − P(x)) |
| CO | Conviction (as published) | P(x) P(¬y) / P(x,¬y) |
| GI | Gini index | P(y)[P(x\|y)² + P(¬x\|y)²] + P(¬y)[P(x\|¬y)² + P(¬x\|¬y)²] − P(x)² − P(¬x)² |
| JM | J-measure | P(x,y) log(P(x\|y)/P(x)) + P(¬x,y) log(P(¬x\|y)/P(¬x)) |
| LP | Laplace | (N P(x,y) + 1) / (N P(y) + 2) |
| CO' | Conviction, y to x (added) | P(y) P(¬x) / P(y,¬x) |

Probabilities are relative document frequencies. Undefined values (a division
by zero) are returned as 0.

## 4. From words to sentences: AISs

InfoSimba compares two bags of words through the similarity of every word of
one bag with every word of the other. Its asymmetric, simplified form is

$$
\mathrm{AISs}(X_i \parallel X_j) = \sum_{k} \sum_{l} X_{ik}\, X_{jl}\, \mathrm{AS}(W_{ik} \parallel W_{jl}), \qquad W_{ik} \neq W_{jl}
$$

where *W_ik* is the k-th word of the first bag and *X_ik* its weight. For
sentences every word has weight 1, and repeated words count once.

The full AIS, with the InfoSimba denominator, is also available as
`pyteg.ais`, but the decision rule below uses AISs, as the thesis does.

## 5. The decision rule

$$
T \rightarrow_G H \iff \mathrm{AISs}(T \parallel H) < \mathrm{AISs}(H \parallel T)
$$

With the conditional probability the left side averages P(t | h) over the
word pairs and the right side averages P(h | t). If the words of H are the
general ones, they appear wherever the words of T do, so P(h | t) is large
and P(t | h) is small. No threshold is needed. A tie, or a pair with no usable
word pair, is read as "no entailment by generality".

## 6. Pre-processing levels

* **All words**: every token.
* **Without stop words**: the Snowball lists for English and Portuguese are
  removed from the sentences. The counts are not affected.
* **With multiword units (MWUs)**: known units are merged into one token
  (`new york` becomes `new_york`) both in the sentences and in the corpus
  before counting, so the MWU level has its own index. Stop words are kept,
  as in the thesis, whose worked example includes the unit *with a*.

## 7. Word level and attributional profiles

`word_generality(index, x, y)` returns AS(x‖y) and AS(y‖x) for two words.

`attributional_generality(index, x, y)` represents each word by a profile of
its main context words and compares the two profiles with AISs, as the
journal version does. With profiles the weights matter. Two options are
offered:

* `count` (default): the 50 words that co-occur most often with the target,
  weighted by P(c | target).
* `ppmi`: the 50 words with the highest positive pointwise mutual
  information, weighted by that value.

Stop words are excluded from profiles in the reproduction pipeline.

## 8. Interpretation choices

These are the points where a reader of the original papers has to make a
decision. Each one is fixed in the code and covered by a test.

1. **Counts.** Web hits are replaced by document counts over a fixed corpus
   (section 2). This is the largest departure and the reason exact numbers
   cannot be expected to match.

2. **Braun-Blanquet equals the conditional probability.** As published,
   BB(x‖y) = f(x,y) / (f(x,y) + f(¬x,y)). With exact counts the denominator
   is f(y), so BB and P give identical values and identical decisions. The
   thesis reports different results for the two, which can only come from Web
   estimates that do not add up (a query for "y −x" is not answered with
   hits(y) − hits(x,y)). We keep the published formula and say so, rather
   than invent a different one. `tests/test_measures.py` checks the identity.

3. **Conviction points the other way.** As published, CO(x‖y) is large when
   *x* implies *y*, that is, when *y* is the general word. Every other measure
   has the opposite orientation, so the decision rule applied to CO picks the
   opposite direction. We keep CO as published and add CO', the same measure
   with the orientation of the others. A pseudo-count of 0.5 keeps both
   finite when the denominator would be zero.

4. **Gini index.** The text extracted from the 2020 article shows a ratio,
   which looks like a typesetting accident. We use the sum form of the CLIB
   2014 papers, which is the standard Gini index of the rule y → x (Tan,
   Kumar & Srivastava, 2004).

5. **Normalising AISs.** The 2014 papers use the raw double sum. The 2020
   article divides by p², where p is described as the length of both
   vectors, which is ambiguous when they differ in length. Our default
   divides both directions by the same number of word pairs, which gives an
   average and the same decisions as the raw sum. `normalize="none"` gives the
   raw sum and `normalize="first"` divides by the squared length of the first
   argument, our best reading of the 2020 formula, which *can* change
   decisions.

6. **Identical words.** Pairs with the same word on both sides are skipped,
   following the condition W_ik ≠ W_jl of the 2020 article.

7. **Unknown words.** Pairs involving a word that the index does not know are
   skipped (section 2). With Web counts almost nothing was unknown, so the
   thesis does not discuss this case.

8. **Stop-word lists.** The thesis used a Microsoft list for English and a
   Linguateca list for Portuguese. We ship the Snowball lists, which are
   BSD-licensed and easy to cite.

9. **Multiword units.** The thesis extracted units with SENTA from the RTE
   test sets. The pipeline extracts them with LocalMaxs and the SCP_f glue
   from the counts corpus, which keeps the test data out of the counts.
   Units produced by SENTA (or pysenta) can be plugged in with
   `step3_extract_mwus.py --from-file`.

10. **Tokenisation.** Lower-cased runs of letters and digits, keeping
    internal hyphens and apostrophes. No stemming or lemmatisation, in line
    with the language-independent spirit of the method.

11. **Logarithms.** The J-measure uses the natural logarithm. Any base gives
    the same decisions.

## 9. A property worth knowing

For the conditional probability, comparing the two directions at word level
is the same as comparing frequencies:

$$
\frac{P(x \mid y)}{P(y \mid x)} = \frac{n(x,y)/n(y)}{n(x,y)/n(x)} = \frac{n(x)}{n(y)}
$$

So, word against word, P (and BB as published) simply says that the more
frequent word is the more general one. Hypernyms are often more frequent than
their hyponyms, which is why this works more often than not, and why it fails
exactly when the hyponym is the more common word. At sentence level the sums
mix many pairs and the equivalence no longer holds, but frequency is still a
large part of what the measure sees. The reproduction report checks this with
a frequency baseline and with a "hard" subset of hypernym pairs. See
[findings.md](findings.md).
