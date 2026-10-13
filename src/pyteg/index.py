"""Document co-occurrence counts, the replacement for Web hit counts.

The original experiments asked a search engine how many pages contained a
word, or a pair of words. Those numbers cannot be obtained again: the APIs are
gone and the Web has changed. :class:`CooccurrenceIndex` gives the same kind
of number, a document frequency, computed over a fixed corpus that anyone can
download. A "document" is whatever unit the corpus file puts on one line; the
reproduction pipeline uses windows of 100 tokens.

Internally the index is a binary word-by-document matrix stored in CSR form,
so a joint count is a sparse dot product.
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from .measures import Contingency
from .text import tokenize

__all__ = ["CooccurrenceIndex"]

_FORMAT_VERSION = 2


class CooccurrenceIndex:
    """Binary word-document incidence matrix with fast count queries.

    Build one with :meth:`build` or :meth:`from_corpus_file`, save it with
    :meth:`save` and load it again with :meth:`load`.
    """

    def __init__(self, vocab: Sequence[str], matrix: sparse.csr_matrix, meta: Optional[dict] = None) -> None:
        if matrix.shape[0] != len(vocab):
            raise ValueError("matrix rows must match the vocabulary size")
        self.vocab: List[str] = list(vocab)
        self.word_id: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}
        self._m: sparse.csr_matrix = matrix  # words x documents
        self._docs: Optional[sparse.csr_matrix] = None  # documents x words, built lazily
        self.df: np.ndarray = np.diff(matrix.indptr).astype(np.int64)
        self.meta: dict = dict(meta or {})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        documents: Iterable[Sequence[str]],
        min_df: int = 1,
        vocabulary: Optional[Iterable[str]] = None,
        meta: Optional[dict] = None,
    ) -> "CooccurrenceIndex":
        """Index tokenised documents.

        Parameters
        ----------
        documents:
            An iterable of token sequences, one per document.
        min_df:
            Words found in fewer documents are dropped from the index.
        vocabulary:
            If given, only these words are indexed. The number of documents
            still counts every document, so probabilities stay correct.
        """
        restrict = set(vocabulary) if vocabulary is not None else None
        word_id: Dict[str, int] = {}
        indices = array("i")
        indptr = array("q", [0])
        for doc in documents:
            ids = set()
            for w in doc:
                if restrict is not None and w not in restrict:
                    continue
                i = word_id.get(w)
                if i is None:
                    i = word_id[w] = len(word_id)
                ids.add(i)
            indices.extend(sorted(ids))
            indptr.append(len(indices))
        n_docs = len(indptr) - 1
        ind = np.frombuffer(indices, dtype=np.int32) if len(indices) else np.zeros(0, np.int32)
        ptr = np.frombuffer(indptr, dtype=np.int64)
        docs = sparse.csr_matrix(
            (np.ones(len(ind), dtype=np.int32), ind, ptr), shape=(n_docs, len(word_id))
        )
        matrix = docs.T.tocsr()
        vocab = [""] * len(word_id)
        for w, i in word_id.items():
            vocab[i] = w
        meta = dict(meta or {})
        meta.setdefault("n_docs", n_docs)
        index = cls(vocab, matrix, meta)
        if min_df > 1:
            index = index.prune(min_df)
        return index

    @classmethod
    def from_corpus_file(
        cls,
        path: Union[str, Path],
        min_df: int = 1,
        merger=None,
        vocabulary: Optional[Iterable[str]] = None,
        meta: Optional[dict] = None,
    ) -> "CooccurrenceIndex":
        """Index a text file with one document per line.

        Each line goes through :func:`pyteg.text.tokenize` and, if ``merger``
        is given, through the multiword-unit merger.
        """

        def docs():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    tokens = tokenize(line)
                    yield merger.merge(tokens) if merger is not None else tokens

        meta = dict(meta or {})
        meta.setdefault("corpus", str(Path(path).name))
        meta.setdefault("mwu", merger is not None)
        if merger is not None:
            meta.setdefault("mwu_count", len(merger))
        return cls.build(docs(), min_df=min_df, vocabulary=vocabulary, meta=meta)

    def prune(self, min_df: int) -> "CooccurrenceIndex":
        """Return a copy without the words that appear in fewer than ``min_df`` documents."""
        keep = np.flatnonzero(self.df >= min_df)
        meta = dict(self.meta, min_df=min_df)
        return CooccurrenceIndex([self.vocab[i] for i in keep], self._m[keep], meta)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save to a compressed ``.npz`` file (no pickling involved)."""
        vocab_blob = np.frombuffer(json.dumps(self.vocab, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
        meta_blob = np.frombuffer(
            json.dumps(dict(self.meta, format_version=_FORMAT_VERSION)).encode("utf-8"), dtype=np.uint8
        )
        np.savez_compressed(
            path,
            indptr=self._m.indptr.astype(np.int64),
            indices=self._m.indices.astype(np.int32),
            shape=np.array(self._m.shape, dtype=np.int64),
            vocab=vocab_blob,
            meta=meta_blob,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CooccurrenceIndex":
        with np.load(path, allow_pickle=False) as data:
            indptr = data["indptr"]
            indices = data["indices"]
            shape = tuple(int(x) for x in data["shape"])
            vocab_text = data["vocab"].tobytes().decode("utf-8")
            meta = json.loads(data["meta"].tobytes().decode("utf-8"))
        if meta.get("format_version", 1) >= 2:
            vocab = json.loads(vocab_text)
        else:  # version 1 stored one word per line
            vocab = vocab_text.split("\n") if vocab_text else []
        matrix = sparse.csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr), shape=shape)
        return cls(vocab, matrix, meta)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return int(self._m.shape[1])

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, word: str) -> bool:
        return word in self.word_id

    def __repr__(self) -> str:
        return f"CooccurrenceIndex(words={len(self.vocab):,}, documents={self.n_docs:,})"

    def freq(self, word: str) -> int:
        """Number of documents that contain ``word`` (0 if unknown)."""
        i = self.word_id.get(word)
        return 0 if i is None else int(self.df[i])

    def ids(self, words: Sequence[str]) -> np.ndarray:
        """Word ids, with -1 for words that are not in the index."""
        return np.array([self.word_id.get(w, -1) for w in words], dtype=np.int64)

    def joint_matrix(self, xs: Sequence[str], ys: Sequence[str]) -> np.ndarray:
        """Matrix of joint document counts ``n(x_i, y_j)``.

        Unknown words give rows or columns of zeros.
        """
        xi, yi = self.ids(xs), self.ids(ys)
        out = np.zeros((len(xs), len(ys)), dtype=np.int64)
        kx, ky = np.flatnonzero(xi >= 0), np.flatnonzero(yi >= 0)
        if len(kx) and len(ky):
            block = (self._m[xi[kx]] @ self._m[yi[ky]].T).toarray()
            out[np.ix_(kx, ky)] = block
        return out

    def joint(self, x: str, y: str) -> int:
        """Number of documents that contain both ``x`` and ``y``."""
        return int(self.joint_matrix([x], [y])[0, 0])

    def freqs(self, words: Sequence[str]) -> np.ndarray:
        ids = self.ids(words)
        out = np.zeros(len(words), dtype=np.int64)
        known = ids >= 0
        out[known] = self.df[ids[known]]
        return out

    def contingency(self, xs: Sequence[str], ys: Sequence[str], smoothing: float = 0.0) -> Tuple[Contingency, np.ndarray]:
        """2x2 tables for every pair ``(x_i, y_j)``.

        Returns the :class:`~pyteg.measures.Contingency` (with array cells of
        shape ``(len(xs), len(ys))``) and a boolean mask that is ``True`` where
        both words are known to the index.
        """
        joint = self.joint_matrix(xs, ys)
        fx = self.freqs(xs)[:, None]
        fy = self.freqs(ys)[None, :]
        known = (fx > 0) & (fy > 0)
        table = Contingency.from_counts(joint, np.broadcast_to(fx, joint.shape), np.broadcast_to(fy, joint.shape), self.n_docs, smoothing)
        return table, known

    # ------------------------------------------------------------------
    # Distributional profiles
    # ------------------------------------------------------------------
    def _doc_matrix(self) -> sparse.csr_matrix:
        if self._docs is None:
            self._docs = self._m.T.tocsr()
        return self._docs

    def cooccurrence_counts(self, word: str) -> np.ndarray:
        """Joint document counts of ``word`` with every word of the vocabulary."""
        i = self.word_id.get(word)
        if i is None:
            return np.zeros(len(self.vocab), dtype=np.int64)
        docs = self._m.indices[self._m.indptr[i] : self._m.indptr[i + 1]]
        return np.asarray(self._doc_matrix()[docs].sum(axis=0)).ravel().astype(np.int64)

    def context_vector(
        self,
        word: str,
        size: int = 50,
        min_df: int = 5,
        exclude: Iterable[str] = (),
        weighting: str = "count",
    ) -> Dict[str, float]:
        """A distributional profile of ``word``: its ``size`` main context words.

        ``weighting`` decides both which context words are kept and their
        weight:

        * ``"count"`` (default): the words that co-occur most often with
          ``word``, weighted by ``P(c | word) = n(word, c) / n(word)``.
        * ``"ppmi"``: the words with the highest positive pointwise mutual
          information, ``log(n(word, c) * N / (n(word) * n(c)))``.

        Context words seen in fewer than ``min_df`` documents, the word itself
        and anything in ``exclude`` (typically stop words) are left out.
        """
        if weighting not in ("count", "ppmi"):
            raise ValueError("weighting must be 'count' or 'ppmi'")
        i = self.word_id.get(word)
        if i is None:
            return {}
        counts = self.cooccurrence_counts(word).astype(float)
        counts[i] = 0.0
        for w in exclude:
            j = self.word_id.get(w)
            if j is not None:
                counts[j] = 0.0
        valid = (counts > 0) & (self.df >= min_df)
        score = np.full(len(self.vocab), -np.inf)
        if weighting == "count":
            score[valid] = counts[valid] / self.df[i]
        else:
            score[valid] = np.log(counts[valid] * self.n_docs / (self.df[i] * self.df[valid]))
        candidates = np.flatnonzero(score > 0)
        if len(candidates) == 0:
            return {}
        top = candidates[np.argsort(-score[candidates], kind="stable")[:size]]
        return {self.vocab[j]: float(score[j]) for j in top}

    def top_words(self, k: int = 20) -> List[Tuple[str, int]]:
        order = np.argsort(-self.df, kind="stable")[:k]
        return [(self.vocab[i], int(self.df[i])) for i in order]

    def word_counts(self) -> Mapping[str, int]:
        return {w: int(c) for w, c in zip(self.vocab, self.df)}
