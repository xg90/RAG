"""TF-IDF sparse retrieval built on an inverted index (pure Python).

Uses the same inverted index `term -> {pid: tf}` as BM25, but ranks with the
classic TF-IDF cosine score (sublinear tf, smooth idf, L2-normalized). It is the
baseline against which BM25's length-normalized saturation scoring is compared.
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from .bm25 import tokenize


def _tf_weight(tf: int) -> float:
    """Sublinear term frequency: 1 + log(tf)."""
    return 1.0 + math.log(tf)


class TFIDF:
    """TF-IDF retriever; ranks by cosine similarity over tf-idf vectors."""

    def __init__(
        self,
        index: dict[str, dict[str, int]],
        idf: dict[str, float],
        doc_norm: dict[str, float],
        n_docs: int,
    ) -> None:
        self.index = index          # term -> {pid: tf}
        self.idf = idf              # term -> idf
        self.doc_norm = doc_norm    # pid -> L2 norm of its tf-idf vector
        self.n_docs = n_docs

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (pid, cosine) by descending score; ties broken by ascending pid."""
        qtf: defaultdict[str, int] = defaultdict(int)
        for t in tokenize(query):
            qtf[t] += 1

        q_weights = {t: _tf_weight(tf) * self.idf.get(t, 0.0) for t, tf in qtf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0.0:
            return []

        dots: dict[str, float] = {}
        for t, qw in q_weights.items():
            if qw == 0.0:
                continue
            postings = self.index.get(t)
            if not postings:
                continue
            idf_t = self.idf[t]
            for pid, tf in postings.items():
                dots[pid] = dots.get(pid, 0.0) + _tf_weight(tf) * idf_t * qw

        ranked: list[tuple[str, float]] = []
        for pid, dot in dots.items():
            denom = self.doc_norm[pid] * q_norm
            ranked.append((pid, dot / denom if denom else 0.0))
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked[:k]


def build_index(
    corpus: dict[str, str] | dict[str, dict[str, Any]],
) -> TFIDF:
    """Build a TF-IDF retriever over an inverted index from a corpus.

    corpus may be {pid: text} or {pid: {..., "text": text, ...}}.
    """
    index: dict[str, dict[str, int]] = {}
    df: defaultdict[str, int] = defaultdict(int)
    tf_by_doc: dict[str, dict[str, int]] = {}

    for pid, value in corpus.items():
        text = value["text"] if isinstance(value, dict) else value
        tf: defaultdict[str, int] = defaultdict(int)
        for t in tokenize(text):
            tf[t] += 1
        tf_by_doc[pid] = dict(tf)
        for t, c in tf.items():
            index.setdefault(t, {})[pid] = c
            df[t] += 1

    n_docs = len(tf_by_doc)
    idf = {t: math.log((1 + n_docs) / (1 + d)) + 1.0 for t, d in df.items()}

    doc_norm: dict[str, float] = {}
    for pid, tf in tf_by_doc.items():
        doc_norm[pid] = math.sqrt(
            sum((_tf_weight(c) * idf[t]) ** 2 for t, c in tf.items())
        )

    return TFIDF(index=index, idf=idf, doc_norm=doc_norm, n_docs=n_docs)


def save_index(idx: TFIDF, path: str | Path) -> None:
    """Serialize the TF-IDF index to disk."""
    with open(path, "wb") as f:
        pickle.dump(idx, f)


def load_index(path: str | Path) -> TFIDF:
    """Deserialize a TF-IDF index from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
