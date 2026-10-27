"""Self-implemented BM25 sparse retrieval (pure Python).

Includes tokenization, inverted-index construction, BM25 scoring,
search, and index persistence.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace; return term list."""
    text = text.lower()
    text = _TOKEN_RE.sub(" ", text)
    return text.split()


class BM25:
    """BM25 retriever. Holds the inverted index and document stats; provides search()."""

    def __init__(
        self,
        index: dict[str, dict[str, int]],
        doc_len: dict[str, int],
        df: dict[str, int],
        n_docs: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index = index          # term -> {pid: tf}
        self.doc_len = doc_len      # pid -> |d|
        self.df = df                # term -> document frequency
        self.n_docs = n_docs
        self.k1 = k1
        self.b = b
        self.avgdl = (sum(doc_len.values()) / n_docs) if n_docs else 0.0

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (pid, score) by descending score; ties broken by ascending pid."""
        tokens = tokenize(query)
        scores: dict[str, float] = {}
        for t in tokens:
            postings = self.index.get(t)
            if not postings:
                continue
            idf = math.log(1 + (self.n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for pid, tf in postings.items():
                dl = self.doc_len[pid]
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[pid] = scores.get(pid, 0.0) + idf * num / denom
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:k]


def build_index(
    corpus: dict[str, str] | dict[str, dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25:
    """Build a BM25 index from a corpus.

    corpus may be {pid: text} or {pid: {..., "text": text, ...}}.
    """
    index: dict[str, dict[str, int]] = {}
    doc_len: dict[str, int] = {}
    df: defaultdict[str, int] = defaultdict(int)

    for pid, value in corpus.items():
        text = value["text"] if isinstance(value, dict) else value
        tokens = tokenize(text)
        doc_len[pid] = len(tokens)

        tf: defaultdict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for t, c in tf.items():
            index.setdefault(t, {})[pid] = c
            df[t] += 1

    return BM25(index=index, doc_len=doc_len, df=df, n_docs=len(doc_len), k1=k1, b=b)


def save_index(bm25: BM25, path: str | Path) -> None:
    """Serialize the BM25 index to disk."""
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def load_index(path: str | Path) -> BM25:
    """Deserialize a BM25 index from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
