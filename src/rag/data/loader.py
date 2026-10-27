"""Load the JSONL datasets under code/dataset into a uniform structure.

All functions only load data; no processing/evaluation logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import config


def _read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file line by line, returning list[dict]."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the corpus as {pid: {text, source_doc, source_url, has_answer, qids}}."""
    path = Path(path) if path else config.CORPUS_PATH
    corpus: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        corpus[row["pid"]] = {
            "text": row["text"],
            "source_doc": row.get("source_doc"),
            "source_url": row.get("source_url"),
            "has_answer": row.get("has_answer", False),
            "qids": row.get("qids", []),
        }
    return corpus


def load_corpus_list(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the corpus as list[dict] (raw rows, including pid)."""
    path = Path(path) if path else config.CORPUS_PATH
    return _read_jsonl(path)


def load_qa_retrieval(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Retrieval eval set: {id, question, gold_pids, negatives}."""
    path = Path(path) if path else config.QA_RETRIEVAL_PATH
    return _read_jsonl(path)


def load_qa_generation(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Generation eval set: {id, question, gold_answer, gold_pids}."""
    path = Path(path) if path else config.QA_GENERATION_PATH
    return _read_jsonl(path)


def load_qa_unanswerable(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Unanswerable eval set: {id, question}."""
    path = Path(path) if path else config.QA_UNANSWERABLE_PATH
    return _read_jsonl(path)
