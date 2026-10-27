"""Generation evaluation metrics: exact match, token F1, citation quality, abstention.

EM / F1 use SQuAD-style answer normalization. Citation recall / precision / F1
measure whether the passages an answer cites align with the gold relevant pids.
Abstention metrics measure whether the model declines to answer unanswerable
questions without wrongly declining answerable ones.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_CITATION = re.compile(r"\[\d+\]")
_ABSTAIN = re.compile(
    r"cannot answer|can'?t answer|unable to answer|no (relevant|sufficient|enough) information",
    re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    """Lowercase, drop articles and punctuation, collapse whitespace."""
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def em(prediction: str, gold: str | Sequence[str]) -> bool:
    """Exact match: true if the prediction equals any gold answer after normalization."""
    golds = [gold] if isinstance(gold, str) else list(gold)
    pred = normalize_answer(prediction)
    return any(normalize_answer(g) == pred for g in golds)


def f1(prediction: str, gold: str | Sequence[str]) -> float:
    """Token-level F1 against the best gold answer (SQuAD-style)."""
    golds = [gold] if isinstance(gold, str) else list(gold)
    pred_toks = _tokens(prediction)
    best = 0.0
    for g in golds:
        g_toks = _tokens(g)
        common = Counter(pred_toks) & Counter(g_toks)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_toks)
        recall = num_common / len(g_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def citation_recall(cited_pids: Sequence[str], gold_pids: Sequence[str]) -> float:
    """Fraction of gold relevant pids that are cited by the answer."""
    gold = set(gold_pids)
    if not gold:
        return 0.0
    return len(gold & set(cited_pids)) / len(gold)


def citation_precision(cited_pids: Sequence[str], gold_pids: Sequence[str]) -> float:
    """Fraction of cited pids that are actually gold relevant."""
    cited = set(cited_pids)
    if not cited:
        return 0.0
    return len(cited & set(gold_pids)) / len(cited)


def citation_f1(cited_pids: Sequence[str], gold_pids: Sequence[str]) -> float:
    """Harmonic mean of citation recall and precision."""
    recall = citation_recall(cited_pids, gold_pids)
    precision = citation_precision(cited_pids, gold_pids)
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def is_abstention(prediction: str) -> bool:
    """Heuristic: an answer abstains if it declines or cites no passage."""
    if _ABSTAIN.search(prediction):
        return True
    return not _CITATION.search(prediction)


def abstention_accuracy(
    predictions: Sequence[str],
    gold_answerable: Sequence[bool],
) -> float:
    """Fraction of samples where abstention is correct.

    predictions: model outputs; gold_answerable: True if the question has an answer.
    Correct when the model abstains exactly on unanswerable questions.
    """
    if not predictions:
        return 0.0
    correct = 0
    for pred, answerable in zip(predictions, gold_answerable):
        if is_abstention(pred) == (not answerable):
            correct += 1
    return correct / len(predictions)


def abstention_recall(predictions: Sequence[str], gold_answerable: Sequence[bool]) -> float:
    """Of unanswerable questions, fraction the model correctly abstains on."""
    total = 0
    correct = 0
    for pred, answerable in zip(predictions, gold_answerable):
        if not answerable:
            total += 1
            if is_abstention(pred):
                correct += 1
    return correct / total if total else 0.0


def abstention_precision(predictions: Sequence[str], gold_answerable: Sequence[bool]) -> float:
    """Of the questions the model abstains on, fraction that are truly unanswerable."""
    total = 0
    correct = 0
    for pred, answerable in zip(predictions, gold_answerable):
        if is_abstention(pred):
            total += 1
            if not answerable:
                correct += 1
    return correct / total if total else 0.0


def false_abstention_rate(predictions: Sequence[str], gold_answerable: Sequence[bool]) -> float:
    """Of answerable questions, fraction the model wrongly abstains on."""
    total = 0
    wrong = 0
    for pred, answerable in zip(predictions, gold_answerable):
        if answerable:
            total += 1
            if is_abstention(pred):
                wrong += 1
    return wrong / total if total else 0.0
