"""Prompt construction for the generation step.

Formats retrieved passages into a numbered context ([1]...[n]), with dedup and
a token budget, and builds the (system, user) prompt pair.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .. import config

PROMPT_VERSION = "structured_answer_v1"

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Use ONLY the provided passages. "
    "Return exactly three lines and nothing else:\n"
    "ANSWER: <shortest exact answer without citations, or UNANSWERABLE>\n"
    "CITATIONS: <comma-separated passage numbers like [1], [3]; or NONE>\n"
    "WHY: <one short supporting sentence with [n] citations; or NONE>\n"
    "If the passages do not contain enough information, output "
    "ANSWER: UNANSWERABLE, CITATIONS: NONE, WHY: NONE."
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (whitespace-separated words) without a tokenizer."""
    return len(text.split())


def build_context(
    passages: Sequence[dict[str, Any]],
    top_n: int = config.TOP_N_CONTEXT,
    max_tokens: int = config.MAX_CONTEXT_TOKENS,
) -> list[dict[str, Any]]:
    """Dedup passages by pid, keep up to top_n, truncate to a token budget."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for p in passages:
        pid = p.get("pid")
        if pid is not None and pid in seen:
            continue
        if pid is not None:
            seen.add(pid)
        deduped.append(p)

    selected: list[dict[str, Any]] = []
    total = 0
    for p in deduped[:top_n]:
        tok = estimate_tokens(p["text"])
        if total + tok > max_tokens:
            break
        selected.append(p)
        total += tok
    return selected


def format_context(passages: Sequence[dict[str, Any]]) -> str:
    """Format passages as numbered lines '[n] text'."""
    return "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))


def format_user_prompt(question: str, context: Sequence[dict[str, Any]]) -> str:
    """Format the question and numbered context into the user prompt."""
    return f"Question: {question}\n\nPassages:\n{format_context(context)}"


def build_prompt(
    question: str,
    passages: Sequence[dict[str, Any]],
    top_n: int = config.TOP_N_CONTEXT,
    max_tokens: int = config.MAX_CONTEXT_TOKENS,
) -> tuple[str, str]:
    """Return (system, user) prompts from a question and retrieved passages."""
    context = build_context(passages, top_n=top_n, max_tokens=max_tokens)
    return SYSTEM_PROMPT, format_user_prompt(question, context)
