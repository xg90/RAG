"""Parse the structured ANSWER / CITATIONS / WHY output from the generator.

The system prompt asks for exactly three lines:

    ANSWER: <short answer or UNANSWERABLE>
    CITATIONS: <passage numbers like [1], [3]; or NONE>
    WHY: <one supporting sentence with [n]; or NONE>

The parser is deliberately tolerant: it strips markdown fences/bold markers and
falls back to the raw text if the model does not follow the format, so one bad
generation cannot break a batch run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FIELD_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(ANSWER|CITATIONS|WHY)\s*:\s*"
    r"(.*?)(?=\n\s*(?:[-*]\s*)?(?:ANSWER|CITATIONS|WHY)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_ABSTAIN_RE = re.compile(
    r"unanswerable|cannot answer|can'?t answer|unable to answer|"
    r"do(?:es)? not contain enough information|not enough information|"
    r"insufficient information",
    re.IGNORECASE,
)


@dataclass
class ParsedAnswer:
    """Structured view of one generated answer."""

    answer: str
    citations: list[int]
    evidence: str
    abstained: bool
    parsed: bool
    raw: str


def _clean(text: str) -> str:
    return text.strip().strip("`*_ \t\r\n")


def _parse_citations(text: str) -> list[int]:
    nums = [int(n) for n in _CITATION_RE.findall(text)]
    if not nums and re.fullmatch(r"[\d,\s]+", text or ""):
        nums = [int(n) for n in re.findall(r"\d+", text)]
    return list(dict.fromkeys(n for n in nums if n > 0))


def parse_structured_answer(raw: str) -> ParsedAnswer:
    """Parse ANSWER / CITATIONS / WHY, with a raw-text fallback."""
    text = raw or ""
    text = text.replace("**", "")
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")

    fields: dict[str, str] = {}
    for match in _FIELD_RE.finditer(text):
        key = match.group(1).upper()
        value = _clean(match.group(2))
        if key not in fields or not fields[key]:
            fields[key] = value

    parsed = all(key in fields for key in ("ANSWER", "CITATIONS", "WHY"))
    answer = fields.get("ANSWER", "")
    evidence = fields.get("WHY", "")
    if not answer:
        answer = _clean(text)
    answer = _clean(_CITATION_RE.sub("", answer))

    citations = _parse_citations(fields.get("CITATIONS", ""))
    if not citations:
        citations = _parse_citations(evidence)
    if not citations:
        citations = _parse_citations(text)

    abstained = bool(_ABSTAIN_RE.search(answer))
    if not abstained and not citations and answer.strip().upper() == "NONE":
        abstained = True

    return ParsedAnswer(
        answer=answer,
        citations=citations,
        evidence=evidence,
        abstained=abstained,
        parsed=parsed,
        raw=text,
    )
