"""Citation parsing and validation for generated answers.

Citations in the answer appear as bracketed numbers [1], [2], ... that index the
numbered passages in the prompt (1-based). This module extracts them, keeps the
ones that fall in a valid range, and maps them to the retrieved pids.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

_CITATION_RE_PATTERN = r"\[(\d+)\]"


def parse_citations(answer: str) -> list[int]:
    """Extract citation numbers [n] from an answer; dedup, preserve order."""
    import re

    return list(dict.fromkeys(int(m) for m in re.findall(_CITATION_RE_PATTERN, answer)))


def validate_citations(citation_nums: Iterable[int], num_passages: int) -> list[int]:
    """Keep only citation numbers within the valid range [1, num_passages]."""
    return [n for n in citation_nums if 1 <= n <= num_passages]


def map_citations(citation_nums: Iterable[int], pids: Sequence[str]) -> dict[int, str]:
    """Map valid citation numbers to pids (1-indexed); invalid numbers are dropped."""
    return {n: pids[n - 1] for n in citation_nums if 1 <= n <= len(pids)}
