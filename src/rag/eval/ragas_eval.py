"""RAGAS-based evaluation for generated RAG answers.

RAGAS is imported lazily so retrieval / generation code can run without it.
Execution entry point: notebooks/03_deploy_generate.ipynb.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .. import config

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness",
    "semantic_similarity",
)


def _require_ragas() -> Any:
    """Import ragas lazily and disable its anonymous analytics."""
    try:
        import ragas
    except ImportError as exc:
        raise ImportError(
            "RAGAS is not installed. Run `pip install -r requirements-ragas.txt` "
            "or `pip install ragas==0.4.3 sentence-transformers`."
        ) from exc
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    return ragas


def build_ragas_rows(
    records: Sequence[dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    sample_size: int | None = None,
    seed: int = config.SEED,
) -> list[dict[str, Any]]:
    """Convert generation records into the rows needed by RAGAS metrics.

    Uses ``context_texts`` / ``context_pids`` stored during generation. Records
    without a gold answer or full retrieved context are skipped, so rerun
    generation after upgrading from an older cache.
    """
    rows: list[dict[str, Any]] = []
    max_contexts = config.RAGAS_MAX_CONTEXTS

    for record in records:
        reference = record.get("gold_answer")
        response = record.get("answer")
        if not reference or not response:
            continue

        context_texts = record.get("context_texts")
        if isinstance(context_texts, str):
            context_texts = [context_texts]
        if not context_texts:
            context_pids = record.get("context_pids") or []
            context_texts = [
                corpus[pid]["text"] for pid in context_pids if pid in corpus
            ]
        context_texts = list(context_texts)
        if max_contexts and max_contexts > 0:
            context_texts = context_texts[:max_contexts]
        if not context_texts:
            continue

        rows.append(
            {
                "id": record.get("id"),
                "user_input": record["question"],
                "response": response,
                "retrieved_contexts": context_texts,
                "reference": reference,
                "abstained": bool(record.get("abstained")),
            }
        )

    if sample_size is not None and sample_size > 0 and sample_size < len(rows):
        rows = random.Random(seed).sample(rows, sample_size)
    return rows


def make_llm() -> Any:
    """Create the RAGAS LLM (DeepSeek by default, OpenAI-compatible)."""
    _require_ragas()
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    if not config.RAGAS_API_KEY:
        raise ValueError("RAGAS_API_KEY / DEEPSEEK_API_KEY is not set")

    client = AsyncOpenAI(
        api_key=config.RAGAS_API_KEY,
        base_url=config.RAGAS_BASE_URL,
    )
    return llm_factory(
        config.RAGAS_LLM_MODEL,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=config.RAGAS_MAX_TOKENS,
    )


def _resolve_embedding_model() -> str:
    """Return a local path for the embedding model, downloading it if needed.

    ``huggingface.co`` is unreachable on some networks, so the configured
    mirror endpoint is tried first with the default endpoint as a fallback.
    """
    model = config.RAGAS_EMBEDDING_MODEL
    local = Path(model).expanduser()
    if local.is_dir():
        return str(local)

    from huggingface_hub import snapshot_download

    endpoints: list[str | None] = []
    if config.RAGAS_HF_ENDPOINT:
        endpoints.append(config.RAGAS_HF_ENDPOINT)
    if None not in endpoints:
        endpoints.append(None)

    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            return snapshot_download(repo_id=model, endpoint=endpoint)
        except Exception as exc:  # network/HTTP failures
            last_error = exc
    raise RuntimeError(
        f"Failed to download embedding model {model!r}. Set RAGAS_HF_ENDPOINT "
        "to a reachable mirror or point RAGAS_EMBEDDING_MODEL at a local folder."
    ) from last_error


def make_embeddings() -> Any:
    """Create local HuggingFace embeddings for RAGAS."""
    _require_ragas()
    from ragas.embeddings.base import embedding_factory

    return embedding_factory(
        "huggingface",
        model=_resolve_embedding_model(),
        device=config.RAGAS_EMBEDDING_DEVICE,
        batch_size=32,
    )


def make_scorers(metric_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Build the requested RAGAS metric scorers."""
    _require_ragas()
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
        SemanticSimilarity,
    )

    names = tuple(metric_names or METRIC_NAMES)
    llm = make_llm()
    embeddings = make_embeddings()

    available = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(
            llm=llm, embeddings=embeddings
        ),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
        "factual_correctness": lambda: FactualCorrectness(llm=llm),
        "semantic_similarity": lambda: SemanticSimilarity(embeddings=embeddings),
    }
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown RAGAS metrics: {sorted(unknown)}")
    return {name: available[name]() for name in names}


async def _score_metric(name: str, scorer: Any, row: dict[str, Any]) -> float:
    if name == "faithfulness":
        result = await scorer.ascore(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
        )
    elif name == "answer_relevancy":
        result = await scorer.ascore(
            user_input=row["user_input"],
            response=row["response"],
        )
    elif name in ("context_precision", "context_recall"):
        result = await scorer.ascore(
            user_input=row["user_input"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        )
    elif name in ("factual_correctness", "semantic_similarity"):
        result = await scorer.ascore(
            response=row["response"],
            reference=row["reference"],
        )
    else:
        raise ValueError(f"Unknown RAGAS metric: {name}")
    return float(result.value)


async def _score_row(
    row: dict[str, Any],
    scorers: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}

    async with semaphore:
        results = await asyncio.gather(
            *(_score_metric(name, scorer, row) for name, scorer in scorers.items()),
            return_exceptions=True,
        )

    for name, result in zip(scorers, results):
        if isinstance(result, Exception):
            scores[name] = None
            errors[name] = f"{type(result).__name__}: {result}"
        else:
            scores[name] = result

    return {
        "id": row.get("id"),
        "question": row["user_input"],
        "answer": row["response"],
        "reference": row["reference"],
        "scores": scores,
        "errors": errors,
    }


def _load_cache(path: Path | None, signature: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"__signature__": signature}
    try:
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"__signature__": signature}
    if cache.get("__signature__") != signature:
        return {"__signature__": signature}
    return cache


def _save_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


async def evaluate_ragas(
    rows: Sequence[dict[str, Any]],
    metric_names: Sequence[str] | None = None,
    concurrency: int | None = None,
    cache_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run RAGAS metrics on rows, reusing a per-record JSON cache."""
    ragas = _require_ragas()
    names = tuple(metric_names or METRIC_NAMES)
    cache_file = Path(cache_path) if cache_path else None
    ragas_version = getattr(ragas, "__version__", "unknown")
    signature = (
        f"ragas:{ragas_version}:{config.RAGAS_LLM_MODEL}:"
        f"{config.RAGAS_EMBEDDING_MODEL}:ctx{config.RAGAS_MAX_CONTEXTS}:"
        f"{','.join(names)}"
    )
    cache = _load_cache(cache_file, signature)

    scorers = make_scorers(names)
    semaphore = asyncio.Semaphore(concurrency or config.RAGAS_CONCURRENCY)

    async def run_one(row: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        key = f"{row.get('id')}:{digest}"
        if key in cache and isinstance(cache[key], dict):
            return cache[key]
        result = await _score_row(row, scorers, semaphore)
        cache[key] = result
        return result

    results = await asyncio.gather(*(run_one(row) for row in rows))
    _save_cache(cache_file, cache)
    return list(results)


def summarize_ragas_results(
    results: Iterable[dict[str, Any]],
    metric_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate per-record RAGAS scores."""
    names = tuple(metric_names or METRIC_NAMES)
    results = list(results)
    scores: dict[str, dict[str, Any]] = {}

    for name in names:
        values: list[float] = []
        errors = 0
        for result in results:
            if name in result.get("errors", {}):
                errors += 1
                continue
            value = result.get("scores", {}).get(name)
            if value is None:
                errors += 1
            else:
                values.append(float(value))
        scores[name] = {
            "mean": sum(values) / len(values) if values else None,
            "n": len(values),
            "errors": errors,
        }

    return {
        "num_samples": len(results),
        "llm_model": config.RAGAS_LLM_MODEL,
        "embedding_model": config.RAGAS_EMBEDDING_MODEL,
        "max_contexts": config.RAGAS_MAX_CONTEXTS,
        "scores": scores,
    }


def write_ragas_result(path: str | Path, summary: dict[str, Any]) -> None:
    """Write a compact RAGAS summary table as markdown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# RAGAS results\n\n")
        f.write(f"Samples: {summary['num_samples']}\n\n")
        f.write(f"LLM judge: `{summary.get('llm_model', 'n/a')}`  \n")
        f.write(f"Embeddings: `{summary.get('embedding_model', 'n/a')}`  \n")
        f.write(f"Max contexts: {summary.get('max_contexts', 'n/a')}\n\n")
        f.write("| metric | mean | n | errors |\n")
        f.write("|---|---:|---:|---:|\n")
        for name, stats in summary["scores"].items():
            mean = "n/a" if stats["mean"] is None else f"{stats['mean']:.4f}"
            f.write(f"| {name} | {mean} | {stats['n']} | {stats['errors']} |\n")
