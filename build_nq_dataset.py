"""Build a local-friendly RAG corpus + QA dataset from a Natural Questions shard.

Input:  validation-00000-of-00007.parquet (HuggingFace NQ validation shard)
Output: code/dataset/
    corpus.jsonl           ~41K unique chunks {pid, text, source_doc, source_url, has_answer, qids}
    qa_generation.jsonl    400  {id, question, gold_answer, gold_pids}
    qa_retrieval.jsonl     400  {id, question, gold_pids, negatives}
    qa_unanswerable.jsonl  200  {id, question}

Everything is reproducible (seeded). The corpus is built by chunking each
example's own Wikipedia document (~180 text tokens per chunk) and then
deduplicating chunks with identical text (some Wikipedia articles are shared
by two questions, which would otherwise create duplicate chunks under
different pids). Gold pids and negatives are remapped to canonical pids.
"""

import json
import os
import random

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(SCRIPT_DIR, "validation-00000-of-00007.parquet")
OUT_DIR = os.path.join(SCRIPT_DIR, "dataset")

CHUNK_SIZE = 180          # text tokens per chunk
NEGATIVES_PER_QUERY = 20  # hard negatives per retrieval query
SEED = 42

N_GEN = 400
N_RETRIEVAL = 400
N_UNANSWERABLE = 200


def int_or_none(x):
    if x is None:
        return None
    try:
        v = int(x)
    except (TypeError, ValueError):
        return None
    return v


def collect_long_spans(annotation):
    """Return list of (start, end) token spans from the 5 long_answer annotators."""
    spans = []
    for item in annotation.get("long_answer", []):
        start = int_or_none(item.get("start_token"))
        end = int_or_none(item.get("end_token"))
        if start is not None and end is not None and start >= 0 and end > start:
            spans.append((start, end))
    return spans


def collect_short_spans(annotation):
    """Return list of (start, end) spans plus the gold answer text.

    NQ short_answers has 5 annotators, each possibly with multiple spans.
    """
    spans = []
    gold_texts = []
    for item in annotation.get("short_answers", []):
        starts = np.asarray(item.get("start_token", []), dtype=int)
        ends = np.asarray(item.get("end_token", []), dtype=int)
        texts = np.asarray(item.get("text", []), dtype=object)
        for s, e in zip(starts, ends):
            s, e = int(s), int(e)
            if s >= 0 and e > s:
                spans.append((s, e))
        for t in texts:
            t = str(t).strip()
            if t:
                gold_texts.append(t)
    return spans, gold_texts


def chunk_document(document):
    """Chunk a document's non-HTML tokens into ~CHUNK_SIZE groups.

    Returns a list of dicts: {text, start_orig, end_orig} where start/end are
    inclusive-exclusive original token index bounds.
    """
    tokens = np.asarray(document["tokens"]["token"], dtype=object)
    is_html = np.asarray(document["tokens"]["is_html"], dtype=bool)

    chunks = []
    buf_tokens = []
    buf_indices = []
    for i, (tok, html) in enumerate(zip(tokens, is_html)):
        if html:
            continue
        buf_tokens.append(str(tok))
        buf_indices.append(i)
        if len(buf_tokens) >= CHUNK_SIZE:
            chunks.append({
                "text": " ".join(buf_tokens),
                "start_orig": buf_indices[0],
                "end_orig": buf_indices[-1] + 1,
            })
            buf_tokens = []
            buf_indices = []
    if buf_tokens:
        chunks.append({
            "text": " ".join(buf_tokens),
            "start_orig": buf_indices[0],
            "end_orig": buf_indices[-1] + 1,
        })
    return chunks


def intersects(span, chunk):
    """Check whether token span (start, end) overlaps a chunk's orig bounds."""
    s, e = span
    return s < chunk["end_orig"] and e > chunk["start_orig"]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    df = pd.read_parquet(PARQUET_PATH)

    chunk_meta = []              # list of dicts per raw chunk (pre-dedup)
    all_chunks_by_qid = {}       # qid -> list of raw pids (ordered)
    gold_long_by_qid = {}        # qid -> set of raw pids (long answer chunks)
    gold_short_by_qid = {}       # qid -> set of raw pids (short answer chunks)
    gold_answer_text = {}        # qid -> short answer string
    question_text = {}           # qid -> question

    long_answer_qids = []
    short_answer_qids = []
    unanswerable_qids = []

    for _, row in df.iterrows():
        qid = str(row["id"])
        question = str(row["question"]["text"]).strip()
        question_text[qid] = question

        long_spans = collect_long_spans(row["annotations"])
        short_spans, gold_texts = collect_short_spans(row["annotations"])

        chunks = chunk_document(row["document"])
        pids = []
        long_hits = set()
        short_hits = set()
        for idx, chunk in enumerate(chunks):
            pid = f"{qid}_c{idx}"
            pids.append(pid)
            hit_long = any(intersects(sp, chunk) for sp in long_spans)
            hit_short = any(intersects(sp, chunk) for sp in short_spans)
            if hit_long:
                long_hits.add(pid)
            if hit_short:
                short_hits.add(pid)
            chunk_meta.append({
                "pid": pid,
                "text": chunk["text"],
                "source_doc": str(row["document"]["title"]),
                "source_url": str(row["document"]["url"]),
                "has_answer": bool(hit_long or hit_short),
                "qid": qid,
            })

        all_chunks_by_qid[qid] = pids
        gold_long_by_qid[qid] = long_hits
        gold_short_by_qid[qid] = short_hits
        if gold_texts:
            gold_answer_text[qid] = gold_texts[0]

        if short_spans and gold_texts:
            short_answer_qids.append(qid)
        if long_spans:
            long_answer_qids.append(qid)
        if not long_spans and not short_spans:
            unanswerable_qids.append(qid)

    # ---- dedup: identical text -> canonical pid (first occurrence) ----
    text_to_canonical = {}
    for m in chunk_meta:
        if m["text"] not in text_to_canonical:
            text_to_canonical[m["text"]] = m["pid"]
    pid_remap = {m["pid"]: text_to_canonical[m["text"]] for m in chunk_meta}

    # remap gold / all-chunk pids to canonical
    gold_long_by_qid = {
        qid: {pid_remap[p] for p in s} for qid, s in gold_long_by_qid.items()
    }
    gold_short_by_qid = {
        qid: {pid_remap[p] for p in s} for qid, s in gold_short_by_qid.items()
    }
    all_chunks_by_qid = {
        qid: list(dict.fromkeys(pid_remap[p] for p in pids))
        for qid, pids in all_chunks_by_qid.items()
    }

    # ---- build deduped corpus entries (merge qids / has_answer) ----
    merged = {}
    for m in chunk_meta:
        cp = pid_remap[m["pid"]]
        if cp not in merged:
            merged[cp] = {
                "text": m["text"],
                "source_doc": m["source_doc"],
                "source_url": m["source_url"],
                "has_answer": m["has_answer"],
                "qids": set(),
            }
        merged[cp]["has_answer"] = merged[cp]["has_answer"] or m["has_answer"]
        merged[cp]["qids"].add(m["qid"])

    corpus_entries = []
    for text, cp in text_to_canonical.items():
        e = merged[cp]
        corpus_entries.append({
            "pid": cp,
            "text": e["text"],
            "source_doc": e["source_doc"],
            "source_url": e["source_url"],
            "has_answer": e["has_answer"],
            "qids": sorted(e["qids"]),
        })

    # ---- sample the three QA sets ----
    gen_qids = rng.choice(np.array(short_answer_qids, dtype=object),
                          size=min(N_GEN, len(short_answer_qids)),
                          replace=False).tolist()
    ret_qids = rng.choice(np.array(long_answer_qids, dtype=object),
                          size=min(N_RETRIEVAL, len(long_answer_qids)),
                          replace=False).tolist()
    noans_qids = rng.choice(np.array(unanswerable_qids, dtype=object),
                            size=min(N_UNANSWERABLE, len(unanswerable_qids)),
                            replace=False).tolist()

    # ---- global pool of non-gold pids for fallback negatives ----
    all_gold = set()
    for s in list(gold_long_by_qid.values()) + list(gold_short_by_qid.values()):
        all_gold |= s
    canonical_pids = list(text_to_canonical.values())
    non_gold_pool = [p for p in canonical_pids if p not in all_gold]

    # ---- write qa_generation.jsonl ----
    gen_rows = []
    for qid in gen_qids:
        gen_rows.append({
            "id": qid,
            "question": question_text[qid],
            "gold_answer": gold_answer_text.get(qid, ""),
            "gold_pids": sorted(gold_short_by_qid[qid]),
        })

    # ---- write qa_retrieval.jsonl (with hard negatives) ----
    ret_rows = []
    for qid in ret_qids:
        gold = gold_long_by_qid[qid]
        same_doc = [p for p in all_chunks_by_qid[qid] if p not in gold]
        hard = rng.choice(np.array(same_doc, dtype=object),
                          size=min(len(same_doc), NEGATIVES_PER_QUERY),
                          replace=False).tolist() if same_doc else []
        if len(hard) < NEGATIVES_PER_QUERY and non_gold_pool:
            extra = rng.choice(np.array(non_gold_pool, dtype=object),
                               size=min(NEGATIVES_PER_QUERY - len(hard), len(non_gold_pool)),
                               replace=False).tolist()
            hard.extend(extra)
        ret_rows.append({
            "id": qid,
            "question": question_text[qid],
            "gold_pids": sorted(gold),
            "negatives": hard,
        })

    # ---- write qa_unanswerable.jsonl ----
    noans_rows = [{"id": qid, "question": question_text[qid]} for qid in noans_qids]

    os.makedirs(OUT_DIR, exist_ok=True)

    def write_jsonl(name, rows):
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    write_jsonl("corpus.jsonl", corpus_entries)
    write_jsonl("qa_generation.jsonl", gen_rows)
    write_jsonl("qa_retrieval.jsonl", ret_rows)
    write_jsonl("qa_unanswerable.jsonl", noans_rows)

    # ---- verification ----
    corpus_by_pid = {e["pid"]: e for e in corpus_entries}
    missing = 0
    for r in gen_rows:
        for p in r["gold_pids"]:
            if p not in corpus_by_pid:
                missing += 1
    for r in ret_rows:
        for p in r["gold_pids"]:
            if p not in corpus_by_pid:
                missing += 1
        for p in r["negatives"]:
            if p not in corpus_by_pid:
                missing += 1

    print("Raw chunks before dedup:", len(chunk_meta))
    print("Corpus chunks after dedup:", len(corpus_entries))
    print("qa_generation:", len(gen_rows))
    print("qa_retrieval:", len(ret_rows))
    print("qa_unanswerable:", len(noans_rows))
    print("missing gold/negative pids in corpus:", missing)
    print("Wrote files to", OUT_DIR)


if __name__ == "__main__":
    main()
