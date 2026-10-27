# End-to-End RAG Question Answering

A compact retrieval-augmented generation (RAG) pipeline over a Wikipedia
passage corpus built from Natural Questions. The system retrieves relevant
passages with sparse retrieval, feeds the top-ranked context to an LLM, and
returns an answer with validated citations or abstains when the retrieved
evidence is insufficient.

```
corpus (~41K chunks)
  -> sparse retrieval: self-implemented BM25 and TF-IDF (shared inverted index)
  -> top-20 context (deduplicated, token-budgeted)
  -> DeepSeek API generation with structured ANSWER / CITATIONS / WHY output
  -> citation validation and abstention when no supporting evidence is found
```

## Code structure

```text
.
├── build_nq_dataset.py          # build the corpus and QA sets
├── notebooks/
│   ├── 01_build_index.ipynb     # build and cache BM25 + TF-IDF indexes
│   ├── 02_evaluate.ipynb        # compare BM25 vs TF-IDF
│   └── 03_deploy_generate.ipynb # retrieval + generation + evaluation
└── src/rag/
    ├── config.py                # paths, model names, top-k, token budget
    ├── data/loader.py           # load corpus and QA files
    ├── retrieval/bm25.py        # self-implemented BM25
    ├── retrieval/tfidf.py       # self-implemented TF-IDF cosine
    ├── generate/prompt.py       # numbered context and structured prompt
    ├── generate/output.py       # parse ANSWER / CITATIONS / WHY
    ├── generate/llm.py          # DeepSeek API client
    ├── generate/citations.py    # citation parsing and validation
    └── eval/
        ├── metrics.py           # Recall@k, MRR@k, nDCG@k, MAP@k
        ├── answer_quality.py    # answer and citation helpers
        └── ragas_eval.py        # optional RAGAS metrics
```

`src/rag/` contains pure-function modules; the notebooks are the only entry points.

## Dataset

`build_nq_dataset.py` builds the local dataset from Natural Questions. Download
the Natural Questions parquet file, place it beside `build_nq_dataset.py`, then run:

```bash
python build_nq_dataset.py
```

It writes `dataset/` (not committed):

| File | Rows | Contents |
|---|---:|---|
| `corpus.jsonl` | ~41K | 180-token Wikipedia chunks, deduplicated by text |
| `qa_generation.jsonl` | 400 | answerable questions with short gold answers |
| `qa_retrieval.jsonl` | 400 | questions with gold passage ids and 20 hard negatives |
| `qa_unanswerable.jsonl` | 200 | questions without an answer span |

Chunking, sampling and negatives are seeded (`SEED=42`) for reproducibility.
The corpus is not redistributed in this repository.
