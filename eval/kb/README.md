# Internal KB single-query evaluation prototype

`queries.jsonl` contains 12 single-turn EEG/MEG questions anchored to the 9-paper,
123-chunk workspace KB used when this prototype was built. `seed_chunk_ids` are
candidate-pool anchors, **not** relevance labels. The dataset is tied to that
KB's chunk IDs and will need regeneration after re-ingestion or re-chunking.
Those 12 queries were manually written. `generated_queries.jsonl` is a separate
automatically generated set of 12 questions spanning 8 papers from the same
KB snapshot; its source quotes, proposed answers and six rejected attempts are
recorded in `generated_queries.audit.jsonl`. To generate another query set
from a frozen KB, run:

```bash
python -m scripts.generate_kb_queries \
  --workspace /tmp/nanobot-kb-eval --count 12 --disable-thinking \
  --output eval/kb/generated_queries.jsonl
```

The generator selects substantive chunks across papers, asks the configured
LLM for a standalone question, and verifies that its proposed supporting
quote occurs in the seed chunk after whitespace normalization. It writes a separate
`generated_queries.audit.jsonl` with the proposed answer, quote, rejected
attempts, prompt version and corpus hash. It never overwrites existing output.
The generated queries are **not** relevance labels: run `pool`, `judge`, and
`evaluate` with distinct output filenames before interpreting Recall@K. The
existing `report.json` still scores the manually written `queries.jsonl`, not
the generated set.

Use a **frozen, writable copy** of the workspace's `kb/` directory. Copy it
while ingestion is stopped; do not run these commands on a live KB because
the KB constructor may update derived indexes. For example:

```bash
mkdir -p /tmp/nanobot-kb-eval
cp -a ~/.nanobot/workspace/kb /tmp/nanobot-kb-eval/kb
python -m scripts.eval_kb pool \
  --workspace /tmp/nanobot-kb-eval \
  --queries eval/kb/generated_queries.jsonl --depth 30 \
  --output eval/kb/pool.json
python -m scripts.eval_kb judge \
  --workspace /tmp/nanobot-kb-eval --pool eval/kb/pool.json \
  --output eval/kb/judgments.jsonl --disable-thinking
python -m scripts.eval_kb evaluate \
  --workspace /tmp/nanobot-kb-eval --pool eval/kb/pool.json \
  --judgments eval/kb/judgments.jsonl \
  --overrides eval/kb/reviews.jsonl --output eval/kb/report.json
```

The evaluation report keeps `summary.metrics` for the configured KB retrieval
pipeline. It also compares the three pooling retrievers in
`summary.source_methods`: `hybrid` (BM25 + dense fusion), `dense` (vector
retrieval without BM25 fusion), and `bm25` (standalone SQLite FTS5 keyword
search). Each has mean Recall, Hit and MRR at the requested K values over the
**same scored queries**; `queries[].source_methods` includes each query's
ranking and metrics. These comparison rankings come from `pool.json` and use
its depth, disabled relevance filter and per-paper limit, whereas
`summary.metrics` reruns the configured pipeline with its normal evaluation
settings. Compare the three `source_methods` with each other, not directly
with `summary.metrics`. The script rejects a pool whose depth is smaller than
the largest requested K or which lacks one of the three retrieval sources.
To produce only this comparison without loading the embedding/rerank models:

```bash
python -m scripts.eval_kb evaluate \
  --workspace /tmp/nanobot-kb-eval --pool eval/kb/pool.json \
  --judgments eval/kb/judgments.jsonl \
  --source-only --output eval/kb/report_methods.json
```

In `--source-only` mode, `summary.metrics` is empty because the configured
runtime pipeline was not rerun; `summary.source_methods` contains the three
comparable results.

Run these commands in the project's Python environment. `judge` uses the
configured OpenAI-compatible provider/model (or `--api-base`, `--model`, and
`--api-key-env`). `--disable-thinking` is for vLLM-served reasoning models
such as the local Qwen model; omit it if an endpoint does not support
`chat_template_kwargs`. It resumes from existing judgments. `--limit 5` performs a
small smoke test. `pool` merges hybrid, dense-only and SQLite BM25 results,
then adds seed chunks. It stores source ranks but no full passage text.
The checked-in pool was built with depth 30; a deeper pool may find more
positives but requires new labeling. If you change queries or rebuild the
pool, start a fresh judgments file rather than reusing old labels.

If `evaluate` reports `unjudged_predictions`, add those exact chunks to the
pool and label only the new pairs, then evaluate again:

```bash
python -m scripts.eval_kb supplement \
  --workspace /tmp/nanobot-kb-eval --pool eval/kb/pool.json \
  --report eval/kb/report.json --output eval/kb/pool.json
python -m scripts.eval_kb judge \
  --workspace /tmp/nanobot-kb-eval --pool eval/kb/pool.json \
  --output eval/kb/judgments.jsonl --disable-thinking
```

For each query-chunk pair, the judge records `0` (irrelevant), `1` (topically
related but not answer evidence), or `2` (direct answer evidence). Grade 2
requires a verbatim supporting quote. Missing/invalid labels and newly
retrieved unjudged chunks cause a query to be **skipped**, never treated as
negative. `evaluate` reports pooled Recall, Hit and MRR at 1/5/10; these are
not whole-corpus recall estimates.

LLM judgments are **silver labels**. Before using them as a release gate,
manually review all grade-2 and `needs_review` rows, and sample grade-0/1
rows to find false negatives. Put corrections in a separate `reviews.jsonl`
and pass `--overrides eval/kb/reviews.jsonl` to `evaluate`; rows with the same
`query_id` and `chunk_id` override the LLM label. Set `status` to `accepted`
only after verifying the quote against the frozen chunk text. Keep the pool,
judgments and report together with the frozen KB snapshot and record the
judge/prompt versions; do not compare scores across different snapshots.

The checked-in silver set has 498 judged query-chunk pairs and 33 grade-2
labels after the reviewed corrections. The example run scored all 12 queries:
pooled Recall@5 = 0.435 and Hit@5 = 0.75. These are smoke-test numbers, not
validated production-quality benchmarks. The original `report.json` predates
the per-source comparison; use `report_methods.json` for that comparison.

For single-turn evidence support, answer adequacy, and latency comparisons, see
[`eval/agent_hallucination/README.md`](../agent_hallucination/README.md).
For their first KB tool-call Recall/Hit/MRR without answer generation, see
[`eval/agent_retrieval/README.md`](../agent_retrieval/README.md).
