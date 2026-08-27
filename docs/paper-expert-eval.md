# Paper Expert Evaluation

This document describes the baseline evaluation loop for the paper expert agent.

## Dataset

Use `eval/paper_eval_dataset.json` with records:

- `query`: question or retrieval intent
- `relevant_paper_ids`: expected paper ids (from arXiv id or your canonical id)
- `top_k`: cutoff for recall@k
- `search_topk`: candidate pool size used by `paper_search` (`max_results` remains a legacy alias)

## Run

From repo root:

```bash
python scripts/eval_paper_agent.py --dataset eval/paper_eval_dataset.json
```

Optional:

```bash
python scripts/eval_paper_agent.py --dataset eval/paper_eval_dataset.json --workspace ~/.nanobot/workspace
```

## Metrics

- `avg_recall_at_k`: relevant hits / total relevant papers.
- `avg_mrr`: mean reciprocal rank of first relevant paper.

Records without a non-empty `relevant_paper_ids` label are reported as
`skipped` and are excluded from aggregate metrics; they are not counted as
zero-recall examples.

## Observability

During runtime, the following tool logs are emitted:

- `paper_search`: query, source, candidate count.
- `paper_similarity`: query and candidate count.
- `paper_rerank`: candidate count and top_k.
- `paper_ingest`: paper id, chunk count, source URL.
- `kb_retrieve`: retrieval hits and top_k.

Use these logs to inspect:

- retrieval miss patterns
- rerank threshold quality
- ingestion failures and parser issues
