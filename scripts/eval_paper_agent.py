"""Offline evaluator for paper expert retrieval/reranking quality.

Usage:
  python scripts/eval_paper_agent.py --dataset eval/paper_eval_dataset.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean

from nanobot.agent.paper_kb import PaperKbConfig, PaperKnowledgeBase
from nanobot.agent.tools.paper import PaperSearchTool


def _recall_at_k(predicted: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    hits = len(set(predicted[:k]) & expected)
    return hits / len(expected)


def _mrr(predicted: list[str], expected: set[str]) -> float:
    for i, pid in enumerate(predicted, start=1):
        if pid in expected:
            return 1.0 / i
    return 0.0


async def _evaluate_item(search_tool: PaperSearchTool, item: dict) -> dict:
    query = item.get("query", "")
    expected = set(item.get("relevant_paper_ids", []))
    if not expected:
        return {
            "query": query,
            "skipped": True,
            "reason": "missing_relevance_labels",
        }
    k = int(item.get("top_k", 10))
    raw_search = await search_tool.execute(
        query=query,
        search_topk=int(item.get("search_topk", item.get("max_results", 60))),
        recall_top_k=max(20, k),
        rerank_top_k=k,
    )
    search_payload = json.loads(raw_search)
    ranked = search_payload.get("results", [])
    predicted_ids = [str(x.get("paper_id", "")) for x in ranked if x.get("paper_id")]
    return {
        "query": query,
        "recall_at_k": _recall_at_k(predicted_ids, expected, k),
        "mrr": _mrr(predicted_ids, expected),
        "predicted": predicted_ids[:k],
        "expected": sorted(expected),
    }


async def main(dataset_path: Path, workspace: Path) -> None:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("items", [])
    kb = PaperKnowledgeBase(workspace, PaperKbConfig(enabled=True))
    search_tool = PaperSearchTool(workspace=workspace, kb=kb)
    reports = []
    for item in items:
        reports.append(await _evaluate_item(search_tool, item))

    labeled = [report for report in reports if not report.get("skipped")]
    if not labeled:
        print(json.dumps({
            "summary": {
                "count": len(reports),
                "labeled_count": 0,
                "skipped_count": len(reports),
                "error": "No labeled evaluation items found.",
            },
            "reports": reports,
        }, ensure_ascii=False, indent=2))
        return
    summary = {
        "count": len(reports),
        "labeled_count": len(labeled),
        "skipped_count": len(reports) - len(labeled),
        "avg_recall_at_k": round(mean(x["recall_at_k"] for x in labeled), 6),
        "avg_mrr": round(mean(x["mrr"] for x in labeled), 6),
    }
    print(json.dumps({"summary": summary, "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path, help="Path to eval dataset json")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("~/.nanobot/workspace").expanduser(),
        help="Workspace root used for KB files",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.workspace))
