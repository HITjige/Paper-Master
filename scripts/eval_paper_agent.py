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
from nanobot.agent.tools.paper import PaperRerankTool, PaperSearchTool


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


async def _evaluate_item(search_tool: PaperSearchTool, rerank_tool: PaperRerankTool, item: dict) -> dict:
    query = item.get("query", "")
    expected = set(item.get("relevant_paper_ids", []))
    raw_search = await search_tool.execute(query=query, max_results=item.get("max_results", 20), top_k=20)
    search_payload = json.loads(raw_search)
    candidates = search_payload.get("results", [])
    reranked_raw = await rerank_tool.execute(query=query, papers=candidates, top_k=item.get("top_k", 10))
    reranked = json.loads(reranked_raw).get("results", [])
    predicted_ids = [str(x.get("paper_id", "")) for x in reranked if x.get("paper_id")]
    k = int(item.get("top_k", 10))
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
    kb = PaperKnowledgeBase(workspace, PaperKbConfig(enable=True))
    search_tool = PaperSearchTool(workspace=workspace, kb=kb)
    rerank_tool = PaperRerankTool(workspace=workspace, kb=kb)
    reports = []
    for item in items:
        reports.append(await _evaluate_item(search_tool, rerank_tool, item))

    if not reports:
        print("No evaluation items found.")
        return
    summary = {
        "count": len(reports),
        "avg_recall_at_k": round(mean(x["recall_at_k"] for x in reports), 6),
        "avg_mrr": round(mean(x["mrr"] for x in reports), 6),
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
