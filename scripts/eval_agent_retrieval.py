"""Evaluate only the first internal-KB retrieval of single and multi-agent runs.

The ordinary agent stops immediately after its first kb_retrieve tool result.
The paper multi-agent runs its router and first retrieval node, then stops.
No answer is saved or judged. See eval/agent_hallucination/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.multi_agent.conditions import router_conditional
from nanobot.agent.multi_agent.state import create_initial_state
from nanobot.agent.paper_evidence import build_evidence_bundle, flatten_evidence_results
from nanobot.utils.helpers import sync_workspace_templates
from scripts.eval_agent_hallucination import (
    EvidenceHook,
    _evaluation_config,
    _hash,
    _make_loop,
    runtime_manifest,
)
from scripts.eval_kb import file_sha256, load_chunks, metrics_at_k, read_jsonl, same_corpus

SCHEMA_VERSION = 1
MODES = ("single", "multi")
SCOREABLE_RUN_STATUSES = {"retrieved", "no_retrieval"}


class FirstRetrievalCapturedError(Exception):
    """Stop the ordinary agent after a completed KB tool call."""


class FirstRetrievalHook(EvidenceHook):
    def __init__(self) -> None:
        super().__init__()
        self.payload: dict[str, Any] | None = None
        self.raw_result = ""
        self.arguments: dict[str, Any] = {}

    def reset(self) -> None:
        self.payload = None
        self.raw_result = ""
        self.arguments = {}
        self.evidence.clear()
        self.tool_calls.clear()
        self.stop_reason = ""
        self._seen.clear()

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        # A model may request several KB tools in one response. The runner
        # normally executes all of them before after_iteration; keep only the
        # first so this evaluation actually performs one logical retrieval.
        if context.response is None:
            return
        first = next((call for call in context.response.tool_calls
                      if call.name == "kb_retrieve"), None)
        if first is not None:
            context.response.tool_calls = [first]
            context.tool_calls = [first]

    async def after_iteration(self, context: AgentHookContext) -> None:
        await super().after_iteration(context)
        calls = {call.id: call for call in context.tool_calls}
        for message in context.messages:
            if message.get("role") != "tool" or message.get("name") != "kb_retrieve":
                continue
            self.raw_result = str(message.get("content") or "")
            call = calls.get(str(message.get("tool_call_id") or ""))
            self.arguments = dict(call.arguments) if call else {}
            try:
                value = json.loads(self.raw_result)
                self.payload = value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                self.payload = None
            raise FirstRetrievalCapturedError


def _unique_chunk_ids(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(
        str(value.get("chunk_id")) for value in values
        if isinstance(value, dict) and value.get("chunk_id")
    ))


def _single_prediction(hook: FirstRetrievalHook) -> dict[str, Any]:
    payload = hook.payload
    if payload is None or payload.get("error"):
        return {"status": "retrieval_error", "predicted": [],
                "error": str(payload.get("error") if payload else hook.raw_result)[:500]}
    papers = payload.get("papers")
    if not isinstance(papers, list):
        return {"status": "retrieval_error", "predicted": [],
                "error": "kb_retrieve returned no paper list"}
    flattened = [chunk for paper in papers if isinstance(paper, dict)
                 for chunk in (paper.get("chunks") or [])]
    predicted = _unique_chunk_ids(flattened)
    if flattened and len(predicted) != len(flattened):
        return {"status": "retrieval_error", "predicted": [],
                "error": "kb_retrieve returned a chunk without a unique chunk_id"}
    return {
        "status": "retrieved", "predicted": predicted,
        "tool_arguments": hook.arguments,
        "retrieval_mode": payload.get("retrieval_mode"),
        "quality": payload.get("quality"),
        "total_hits_before_gate": payload.get("total_hits"),
        "returned_hits": payload.get("returned_hits"),
    }


async def _first_single(loop, query: str, run_id: str,
                        hook: FirstRetrievalHook) -> dict[str, Any]:
    try:
        # A direct answer without a tool call is intentionally discarded.
        await loop.process_direct(query, session_key=run_id, chat_id=run_id)
    except FirstRetrievalCapturedError:
        return _single_prediction(hook)
    return {"status": "no_retrieval", "predicted": []}


async def _first_multi(loop, query: str, run_id: str) -> dict[str, Any]:
    graph = loop._multi_agent_graph
    snapshot = loop._build_multi_agent_context_snapshot(
        content=query, history=[], channel="cli", chat_id=run_id,
        session_summary=None, media=None,
    )
    state = create_initial_state(query, session_id=run_id, config=graph.agent_config,
                                 **snapshot)
    state.update(snapshot)
    state.update({
        "last_routing_decision": "none", "routing_context": {"active_papers": []},
        "referenced_papers": [], "presented_paper_ids": [], "last_search_topic": "",
        "orchestrator_decision": "multi_agent", "orchestrator_reasoning": "",
        "orchestrator_confidence": 0.0,
        "retrieval_judge_margin": loop.tools_config.paper.multi_agent_retrieval_judge_margin,
    })
    state = await graph.nodes.router_node(state)
    route = router_conditional(state)
    if route not in {"retrieval", "hybrid_entry"}:
        return {"status": "no_retrieval", "predicted": [],
                "routing_decision": state.get("routing_decision"), "router_next": route}
    if route == "hybrid_entry":
        state = await graph.nodes.hybrid_entry_node(state)
    state = await graph.nodes.retrieval_node(state)
    if state.get("requires_clarification"):
        return {"status": "no_retrieval", "predicted": [],
                "routing_decision": state.get("routing_decision"),
                "reason": "requires_clarification"}
    if state.get("error_message"):
        return {"status": "retrieval_error", "predicted": [],
                "routing_decision": state.get("routing_decision"),
                "error": str(state["error_message"])[:500]}
    results = state.get("retrieval_results") or []
    # The synthesis renderer groups by paper and source order. Use that
    # deterministic ranking for comparable MRR@K, without running synthesis.
    bundle = build_evidence_bundle(results, query=query)
    predicted = _unique_chunk_ids(flatten_evidence_results(bundle))
    if results and len(predicted) != len(results):
        return {"status": "retrieval_error", "predicted": [],
                "routing_decision": state.get("routing_decision"),
                "error": "Retrieval node returned a chunk without a unique chunk_id"}
    return {
        "status": "retrieved",
        "predicted": predicted,
        "routing_decision": state.get("routing_decision"),
        "quality": state.get("retrieval_quality"),
        "rewritten_queries": state.get("rewritten_queries", []),
        "rewrite_fallback_used": state.get("rewrite_fallback_used", False),
        "raw_retrieved_count": len(results),
    }


async def _run_on_loop(loop, hook: FirstRetrievalHook, item: dict[str, Any],
                       mode: str, repeat: int, timeout_s: float) -> dict[str, Any]:
    run_id = f"first:{item['id']}:{repeat}:{mode}"
    hook.reset()
    previous_calls = getattr(loop, "_eval_model_calls", [0])[0]
    previous_usage = dict(getattr(loop, "_eval_usage", {}))
    started = time.monotonic()
    try:
        if mode == "single":
            result = await asyncio.wait_for(
                _first_single(loop, item["query"], run_id, hook), timeout_s,
            )
        else:
            result = await asyncio.wait_for(
                _first_multi(loop, item["query"], run_id), timeout_s,
            )
    except TimeoutError:
        result = {"status": "timeout", "predicted": []}
    except Exception as exc:
        result = {"status": "error", "predicted": [],
                  "error": f"{type(exc).__name__}: {exc}"[:500]}
    current_usage = dict(getattr(loop, "_eval_usage", {}))
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "query_id": item["id"], "query": item["query"],
        "tags": item.get("tags", []), "mode": mode, "repeat": repeat,
        "elapsed_s": round(time.monotonic() - started, 3),
        "model_calls": getattr(loop, "_eval_model_calls", [0])[0] - previous_calls,
        "usage": {key: current_usage.get(key, 0) - previous_usage.get(key, 0)
                  for key in current_usage},
        "embedding": loop.kb.get_embedding_status(),
        "lexical": loop.kb.get_lexical_status(),
        **result,
    }


def _queries_manifest(queries: list[dict[str, Any]]) -> str:
    return _hash([{"id": item["id"], "query": item["query"]} for item in queries])


async def run_command(args: argparse.Namespace) -> None:
    workspace = args.workspace.expanduser().resolve()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    same_corpus(pool, workspace)
    all_queries = list(pool["queries"])
    query_hash = _queries_manifest(all_queries)
    queries = all_queries
    if args.limit:
        queries = queries[:args.limit]
    config = _evaluation_config(args.config)
    runtime = {**runtime_manifest(),
               "scripts/eval_agent_retrieval.py": file_sha256(Path(__file__))}
    config_hash = _hash({"config": config.model_dump(), "runtime": runtime,
                         "timeout_s": args.timeout})
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        if (row.get("schema_version") != SCHEMA_VERSION or
            row.get("corpus") != pool["corpus"] or
            row.get("config_sha256") != config_hash or
            row.get("query_manifest_sha256") != query_hash):
            raise ValueError("Existing first-retrieval runs use another KB or runtime config")
        run_id = row.get("run_id")
        if not run_id or run_id in existing:
            raise ValueError(f"Missing or duplicate run_id: {run_id}")
        existing[run_id] = row
    pending = []
    for item in queries:
        for repeat in range(args.repeats):
            modes = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
            for mode in modes:
                run_id = f"first:{item['id']}:{repeat}:{mode}"
                previous = existing.get(run_id)
                if previous:
                    if previous.get("query") != item["query"]:
                        raise ValueError(f"Query changed for existing run {run_id}")
                else:
                    pending.append((item, repeat, mode))
    if not pending:
        print("All requested first-retrieval runs already exist")
        return
    # Reuse one private KB copy and provider for the whole batch. Each ordinary
    # run has a unique session key; neither path writes an answer or memory.
    with tempfile.TemporaryDirectory(prefix="nanobot-retrieval-eval-") as temporary:
        run_workspace = Path(temporary)
        shutil.copytree(workspace / "kb", run_workspace / "kb")
        sync_workspace_templates(run_workspace, silent=True)
        hook = FirstRetrievalHook()
        loop = _make_loop(run_workspace, config, hook)
        embedding = loop.kb.get_embedding_status()
        lexical = loop.kb.get_lexical_status()
        if (embedding.get("degraded") or lexical.get("degraded")) and not args.allow_degraded:
            raise RuntimeError("KB retrieval backend is degraded; pass --allow-degraded intentionally")
        try:
            for item, repeat, mode in pending:
                run_id = f"first:{item['id']}:{repeat}:{mode}"
                row = await _run_on_loop(loop, hook, item, mode, repeat, args.timeout)
                row.update({"corpus": pool["corpus"], "config_sha256": config_hash,
                            "query_manifest_sha256": query_hash, "runtime": runtime})
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                print(f"{run_id}: {row['status']} hits={len(row['predicted'])} "
                      f"time={row['elapsed_s']}s")
        finally:
            await loop.close_mcp()


def _labels(pool: dict[str, Any], judgments: list[dict[str, Any]],
            overrides: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidate_keys = {(row["query_id"], row["chunk_id"]) for row in pool["candidates"]}
    if len(candidate_keys) != len(pool["candidates"]):
        raise ValueError("Duplicate query/chunk candidate in pool")
    decisions = {(row["query_id"], row["chunk_id"]): row for row in judgments}
    if len(decisions) != len(judgments):
        raise ValueError("Duplicate query/chunk judgments")
    for row in overrides:
        if (row["query_id"], row["chunk_id"]) not in candidate_keys:
            raise ValueError("Override refers to a pair outside the pool")
        decisions[(row["query_id"], row["chunk_id"])] = row
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in pool["candidates"]:
        qid, cid = candidate["query_id"], candidate["chunk_id"]
        decision = decisions.get((qid, cid))
        if (decision is None or decision.get("status") != "accepted" or
            (decision.get("chunk_sha256") and
             decision["chunk_sha256"] != candidate.get("chunk_sha256"))):
            by_query[qid].append({"chunk_id": cid, "status": "missing_or_needs_review"})
        elif type(decision.get("grade")) is not int or decision["grade"] not in (0, 1, 2):
            by_query[qid].append({"chunk_id": cid, "status": "invalid_grade"})
        else:
            by_query[qid].append({"chunk_id": cid, "grade": decision["grade"],
                                  "status": "accepted"})
    return {qid: {"candidate_ids": {row["chunk_id"] for row in rows},
                  "unresolved": [row["chunk_id"] for row in rows if row["status"] != "accepted"],
                  "relevant": {row["chunk_id"] for row in rows if row.get("grade") == 2}}
            for qid, rows in by_query.items()}


def _summarize(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode]
        scored = [row for row in subset if row["status"] == "scored"]
        conditional = [row for row in scored if row["run_status"] == "retrieved"]
        summary[mode] = {
            "runs": len(subset), "scored_runs": len(scored),
            "status_counts": dict(Counter(row["status"] for row in subset)),
            "retrieval_attempt_rate": (sum(row["run_status"] in {"retrieved", "retrieval_error"}
                                           for row in subset)
                                    / len(subset) if subset else None),
            "metrics": {
                str(k): {metric: mean(row["metrics"][str(k)][metric] for row in scored)
                         for metric in ("recall", "hit", "mrr")}
                for k in ks
            } if scored else {},
            "conditional_on_retrieval": {
                str(k): {metric: mean(row["metrics"][str(k)][metric] for row in conditional)
                         for metric in ("recall", "hit", "mrr")}
                for k in ks
            } if conditional else {},
            "mean_elapsed_s": mean(row["elapsed_s"] for row in subset) if subset else None,
        }
    paired = defaultdict(dict)
    for row in rows:
        if row["status"] == "scored":
            paired[(row["query_id"], row["repeat"])][row["mode"]] = row
    pairs = [group for group in paired.values() if set(group) == set(MODES)]
    summary["paired"] = {
        "count": len(pairs),
        "metrics": {
            str(k): {metric: mean(pair["multi"]["metrics"][str(k)][metric]
                                  - pair["single"]["metrics"][str(k)][metric]
                                  for pair in pairs)
                     for metric in ("recall", "hit", "mrr")}
            for k in ks
        } if pairs else {},
    }
    return summary


def build_report(pool: dict[str, Any], runs: list[dict[str, Any]],
                 judgments: list[dict[str, Any]], overrides: list[dict[str, Any]],
                 ks: list[int]) -> dict[str, Any]:
    if not runs:
        raise ValueError("No first-retrieval runs")
    if len({row["run_id"] for row in runs}) != len(runs):
        raise ValueError("Duplicate run_id")
    if len({row.get("config_sha256") for row in runs}) != 1:
        raise ValueError("Runs use different agent configurations")
    if any(row.get("corpus") != pool["corpus"] for row in runs):
        raise ValueError("Runs and pool use different KB snapshots")
    if any(row.get("mode") not in MODES for row in runs):
        raise ValueError("Unknown agent mode in runs")
    queries = {item["id"]: item for item in pool["queries"]}
    labels = _labels(pool, judgments, overrides)
    reports = []
    unjudged_by_query: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        qid = run["query_id"]
        if qid not in queries or run.get("query") != queries[qid]["query"]:
            raise ValueError(f"Unknown or changed query: {qid}")
        prediction = list(run.get("predicted") or [])
        item = {"run_id": run["run_id"], "query_id": qid,
                "mode": run["mode"], "repeat": run["repeat"],
                "run_status": run["status"], "predicted": prediction,
                "elapsed_s": run.get("elapsed_s", 0),
                "tags": queries[qid].get("tags", [])}
        gold = labels.get(qid, {"candidate_ids": set(), "unresolved": [], "relevant": set()})
        unjudged = sorted(set(prediction) - gold["candidate_ids"])
        if run["status"] in SCOREABLE_RUN_STATUSES and unjudged:
            unjudged_by_query[qid].update(unjudged)
        if run["status"] not in SCOREABLE_RUN_STATUSES:
            item["status"] = run["status"]
        elif gold["unresolved"]:
            item.update({"status": "incomplete_labels", "unresolved": gold["unresolved"],
                         "unjudged": unjudged})
        elif not gold["relevant"]:
            item.update({"status": "no_positive_labels", "unjudged": unjudged})
        else:
            if unjudged:
                item.update({"status": "unjudged_predictions", "unjudged": unjudged})
            else:
                item.update({"status": "scored", "relevant": sorted(gold["relevant"]),
                             "metrics": {str(k): metrics_at_k(prediction, gold["relevant"], k)
                                         for k in ks}})
        reports.append(item)
    return {
        "schema_version": SCHEMA_VERSION, "metric_scope": "judged_pool",
        "ranking_scope": "first_kb_retrieval_result",
        "corpus": pool["corpus"], "config_sha256": runs[0].get("config_sha256"),
        "ks": ks, "summary": _summarize(reports, ks),
        "runs": reports,
        # Compatible with `python -m scripts.eval_kb supplement --report ...`.
        "queries": [{"id": qid, "unjudged": sorted(ids)}
                    for qid, ids in sorted(unjudged_by_query.items())],
    }


def report_command(args: argparse.Namespace) -> None:
    workspace = args.workspace.expanduser().resolve()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    same_corpus(pool, workspace)
    chunks = load_chunks(workspace)  # Validate unique IDs before interpreting labels.
    runs = read_jsonl(args.runs)
    missing_chunk_ids = {cid for row in runs for cid in (row.get("predicted") or [])
                         if cid not in chunks}
    if missing_chunk_ids:
        raise ValueError(f"Predicted chunk IDs are absent from the frozen KB: "
                         f"{sorted(missing_chunk_ids)[:5]}")
    judgments = read_jsonl(args.judgments)
    overrides = read_jsonl(args.overrides) if args.overrides else []
    ks = sorted({int(value) for value in args.ks.split(",")})
    if not ks or ks[0] < 1:
        raise ValueError("--ks must contain positive integers")
    if type(pool.get("depth_per_source")) is int and pool["depth_per_source"] < max(ks):
        raise ValueError("Pool depth must cover the largest requested K")
    report = build_report(pool, runs, judgments, overrides, ks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report"):
        command = sub.add_parser(name)
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--pool", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    run = sub.choices["run"]
    run.add_argument("--config", type=Path)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--allow-degraded", action="store_true")
    report = sub.choices["report"]
    report.add_argument("--judgments", type=Path, required=True)
    report.add_argument("--runs", type=Path, required=True)
    report.add_argument("--overrides", type=Path)
    report.add_argument("--ks", default="1,5,10")
    args = parser.parse_args()
    if args.command == "run":
        if args.repeats < 1 or args.limit < 0 or args.timeout <= 0:
            parser.error("--repeats and --timeout must be positive; --limit nonnegative")
        asyncio.run(run_command(args))
    else:
        report_command(args)


if __name__ == "__main__":
    main()
