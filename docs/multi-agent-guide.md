# Multi-Agent System Guide

## Overview

The multi-agent system provides an intelligent workflow for paper retrieval and analysis using 5 specialized agents:

1. **Router Agent** - Analyzes user intent and decides execution path
2. **Retrieval Agent** - Queries internal knowledge base with quality assessment
3. **Research Agent** - Searches and ingests external papers from arXiv
4. **Synthesis Agent** - Generates comprehensive answers from sources
5. **Critic Agent** - Reviews answer quality and detects hallucinations

## Architecture

```
┌─────────────┐
│    User     │
└──────┬──────┘
       ▼
┌─────────────┐
│   Router    │ ◄── Decides: internal / external / hybrid / direct
└──────┬──────┘
       │
   ┌───┼───┐
   ▼   ▼   ▼
┌────┐┌────┐┌────┐
│Int ││Ext ││Dir │
│KB  ││Src ││Ans │
└──┬─┘└─┬──┘└────┘
   │    │
   ▼    ▼
┌─────────────┐
│  Synthesis  │ ◄── Combines all sources
└──────┬──────┘
       ▼
┌─────────────┐
│   Critic    │ ◄── Quality check
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌────┐  ┌────┐
│Pass│  │Redo│
└──┬─┘  └────┘
   ▼
┌─────────────┐
│   Output    │
└─────────────┘
```

## Routing Logic

The Router Agent analyzes queries and routes them to appropriate paths:

| Pattern | Route | Example |
|---------|-------|---------|
| "知识库", "已有", "stored" | **internal** | "总结一下知识库里的Transformer论文" |
| "最新", "2026", "recent", "latest" | **external** | "找几篇2026年最新的Mamba论文" |
| "对比", "比较", "vs", "compare" | **hybrid** | "对比知识库里的A论文和最新B论文" |
| Basic questions, greetings | **direct** | "什么是Transformer?" |

## Usage

### Automatic Detection

The system automatically uses multi-agent mode for paper-related queries:

```
User: "帮我找几篇关于大模型推理优化的最新论文"
→ Automatically triggers multi-agent workflow
```

### Manual Trigger

Use the `/multi-agent` command:

```
/multi-agent 总结一下Transformer架构的发展历程
```

### Check Status

```
/multi-agent
→ Shows multi-agent system status
```

## Workflow Steps

### 1. Router Decision

The Router analyzes your query and decides:
- **internal**: Query existing knowledge base
- **external**: Search arXiv for new papers
- **hybrid**: Combine both approaches
- **direct**: Answer without retrieval

### 2. Retrieval (if internal/hybrid)

- Searches the summary/question/original-chunk indexes using dense + lexical retrieval
- Evaluates result quality
- If insufficient, triggers external search

### 3. Research (if external/hybrid)

- Generates multiple search queries
- Searches arXiv
- Reranks results using semantic similarity, recency, and source priors
- Returns abstract-level evidence and lets the user select papers
- Downloads, parses, and ingests only the selected papers

### 4. Synthesis

- Combines internal and external sources
- Generates structured answer
- Adds citations

### 5. Critic Review

- Checks completeness
- Detects hallucinations
- Verifies citations
- Routes back if issues found

## Configuration

Add to `~/.nanobot/config.json` (the MinerU token is optional):

```json
{
  "tools": {
    "paper": {
      "enable": true,
      "multiAgentOrchestratorEnabled": true,
      "embeddingModel": "text-embedding-3-small",
      "embeddingApiKey": "${OPENAI_API_KEY}",
      "embeddingFallback": "error",
      "embeddingBatchSize": 64,
      "rrfK": 60,
      "denseRrfWeight": 0.5,
      "sparseRrfWeight": 0.5,
      "bm25TitleWeight": 5.0,
      "bm25KeywordsWeight": 3.0,
      "bm25SummaryWeight": 1.5,
      "bm25QuestionsWeight": 2.0,
      "bm25BodyWeight": 1.0,
      "mineruApiToken": "${MINERU_API_TOKEN}",
      "retrievalTopK": 5,
      "autoContextRetrieve": true,
      "autoContextTopK": 5
    }
  }
}
```

If MinerU is not used, omit `mineruApiToken`. If it is used, export
`MINERU_API_TOKEN` before starting nanobot; unresolved `${...}` variables are
treated as configuration errors. The Paper configuration accepts camelCase
(recommended), snake_case, and `MINERU_API_TOKEN` as a compatibility alias for
the token key. In all cases, prefer keeping the secret in the environment and
using `"mineruApiToken": "${MINERU_API_TOKEN}"` in the JSON file.

For production retrieval, configure either an embedding API key or a local
SentenceTransformer directory. `embeddingFallback: "error"` prevents startup
configuration mistakes from silently becoming lexical-only retrieval. The
default `"hash"` fallback remains available for offline development and is
reported as `hash_lexical` with `degraded: true` by KB APIs. Embedding requests
are sent in batches controlled by `embeddingBatchSize`.

Sparse retrieval is stored in `kb/lexical.db` using SQLite FTS5. Each parent
chunk is one weighted document with title, keywords, summary, hypothetical
questions, and body fields. `denseRrfWeight` and `sparseRrfWeight` are family
weights: each family's weight is divided among its query rewrites and views,
so adding rewrites does not give that family extra RRF votes. The index is
automatically rebuilt from canonical JSONL when its schema/tokenizer version
or source file modification marker changes.

## Requirements

```bash
pip install -e ".[paper]"

# Ensure paper tools are enabled
# See paper-tools-guide.md
```

## API

### Direct Usage

```python
from nanobot.agent.multi_agent import build_multi_agent_graph

# Build graph
graph = build_multi_agent_graph(
    provider=llm_provider,
    kb=paper_kb,
    tools={
        "paper_search": search_tool,
        "paper_ingest": ingest_tool,
        "kb_retrieve": retrieve_tool,
    },
)

# Run workflow
result = await graph.run("What are the latest Mamba papers?")
print(result["final_answer"])
```

### Via AgentLoop

```python
from nanobot.agent.loop import AgentLoop

loop = AgentLoop(...)

# Check if query should use multi-agent
if loop.should_use_multi_agent(query):
    result = await loop.process_with_multi_agent(query)
```

## State Tracking

The system tracks:

- `routing_decision`: Which path was chosen
- `retrieval_quality`: Whether internal results were sufficient
- `iteration_count`: Number of critic review cycles
- `critic_verdict`: Final quality assessment
- `sources_used`: Which sources contributed

## Debugging

Enable debug logging:

```bash
export NANOBOT_LOG_LEVEL=debug
```

View workflow trace:

```
[Multi-Agent Workflow: hybrid mode, 2 iteration(s)]

Your answer here...

References:
1. arxiv:2405.04517
2. arxiv:2312.00752
```

## Limitations

- Requires LangGraph installation
- Paper tools must be enabled
- External search limited to arXiv
- Max 3 iterations for critic review
- Requires LLM provider for agent reasoning

## Future Enhancements

- [ ] Support for more external sources (PubMed, Semantic Scholar)
- [ ] Parallel agent execution
- [ ] User feedback integration
- [ ] Custom agent prompts
- [ ] Workflow visualization
