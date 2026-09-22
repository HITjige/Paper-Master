# 普通 agent 与 multi-agent 的首次 KB 检索评测

本评测回答“单轮问题交给 agent 后，**第一次内部 KB 检索**能否找到可直接回答的 chunk”。它复用 `eval/kb/pool.json` 的 query–chunk 标签：等级 2 是正例；等级 0/1 不是正例。普通 agent 在首个 `kb_retrieve` 工具结果返回后立即停止；multi-agent 只运行 router 与首次 retrieval 节点。两边都不执行后续生成答案或后续检索，`first_retrieval_runs.jsonl` 也不保存答案。

这里的“一次”指普通 agent 的一次 `kb_retrieve` 工具调用，或 multi-agent 的一次 retrieval 节点。工具/节点内部的多 query 检索、重写回退和质量判断属于这次逻辑检索。指标使用**首次检索结果的 chunk 顺序**：普通 agent 取工具返回的 `papers[].chunks`，multi-agent 按 synthesis 证据格式的论文/原文顺序重建排名，但不运行 synthesis 或模拟它的上下文截断。质量门控清空结果时记为空排名；没有调用 KB 时也记为空排名，并单独报告 `retrieval_attempt_rate`。

使用停止入库后的可写 KB 副本，并让 pool、judgments 对应同一 KB 快照：

```bash
python -m scripts.eval_agent_retrieval run \
  --workspace /tmp/nanobot-kb-eval \
  --pool eval/kb/pool.json \
  --limit 32 --repeats 1 \
  --output eval/agent_retrieval/first_retrieval_runs.jsonl

python -m scripts.eval_agent_retrieval report \
  --workspace /tmp/nanobot-kb-eval \
  --pool eval/kb/pool.json \
  --runs eval/agent_retrieval/first_retrieval_runs.jsonl \
  --judgments eval/kb/judgments.jsonl \
  --output eval/agent_retrieval/first_retrieval_report.json
```

检查前 5 题后，去掉 `--limit` 运行完整 pool；`run` 会跳过已有的同配置结果。若有人工修正，在 `report` 命令中加 `--overrides path/to/reviews.jsonl`。`--repeats` 默认为 1，以减少模型调用。普通 agent 和 multi-agent 使用相同模型配置、冻结 KB 与全新 session；外部搜索和其他工具被禁用。multi-agent 若路由到非 KB 路径，本轮标为 `no_retrieval`，不会进入 research 或 synthesis。普通 agent 若直接回答，也只记 `no_retrieval`，其答案被丢弃。

报告的 `summary.single` 和 `summary.multi` 给出 Recall/Hit/MRR@1/5/10；`summary.paired` 是同题同次运行的 multi 减普通 agent 的均值差。主指标把没有调用检索的已标注题目计为零；`conditional_on_retrieval` 只统计实际调用且成功返回的轮次。两个指标都以**已标注候选池内的正例**为分母，不能当作全库召回率。`incomplete_labels`、`no_positive_labels`、`unjudged_predictions`、错误和超时单独计数，不混入均值。
若运行配置只返回 5 条 chunk，@10 可能与 @5 相同；报告保留实际返回数量供解释。

若报告中有 `queries[].unjudged`，可用现有工具增量补标，再用更新后的 pool/judgments 重跑 **report**，无需重新调用 agent：

```bash
python -m scripts.eval_kb supplement \
  --workspace /tmp/nanobot-kb-eval \
  --pool eval/kb/pool.json \
  --report eval/agent_retrieval/first_retrieval_report.json \
  --output eval/agent_retrieval/augmented_pool.json

cp eval/kb/judgments.jsonl eval/agent_retrieval/augmented_judgments.jsonl
python -m scripts.eval_kb judge \
  --workspace /tmp/nanobot-kb-eval \
  --pool eval/agent_retrieval/augmented_pool.json \
  --output eval/agent_retrieval/augmented_judgments.jsonl \
  --disable-thinking
```

然后将 `report` 命令中的 `--pool`、`--judgments` 改为对应的 `augmented_*` 文件。`needs_review` 的旧标签仍需人工复核并以 `--overrides` 传入；新命中的 chunk 不会自动视为负例。`--disable-thinking` 仅适用于支持该参数的本地模型服务。
