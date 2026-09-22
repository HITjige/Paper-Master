# 内部 KB 单轮检索评测

本原型评测「一个问题能否检索到可直接回答它的证据 chunk」，不是端到端答案质量评测。现有 `queries.jsonl` 是 12 条人工编写的问题；`generated_queries.jsonl` 是基于同一 KB 快照自动生成的另一组问题。两组数据不能共用候选池、标注或报告。

## 流程与脚本

1. **固定知识库快照**：停止入库后，将工作区的 `kb/` 复制到可写目录。评测脚本可能更新派生索引，不要直接对在线 KB 运行。
2. **准备 query**：使用 `queries.jsonl`，或运行 `python -m scripts.generate_kb_queries` 从 KB chunk 生成新问题。`seed_chunk_ids` 只是候选锚点，**不是相关性真值**；生成器的答案和引文见 `generated_queries.audit.jsonl`。
3. **召回候选**：运行 `python -m scripts.eval_kb pool`，合并 hybrid、dense、BM25 各自的 Top-K 与 seed chunk，保存各检索方式的排名。
4. **标注证据**：运行 `python -m scripts.eval_kb judge`，由 LLM 将每个 query–chunk 对标为 `0`（无关）、`1`（相关但不能回答）、`2`（直接证据）。人工复核后，可将修正写入 `reviews.jsonl`；只有等级 2 算正例。
5. **计算指标**：运行 `python -m scripts.eval_kb evaluate`，输出 Recall@K、Hit@K、MRR@K。`--source-only` 直接用候选池中的排名比较 hybrid、dense、BM25，无须加载检索模型；不加该参数还会重跑当前配置的检索流程。

以下命令在项目根目录、项目 Python 环境中执行。新建输出文件名，避免覆盖目录中已有的示例数据：

```bash
KB_EVAL_DIR="$(mktemp -d /tmp/nanobot-kb-eval.XXXXXX)"
cp -a ~/.nanobot/workspace/kb "$KB_EVAL_DIR/kb"

python -m scripts.eval_kb pool \
  --workspace "$KB_EVAL_DIR" --queries eval/kb/queries.jsonl \
  --depth 30 --output eval/kb/my_pool.json
python -m scripts.eval_kb judge \
  --workspace "$KB_EVAL_DIR" --pool eval/kb/my_pool.json \
  --disable-thinking --output eval/kb/my_judgments.jsonl
python -m scripts.eval_kb evaluate \
  --workspace "$KB_EVAL_DIR" --pool eval/kb/my_pool.json \
  --judgments eval/kb/my_judgments.jsonl --source-only \
  --output eval/kb/my_report_methods.json
```

如需同时评测当前配置的完整检索流程，去掉 `--source-only`，输出到另一个报告文件；若有人工修正，加上 `--overrides eval/kb/reviews.jsonl`。`--disable-thinking` 仅适用于支持该参数的本地推理服务；其他服务应去掉它。

自动生成新 query 的命令如下。生成器不会覆盖已有输出；将后续 `pool` 的 `--queries` 指向新文件，并使用全新的 pool、judgments 和 report 文件：

```bash
python -m scripts.generate_kb_queries \
  --workspace "$KB_EVAL_DIR" --count 12 --disable-thinking \
  --output eval/kb/my_generated_queries.jsonl
```

## 看报告时注意

`report_methods.json` 使用已有人工 query 和标注，三种检索方式的指标见 `summary.source_methods`，逐题排名见 `queries[].source_methods`。`report.json` 是旧版完整流程报告，不含三种方式的对比；完整流程的 `summary.metrics` 与候选池比较使用的检索设置不同，**不能直接横比**。两种报告的 Recall 都以已标注候选池中的正例为分母，并非全库真实召回率。

若完整评测出现 `unjudged_predictions`，先运行以下命令将新命中 chunk 加入 pool，再重复上面的 `judge` 命令增量补标，最后重跑 `evaluate`：

```bash
python -m scripts.eval_kb supplement \
  --workspace "$KB_EVAL_DIR" --pool eval/kb/my_pool.json \
  --report eval/kb/my_report.json --output eval/kb/my_pool.json
```

未标注或待复核样本不会被当成负例。LLM 标签属于银标；用作正式基准前，应复核等级 2 和 `needs_review`，并抽查等级 0/1 的漏标。

普通 agent 与 multi-agent 的单轮答案幻觉率评测见 [单轮事实性评测](../agent_hallucination/README.md)。
只评测两种 agent 首次 KB 检索的 Recall/Hit/MRR，见 [首次 KB 检索评测](../agent_retrieval/README.md)。
