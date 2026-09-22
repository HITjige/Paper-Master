# 普通 agent 与 multi-agent 的单轮证据支撑评测

若只想评测首次调用 KB 的检索质量、无需生成答案，见[首次 KB 检索评测](../agent_retrieval/README.md)。

本评测使用一个停止入库后的、**可写的 KB 副本**。`run` 为每个 query、重复次数和模式再建立临时工作区，使用全新 session；不会修改源工作区。两种模式使用同一个模型配置。普通 agent 显式关闭自动 multi-agent 路由；multi-agent 必须成功进入 graph，否则该 run 报错。外部搜索、入库、Web、shell 和 MCP 均被禁用。暂停等待外部搜索确认的结果记为 `paused`，不追加第二轮消息。

在项目 Python 环境中，从项目根目录运行：

```bash
python -m scripts.eval_agent_hallucination run \
  --workspace /tmp/nanobot-kb-eval \
  --queries eval/kb/generated_queries.jsonl --protocol workflow \
  --repeats 1 --limit 32 \
  --output eval/agent_hallucination/workflow_runs.jsonl

python -m scripts.eval_agent_hallucination judge \
  --workspace /tmp/nanobot-kb-eval \
  --runs eval/agent_hallucination/workflow_runs.jsonl \
  --output eval/agent_hallucination/workflow_grounding_claims.jsonl \
  --disable-thinking

python -m scripts.eval_agent_hallucination report \
  --workspace /tmp/nanobot-kb-eval \
  --runs eval/agent_hallucination/workflow_runs.jsonl \
  --judgments eval/agent_hallucination/workflow_grounding_claims.jsonl \
  --output eval/agent_hallucination/workflow_grounding_report.json
```

`judge` 的 `--disable-thinking` 只适用于支持 `chat_template_kwargs.enable_thinking` 的服务；其他服务去掉该参数。也可指定 `--model`、`--api-base` 和 `--api-key-env`。建议判定模型与被评测模型不同。三个阶段均校验 `documents.jsonl`、`chunks.jsonl` 的 SHA-256。`run` 和 `judge` 支持增量续跑；更换 KB、模型配置、题目、固定证据或运行时代码时应使用新的 runs 文件，更换判定模型或判定协议时应使用新的 claims 文件。旧版 `workflow_claims.jsonl` 使用 KB 全文事实性判定协议，不能与新版判定混用；现有 `workflow_runs.jsonl` 可以作为新版 `judge` 的输入，但不能继续追加新版 `run`。`--limit` 可先做小规模试运行。

`negative_queries.jsonl` 提供 8 条无答案或错误前提的候选题，标记为 `provisional_unanswerable`。它们面向当前九篇论文的 KB 快照；正式使用前须逐条核对冻结全文。`--queries` 可重复指定，例如 `--queries eval/kb/queries.jsonl --queries eval/agent_hallucination/negative_queries.jsonl`。报告的 `by_tag` 会分别给出这类题的结果。

## 固定证据实验

`workflow` 是两种系统的实际单轮流程，检索结果可能不同。`fixed` 向普通 agent 注入一次标准 `kb_retrieve` 工具结果，并将同一批 chunk 提供给 multi-agent 的 synthesis/critic 节点；它比较给定证据后的生成环节。两边保留各自正常的证据呈现格式和截断策略。先人工核查证据列表，然后建立 JSONL 文件：

```json
{"query_id":"kb001","chunk_ids":["2038c6a08557:8","2038c6a08557:9"]}
```

每条 query 都需要一行；KB 无法回答的题目可用空数组。运行命令与上面相同，只需改为 `--protocol fixed --fixed-evidence path/to/evidence.jsonl`，并使用单独的 runs、claims、report 文件。`--fixed-from-seeds` 仅用于快速检查流程：原题目的 `seed_chunk_ids` 是候选锚点，未经核验，不能当作正式证据真值。

## 逐事实证据核验与指标

`judge` 将回答拆为原子事实，每批最多八条，只检查 `runs.jsonl` 记录的、实际送入回答模型的完整证据，不重新检索 KB。`supported` 和 `contradicted` 必须附能在这批证据中找到的原文引句；证据既不支持也不直接反驳时标为 `unsupported`。判定格式错误、引句不存在等情况标为 `needs_review`。这里的 `unsupported` 表示**当时证据没有支撑**，不能据此断言模型编造或冻结 KB 没有对应事实。

人工复核后，可另建 `reviews.jsonl`，每行覆盖一条声明，例如：

```json
{"run_id":"workflow:kb001:0:single","index":0,"status":"unsupported","reviewer":"manual","reason":"已检查该 run 的全部可见证据"}
{"run_id":"workflow:kb001:0:multi","index":1,"status":"supported","quote":"The diffusion pipeline is then initialized by these latent representations","reviewer":"manual"}
```

允许的人工状态为 `supported`、`contradicted`、`unsupported`、`needs_review` 和 `not_factual`。前两者必须提供当时可见证据中的精确引句；人工标记 `unsupported` 需注明复核者和原因。报告只给声明已全部定案的回答计算无证据支撑率；待复核回答、拒答、暂停和错误分别计数。

建立复核文件后，在 `report` 命令中加上 `--reviews eval/agent_hallucination/workflow_grounding_reviews.jsonl`。

### 批量自动复核待审核事实

`auto_review_grounding` 只处理原始 `needs_review` 声明。它从每次 run 当时可见的证据抽取带编号的原文段落，先让判定模型选一段，再单独核对该段是否完整支持声明；两次判定一致才提出支持标签。输出单独保存，原始 `workflow_grounding_claims.jsonl` 不变，可增量续跑。对证据缺失、两轮不一致和格式异常的条目继续保留 `needs_review`；不会仅凭模型未找到证据就自动断言 `unsupported`。

```bash
python -m scripts.auto_review_grounding \
  --workspace /tmp/nanobot-kb-eval \
  --runs eval/agent_hallucination/workflow_runs.jsonl \
  --judgments eval/agent_hallucination/workflow_grounding_claims.jsonl \
  --output eval/agent_hallucination/workflow_grounding_auto_reviews.jsonl \
  --conservative-output eval/agent_hallucination/workflow_grounding_auto_reviews_conservative.jsonl \
  --disable-thinking
```

`--conservative-output` 将模型提出的 `contradicted` 留给人工确认；只有孤立的 `"year": NNNN` 引句也不足以证明是哪篇论文的年份。这些条目会回到 `needs_review`，并保留候选标签和引句供复核。要逐条重试未定案的条目，可用 `--seed-reviews` 指向首轮输出，另设 `--output`，并加 `--batch-size 1`；已定案的首轮结果会复制到新文件，复核时使用新文件生成的 conservative 输出。

抽查表明，同一判定模型有时会把**另一方法的数值**误判为支持。因此在将银标纳入报告前，再用来源论文标题、原文段落和更严格的提示核对所有候选 `supported`；方法名与来源或原文不一致、无法完整证实的条目仍待审核。`--remaining-output` 另外生成简短的人工复核队列，含未定案的声明和抽取失败的回答：

```bash
python -m scripts.audit_grounding_support \
  --workspace /tmp/nanobot-kb-eval \
  --runs eval/agent_hallucination/workflow_runs.jsonl \
  --judgments eval/agent_hallucination/workflow_grounding_claims.jsonl \
  --candidates eval/agent_hallucination/workflow_grounding_auto_reviews_conservative.jsonl \
  --output eval/agent_hallucination/workflow_grounding_auto_reviews_audited.jsonl \
  --remaining-output eval/agent_hallucination/workflow_grounding_remaining_review.jsonl \
  --disable-thinking

python -m scripts.eval_agent_hallucination report \
  --workspace /tmp/nanobot-kb-eval \
  --runs eval/agent_hallucination/workflow_runs.jsonl \
  --judgments eval/agent_hallucination/workflow_grounding_claims.jsonl \
  --reviews eval/agent_hallucination/workflow_grounding_auto_reviews_audited.jsonl \
  --rubrics eval/agent_hallucination/workflow_answer_rubrics.jsonl \
  --answer-judgments eval/agent_hallucination/workflow_answer_judgments.jsonl \
  --answer-reviews eval/agent_hallucination/workflow_answer_reviews.jsonl \
  --output eval/agent_hallucination/workflow_full_report_auto_grounding.json
```

自动确认的支持标签仍属于**银标**，建议抽样人工核对。剩余待审核条目不能当作 `unsupported`，也不能用于推断最终幻觉率。

主指标 `query_ungrounded_rate` 是“含至少一条 `unsupported` 或 `contradicted` 声明的已评分回答数 / 已评分回答数”；`claim_ungrounded_rate` 以事实声明为分母。`paired_query_difference` 为 multi-agent 减普通 agent 的同题差值及按 query 聚类的 bootstrap 95% 区间。`workflow` 的分数同时受检索与生成影响；`fixed` 用同一批 chunk 比较给定证据后的生成环节。已有 `eval_kb.py` 的 query–chunk 等级 0/1/2 只评检索相关性，不能替代这里的逐事实复核。

建议先核查所有待复核与判为无证据支撑的声明，再抽查判为支持的声明和抽取结果。报告中的 `pending_review` 应清零后再比较两个模式。

## 回答充分性与拒答

“回答是否解答了问题”在此称为**回答充分性**；上面的证据支撑率衡量回答中的事实是否被 agent 当时看到的证据支持。两个指标分别报告，避免把流畅但遗漏要点的回答或一律拒答的系统评为更好。

先复制 [`workflow_answer_rubrics.draft.jsonl`](workflow_answer_rubrics.draft.jsonl) 为 `workflow_answer_rubrics.jsonl`。草稿含 32 条自动生成题的候选要点和引句，以及 8 条负例题的候选“不可回答”标签；**全部是 `provisional`，不能直接计分**。逐题核查冻结 KB、题目是否可回答、必需覆盖的要点及其 `chunk_id`、原文引句；需要时增删 `required_points`。确认后把 `review_status` 改为 `reviewed`，填写 `reviewer` 和 `review_reason`。不可回答题尤其需要检查全文；代码只验证可回答题的引句存在于指定 chunk，不能替代语义复核。

审核后的可回答题保留如下字段；不可回答题用 `"answerability":"unanswerable"` 和空的 `required_points`：

```json
{"query_id":"auto_kb_001","query":"...","answerability":"answerable","required_points":[{"id":"p1","text":"...","chunk_id":"0e5315955f0a:6","quote":"..."}],"review_status":"reviewed","reviewer":"姓名或标识","review_reason":"已核对冻结 KB 全文与题目要点","corpus":{"documents.jsonl":"<SHA-256>","chunks.jsonl":"<SHA-256>"}}
```

评分只依据已审核的 rubric；未审核的题目在报告中列入 `answer_quality.pending_review`。`judge-answers` 对每个已完成回答判定 `answer`／`abstain`，逐要点检查是否覆盖，并要求覆盖的要点提供能在回答中找到的引句。判定结果是银标，可在报告中用人工复核覆盖。使用新的输出文件：

```bash
python -m scripts.eval_agent_hallucination judge-answers \
  --workspace /tmp/nanobot-kb-eval \
  --runs eval/agent_hallucination/workflow_runs.jsonl \
  --rubrics eval/agent_hallucination/workflow_answer_rubrics.jsonl \
  --output eval/agent_hallucination/workflow_answer_judgments.jsonl \
  --disable-thinking

python -m scripts.eval_agent_hallucination report \
  --workspace /tmp/nanobot-kb-eval \
  --runs eval/agent_hallucination/workflow_runs.jsonl \
  --judgments eval/agent_hallucination/workflow_grounding_claims.jsonl \
  --rubrics eval/agent_hallucination/workflow_answer_rubrics.jsonl \
  --answer-judgments eval/agent_hallucination/workflow_answer_judgments.jsonl \
  --output eval/agent_hallucination/workflow_full_report.json
```

若需修正某条充分性判定，另建 `workflow_answer_reviews.jsonl`，每行覆盖一个 run。`covered_points` 中的引句必须出现在该 run 的回答中；无须重写未覆盖的要点：

```json
{"run_id":"workflow:auto_kb_001:0:single","answer_type":"answer","covered_points":[{"id":"p1","quote":"对原始图像应用高斯模糊"}],"reviewer":"manual","reason":"已核对答案与评分要点"}
```

在 `report` 加 `--answer-reviews eval/agent_hallucination/workflow_answer_reviews.jsonl` 即可。`full_answer_rate` 衡量可回答题全部要点覆盖的比例；`mean_point_coverage` 衡量平均要点覆盖率；`incorrect_abstain_rate` 衡量可回答题的错误拒答；`correct_abstain_rate` 衡量不可回答题的正确拒答。`task_success_rate` 在可回答题要求完整覆盖，在不可回答题要求正确拒答。报告保留各项分数，不合并为单一质量分。

## 时延与成本

`runs.jsonl` 已记录每次的 `elapsed_s`、`model_calls` 和 `usage`。报告新增完成回答的 `completed_p50_elapsed_s`、`completed_p95_elapsed_s`、`completion_rate`，以及同题同次运行的 `paired_latency_difference_s`（multi-agent 减普通 agent，按 query 聚类的 bootstrap 95% 区间）。`status_counts` 分别列出完成、暂停和错误，避免把暂停耗时混入完成回答的分位数。`elapsed_s` 从 agent 初始化完成后开始计时，不包括临时 KB 副本和工作区准备时间；`mean_elapsed_s` 仍是所有状态的均值。报告还保留平均模型调用次数和总 token 数。
