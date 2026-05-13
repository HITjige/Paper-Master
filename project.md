## 论文检索与分析流程解析

这是一个完整的多智能体论文检索、入库、分析和技能提取流程，核心由 **4 层构成**：

---

### 🧭 第一层：路由层 (Router Agent)

文件：nodes.py → `router_node()`

用户提问后，**Router Agent** 先判断意图，决定走哪条路径：

| 路由决策 | 说明 | 触发场景 |
|---|---|---|
| `internal` | 仅检索本地知识库 | 一般论文/概念问题（默认路径） |
| `external` | 搜索 arXiv | 含"最新""latest"等关键词 |
| `hybrid` | 先内部检索，再外部搜索 | 含"对比""compare"等关键词 |
| `direct` | 直接回答，无需检索 | 打招呼、非学术问题 |

路由规则定义在 agents.py → `ROUTER_SYSTEM_PROMPT`。

---

### 📚 第二层：检索层 (Retrieval Agent)

文件：nodes.py → `retrieval_node()`

#### 1️⃣ 查询优化（核心预处理）

- **Rule-based cleaning**：去除语气词、填充词（"帮我找一下" → ""）
- **LLM Rewrite**：根据对话上下文压缩历史，用大模型重写查询（中译英、代词消解、去噪）
- **Query Decomposition**：将一个复杂问题拆解成 2~4 个子查询，覆盖不同角度
- **Entity Extraction**：从查询中提取显式的 paper_id 或论文标题，实现精准实体检索

结果缓存在 `state["rewritten_queries"]` 中，检索和研究节点共享。

#### 2️⃣ 知识库检索 — HyDE（Hypothetical Document Embeddings）

`PaperKnowledgeBase` (paper_kb.py) 实现了 **HyDE 检索范式**，使用 Chroma 向量数据库配合 **三层索引**：

| 集合 (Collection) | 存储内容 | 用途 |
|---|---|---|
| `paper_summaries` | 每个chunk的LLM摘要嵌入 | 通过摘要匹配查询意图 |
| `paper_questions` | 每个chunk的假设性问题嵌入 | 通过用户可能问的问题来检索 |
| `paper_chunks` | 原始chunk文本+元数据 | 最终返回的父文档内容 |

**检索流程：**
1. 对查询向量化，同时搜索 `summaries` 和 `questions` 集合
2. **BM25 稀疏索引**做关键词检索
3. **RRF (Reciprocal Rank Fusion)** 融合稠密向量和稀疏结果
4. **MMR (Maximal Marginal Relevance)** 多样化选择，防止单篇论文垄断
5. 评分低于 best_score × 0.3 的结果被过滤

**质量评估：**
- best_score ≤ threshold - margin → `insufficient` → 转外部搜索
- best_score ≥ threshold + margin → `sufficient` → 直接合成答案
- 中间灰色地带 → LLM 判断

---

### 🔬 第三层：外部研究层 (Research Agent)

文件：nodes.py → `research_node()`

当内部检索不足或用户要求最新论文时触发：

```
research_node()
  ├─ 复用 rewritten_queries（与检索层共享）
  ├─ PaperSearchTool.execute()     ← 搜索 arXiv
  │   ├─ 多候选查询并发检索
  │   ├─ 去重 (paper_id去重)
  │   ├─ PaperSimilarityTool      ← 粗排 (cosine + lexical混合)
  │   └─ PaperRerankTool          ← 精排 (sim × 0.6 + recency × 0.25 + source×0.15)
  ├─ LLM选择 top ingest_limit 篇论文
  ├─ PaperIngestTool.execute()     ← 下载+解析+入库
  │   ├─ 下载PDF/HTML
  │   ├─ MinerU 解析PDF (结构保留)
  │   ├─ MarkdownHeaderTextSplitter 语义分块
  │   ├─ LLM生成每个chunk的元数据 (summary + 假设性问题 + keywords + entities + claims)
  │   └─ upsert_semantic_chunks() → 入库到Chroma三层索引
  └─ 设置 post_research_retrieval=true → 回检索层检索新入库内容
```

**`PaperSearchTool`** (`tools/paper.py`) 的内部工作流：
```
用户查询 → 候选查询生成(LLM/keyword) → 并发arXiv检索 → 去重 → 粗排 → 精排 → 输出
```

**`PaperIngestTool`** 的入库流程：
```
论文元数据 → 下载PDF → MinerU解析 → 去噪(去掉Reference/Acknowledgement) →
Markdown语义分块 → LLM为每块生成摘要+假设性问题+关键词+实体+声明 →
upsert到Chroma (summaries + questions + chunks三层)
```

---

### 🧠 第四层：合成与审查层 (Synthesis + Critic Agent)

```
synthesis_node()
  ├─ 格式化检索结果（按paper_id分组，含section路径）
  ├─ 调用LLM综合生成答案
  │   ├─ 回答语言与用户一致
  │   ├─ 使用Markdown表格/列表/粗体
  │   ├─ 引用格式 [paper_id]
  │   └─ 如果critic反馈，必须处理
  └─ 输出 draft_answer

critic_node()
  ├─ 评估答案质量: passed / needs_revision / needs_more_info
  │   ├─ passed → 结束，输出 final_answer
  │   ├─ needs_revision → 回 synthesis 修改 (最多3轮)
  │   └─ needs_more_info → 再触发 research
  └─ 防止死循环（max_iterations=3）
```

---

### 🔄 完整工作流图

```
用户查询
    │
Router Agent ──direct──→ Synthesis ──→ Critic ──passed──→ 输出答案
    │                              ↑       │
    │                              │       └─needs_revision (→ Synthesis)
    │                              │
    ├─internal ──→ Retrieval Node ─┤
    │                 │            │
    │           insufficient  sufficient
    │                 │            │
    │                 ▼            │
    └─hybrid ──→ (先 internal)     │
    │                 │            │
    │                 ▼            │
    └─external ──→ Research Node ──┘
                      │
                      ▼ (搜索+入库后)
                再次调用Retrieval
                检索刚入库的论文
```

---

### 📦 后续管线：Skill Extractor（论文知识沉淀）

论文对话完成后，`SkillExtractor` (skill_extractor.py) 异步运行：

1. **Phase 1**：分析 multi-agent 完整对话 + 论文内容，判断是否有可复用的技能
2. **Phase 2**：委托 `AgentRunner` 创建/编辑 `SKILL.md` 文件
3. 将论文中得出的洞察沉淀为可复用的 agent 技能

---

### 🧪 评估系统

eval_paper_agent.py 用 paper_eval_dataset.json 评测检索/重排质量：
- **Recall@k**：前 k 个结果中的相关论文覆盖率
- **MRR**：第一个相关论文的倒数排名

---

### 关键设计亮点

1. **HyDE 假设性文档嵌入**：不直接存论文文本嵌入，而是存：
   - LLM 生成的摘要嵌入（capture 全局内容）
   - LLM 生成的假设性问题嵌入（capture 用户可能的问法）
   
2. **RRF 混合检索**：稠密向量 + BM25 稀疏索引融合，兼顾语义和关键词

3. **MMR 多样化**：Max Marginal Relevance 避免同一篇论文的多个 chunk 垄断结果

4. **查询重写共享**：检索和研究节点复用同一组优化后的查询，减少 LLM 调用

5. **防循环机制**：`loop_guard_count` + `post_research_retrieval` 标志，确保研究→检索循环后强制进入合成