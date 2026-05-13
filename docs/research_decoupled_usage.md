# Research Node 解耦使用指南

## 概述

`research_node` 现在支持 search 和 ingest 的解耦执行，允许用户先查看搜索结果，再决定是否摄取以及摄取哪些论文。

## 工作流程

```
Router → Research (search phase) → Synthesis (show selection UI) → [External: user input] → Research (ingest phase) → Retrieval → Synthesis
```

**注意**: 由于 LangGraph 的限制，无法真正"暂停"等待用户输入。因此流程设计为：
1. Research 节点完成搜索后，跳转到 Synthesis 节点展示选择界面
2. 外部系统捕获这个输出，获取用户选择
3. 外部系统更新 state 后重新调用 graph，进入 ingest 阶段

## State 字段说明

### 新增字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `research_phase` | str | 当前阶段: `"search"` \| `"select"` \| `"ingest"` \| `"complete"` |
| `search_completed` | bool | 搜索是否已完成 |
| `papers_for_selection` | List[Dict] | 搜索到的论文列表（供用户选择） |
| `user_selected_papers` | List[str] | 用户选择的 paper_id 列表 |
| `user_skip_ingest` | bool | 用户是否选择跳过摄取 |

### 流程控制

1. **search 阶段**: 执行搜索，结果存入 `papers_for_selection`，`research_phase` 变为 `"select"`
2. **select 阶段**: 跳转到 Synthesis 展示选择界面，`draft_answer` 包含格式化内容
3. **ingest 阶段**: 根据用户选择执行摄取
4. **complete 阶段**: 完成，跳转到 Retrieval 进行后续检索

## 使用示例

### 基本用法

```python
from nanobot.agent.multi_agent.graph import MultiAgentGraph
from nanobot.agent.multi_agent.state import create_initial_state

# 初始化 graph
graph = MultiAgentGraph(provider, kb, tools)

# 创建初始 state
initial_state = create_initial_state(user_query="Transformer architecture survey")

# 第一次调用 - 执行 search 阶段，完成后会跳转到 synthesis 展示选择界面
state = await graph.graph.ainvoke(initial_state)

# 检查是否处于 select 阶段（在 synthesis 节点展示了选择界面）
if state.get("research_phase") == "select":
    # 从 draft_answer 获取选择界面内容
    selection_ui = state["draft_answer"]
    print(selection_ui)  # 展示给用户
    
    # 获取用户选择（假设用户选择了前2篇）
    papers = state["papers_for_selection"]
    selected_ids = [papers[0]["paper_id"], papers[1]["paper_id"]]
    
    # 更新 state 以触发 ingest 阶段
    state["user_selected_papers"] = selected_ids
    state["research_phase"] = "ingest"
    
    # 第二次调用 - 执行 ingest 阶段
    final_state = await graph.graph.ainvoke(state)
```

### 跳过摄取

```python
if state.get("research_phase") == "select":
    # 用户选择跳过
    state["user_skip_ingest"] = True
    # 可以设置一个空列表或保持 user_selected_papers 为空
    state["user_selected_papers"] = []
    state["research_phase"] = "ingest"
    
    final_state = await graph.graph.ainvoke(state)
```

### 摄取全部

```python
if state.get("research_phase") == "select":
    # 用户选择全部
    papers = state["papers_for_selection"]
    state["user_selected_papers"] = [p["paper_id"] for p in papers]
    state["research_phase"] = "ingest"
    
    final_state = await graph.graph.ainvoke(state)
```

## 选择界面格式

当搜索完成后，`draft_answer` 字段包含格式化的选择界面：

```markdown
## 📚 搜索到的论文

请选择要摄取到知识库的论文（输入 paper_id 列表，或回复指令）：

### [1] Attention Is All You Need
- **ID**: `1706.03762`
- **年份**: 2017
- **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar
- **摘要**: We propose a new simple network architecture, the Transformer...

### [2] BERT: Pre-training of Deep Bidirectional Transformers
- **ID**: `1810.04805`
- **年份**: 2018
- **作者**: Jacob Devlin, Ming-Wei Chang, Kenton Lee
- **摘要**: We introduce a new language representation model called BERT...

---

**操作说明**：
- 回复论文 ID 列表（如：`1706.03762, 1810.04805`）选择要摄取的论文
- 回复 `skip` 跳过摄取
- 回复 `all` 摄取全部
```

## 与现有流程的兼容性

- 如果不使用新字段，系统会保持原有行为（自动摄取 top_k 论文）
- `external_papers` 和 `ingested_papers` 字段仍然可用
- 可以通过设置 `research_phase` 初始值为 `"ingest"` 来跳过选择过程

## 注意事项

1. **状态持久化**: 在两次调用之间需要保存 state，因为 LangGraph 是无状态的
2. **超时处理**: 建议在外部实现超时逻辑，超时后默认视为 `skip`
3. **错误处理**: 如果摄取失败，会记录在 `error_message` 中，但流程会继续
