# Paper-Master

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/based_on-nanobot-8A2BE2" alt="Based on nanobot">
  </p>
</div>

**Paper-Master** 是一款基于 [nanobot](https://github.com/HKUDS/nanobot) 二次开发的多智能体论文检索与问答系统。
它在 nanobot 轻量级 AI Agent 框架的基础上，深度集成了 **LangGraph 多 Agent 工作流**、**HyDE 假设性问题检索**、**Chroma + BM25 混合知识库**，实现了从论文搜索、PDF 解析、语义入库到引用级问答的端到端闭环。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **🧠 多智能体论文系统** | 基于 LangGraph 的 5 Agent 协作，从搜索、理解到引用的端到端论文问答 |
| **📚 智能知识库 (PaperKB)** | Chroma 向量库 + BM25 混合检索，支持 **HyDE 假设性问题检索**范式 |
| **🔧 丰富工具生态** | 20+ 内置工具：代码/文件操作、Web 搜索、Shell、论文检索、知识库等 |
| **💬 多聊天渠道** | Telegram、Discord、飞书、微信、Slack、钉钉、QQ、Matrix、Email、WebSocket |
| **🧠 三层记忆系统** | 短时会话 → Consolidator 历史归档 → Dream 长期记忆固化 |
| **📖 可复用技能** | SKILL.md 驱动，从对话中自动萃取知识，跨对话复用 |
| **🔌 MCP 支持** | 兼容 Model Context Protocol，集成外部工具和数据源 |
| **📡 多 Provider** | OpenAI、Anthropic、OpenRouter、DeepSeek、Qwen、本地模型等 |

---

## 🏗️ 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Interface                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────┐ ┌──────┐          │
│  │ Telegram  │ │  Discord  │ │  WeChat   │ │CLI │ │ WebUI│   ...   │
│  └──────────┘ └──────────┘ └──────────┘ └────┘ └──────┘          │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│                      Agent Loop (agent/loop.py)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │  Router   │  │  Context  │  │   Tools   │  │   Memory/Skills    │ │
│  │(MultiAgent)│  │  Builder  │  │  Registry  │  │   Injection        │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                               │
     ┌─────────────────────────┼─────────────────────────┐
     │                         │                         │
     ▼                         ▼                         ▼
┌───────────┐         ┌──────────────┐          ┌─────────────┐
│  Multi-   │         │   Provider    │          │    MCP      │
│  Agent    │         │   (LLM)      │          │   Server    │
│  System   │         └──────────────┘          └─────────────┘
└───────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────┐
│          Multi-Agent Paper System (LangGraph 编排)                  │
│  ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐   │
│  │ Router │→│Retrieval│→│Research  │→│Synthesis │→│ Critic │   │
│  │ Agent  │ │  Agent   │ │  Agent   │ │  Agent   │ │  Agent │   │
│  └────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘   │
│       │        │        │               ↑        │              │
│       │        │        └───arXiv API───┘        │              │
│       │        │               │                  │              │
│       │        │         ┌─────▼──────┐           │              │
│       │        │         │ PaperKB    │           │              │
│       │        └────────→│ (Chroma +   │──────────┘              │
│       │                  │  BM25)     │                         │
│       └─────────────────→│ JSONL      │                         │
│                          └────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔬 核心系统设计

### 5 个专用 Agent 协作

| Agent | 职责 | 关键能力 |
|-------|------|---------|
| **Router** | 路由裁决 | 基于 LLM+ 7 层上下文判断走 internal/external/hybrid/direct |
| **Retrieval** | 知识库检索 | 统一查询改写 + HyDE 混合检索 + 质量评估 |
| **Research** | 外部搜索 | 多候选 Query 并发搜 arXiv + 去重 + 三重排序 |
| **Synthesis** | 综合写作 | 证据驱动，语言匹配，支持 Critic 迭代反馈 |
| **Critic** | 质量审查 | Pass-First 理念，3 路分支 + 最大迭代防死循环 |

### 完整工作流

```
                    用户 Query
                        │
                    ┌───▼───┐
                    │ Router │
                    └───┬───┘
          ┌──────────────┼──────────────┐
          │              │              │
     ┌────▼───┐    ┌────▼───┐     ┌───▼───┐
     │internal │    │external │     │direct │
     └────┬───┘    └────┬───┘     └───┬───┘
          │              │             │
     ┌────▼────────┐     │        ┌────▼────┐
     │ Retrieval   │←────┘        │Synthesis │
     │ Agent       │              └────┬────┘
     └────┬────────┘                   │
          │                        ┌───▼───┐
     不足  │ 充足               ┌──┤ Critic │
          │                   │   └───┬───┘
     ┌────▼────────┐          │   ┌───┴───┐
     │  Research   │──────────┘   │passed │
     │  Agent      │              └───┬───┘
     └────┬────────┘                  │
          │                      ┌────▼────┐
     ┌────▼────────┐              │  答案   │
     │ paper_ingest │              │ +引用    │
     │ → MinerU     │              └─────────┘
     │ → 语义分块    │
     │ → Chroma 入库 │
     └──────────────┘
```

### 关键技术

**① HyDE 假设性问题检索**

入库时 LLM 为每个文本块生成假设性问题；检索时用户 Query 匹配这些问题——"意图到意图"：

```
入库：文本块 → LLM 生成 {摘要, 假设性问题×3, 关键词, 实体, 声明}
       → Chroma 三层索引 (summaries / questions / chunks)

检索：用户 Query → 多候选改写 → 并搜 summaries + questions + BM25
       → RRF 融合 → MMR 多样化 → 0.3 阈值过滤
```

**② 多候选查询**

```
原始 Query: "最新的 EEG 分类方法"
         ↓ LLM 改写
候1: "EEG signal classification"
候2: "brain wave pattern recognition"
候3: "electroencephalography deep learning"
         ↓ 并发 arXiv（3s 间隔防限流）→ 去重 → 粗排 → 精排
```

**③ PDF 智能解析**

```
MinerU 解析 → 去 References → 图/表自动提取(KV)
→ 合并断裂段落 → 剥离前言 → 语义分块
→ LLM 生成元数据 → upsert 到 Chroma
```

**④ 三重排序**

```
rerank_score = 0.6×CrossEncoder + 0.25×recency + 0.15×source_prior
```

**⑤ 防死循环**

Critic: Pass-First 理念 / 3 路条件边 / max_iterations=3 强制输出

---

## 🧠 记忆系统

```
短时记忆             存档记忆                 长期记忆
session.messages → history.jsonl → MEMORY.md / SOUL.md / USER.md
                      ↑                    ↑
              Consolidator            Dream（两阶段处理）
              (Token预算压缩)       (定期消化→精准编辑)
                                        ↑
                                    GitStore
                              (版本管理 + 逐行老化标注)
```

| 层级 | 存储 | 能力 |
|------|------|------|
| **L1 短时** | 会话消息列表 | 存活于当前对话，用完即弃 |
| **L2 存档** | `history.jsonl` | Consolidator 自动压缩归档（有并发锁 + 降级保护） |
| **L3 固化** | `MEMORY.md` / `SOUL.md` / `USER.md` | Dream 两阶段分析 + 精准编辑，Git 版本管理 |

---

## 📦 安装

```bash
# 克隆本仓库
git clone https://github.com/your-username/paper-master.git
cd paper-master
pip install -e .
```

> **注**：Paper-Master 基于 [nanobot](https://github.com/HKUDS/nanobot) 二次开发，支持 nanobot 原生框架的全部功能（聊天渠道、MCP 等）。

## 🚀 快速开始（WebUI）

<p align="center">
  <img src="images/nanobot_webui.png" alt="nanobot webui preview" width="900">
</p>

**1. Enable the WebSocket channel in `~/.nanobot/config.json`**

```json
{ "channels": { "websocket": { "enabled": true } } }
```

**2. Build**

```bash
nanobot onboard

cd webui
bun --bun install
bun --bun run build
```

**3. Start the gateway**

```bash
nanobot gateway
```

论文相关 Query 自动触发多 Agent 检索，例如：
- `"最新的 Mamba 架构论文有哪些？"`
- `"总结一下 Transformer 注意力机制的原理"`
- `"对比一下 GPT-4 和 Claude-3 的架构差异"`

支持 pdf 论文上传，自主构建知识库

## 🔧 核心工具

| 类别 | 工具 |
|------|------|
| **论文检索** | `paper_search` / `paper_ingest` / `kb_retrieve` / `paper_rerank` |
| **文件操作** | `read_file` / `write_file` / `edit_file` / `list_dir` |
| **代码执行** | `exec`（沙箱 Shell） |
| **Web** | `web_fetch` / `web_search` |
| **搜索** | `grep` / `glob` |
| **记忆** | `memory`（读写 MEMORY.md） |
| **消息** | `message`（跨会话通知） |
| **MCP** | 任意 MCP 服务器工具 |
| **规划** | `notebook_edit`（Jupyter）/ `spawn`（子 Agent） |

## 📚 相关文档

| 主题 | 文档 |
|------|------|
| 多智能体工作流 | [`docs/multi-agent-guide.md`](./docs/multi-agent-guide.md) |
| 论文评测说明 | [`docs/paper-expert-eval.md`](./docs/paper-expert-eval.md) |
| 快速配置指南 | [`docs/configuration.md`](./docs/configuration.md) |
| 论文解析流程 | [`project.md`](./project.md) |

> Paper-Master 继承了 nanobot 的全部文档能力。nanobot 的完整文档请见 [docs/README.md](./docs/README.md) 或 [nanobot.wiki](https://nanobot.wiki)。

## 🤝 贡献

PRs welcome！本项目专注于论文检索与问答场景，欢迎提交改进。

如需 nanobot 原生框架的完整功能，请访问 [nanobot GitHub](https://github.com/HKUDS/nanobot)。
