# from sentence_transformers import CrossEncoder

# rerank_model = "/data1/project/models/Qwen3-Reranker-0.6B"
# rerank_model = CrossEncoder(rerank_model)

# query = "What are the latest advancements in EEG synthesis using diffusion models?"
# doc = "This work studies EEG synthesis with diffusion models and guidance."
# score = rerank_model.predict([(query, doc)])[0]
# print("Relevance score:", score)


# from pypdf import PdfReader
# from pathlib import Path

# _MAX_TEXT_LENGTH = 200_000

# def _truncate(text: str, max_length: int) -> str:
#     """Truncate text with a suffix indicating truncation."""
#     if len(text) <= max_length:
#         return text
#     return text[:max_length] + f"... (truncated, {len(text)} chars total)"

# def _extract_pdf(path: Path) -> str:
#     """Extract text from PDF using pypdf."""
#     try:
#         reader = PdfReader(path)
#         pages: list[str] = []
#         for i, page in enumerate(reader.pages, 1):
#             text = page.extract_text() or ""
#             pages.append(f"--- Page {i} ---\n{text}")
#         return _truncate("\n\n".join(pages), _MAX_TEXT_LENGTH)
#     except Exception as e:
#         return f"[error: failed to extract PDF: {e!s}]"
    
# print(_extract_pdf(Path("/home/eeg2img/.nanobot/workspace/kb/downloads/2604.20834v1.pdf")))


# from langchain_mineru import MinerULoader
from nanobot.agent.tools.paper import _strip_front_matter, _merge_split_paragraphs, _remove_noisy_blocks, _extract_assets_and_strip, _extract_figure_table_blocks, _normalize_blank_lines, _parse_asset_refs

# loader = MinerULoader(
#     source="/home/eeg2img/.nanobot/workspace/kb/BraVL：Decoding Visual Neural Representations by Multimodal Learning of Brain-Visual-Linguistic Features (TPAMI23).pdf",
#     language="en",
#     mode="precision",
#     token="eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIzMDkwMDk0MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3NzEyNTM5NiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiM2RkZGExYzMtNWNhNC00YWRmLThkZGUtN2NlZTMyODRmODUyIiwiZW1haWwiOiIiLCJleHAiOjE3ODQ5MDEzOTZ9.LyGD4DRvRzZffORrZDlcygd0VlylT9jitn0jIkU6t3v2rRxT8xOQW4GyP3YcQsk_VNIwVstA-adePkEwgu8w5Q"
# )
# docs = loader.load()
# text_content = docs[0].page_content
# with open("/home/eeg2img/.nanobot/workspace/kb/paper_4.md", "w") as f:
#     f.write(text_content)

with open("/home/eeg2img/.nanobot/workspace/kb/paper_2.md", "r") as f:
    text_content = f.read()
paper_id = "2604.28192v1"
text_content = _remove_noisy_blocks(text_content)
text_content, assets_by_key = _extract_figure_table_blocks(
    text_content, paper_id=paper_id,
)
# for key, asset in assets_by_key.items():
#     print(f"Extracted asset: {key}, type: {asset['type']}\nCaption: {asset.get('caption', 'N/A')}\nContent:\n{asset.get('content', 'N/A')}\n{'-'*80}")
# text_content = _merge_split_paragraphs(text_content)
# # Strip front matter (title/authors/abstract before Introduction)
# text_content = _strip_front_matter(text_content)
# # Final cleanup: discard any remaining image links / HTML tables
# text_content, _ = _extract_assets_and_strip(text_content, paper_id=paper_id)
# text_content = _normalize_blank_lines(text_content)

# with open("/home/eeg2img/.nanobot/workspace/kb/paper_cleaned_4.md", "w") as f:
#     f.write(text_content)

from nanobot.agent.tools.paper import _split_markdown_semantic, _parse_asset_refs
from collections import Counter

with open("/home/eeg2img/.nanobot/workspace/kb/paper_cleaned_2.md", "r") as f:
    text_content = f.read()

semantic_chunks = _split_markdown_semantic(
    text_content,
    max_chunk_chars=4096,
    min_chunk_chars=100,
)

all_parsed_refs = []
for chunk in semantic_chunks:
    text = chunk.get("text", "")
    print(f"Chunk (length={len(text)}):\n{text[:500]}\n{'-'*20}")
    parsed_refs = _parse_asset_refs(text)
    all_parsed_refs.extend(parsed_refs)
    parsed_refs = list(set(parsed_refs))  # deduplicate
    for kind, num in parsed_refs:
        key = f"{paper_id}_{'Table' if kind == 'table' else 'Figure'}_{num}"
        asset = assets_by_key.get(key)
        print(f"Parsed reference: {kind} {num}, key: {key}, asset found: {'Yes' if asset else 'No'}")
    print("="*100)
print("All parsed asset references across chunks:\n", Counter(all_parsed_refs))


# from mineru import MinerU

# # Get your free token from https://mineru.net/apiManage/token
# client = MinerU("eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIzMDkwMDk0MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3NzEyNTM5NiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiM2RkZGExYzMtNWNhNC00YWRmLThkZGUtN2NlZTMyODRmODUyIiwiZW1haWwiOiIiLCJleHAiOjE3ODQ5MDEzOTZ9.LyGD4DRvRzZffORrZDlcygd0VlylT9jitn0jIkU6t3v2rRxT8xOQW4GyP3YcQsk_VNIwVstA-adePkEwgu8w5Q")
# result = client.extract("/home/eeg2img/.nanobot/workspace/kb/Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion (NIPS24).pdf", pages="1-3", language="en", mode="precision")

# print(type(result.markdown))
# print(result.markdown)
# print(type(result.images))
# print(result.images)


# from pypdf import PdfReader
# from pathlib import Path

# path = "https://arxiv.org/pdf/2507.07157"
# reader = PdfReader(path)
# for i, page in enumerate(reader.pages, 1):
#     text = page.extract_text()
#     with open(f"./pdf_reader.txt", "a") as f:
#         f.write(f"--- Page {i} ---\n")
#         f.write(text or "")
#         f.write("\n\n")


# num_questions = 3
# section = "Methodology"
# text = "In this paper, we propose a novel EEG synthesis framework based on diffusion models."
# prompt = """You are a scientific paper analyzer. Given a text excerpt from a paper, generate a JSON response with the following structure:

# ```json
# {{
#   "summary": "A concise 1-2 sentence summary of the core content",
#   "hypothetical_questions": [
#     "Question 1: What would a user ask to find this content?",
#     "Question 2: ...",
#     "Question 3: ..."
#   ],
#   "keywords": ["key term 1", "key term 2", ...]
# }}
# ```

# Requirements:
# - Summary should be 1-2 sentences capturing the essential information
# - Generate exactly {num_questions} hypothetical questions that a user might ask to retrieve this content
# - Questions should be natural search queries, not formal academic questions
# - Keywords should be 3-6 most important technical terms

# Section: {section}
# Text excerpt:
# {text_excerpt}

# Return ONLY the JSON, no explanation.""".format(
#         num_questions=num_questions,
#         section=section,
#         text_excerpt=text[:4000] if len(text) > 4000 else text,
#     )
# print(prompt)


# from urllib.parse import quote_plus
# import urllib.parse

# arbitrary_query = 'all:"multi-agent systems recent research and developments"'

# query = "multi-agent systems recent research and developments"
# keywords = ['multi-agent', 'systems']
# query = f"all:\"{query}\""
# if keywords:
#     keyword_set = " AND ".join([f'all:"{k}"' for k in keywords])
#     query += f" OR ({keyword_set})"
# params = {
#     "search_query": query,
#     "start": 0,
#     "max_results": 20,
#     "sortBy": "submittedDate",
#     "sortOrder": "descending"
# }
# query_string = urllib.parse.urlencode(params)
# print(query_string)


# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from typing import Any
# import re

# with open("/home/eeg2img/.nanobot/workspace/kb/uploads/BraVL：Decoding Visual Neural Representations by Multimodal Learning of Brain-Visual-Linguistic Features (TPAMI23).md", "r") as f:
#     text = f.read()

# headers_to_split_on = [
#     ("#", "heading_1"),
#     ("##", "heading_2"),
#     ("###", "heading_3"),
#     ("####", "heading_4"),
# ]

# splitter = MarkdownHeaderTextSplitter(
#     headers_to_split_on=headers_to_split_on,
#     return_each_line=False,
#     strip_headers=False,
# )
# md_chunks = splitter.split_text(text)

# min_chunk_chars = 100
# max_chunk_chars = 4096

# result: list[dict[str, Any]] = []
# for chunk in md_chunks:
#     # Extract header info from metadata
#     metadata = chunk.metadata or {}
#     heading_1 = metadata.get("heading_1", "")
#     heading_2 = metadata.get("heading_2", "")
#     heading_3 = metadata.get("heading_3", "")
#     heading_4 = metadata.get("heading_4", "")
    
#     # Build section name and heading path
#     section_parts = [h for h in [heading_1, heading_2, heading_3, heading_4] if h]
#     section = section_parts[-1] if section_parts else "content"
#     heading_path = " > ".join(section_parts) if section_parts else "content"
    
#     # Determine heading level
#     heading_level = 0
#     if heading_4:
#         heading_level = 4
#     elif heading_3:
#         heading_level = 3
#     elif heading_2:
#         heading_level = 2
#     elif heading_1:
#         heading_level = 1
    
#     chunk_text = chunk.page_content.strip()
    
#     # Further split if chunk is too long
#     if len(chunk_text) <= max_chunk_chars:
#         if len(chunk_text) >= min_chunk_chars:
#             result.append({
#                 "section": section,
#                 "heading_level": heading_level,
#                 "text": chunk_text,
#                 "heading_path": heading_path,
#             })
#     else:
#         # Split long chunks by paragraphs
#         paragraphs = [p.strip() for p in re.split(r"\n{2,}", chunk_text) if p.strip()]
#         buf = ""
#         for para in paragraphs:
#             candidate = f"{buf}\n\n{para}".strip() if buf else para
#             if len(candidate) <= max_chunk_chars:
#                 buf = candidate
#                 continue
#             if buf and len(buf) >= min_chunk_chars:
#                 result.append({
#                     "section": section,
#                     "heading_level": heading_level,
#                     "text": buf,
#                     "heading_path": heading_path,
#                 })
#             buf = para
#         if buf and len(buf) >= min_chunk_chars:
#             result.append({
#                 "section": section,
#                 "heading_level": heading_level,
#                 "text": buf,
#                 "heading_path": heading_path,
#             })

# for item in result:
#     print(item["heading_path"])
#     print("-" * 50)


# import json_repair
# import re

# content = "```json\n{\n    \"answer\": \"LAPO（Latent-to-Action Policy Optimization）是用于视觉-语言-动作（VLA）模型的一种强化学习框架，旨在联合优化模型的**内部推理过程**（latent reasoning）和**动作生成**，从而提升决策能力与任务表现。\\n\\n### 核心思想\\nLAPO将推理过程视为决策变量，通过强化学习让环境奖励信号直接指导模型在推理阶段和动作阶段的决策，实现更高效、自适应的决策路径。\\n\\n### LAPO 的关键实现细节\\n\\n#### 1. 框架结构\\n- 在每个时间步 $t$，模型首先基于视觉和语言输入生成一组**连续的潜在令牌**（latent tokens）$\\mathbf{Z}_t^\\\\theta$，这些令牌代表对任务的内部推理。\\n- 推理结束后，通过一个特殊标记 `<latent_end>` 表示推理阶段结束，随后模型生成动作序列 $\\mathbf{C}_t$。\\n- 动作生成使用相同的KV缓存，通过一次前向传播完成，支持并行解码，提升效率。\\n- 最终的隐藏状态被用于价值头（value head）估计当前状态的价值 $v_t$，用于优势估计（advantage estimation）。\\n\\n#### 2. 推理长度的自适应机制（Adaptive Latent CoT Reasoning）\\n为了避免固定长度推理带来的计算浪费或推理不足，LAPO引入了**动态推理长度控制**：\\n\\n- **动态终止信号**：将 `<latent_end>` 从固定终止符变为动态生成的信号。当模型预测该令牌的概率 $p \\\\geq 0.99$ 时，推理终止，进入动作预测阶段。\\n- **长度采样探索**：在训练阶段，为鼓励探索，从预设的 $M=4$ 个候选长度（如2、4、6、8）中，通过温度参数 $\\beta$ 调整概率分布，采样一个推理长度 $m \\\\sim \\\\text{Categorical}(p_1, \\\\dots, p_M)$。\\n- **概率分布公式**：\\n  $$\\n  p_m = \\\\frac{\\\\exp(l_m / \\\\beta)}{\\\\sum_{j=1}^M \\\\exp(l_j / \\\\beta)}, \\\\quad \\\\forall m \\\\in \\\\{1, \\\\dots, M\\\\}\\n  $$\\n  其中 $l_m$ 是 `<latent_end>` 在各位置的预softmax对数概率。\\n\\n- **训练中的监督优化**：引入额外的损失项 $\\mathcal{L}_{\\\\text{end}}(\\\\theta)$，基于采样位置 $m$ 的 `<latent_end>` 令牌的对数概率，构建一个**离散似然比**：\\n  $$\\n  r_t^{\\\\text{end}}(\\\\theta) = \\\\frac{\\\\pi_\\\\theta(\\\\mathbf{z}_t^{\\\\text{end}} | \\\\cdot)}{\\\\pi_{\\\\theta_{\\\\text{old}}}(\\\\mathbf{z}_t^{\\\\text{end}} | \\\\cdot)}\\n  $$\\n  该比值被加入总损失，权重为 $\\\\lambda_3$，以确保模型在需要复杂推理时能保留足够深度，在简单任务中能快速退出。\\n\\n#### 3. 政策优化目标\\nLAPO的总训练目标函数为：\\n$$\\n\\\\mathcal{L}_{\\\\text{total}}(\\\\theta) = \\\\mathcal{L}_{\\\\text{action}}(\\\\theta) + \\\\lambda_1 \\\\mathcal{L}_{\\\\text{latent}}(\\\\theta) + \\\\lambda_2 \\\\mathcal{L}_{\\\\text{value}}(\\\\theta) + \\\\lambda_3 \\\\mathcal{L}_{\\\\text{end}}(\\\\theta)\\n$$\\n\\n- **动作损失**：基于动作序列的对数概率差异，使用优势函数 $\\hat{A}_t$ 加权。\\n- **潜在损失**：使用高斯近似，计算旧潜在序列与新潜在序列之间的欧氏距离，定义为：\\n  $$\\n  r_t^z(\\\\theta) = \\\\exp\\\\left(-\\\\frac{1}{2\\\\sigma^2} \\\\sum_{k=1}^{N_z} \\\\|\\\\mathbf{z}_{t,k}^{\\\\text{old}} - \\\\mathbf{z}_{t,k}^{\\\\theta}\\\\|^2\\\\right)\\n  $$\\n  其中 $\\\\sigma$ 是固定方差超参数。\\n- **价值损失**：采用均方误差（MSE）优化状态价值估计。\\n\\n- **策略梯度**：使用**裁剪策略梯度**（clipped surrogate loss）防止策略更新过大，确保训练稳定性。\\n\\n#### 4. 优势与贡献\\n- **自适应推理**：模型能根据任务复杂度动态调整推理长度，避免冗余计算。\\n- **端到端优化**：奖励信号直接作用于推理和动作空间，提升整体决策质量。\\n- **高效推理-动作耦合**：通过共享KV缓存和统一前向过程，实现推理与动作的高效协同。\\n\\n### 实验验证\\n在LIBERO基准上，LaST-R1（基于LAPO框架）显著优于SOTA基线，尤其在需要复杂认知规划的任务中表现突出。在真实机器人部署中也验证了其鲁棒性和泛化能力 [2604.28192v1]。\\n\\n> **总结**：LAPO不是简单的推理增强，而是一个**融合推理与动作的强化学习框架**，通过自适应推理长度和联合优化机制，使VLA模型在复杂任务中实现更智能、更高效的决策。\\n\\n**参考文献**：\\n- [2604.28192v1] LaST-R1: Reinforcing Action via Adaptive Physical Latent Reasoning for VLA Models\",\n    \"citations\": [\n        \"arxiv:2604.28192v1\"\n    ],\n    \"sections_covered\": [\n        \"overview\",\n        \"method\",\n        \"results\"\n    ],\n    \"confidence\": 0.98\n}\n```"
# # content = re.sub(r'\\\\', r'\\', content)
# # content = re.sub(r'\(?![\/bfnrtu"])', r'\\', content)
# if "```json" in content:
#     json_content = content.split("```json")[1].split("```")[0]
# elif "```" in content:
#     json_content = content.split("```")[1].split("```")[0]
# else:
#     json_content = content

# result = json_repair.loads(json_content.strip())
# print(result)
