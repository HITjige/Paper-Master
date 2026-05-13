"""Lightweight paper knowledge base and retrieval helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


# ---------------------------------------------------------------------------
# BM25 sparse retrieval (lightweight, no external deps)
# ---------------------------------------------------------------------------

class _BM25Index:
    """A minimal BM25 index over a corpus of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[tuple[str, list[str]]] = []  # (doc_id, tokens)
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = defaultdict(int)
        self._idf: dict[str, float] = {}
        self._built = False

    def _tokenize(self, text: str) -> list[str]:
        return _split_tokens(text)

    def index(self, doc_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        self._corpus.append((doc_id, tokens))
        self._doc_len.append(len(tokens))
        self._built = False

    def remove(self, doc_id: str) -> None:
        self._corpus = [(did, toks) for did, toks in self._corpus if did != doc_id]
        self._doc_len = [len(toks) for _, toks in self._corpus]
        self._built = False

    def remove_by_prefix(self, prefix: str) -> None:
        self._corpus = [(did, toks) for did, toks in self._corpus if not did.startswith(prefix)]
        self._doc_len = [len(toks) for _, toks in self._corpus]
        self._built = False

    def _build(self) -> None:
        if self._built:
            return
        n = len(self._corpus)
        self._avgdl = sum(self._doc_len) / max(1, n)
        self._df.clear()
        for _, toks in self._corpus:
            seen: set[str] = set()
            for t in toks:
                if t not in seen:
                    self._df[t] += 1
                    seen.add(t)
        self._idf.clear()
        for term, df in self._df.items():
            self._idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        self._built = True

    def search(self, query: str, top_n: int = 20) -> list[tuple[str, float]]:
        self._build()
        q_tokens = self._tokenize(query)
        if not q_tokens or not self._corpus:
            return []
        scores: list[tuple[str, float]] = []
        for (doc_id, toks), dl in zip(self._corpus, self._doc_len):
            score = 0.0
            for qt in q_tokens:
                idf = self._idf.get(qt, 0.0)
                if idf == 0.0:
                    continue
                tf = toks.count(qt)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if score > 0:
                scores.append((doc_id, score))
        if scores:
            normalized_scores = {}
            for doc_id, score in scores:
                if doc_id.count(":") == 2:
                    prefix = doc_id.rsplit(":", 1)[0]
                    normalized_scores[prefix] = max(normalized_scores.get(prefix, 0.0), score)
            scores = list(normalized_scores.items())
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


def _rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion (RRF) across multiple ranked lists.

    Each list is a list of (id, score_or_distance) sorted by relevance
    (best first).  Returns a dict of id → fused_score.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranked in rankings:
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] += 1.0 / (k + rank + 1)
    return dict(fused)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _split_tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9]+", text.lower()) if tok and len(tok) > 1]


def _tokenize(text: str) -> set[str]:
    return set(_split_tokens(text))


def _parse_asset_refs(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"\b(Figure|Fig\.|Table|Tab\.?)\s*(\d+(?:\.\d+)*)",
        re.IGNORECASE,
    )
    refs: list[tuple[str, str]] = []
    for match in pattern.finditer(text or ""):
        raw_kind = match.group(1).lower()
        num = match.group(2)
        kind = "table" if raw_kind.startswith("tab") else "figure"
        refs.append((kind, num))
    return refs


def _load_asset_kv(kv_path: Path) -> dict[str, dict[str, Any]]:
    if not kv_path.exists():
        return {}
    assets: dict[str, dict[str, Any]] = {}
    with kv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = payload.get("key")
            value = payload.get("value")
            if isinstance(key, str) and isinstance(value, dict):
                assets[key] = value
    return assets


_INTENT_SECTION_HINTS: dict[str, set[str]] = {
    "limitations": {"limitations", "discussion"},
    "method": {"method", "methods", "approach"},
    "experiment": {"experiment", "results"},
    "conclusion": {"conclusion", "discussion"},
}


def _parse_query_intent(query: str) -> dict[str, Any]:
    q = (query or "").lower()
    target_sections: set[str] = set()
    intent = "general"

    if any(x in q for x in ("limitation", "limitations", "局限", "缺点", "不足")):
        intent = "limitations"
    elif any(x in q for x in ("method", "methods", "approach", "方法", "原理")):
        intent = "method"
    elif any(x in q for x in ("experiment", "results", "benchmark", "实验", "结果", "评测")):
        intent = "experiment"
    elif any(x in q for x in ("conclusion", "总结", "结论")):
        intent = "conclusion"

    target_sections |= _INTENT_SECTION_HINTS.get(intent, set())
    latest = any(x in q for x in ("latest", "recent", "newest", "最新", "近期", "最近"))

    return {
        "intent": intent,
        "target_sections": target_sections,
        "latest": latest,
    }


def _metadata_score(chunk: dict[str, Any], q_tokens: set[str], intent: dict[str, Any]) -> float:
    text = str(chunk.get("text", ""))
    section = str(chunk.get("section", "")).lower()
    kind = str(chunk.get("kind", "")).lower()

    kw_tokens: set[str] = set()
    for kw in chunk.get("keywords", []) or []:
        kw_tokens |= _tokenize(str(kw))

    claim_tokens: set[str] = set()
    for item in (chunk.get("claims", []) or []) + (chunk.get("limitations", []) or []):
        claim_tokens |= _tokenize(str(item))

    kw_overlap = len(q_tokens & kw_tokens) / max(1, len(q_tokens))
    claim_overlap = len(q_tokens & claim_tokens) / max(1, len(q_tokens))

    section_boost = 0.0
    target_sections: set[str] = intent.get("target_sections", set())
    if target_sections and section:
        if any(target in section for target in target_sections):
            section_boost = 1.0

    kind_boost = 0.15 if kind == "distilled_summary" else 0.0
    text_overlap = len(q_tokens & _tokenize(text)) / max(1, len(q_tokens))

    return 0.35 * kw_overlap + 0.30 * claim_overlap + 0.20 * section_boost + 0.10 * text_overlap + 0.05 * kind_boost


def _recency_score(year: Any) -> float:
    try:
        y = int(year)
    except (TypeError, ValueError):
        return 0.0
    now = datetime.utcnow().year
    return max(0.0, 1.0 - (now - y) / 10.0)


def _compute_chunk_signals(
    chunk: dict[str, Any],
    q_tokens: set[str],
    intent: dict[str, Any],
    docs: dict[str, dict[str, Any]],
    q_emb: list[float] | None,
    *,
    prefer_distilled: bool,
    distilled_boost_value: float,
) -> tuple[float, float, float, float, float]:
    text = str(chunk.get("text", ""))
    c_emb = chunk.get("embedding") or []
    emb_score = _cosine_similarity(q_emb, c_emb) if q_emb and isinstance(c_emb, list) else 0.0
    overlap = len(q_tokens & _tokenize(text))
    lexical = overlap / max(1, len(q_tokens))
    meta = docs.get(chunk.get("paper_id"), {})
    metadata = _metadata_score(chunk, q_tokens, intent)
    recency = _recency_score(meta.get("year"))
    distilled_boost = (
        distilled_boost_value
        if prefer_distilled and str(chunk.get("kind", "")) == "distilled_summary"
        else 0.0
    )
    return emb_score, lexical, metadata, recency, distilled_boost


def _merge_chunk_doc(chunk: dict[str, Any], meta: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "paper_id": chunk.get("paper_id"),
        "score": round(float(score), 5),
        "text": chunk.get("text", ""),
        "title": meta.get("title", ""),
        "url": meta.get("url", ""),
        "source": meta.get("source", ""),
        "year": meta.get("year"),
        "section": chunk.get("section", ""),
        "kind": chunk.get("kind", ""),
        "keywords": chunk.get("keywords", []),
        "claims": chunk.get("claims", []),
        "limitations": chunk.get("limitations", []),
        "linked_assets": chunk.get("linked_assets", []),
    }


def _select_diverse(
    scored: list[dict[str, Any]],
    *,
    k: int,
    per_paper_limit: int,
    mmr_lambda: float = 0.7,
) -> list[dict[str, Any]]:
    """MMR-based diverse selection with per-paper cap.
    
    Uses Maximal Marginal Relevance to balance relevance and diversity:
    MMR(d_i) = lambda * score(d_i) - (1-lambda) * max_{d_j in S} sim(d_i, d_j)
    
    Falls back to greedy selection when embedding vectors are not available.
    
    Args:
        scored: Scored chunks sorted by relevance (descending)
        k: Number of results to return
        per_paper_limit: Max chunks per paper
        mmr_lambda: Relevance-diversity trade-off (1.0 = relevance-only, 0.0 = diversity-only)
        
    Returns:
        Diverse top-k results
    """
    if not scored or k <= 0:
        return []
    
    # Check if embeddings are available for MMR
    has_embeddings = any(
        isinstance(c.get("embedding"), list) and len(c.get("embedding", []) or []) > 0
        for c in scored[:3]
    )
    
    if not has_embeddings:
        # Fallback to greedy selection
        selected: list[dict[str, Any]] = []
        per_paper_count: dict[str, int] = {}
        for item in scored:
            pid = str(item.get("paper_id", ""))
            if pid and per_paper_count.get(pid, 0) >= per_paper_limit:
                continue
            selected.append(item)
            if pid:
                per_paper_count[pid] = per_paper_count.get(pid, 0) + 1
            if len(selected) >= k:
                break
        return selected
    
    # MMR selection
    selected: list[dict[str, Any]] = []
    remaining = list(scored)
    per_paper_count: dict[str, int] = {}
    lam = max(0.0, min(1.0, mmr_lambda))
    
    while len(selected) < k and remaining:
        best_idx = 0
        best_mmr = float("-inf")
        
        for i, item in enumerate(remaining):
            pid = str(item.get("paper_id", ""))
            if pid and per_paper_count.get(pid, 0) >= per_paper_limit:
                continue
            
            relevance = float(item.get("score", 0.0))
            
            # Compute max cosine similarity to already-selected items
            max_sim = 0.0
            item_emb = item.get("embedding")
            if item_emb and selected:
                for sel in selected:
                    sel_emb = sel.get("embedding")
                    if sel_emb:
                        sim = _cosine_similarity(
                            item_emb if isinstance(item_emb, list) else [],
                            sel_emb if isinstance(sel_emb, list) else [],
                        )
                        max_sim = max(max_sim, sim)
            
            mmr = lam * relevance - (1.0 - lam) * max_sim
            
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        
        winner = remaining.pop(best_idx)
        selected.append(winner)
        pid = str(winner.get("paper_id", ""))
        if pid:
            per_paper_count[pid] = per_paper_count.get(pid, 0) + 1

    # --- 后处理：确保每篇论文至少有一个 chunk 被选中 ---
    selected_pids = {str(item.get("paper_id", "")) for item in selected}
    missing_pids: set[str] = set()
    for item in remaining:
        pid = str(item.get("paper_id", ""))
        if pid and pid not in selected_pids:
            missing_pids.add(pid)

    if missing_pids:
        # 找出每篇缺失论文的最佳 chunk
        best_per_paper: dict[str, dict[str, Any]] = {}
        for item in remaining:
            pid = str(item.get("paper_id", ""))
            if pid in missing_pids:
                score = float(item.get("score", 0.0))
                if pid not in best_per_paper or score > best_per_paper[pid].get("score", 0.0):
                    best_per_paper[pid] = item
                    
    selected.extend(best_per_paper.values())
    
    return selected


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _hash_embedding(text: str, dims: int = 256) -> list[float]:
    vec = [0.0] * dims
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i, byt in enumerate(digest):
            idx = (i * 31 + byt) % dims
            sign = -1.0 if (byt % 2) else 1.0
            vec[idx] += sign * ((byt / 255.0) + 0.1)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


@dataclass(slots=True)
class PaperKbConfig:
    enabled: bool = True
    embedding_api_key: str = ""
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_model: str = "/data1/project/models/bge-small-zh-v1.5"
    rerank_model: str = "/data1/project/models/Qwen3-Reranker-0.6B"
    retrieval_top_k: int = 5
    max_chunk_chars: int = 4096
    min_chunk_chars: int = 300
    # New config for hypothetical question retrieval
    num_hypothetical_questions: int = 3
    enable_hypothetical_retrieval: bool = True
    chroma_persist_dir: str = ""  # Will default to {workspace}/kb/chroma
    mmr_lambda: float = 0.7  # Relevance-diversity trade-off for MMR selection (1.0=relevance-only)
    use_hybrid_retrieval: bool = True  # Enable BM25+dense RRF hybrid retrieval


class PaperKnowledgeBase:
    """Paper knowledge base with Chroma vector storage for semantic retrieval.
    
    Supports both:
    - Traditional JSONL-based storage (for backup/export)
    - Chroma vector database (for efficient semantic search)
    
    Chroma Collections:
    - paper_summaries: stores summary embeddings
    - paper_questions: stores hypothetical question embeddings
    - paper_chunks: stores parent document metadata and text
    """

    def __init__(self, workspace: Path, config: PaperKbConfig):
        self.workspace = workspace
        self.config = config
        self.base_dir = workspace / "kb"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.docs_file = self.base_dir / "documents.jsonl"
        self.chunks_file = self.base_dir / "chunks.jsonl"
        self.embedding_model = None
        self.rerank_model = None
        
        # Initialize Chroma collections for semantic retrieval
        self._chroma_client = None
        self._summary_collection = None
        self._question_collection = None
        self._chunk_collection = None
        # BM25 sparse indexes (keyed by Chroma collection name)
        self._bm25_indexes: dict[str, _BM25Index] = {}
        self._init_chroma_collections()

    def _init_chroma_collections(self) -> None:
        """Initialize Chroma vector database collections."""
        if not self.config.enable_hypothetical_retrieval:
            return
        
        try:
            import chromadb
            
            # Determine persist directory
            persist_dir = self.config.chroma_persist_dir or str(self.base_dir / "chroma")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            
            # Create persistent client
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            
            # Create or get collections
            self._summary_collection = self._chroma_client.get_or_create_collection(
                name="paper_summaries",
                metadata={"hnsw:space": "cosine"},
            )
            self._question_collection = self._chroma_client.get_or_create_collection(
                name="paper_questions",
                metadata={"hnsw:space": "cosine"},
            )
            self._chunk_collection = self._chroma_client.get_or_create_collection(
                name="paper_chunks",
                metadata={"hnsw:space": "cosine"},
            )
            
            logger.info(
                "Chroma initialized: summaries={}, questions={}, chunks={}",
                self._summary_collection.count(),
                self._question_collection.count(),
                self._chunk_collection.count(),
            )
            # Rebuild BM25 indexes from existing Chroma data
            self._init_bm25_from_chroma()
        except Exception as e:
            logger.warning("Chroma initialization failed: {}, falling back to JSONL-only mode", e)
            self._chroma_client = None
            self._summary_collection = None
            self._question_collection = None
            self._chunk_collection = None

    def _init_bm25_from_chroma(self) -> None:
        """Rebuild BM25 indexes from existing Chroma collections on startup."""
        if not self.config.use_hybrid_retrieval:
            return
        if not self._chroma_client or not self._summary_collection:
            return
        
        try:
            self._rebuild_bm25_index(
                collection=self._summary_collection,
                index_key="summaries",
                label="summaries",
            )
            if self._question_collection:
                self._rebuild_bm25_index(
                    collection=self._question_collection,
                    index_key="questions",
                    label="questions",
                )
        except Exception as e:
            logger.warning("BM25 index rebuild failed: {}", e)

    def _rebuild_bm25_index(self, *, collection: Any, index_key: str, label: str) -> None:
        if collection.count() <= 0:
            return
        idx = _BM25Index()
        self._bm25_indexes[index_key] = idx
        page_size = 500
        offset = 0
        while True:
            batch = collection.get(
                limit=page_size,
                offset=offset,
                include=["documents"],
            )
            ids = batch.get("ids", [])
            docs = batch.get("documents", [])
            if not ids:
                break
            for doc_id, doc_text in zip(ids, docs):
                if doc_text:
                    idx.index(doc_id, str(doc_text))
            offset += len(ids)
            if len(ids) < page_size:
                break
        logger.info(
            "BM25 {} index rebuilt: {} docs",
            label,
            len(idx._corpus),
        )

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    val = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(val, dict):
                    records.append(val)
        return records

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def embed_text(self, text: str) -> list[float]:
        clean = text.strip()
        if not clean:
            return []
        if Path(self.config.embedding_model).exists():
            try:
                if self.embedding_model is None:
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer(self.config.embedding_model)
                return self.embedding_model.encode(clean).tolist()
            except Exception as e:
                logger.warning("Embedding model failed, try embedding API")

        if not self.config.embedding_api_key:
            return _hash_embedding(clean)
        headers = {
            "Authorization": f"Bearer {self.config.embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.config.embedding_model, "input": clean}
        url = self.config.embedding_api_base.rstrip("/") + "/embeddings"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            data = resp.json().get("data", [])
            if data and isinstance(data[0], dict) and isinstance(data[0].get("embedding"), list):
                return [float(x) for x in data[0]["embedding"]]
        except Exception as exc:
            logger.warning("Embedding API failed, fallback to hash embedding: {}", exc)
        return _hash_embedding(clean)
    
    def rerank_similarity(self, query: str, doc: str) -> float:
        if self.rerank_model is None:
            from sentence_transformers import CrossEncoder
            self.rerank_model = CrossEncoder(self.config.rerank_model)

        score = self.rerank_model.predict([(query, doc)])[0]

        return score

    def split_into_chunks(self, text: str) -> list[str]:
        clean = text.strip()
        if not clean:
            return []
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: list[str] = []
        buf = ""
        for para in paras or [clean]:
            candidate = f"{buf}\n\n{para}".strip() if buf else para
            if len(candidate) <= self.config.max_chunk_chars:
                buf = candidate
                continue
            if buf:
                if len(buf) >= self.config.min_chunk_chars:
                    chunks.append(buf)
                else:
                    chunks.append(buf[: self.config.max_chunk_chars])
                buf = ""
            for i in range(0, len(para), self.config.max_chunk_chars):
                part = para[i : i + self.config.max_chunk_chars]
                if part:
                    chunks.append(part)
        if buf:
            chunks.append(buf)
        return chunks

    async def upsert_document(
        self,
        doc: dict[str, Any],
        text: str,
        distilled_chunks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        paper_id = str(doc.get("paper_id") or doc.get("id") or "").strip()
        if not paper_id:
            paper_id = hashlib.md5((_normalize_space(doc.get("title", "")) or "paper").encode()).hexdigest()[:12]
        now = datetime.utcnow().isoformat()
        documents = self._read_jsonl(self.docs_file)
        doc_row = {
            "paper_id": paper_id,
            "title": doc.get("title", ""),
            "authors": doc.get("authors", []),
            "abstract": doc.get("abstract", ""),
            "url": doc.get("url", ""),
            "source": doc.get("source", "unknown"),
            "year": doc.get("year"),
            "venue": doc.get("venue", ""),
            "updated_at": now,
        }
        kept_docs = [d for d in documents if d.get("paper_id") != paper_id]
        kept_docs.append(doc_row)
        self._write_jsonl(self.docs_file, kept_docs)

        existing_chunks = self._read_jsonl(self.chunks_file)
        existing_chunks = [c for c in existing_chunks if c.get("paper_id") != paper_id]
        chunk_count = 0
        if distilled_chunks:
            for idx, chunk in enumerate(distilled_chunks):
                chunk_text = str(chunk.get("text", "")).strip()
                if not chunk_text:
                    continue
                emb = await self.embed_text(chunk_text)
                row = {
                    "chunk_id": f"{paper_id}:{idx}",
                    "paper_id": paper_id,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "embedding": emb,
                    "updated_at": now,
                }
                for key in ("section", "kind", "keywords", "claims", "limitations", "source_span"):
                    if key in chunk:
                        row[key] = chunk[key]
                existing_chunks.append(row)
                chunk_count += 1
        else:
            chunks = self.split_into_chunks(text)
            for idx, chunk in enumerate(chunks):
                emb = await self.embed_text(chunk)
                existing_chunks.append(
                    {
                        "chunk_id": f"{paper_id}:{idx}",
                        "paper_id": paper_id,
                        "chunk_index": idx,
                        "text": chunk,
                        "embedding": emb,
                        "updated_at": now,
                    }
                )
            chunk_count = len(chunks)
        self._write_jsonl(self.chunks_file, existing_chunks)
        return {
            "paper_id": paper_id,
            "chunk_count": chunk_count,
            "distilled": bool(distilled_chunks),
        }

    async def upsert_semantic_chunks(
        self,
        doc: dict[str, Any],
        semantic_chunks: list[dict[str, Any]],
        chunk_metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Upsert semantic chunks with summary and hypothetical question embeddings.
        
        This implements the Hypothetical Document Embeddings (HyDE) approach:
        - Store parent document in paper_chunks collection
        - Store summary embedding in paper_summaries collection
        - Store hypothetical question embeddings in paper_questions collection
        
        Args:
            doc: Paper metadata (paper_id, title, url, source, year, etc.)
            semantic_chunks: List of chunks from _split_markdown_semantic:
                [{"section", "heading_level", "text", "heading_path"}]
            chunk_metadata: Optional list of LLM-generated metadata:
                [{"summary", "hypothetical_questions", "keywords"}]
        
        Returns:
            {"paper_id", "chunk_count", "question_count", "success"}
        """
        paper_id = str(doc.get("paper_id") or doc.get("id") or "").strip()
        title = str(doc.get("title", "")).strip()
        if not paper_id:
            paper_id = hashlib.md5((_normalize_space(doc.get("title", "")) or "paper").encode()).hexdigest()[:12]
        
        now = datetime.utcnow().isoformat()
        
        # Update documents.jsonl (metadata)
        documents = self._read_jsonl(self.docs_file)
        doc_row = {
            "paper_id": paper_id,
            "title": title,
            "authors": doc.get("authors", []),
            "abstract": doc.get("abstract", ""),
            "url": doc.get("url", ""),
            "source": doc.get("source", "unknown"),
            "year": doc.get("year"),
            "venue": doc.get("venue", ""),
            "updated_at": now,
        }
        kept_docs = [d for d in documents if d.get("paper_id") != paper_id]
        kept_docs.append(doc_row)
        self._write_jsonl(self.docs_file, kept_docs)
        
        # If Chroma not available, fallback to traditional storage
        if not self._chroma_client or not self._summary_collection:
            logger.warning("Chroma not available, falling back to traditional upsert")
            return await self.upsert_document(doc, "", distilled_chunks=chunk_metadata)
        
        # Clear existing chunks for this paper
        self._delete_paper_from_chroma(paper_id)
        
        chunk_count = 0
        question_count = 0
        
        for idx, chunk in enumerate(semantic_chunks):
            chunk_text = str(chunk.get("text", "")).strip()
            if not chunk_text or len(chunk_text) < self.config.min_chunk_chars:
                continue
            
            chunk_id = f"{paper_id}:{idx}"
            section = chunk.get("section", "content")
            heading_level = chunk.get("heading_level", 0)
            heading_path = chunk.get("heading_path", "content")
            
            # Get or generate metadata for this chunk
            meta = None
            if chunk_metadata and idx < len(chunk_metadata):
                meta = chunk_metadata[idx]
            
            summary = meta.get("summary", "") if meta else chunk_text[:200]
            questions = meta.get("hypothetical_questions", []) if meta else []
            keywords = meta.get("keywords", []) if meta else []
            
            # Store parent chunk (without embedding, just metadata)
            ents = meta.get("entities", []) if meta else []
            clms = meta.get("claims", []) if meta else []
            self._chunk_collection.upsert(
                ids=[chunk_id],
                documents=[chunk_text],
                metadatas=[{
                    "paper_id": paper_id,
                    "paper_title": title,
                    "paper_year": str(doc.get("year") or ""),
                    "paper_source": doc.get("source", "unknown"),
                    "section": section,
                    "heading_level": str(heading_level),
                    "heading_path": heading_path,
                    "keywords": json.dumps(keywords),
                    "entities": json.dumps(ents),
                    "claims": json.dumps(clms),
                    "updated_at": now,
                }],
            )
            chunk_count += 1
            
            # Generate and store summary embedding
            if summary:
                summary_emb = await self.embed_text(summary)
                if summary_emb:
                    self._summary_collection.upsert(
                        ids=[f"{chunk_id}:summary"],
                        embeddings=[summary_emb],
                        documents=[summary],
                        metadatas=[{
                            "paper_id": paper_id,
                            "paper_title": title,
                            "paper_source": doc.get("source", "unknown"),
                            "paper_year": str(doc.get("year") or ""),
                            "chunk_id": chunk_id,
                            "section": section,
                            "heading_path": heading_path,
                            "type": "summary",
                        }],
                    )
                    # BM25 index for hybrid retrieval
                    if self.config.use_hybrid_retrieval:
                        if "summaries" not in self._bm25_indexes:
                            self._bm25_indexes["summaries"] = _BM25Index()
                        self._bm25_indexes["summaries"].index(
                            f"{chunk_id}:summary", summary
                        )
            
            # Generate and store question embeddings
            for q_idx, question in enumerate(questions[:self.config.num_hypothetical_questions]):
                if not question:
                    continue
                q_emb = await self.embed_text(question)
                if q_emb:
                    self._question_collection.upsert(
                        ids=[f"{chunk_id}:q{q_idx}"],
                        embeddings=[q_emb],
                        documents=[question],
                        metadatas=[{
                            "paper_id": paper_id,
                            "paper_title": title,
                            "paper_source": doc.get("source", "unknown"),
                            "paper_year": str(doc.get("year") or ""),
                            "chunk_id": chunk_id,
                            "section": section,
                            "heading_path": heading_path,
                            "question_idx": str(q_idx),
                            "type": "hypothetical_question",
                        }],
                    )
                    question_count += 1
                    # BM25 index for hybrid retrieval
                    if self.config.use_hybrid_retrieval:
                        if "questions" not in self._bm25_indexes:
                            self._bm25_indexes["questions"] = _BM25Index()
                        self._bm25_indexes["questions"].index(
                            f"{chunk_id}:q{q_idx}", question
                        )
        
        logger.info(
            "Upserted semantic chunks: paper_id={}, chunks={}, questions={}, summaries={}",
            paper_id,
            chunk_count,
            question_count,
            chunk_count,
        )
        
        return {
            "paper_id": paper_id,
            "chunk_count": chunk_count,
            "question_count": question_count,
            "success": True,
        }

    def _delete_paper_from_chroma(self, paper_id: str) -> None:
        """Delete all chunks, summaries, and questions for a paper from Chroma."""
        if not self._chroma_client:
            return
        
        try:
            # Get all IDs for this paper
            # Note: Chroma doesn't support delete by metadata filter directly,
            # so we need to query and delete by IDs
            
            # Delete from chunks
            chunk_ids = self._chunk_collection.get(
                where={"paper_id": paper_id},
            ).get("ids", [])
            if chunk_ids:
                self._chunk_collection.delete(ids=chunk_ids)
            
            # Delete from summaries
            summary_ids = self._summary_collection.get(
                where={"paper_id": paper_id},
            ).get("ids", [])
            if summary_ids:
                self._summary_collection.delete(ids=summary_ids)
            
            # Delete from questions
            question_ids = self._question_collection.get(
                where={"paper_id": paper_id},
            ).get("ids", [])
            if question_ids:
                self._question_collection.delete(ids=question_ids)
            
            if summary_ids or question_ids:
                logger.debug("Deleted paper {} from Chroma: chunks={}, summaries={}, questions={}", 
                            paper_id, len(chunk_ids), len(summary_ids), len(question_ids))
            
            # Also clean BM25 indexes
            prefix = paper_id + ":"
            for idx in self._bm25_indexes.values():
                idx.remove_by_prefix(prefix)
        except Exception as e:
            logger.warning("Failed to delete paper {} from Chroma: {}", paper_id, e)

    async def retrieve_by_hypothetical_questions(
        self,
        query: str = "",
        top_k: int | None = None,
        *,
        queries: list[str] | None = None,
        entities: list[dict[str, str]] | None = None,
        per_paper_limit: int = 3,
        search_mode: str = "hybrid",
        where_filter: dict[str, Any] | None = None,
        use_hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks using hypothetical question embeddings (HyDE approach).
        
        This method searches across:
        1. Summary embeddings (paper_summaries collection)
        2. Hypothetical question embeddings (paper_questions collection)
        
        Then retrieves the corresponding parent documents from paper_chunks.
        
        Supports:
        - Single-query (query=) and multi-query (queries=) modes
        - Entity-aware retrieval (entities=) with per-entity metadata filtering
        - Hybrid BM25 + dense RRF fusion (use_hybrid=True)
        
        Args:
            query: User's query text (single-query mode, backward-compatible)
            top_k: Number of results to return
            queries: List of query strings (multi-query mode)
            entities: List of entity dicts with keys paper_id/title/query.
                     When provided, each entity is searched with its own
                     metadata filter and query, then results are merged.
            per_paper_limit: Max chunks per paper to avoid over-concentration
            search_mode: "hybrid" | "questions_only" | "summaries_only"
            where_filter: Optional Chroma where filter for metadata
            use_hybrid: Enable BM25+dense RRF fusion (when BM25 index exists)
        
        Returns:
            List of dicts with chunk info and parent text
        """
        if not self._chroma_client or not self._summary_collection:
            logger.warning("Chroma not available, falling back to traditional retrieve")
            effective_query = query or (queries[0] if queries else "")
            return await self.retrieve(effective_query, top_k, prefer_distilled=True, per_paper_limit=per_paper_limit)
        
        k = max(1, top_k or self.config.retrieval_top_k)
        
        # --- Entity-aware retrieval ---
        if entities:
            return await self._retrieve_by_entities(
                entities=entities,
                query=query,
                queries=queries,
                top_k=k,
                per_paper_limit=per_paper_limit,
                search_mode=search_mode,
                use_hybrid=use_hybrid,
            )
        
        return await self._retrieve_dense_hybrid(
            query=query,
            queries=queries,
            top_k=k,
            per_paper_limit=per_paper_limit,
            search_mode=search_mode,
            where_filter=where_filter,
            use_hybrid=use_hybrid,
        )

    async def _retrieve_by_entities(
        self,
        *,
        entities: list[dict[str, str]],
        query: str,
        queries: list[str] | None = None,
        top_k: int,
        per_paper_limit: int,
        search_mode: str,
        use_hybrid: bool,
    ) -> list[dict[str, Any]]:
        """Entity-aware retrieval: filter by paper_id/title per entity.
        
        Supports multi-query mode via the queries parameter: when queries
        are provided, each entity is searched with all of them (combined
        with the entity-specific query from the entity dict).
        """
        all_entity_results: list[dict[str, Any]] = []
        for ent in entities:
            ent_filter: dict[str, Any] = {}
            pid = (ent.get("paper_id") or "").strip()
            title = (ent.get("title") or "").strip()
            entity_query_list = ent.get("queries") or queries or [query] or []
            
            if pid:
                # arXiv IDs may have version suffixes; match exact paper_id
                ent_filter["paper_id"] = pid
            elif title:
                ent_filter["paper_title"] = {"$contains": title}
            else:
                continue
            
            logger.info(
                "Entity retrieval: filter={} queries={}",
                ent_filter,
                entity_query_list,
            )
            
            ent_results = await self._retrieve_dense_hybrid(
                query=query,
                queries=entity_query_list,
                top_k=top_k,
                per_paper_limit=per_paper_limit,
                search_mode=search_mode,
                where_filter=ent_filter,
                use_hybrid=use_hybrid,
                apply_diversity=False,
            )
            all_entity_results.extend(ent_results)
        
        # Merge: keep best score per chunk_id
        merged: dict[str, dict[str, Any]] = {}
        for r in all_entity_results:
            cid = r.get("chunk_id", "")
            if cid not in merged or r.get("score", 0) > merged[cid].get("score", 0):
                merged[cid] = r
        
        final_entity = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)
        final_entity = _select_diverse(
            final_entity,
            k=min(max(top_k, len(entities)), len(final_entity)),
            per_paper_limit=max(1, per_paper_limit),
            mmr_lambda=self.config.mmr_lambda,
        )
        logger.info(
            "Entity retrieval: {} entities → {} merged → {} final",
            len(entities), len(merged), len(final_entity),
        )
        return final_entity

    async def _retrieve_dense_hybrid(
        self,
        *,
        query: str = "",
        queries: list[str] | None = None,
        top_k: int,
        per_paper_limit: int,
        search_mode: str,
        where_filter: dict[str, Any] | None,
        use_hybrid: bool,
        apply_diversity: bool = True,
    ) -> list[dict[str, Any]]:
        """Core retrieval: dense vectors + optional BM25 RRF fusion."""
        # Build query list: multi-query mode takes precedence
        query_list: list[str] = [query] if query else []
        query_list += list(queries) if queries else []
        if not query_list:
            return []
        
        matched_chunks: dict[str, dict[str, Any]] = {}
        max_candidates = top_k * 10
        bm25_enabled = use_hybrid and self.config.use_hybrid_retrieval
        
        # Accumulate BM25 rankings for RRF
        bm25_rankings: list[list[tuple[str, float]]] = []
        
        for q in query_list:
            q_emb = await self.embed_text(q)
            if not q_emb:
                continue
            
            # --- Dense: search summaries ---
            if search_mode in ("hybrid", "summaries_only"):
                try:
                    summary_results = self._summary_collection.query(
                        query_embeddings=[q_emb],
                        n_results=top_k * 2,
                        include=["metadatas", "documents", "distances"],
                        where=where_filter,
                    )
                    
                    if summary_results and summary_results.get("ids"):
                        ids = summary_results["ids"][0]
                        metas = summary_results["metadatas"][0] if summary_results.get("metadatas") else []
                        docs = summary_results["documents"][0] if summary_results.get("documents") else []
                        distances = summary_results["distances"][0] if summary_results.get("distances") else []
                        
                        for i, id_ in enumerate(ids):
                            meta = metas[i] if i < len(metas) else {}
                            chunk_id = meta.get("chunk_id", "")
                            if not chunk_id:
                                continue
                            
                            dist = distances[i] if i < len(distances) else 1.0
                            score = 1.0 - dist
                            
                            if chunk_id not in matched_chunks or score > matched_chunks[chunk_id].get("score", 0):
                                matched_chunks[chunk_id] = {
                                    "chunk_id": chunk_id,
                                    "paper_id": meta.get("paper_id", ""),
                                    "section": meta.get("section", ""),
                                    "heading_path": meta.get("heading_path", ""),
                                    "score": score,
                                    "matched_by": "summary",
                                    "matched_text": docs[i] if i < len(docs) else "",
                                }
                except Exception as e:
                    logger.warning("Summary search failed: {}", e)
            
            # --- Dense: search hypothetical questions ---
            if search_mode in ("hybrid", "questions_only"):
                try:
                    question_results = self._question_collection.query(
                        query_embeddings=[q_emb],
                        n_results=top_k * 3,
                        include=["metadatas", "documents", "distances"],
                        where=where_filter,
                    )
                    
                    if question_results and question_results.get("ids"):
                        ids = question_results["ids"][0]
                        metas = question_results["metadatas"][0] if question_results.get("metadatas") else []
                        docs = question_results["documents"][0] if question_results.get("documents") else []
                        distances = question_results["distances"][0] if question_results.get("distances") else []
                        
                        for i, id_ in enumerate(ids):
                            meta = metas[i] if i < len(metas) else {}
                            chunk_id = meta.get("chunk_id", "")
                            if not chunk_id:
                                continue
                            
                            dist = distances[i] if i < len(distances) else 1.0
                            score = 1.0 - dist
                            
                            if chunk_id not in matched_chunks or score > matched_chunks[chunk_id].get("score", 0):
                                matched_chunks[chunk_id] = {
                                    "chunk_id": chunk_id,
                                    "paper_id": meta.get("paper_id", ""),
                                    "section": meta.get("section", ""),
                                    "heading_path": meta.get("heading_path", ""),
                                    "score": score,
                                    "matched_by": "question",
                                    "matched_text": docs[i] if i < len(docs) else "",
                                }
                except Exception as e:
                    logger.warning("Question search failed: {}", e)
            
            # --- Sparse: BM25 (if enabled) ---
            if bm25_enabled and not where_filter:
                if search_mode in ("hybrid", "summaries_only"):
                    bm25_idx = self._bm25_indexes.get("summaries")
                    if bm25_idx:
                        bm25_rankings.append(bm25_idx.search(q, top_n=top_k * 2))
                
                if search_mode in ("hybrid", "questions_only"):
                    bm25_idx = self._bm25_indexes.get("questions")
                    if bm25_idx:
                        bm25_rankings.append(bm25_idx.search(q, top_n=top_k * 3))
            
            # Early trim: keep only top max_candidates by score
            if len(matched_chunks) > max_candidates:
                sorted_items = sorted(matched_chunks.values(), key=lambda x: x.get("score", 0), reverse=True)
                matched_chunks = {item["chunk_id"]: item for item in sorted_items[:max_candidates]}
        
        # --- RRF fusion: dense + BM25 ---
        if bm25_enabled and bm25_rankings:
            dense_scores: list[tuple[str, float]] = [
                (cid, info["score"]) for cid, info in matched_chunks.items()
            ]
            dense_scores.sort(key=lambda x: x[1], reverse=True)
            
            all_rankings = [dense_scores[:top_k * 10]] + bm25_rankings
            fused = _rrf_fuse(all_rankings, k=60)
            
            # Normalise RRF scores to the same scale as dense scores before blending.
            # Raw RRF scores are typically 0.01-0.05 whereas dense cosine scores
            # are 0.7-0.95.  Without normalisation the blended score for a perfect
            # match drops from ~0.90 to ~0.45, which can fall below the retrieval
            # quality threshold and cause false "insufficient" verdicts.
            max_dense = max((info.get("score", 0) for info in matched_chunks.values()), default=1.0)
            max_rrf = max(fused.values(), default=1.0)
            scale = max_dense / max(1e-6, max_rrf) if max_rrf > 0 else 1.0

            logger.info("RRF fusion: max_dense={} max_rrf={} scale={}", max_dense, max_rrf, scale)
            
            # Blend: 50% dense + 50% RRF (normalised)
            for chunk_id, fused_score in fused.items():
                prev = matched_chunks.get(chunk_id)
                if prev is not None:
                    prev["score"] = 0.5 * prev.get("score", 0) + 0.5 * (fused_score * scale)
                    prev["matched_by"] = (prev.get("matched_by", "") or "") + "+bm25"
            
            # For chunks that *only* appear in BM25 (not in dense), add them
            # with the normalised RRF score as their base score.
            for chunk_id, fused_score in fused.items():
                if chunk_id not in matched_chunks:
                    matched_chunks[chunk_id] = {
                        "chunk_id": chunk_id,
                        "score": fused_score * scale,
                        "matched_by": "bm25_only",
                    }
        
        # Sort by score and apply per_paper_limit
        sorted_chunks = sorted(matched_chunks.values(), key=lambda x: x.get("score", 0), reverse=True)
        
        if apply_diversity:
            final_chunks = _select_diverse(
                sorted_chunks,
                k=top_k,
                per_paper_limit=max(1, per_paper_limit),
                mmr_lambda=self.config.mmr_lambda,
            )
        else:
            final_chunks = sorted_chunks[:top_k]
        
        # Retrieve parent documents for matched chunks
        if final_chunks:
            chunk_ids = [c["chunk_id"] for c in final_chunks]
            # logger.info("Retrieving parent documents for chunk_ids: {}", chunk_ids[:10])
            # chunk_ids = [cid.rsplit(":", 1)[0] if cid.count(":") == 2 else cid for cid in chunk_ids]  # Extract paper_id:chunk_index
            # chunk_ids = list(set(chunk_ids))  # Unique chunk_ids
            # logger.info("Normalized chunk_ids for retrieval: {}", chunk_ids[:10])
            # visited = set()
            try:
                parent_results = self._chunk_collection.get(
                    ids=chunk_ids,
                    include=["documents", "metadatas"],
                )

                id_to_index = {id_: idx for idx, id_ in enumerate(parent_results["ids"])}
                
                parent_docs = parent_results.get("documents", [])
                parent_metas = parent_results.get("metadatas", [])
                # logger.info("Retrieved parent metadata: {}", parent_metas[:10])
                
                for i, chunk in enumerate(final_chunks):
                    chunk_id = chunk["chunk_id"]
                    # chunk_id = chunk_id.rsplit(":", 1)[0] if chunk_id.count(":") == 2 else chunk_id
                    # if chunk_id in visited:
                    #     continue
                    # visited.add(chunk_id)
                    try:
                        idx = id_to_index.get(chunk_id)
                        chunk["text"] = parent_docs[idx] if idx < len(parent_docs) else ""
                        if idx < len(parent_metas):
                            meta = parent_metas[idx]
                            chunk["chunk_id"] = chunk_id
                            chunk["paper_title"] = meta.get("paper_title", "")
                            chunk["paper_year"] = meta.get("paper_year")
                            chunk["paper_source"] = meta.get("paper_source", "")
                            chunk["keywords"] = json.loads(meta.get("keywords", "[]"))
                            chunk["entities"] = json.loads(meta.get("entities", "[]"))
                            chunk["claims"] = json.loads(meta.get("claims", "[]"))
                    except (ValueError, IndexError):
                        pass
            except Exception as e:
                logger.warning("Failed to retrieve parent documents: {}", e)

        if final_chunks:
            assets_path = self.base_dir / "figures.jsonl"
            assets_by_key = _load_asset_kv(assets_path)
            if assets_by_key:
                for chunk in final_chunks:
                    text = str(chunk.get("text", ""))
                    pid = str(chunk.get("paper_id", ""))
                    if not text or not pid:
                        continue
                    linked: list[dict[str, Any]] = []
                    seen_keys: set[str] = set()
                    for kind, num in _parse_asset_refs(text):
                        key = f"{pid}_{'Table' if kind == 'table' else 'Figure'}_{num}"
                        if key in seen_keys:
                            continue
                        asset = assets_by_key.get(key)
                        if not asset:
                            continue
                        linked.append({"key": key, **asset})
                        seen_keys.add(key)
                    if linked:
                        chunk["linked_assets"] = linked
        
        logger.info(
            "retrieve_by_hypothetical_questions: query='{}' queries={} mode={} matched={} final={}",
            query[:50],
            len(query_list),
            search_mode,
            len(matched_chunks),
            len(final_chunks),
        )

        # logger.info("final chunks: {}", final_chunks[:3])
        
        return final_chunks

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        prefer_distilled: bool = True,
        per_paper_limit: int = 3,
    ) -> list[dict[str, Any]]:
        k = max(1, top_k or self.config.retrieval_top_k)
        chunks = self._read_jsonl(self.chunks_file)
        if not chunks:
            return []
        q_emb = await self.embed_text(query)
        q_tokens = _tokenize(query)
        intent = _parse_query_intent(query)
        docs = {d.get("paper_id"): d for d in self._read_jsonl(self.docs_file)}
        scored: list[dict[str, Any]] = []
        for chunk in chunks:
            emb_score, lexical, metadata, recency, distilled_boost = _compute_chunk_signals(
                chunk,
                q_tokens,
                intent,
                docs,
                q_emb,
                prefer_distilled=prefer_distilled,
                distilled_boost_value=0.08,
            )

            if intent.get("latest"):
                score = 0.45 * emb_score + 0.15 * lexical + 0.20 * metadata + 0.20 * recency + distilled_boost
            else:
                score = 0.55 * emb_score + 0.18 * lexical + 0.22 * metadata + 0.05 * recency + distilled_boost
            scored.append({**chunk, "score": score})
        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        selected = _select_diverse(
            scored,
            k=k,
            per_paper_limit=max(1, per_paper_limit),
            mmr_lambda=self.config.mmr_lambda,
        )
        assets_by_key = _load_asset_kv(self.base_dir / "figures.jsonl")
        if assets_by_key:
            for chunk in selected:
                text = str(chunk.get("text", ""))
                pid = str(chunk.get("paper_id", ""))
                if not text or not pid:
                    continue
                linked: list[dict[str, Any]] = []
                seen_keys: set[str] = set()
                for kind, num in _parse_asset_refs(text):
                    key = f"{pid}_{'Table' if kind == 'table' else 'Figure'}_{num}"
                    if key in seen_keys:
                        continue
                    asset = assets_by_key.get(key)
                    if not asset:
                        continue
                    linked.append({"key": key, **asset})
                    seen_keys.add(key)
                if linked:
                    chunk["linked_assets"] = linked
        return [
            _merge_chunk_doc(item, docs.get(item.get("paper_id"), {}), float(item.get("score", 0.0)))
            for item in selected
        ]

    def load_docs_meta(self) -> dict[str, dict[str, Any]]:
        """Load paper metadata from documents.jsonl, keyed by paper_id.

        Returns:
            {paper_id: {"title", "authors", "abstract", "year", "source"}}
        """
        docs = self._read_jsonl(self.docs_file)
        meta: dict[str, dict[str, Any]] = {}
        for d in docs:
            pid = str(d.get("paper_id", ""))
            if not pid:
                continue
            meta[pid] = {
                "title": d.get("title", ""),
                "authors": d.get("authors", []),
                "abstract": d.get("abstract", ""),
                "year": d.get("year"),
                "source": d.get("source", ""),
            }
        return meta

    def retrieve_lexical(
        self,
        query: str,
        top_k: int | None = None,
        *,
        prefer_distilled: bool = True,
        per_paper_limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Sync-only retrieval for prompt assembly paths."""
        k = max(1, top_k or self.config.retrieval_top_k)
        chunks = self._read_jsonl(self.chunks_file)
        if not chunks:
            return []
        q_tokens = _tokenize(query)
        intent = _parse_query_intent(query)
        docs = {d.get("paper_id"): d for d in self._read_jsonl(self.docs_file)}
        scored: list[dict[str, Any]] = []
        for chunk in chunks:
            _emb, lexical, metadata, recency, distilled_boost = _compute_chunk_signals(
                chunk,
                q_tokens,
                intent,
                docs,
                None,
                prefer_distilled=prefer_distilled,
                distilled_boost_value=0.06,
            )
            if intent.get("latest"):
                score = 0.55 * lexical + 0.25 * metadata + 0.20 * recency + distilled_boost
            else:
                score = 0.65 * lexical + 0.25 * metadata + 0.10 * recency + distilled_boost
            scored.append({**chunk, "score": score})
        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        selected = _select_diverse(
            scored,
            k=k,
            per_paper_limit=max(1, per_paper_limit),
            mmr_lambda=self.config.mmr_lambda,
        )
        assets_by_key = _load_asset_kv(self.base_dir / "figures.jsonl")
        if assets_by_key:
            for chunk in selected:
                text = str(chunk.get("text", ""))
                pid = str(chunk.get("paper_id", ""))
                if not text or not pid:
                    continue
                linked: list[dict[str, Any]] = []
                seen_keys: set[str] = set()
                for kind, num in _parse_asset_refs(text):
                    key = f"{pid}_{'Table' if kind == 'table' else 'Figure'}_{num}"
                    if key in seen_keys:
                        continue
                    asset = assets_by_key.get(key)
                    if not asset:
                        continue
                    linked.append({"key": key, **asset})
                    seen_keys.add(key)
                if linked:
                    chunk["linked_assets"] = linked
        return [
            _merge_chunk_doc(item, docs.get(item.get("paper_id"), {}), float(item.get("score", 0.0)))
            for item in selected
        ]
