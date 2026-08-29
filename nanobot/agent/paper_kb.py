"""Lightweight paper knowledge base and retrieval helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Persistent BM25 sparse retrieval
# ---------------------------------------------------------------------------

class _SQLiteBM25Index:
    """Persistent multi-field BM25 index backed by SQLite FTS5.

    One FTS row represents one parent chunk.  Generated summaries and
    hypothetical questions are fields of that row rather than independent
    pseudo-documents, so they can be weighted without inflating document
    frequency or allowing one chunk to occupy several result slots.
    """

    SCHEMA_VERSION = "1"
    TOKENIZER_VERSION = "mixed-cjk-unigram-bigram-v1"
    FIELD_WEIGHTS = (5.0, 3.0, 1.5, 2.0, 1.0)
    SEARCH_FIELDS = frozenset({"title", "keywords", "summary", "questions", "body"})
    MAX_QUERY_TERMS = 64

    def __init__(
        self,
        path: Path,
        *,
        field_weights: tuple[float, float, float, float, float] | None = None,
    ) -> None:
        self.path = path
        self.field_weights = tuple(
            max(0.0, float(weight))
            for weight in (field_weights or self.FIELD_WEIGHTS)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS lexical_meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        existing_version = self._get_meta("schema_version")
        existing_tokenizer = self._get_meta("tokenizer_version")
        if (
            existing_version not in {None, self.SCHEMA_VERSION}
            or existing_tokenizer not in {None, self.TOKENIZER_VERSION}
        ):
            self._drop_index_schema()

        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS lexical_documents (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                paper_id TEXT NOT NULL,
                title_raw TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                keywords TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                questions TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_lexical_documents_paper_id
                ON lexical_documents(paper_id);
            CREATE INDEX IF NOT EXISTS idx_lexical_documents_title_raw
                ON lexical_documents(title_raw);

            CREATE VIRTUAL TABLE IF NOT EXISTS paper_fts USING fts5(
                title,
                keywords,
                summary,
                questions,
                body,
                content='lexical_documents',
                content_rowid='rowid',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS lexical_documents_ai
            AFTER INSERT ON lexical_documents BEGIN
                INSERT INTO paper_fts(rowid, title, keywords, summary, questions, body)
                VALUES (new.rowid, new.title, new.keywords, new.summary, new.questions, new.body);
            END;

            CREATE TRIGGER IF NOT EXISTS lexical_documents_ad
            AFTER DELETE ON lexical_documents BEGIN
                INSERT INTO paper_fts(
                    paper_fts, rowid, title, keywords, summary, questions, body
                ) VALUES (
                    'delete', old.rowid, old.title, old.keywords,
                    old.summary, old.questions, old.body
                );
            END;

            CREATE TRIGGER IF NOT EXISTS lexical_documents_au
            AFTER UPDATE ON lexical_documents BEGIN
                INSERT INTO paper_fts(
                    paper_fts, rowid, title, keywords, summary, questions, body
                ) VALUES (
                    'delete', old.rowid, old.title, old.keywords,
                    old.summary, old.questions, old.body
                );
                INSERT INTO paper_fts(rowid, title, keywords, summary, questions, body)
                VALUES (new.rowid, new.title, new.keywords, new.summary, new.questions, new.body);
            END;
            """
        )
        weights = ", ".join(str(weight) for weight in self.field_weights)
        self._connection.execute(
            "INSERT INTO paper_fts(paper_fts, rank) VALUES('rank', ?)",
            (f"bm25({weights})",),
        )
        self._set_meta("schema_version", self.SCHEMA_VERSION)
        self._set_meta("tokenizer_version", self.TOKENIZER_VERSION)
        self._connection.commit()

    def _drop_index_schema(self) -> None:
        self._connection.executescript(
            """
            DROP TRIGGER IF EXISTS lexical_documents_ai;
            DROP TRIGGER IF EXISTS lexical_documents_ad;
            DROP TRIGGER IF EXISTS lexical_documents_au;
            DROP TABLE IF EXISTS paper_fts;
            DROP TABLE IF EXISTS lexical_documents;
            DELETE FROM lexical_meta;
            """
        )

    def _get_meta(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM lexical_meta WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row[0]) if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO lexical_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            return " ".join(str(item) for item in value if item)
        return str(value or "")

    @classmethod
    def _indexed_text(cls, value: Any) -> str:
        return " ".join(tokenize_text(cls._as_text(value)))

    @classmethod
    def _prepare_row(cls, row: dict[str, Any]) -> tuple[str, ...]:
        title_raw = cls._as_text(row.get("title", ""))
        return (
            str(row.get("chunk_id", "")),
            str(row.get("paper_id", "")),
            title_raw,
            cls._indexed_text(title_raw),
            cls._indexed_text(row.get("keywords", "")),
            cls._indexed_text(row.get("summary", "")),
            cls._indexed_text(row.get("questions", "")),
            cls._indexed_text(row.get("body", "")),
        )

    def replace_paper(
        self,
        paper_id: str,
        rows: list[dict[str, Any]],
        *,
        source_mtime_ns: int | None = None,
    ) -> None:
        prepared = [self._prepare_row(row) for row in rows if row.get("chunk_id")]
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM lexical_documents WHERE paper_id = ?",
                (paper_id,),
            )
            self._connection.executemany(
                "INSERT INTO lexical_documents("
                "chunk_id, paper_id, title_raw, title, keywords, summary, questions, body"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                prepared,
            )
            if source_mtime_ns is not None:
                self._set_meta("source_mtime_ns", str(source_mtime_ns))

    def rebuild(
        self,
        rows: list[dict[str, Any]],
        *,
        source_mtime_ns: int,
    ) -> None:
        prepared = [self._prepare_row(row) for row in rows if row.get("chunk_id")]
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM lexical_documents")
            self._connection.executemany(
                "INSERT INTO lexical_documents("
                "chunk_id, paper_id, title_raw, title, keywords, summary, questions, body"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                prepared,
            )
            self._set_meta("source_mtime_ns", str(source_mtime_ns))

    def needs_sync(self, source_mtime_ns: int) -> bool:
        with self._lock:
            return self._get_meta("source_mtime_ns") != str(source_mtime_ns)

    @classmethod
    def _match_query(cls, query: str, fields: tuple[str, ...] | None) -> str:
        tokens = list(dict.fromkeys(tokenize_text(query)))
        if not tokens:
            return ""
        if len(tokens) > cls.MAX_QUERY_TERMS:
            multi_character = [token for token in tokens if len(token) > 1]
            single_character = [token for token in tokens if len(token) == 1]
            tokens = (multi_character + single_character)[:cls.MAX_QUERY_TERMS]
        terms = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        if fields:
            valid_fields = [field for field in fields if field in cls.SEARCH_FIELDS]
            if valid_fields:
                return "{" + " ".join(valid_fields) + "} : (" + terms + ")"
        return "(" + terms + ")"

    def search(
        self,
        query: str,
        *,
        top_n: int = 20,
        fields: tuple[str, ...] | None = None,
        paper_id: str = "",
        title_contains: str = "",
    ) -> list[tuple[str, float]]:
        match_query = self._match_query(query, fields)
        if not match_query:
            return []

        filters: list[str] = []
        params: list[Any] = [match_query]
        if paper_id:
            filters.append("d.paper_id = ?")
            params.append(paper_id)
        if title_contains:
            filters.append("LOWER(d.title_raw) LIKE ?")
            params.append(f"%{title_contains.lower()}%")
        params.append(max(1, top_n))
        filter_sql = "" if not filters else " AND " + " AND ".join(filters)
        sql = (
            "SELECT d.chunk_id, -rank AS score "
            "FROM paper_fts "
            "JOIN lexical_documents d ON d.rowid = paper_fts.rowid "
            "WHERE paper_fts MATCH ?"
            f"{filter_sql} ORDER BY rank LIMIT ?"
        )
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [(str(row["chunk_id"]), float(row["score"])) for row in rows]

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM lexical_documents"
            ).fetchone()
        return int(row[0]) if row else 0

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            source_mtime_ns = self._get_meta("source_mtime_ns")
        return {
            "backend": "sqlite_fts5",
            "path": str(self.path),
            "document_count": self.count(),
            "schema_version": self.SCHEMA_VERSION,
            "tokenizer_version": self.TOKENIZER_VERSION,
            "field_weights": {
                field: weight
                for field, weight in zip(
                    ("title", "keywords", "summary", "questions", "body"),
                    self.field_weights,
                )
            },
            "source_mtime_ns": int(source_mtime_ns) if source_mtime_ns else None,
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _weighted_rrf_fuse(
    families: list[tuple[list[list[tuple[str, float]]], float]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Fuse ranked-list families without giving larger families more weight."""
    available = [
        ([ranking for ranking in rankings if ranking], max(0.0, weight))
        for rankings, weight in families
        if any(rankings)
    ]
    active = [(rankings, weight) for rankings, weight in available if weight > 0]
    if not active and available:
        active = [(rankings, 1.0) for rankings, _weight in available]
    total_weight = sum(weight for _, weight in active)
    if total_weight <= 0:
        return {}

    fused: dict[str, float] = defaultdict(float)
    for rankings, family_weight in active:
        per_ranking_weight = family_weight / total_weight / len(rankings)
        for ranking in rankings:
            seen: set[str] = set()
            for rank, (doc_id, _score) in enumerate(ranking, start=1):
                if not doc_id or doc_id in seen:
                    continue
                fused[doc_id] += per_ranking_weight / (k + rank)
                seen.add(doc_id)
    return dict(fused)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


_MIXED_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)


def tokenize_text(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text without requiring a segmenter.

    English and numeric spans are lower-cased.  Contiguous CJK spans produce
    both unigrams and overlapping bigrams: unigrams preserve recall for short
    queries while bigrams add enough phrase precision for BM25 and the lexical
    hash fallback.
    """
    tokens: list[str] = []
    for segment in _MIXED_TOKEN_PATTERN.findall((text or "").lower()):
        if segment.isascii():
            if len(segment) > 1:
                tokens.append(segment)
            continue

        characters = list(segment)
        tokens.extend(characters)
        tokens.extend(
            "".join(characters[index:index + 2])
            for index in range(len(characters) - 1)
        )
    return tokens


def _split_tokens(text: str) -> list[str]:
    return tokenize_text(text)


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
        eligible: list[tuple[int, dict[str, Any]]] = []
        for i, item in enumerate(remaining):
            pid = str(item.get("paper_id", ""))
            if pid and per_paper_count.get(pid, 0) >= per_paper_limit:
                continue
            eligible.append((i, item))

        if not eligible:
            break

        best_idx = eligible[0][0]
        best_mmr = float("-inf")
        for i, item in eligible:
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
    embedding_model: str = "text-embedding-3-small"
    embedding_fallback: str = "hash"
    embedding_batch_size: int = 64
    rerank_model: str = ""
    retrieval_top_k: int = 5
    max_chunk_chars: int = 4096
    min_chunk_chars: int = 300
    # New config for hypothetical question retrieval
    num_hypothetical_questions: int = 3
    enable_hypothetical_retrieval: bool = True
    chroma_persist_dir: str = ""  # Will default to {workspace}/kb/chroma
    mmr_lambda: float = 0.7  # Relevance-diversity trade-off for MMR selection (1.0=relevance-only)
    use_hybrid_retrieval: bool = True  # Enable BM25+dense RRF hybrid retrieval
    rrf_k: int = 60
    dense_rrf_weight: float = 0.5
    sparse_rrf_weight: float = 0.5
    bm25_title_weight: float = 5.0
    bm25_keywords_weight: float = 3.0
    bm25_summary_weight: float = 1.5
    bm25_questions_weight: float = 2.0
    bm25_body_weight: float = 1.0


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
        self._jsonl_lock = asyncio.Lock()
        self._embedding_model_lock = asyncio.Lock()
        self._rerank_model_lock = asyncio.Lock()
        self._embedding_backend = self._resolve_embedding_backend()
        self._embedding_last_error = ""
        self._embedding_dimension: int | None = (
            256 if self._embedding_backend == "hash_lexical" else None
        )
        if self._embedding_backend == "hash_lexical":
            logger.warning(
                "No semantic embedding backend is configured; using the explicit "
                "hash_lexical fallback. Retrieval is degraded to lexical similarity."
            )
        
        # Initialize Chroma collections for semantic retrieval
        self._chroma_client = None
        self._summary_collection = None
        self._question_collection = None
        self._chunk_collection = None
        self._lexical_index: _SQLiteBM25Index | None = None
        self._lexical_last_error = ""
        self._init_lexical_index()
        self._init_chroma_collections()

    def _chunks_source_mtime_ns(self) -> int:
        try:
            return self.chunks_file.stat().st_mtime_ns
        except OSError:
            return 0

    def _lexical_rows_from_jsonl(self) -> list[dict[str, Any]]:
        docs = {
            str(row.get("paper_id", "")): row
            for row in self._read_jsonl(self.docs_file)
            if row.get("paper_id")
        }
        rows: list[dict[str, Any]] = []
        for chunk in self._read_jsonl(self.chunks_file):
            chunk_id = str(chunk.get("chunk_id", ""))
            paper_id = str(chunk.get("paper_id", ""))
            if not chunk_id or not paper_id:
                continue
            doc = docs.get(paper_id, {})
            rows.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "title": doc.get("title", chunk.get("paper_title", "")),
                "keywords": chunk.get("keywords", []),
                "summary": chunk.get("summary", ""),
                "questions": chunk.get("hypothetical_questions", []),
                "body": chunk.get("text", ""),
            })
        return rows

    def _init_lexical_index(self) -> None:
        """Open the persistent sparse index and repair it from canonical JSONL."""
        try:
            self._lexical_index = _SQLiteBM25Index(
                self.base_dir / "lexical.db",
                field_weights=(
                    self.config.bm25_title_weight,
                    self.config.bm25_keywords_weight,
                    self.config.bm25_summary_weight,
                    self.config.bm25_questions_weight,
                    self.config.bm25_body_weight,
                ),
            )
            source_mtime_ns = self._chunks_source_mtime_ns()
            if self._lexical_index.needs_sync(source_mtime_ns):
                rows = self._lexical_rows_from_jsonl()
                self._lexical_index.rebuild(
                    rows,
                    source_mtime_ns=source_mtime_ns,
                )
                logger.info(
                    "SQLite FTS5 index rebuilt from JSONL: {} parent chunks",
                    len(rows),
                )
            self._lexical_last_error = ""
        except Exception as exc:
            self._lexical_last_error = str(exc)
            self._lexical_index = None
            logger.warning(
                "SQLite FTS5 initialization failed; sparse retrieval disabled: {}",
                exc,
            )

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
        except Exception as e:
            logger.warning("Chroma initialization failed: {}, falling back to JSONL-only mode", e)
            self._chroma_client = None
            self._summary_collection = None
            self._question_collection = None
            self._chunk_collection = None

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
        """Atomically replace a JSONL file after fully writing a sibling temp file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_name = f.name
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            if tmp_name:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except OSError:
                    pass

    async def _persist_paper_rows(
        self,
        *,
        doc_row: dict[str, Any],
        chunk_rows: list[dict[str, Any]],
    ) -> None:
        """Idempotently replace one paper's JSONL rows under an in-process lock."""
        paper_id = str(doc_row.get("paper_id", ""))
        async with self._jsonl_lock:
            documents = [
                row for row in self._read_jsonl(self.docs_file)
                if str(row.get("paper_id", "")) != paper_id
            ]
            chunks = [
                row for row in self._read_jsonl(self.chunks_file)
                if str(row.get("paper_id", "")) != paper_id
            ]
            documents.append(doc_row)
            chunks.extend(chunk_rows)
            self._write_jsonl(self.docs_file, documents)
            self._write_jsonl(self.chunks_file, chunks)

    async def _replace_lexical_paper(
        self,
        *,
        doc_row: dict[str, Any],
        chunk_rows: list[dict[str, Any]],
    ) -> None:
        """Replace one paper in the derived FTS index without blocking the loop."""
        if self._lexical_index is None:
            return
        paper_id = str(doc_row.get("paper_id", ""))
        rows = [{
            "chunk_id": row.get("chunk_id", ""),
            "paper_id": paper_id,
            "title": doc_row.get("title", ""),
            "keywords": row.get("keywords", []),
            "summary": row.get("summary", ""),
            "questions": row.get("hypothetical_questions", []),
            "body": row.get("text", ""),
        } for row in chunk_rows]
        try:
            await asyncio.to_thread(
                self._lexical_index.replace_paper,
                paper_id,
                rows,
                source_mtime_ns=self._chunks_source_mtime_ns(),
            )
            self._lexical_last_error = ""
        except Exception as exc:
            # JSONL is canonical.  Leave its mtime marker unmatched so the
            # next process start repairs the derived FTS index automatically.
            self._lexical_last_error = str(exc)
            logger.warning("SQLite FTS5 update failed for paper {}: {}", paper_id, exc)

    def get_lexical_status(self) -> dict[str, Any]:
        if self._lexical_index is None:
            return {
                "backend": "unavailable",
                "document_count": None,
                "degraded": True,
                "reason": self._lexical_last_error or "sqlite_fts5_unavailable",
            }
        try:
            status = self._lexical_index.get_status()
        except Exception as exc:
            self._lexical_last_error = str(exc)
            return {
                "backend": "sqlite_fts5",
                "document_count": None,
                "degraded": True,
                "reason": str(exc),
            }
        status["degraded"] = bool(self._lexical_last_error)
        status["reason"] = self._lexical_last_error
        return status

    def _resolve_embedding_backend(self) -> str:
        """Choose one stable backend for the lifetime of this KB instance."""
        model_path = Path(self.config.embedding_model).expanduser()
        if self.config.embedding_model and model_path.exists():
            return "local_sentence_transformer"
        if self.config.embedding_api_key:
            return "openai_compatible_api"
        if self.config.embedding_fallback == "hash":
            return "hash_lexical"
        return "unavailable"

    def get_embedding_status(self) -> dict[str, Any]:
        """Return observable backend health without exposing credentials."""
        reason = ""
        if self._embedding_backend == "hash_lexical":
            reason = "no_semantic_embedding_backend_configured"
        elif self._embedding_backend == "unavailable":
            reason = "embedding_backend_unavailable"
        if self._embedding_last_error:
            reason = self._embedding_last_error
        active_model = (
            self.config.embedding_model
            if self._embedding_backend in {
                "local_sentence_transformer",
                "openai_compatible_api",
            }
            else "hash-lexical-256" if self._embedding_backend == "hash_lexical" else None
        )
        return {
            "backend": self._embedding_backend,
            "model": active_model,
            "configured_model": self.config.embedding_model,
            "batch_size": max(1, self.config.embedding_batch_size),
            "dimension": self._embedding_dimension,
            "degraded": self._embedding_backend in {"hash_lexical", "unavailable"}
            or bool(self._embedding_last_error),
            "reason": reason,
        }

    async def _embed_api_batch(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.config.embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.config.embedding_model, "input": texts}
        url = self.config.embedding_api_base.rstrip("/") + "/embeddings"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json().get("data", [])
        ordered: list[list[float] | None] = [None] * len(texts)
        for position, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                continue
            try:
                index = int(item.get("index", position))
            except (TypeError, ValueError):
                index = position
            if 0 <= index < len(ordered):
                ordered[index] = [float(value) for value in item["embedding"]]
        if any(vector is None for vector in ordered):
            raise RuntimeError(
                f"Embedding API returned {len(data)} vectors for {len(texts)} inputs"
            )
        return [vector for vector in ordered if vector is not None]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in stable, bounded batches while preserving input order.

        Once an API or local semantic backend is selected, failures are raised
        instead of silently switching to hash vectors with a different
        dimension.  Hash fallback is used only when explicitly configured as
        the backend at initialization time.
        """
        if not texts:
            return []

        results: list[list[float]] = [[] for _ in texts]
        nonempty = [(index, text.strip()) for index, text in enumerate(texts) if text.strip()]
        if not nonempty:
            return results

        backend = self._embedding_backend
        batch_size = max(1, self.config.embedding_batch_size)
        try:
            for start in range(0, len(nonempty), batch_size):
                batch = nonempty[start:start + batch_size]
                batch_texts = [text for _, text in batch]

                if backend == "hash_lexical":
                    vectors = [_hash_embedding(text) for text in batch_texts]
                elif backend == "local_sentence_transformer":
                    if self.embedding_model is None:
                        async with self._embedding_model_lock:
                            if self.embedding_model is None:
                                from sentence_transformers import SentenceTransformer

                                self.embedding_model = await asyncio.to_thread(
                                    SentenceTransformer,
                                    str(Path(self.config.embedding_model).expanduser()),
                                )
                    encoded = await asyncio.to_thread(
                        self.embedding_model.encode,
                        batch_texts,
                        normalize_embeddings=True,
                    )
                    vectors = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
                elif backend == "openai_compatible_api":
                    vectors = await self._embed_api_batch(batch_texts)
                else:
                    raise RuntimeError(
                        "No embedding backend is available. Configure embeddingApiKey, "
                        "a local embeddingModel path, or embeddingFallback='hash'."
                    )

                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"Embedding backend returned {len(vectors)} vectors for {len(batch)} inputs"
                    )
                normalized_vectors: list[list[float]] = []
                for vector in vectors:
                    normalized = [float(value) for value in vector]
                    if not normalized:
                        raise RuntimeError("Embedding backend returned an empty vector")
                    if not all(math.isfinite(value) for value in normalized):
                        raise RuntimeError("Embedding backend returned a non-finite vector")
                    if self._embedding_dimension is None:
                        self._embedding_dimension = len(normalized)
                    elif len(normalized) != self._embedding_dimension:
                        raise RuntimeError(
                            "Embedding dimension changed from "
                            f"{self._embedding_dimension} to {len(normalized)}"
                        )
                    normalized_vectors.append(normalized)

                for (original_index, _), vector in zip(batch, normalized_vectors):
                    results[original_index] = vector

            self._embedding_last_error = ""
            return results
        except Exception as exc:
            self._embedding_last_error = f"{backend}_failed: {exc}"
            raise RuntimeError(self._embedding_last_error) from exc

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def _upsert_chroma_batches(
        self,
        collection: Any,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert a collection without blocking the event loop per record."""
        batch_size = max(1, min(500, self.config.embedding_batch_size * 4))
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            await asyncio.to_thread(
                collection.upsert,
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
    
    def rerank_similarity(self, query: str, doc: str) -> float:
        if not self.config.rerank_model:
            raise RuntimeError("No cross-encoder rerank model is configured")
        if self.rerank_model is None:
            from sentence_transformers import CrossEncoder
            self.rerank_model = CrossEncoder(self.config.rerank_model)

        score = self.rerank_model.predict([(query, doc)])[0]

        return score

    async def rerank_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run the configured Cross-Encoder once for a batch of query/document pairs."""
        if not pairs:
            return []
        if not self.config.rerank_model:
            raise RuntimeError("No cross-encoder rerank model is configured")
        if self.rerank_model is None:
            async with self._rerank_model_lock:
                if self.rerank_model is None:
                    from sentence_transformers import CrossEncoder

                    self.rerank_model = await asyncio.to_thread(
                        CrossEncoder,
                        self.config.rerank_model,
                    )
        raw_scores = await asyncio.to_thread(self.rerank_model.predict, pairs)
        values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        normalized: list[float] = []
        for raw_score in values:
            score = float(raw_score)
            if not math.isfinite(score):
                score = 0.0
            elif not 0.0 <= score <= 1.0:
                # Cross-Encoders commonly return logits; map them to a stable
                # relevance range before mixing with RRF/recency features.
                score = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))
            normalized.append(score)
        if len(normalized) != len(pairs):
            raise RuntimeError(
                f"Cross-encoder returned {len(normalized)} scores for {len(pairs)} pairs"
            )
        return normalized

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
        new_chunks: list[dict[str, Any]] = []
        if distilled_chunks:
            for idx, chunk in enumerate(distilled_chunks):
                chunk_text = str(chunk.get("text", "")).strip()
                if not chunk_text:
                    continue
                row = {
                    "chunk_id": f"{paper_id}:{idx}",
                    "paper_id": paper_id,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "updated_at": now,
                }
                for key in ("section", "kind", "keywords", "claims", "limitations", "source_span"):
                    if key in chunk:
                        row[key] = chunk[key]
                new_chunks.append(row)
        else:
            chunks = self.split_into_chunks(text)
            for idx, chunk in enumerate(chunks):
                new_chunks.append(
                    {
                        "chunk_id": f"{paper_id}:{idx}",
                        "paper_id": paper_id,
                        "chunk_index": idx,
                        "text": chunk,
                        "updated_at": now,
                    }
                )
        embeddings = await self.embed_texts([str(row["text"]) for row in new_chunks])
        for row, embedding in zip(new_chunks, embeddings):
            row["embedding"] = embedding
        await self._persist_paper_rows(doc_row=doc_row, chunk_rows=new_chunks)
        await self._replace_lexical_paper(doc_row=doc_row, chunk_rows=new_chunks)
        embedding_status = self.get_embedding_status()
        lexical_status = self.get_lexical_status()
        return {
            "paper_id": paper_id,
            "chunk_count": len(new_chunks),
            "distilled": bool(distilled_chunks),
            "storage_backend": "jsonl",
            "embedding": embedding_status,
            "lexical": lexical_status,
            "degraded": bool(embedding_status["degraded"] or lexical_status["degraded"]),
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

        # Build the parent rows once and always persist them to JSONL.  JSONL is
        # the durable fallback/export representation; Chroma is the semantic index.
        parent_rows: list[dict[str, Any]] = []
        for idx, chunk in enumerate(semantic_chunks):
            chunk_text = str(chunk.get("text", "")).strip()
            if not chunk_text or len(chunk_text) < self.config.min_chunk_chars:
                continue
            meta = chunk_metadata[idx] if chunk_metadata and idx < len(chunk_metadata) else {}
            parent_rows.append({
                "chunk_id": f"{paper_id}:{idx}",
                "paper_id": paper_id,
                "chunk_index": idx,
                "text": chunk_text,
                "section": chunk.get("section", "content"),
                "heading_level": chunk.get("heading_level", 0),
                "heading_path": chunk.get("heading_path", "content"),
                "kind": "semantic_chunk",
                "summary": meta.get("summary", chunk_text[:200]),
                "hypothetical_questions": meta.get("hypothetical_questions", []),
                "keywords": meta.get("keywords", []),
                "entities": meta.get("entities", []),
                "claims": meta.get("claims", []),
                "updated_at": now,
            })

        parent_embeddings = await self.embed_texts(
            [str(row["text"]) for row in parent_rows]
        )
        for row, embedding in zip(parent_rows, parent_embeddings):
            row["embedding"] = embedding

        await self._persist_paper_rows(doc_row=doc_row, chunk_rows=parent_rows)
        await self._replace_lexical_paper(doc_row=doc_row, chunk_rows=parent_rows)
        embedding_status = self.get_embedding_status()
        lexical_status = self.get_lexical_status()

        # If Chroma is unavailable, parent-chunk retrieval remains functional.
        if not self._chroma_client or not self._summary_collection:
            logger.warning("Chroma not available, falling back to traditional upsert")
            return {
                "paper_id": paper_id,
                "chunk_count": len(parent_rows),
                "question_count": 0,
                "success": True,
                "storage_backend": "jsonl",
                "degraded": True,
                "embedding": embedding_status,
                "lexical": lexical_status,
                "degradation_reasons": [
                    reason for reason in (
                        "chroma_unavailable",
                        str(embedding_status.get("reason", "")),
                        str(lexical_status.get("reason", "")),
                    ) if reason
                ],
            }

        # Clear existing chunks for this paper
        self._delete_paper_from_chroma(paper_id)

        parent_ids: list[str] = []
        parent_documents: list[str] = []
        parent_metadatas: list[dict[str, Any]] = []
        summary_records: list[tuple[str, str, dict[str, Any]]] = []
        question_records: list[tuple[str, str, dict[str, Any]]] = []

        for row in parent_rows:
            chunk_id = str(row["chunk_id"])
            chunk_text = str(row["text"])
            section = str(row.get("section", "content"))
            heading_level = row.get("heading_level", 0)
            heading_path = str(row.get("heading_path", "content"))
            keywords = row.get("keywords", []) or []
            entities = row.get("entities", []) or []
            claims = row.get("claims", []) or []

            parent_ids.append(chunk_id)
            parent_documents.append(chunk_text)
            parent_metadatas.append({
                "paper_id": paper_id,
                "paper_title": title,
                "paper_year": str(doc.get("year") or ""),
                "paper_source": doc.get("source", "unknown"),
                "section": section,
                "heading_level": str(heading_level),
                "heading_path": heading_path,
                "keywords": json.dumps(keywords),
                "entities": json.dumps(entities),
                "claims": json.dumps(claims),
                "updated_at": now,
            })

            summary = str(row.get("summary", "")).strip()
            if summary:
                summary_records.append((
                    f"{chunk_id}:summary",
                    summary,
                    {
                        "paper_id": paper_id,
                        "paper_title": title,
                        "paper_source": doc.get("source", "unknown"),
                        "paper_year": str(doc.get("year") or ""),
                        "chunk_id": chunk_id,
                        "section": section,
                        "heading_path": heading_path,
                        "type": "summary",
                    },
                ))

            questions = row.get("hypothetical_questions", []) or []
            for question_index, question in enumerate(
                questions[:self.config.num_hypothetical_questions]
            ):
                if not question:
                    continue
                question_records.append((
                    f"{chunk_id}:q{question_index}",
                    str(question),
                    {
                        "paper_id": paper_id,
                        "paper_title": title,
                        "paper_source": doc.get("source", "unknown"),
                        "paper_year": str(doc.get("year") or ""),
                        "chunk_id": chunk_id,
                        "section": section,
                        "heading_path": heading_path,
                        "question_idx": str(question_index),
                        "type": "hypothetical_question",
                    },
                ))

        if parent_ids:
            await self._upsert_chroma_batches(
                self._chunk_collection,
                ids=parent_ids,
                embeddings=[list(row.get("embedding", [])) for row in parent_rows],
                documents=parent_documents,
                metadatas=parent_metadatas,
            )

        auxiliary_records = summary_records + question_records
        auxiliary_embeddings = await self.embed_texts(
            [record[1] for record in auxiliary_records]
        )
        summary_embeddings = auxiliary_embeddings[:len(summary_records)]
        question_embeddings = auxiliary_embeddings[len(summary_records):]

        if summary_records:
            await self._upsert_chroma_batches(
                self._summary_collection,
                ids=[record[0] for record in summary_records],
                embeddings=summary_embeddings,
                documents=[record[1] for record in summary_records],
                metadatas=[record[2] for record in summary_records],
            )
        if question_records:
            await self._upsert_chroma_batches(
                self._question_collection,
                ids=[record[0] for record in question_records],
                embeddings=question_embeddings,
                documents=[record[1] for record in question_records],
                metadatas=[record[2] for record in question_records],
            )

        question_count = len(question_records)

        logger.info(
            "Upserted semantic chunks: paper_id={}, chunks={}, questions={}, summaries={}",
            paper_id,
            len(parent_rows),
            question_count,
            len(parent_rows),
        )

        return {
            "paper_id": paper_id,
            "chunk_count": len(parent_rows),
            "question_count": question_count,
            "success": True,
            "storage_backend": "chroma+jsonl",
            "degraded": bool(embedding_status["degraded"] or lexical_status["degraded"]),
            "embedding": embedding_status,
            "lexical": lexical_status,
            "degradation_reasons": [
                reason for reason in (
                    str(embedding_status.get("reason", "")),
                    str(lexical_status.get("reason", "")),
                ) if reason
            ],
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
            
        except Exception as e:
            logger.warning("Failed to delete paper {} from Chroma: {}", paper_id, e)

    async def _retrieve_jsonl_multiquery(
        self,
        *,
        query: str,
        queries: list[str] | None,
        entities: list[dict[str, str]] | None,
        top_k: int,
        per_paper_limit: int,
    ) -> list[dict[str, Any]]:
        """Retrieve and merge JSONL results for backend fallback/recovery."""
        query_list = [item for item in ([query] + list(queries or [])) if item]
        if not query_list:
            return []

        merged: dict[str, dict[str, Any]] = {}
        for fallback_query in dict.fromkeys(query_list):
            results = await self.retrieve(
                fallback_query,
                max(top_k * 2, top_k),
                prefer_distilled=True,
                per_paper_limit=max(per_paper_limit, top_k),
            )
            for item in results:
                chunk_id = str(item.get("chunk_id", ""))
                current = merged.get(chunk_id, {})
                if chunk_id and float(item.get("score", 0.0)) > float(current.get("score", -1.0)):
                    merged[chunk_id] = item

        fallback_results = list(merged.values())
        if entities:
            ids = {str(item.get("paper_id", "")).strip() for item in entities if item.get("paper_id")}
            titles = {str(item.get("title", "")).strip().lower() for item in entities if item.get("title")}
            fallback_results = [
                item for item in fallback_results
                if (not ids and not titles)
                or str(item.get("paper_id", "")) in ids
                or any(title in str(item.get("title", "")).lower() for title in titles)
            ]

        fallback_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return _select_diverse(
            fallback_results,
            k=top_k,
            per_paper_limit=max(1, per_paper_limit),
            mmr_lambda=self.config.mmr_lambda,
        )

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
        1. Parent chunk embeddings (paper_chunks collection)
        2. Summary embeddings (paper_summaries collection)
        3. Hypothetical question embeddings (paper_questions collection)
        
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
            search_mode: "hybrid" | "questions_only" | "summaries_only" |
                "chunks_only"
            where_filter: Optional Chroma where filter for metadata
            use_hybrid: Enable BM25+dense RRF fusion (when BM25 index exists)
        
        Returns:
            List of dicts with chunk info and parent text
        """
        valid_modes = {"hybrid", "questions_only", "summaries_only", "chunks_only"}
        if search_mode not in valid_modes:
            raise ValueError(
                f"Unsupported search_mode={search_mode!r}; expected one of {sorted(valid_modes)}"
            )

        k = max(1, top_k or self.config.retrieval_top_k)
        mode_collections = {
            "hybrid": (self._chunk_collection, self._summary_collection, self._question_collection),
            "chunks_only": (self._chunk_collection,),
            "summaries_only": (self._summary_collection,),
            "questions_only": (self._question_collection,),
        }
        if not self._chroma_client or not any(
            collection is not None for collection in mode_collections[search_mode]
        ):
            logger.warning("Chroma not available, falling back to traditional retrieve")
            return await self._retrieve_jsonl_multiquery(
                query=query,
                queries=queries,
                entities=entities,
                top_k=k,
                per_paper_limit=per_paper_limit,
            )
        
        # --- Entity-aware retrieval ---
        if entities:
            entity_results = await self._retrieve_by_entities(
                entities=entities,
                query=query,
                queries=queries,
                top_k=k,
                per_paper_limit=per_paper_limit,
                search_mode=search_mode,
                use_hybrid=use_hybrid,
            )
            if entity_results or not self.chunks_file.exists():
                return entity_results
            logger.warning("Chroma entity lookup returned no chunks; retrying against JSONL fallback")
            return await self._retrieve_jsonl_multiquery(
                query=query,
                queries=queries,
                entities=entities,
                top_k=k,
                per_paper_limit=per_paper_limit,
            )
        
        results = await self._retrieve_dense_hybrid(
            query=query,
            queries=queries,
            top_k=k,
            per_paper_limit=per_paper_limit,
            search_mode=search_mode,
            where_filter=where_filter,
            use_hybrid=use_hybrid,
        )
        if results or not self.chunks_file.exists():
            return results

        logger.warning("Chroma returned no paper chunks; retrying against JSONL fallback")
        return await self._retrieve_jsonl_multiquery(
            query=query,
            queries=queries,
            entities=entities,
            top_k=k,
            per_paper_limit=per_paper_limit,
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
            entity_query_list: list[str] = []
            seen_entity_queries: set[str] = set()
            for entity_query in [
                query,
                *(queries or []),
                *(ent.get("queries") or []),
            ]:
                normalized_query = re.sub(r"\s+", " ", str(entity_query or "")).strip()
                query_key = normalized_query.casefold()
                if normalized_query and query_key not in seen_entity_queries:
                    seen_entity_queries.add(query_key)
                    entity_query_list.append(normalized_query)
            
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

    def _hydrate_parent_chunks(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach parent text, metadata and embeddings before diversity selection."""
        if not candidates or not self._chunk_collection:
            return candidates
        try:
            parent_results = self._chunk_collection.get(
                ids=[str(item.get("chunk_id", "")) for item in candidates if item.get("chunk_id")],
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as exc:
            logger.warning("Failed to hydrate parent chunks: {}", exc)
            return candidates

        ids = parent_results.get("ids", []) or []
        documents = parent_results.get("documents", []) or []
        metadatas = parent_results.get("metadatas", []) or []
        embeddings = parent_results.get("embeddings", [])
        if embeddings is None:
            embeddings = []
        id_to_index = {str(chunk_id): idx for idx, chunk_id in enumerate(ids)}

        hydrated: list[dict[str, Any]] = []
        for candidate in candidates:
            chunk_id = str(candidate.get("chunk_id", ""))
            idx = id_to_index.get(chunk_id)
            if idx is None:
                continue
            item = dict(candidate)
            item["text"] = documents[idx] if idx < len(documents) else ""
            if idx < len(embeddings) and embeddings[idx] is not None:
                emb = embeddings[idx]
                item["embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            if idx < len(metadatas):
                meta = metadatas[idx] or {}
                item["paper_id"] = meta.get("paper_id", item.get("paper_id", ""))
                item["paper_title"] = meta.get("paper_title", "")
                item["paper_year"] = meta.get("paper_year")
                item["paper_source"] = meta.get("paper_source", "")
                for key in ("keywords", "entities", "claims"):
                    try:
                        item[key] = json.loads(meta.get(key, "[]"))
                    except (TypeError, json.JSONDecodeError):
                        item[key] = []
            hydrated.append(item)
        return hydrated

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
        query_list = list(dict.fromkeys(item for item in query_list if item))
        if not query_list:
            return []
        
        matched_chunks: dict[str, dict[str, Any]] = {}
        dense_rankings: list[list[tuple[str, float]]] = []
        sparse_rankings: list[list[tuple[str, float]]] = []
        sparse_scores: dict[str, float] = {}
        sparse_enabled = (
            use_hybrid
            and self.config.use_hybrid_retrieval
            and self._lexical_index is not None
        )

        def _collapse_ranking(
            ranking: list[tuple[str, float]],
        ) -> list[tuple[str, float]]:
            best: dict[str, float] = {}
            for chunk_id, score in ranking:
                if chunk_id:
                    best[chunk_id] = max(best.get(chunk_id, float("-inf")), score)
            return sorted(best.items(), key=lambda item: item[1], reverse=True)

        def _record_dense(
            *,
            chunk_id: str,
            score: float,
            metadata: dict[str, Any],
            matched_by: str,
            matched_text: str,
        ) -> None:
            item = matched_chunks.setdefault(chunk_id, {"chunk_id": chunk_id})
            previous_dense = float(item.get("dense_score", float("-inf")))
            if score >= previous_dense:
                item.update({
                    "paper_id": metadata.get("paper_id", ""),
                    "section": metadata.get("section", ""),
                    "heading_path": metadata.get("heading_path", ""),
                    "matched_text": matched_text,
                })
            item["dense_score"] = max(previous_dense, score)
            item["score"] = item["dense_score"]
            item["score_type"] = "dense_cosine"
            matched_sources = set(filter(None, str(item.get("matched_by", "")).split("+")))
            matched_sources.add(matched_by)
            item["matched_by"] = "+".join(sorted(matched_sources))
        
        query_embeddings = await self.embed_texts(query_list)
        valid_query_embeddings = [
            embedding for embedding in query_embeddings if embedding
        ]

        # One batched Chroma query per enabled view instead of one call per
        # rewritten query. Each returned row remains an independent RRF list.
        dense_views: list[tuple[str, Any, int]] = []
        if search_mode in ("hybrid", "chunks_only") and self._chunk_collection is not None:
            dense_views.append(("chunk", self._chunk_collection, top_k * 3))
        if search_mode in ("hybrid", "summaries_only") and self._summary_collection is not None:
            dense_views.append(("summary", self._summary_collection, top_k * 2))
        if search_mode in ("hybrid", "questions_only") and self._question_collection is not None:
            dense_views.append(("question", self._question_collection, top_k * 3))

        if valid_query_embeddings:
            for matched_by, collection, candidate_count in dense_views:
                try:
                    dense_results = await asyncio.to_thread(
                        collection.query,
                        query_embeddings=valid_query_embeddings,
                        n_results=max(top_k, candidate_count),
                        include=["metadatas", "documents", "distances"],
                        where=where_filter,
                    )
                    result_ids = dense_results.get("ids", []) if dense_results else []
                    result_metas = dense_results.get("metadatas", []) if dense_results else []
                    result_docs = dense_results.get("documents", []) if dense_results else []
                    result_distances = dense_results.get("distances", []) if dense_results else []

                    for query_index in range(len(valid_query_embeddings)):
                        ids = result_ids[query_index] if query_index < len(result_ids) else []
                        metas = result_metas[query_index] if query_index < len(result_metas) else []
                        docs = result_docs[query_index] if query_index < len(result_docs) else []
                        distances = (
                            result_distances[query_index]
                            if query_index < len(result_distances)
                            else []
                        )
                        ranking: list[tuple[str, float]] = []
                        for result_index, result_id in enumerate(ids or []):
                            meta = metas[result_index] if result_index < len(metas) else {}
                            # Parent chunks use their Chroma ID directly; summary
                            # and question views carry the parent ID in metadata.
                            chunk_id = str(meta.get("chunk_id") or result_id or "")
                            if not chunk_id:
                                continue
                            distance = (
                                distances[result_index]
                                if result_index < len(distances)
                                else 1.0
                            )
                            score = 1.0 - float(distance)
                            ranking.append((chunk_id, score))
                            _record_dense(
                                chunk_id=chunk_id,
                                score=score,
                                metadata=meta,
                                matched_by=matched_by,
                                matched_text=(
                                    docs[result_index]
                                    if result_index < len(docs)
                                    else ""
                                ),
                            )
                        collapsed = _collapse_ranking(ranking)
                        if collapsed:
                            dense_rankings.append(collapsed)
                except Exception as exc:
                    logger.warning("{} dense search failed: {}", matched_by, exc)

        # --- Sparse: one weighted multi-field FTS ranking per rewritten query ---
        if sparse_enabled:
            if search_mode == "summaries_only":
                sparse_fields: tuple[str, ...] | None = ("summary",)
            elif search_mode == "questions_only":
                sparse_fields = ("questions",)
            elif search_mode == "chunks_only":
                sparse_fields = ("title", "keywords", "body")
            else:
                sparse_fields = None

            paper_id_filter = ""
            title_filter = ""
            if where_filter:
                paper_id_value = where_filter.get("paper_id")
                if isinstance(paper_id_value, str):
                    paper_id_filter = paper_id_value
                title_value = where_filter.get("paper_title")
                if isinstance(title_value, dict):
                    title_filter = str(title_value.get("$contains", ""))
                elif isinstance(title_value, str):
                    title_filter = title_value

            def _search_sparse_queries() -> list[list[tuple[str, float]]]:
                assert self._lexical_index is not None
                return [
                    self._lexical_index.search(
                        sparse_query,
                        top_n=top_k * 10,
                        fields=sparse_fields,
                        paper_id=paper_id_filter,
                        title_contains=title_filter,
                    )
                    for sparse_query in query_list
                ]

            try:
                sparse_rankings = [
                    ranking
                    for ranking in await asyncio.to_thread(_search_sparse_queries)
                    if ranking
                ]
                self._lexical_last_error = ""
                for ranking in sparse_rankings:
                    for chunk_id, bm25_score in ranking:
                        sparse_scores[chunk_id] = max(
                            sparse_scores.get(chunk_id, float("-inf")),
                            bm25_score,
                        )
            except Exception as exc:
                self._lexical_last_error = str(exc)
                sparse_rankings = []
                logger.warning("SQLite FTS5 search failed: {}", exc)

        # RRF is a ranking score, not a semantic similarity.  Give Dense and
        # Sparse fixed family weights, then divide each family weight equally
        # among its views/rewrites so query decomposition cannot bias fusion.
        if sparse_rankings:
            fused = _weighted_rrf_fuse(
                [
                    (dense_rankings, self.config.dense_rrf_weight),
                    (sparse_rankings, self.config.sparse_rrf_weight),
                ],
                k=max(1, self.config.rrf_k),
            )
            max_rrf = max(fused.values(), default=1.0)
            for chunk_id, raw_rrf_score in fused.items():
                item = matched_chunks.setdefault(chunk_id, {"chunk_id": chunk_id})
                dense_score = item.get("dense_score")
                bm25_score = sparse_scores.get(chunk_id)
                item["dense_score"] = dense_score
                item["bm25_score"] = bm25_score
                item["rrf_score"] = raw_rrf_score
                item["score"] = raw_rrf_score / max(1e-12, max_rrf)
                item["score_type"] = "weighted_rrf"
                if dense_score is not None and bm25_score is not None:
                    matched_sources = set(filter(
                        None,
                        str(item.get("matched_by", "dense")).split("+"),
                    ))
                    matched_sources.add("bm25")
                    item["matched_by"] = "+".join(sorted(matched_sources))
                elif bm25_score is not None:
                    item["matched_by"] = "bm25_only"
                else:
                    item["matched_by"] = item.get("matched_by", "dense")
        
        # Hydrate parent embeddings before MMR; otherwise this path silently
        # degrades to relevance-only greedy selection.
        sorted_chunks = sorted(matched_chunks.values(), key=lambda x: x.get("score", 0), reverse=True)
        sorted_chunks = self._hydrate_parent_chunks(sorted_chunks)
        
        if apply_diversity:
            final_chunks = _select_diverse(
                sorted_chunks,
                k=top_k,
                per_paper_limit=max(1, per_paper_limit),
                mmr_lambda=self.config.mmr_lambda,
            )
        else:
            final_chunks = sorted_chunks[:top_k]
        
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
                "url": d.get("url", ""),
                "year": d.get("year"),
                "source": d.get("source", ""),
            }
        return meta

    def get_stats(self) -> dict[str, Any]:
        """Return storage statistics without exposing backend internals to the API."""
        docs = self._read_jsonl(self.docs_file)
        chunks = self._read_jsonl(self.chunks_file)
        chunk_counts: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            pid = str(chunk.get("paper_id", ""))
            if pid:
                chunk_counts[pid] += 1

        chroma_count: int | None = None
        chroma_consistent: bool | None = None
        if self._chunk_collection is not None:
            try:
                chroma_count = int(self._chunk_collection.count())
                chroma_consistent = chroma_count == len(chunks)
                if not chroma_consistent:
                    logger.warning(
                        "Paper KB backend count mismatch: chroma={} jsonl={}",
                        chroma_count,
                        len(chunks),
                    )
            except Exception as exc:
                logger.debug("Unable to read Chroma stats: {}", exc)

        embedding_status = self.get_embedding_status()
        lexical_status = self.get_lexical_status()
        lexical_count = lexical_status.get("document_count")
        lexical_consistent = (
            lexical_count == len(chunks)
            if isinstance(lexical_count, int)
            else None
        )
        backends_consistent = chroma_consistent is True and lexical_consistent is True
        storage_degraded = (
            self._chunk_collection is None
            or chroma_consistent is not True
            or lexical_consistent is not True
        )
        degradation_reasons: list[str] = []
        if self._chunk_collection is None:
            degradation_reasons.append("chroma_unavailable")
        elif chroma_consistent is not True:
            degradation_reasons.append("chroma_jsonl_inconsistent")
        if lexical_status.get("reason"):
            degradation_reasons.append(str(lexical_status["reason"]))
        elif lexical_consistent is not True:
            degradation_reasons.append("lexical_jsonl_inconsistent")
        if embedding_status.get("reason"):
            degradation_reasons.append(str(embedding_status["reason"]))

        if self._chunk_collection is not None and self._lexical_index is not None:
            storage_backend = "chroma+sqlite_fts5+jsonl"
        elif self._chunk_collection is not None:
            storage_backend = "chroma+jsonl"
        elif self._lexical_index is not None:
            storage_backend = "sqlite_fts5+jsonl"
        else:
            storage_backend = "jsonl"

        recent = sorted(docs, key=lambda row: str(row.get("updated_at", "")), reverse=True)[:10]
        return {
            "paper_count": len({str(row.get("paper_id")) for row in docs if row.get("paper_id")}),
            "chunk_count": len(chunks),
            "chroma_chunk_count": chroma_count,
            "lexical_chunk_count": lexical_count,
            "storage_backend": storage_backend,
            "chroma_consistent": chroma_consistent,
            "lexical_consistent": lexical_consistent,
            "backends_consistent": backends_consistent,
            "embedding": embedding_status,
            "lexical": lexical_status,
            "degraded": storage_degraded or bool(embedding_status["degraded"]),
            "degradation_reasons": degradation_reasons,
            "recent_papers": [
                {
                    "paper_id": row.get("paper_id", ""),
                    "title": row.get("title", ""),
                    "source": row.get("source", ""),
                    "year": row.get("year"),
                    "chunk_count": chunk_counts.get(str(row.get("paper_id", "")), 0),
                    "updated_at": row.get("updated_at", ""),
                }
                for row in recent
            ],
        }

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
