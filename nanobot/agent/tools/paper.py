"""Paper domain tools: search, similarity, rerank, ingest, retrieve."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import random
import re
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx
from loguru import logger

from nanobot.agent.paper_kb import PaperKnowledgeBase, tokenize_text
from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    NumberSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.security.network import validate_resolved_url, validate_url_target
from nanobot.utils.document import extract_text

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

MAX_PAPER_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_ARXIV_RESPONSE_BYTES = 10 * 1024 * 1024


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tok(text: str) -> set[str]:
    return set(tokenize_text(text))


def _safe_paper_id(value: Any) -> str:
    """Return a filesystem-safe, stable paper identifier."""
    raw = str(value or "paper")
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "paper"
    if clean != raw or len(clean) > 120:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        clean = f"{clean[:108]}-{digest}"
    return clean


def _valid_extracted_text(text: str, *, min_chars: int = 100) -> bool:
    """Reject empty or obviously binary/garbled parser output."""
    clean = (text or "").strip()
    if len(clean) < min_chars or clean.lower().startswith("[error:"):
        return False
    replacement_ratio = clean.count("\ufffd") / max(1, len(clean))
    printable_ratio = sum(ch.isprintable() or ch in "\n\r\t" for ch in clean) / len(clean)
    return replacement_ratio < 0.01 and printable_ratio > 0.90


@dataclass(slots=True)
class _ArxivSearchResult:
    papers: list[dict[str, Any]]
    status: str
    query: str
    sort_by: str
    attempts: int = 1
    latency_ms: int = 0
    error: str = ""


class _ArxivRateLimiter:
    """Serialize request starts so concurrent queries share one rate limit."""

    def __init__(self, min_interval_seconds: float = 3.0):
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self.min_interval_seconds - (now - self._last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request_at = time.monotonic()


def _escape_arxiv_term(value: str) -> str:
    """Remove query-language control characters while preserving term text."""
    return _norm(re.sub(r'["\\()\[\]{}]', " ", str(value or "")))


def _build_arxiv_query(
    query: str,
    keywords: list[str] | None = None,
    *,
    from_year: int | None = None,
    to_year: int | None = None,
) -> str:
    """Build a field-aware arXiv query with an optional submitted-date range."""
    clean_query = _escape_arxiv_term(query)
    arxiv_id = re.fullmatch(
        r"(?:arxiv:)?((?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?)",
        clean_query,
        flags=re.IGNORECASE,
    )
    if arxiv_id:
        search_query = f"id:{arxiv_id.group(1)}"
    else:
        clauses: list[str] = []
        if clean_query:
            clauses.append(f'(ti:"{clean_query}" OR abs:"{clean_query}")')
        keyword_clauses = []
        for keyword in keywords or []:
            clean_keyword = _escape_arxiv_term(keyword)
            if clean_keyword:
                keyword_clauses.append(
                    f'(ti:"{clean_keyword}" OR abs:"{clean_keyword}")'
                )
        if keyword_clauses:
            clauses.append("(" + " AND ".join(keyword_clauses) + ")")
        search_query = " OR ".join(clauses) or "all:*"

    if from_year is not None or to_year is not None:
        lower = max(1991, int(from_year or 1991))
        upper = min(2100, int(to_year or datetime.now().year))
        if lower > upper:
            lower, upper = upper, lower
        search_query = (
            f"({search_query}) AND "
            f"submittedDate:[{lower}01010000 TO {upper}12312359]"
        )
    return search_query


async def _parse_arxiv(
    query: str,
    keywords: list[str] | None = None,
    max_results: int = 20,
    *,
    sort_by: str = "relevance",
    from_year: int | None = None,
    to_year: int | None = None,
    start: int = 0,
    client: httpx.AsyncClient | None = None,
    rate_limiter: _ArxivRateLimiter | None = None,
) -> _ArxivSearchResult:
    """Search arXiv with connection reuse, shared throttling and diagnostics."""
    if client is None:
        async with httpx.AsyncClient(
            timeout=20.0,
            headers={"User-Agent": "nanobot-paper-search/1.0"},
        ) as owned_client:
            return await _parse_arxiv(
                query,
                keywords,
                max_results,
                sort_by=sort_by,
                from_year=from_year,
                to_year=to_year,
                start=start,
                client=owned_client,
                rate_limiter=rate_limiter or _ArxivRateLimiter(),
            )

    normalized_sort = "submittedDate" if sort_by == "submittedDate" else "relevance"
    search_query = _build_arxiv_query(
        query,
        keywords,
        from_year=from_year,
        to_year=to_year,
    )
    params = {
        "search_query": search_query,
        "start": max(0, int(start)),
        "max_results": max(1, min(100, int(max_results))),
        "sortBy": normalized_sort,
        "sortOrder": "descending",
    }
    url = f"https://export.arxiv.org/api/query?{urlencode(params)}"
    limiter = rate_limiter or _ArxivRateLimiter()
    max_retries = 5
    started_at = time.monotonic()
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            await limiter.wait()
            response = await client.get(url)
            if len(response.content) > MAX_ARXIV_RESPONSE_BYTES:
                raise ValueError("arXiv response exceeded size limit")
            response.raise_for_status()

            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            root = ET.fromstring(response.text)
            papers: list[dict[str, Any]] = []
            for entry in root.findall("atom:entry", namespaces):
                entry_url = _norm(entry.findtext("atom:id", "", namespaces))
                paper_id = _norm(entry_url.split("/")[-1])
                title = _norm(entry.findtext("atom:title", "", namespaces))
                abstract = _norm(entry.findtext("atom:summary", "", namespaces))
                published = _norm(entry.findtext("atom:published", "", namespaces))
                updated = _norm(entry.findtext("atom:updated", "", namespaces))
                try:
                    year = int(published[:4]) if published else None
                except ValueError:
                    year = None
                authors = [
                    _norm(author.findtext("atom:name", "", namespaces))
                    for author in entry.findall("atom:author", namespaces)
                ]
                pdf_url = ""
                for link in entry.findall("atom:link", namespaces):
                    if link.attrib.get("title") == "pdf":
                        pdf_url = link.attrib.get("href", "")
                        break
                categories = [
                    category.attrib.get("term", "")
                    for category in entry.findall("atom:category", namespaces)
                    if category.attrib.get("term")
                ]
                primary = entry.find("arxiv:primary_category", namespaces)
                papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "url": entry_url,
                    "pdf_url": pdf_url,
                    "source": "arxiv",
                    "published": published,
                    "updated": updated,
                    "year": year,
                    "authors": [author for author in authors if author],
                    "categories": categories,
                    "primary_category": primary.attrib.get("term", "") if primary is not None else "",
                    "doi": _norm(entry.findtext("arxiv:doi", "", namespaces)),
                    "journal_ref": _norm(entry.findtext("arxiv:journal_ref", "", namespaces)),
                    "comment": _norm(entry.findtext("arxiv:comment", "", namespaces)),
                })
            return _ArxivSearchResult(
                papers=papers,
                status="ok",
                query=query,
                sort_by=normalized_sort,
                attempts=attempt,
                latency_ms=int((time.monotonic() - started_at) * 1000),
            )
        except ET.ParseError as exc:
            last_error = f"parse_error: {exc}"
            break
        except (httpx.HTTPError, ValueError) as exc:
            last_error = str(exc)
            status_code = (
                exc.response.status_code
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None
                else None
            )
            retryable = (
                status_code in {429, 500, 502, 503, 504}
                or isinstance(exc, httpx.RequestError)
            )
            if not retryable or attempt >= max_retries:
                break
            retry_after = 0.0
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                try:
                    retry_after = float(exc.response.headers.get("Retry-After", "0") or 0)
                except ValueError:
                    retry_after = 0.0
            delay = max(retry_after, min(30.0, 2.0 ** (attempt - 1)))
            delay += random.uniform(0.0, 0.5)
            logger.warning(
                "arXiv request failed (attempt {}/{}), retrying in {:.1f}s: {}",
                attempt,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    if last_error.startswith("parse_error:"):
        status = "parse_error"
    else:
        status = "rate_limited" if "429" in last_error else "request_error"
    logger.warning("arXiv search failed for query='{}': {}", query[:60], last_error)
    return _ArxivSearchResult(
        papers=[],
        status=status,
        query=query,
        sort_by=normalized_sort,
        attempts=attempt,
        latency_ms=int((time.monotonic() - started_at) * 1000),
        error=last_error,
    )


def _strip_front_matter(text: str) -> str:
    """Remove everything before the first Introduction/Abstract heading.

    Handles patterns like:
    - ``# Introduction``
    - ``# 1 Introduction``
    - ``# I. Introduction``
    - ``# 1.1 Introduction``
    - ``# Chapter 1 Introduction``
    """
    pattern = re.compile(
        r'^#{1,6}\s+(?:'                                # heading up to ######
        r'(?:\d+(?:\.\d+)*\s*\.?\s*)?'              # optional "1", "1.1", "1."
        r'(?:[IVXLCDM]+\.?\s*)?'                        # optional "I", "II."
        r'(?:Chapter\s+\d+\s+)?'                        # optional "Chapter 1 "
        r'(introduction)\b'                               # the keyword itself
        r')',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text.strip())
    if match:
        return text[match.start():].strip()
    return text.strip()


def _extract_front_matter(text: str) -> str:
    """Extract the front matter (everything before Introduction heading).

    Handles patterns like:
    - ``# Introduction``
    - ``# 1 Introduction``
    - ``# I. Introduction``
    - ``# 1.1 Introduction``
    - ``# Chapter 1 Introduction``

    Returns the text before the heading, or empty string if not found.
    """
    pattern = re.compile(
        r'^#{1,6}\s+(?:'
        r'(?:\d+(?:\.\d+)*\s*\.?\s*)?'
        r'(?:[IVXLCDM]+\.?\s*)?'
        r'(?:Chapter\s+\d+\s+)?'
        r'(introduction)\b'
        r')',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text.strip())
    if match:
        return text[:match.start()].strip()
    return ""


async def _parse_front_matter_metadata(
    front_matter: str,
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Extract paper metadata (title, authors, abstract) from front matter text.

    Called when arXiv API metadata is unavailable (e.g. local PDF uploads).
    """
    if not front_matter or not provider or not model:
        # Fallback: heuristic title extraction
        title = ""
        lines = front_matter.strip().splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break
        return {"title": title, "authors": [], "abstract": "", "year": None}

    prompt = (
        "Extract paper metadata from this markdown front matter. "
        "Return ONLY JSON:\n"
        '{"title": "...", "authors": ["Author1", "Author2"], '
        '"abstract": "...", "year": 2024 | None}\n\n'
        f"Front matter:\n{front_matter}"
    )
    try:
        resp = await provider.chat_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": "You extract paper metadata from markdown front matter."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            tool_choice=None,
        )
        raw = (resp.content or "").strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        import json_repair
        payload = json_repair.loads(raw.strip())
        year = payload.get("year", "")
        if year:
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = None
        return {
            "title": str(payload.get("title", "") or "").strip(),
            "authors": payload.get("authors", []),
            "abstract": str(payload.get("abstract", "") or "").strip(),
            "year": year,
        }
    except Exception as e:
        logger.warning("Front matter metadata extraction failed: {}", e)
        return {"title": "", "authors": [], "abstract": "", "year": None}


def _heuristic_similarity(query: str, title: str, abstract: str) -> float:
    q = _tok(query)
    doc = _tok(f"{title} {abstract}")
    if not q:
        return 0.0
    overlap = len(q & doc) / len(q)
    return float(min(1.0, overlap))


async def _generate_candidate_queries(
    original_query: str,
    provider: LLMProvider | None = None,
    model: str | None = None,
    num_queries: int = 3,
) -> list[str]:
    """Generate multiple candidate search queries from a single user question.
    
    Uses LLM to reformulate the question from different perspectives/angles,
    producing diverse search queries that improve recall.
    
    Args:
        original_query: User's original question
        provider: LLM provider for query generation
        model: Model name for query generation
        num_queries: Number of candidate queries to generate (default 3)
    
    Returns:
        List of candidate queries, always including the original query as first element.
    """
    # Always include original query
    result = [original_query]
    
    if not provider or not model:
        # Fallback: generate keyword-based variant queries
        keywords = _extract_keywords_from_query(original_query, top_n=6)
        if len(keywords) >= 2:
            result.append(" ".join(keywords[:2]))
            if len(keywords) >= 3:
                result.append(" ".join(keywords[:3]))
        return result[:num_queries + 1]
    
    prompt = (
        "You are a search query optimizer for academic paper retrieval. "
        "Given a user's research question, generate {num_queries} alternative search queries "
        "that would find relevant papers from different perspectives.\n\n"
        "Rules:\n"
        "- Each query should be 2-8 words, concise and search-friendly\n"
        "- Use different terminology or synonyms\n"
        "- Remove filler words (latest, recent, about, etc.)\n"
        "- Queries should complement each other, not overlap or alter the original meaning\n"
        "- Complex queries involving multiple domains or perspectives should be appropriately decomposed\n"
        "- Return ONLY a JSON array of strings, no explanation\n\n"
        "Original question: {original_query}\n\n"
        "Example output for 'What are the latest methods for EEG signal classification?':\n"
        '["EEG signal classification", '
        '"brain wave pattern recognition", '
        '"electroencephalography classification"]\n\n'
        "Return ONLY the JSON array."
    ).format(num_queries=num_queries, original_query=original_query)
    
    try:
        resp = await provider.chat_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise JSON generator for search query optimization."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            tool_choice=None,
        )
        raw = (resp.content or "").strip()
        
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        
        queries = json.loads(raw)
        if isinstance(queries, list):
            for q in queries:
                q_str = str(q).strip()
                if q_str and q_str.lower() != original_query.lower():
                    result.append(q_str)
        
        logger.info("Generated candidate queries: original='{}' candidates={}", 
                     original_query[:50], len(result) - 1)
        return result[:num_queries + 1]
    except Exception as e:
        logger.warning("LLM query generation failed: {}, using keyword fallback", e)
        keywords = _extract_keywords_from_query(original_query, top_n=6)
        if len(keywords) >= 2:
            result.append(" ".join(keywords[:2]))
            if len(keywords) >= 3:
                result.append(" ".join(keywords[:3]))
        return result[:num_queries + 1]


def _canonical_paper_id(value: Any) -> str:
    paper_id = re.sub(r"^arxiv:", "", str(value or "").strip(), flags=re.IGNORECASE)
    versioned_arxiv = re.fullmatch(
        r"(?P<base>(?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7}))v\d+",
        paper_id,
        flags=re.IGNORECASE,
    )
    if versioned_arxiv:
        paper_id = versioned_arxiv.group("base")
    return paper_id.casefold()


def _paper_fusion_key(paper: dict[str, Any]) -> str:
    """Build a canonical identity for cross-query result fusion."""
    canonical_id = _canonical_paper_id(
        paper.get("paper_id", "") or paper.get("id", "")
    )
    if canonical_id:
        return f"id:{canonical_id}"
    title = _norm(str(paper.get("title", ""))).casefold()
    return "title:" + hashlib.sha256(title.encode("utf-8")).hexdigest()


def _rrf_fuse_paper_rankings(
    rankings: list[list[dict[str, Any]]],
    queries: list[str] | None = None,
    *,
    k: int = 60,
    ranking_weights: list[float] | None = None,
    ranking_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fuse external query rankings while retaining rank provenance."""
    records: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    matches: dict[str, list[dict[str, Any]]] = {}

    for query_index, ranking in enumerate(rankings):
        ranking_weight = (
            max(0.0, float(ranking_weights[query_index]))
            if ranking_weights and query_index < len(ranking_weights)
            else 1.0
        )
        if ranking_weight <= 0:
            continue
        seen_in_ranking: set[str] = set()
        for rank, paper in enumerate(ranking, 1):
            key = _paper_fusion_key(paper)
            if key in seen_in_ranking:
                continue
            seen_in_ranking.add(key)
            records.setdefault(key, dict(paper))
            scores[key] = scores.get(key, 0.0) + ranking_weight / (max(1, k) + rank)
            match: dict[str, Any] = {
                "query_index": query_index,
                "rank": rank,
            }
            if queries and query_index < len(queries):
                match["query"] = str(queries[query_index])[:200]
            if ranking_labels and query_index < len(ranking_labels):
                match["route"] = ranking_labels[query_index]
            matches.setdefault(key, []).append(match)

    max_score = max(scores.values(), default=1.0)
    fused: list[dict[str, Any]] = []
    for key, raw_score in scores.items():
        paper = records[key]
        paper["query_rrf_score"] = round(raw_score / max(1e-12, max_score), 6)
        paper["query_rrf_raw_score"] = round(raw_score, 8)
        paper["query_hit_count"] = len(matches.get(key, []))
        paper["query_matches"] = matches.get(key, [])
        fused.append(paper)
    fused.sort(key=lambda paper: paper.get("query_rrf_raw_score", 0.0), reverse=True)
    return fused


def _paper_in_time_range(
    paper: dict[str, Any],
    from_year: int | None,
    to_year: int | None,
) -> bool:
    if from_year is None and to_year is None:
        return True
    try:
        year = int(paper.get("year"))
    except (TypeError, ValueError):
        return False
    return (from_year is None or year >= from_year) and (to_year is None or year <= to_year)


def _select_external_candidate_pool(
    fused: list[dict[str, Any]],
    rankings: list[list[dict[str, Any]]],
    *,
    budget: int,
    quota_per_ranking: int = 2,
) -> list[dict[str, Any]]:
    """Guarantee minimal route coverage, then fill the pool by fused rank."""
    budget = max(1, budget)
    fused_by_key = {_paper_fusion_key(paper): paper for paper in fused}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ranking in rankings:
        added = 0
        for paper in ranking:
            key = _paper_fusion_key(paper)
            if key in seen or key not in fused_by_key:
                continue
            selected.append(fused_by_key[key])
            seen.add(key)
            added += 1
            if len(selected) >= budget or added >= quota_per_ranking:
                break
        if len(selected) >= budget:
            return selected
    for paper in fused:
        key = _paper_fusion_key(paper)
        if key not in seen:
            selected.append(paper)
            seen.add(key)
            if len(selected) >= budget:
                break
    return selected


def _query_requests_recency(query: str) -> bool:
    return bool(re.search(
        r"(?:最新|近期|最近|近年|newest|latest|recent|state[ -]of[ -]the[ -]art|\bSOTA\b)",
        query,
        flags=re.IGNORECASE,
    ))


def _extract_keywords_from_query(query: str, top_n: int = 8) -> list[str]:
    ignore = {"paper", "papers", "latest", "recent", "survey", "about", "for", "with", "from", "that", 
    "论文", "文章", "最新", "最近", "近期", "综述", "有关", "关于", "领域", "的"}
    toks = [t for t in _tok(query) if t not in ignore]
    uniq: list[str] = []
    for t in toks:
        if t not in uniq:
            uniq.append(t)
    return uniq[:top_n]


def _trim_papers_for_payload(
    papers: list[dict[str, Any]],
    *,
    max_abstract_chars: int = 800,
    max_title_chars: int = 220,
) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for paper in papers:
        title = str(paper.get("title", ""))[:max_title_chars]
        abstract = str(paper.get("abstract", ""))[:max_abstract_chars]
        if len(str(paper.get("abstract", ""))) > max_abstract_chars:
            abstract += "..."
        trimmed.append({**paper, "title": title, "abstract": abstract})
    return trimmed


class _PaperTool(Tool):
    def __init__(
        self, 
        workspace: Path, 
        kb: PaperKnowledgeBase, 
        provider: LLMProvider | None = None,
        model: str | None = None
    ):
        self.workspace = workspace
        self.kb = kb
        self.provider = provider
        self.model = model

    @property
    def read_only(self) -> bool:
        return True


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("User query / research question"),
        source=StringSchema("Search source", enum=["arxiv"]),
        search_topk=IntegerSchema(60, minimum=1, maximum=100),
        recall_top_k=IntegerSchema(20, minimum=1, maximum=50),
        rerank_top_k=IntegerSchema(5, minimum=1, maximum=20),
        candidate_queries=ArraySchema(
            StringSchema("English search query"),
            description="Optional prepared query variants",
            max_items=10,
        ),
        exclude_paper_ids=ArraySchema(
            StringSchema("paper id"),
            description="Already ingested arXiv IDs to exclude before ranking",
            max_items=500,
        ),
        from_year=IntegerSchema(description="Optional inclusive start year", minimum=1991, maximum=2100),
        to_year=IntegerSchema(description="Optional inclusive end year", minimum=1991, maximum=2100),
        sort_mode=StringSchema(
            "Search mode: relevance, balanced relevance+recency, or auto",
            enum=["auto", "relevance", "balanced", "recent"],
        ),
        required=["query"],
    )
)
class PaperSearchTool(_PaperTool):
    name = "paper_search"
    description = (
        "Search latest papers (arXiv) through query. "
        "Generates multiple candidate queries from different perspectives, "
        "retrieves and merges results, then applies similarity scoring and reranking. "
        "Include complete retrieval process including paper_similarity and paper_rerank."
    )

    def __init__(
        self,
        workspace: Path,
        kb: PaperKnowledgeBase,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ):
        super().__init__(workspace=workspace, kb=kb, provider=provider, model=model)
        self._search_cache: dict[
            tuple[Any, ...], tuple[float, _ArxivSearchResult]
        ] = {}

    async def execute(
        self,
        query: str,
        source: str = "arxiv",
        search_topk: int = 60,
        recall_top_k: int = 20,
        rerank_top_k: int = 5,
        num_candidate_queries: int = 3,
        candidate_queries: list[str] | None = None,
        keywords: list[list[str]] | None = None,
        exclude_paper_ids: list[str] | None = None,
        from_year: int | None = None,
        to_year: int | None = None,
        sort_mode: str = "auto",
        **kwargs: Any,
    ) -> str:
        if source != "arxiv":
            return json.dumps({"error": "Only arxiv is supported for now."}, ensure_ascii=False)
        
        # Step 1: Use externally-provided candidate queries when available,
        # otherwise generate them via LLM internally
        keyword_by_query: dict[str, list[str]] = {}
        if candidate_queries:
            provided_queries = [
                str(q).strip() for q in candidate_queries if q and str(q).strip()
            ]
            for index, candidate_query in enumerate(provided_queries):
                if keywords and index < len(keywords) and keywords[index]:
                    keyword_key = re.sub(r"\s+", " ", candidate_query).strip().casefold()
                    keyword_by_query[keyword_key] = keywords[index]
            search_queries = []
            seen_queries: set[str] = set()
            for candidate_query in provided_queries or [query]:
                normalized = re.sub(r"\s+", " ", candidate_query).strip()
                key = normalized.casefold()
                if normalized and key not in seen_queries:
                    seen_queries.add(key)
                    search_queries.append(normalized)
            logger.info("paper_search: using {} external candidate_queries", len(search_queries))
        else:
            search_queries = await _generate_candidate_queries(
                original_query=query,
                provider=self.provider,
                model=self.model,
                num_queries=num_candidate_queries,
            )
            logger.info("paper_search: original='{}' candidates={}", query[:50], search_queries)
        
        normalized_sort_mode = sort_mode if sort_mode in {
            "relevance", "balanced", "recent"
        } else "auto"
        prefer_recent = (
            normalized_sort_mode in {"balanced", "recent"}
            or (normalized_sort_mode == "auto" and _query_requests_recency(query))
        )
        sort_routes = (
            [("relevance", 0.6), ("submittedDate", 0.4)]
            if prefer_recent
            else [("relevance", 1.0)]
        )
        route_specs = [
            (search_query, sort_by, weight)
            for search_query in search_queries
            for sort_by, weight in sort_routes
        ]
        per_route_topk = max(
            5,
            min(100, math.ceil(max(1, search_topk) / max(1, len(route_specs)))),
        )
        excluded_ids = {
            _canonical_paper_id(paper_id)
            for paper_id in exclude_paper_ids or []
            if _canonical_paper_id(paper_id)
        }
        limiter = _ArxivRateLimiter()
        cache_ttl_seconds = 15 * 60

        async with httpx.AsyncClient(
            timeout=20.0,
            headers={"User-Agent": "nanobot-paper-search/1.0"},
        ) as client:
            async def _search_one_route(
                search_query: str,
                sort_by: str,
            ) -> _ArxivSearchResult:
                keyword_key = re.sub(r"\s+", " ", search_query).strip().casefold()
                route_keywords = keyword_by_query.get(keyword_key) or _extract_keywords_from_query(search_query)
                cache_key = (
                    keyword_key,
                    tuple(route_keywords),
                    sort_by,
                    from_year,
                    to_year,
                    per_route_topk,
                )
                cached = self._search_cache.get(cache_key)
                if cached and cached[0] > time.monotonic():
                    return cached[1]
                route_result = await _parse_arxiv(
                    search_query,
                    route_keywords,
                    max_results=per_route_topk,
                    sort_by=sort_by,
                    from_year=from_year,
                    to_year=to_year,
                    client=client,
                    rate_limiter=limiter,
                )
                # Keep compatibility with light-weight injected test/search adapters.
                if isinstance(route_result, list):
                    route_result = _ArxivSearchResult(
                        papers=route_result,
                        status="ok",
                        query=search_query,
                        sort_by=sort_by,
                    )
                if route_result.status == "ok":
                    self._search_cache[cache_key] = (
                        time.monotonic() + cache_ttl_seconds,
                        route_result,
                    )
                    if len(self._search_cache) > 256:
                        now = time.monotonic()
                        self._search_cache = {
                            key: value
                            for key, value in self._search_cache.items()
                            if value[0] > now
                        }
                return route_result

            route_results = await asyncio.gather(*[
                _search_one_route(search_query, sort_by)
                for search_query, sort_by, _weight in route_specs
            ])

        raw_total = sum(len(result.papers) for result in route_results)
        excluded_total = 0
        results_per_route: list[list[dict[str, Any]]] = []
        for result in route_results:
            filtered_ranking = []
            for paper in result.papers:
                if _canonical_paper_id(paper.get("paper_id")) in excluded_ids:
                    excluded_total += 1
                    continue
                if not _paper_in_time_range(paper, from_year, to_year):
                    continue
                filtered_ranking.append(paper)
            results_per_route.append(filtered_ranking)

        route_queries = [spec[0] for spec in route_specs]
        route_weights = [spec[2] for spec in route_specs]
        route_labels = [spec[1] for spec in route_specs]
        fused_papers = _rrf_fuse_paper_rankings(
            results_per_route,
            route_queries,
            ranking_weights=route_weights,
            ranking_labels=route_labels,
        )
        deduped_total = len(fused_papers)
        all_papers = _select_external_candidate_pool(
            fused_papers,
            results_per_route,
            budget=max(1, search_topk),
        ) if fused_papers else []

        failed_routes = [result for result in route_results if result.status != "ok"]
        successful_routes = [result for result in route_results if result.status == "ok"]
        if failed_routes and successful_routes:
            search_status = "partial"
        elif failed_routes:
            search_status = "error"
        else:
            search_status = "ok"
        route_diagnostics = [
            {
                **{key: value for key, value in asdict(result).items() if key != "papers"},
                "result_count": len(result.papers),
            }
            for result in route_results
        ]

        if not all_papers:
            reason = "search_failed" if search_status == "error" else "no_results"
            logger.info("paper_search: query='{}' source={} results=0 status={}", query, source, search_status)
            return json.dumps(
                {
                    "query": query,
                    "candidate_queries": search_queries,
                    "results": [],
                    "reason": reason,
                    "search_status": search_status,
                    "route_diagnostics": route_diagnostics,
                    "embedding": self.kb.get_embedding_status(),
                    "workflow_hint": "Broaden query and retry only when the search completed successfully with no results.",
                },
                ensure_ascii=False,
            )

        logger.info(
            "paper_search: query='{}' routes={} raw_total={} deduped_total={} excluded={}",
            query[:50],
            len(route_specs),
            raw_total,
            deduped_total,
            excluded_total,
        )

        # Step 4: all prepared queries participate in the batched semantic coarse rank.
        sim_tool = PaperSimilarityTool(workspace=self.workspace, kb=self.kb)
        sim_payload = json.loads(await sim_tool.execute(
            query=query,
            queries=search_queries,
            papers=all_papers,
            top_k=recall_top_k,
        ))
        candidates = sim_payload.get("results", []) if isinstance(sim_payload, dict) else []

        # Step 5: use the configured Cross-Encoder in one batch, if available.
        rerank_tool = PaperRerankTool(workspace=self.workspace, kb=self.kb)
        rerank_payload = json.loads(await rerank_tool.execute(
            query=query,
            queries=search_queries,
            papers=candidates,
            top_k=rerank_top_k,
            prefer_recent=prefer_recent,
        ))
        ranked = rerank_payload.get("results", []) if isinstance(rerank_payload, dict) else all_papers
        next_step = "Use ranked results directly; call paper_ingest for deep internalization."
        
        return json.dumps(
            {
                "query": query,
                "candidate_queries": search_queries,
                "keywords_used": keywords or [_extract_keywords_from_query(query)],
                "source": source,
                "sort_mode": "balanced" if prefer_recent else "relevance",
                "next_step": next_step,
                "result_nonempty": len(all_papers) > 0,
                "search_status": search_status,
                "route_diagnostics": route_diagnostics,
                "raw_total": raw_total,
                "excluded_total": excluded_total,
                "deduped_total": deduped_total,
                "candidate_pool_total": len(all_papers),
                "total": len(ranked),
                "results": _trim_papers_for_payload(ranked),
                "embedding": self.kb.get_embedding_status(),
            },
            ensure_ascii=False,
        )
        
@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("Query text"),
        queries=ArraySchema(
            StringSchema("query variant"),
            description="Optional prepared query variants used for multi-query scoring",
            max_items=10,
        ),
        papers=ArraySchema(
            ObjectSchema(
                properties={
                    "paper_id": StringSchema("paper id"),
                    "title": StringSchema("title"),
                    "abstract": StringSchema("abstract"),
                    "year": IntegerSchema(description="year"),
                },
                required=["title", "abstract"],
            ),
            description="Paper candidates",
            min_items=1,
            max_items=200,
        ),
        top_k=IntegerSchema(10, minimum=1, maximum=50),
        required=["query"],
    )
)
class PaperSimilarityTool(_PaperTool):
    name = "paper_similarity"
    description = (
        "Score query-paper relevance for candidate papers. "
        "Expected to run after paper_search and before paper_rerank."
    )

    async def execute(
        self,
        query: str,
        queries: list[str] | None = None,
        papers: list[dict[str, Any]] | None = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> str:
        papers = papers or []
        if not papers:
            return json.dumps(
                {
                    "error": "papers is required",
                    "query": query,
                },
                ensure_ascii=False,
            )
        query_variants: list[str] = []
        seen_queries: set[str] = set()
        for query_variant in [query, *(queries or [])]:
            normalized_query = re.sub(r"\s+", " ", str(query_variant or "")).strip()
            query_key = normalized_query.casefold()
            if normalized_query and query_key not in seen_queries:
                seen_queries.add(query_key)
                query_variants.append(normalized_query)
        documents = [
            f"{str(paper.get('title', ''))}\n{str(paper.get('abstract', ''))}"
            for paper in papers
        ]
        vectors = await self.kb.embed_texts([*query_variants, *documents])
        query_embeddings = vectors[:len(query_variants)]
        document_embeddings = vectors[len(query_variants):]
        scored: list[dict[str, Any]] = []
        from nanobot.agent.paper_kb import _cosine_similarity  # noqa: PLC2701

        for paper, document_embedding in zip(papers, document_embeddings):
            title = str(paper.get("title", ""))
            abstract = str(paper.get("abstract", ""))
            semantic_scores = [
                _cosine_similarity(query_embedding, document_embedding)
                if query_embedding and document_embedding else 0.0
                for query_embedding in query_embeddings
            ]
            core_semantic = semantic_scores[0] if semantic_scores else 0.0
            max_semantic = max(semantic_scores, default=0.0)
            semantic_relevance = 0.7 * max_semantic + 0.3 * core_semantic
            lexical_scores = [
                _heuristic_similarity(query_variant, title, abstract)
                for query_variant in query_variants
            ]
            lexical_relevance = max(lexical_scores, default=0.0)
            coarse_relevance = 0.85 * semantic_relevance + 0.15 * lexical_relevance
            rrf_prior = min(1.0, max(0.0, float(paper.get("query_rrf_score", 0.0))))
            matched_queries = {
                str(match.get("query", "")).casefold()
                for match in paper.get("query_matches", [])
                if match.get("query")
            }
            retrieval_query_count = max(1, len(queries or [query]))
            coverage = min(1.0, len(matched_queries) / retrieval_query_count)
            if paper.get("query_rrf_score") is None:
                similarity_score = coarse_relevance
            else:
                similarity_score = (
                    0.85 * coarse_relevance
                    + 0.10 * rrf_prior
                    + 0.05 * coverage
                )
            best_query_index = (
                max(range(len(semantic_scores)), key=semantic_scores.__getitem__)
                if semantic_scores else 0
            )
            scored.append({
                **paper,
                "similarity_score": round(similarity_score, 6),
                "coarse_relevance_score": round(coarse_relevance, 6),
                "query_similarity_core": round(core_semantic, 6),
                "query_similarity_max": round(max_semantic, 6),
                "query_coverage_score": round(coverage, 6),
                "matched_query": (
                    query_variants[best_query_index]
                    if query_variants and best_query_index < len(query_variants)
                    else query
                ),
            })
        scored.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        logger.info("paper_similarity: query='{}' candidates={}", query, len(papers))
        return json.dumps(
            {
                "query": query,
                "queries": query_variants,
                "results": _trim_papers_for_payload(scored[:top_k]),
                "embedding": self.kb.get_embedding_status(),
            },
            ensure_ascii=False,
        )


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("Query text"),
        queries=ArraySchema(
            StringSchema("query variant"),
            description="Optional prepared query variants used by the reranker",
            max_items=10,
        ),
        papers=ArraySchema(
            ObjectSchema(
                properties={
                    "paper_id": StringSchema("paper id"),
                    "title": StringSchema("title"),
                    "abstract": StringSchema("abstract"),
                    "source": StringSchema("source"),
                    "year": IntegerSchema(description="year"),
                    "similarity_score": NumberSchema(description="optional precomputed similarity"),
                },
                required=["title", "abstract"],
            ),
            description="Paper candidates",
            min_items=1,
            max_items=300,
        ),
        top_k=IntegerSchema(10, minimum=1, maximum=100),
        prefer_recent=BooleanSchema(
            description="Apply a small recency weight in final ranking",
            default=False,
        ),
        required=["query"],
    )
)
class PaperRerankTool(_PaperTool):
    name = "paper_rerank"
    description = (
        "Final ranking stage for paper candidates using semantic relevance, query coverage, "
        "retrieval consensus, and optional recency. "
        "Use this output for conclusions instead of raw paper_search output."
    )

    async def execute(
        self,
        query: str,
        queries: list[str] | None = None,
        papers: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        prefer_recent: bool = False,
        **kwargs: Any,
    ) -> str:
        papers = papers or []
        if not papers:
            return json.dumps(
                {
                    "error": "papers is required",
                    "query": query,
                },
                ensure_ascii=False,
            )
        query_variants: list[str] = []
        seen_queries: set[str] = set()
        for query_variant in [query, *(queries or [])]:
            normalized_query = re.sub(r"\s+", " ", str(query_variant or "")).strip()
            query_key = normalized_query.casefold()
            if normalized_query and query_key not in seen_queries:
                seen_queries.add(query_key)
                query_variants.append(normalized_query)

        cross_encoder_scores: list[float] | None = None
        reranker_name = "coarse_fallback"
        if self.kb.config.rerank_model and query_variants:
            pairs = [
                (
                    query_variant,
                    f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}",
                )
                for paper in papers
                for query_variant in query_variants
            ]
            try:
                pair_scores = await self.kb.rerank_pairs(pairs)
                query_count = len(query_variants)
                cross_encoder_scores = [
                    max(pair_scores[index:index + query_count], default=0.0)
                    for index in range(0, len(pair_scores), query_count)
                ]
                reranker_name = "cross_encoder"
            except Exception as exc:
                logger.warning("paper_rerank: batch cross-encoder failed, using coarse scores: {}", exc)

        now_year = datetime.now().year
        ranked: list[dict[str, Any]] = []
        for paper_index, paper in enumerate(papers):
            relevance = (
                cross_encoder_scores[paper_index]
                if cross_encoder_scores and paper_index < len(cross_encoder_scores)
                else float(paper.get("coarse_relevance_score", paper.get("similarity_score", 0.0)))
            )
            try:
                year = int(paper.get("year"))
                recency = min(1.0, max(0.0, 1.0 - (now_year - year) / 10.0))
            except (TypeError, ValueError):
                recency = 0.5
            has_rrf = paper.get("query_rrf_score") is not None
            has_coverage = paper.get("query_coverage_score") is not None
            rrf_prior = min(1.0, max(0.0, float(paper.get("query_rrf_score", 0.0))))
            coverage = min(1.0, max(0.0, float(paper.get("query_coverage_score", 0.0))))
            recency_weight = 0.15 if prefer_recent else 0.0
            rrf_weight = 0.10 if has_rrf else 0.0
            coverage_weight = 0.05 if has_coverage else 0.0
            relevance_weight = 1.0 - recency_weight - rrf_weight - coverage_weight
            score = (
                relevance_weight * relevance
                + rrf_weight * rrf_prior
                + coverage_weight * coverage
                + recency_weight * recency
            )
            ranked.append({
                **paper,
                "rerank_score": round(score, 6),
                "rerank_relevance_score": round(relevance, 6),
                "recency_score": round(recency, 6),
                "reranker": reranker_name,
            })
        ranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        logger.info("paper_rerank: query='{}' candidates={} top_k={}", query, len(papers), top_k)
        return json.dumps(
            {
                "query": query,
                "queries": query_variants,
                "reranker": reranker_name,
                "results": _trim_papers_for_payload(ranked[:top_k]),
                "total": len(ranked),
            },
            ensure_ascii=False,
        )


_SECTION_HINTS = [
    "abstract",
    "introduction",
    "method",
    "methods",
    "approach",
    "experiment",
    "results",
    "discussion",
    "conclusion",
    "limitations",
]


def _replace_unparsed_images(text: str) -> str:
    """Replace unparsed image references like ``![](hash.jpg)`` with a placeholder.

    MinerU may leave raw image links that look like noise; replacing them
    makes the text cleaner for chunking and embedding.
    """
    pattern = re.compile(r'!\[.*?\]\([^)]+\.(?:jpg|jpeg|png|gif|webp|bmp|svg)(?:\?[^)]*)?\)')
    return pattern.sub("[Images omitted]", text)


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


def _save_asset_kv(kv_path: Path, assets: dict[str, dict[str, Any]], *, paper_id: str) -> None:
    kv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if kv_path.exists():
        with kv_path.open("r", encoding="utf-8") as f:
            existing = [line.rstrip("\n") for line in f if line.strip()]
    prefix = f"{paper_id}_"
    filtered: list[str] = []
    for line in existing:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = payload.get("key")
        if isinstance(key, str) and key.startswith(prefix):
            continue
        filtered.append(line)
    for key, value in assets.items():
        filtered.append(json.dumps({"key": key, "value": value}, ensure_ascii=False))
    kv_path.write_text("\n".join(filtered) + ("\n" if filtered else ""), encoding="utf-8")


def _extract_assets_and_strip(
    text: str,
    *,
    paper_id: str,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Extract figure/table captions and remove physical assets from text."""
    lines = (text or "").splitlines()
    assets: dict[str, dict[str, Any]] = {}
    cleaned_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if re.match(r"^!\[.*?\]\([^)]+\)$", stripped):
            i += 1
            continue

        if re.match(r"^[a-z]\)?\s*$", stripped):
            i += 1
            continue

        cap = re.match(
            r"^(Figure|Fig\.|Table|Tab\.?)\s+(\d+(?:\.\d+)*)\s*[:.,]?\s*(.*)",
            stripped,
            re.IGNORECASE,
        )
        if cap:
            raw_kind = cap.group(1).lower()
            num = cap.group(2)
            caption_text = cap.group(0).strip()
            kind = "table" if raw_kind.startswith("tab") else "figure"
            content = ""

            if kind == "table":
                j = i + 1
                table_lines: list[str] = []
                in_table = False
                while j < len(lines):
                    cur = lines[j]
                    cur_strip = cur.strip()
                    if "<table" in cur_strip.lower():
                        in_table = True
                    if in_table:
                        table_lines.append(cur)
                        if "</table>" in cur_strip.lower():
                            j += 1
                            break
                    else:
                        if cur_strip == "":
                            j += 1
                            break
                    j += 1
                if table_lines:
                    content = "\n".join(table_lines).strip()
                i = j
            else:
                i += 1

            key = f"{paper_id}_{'Table' if kind == 'table' else 'Figure'}_{num}"
            assets[key] = {
                "caption": caption_text,
                "content": content,
                "type": kind,
            }
            continue

        if "<table" in stripped.lower():
            j = i + 1
            while j < len(lines):
                if "</table>" in lines[j].strip().lower():
                    j += 1
                    break
                j += 1
            i = j
            continue

        cleaned_lines.append(line)
        i += 1

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned, assets


def _merge_split_paragraphs(text: str) -> str:
    """Merge paragraphs that were split across pages by MinerU.

    Two heuristic scenarios:

    **Scenario 1 — Hyphenation**:
       Previous para ends with ``-`` and next para starts with a lowercase letter.
       Remove the hyphen and extra newlines, e.g. ``trans-\n\nformation`` → ``transformation``.

    **Scenario 2 — Mid-sentence split**:
       Previous para ends with a lowercase letter or comma (no period) and next
       para starts with a lowercase letter.  Replace double newline with space.
    """
    paragraphs = re.split(r'\n{2,}', text)
    merged: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if merged:
            prev = merged[-1]
            # Scenario 1: hyphenation
            m1 = re.match(r'^.*-\s*$', prev, re.DOTALL)
            m2 = re.match(r'^[a-z]', para)
            if m1 and m2:
                merged[-1] = re.sub(r'-\s*$', '', prev) + para
                continue
            # Scenario 2: mid-sentence split
            # prev ends with lowercase or comma, no period/semicolon/question-mark
            if re.search(r'[a-z,]$', prev) and re.match(r'^[a-z]', para):
                merged[-1] = prev + ' ' + para
                continue
        merged.append(para)
    return '\n\n'.join(merged)


def _merge_consecutive_images(text: str) -> str:
    """Remove consecutive multi-panel image paragraphs, keep caption only.

    MinerU often renders sub-figures (a)(b)(c) as separate image links.
    A run of consecutive image paragraphs plus an optional trailing caption
    is collapsed into just the caption (image links discarded).
    """
    paragraphs = re.split(r'(\n\n+)', text)
    result: list[str] = []
    i = 0
    while i < len(paragraphs):
        stripped = paragraphs[i].strip()
        # Match an image paragraph (may include a trailing sub-label like "  a")
        is_img_para = bool(re.match(
            r'^!\[.*?\]\([^)]+\)(?:\s+[a-z]\)?)?\s*$', stripped
        ))
        # Standalone sub-label paragraph between images
        is_sub_label = bool(re.match(r'^[a-z]\)?\s*$', stripped))

        if is_img_para:
            # Skip all consecutive images and sub-label paragraphs
            while i < len(paragraphs):
                cur = paragraphs[i].strip()
                cur_img = bool(re.match(
                    r'^!\[.*?\]\([^)]+\)(?:\s+[a-z]\)?)?\s*$', cur
                ))
                cur_label = bool(re.match(r'^[a-z]\)?\s*$', cur))
                if cur_img or cur_label:
                    i += 1
                    if i < len(paragraphs) and re.match(r'^\n+$', paragraphs[i]):
                        i += 1
                else:
                    break
            # Check for trailing caption — keep only the caption, discard images
            if i < len(paragraphs):
                after = paragraphs[i].strip()
                cap_m = re.search(
                    r'(Figure|Table|Fig\.|Tab\.?)\s+\d+', after, re.IGNORECASE,
                )
                if cap_m:
                    caption_text = after[cap_m.start(1):]
                    result.append(caption_text)
                    i += 1
                    if i < len(paragraphs) and re.match(r'^\n+$', paragraphs[i]):
                        i += 1
                    continue
            # No caption found after image sequence — discard entirely
        else:
            result.append(paragraphs[i])
            i += 1
    return '\n'.join(result).replace('\n\n\n', '\n\n')


def _reflow_figures_tables(text: str) -> str:
    """Re-flow figure/table blocks to their first-mention paragraph.

    Academic PDF排版经常为了凑版面把图表"漂移"到远离首次引用的位置，
    甚至漂移到下一个章节中。 此函数实现三步骤逻辑重排：

    **Step 1: 提取并剥离图表块**
      识别 ``![]()`` + caption (``Figure X: ...`` / ``Table X: ...``)
      的连续段落，存入字典并从原文中删除。

    **Step 2: 寻找首次提及锚点**
      在剥离后的纯文本中搜索 ``Figure X`` / ``Table X`` / ``Fig. X``。

    **Step 3: 在锚点段落末尾重新插入**
      将图表块插入到首次提及所在段落的 ``\n\n`` 结尾处。
      未被引用的图表追加到文档末尾。
    """
    # ── Step 1: extract figure/table blocks ──
    paragraphs = re.split(r'(\n\n+)', text)
    body_parts: list[str] = []
    blocks: dict[str, str] = {}
    current_label: str | None = None
    current_block_parts: list[str] = []

    def _flush_block() -> None:
        nonlocal current_label, current_block_parts
        if current_label is not None and current_block_parts:
            blocks[current_label] = ''.join(current_block_parts)
        current_label = None
        current_block_parts.clear()

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            body_parts.append(para)
            continue

        # ---- Check if this para starts a figure/table caption ----
        # Pattern: "Figure N:" / "Table N:" / "Fig. N:" / "Tab. N:"
        caption_start = re.match(
            r'(Figure|Table|Fig\.|Tab\.?)'            # keyword
            r'\s+(\d+(?:\.\d+)*)'                   # number
            r'[\s:,]',                                # space or colon or comma
            stripped, re.IGNORECASE,
        )
        is_image = bool(re.match(r'^!\[.*?\]\([^)]+\)\s*$', stripped))

        if caption_start:
            _flush_block()
            keyword_raw = caption_start.group(1)
            if keyword_raw.lower().startswith('fig'):
                keyword = 'Figure'
            elif keyword_raw.lower().startswith('tab'):
                keyword = 'Table'
            else:
                keyword = keyword_raw
            num = caption_start.group(2)
            current_label = f"{keyword} {num}"
            current_block_parts.append(para)
        elif current_label is not None and is_image:
            current_block_parts.append(para)
        elif current_label is not None and re.match(r'^<table>', stripped, re.IGNORECASE):
            if not para.startswith('\n'):
                para = '\n' + para
            current_block_parts.append(para)
        elif current_label is not None and re.match(r'^</table>', stripped, re.IGNORECASE):
            current_block_parts.append(para)
            _flush_block()
        else:
            _flush_block()
            body_parts.append(para)
    _flush_block()

    if not blocks:
        return text

    clean_text = ''.join(body_parts)

    # ── Step 2 & 3: find first mention, re-insert ──
    for label in sorted(blocks, key=len, reverse=True):
        num_m = re.search(r'(\d+(?:\.\d+)*)', label)
        if not num_m:
            continue
        num = num_m.group(1)
        keyword = label.split()[0]  # "Figure" or "Table"

        # Build mention regex: include "Tab." / "Tabs." for tables
        if keyword == 'Table':
            mention_variants = r'Table|Tab\.' 
        else:
            mention_variants = r'Figure|Fig\.'
        # Use word boundary + negative lookahead to avoid matching
        # "Fig. 6a" when searching for "Fig. 6", or "Figs. 6, 7"
        # Allow letters after the number (sub-figure refs like "Fig. 6a")
        # but prevent digits (e.g. "Fig. 61" → different figure)
        mention = re.search(
            rf'(?:{mention_variants})\s*{re.escape(num)}(?![0-9])',
            clean_text, re.IGNORECASE,
        )
        if mention:
            # Insert at the end of the paragraph containing the mention
            pos = mention.end()
            next_gap = clean_text.find('\n\n', pos)
            if next_gap == -1:
                next_gap = len(clean_text)
            block_text = f"\n\n{blocks[label]}"
            clean_text = clean_text[:next_gap] + block_text + clean_text[next_gap:]
        else:
            # Unreferenced — append to document end
            clean_text += f"\n\n{blocks[label]}"

    return clean_text


def _remove_noisy_blocks(text: str) -> str:
    """
    去除 References, Acknowledgements, Copyrights 等尾部章节
    """
    # 正则匹配标题，兼容 "# REFERENCES", "## VIII. REFERENCES", "References" 等写法
    # (?i) 表示忽略大小写，^ 表示行首，re.MULTILINE 使 ^ 匹配每一行的开头
    pattern = re.compile(
        r'^(#{1,6}\s+)?([IVX\d]+\.?\s+)?(reference(s)?|acknowledg(e)?ment(s)?|copyright(s)?)\s*$', 
        re.IGNORECASE | re.MULTILINE
    )
    
    match = pattern.search(text)
    if match:
        # 找到匹配项后，截取匹配项之前的所有内容
        return text[:match.start()].strip()
    return text.strip()


def _split_sections(text: str) -> list[tuple[str, str]]:
    normalized = _remove_noisy_blocks(text)
    if not normalized:
        return []

    lines = normalized.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "content"
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current_title
        body = "\n".join(buf).strip()
        if body:
            sections.append((current_title, body))
        buf = []

    heading_re = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?([A-Za-z][A-Za-z\s]{2,40})\s*$")
    for line in lines:
        m = heading_re.match(line)
        if m:
            title = m.group(1).strip().lower()
            if any(h in title for h in _SECTION_HINTS):
                flush()
                current_title = title
                continue
        buf.append(line)
    flush()

    if not sections:
        return [("content", normalized)]
    return sections


def _split_markdown_semantic(
    text: str,
    max_chunk_chars: int = 4096,
    min_chunk_chars: int = 100,
) -> list[dict[str, Any]]:
    """Split markdown text into semantic chunks based on headers.
    
    Uses MarkdownHeaderTextSplitter from langchain to split by header levels,
    then further splits long chunks to fit within max_chunk_chars.
    
    Returns a list of dicts with:
        - section: header title (e.g., "Introduction")
        - heading_level: header level (1-6)
        - text: chunk content
        - heading_path: full heading path (e.g., "# Introduction ## Method")
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    
    normalized = _remove_noisy_blocks(text)
    if not normalized:
        return []
    
    # Define headers to split on
    headers_to_split_on = [
        ("#", "heading_1"),
        ("##", "heading_2"),
        ("###", "heading_3"),
        ("####", "heading_4"),
    ]
    
    try:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False,
            strip_headers=False,
        )
        md_chunks = splitter.split_text(normalized)
    except Exception as e:
        logger.warning("MarkdownHeaderTextSplitter failed: {}, fallback to _split_sections", e)
        # Fallback to original section splitting
        sections = _split_sections(normalized)
        return [
            {
                "section": title,
                "heading_level": 1,
                "text": body,
                "heading_path": title,
            }
            for title, body in sections
            if len(body) >= min_chunk_chars
        ]
    
    result: list[dict[str, Any]] = []
    for chunk in md_chunks:
        # Extract header info from metadata
        metadata = chunk.metadata or {}
        heading_1 = metadata.get("heading_1", "")
        heading_2 = metadata.get("heading_2", "")
        heading_3 = metadata.get("heading_3", "")
        heading_4 = metadata.get("heading_4", "")
        
        # Build section name and heading path
        section_parts = [h for h in [heading_1, heading_2, heading_3, heading_4] if h]
        section = section_parts[-1] if section_parts else "content"
        heading_path = " > ".join(section_parts) if section_parts else "content"
        
        # Determine heading level
        heading_level = 0
        if heading_4:
            heading_level = 4
        elif heading_3:
            heading_level = 3
        elif heading_2:
            heading_level = 2
        elif heading_1:
            heading_level = 1
        
        chunk_text = chunk.page_content.strip()
        
        # Further split if chunk is too long
        if len(chunk_text) <= max_chunk_chars:
            if len(chunk_text) >= min_chunk_chars:
                result.append({
                    "section": section,
                    "heading_level": heading_level,
                    "text": chunk_text,
                    "heading_path": heading_path,
                })
        else:
            # Split long chunks by paragraphs
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", chunk_text) if p.strip()]
            buf = ""
            for para in paragraphs:
                candidate = f"{buf}\n\n{para}".strip() if buf else para
                if len(candidate) <= max_chunk_chars:
                    buf = candidate
                    continue
                if buf and len(buf) >= min_chunk_chars:
                    result.append({
                        "section": section,
                        "heading_level": heading_level,
                        "text": buf,
                        "heading_path": heading_path,
                    })
                buf = para
            if buf and len(buf) >= min_chunk_chars:
                result.append({
                    "section": section,
                    "heading_level": heading_level,
                    "text": buf,
                    "heading_path": heading_path,
                })
    
    return result


def _extract_keywords(text: str, max_items: int = 8) -> list[str]:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "into", "using", "based",
        "论文", "研究", "方法", "结果", "我们", "以及", "进行", "通过", "模型",
    }
    toks = [t for t in _tok(text) if t not in stop and len(t) > 2]
    uniq: list[str] = []
    for t in toks:
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= max_items:
            break
    return uniq


def _fallback_section_summary(section: str, text: str) -> dict[str, Any]:
    body = _norm(text)
    snippet = body[:900] + ("..." if len(body) > 900 else "")
    return {
        "section": section,
        "summary": snippet,
        "claims": [],
        "limitations": [],
        "keywords": _extract_keywords(body),
    }


async def _generate_chunk_metadata(
    text: str,
    section: str,
    provider: LLMProvider | None = None,
    model: str | None = None,
    num_questions: int = 3,
    summarize: bool = True,
    *,
    title: str = "",
    abstract: str = "",
    paper_id: str = "",
    assets_by_key: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate summary and hypothetical questions for a text chunk using LLM.

    This implements the "Hypothetical Document Embeddings" (HyDE) approach:
    1. Generate a concise summary (1-2 sentences, grounded in both paper-level
       and chunk-level context)
    2. Generate hypothetical questions a user might ask to find this content
    3. Extract proper nouns, datasets, metrics, and numeric values

    Args:
        title: Paper title (provides global context)
        abstract: Paper abstract (provides global context)

    Returns:
        {
            "summary": "...",
            "hypothetical_questions": ["...", ...],
            "keywords": ["...", ...],
            "entities": ["ProperNoun", "DatasetName", "MetricName", ...],
            "claims": ["Key claim 1", "Key claim 2", ...]
        }
    """
    # Fallback if no LLM provider available
    if not provider or not model or not summarize:
        keywords = _extract_keywords(text, max_items=6)
        fallback_questions = [
            f"What is discussed in the {section} section?",
            f"How does the {section} relate to the main topic?",
            f"What are the key findings in {section}?",
        ]
        return {
            "summary": _norm(text)[:200] + ("..." if len(text) > 200 else ""),
            "hypothetical_questions": fallback_questions[:num_questions],
            "keywords": keywords,
            "entities": [],
            "claims": [],
        }
    
    # Build paper-level context (truncated for prompt budget)
    paper_context_parts: list[str] = []
    if title:
        paper_context_parts.append(f"Paper Title: {title[:300]}")
    if abstract:
        paper_context_parts.append(f"Paper Abstract: {abstract[:800]}")
    paper_context = "\n".join(paper_context_parts) if paper_context_parts else "(no paper-level context)"
    
    prompt = """You are a scientific paper analyzer. Given a paper's metadata and a text excerpt from one of its sections, produce a structured JSON analysis.

## Paper-level Context (for reference)
{paper_context}

## Chunk to Analyze
Section: {section}
Text excerpt:
{text_excerpt}

## Output JSON Structure

```json
{{
  "summary": "1-2 sentence summary that connects paper-level purpose with this section's specific content.",
  "hypothetical_questions": [
    "Question 1: natural search query a user would type",
    "Question 2: ...",
    "Question 3: ..."
  ],
  "keywords": ["technical term 1", "technical term 2", ...],
  "entities": ["ProperNoun", "DatasetName", "MetricName", "NumericValue with unit", ...],
  "claims": ["This section claims that X outperforms Y by Z%", ...]
}}
```

### summary
- 1-2 sentences capturing what THIS section contributes to the paper's overall argument
- Include paper title reference so the summary can stand alone

### hypothetical_questions
- Generate exactly {num_questions} natural search queries a user might type
- Questions should reference concrete details: model names, datasets, metrics
- NOT generic like "What is discussed here?" — be specific

### keywords
- 5-10 most important technical terms from this chunk

### entities
- Proper nouns: model names, architecture names, framework names (e.g., "BERT", "ResNet-50")
- Dataset names: (e.g., "ImageNet", "SQuAD v2.0")
- Metric names: (e.g., "BLEU", "F1-score", "top-1 accuracy")
- Specific numeric values with context: (e.g., "94.7% accuracy", "3.2x speedup")
- Author/institution names mentioned in this chunk

### claims
- Concise factual claims made in this chunk
- Include numbers where present (e.g., "Transformer achieves 28.4 BLEU on WMT 2014 EN-DE")
- One claim per array item, max 5

## Critical Rules
- ALL output fields must be grounded in the provided text — NO hallucination
- Return ONLY the JSON object, no markdown fences, no explanation
- If nothing to extract for a field, return []""".format(
        num_questions=num_questions,
        section=section,
        text_excerpt=text[:4096] + ("..." if len(text) > 4096 else ""),
        paper_context=paper_context,
    )
    
    try:
        resp = await provider.chat_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise JSON generator for scientific paper analysis. Extract ALL named entities, datasets, metrics, and specific values."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            tool_choice=None,
        )
        raw = (resp.content or "").strip()
        
        # Handle potential markdown code block wrapping
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        
        import json_repair
        
        payload = json_repair.loads(raw)
        
        # Validate and normalize the response
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            summary = _norm(text)[:200]
        
        questions = payload.get("hypothetical_questions", [])
        if isinstance(questions, list):
            questions = [str(q).strip() for q in questions if q]
        else:
            questions = []
        questions = questions[:num_questions]
        if len(questions) < num_questions:
            questions.extend([
                f"How does the {section} section contribute to the overall findings?",
                f"What are the key results reported in the {section} section?",
            ][:num_questions - len(questions)])
        
        keywords = payload.get("keywords", [])
        if isinstance(keywords, list):
            keywords = [str(k).strip() for k in keywords if k][:10]
        else:
            keywords = _extract_keywords(text)
        
        entities = payload.get("entities", [])
        if isinstance(entities, list):
            entities = [str(e).strip() for e in entities if e and str(e).strip()]
        else:
            entities = []
        
        claims = payload.get("claims", [])
        if isinstance(claims, list):
            claims = [str(c).strip() for c in claims if c and str(c).strip()][:5]
        else:
            claims = []
        
        logger.info(
            "Generated chunk metadata: summary_len={}, questions={}, keywords={}, entities={}, claims={}",
            len(summary),
            len(questions),
            len(keywords),
            len(entities),
            len(claims),
        )
        
        return {
            "summary": summary,
            "hypothetical_questions": questions,
            "keywords": keywords,
            "entities": entities,
            "claims": claims,
        }
    except Exception as e:
        logger.warning("LLM metadata generation failed: {}, using fallback", e)
        keywords = _extract_keywords(text, max_items=6)
        fallback_questions = [
            f"What is discussed in the {section} section?",
            f"How does the {section} relate to the main topic?",
            f"What are the key findings in {section}?",
        ]
        return {
            "summary": _norm(text)[:200],
            "hypothetical_questions": fallback_questions[:num_questions],
            "keywords": keywords,
            "entities": [],
            "claims": [],
        }


@tool_parameters(
    tool_parameters_schema(
        paper=ObjectSchema(
            properties={
                "paper_id": StringSchema("paper id"),
                "title": StringSchema("title"),
                "url": StringSchema("paper url"),
                "pdf_url": StringSchema("pdf url"),
                "source": StringSchema("source"),
                "year": IntegerSchema(description="year"),
                "venue": StringSchema("venue"),
            },
            required=["title"],
        ),
        papers=ArraySchema(
            ObjectSchema(
                properties={
                    "paper_id": StringSchema("paper id"),
                    "title": StringSchema("title"),
                    "url": StringSchema("paper url"),
                    "pdf_url": StringSchema("pdf url"),
                    "source": StringSchema("source"),
                    "year": IntegerSchema(description="year"),
                    "venue": StringSchema("venue"),
                },
                required=["title"],
            ),
            description="Batch papers for parallel ingest",
            min_items=1,
            max_items=30,
        ),
        parse_mode=StringSchema("parse mode", enum=["auto", "pdf", "text"]),
        concurrency=IntegerSchema(3, minimum=1, maximum=10),
        summarize=BooleanSchema(description="Summarize sections before upsert", default=True),
        keep_pdf=BooleanSchema(description="Keep the downloaded PDF after successful parsing", default=False),
    )
)
class PaperIngestTool(_PaperTool):
    name = "paper_ingest"
    description = (
        "Download, parse, chunk, and upsert selected paper(s) to local knowledge base. "
        "This tool prepares papers for future retrieval but does NOT return knowledge directly. "
        "After ingestion, you MUST call `kb_retrieve` to query the knowledge base for grounded evidence. "
        "Supports batch parallel ingestion with optional LLM-based summarization."
    )

    def __init__(
        self,
        workspace: Path,
        kb: PaperKnowledgeBase,
        provider: LLMProvider | None = None,
        model: str | None = None,
        *,
        mineru_api_token: str = "",
    ) -> None:
        super().__init__(workspace=workspace, kb=kb, provider=provider, model=model)
        self.mineru_api_token = mineru_api_token

    @property
    def read_only(self) -> bool:
        return False

    async def _summarize_section(self, section: str, text: str) -> dict[str, Any]:
        if not self.provider or not self.model:
            return _fallback_section_summary(section, text)

        prompt = (
            "Summarize this paper section into strict JSON with keys: "
            "summary(string), claims(array of strings), limitations(array of strings), keywords(array of strings). "
            "Keep summary concise and factual.\n\n"
            f"Section: {section}\n"
            f"Content:\n{text}"
        )
        try:
            resp = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a scientific paper summarizer."},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            raw = (resp.content or "").strip()
            import json_repair
            payload = json_repair.loads(raw)
            logger.info("paper_ingest: summary_len={} claims={} limitations={} keywords={}", len(str(payload.get("summary", ""))), len(payload.get("claims", [])), len(payload.get("limitations", [])), len(payload.get("keywords", [])))
            return {
                "section": section,
                "summary": str(payload.get("summary", "")).strip(),
                "claims": payload.get("claims", []) if isinstance(payload.get("claims", []), list) else [],
                "limitations": payload.get("limitations", []) if isinstance(payload.get("limitations", []), list) else [],
                "keywords": payload.get("keywords", []) if isinstance(payload.get("keywords", []), list) else [],
            }
        except Exception:
            logger.warning("paper_ingest: section summarization failed, fallback to extractive")
            return _fallback_section_summary(section, text)

    async def _distill_text(self, text_content: str) -> list[dict[str, Any]]:
        sections = _split_sections(text_content)
        if not sections:
            return []
        distilled: list[dict[str, Any]] = []
        for idx, (section, body) in enumerate(sections):
            info = await self._summarize_section(section, body)
            summary = _norm(str(info.get("summary", "")))
            if not summary:
                continue
            distilled.append(
                {
                    "section": section,
                    "kind": "distilled_summary",
                    "text": summary,
                    "keywords": info.get("keywords", []),
                    "claims": info.get("claims", []),
                    "limitations": info.get("limitations", []),
                    "source_span": idx,
                }
            )
        return distilled

    async def _ingest_one(
        self,
        paper: dict[str, Any],
        parse_mode: str = "auto",
        summarize: bool = True,
        use_hypothetical: bool = True,
        keep_pdf: bool = False,
    ) -> dict[str, Any]:
        """Internalize one paper into the knowledge base.
        
        Args:
            paper: Paper metadata dict
            parse_mode: "auto" | "pdf" | "text"
            summarize: Whether to generate LLM summaries
            use_hypothetical: Use hypothetical question retrieval (HyDE approach)
        
        Returns:
            {"status": "ok" | "error", "paper_id", "chunk_count", ...}
        """
        url = str(paper.get("pdf_url") or paper.get("url") or "")
        if not paper.get("pdf_url") and "arxiv.org" in url:
            url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")
        if not url:
            return {"status": "error", "error": "paper.url or paper.pdf_url is required", "paper": paper}

        url_ok, url_error = validate_url_target(url)
        if not url_ok:
            return {"status": "error", "error": f"unsafe_url: {url_error}", "paper": paper}

        downloads_dir = self.workspace / "kb" / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        paper_id = str(paper.get("paper_id") or "paper")
        safe_paper_id = _safe_paper_id(paper_id)
        local_pdf = downloads_dir / f"{safe_paper_id}.pdf"
        # Keep a .pdf suffix so both MinerU and the local extractor select the
        # correct parser while the leading dot still marks the file temporary.
        temp_pdf = downloads_dir / f".{safe_paper_id}.{uuid.uuid4().hex}.part.pdf"
        local_md = downloads_dir / f"{safe_paper_id}.md"
        text_content = ""
        try:
            async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
                logger.info("paper_ingest: downloading paper_id={} url={}", paper_id, url)
                resp = await client.get(url)
                resp.raise_for_status()
            data = resp.content
            if len(data) > MAX_PAPER_DOWNLOAD_BYTES:
                return {
                    "status": "error",
                    "error": f"paper exceeds {MAX_PAPER_DOWNLOAD_BYTES // (1024 * 1024)}MB limit",
                    "paper_id": paper_id,
                }

            response_url = str(getattr(resp, "url", "") or "")
            if response_url:
                redirect_ok, redirect_error = validate_resolved_url(response_url)
                if not redirect_ok:
                    return {
                        "status": "error",
                        "error": f"unsafe_redirect: {redirect_error}",
                        "paper_id": paper_id,
                    }

            headers = getattr(resp, "headers", {}) or {}
            content_type = str(headers.get("content-type", "")).lower()
            is_pdf = data.startswith(b"%PDF") or "application/pdf" in content_type
            if parse_mode == "pdf" and not is_pdf:
                return {"status": "error", "error": "expected_pdf_content", "paper_id": paper_id}

            if parse_mode in {"auto", "pdf"} and is_pdf:
                temp_pdf.write_bytes(data)
                try:
                    if self.mineru_api_token:
                        from langchain_mineru import MinerULoader

                        loader = MinerULoader(
                            source=str(temp_pdf),
                            mode="precision",
                            token=self.mineru_api_token,
                        )
                        docs = await asyncio.to_thread(loader.load)
                        text_content = docs[0].page_content if docs else ""
                except Exception as exc:
                    logger.warning("paper_ingest: MinerU parse failed for {}: {}", paper_id, exc)

                if not _valid_extracted_text(text_content):
                    extracted = await asyncio.to_thread(extract_text, temp_pdf)
                    text_content = extracted if isinstance(extracted, str) else ""

                if keep_pdf and _valid_extracted_text(text_content):
                    temp_pdf.replace(local_pdf)
            elif parse_mode in {"auto", "text"}:
                try:
                    text_content = data.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    text_content = ""

            if not _valid_extracted_text(text_content):
                logger.warning("paper_ingest: invalid extracted text for paper_id={} url={}", paper_id, url)
                return {"status": "error", "error": "failed_to_parse_content", "paper_id": paper_id, "url": url}

            text_content = _remove_noisy_blocks(text_content)
            # Clean MinerU artifacts and re-flow figures/tables
            # Order matters: image sequence ops and figure reflow need raw ![]() links
            text_content = _merge_consecutive_images(text_content)
            text_content = _reflow_figures_tables(text_content)
            text_content = _replace_unparsed_images(text_content)
            text_content = _merge_split_paragraphs(text_content)
            # Save front matter before stripping, for metadata extraction
            raw_front_matter = _extract_front_matter(text_content)
            # Strip front matter (title/authors/abstract before Introduction)
            text_content = _strip_front_matter(text_content)
            logger.info("paper_ingest: writing text content to local_md={} (length={})", local_md, len(text_content))
            local_md.write_text(text_content, encoding="utf-8")

            # Enrich paper metadata from front matter when arXiv data is missing
            if (not paper.get("authors") or not paper.get("abstract")) and raw_front_matter:
                fm_meta = await _parse_front_matter_metadata(
                    raw_front_matter, self.provider, self.model,
                )
                if fm_meta.get("title"):
                    paper["title"] = fm_meta["title"]
                if fm_meta.get("authors"):
                    paper["authors"] = fm_meta["authors"]
                if fm_meta.get("abstract"):
                    paper["abstract"] = fm_meta["abstract"]
                if fm_meta.get("year"):
                    paper["year"] = fm_meta["year"]
                logger.info(
                    "paper_ingest: enriched metadata from front matter: title='{}' authors={} year={}",
                    paper.get("title", "")[:60],
                    len(paper.get("authors", [])),
                    paper.get("year", ""),
                )
            
            # Use hypothetical question retrieval approach
            if use_hypothetical and self.kb.config.enable_hypothetical_retrieval:
                # 1. Split into semantic chunks based on Markdown headers
                semantic_chunks = _split_markdown_semantic(
                    text_content,
                    max_chunk_chars=self.kb.config.max_chunk_chars,
                    min_chunk_chars=self.kb.config.min_chunk_chars,
                )
                
                if not semantic_chunks:
                    logger.warning("paper_ingest: no semantic chunks generated for paper_id={}", paper_id)
                    return {"status": "error", "error": "no_chunks_generated", "paper_id": paper_id}
                
                # 2. Generate metadata (summary + hypothetical questions) for each chunk
                chunk_metadata: list[dict[str, Any]] = []
                for chunk in semantic_chunks:
                    meta = await _generate_chunk_metadata(
                        text=chunk.get("text", ""),
                        section=chunk.get("section", "content"),
                        provider=self.provider,
                        model=self.model,
                        num_questions=self.kb.config.num_hypothetical_questions,
                        summarize=summarize,
                        title=paper.get("title", ""),
                        abstract=paper.get("abstract", ""),
                    )
                    chunk_metadata.append(meta)
                
                # 3. Upsert to Chroma with multiple embeddings
                logger.info("paper_ingest: upserting paper_id={} with {} semantic chunks and metadata", paper_id, len(semantic_chunks))
                result = await self.kb.upsert_semantic_chunks(
                    doc=paper,
                    semantic_chunks=semantic_chunks,
                    chunk_metadata=chunk_metadata,
                )
                
                logger.info(
                    "paper_ingest (hypothetical): paper_id={} chunks={} questions={} url={}",
                    result.get("paper_id"),
                    result.get("chunk_count"),
                    result.get("question_count"),
                    url,
                )
                return {
                    "status": "ok",
                    "paper_id": result.get("paper_id"),
                    "chunk_count": result.get("chunk_count"),
                    "question_count": result.get("question_count"),
                    "mode": "hypothetical",
                    "storage_backend": result.get("storage_backend", "unknown"),
                    "degraded": bool(result.get("degraded", False)),
                    "embedding": result.get("embedding", {}),
                    "lexical": result.get("lexical", {}),
                    "degradation_reasons": result.get("degradation_reasons", []),
                    "local_file": str(local_md),
                    "next_step_hint": "After ingestion, call `kb_retrieve(query, top_k=...)` to retrieve relevant knowledge chunks.",
                }
            else:
                # Traditional approach: section-level summarization
                distilled = await self._distill_text(text_content) if summarize else []
                result = await self.kb.upsert_document(paper, text_content, distilled_chunks=distilled or None)
                logger.info(
                    "paper_ingest (traditional): paper_id={} chunks={} distilled_len={} url={}",
                    result.get("paper_id"),
                    result.get("chunk_count"),
                    len(result.get("distilled", "")) if result.get("distilled") else 0,
                    url,
                )
                return {
                    "status": "ok",
                    "paper_id": result.get("paper_id"),
                    "chunk_count": result.get("chunk_count"),
                    "mode": "traditional",
                    "storage_backend": result.get("storage_backend", "jsonl"),
                    "degraded": bool(result.get("degraded", False)),
                    "embedding": result.get("embedding", {}),
                    "lexical": result.get("lexical", {}),
                    "local_file": str(local_md),
                    "next_step_hint": "After ingestion, call `kb_retrieve(query, top_k=...)` to retrieve relevant knowledge chunks.",
                }
        except Exception as exc:
            logger.warning("paper_ingest: failed to ingest paper_id={} url={} error={}", paper_id, url, exc)
            return {"status": "error", "error": str(exc), "paper_id": paper_id, "url": url}
        finally:
            with contextlib.suppress(OSError):
                temp_pdf.unlink(missing_ok=True)

    async def execute(
        self,
        paper: dict[str, Any] | None = None,
        papers: list[dict[str, Any]] | None = None,
        parse_mode: str = "auto",
        concurrency: int = 3,
        summarize: bool = True,
        keep_pdf: bool = False,
        **kwargs: Any,
    ) -> str:
        targets: list[dict[str, Any]] = []
        if isinstance(paper, dict):
            targets.append(paper)
        if papers:
            targets.extend([p for p in papers if isinstance(p, dict)])
        if not targets:
            return json.dumps(
                {"error": "paper or papers is required"},
                ensure_ascii=False,
            )

        if len(targets) == 1:
            result = await self._ingest_one(
                targets[0],
                parse_mode=parse_mode,
                summarize=summarize,
                keep_pdf=keep_pdf,
            )
            return json.dumps(result, ensure_ascii=False)

        sem = asyncio.Semaphore(max(1, min(concurrency, 10)))

        async def _run_one(item: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await self._ingest_one(
                    item,
                    parse_mode=parse_mode,
                    summarize=summarize,
                    keep_pdf=keep_pdf,
                )

        batch = await asyncio.gather(*[_run_one(p) for p in targets])
        succeeded = [r for r in batch if r.get("status") == "ok"]
        failed = [r for r in batch if r.get("status") != "ok"]
        return json.dumps(
            {
                "status": "ok" if succeeded else "error",
                "total": len(batch),
                "succeeded": len(succeeded),
                "failed": len(failed),
                "results": batch,
                "next_step_hint": "After ingestion, call `kb_retrieve(query, top_k=...)` to retrieve relevant knowledge chunks.",
            },
            ensure_ascii=False,
        )


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("Query text"),
        top_k=IntegerSchema(5, minimum=1, maximum=30),
        prefer_distilled=BooleanSchema(description="Prefer distilled summary chunks", default=True),
        per_paper_limit=IntegerSchema(2, minimum=1, maximum=10),
        retrieval_mode=StringSchema(
            "Retrieval mode: 'hypothetical' (question view), 'traditional' (parent chunks), 'hybrid' (parent + summary + question views)",
            enum=["hypothetical", "traditional", "hybrid"],
        ),
        required=["query"],
    )
)
class KBRetrieveTool(_PaperTool):
    name = "kb_retrieve"
    description = (
        "Retrieve semantically relevant chunks from local paper KB for grounded answers. "
        "Supports three retrieval modes: "
        "- 'hypothetical': Search via hypothetical question embeddings (HyDE approach, best for natural language queries) "
        "- 'traditional': Search via parent document embeddings directly "
        "- 'hybrid': Combine parent, summary, and question views for best recall"
    )

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        prefer_distilled: bool = True,
        per_paper_limit: int = 3,
        retrieval_mode: str = "hypothetical",
        **kwargs: Any,
    ) -> str:
        search_modes = {
            "hypothetical": "questions_only",
            "traditional": "chunks_only",
            "hybrid": "hybrid",
        }
        if retrieval_mode not in search_modes:
            return json.dumps(
                {"error": f"Unsupported retrieval_mode: {retrieval_mode}"},
                ensure_ascii=False,
            )

        # When Chroma multi-view retrieval is enabled, all public modes use the
        # same fusion path but activate different vector and FTS fields.
        if self.kb.config.enable_hypothetical_retrieval:
            search_mode = search_modes[retrieval_mode]
            results = await self.kb.retrieve_by_hypothetical_questions(
                query,
                top_k=top_k,
                per_paper_limit=per_paper_limit,
                search_mode=search_mode,
            )
            logger.info(
                "kb_retrieve: query='{}' mode={} search_mode={} top_k={} per_paper_limit={} hits={}",
                query,
                retrieval_mode,
                search_mode,
                top_k,
                per_paper_limit,
                len(results),
            )
        else:
            # Traditional retrieval
            results = await self.kb.retrieve(
                query,
                top_k=top_k,
                prefer_distilled=prefer_distilled,
                per_paper_limit=per_paper_limit,
            )
            logger.info(
                "kb_retrieve (traditional): query='{}' top_k={} prefer_distilled={} per_paper_limit={} hits={}",
                query,
                top_k,
                prefer_distilled,
                per_paper_limit,
                len(results),
            )
        return json.dumps(
            {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "results": results,
                "embedding": self.kb.get_embedding_status(),
                "lexical": self.kb.get_lexical_status(),
            },
            ensure_ascii=False,
        )
