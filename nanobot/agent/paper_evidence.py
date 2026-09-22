"""Shared paper evidence representation and renderers.

The ordinary agent and the multi-agent workflow intentionally expose evidence
in different wire formats (compact JSON versus XML).  Keeping the source
assembly here ensures both paths agree on paper metadata, evidence levels and
the citation IDs that an answer is allowed to use.
"""

from __future__ import annotations

import re
import xml.sax.saxutils as saxutils
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidencePaper:
    """Paper-level evidence plus the retrieved full-text chunks, if any."""

    paper_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Any = ""
    url: str = ""
    abstract: str = ""
    evidence_level: str = "unknown"
    chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceBundle:
    """Neutral evidence shared by renderers and citation validation."""

    papers: list[EvidencePaper] = field(default_factory=list)
    query: str = ""
    queries: list[str] = field(default_factory=list)
    quality: str = "unknown"
    quality_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed_citation_ids(self) -> list[str]:
        return [paper.paper_id for paper in self.papers if paper.paper_id]

    def source_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "url": paper.url,
                "evidence_level": paper.evidence_level,
            }
            for paper in self.papers
            if paper.paper_id
        ]


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _authors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value not in (None, ""):
        return [str(value)]
    return []


def _chunk_source_order(chunk: dict[str, Any]) -> tuple[int, int, int, str]:
    chunk_id = str(chunk.get("chunk_id", ""))
    chunk_index = _as_int(chunk.get("chunk_index"))
    if chunk_index is None:
        chunk_index = _as_int(chunk_id.rsplit(":", 1)[-1])
    page_start = _as_int(chunk.get("page_start"))
    if chunk_index is not None:
        return (0, chunk_index, page_start if page_start is not None else 10**9, chunk_id)
    return (1, page_start if page_start is not None else 10**9, 10**9, chunk_id)


def _normalize_queries(query: str, queries: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in [query, *(queries or [])]:
        item = re.sub(r"\s+", " ", str(value or "")).strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            normalized.append(item)
    return normalized


def build_evidence_bundle(
    retrieval_results: list[dict[str, Any]] | None = None,
    docs_meta: dict[str, dict[str, Any]] | None = None,
    external_papers: list[dict[str, Any]] | None = None,
    *,
    query: str = "",
    queries: list[str] | None = None,
    quality: str = "unknown",
    quality_reason: str = "",
    diagnostics: dict[str, Any] | None = None,
) -> EvidenceBundle:
    """Assemble raw retrieval/search output into one paper-centric bundle."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in retrieval_results or []:
        if not isinstance(raw, dict):
            continue
        paper_id = str(raw.get("paper_id") or "unknown")
        grouped.setdefault(paper_id, []).append(dict(raw))

    papers: list[EvidencePaper] = []
    full_text_ids: set[str] = set()
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: max(
            (float(chunk.get("score", 0) or 0) for chunk in item[1]),
            default=0,
        ),
        reverse=True,
    )
    for paper_id, raw_chunks in sorted_groups:
        full_text_ids.add(paper_id)
        first = raw_chunks[0]
        meta = (docs_meta or {}).get(paper_id, {})
        title = str(
            meta.get("title")
            or first.get("paper_title")
            or first.get("title")
            or paper_id
        )
        papers.append(EvidencePaper(
            paper_id=paper_id,
            title=title,
            authors=_authors(meta.get("authors") or first.get("authors")),
            year=(
                meta.get("year")
                or first.get("paper_year")
                or first.get("year")
                or ""
            ),
            url=str(meta.get("url") or first.get("url") or ""),
            abstract=str(meta.get("abstract") or first.get("abstract") or ""),
            evidence_level="full_text",
            chunks=sorted(raw_chunks, key=_chunk_source_order),
        ))

    for raw in external_papers or []:
        if not isinstance(raw, dict):
            continue
        paper_id = str(raw.get("paper_id") or raw.get("arxiv_id") or "").strip()
        if not paper_id or paper_id in full_text_ids:
            continue
        papers.append(EvidencePaper(
            paper_id=paper_id,
            title=str(raw.get("title") or paper_id),
            authors=_authors(raw.get("authors")),
            year=raw.get("year") or "",
            url=str(raw.get("url") or ""),
            abstract=str(raw.get("abstract") or "")[:3000],
            evidence_level="abstract_only",
        ))

    normalized_queries = _normalize_queries(query, queries)
    return EvidenceBundle(
        papers=papers,
        query=str(query or ""),
        queries=normalized_queries,
        quality=str(quality or "unknown"),
        quality_reason=str(quality_reason or ""),
        diagnostics=dict(diagnostics or {}),
    )


def flatten_evidence_results(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    """Return chunk-shaped JSON enriched with its canonical paper metadata."""
    flattened: list[dict[str, Any]] = []
    for paper in bundle.papers:
        for chunk in paper.chunks:
            item = dict(chunk)
            item["paper_id"] = paper.paper_id
            item["paper_title"] = paper.title
            item["authors"] = list(paper.authors)
            item["abstract"] = paper.abstract
            item["evidence_level"] = paper.evidence_level
            if paper.year not in (None, ""):
                item["paper_year"] = paper.year
            if paper.url:
                item["url"] = paper.url
            flattened.append(item)
    return flattened


def render_evidence_bundle_xml(bundle: EvidenceBundle) -> str:
    """Render an evidence bundle for the multi-agent synthesis prompt."""
    parts: list[str] = []
    for paper in bundle.papers:
        paper_id = saxutils.escape(paper.paper_id)
        title = saxutils.escape(paper.title)
        authors = paper.authors[:5]
        authors_text = ", ".join(authors)
        if len(paper.authors) > 5:
            authors_text += " et al."

        if paper.evidence_level == "full_text":
            parts.append(f'  <paper id="{paper_id}" evidence_level="full_text">')
            parts.append("    <metadata>")
            parts.append(f"      <title>{title}</title>")
            if authors_text:
                parts.append(
                    f"      <authors>{saxutils.escape(authors_text)}</authors>"
                )
            if paper.year not in (None, ""):
                parts.append(f"      <year>{saxutils.escape(str(paper.year))}</year>")
            parts.append("    </metadata>")
            if paper.abstract:
                parts.append("    <global_abstract>")
                parts.append(f"      {saxutils.escape(paper.abstract)}")
                parts.append("    </global_abstract>")

            parts.append("    <retrieved_chunks>")
            for chunk in paper.chunks:
                section = chunk.get("heading_path", chunk.get("section", ""))
                text = saxutils.escape(str(chunk.get("text", "")))
                score = float(chunk.get("score", 0) or 0)
                chunk_index = _as_int(chunk.get("chunk_index"))
                if chunk_index is None:
                    chunk_index = _as_int(
                        str(chunk.get("chunk_id", "")).rsplit(":", 1)[-1]
                    )
                page_start = _as_int(chunk.get("page_start"))
                order_attributes = ""
                if chunk_index is not None:
                    order_attributes += f' chunk_index="{chunk_index}"'
                if page_start is not None:
                    order_attributes += f' page_start="{page_start}"'
                parts.append(
                    f'      <chunk section="{saxutils.escape(str(section))}" '
                    f'score="{score:.3f}"{order_attributes}>'
                )
                parts.append(f"        {text}")
                parts.append("      </chunk>")
            parts.append("    </retrieved_chunks>")

            seen_assets: set[str] = set()
            assets: list[dict[str, Any]] = []
            for chunk in paper.chunks:
                for asset in chunk.get("linked_assets", []):
                    key = str(asset.get("key", ""))
                    if key and key not in seen_assets:
                        seen_assets.add(key)
                        assets.append(asset)
            if assets:
                parts.append("    <referenced_figures_tables>")
                for asset in assets:
                    key = saxutils.escape(str(asset.get("key", "")))
                    caption = saxutils.escape(str(asset.get("caption", "")))
                    asset_type = str(asset.get("type", "figure"))
                    content = asset.get("content", "")
                    line = (
                        f'      <asset key="{key}" '
                        f'type="{saxutils.escape(asset_type)}" caption="{caption}"'
                    )
                    if content and asset_type == "table":
                        parts.append(line + ">")
                        parts.append(f"        {saxutils.escape(str(content))}")
                        parts.append("      </asset>")
                    else:
                        parts.append(line + " />")
                parts.append("    </referenced_figures_tables>")
            parts.append("  </paper>")
            continue

        parts.append(f'  <paper id="{paper_id}" evidence_level="abstract_only">')
        parts.append(f"    <title>{title}</title>")
        if authors_text:
            parts.append(f"    <authors>{saxutils.escape(authors_text)}</authors>")
        if paper.year not in (None, ""):
            parts.append(f"    <year>{saxutils.escape(str(paper.year))}</year>")
        if paper.url:
            parts.append(f"    <url>{saxutils.escape(paper.url)}</url>")
        if paper.abstract:
            parts.append(f"    <abstract>{saxutils.escape(paper.abstract)}</abstract>")
        parts.append(
            "    <usage_constraint>Abstract-only evidence; do not infer unreported "
            "method details or experiment values.</usage_constraint>"
        )
        parts.append("  </paper>")

    if not parts:
        return "<sources>No external sources provided. Answer based on your knowledge.</sources>"
    return "<sources>\n" + "\n".join(parts) + "\n</sources>"


def collect_answer_citations(
    answer: str,
    *,
    bundle: EvidenceBundle | None = None,
    retrieval_results: list[dict[str, Any]] | None = None,
    docs_meta: dict[str, dict[str, Any]] | None = None,
    external_papers: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve inline ``[paper_id]`` markers and report fabricated IDs."""
    evidence = bundle or build_evidence_bundle(
        retrieval_results,
        docs_meta,
        external_papers,
    )
    available = {paper.paper_id: paper for paper in evidence.papers}
    marker_ids = list(dict.fromkeys(re.findall(
        r"\[([A-Za-z0-9][A-Za-z0-9._:/-]{1,127})\](?!\()",
        answer or "",
    )))
    cited_ids = [paper_id for paper_id in marker_ids if paper_id in available]
    invalid = [
        paper_id
        for paper_id in marker_ids
        if paper_id not in available
        and any(character.isdigit() for character in paper_id)
        and any(separator in paper_id for separator in (".", ":", "-"))
    ]
    citations: list[str] = []
    for paper_id in cited_ids:
        paper = available[paper_id]
        citations.append(
            f"[{paper_id}] {paper.title or paper_id}"
            + (f" — {paper.url}" if paper.url else "")
            + f" ({paper.evidence_level})"
        )
    return citations, invalid
