"""Node implementations for multi-agent workflow.

Each node represents an agent in the workflow graph.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import json_repair
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from loguru import logger

from nanobot.agent.multi_agent.agents import (
    CRITIC_USER_TEMPLATE,
    ENTITY_EXTRACT_SYSTEM_PROMPT,
    ENTITY_EXTRACT_USER_TEMPLATE,
    HISTORY_COMPRESS_SYSTEM_PROMPT,
    HISTORY_COMPRESS_USER_TEMPLATE,
    QUERY_DECOMPOSE_SYSTEM_PROMPT,
    QUERY_DECOMPOSE_USER_TEMPLATE,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_TEMPLATE,
    RESEARCH_USER_TEMPLATE,
    RETRIEVAL_USER_TEMPLATE,
    ROUTER_USER_TEMPLATE,
    SYNTHESIS_USER_TEMPLATE,
    UNIFIED_QUERY_SYSTEM_PROMPT,
    UNIFIED_QUERY_USER_TEMPLATE,
    format_sources_section,
    get_agent_prompts,
)
from nanobot.agent.multi_agent.state import MultiAgentState

if TYPE_CHECKING:
    from nanobot.agent.memory import MemoryStore
    from nanobot.agent.paper_kb import PaperKnowledgeBase
    from nanobot.agent.tools.paper import (
        KBRetrieveTool,
        PaperIngestTool,
        PaperRerankTool,
        PaperSearchTool,
        PaperSimilarityTool,
    )
    from nanobot.providers.base import LLMProvider


class AgentNodes:
    """Container for all agent node implementations."""
    
    def __init__(
        self,
        provider: LLMProvider,
        kb: PaperKnowledgeBase,
        tools: Dict[str, Any],
        similarity_threshold: float = 0.2,
        top_k: int = 5,
        ingest_limit: int = 3,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
        memory_store: "MemoryStore | None" = None,
    ):
        self.provider = provider
        self.kb = kb
        self.tools = tools
        self.prompts = get_agent_prompts(similarity_threshold, top_k, ingest_limit)
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.ingest_limit = ingest_limit
        self._progress_callback = progress_callback
        self._memory_store = memory_store

    @staticmethod
    def _results_preview(results: list[dict[str, Any]], limit: int = 5) -> str:
        if not results:
            return "(no retrieval results)"
        lines: list[str] = []
        for idx, item in enumerate(results[:limit], 1):
            title = str(item.get("paper_title") or item.get("title") or item.get("paper_id") or "unknown")
            section = str(item.get("section", ""))
            score = float(item.get("score", 0.0) or 0.0)
            lines.append(f"{idx}. score={score:.3f}; section={section}; title={title[:120]}")
        return "\n".join(lines)

    async def _emit_progress(self, msg: str) -> None:
        """Send a progress update via the callback, if configured."""
        if self._progress_callback:
            try:
                ret = self._progress_callback(msg)
                if asyncio.isfuture(ret) or inspect.isawaitable(ret):
                    await ret
            except Exception as e:
                logger.debug("Progress callback failed: {}", e)

    def _refresh_memory_context(self, state: MultiAgentState) -> None:
        if not self._memory_store:
            return
        try:
            long_term = self._memory_store.read_memory() or "(empty)"
            user_profile = self._memory_store.read_user() or "(empty)"
            soul = self._memory_store.read_soul() or "(empty)"
        except Exception as e:
            logger.debug("Memory refresh failed: {}", e)
            return
        state["long_term_memory_context"] = long_term
        state["user_profile_context"] = user_profile
        state["soul_context"] = soul

    # ================================================================
    # Query Rewrite Methods
    # ================================================================

    async def _compress_history_for_rewrite(
        self,
        user_query: str,
        state: MultiAgentState,
    ) -> str:
        """Compress conversation history into a concise context block for query rewriting.
        
        Produces a short JSON with intent summary, last source, failure reason,
        key entities, and active constraints.
        
        Args:
            user_query: The current user query
            state: The workflow state
            
        Returns:
            Compressed context string (JSON) or empty string on failure
        """
        rewrite_history_context = state.get("rewrite_history_context", "")
        if rewrite_history_context:
            return str(rewrite_history_context)
        
        retrieval_history = str(state.get("retrieval_history_context", "") or "(empty)")
        research_history = str(state.get("research_history_context", "") or "(empty)")
        recent_dialog = str(state.get("recent_dialog_context", "") or "(empty)")
        # long_term_memory = str(state.get("long_term_memory_context", "") or "(empty)")
        # user_profile = str(state.get("user_profile_context", "") or "(empty)")
        long_term_memory = "(empty)"
        user_profile = "(empty)"
        session_summary = str(state.get("session_summary_context", "") or "(empty)")
        last_routing = str(state.get("last_routing_decision", "") or "none")
        last_quality = str(state.get("retrieval_quality", "") or "unknown")
        rewrite_context_chars = int(state.get("rewrite_context_chars", 60000) or 60000)
        
        # If no usable context at all, return empty
        if (
            retrieval_history == "(empty)"
            and research_history == "(empty)"
            and recent_dialog == "(empty)"
            and long_term_memory == "(empty)"
            and user_profile == "(empty)"
            and session_summary == "(empty)"
        ):
            return ""
        
        prompt = HISTORY_COMPRESS_USER_TEMPLATE.format(
            user_query=user_query[:rewrite_context_chars],
            recent_dialog_context=recent_dialog[:rewrite_context_chars],
            long_term_memory_context=long_term_memory[:rewrite_context_chars],
            user_profile_context=user_profile[:rewrite_context_chars],
            session_summary_context=session_summary[:rewrite_context_chars],
            retrieval_history_context=retrieval_history[:rewrite_context_chars],
            research_history_context=research_history[:rewrite_context_chars],
            last_routing_decision=last_routing,
            last_retrieval_quality=last_quality,
        )

        logger.info("Compressing history for query rewrite: {}", prompt)
        
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": HISTORY_COMPRESS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            content = (response.content or "").strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            # Store for reuse
            state["rewrite_history_context"] = json.dumps(result, ensure_ascii=False)
            
            # Extract referenced_papers from compression result for downstream entity resolution
            referenced_papers = result.get("referenced_papers", [])
            if referenced_papers and isinstance(referenced_papers, list):
                state["referenced_papers"] = [
                    {
                        "paper_id": str(p.get("paper_id", "")).strip(),
                        "title": str(p.get("title", "")).strip(),
                    }
                    for p in referenced_papers
                    if isinstance(p, dict) and (p.get("paper_id") or p.get("title"))
                ]
                logger.info(
                    "Extracted {} referenced papers: {}",
                    len(state["referenced_papers"]),
                    [(p["paper_id"] or p["title"][:60]) for p in state["referenced_papers"]],
                )
            
            logger.info("History compressed for rewrite: {}", state["rewrite_history_context"])
            return state["rewrite_history_context"]
        except Exception as e:
            logger.warning("History compression failed: {}", e)
            return ""

    @staticmethod
    def _rule_based_rewrite(query: str) -> str:
        """Apply rule-based query cleaning before LLM rewrite.
        
        Handles:
        - Noise word removal
        - Basic denoising
        
        Does NOT handle pronoun resolution (that requires context/LLM).
        
        Args:
            query: Original user query
            
        Returns:
            Cleaned query string
        """
        import re
        
        cleaned = query.strip()
        
        # Remove common filler prefixes
        filler_prefixes = [
            r"^(please|pls|帮我|请|麻烦|能不能|可以|能否)\s*",
            r"^(can you|could you|would you)\s*",
            r"^(I want to|I need to|I would like to)\s*",
            r"^(find|search|look for|检索|查找|搜索)\s*(me\s*)?(some\s*)?(the\s*)?",
            # Chinese natural-language wrappers
            r"^(有没有|有哪些|什么是|怎么样|如何|是否|是不是)\s*",
            r"^(告诉我|介绍一下|解释一下|说说|讲一下|聊聊)\s*",
        ]
        for pattern in filler_prefixes:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        
        # Remove trailing filler
        filler_suffixes = [
            r"\s*(please|pls|谢谢|thanks?|thank you)[.!]*$",
            r"\s*\?+\s*$",
            # Chinese trailing filler
            r"\s*(的论文|的文献|的文章|方面的|相关的|是什么|怎么做|怎么样|吗)[?？!！.]*$",
            # English trailing filler
            r"\s*(papers?|articles?|literature|studies?|research)[.?]*$",
        ]
        for pattern in filler_suffixes:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        # If the cleaned query is still pure Chinese (no Latin chars), leave
        # it as-is — the LLM rewrite step will translate to English for search.
        # Rule-based rewrite only strips filler; it doesn't translate.
        return cleaned if cleaned else query

    @staticmethod
    def _should_trigger_llm_rewrite(
        query: str,
        previous_retrieval_quality: str,
        state: MultiAgentState,
    ) -> tuple[bool, str]:
        """Determine if LLM-based query rewrite should be triggered.

        **Default policy: ALWAYS trigger LLM rewrite** — most user queries
        contain natural-language filler that degrades vector-search recall.
        The only exceptions are trivially simple queries that are already
        clean keyword-style search terms.

        Skip rewrite ONLY when ALL of these are true:
        1. Query is already short keyword-style (≤5 words, no verbs/filler)
        2. Query has no natural-language wrapping patterns
        3. Previous retrieval was sufficient (no need to retry)

        Args:
            query: The (already rule-cleaned) query
            previous_retrieval_quality: "sufficient" / "insufficient" / ""
            state: Current workflow state

        Returns:
            (should_trigger, reason) tuple
        """
        import re

        # --- Fast-path skip conditions (only skip when query is already clean) ---

        # Skip 1: trivial single keyword (≤2 words, ≤20 chars)
        word_count = len(query.split())
        if word_count <= 2 and len(query) <= 20:
            return False, "trivial_keyword"

        # Skip 2: query already looks like a clean scientific search term
        # (e.g. "EEG signal classification", "transformer attention mechanism")
        natural_language_markers = [
            r"\b(有没有|有哪些|什么是|怎么样|如何|能否|可以|帮我|请|麻烦|能不能)\b",
            r"\b(what is|how to|how does|can you|tell me|find me|search for|look up)\b",
            r"\b(is there|are there|do you|does the)\b",
            r"\b(的论文|的文献|的文章|方面的|相关的)\b",
            r"\b(papers? about|papers? on|articles? about|literature on)\b",
        ]
        has_natural_language = any(
            re.search(pattern, query, re.IGNORECASE) for pattern in natural_language_markers
        )
        if not has_natural_language and word_count <= 5:
            return False, "already_clean_keyword"

        # --- Default: trigger LLM rewrite for everything else ---
        reasons: list[str] = ["default_llm_rewrite"]

        if has_natural_language:
            reasons.append("natural_language_detected")
        if previous_retrieval_quality == "insufficient":
            reasons.append("previous_retrieval_insufficient")
        if word_count > 5:
            reasons.append("verbose_query")

        # Pronoun/vague reference detection (added for logging)
        pronoun_patterns = [
            r"\b(它|他|她|这个|那个|这些|那些|这里|那里|这|那)\b",
            r"\b(this|that|these|those|it|they|them|the above|the previous|the same)\b",
            r"\b(前者|后者|上述|前面|上面|之前|刚刚|刚才)\b",
        ]
        for pattern in pronoun_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                reasons.append("pronoun_reference")
                break

        reason_str = "; ".join(reasons)
        return True, reason_str

    async def _rewrite_query_with_context(
        self,
        user_query: str,
        state: MultiAgentState,
    ) -> dict[str, Any]:
        """Rewrite a user query for internal KB retrieval using LLM with compressed context.
        
        Args:
            user_query: The original user query
            state: Current workflow state with history context
            
        Returns:
            {
                "rewritten_query": str,
                "confidence": float,
                "reasoning": str,
                "constraints": list[str],
            }
        """
        rewrite_history_context = state.get("rewrite_history_context", "") or "(empty)"
        previous_quality = str(state.get("retrieval_quality", "") or "none")
        confidence_threshold = float(state.get("retrieval_rewrite_confidence_threshold", 0.6) or 0.6)
        
        _, trigger_reason = self._should_trigger_llm_rewrite(user_query, previous_quality, state)
        
        # Append referenced_papers to context for pronoun/resolution awareness
        referenced_papers = state.get("referenced_papers", [])
        papers_context = ""
        if referenced_papers:
            papers_lines = []
            for i, p in enumerate(referenced_papers, 1):
                pid = p.get("paper_id", "")
                title = p.get("title", "")
                papers_lines.append(f"  [{i}] paper_id={pid or '(unknown)'}; title={title[:120] or '(unknown)'}")
            papers_context = "\nReferenced Papers:\n" + "\n".join(papers_lines)
        
        prompt = QUERY_REWRITE_USER_TEMPLATE.format(
            user_query=user_query,
            rewrite_history_context=rewrite_history_context + papers_context,
            previous_retrieval_quality=previous_quality,
            trigger_reason=f"Trigger reason: {trigger_reason}" if trigger_reason != "no_trigger_condition" else "",
        )

        logger.info("LLM query rewrite triggered: {}", prompt)
        
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            content = (response.content or "").strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            rewritten = str(result.get("rewritten_query", "")).strip()
            confidence = float(result.get("confidence", 0.5))
            
            # Validate: if confidence below threshold, fallback to original
            if not rewritten or confidence < confidence_threshold:
                logger.info(
                    "Query rewrite low confidence ({:.2f} < {:.2f}), using original query",
                    confidence,
                    confidence_threshold,
                )
                return {
                    "rewritten_query": user_query,
                    "confidence": confidence,
                    "reasoning": "confidence_below_threshold",
                    "constraints": [],
                }
            
            logger.info(
                "Query rewritten: '{}' -> '{}' (confidence={:.2f})",
                user_query[:50],
                rewritten[:50],
                confidence,
            )
            
            return {
                "rewritten_query": rewritten,
                "confidence": confidence,
                "reasoning": str(result.get("reasoning", "")),
                "constraints": result.get("constraints", []) if isinstance(result.get("constraints"), list) else [],
            }
        except Exception as e:
            logger.warning("Query rewrite failed: {}, using original query", e)
            return {
                "rewritten_query": user_query,
                "confidence": 0.0,
                "reasoning": f"rewrite_failed: {e}",
                "constraints": [],
            }

    async def _decompose_query_for_research(
        self,
        user_query: str,
        state: MultiAgentState,
    ) -> dict[str, Any]:
        """Decompose a user query into multiple sub-queries for external research.
        
        Args:
            user_query: The original user query
            state: Current workflow state
            
        Returns:
            {
                "sub_queries": list[str],
                "confidence": float,
                "reasoning": str,
                "coverage": list[str],
            }
        """
        num_queries = int(state.get("research_query_decompose_count", 3) or 3)
        rewrite_history_context = state.get("rewrite_history_context", "") or "(empty)"
        
        prompt = QUERY_DECOMPOSE_USER_TEMPLATE.format(
            user_query=user_query,
            rewrite_history_context=rewrite_history_context,
            num_queries=num_queries,
        )
        
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": QUERY_DECOMPOSE_SYSTEM_PROMPT.format(num_queries=num_queries)},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            content = (response.content or "").strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            sub_queries = result.get("sub_queries", [])
            if isinstance(sub_queries, list):
                sub_queries = [str(q).strip() for q in sub_queries if q and str(q).strip()]
            else:
                sub_queries = []
            
            if not sub_queries:
                sub_queries = [user_query]
            
            logger.info(
                "Query decomposed: '{}' -> {} sub-queries: {}",
                user_query[:50],
                len(sub_queries),
                [q[:50] for q in sub_queries],
            )
            
            return {
                "sub_queries": sub_queries[:num_queries],
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": str(result.get("reasoning", "")),
                "coverage": result.get("coverage", []) if isinstance(result.get("coverage"), list) else [],
            }
        except Exception as e:
            logger.warning("Query decomposition failed: {}, using original query", e)
            return {
                "sub_queries": [user_query],
                "confidence": 0.0,
                "reasoning": f"decomposition_failed: {e}",
                "coverage": [],
            }

    async def _extract_entities(
        self,
        user_query: str,
        cleaned_query: str,
        state: MultiAgentState,
        *,
        rewrite_history_context: str = "",
        rewritten_query: str = "",
    ) -> list[dict[str, str]]:
        """Extract specific paper references (IDs, titles) from the user query.
        
        Uses the compressed conversation context to resolve vague references
        like "第一篇论文", "that paper", "it" to actual paper IDs/titles.
        Falls back to regex extraction of [paper_id] markers from rewritten_query.
        
        Args:
            user_query: Original user query
            cleaned_query: Rule-cleaned query
            state: Workflow state (may contain referenced_papers from history compression)
            rewrite_history_context: Compressed conversation context with referenced_papers
            rewritten_query: LLM-rewritten query (may contain [paper_id] markers)
            
        Returns:
            List of entity dicts:
            [{"paper_id": "2604.24729v1", "title": "", "query": "algorithm design"}, ...]
        """
        # Check cache first
        cached = state.get("extracted_entities")
        if cached is not None:
            return list(cached) if isinstance(cached, list) else []
        
        prompt = ENTITY_EXTRACT_USER_TEMPLATE.format(
            user_query=user_query,
            cleaned_query=cleaned_query,
            rewrite_history_context=rewrite_history_context or state.get("rewrite_history_context", "(empty)"),
            rewritten_query=rewritten_query or "(none)",
        )
        
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            content = (response.content or "").strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            entities = result.get("entities", [])
            if isinstance(entities, list):
                entities = [
                    {
                        "paper_id": str(e.get("paper_id", "")).strip(),
                        "title": str(e.get("title", "")).strip(),
                        "query": str(e.get("query", "")).strip(),
                    }
                    for e in entities
                    if isinstance(e, dict) and (e.get("paper_id") or e.get("title"))
                ]
            else:
                entities = []
            
            # Cache in state
            state["extracted_entities"] = entities
            if entities:
                logger.info(
                    "_extract_entities: found {} entities: {}",
                    len(entities),
                    [(e.get("paper_id") or e.get("title", "")[:60]) for e in entities],
                )
                return entities
            
            # LLM returned no entities — try regex fallback from rewritten_query
            logger.info("_extract_entities: LLM returned no entities, trying regex fallback")
        except Exception as e:
            logger.warning("Entity extraction failed: {}, trying regex fallback", e)
            entities = []
        
        # ---- Regex fallback: extract [paper_id] markers from rewritten_query ----
        if not entities:
            query_with_markers = rewritten_query or ""
            if not query_with_markers:
                # Also check cached rewritten_queries in state
                cached_queries = state.get("rewritten_queries", [])
                if cached_queries and isinstance(cached_queries, list):
                    query_with_markers = " ".join(str(q) for q in cached_queries)
            
            if query_with_markers:
                paper_ids = re.findall(r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]', query_with_markers)
                if paper_ids:
                    # Extract query text by removing [paper_id] markers
                    clean_query = re.sub(r'\[\d{4}\.\d{4,5}(?:v\d+)?\]', '', query_with_markers).strip()
                    for pid in paper_ids:
                        entities.append({
                            "paper_id": pid,
                            "title": "",
                            "query": clean_query[:200] or "overview",
                        })
                    logger.info(
                        "_extract_entities: regex fallback found {} paper_ids: {}",
                        len(entities),
                        paper_ids,
                    )
        
        state["extracted_entities"] = entities
        return entities

    @staticmethod
    def _check_rewrite_degradation(
        original_results: list[dict[str, Any]],
        rewritten_results: list[dict[str, Any]],
        delta_threshold: float = 0.05,
    ) -> bool:
        """Check if rewritten query results are significantly worse than original.
        
        Compares:
        - Hit count: if rewritten returns fewer results
        - Top-1 score: if rewritten top score is lower by > delta_threshold
        
        Args:
            original_results: Results from original query
            rewritten_results: Results from rewritten query
            delta_threshold: Maximum acceptable score degradation
            
        Returns:
            True if rewritten results are degraded (fallback needed), False otherwise
        """
        # If rewritten has no results but original does, it's degraded
        if not rewritten_results and original_results:
            logger.info("Rewrite degradation: rewritten returned 0 results, original had {}", len(original_results))
            return True
        
        if not original_results and not rewritten_results:
            return False
        
        if not original_results:
            return False
        
        # Compare top-1 scores
        orig_top1 = max((r.get("score", 0) for r in original_results), default=0)
        rewrite_top1 = max((r.get("score", 0) for r in rewritten_results), default=0)
        
        degradation = orig_top1 - rewrite_top1
        
        if degradation > delta_threshold:
            logger.info(
                "Rewrite degradation: top1_score dropped from {:.4f} to {:.4f} (delta={:.4f} > {:.4f})",
                orig_top1,
                rewrite_top1,
                degradation,
                delta_threshold,
            )
            return True
        
        logger.debug(
            "Rewrite OK: top1_score {:.4f} -> {:.4f} (delta={:.4f})",
            orig_top1,
            rewrite_top1,
            degradation,
        )
        return False

    # ================================================================
    # Unified Query Rewrite (single LLM call)
    # ================================================================

    async def _unified_query_rewrite(
        self,
        user_query: str,
        state: MultiAgentState,
    ) -> dict[str, Any]:
        """Unified query rewrite: merge history-compress + rewrite + decompose + entity-extract
        into a single LLM call for reduced latency.
        
        Reads context directly from state fields (no compression call needed).
        Formats the UNIFIED_QUERY_USER_TEMPLATE with all context sections.
        Makes one LLM call with UNIFIED_QUERY_SYSTEM_PROMPT.
        Parses the structured JSON output and writes results into state.
        
        Args:
            user_query: Original user query
            state: Current workflow state
            
        Returns:
            {
                "rewritten_queries": list[str],
                "sub_queries_detail": list[dict],
                "extracted_entities": list[dict],
                "rewrite_reasoning": str,
                "rewrite_confidence": float,
                "requires_clarification": bool,
                "referenced_papers": list[dict],
            }
        """
        rewrite_context_chars = int(state.get("rewrite_context_chars", 60000) or 60000)
        num_sub_queries = int(state.get("research_query_decompose_count", 3) or 3)
        
        # Build referenced_papers section for the prompt
        referenced_papers = state.get("referenced_papers", [])
        referenced_papers_section = ""
        if referenced_papers:
            papers_lines = []
            for i, p in enumerate(referenced_papers, 1):
                pid = p.get("paper_id", "")
                title = p.get("title", "")
                papers_lines.append(f"  [{i}] paper_id={pid or '(unknown)'}; title={title[:120] or '(unknown)'}")
            referenced_papers_section = "<referenced_papers>\n" + "\n".join(papers_lines) + "\n</referenced_papers>"
        
        # Format context fields (truncate to rewrite_context_chars)
        user_profile = str(state.get("user_profile_context", "") or "(empty)")[:rewrite_context_chars]
        long_term_memory = str(state.get("long_term_memory_context", "") or "(empty)")[:rewrite_context_chars]
        session_summary = str(state.get("session_summary_context", "") or "(empty)")[:rewrite_context_chars]
        recent_dialog = str(state.get("recent_dialog_context", "") or "(empty)")[:rewrite_context_chars]
        research_history = str(state.get("research_history_context", "") or "(empty)")[:rewrite_context_chars]
        retrieval_history = str(state.get("retrieval_history_context", "") or "(empty)")[:rewrite_context_chars]
        last_routing = str(state.get("last_routing_decision", "") or "none")
        last_quality = str(state.get("retrieval_quality", "") or "unknown")
        
        # Build the system prompt with num_sub_queries
        system_prompt = UNIFIED_QUERY_SYSTEM_PROMPT.format(num_sub_queries=num_sub_queries)
        
        # Build the user prompt
        user_prompt = UNIFIED_QUERY_USER_TEMPLATE.format(
            user_profile_context=user_profile,
            long_term_memory_context=long_term_memory,
            session_summary_context=session_summary,
            recent_dialog_context=recent_dialog,
            research_history_context=research_history,
            retrieval_history_context=retrieval_history,
            last_routing_decision=last_routing,
            last_retrieval_quality=last_quality,
            user_query=user_query,
            referenced_papers_section=referenced_papers_section,
        )
        
        logger.info("_unified_query_rewrite: calling LLM with unified prompt for query '{}'", user_query[:60])
        
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            content = (response.content or "").strip()
            # Strip markdown fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json_repair.loads(content)

            logger.info("_unified_query_rewrite: result from LLM: {}", result)
            
            # ---- Parse structured output ----
            rewrite_reasoning = str(result.get("rewrite_reasoning", ""))
            confidence_score = float(result.get("confidence_score", 0.5))
            requires_clarification = bool(result.get("requires_clarification", False))
            sub_queries_raw = result.get("sub_queries", [])
            
            # ---- Build rewritten_queries flat list ----
            rewritten_queries: list[str] = []
            sub_queries_detail: list[dict[str, Any]] = []
            extracted_entities: dict[str, dict[str, str]] = {}
            all_referenced_papers: list[dict[str, str]] = []
            
            if isinstance(sub_queries_raw, list):
                for sq in sub_queries_raw:
                    if not isinstance(sq, dict):
                        continue
                    
                    # rewritten_queries from this sub-query
                    rq_list = sq.get("rewritten_queries", [])
                    if isinstance(rq_list, str):
                        rq_list = [rq_list]
                    for rq in rq_list:
                        rq_str = str(rq).strip()
                        if rq_str and rq_str not in rewritten_queries:
                            rewritten_queries.append(rq_str)
                    
                    # target_paper → entity + referenced_paper
                    target_paper = sq.get("target_paper", {})
                    if isinstance(target_paper, dict) and (target_paper.get("paper_id") or target_paper.get("title")):
                        pid = str(target_paper.get("paper_id", "")).strip()
                        title = str(target_paper.get("title", "")).strip()
                        # Remove 'arxiv:' prefix if present
                        if pid.startswith("arxiv:"):
                            pid = pid[6:]
                        if pid not in extracted_entities:
                            all_referenced_papers.append({
                                "paper_id": pid,
                                "title": title,
                            })
                            extracted_entities[pid] = {
                                "paper_id": pid,
                                "title": title,
                                "queries": rq_list if rq_list else [],
                            }
                        else:
                            pid_queries = extracted_entities[pid].get("queries", [])
                            for rq in rq_list:
                                rq_str = str(rq).strip()
                                if rq_str and rq_str not in pid_queries:
                                    pid_queries.append(rq_str)
                            extracted_entities[pid]["queries"] = pid_queries
                    
                    # keywords
                    keywords = sq.get("keywords", [])
                    if not isinstance(keywords, list):
                        keywords = []
                    
                    # time_filter
                    time_filter = sq.get("time_filter")
                    if time_filter:
                        time_filter = str(time_filter).strip()
                    else:
                        time_filter = None
                    
                    # Build sub_queries_detail entry
                    sub_queries_detail.append({
                        "rewritten_queries": [str(q).strip() for q in rq_list if str(q).strip()],
                        "target_paper": target_paper if isinstance(target_paper, dict) else {},
                        "keywords": [str(k).strip() for k in keywords if str(k).strip()],
                        "time_filter": time_filter,
                    })

            extracted_entities = list(extracted_entities.values()) if extracted_entities else []
            
            # Ensure at least one query exists (fallback to cleaned original)
            if not rewritten_queries:
                rewritten_queries = [self._rule_based_rewrite(user_query)]
                sub_queries_detail = [{
                    "rewritten_queries": rewritten_queries,
                    "target_paper": {},
                    "keywords": [],
                    "time_filter": None,
                }]
            
            # Ensure the cleaned original is always in the list (as first element)
            cleaned_query = self._rule_based_rewrite(user_query)
            # if cleaned_query.lower() not in [q.lower() for q in rewritten_queries]:
            #     rewritten_queries.insert(0, cleaned_query)
            
            # Merge referenced_papers from state with newly resolved ones
            existing_refs = state.get("referenced_papers", [])
            for ref in all_referenced_papers:
                if not any(r.get("paper_id") == ref["paper_id"] or r.get("title") == ref["title"] for r in existing_refs):
                    existing_refs.append(ref)
            
            # ---- Write results into state ----
            state["rewritten_queries"] = rewritten_queries
            state["sub_queries_detail"] = sub_queries_detail
            state["extracted_entities"] = extracted_entities
            state["rewrite_reasoning"] = rewrite_reasoning
            state["rewrite_confidence"] = confidence_score
            state["requires_clarification"] = requires_clarification
            state["referenced_papers"] = existing_refs
            state["rewrite_fallback_used"] = False
            
            logger.info(
                "_unified_query_rewrite: {} queries, {} sub_queries_detail, {} entities, confidence={:.2f}, clarification={}",
                len(rewritten_queries),
                len(sub_queries_detail),
                len(extracted_entities),
                confidence_score,
                requires_clarification,
            )
            
            return {
                "rewritten_queries": rewritten_queries,
                "sub_queries_detail": sub_queries_detail,
                "extracted_entities": extracted_entities,
                "rewrite_reasoning": rewrite_reasoning,
                "rewrite_confidence": confidence_score,
                "requires_clarification": requires_clarification,
                "referenced_papers": existing_refs,
            }
        except Exception as e:
            logger.warning("_unified_query_rewrite failed: {}, falling back to rule-based", e)
            # Fallback: rule-based rewrite only
            cleaned_query = self._rule_based_rewrite(user_query)
            state["rewritten_queries"] = [cleaned_query]
            state["sub_queries_detail"] = [{
                "rewritten_queries": [cleaned_query],
                "target_paper": {},
                "keywords": [],
                "time_filter": None,
            }]
            state["extracted_entities"] = []
            state["rewrite_reasoning"] = f"unified_rewrite_failed: {e}"
            state["rewrite_confidence"] = 0.0
            state["requires_clarification"] = False
            state["rewrite_fallback_used"] = True
            
            return {
                "rewritten_queries": [cleaned_query],
                "sub_queries_detail": state["sub_queries_detail"],
                "extracted_entities": [],
                "rewrite_reasoning": f"unified_rewrite_failed: {e}",
                "rewrite_confidence": 0.0,
                "requires_clarification": False,
                "referenced_papers": state.get("referenced_papers", []),
            }

    # ================================================================
    # Shared Query Preparation (rewrite + decomposition)
    # ================================================================

    async def _prepare_queries(
        self,
        user_query: str,
        state: MultiAgentState,
    ) -> list[str]:
        """Prepare optimized search queries via unified LLM call (rewrite + decompose + entity extract).
        
        This is the SHARED entry point for both retrieval_node and research_node.
        Results are cached in state["rewritten_queries"] so subsequent nodes
        reuse the same set without repeating LLM calls.
        
        Flow (unified):
        1. Return cached queries if already present
        2. If query_rewrite_enabled=False, return rule-cleaned query only
        3. If query_rewrite_use_llm=True, call _unified_query_rewrite (single LLM call)
        4. Otherwise, fall back to rule-based cleaning only
        
        Args:
            user_query: Original user query
            state: Current workflow state
            
        Returns:
            Deduplicated list of optimized query strings (always includes
            the cleaned original as first element)
        """
        # Return cached queries if already prepared
        cached = state.get("rewritten_queries")
        if cached and isinstance(cached, list) and len(cached) > 0:
            logger.info(
                "_prepare_queries: reusing cached {} queries",
                len(cached),
            )
            return [str(q) for q in cached]
        
        query_rewrite_enabled = state.get("query_rewrite_enabled", True)
        if not query_rewrite_enabled:
            # No rewrite, just return cleaned original
            cleaned = self._rule_based_rewrite(user_query)
            state["rewritten_queries"] = [cleaned]
            return [cleaned]
        
        query_rewrite_use_llm = state.get("query_rewrite_use_llm", True)
        if query_rewrite_use_llm:
            # Unified single LLM call: rewrite + decompose + entity extract
            result = await self._unified_query_rewrite(user_query, state)
            all_queries = result.get("rewritten_queries", [])
            logger.info(
                "_prepare_queries: unified rewrite produced {} queries, confidence={:.2f}",
                len(all_queries),
                result.get("rewrite_confidence", 0.0),
            )
            return all_queries
        else:
            # LLM disabled — only rule-based cleaning
            cleaned = self._rule_based_rewrite(user_query)
            state["rewritten_queries"] = [cleaned]
            state["sub_queries_detail"] = [{
                "rewritten_queries": [cleaned],
                "target_paper": {},
                "keywords": [],
                "time_filter": None,
            }]
            return [cleaned]

    # ================================================================
    # Research candidate selection
    # ================================================================

    async def _select_research_candidates_with_llm(
        self,
        *,
        user_query: str,
        papers: list[dict[str, Any]],
        research_history_context: str,
        ingest_limit: int,
    ) -> list[dict[str, Any]]:
        if not papers:
            return []
        paper_briefs = []
        for i, p in enumerate(papers[:20], 1):
            paper_briefs.append(
                f"{i}. id={p.get('paper_id','')}; title={str(p.get('title',''))[:140]}; "
                f"year={p.get('year','')}; score={p.get('rerank_score', p.get('similarity_score', ''))}"
            )

        prompt = (
            "Select the most useful papers for ingestion and return strict JSON.\n"
            "JSON format: {\"selected_indices\": [1,2,...], \"reasoning\": \"...\"}.\n"
            f"Maximum selections: {ingest_limit}.\n"
            f"User query: {user_query}\n"
            f"Research context: {research_history_context or '(empty)'}\n"
            "Candidates:\n"
            + "\n".join(paper_briefs)
        )
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": "You are a precise selector for paper ingestion."},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            content = (response.content or "").strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content)
            selected = data.get("selected_indices", [])
            selected_idx = {
                int(x) - 1
                for x in selected
                if isinstance(x, (int, str)) and str(x).isdigit()
            }
            output = [papers[i] for i in sorted(selected_idx) if 0 <= i < len(papers)]
            return output[:ingest_limit] if output else papers[:ingest_limit]
        except Exception as e:
            logger.warning("Research Agent LLM selector failed: {}", e)
            return papers[:ingest_limit]
    
    async def router_node(self, state: MultiAgentState) -> MultiAgentState:
        """Router Agent: Analyze user intent and decide execution path.
        
        In resume mode, skip LLM call and direct to research for ingest continuation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with routing_decision and routing_reasoning
        """
        # Resume mode: skip routing, go directly to research (ingest phase)
        if state.get("resume_mode"):
            logger.info(
                "Router Agent: Resume mode — skipping routing, directing to research (phase={})",
                state.get("research_phase", "unknown"),
            )
            state["routing_decision"] = "external"
            state["routing_reasoning"] = "Resume mode: continuing with user-selected paper ingest"
            await self._emit_progress("🔄 恢复工作流：继续处理论文摄取...")
            return state

        logger.info("Router Agent: Analyzing user query")
        
        await self._emit_progress("🧭 分析用户意图，判断执行路径...")
        
        user_query = state.get("user_query", "")
        session_memory_short = state.get("router_memory_short", "") or "(empty)"
        session_memory_long = state.get("router_memory_long", "") or "(empty)"
        recent_dialog = state.get("recent_dialog_context", "") or "(empty)"
        long_term_memory = state.get("long_term_memory_context", "") or "(empty)"
        user_profile = state.get("user_profile_context", "") or "(empty)"
        last_routing_decision = state.get("last_routing_decision", "") or "none"
        
        # Build prompt
        prompt = ROUTER_USER_TEMPLATE.format(
            user_query=user_query,
            recent_dialog_context=recent_dialog,
            session_memory_short=session_memory_short,
            session_memory_long=session_memory_long,
            long_term_memory_context=long_term_memory,
            user_profile_context=user_profile,
            last_routing_decision=last_routing_decision,
        )
        
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": self.prompts.router},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            
            content = response.content or ""
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            state["routing_decision"] = result.get("decision", "hybrid")
            state["routing_reasoning"] = result.get("reasoning", "")
            
            logger.info(
                "Router Agent: decision={}, reasoning={}",
                state["routing_decision"],
                state["routing_reasoning"],
            )
            
            await self._emit_progress(
                f"📋 路由决策: {state['routing_decision']} — {state.get('routing_reasoning', '')[:100]}"
            )
            
        except Exception as e:
            logger.error("Router Agent failed: {}", e)
            # Fallback to internal — prefer KB by default
            state["routing_decision"] = "internal"
            state["routing_reasoning"] = f"Router failed ({e}), defaulting to internal"
        
        return state
    
    async def retrieval_node(self, state: MultiAgentState) -> MultiAgentState:
        """Retrieval Agent: Query internal knowledge base with query rewrite support.
        
        Flow:
        1. Apply rule-based rewrite (denoising)
        2. Compress history context if needed
        3. Check trigger conditions for LLM rewrite
        4. If triggered, use LLM to rewrite query
        5. Retrieve with rewritten query
        6. Check degradation vs original; fallback if worse
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with retrieval_results and retrieval_quality
        """
        logger.info("Retrieval Agent: Querying knowledge base")
        
        user_query = state.get("user_query", "")
        retrieval_history_context = state.get("retrieval_history_context", "") or "(empty)"
        recent_dialog = state.get("recent_dialog_context", "") or "(empty)"
        long_term_memory = state.get("long_term_memory_context", "") or "(empty)"
        user_profile = state.get("user_profile_context", "") or "(empty)"
        fallback_delta = float(state.get("rewrite_fallback_delta_threshold", 0.05) or 0.05)
        
        # Handle post-research loop: clear the flag so we don't loop forever
        post_research = state.get("post_research_retrieval", False)
        if post_research:
            state["post_research_retrieval"] = False
            logger.info("Retrieval Agent: Post-research retrieval — retrieving newly ingested papers")
            await self._emit_progress("📚 检索刚入库的论文内容...")
        else:
            await self._emit_progress("📚 正在检索内部知识库...")
        
        try:
            # Prepare queries (shared rewrite + decomposition, cached on first call)
            if not state.get("rewritten_queries", ""):
                await self._emit_progress("✏️ 优化查询关键词...")
                all_queries = await self._prepare_queries(user_query, state)
            else:
                # Post-research: reuse cached queries without re-invoking LLM
                all_queries = state.get("rewritten_queries", [user_query])

            logger.info("Retrieval Agent: all queries = {}", all_queries)
            
            # ---- Handle requires_clarification: skip retrieval, go straight to synthesis ----
            if state.get("requires_clarification", False):
                logger.warning(
                    "Retrieval Agent: Query references too ambiguous (requires_clarification=True), "
                    "skipping retrieval and routing to synthesis for user clarification"
                )
                state["retrieval_results"] = []
                state["retrieval_quality"] = "insufficient"
                # Store clarification request for synthesis to address
                state["draft_answer"] = (
                    f"I'm not sure which paper or reference you're referring to. "
                    f"Could you please clarify? For example, specify the paper title, arXiv ID, "
                    f"or describe it more specifically.\n\n"
                    f"Your question was: {user_query}"
                )
                await self._emit_progress("❓ 无法确定您引用的论文，需要您的澄清")
                return state
            # ---- End requires_clarification handling ----
            
            # Retrieve: entity-aware if specific papers targeted, else multi-query
            logger.info("Retrieval Agent: entities {}", state.get("extracted_entities", []))
            entities = state.get("extracted_entities", [])
            if entities:
                logger.info(
                    "Retrieval Agent: entity-aware retrieval for {} entities",
                    len(entities),
                )
                await self._emit_progress(
                    f"🎯 精准检索: {', '.join(e.get('paper_id') or e.get('title','')[:30] for e in entities[:3])}"
                )
                results = await self.kb.retrieve_by_hypothetical_questions(
                    entities=entities,
                    queries=all_queries,
                    top_k=self.top_k,
                    per_paper_limit=4,  # More per-paper for targeted retrieval
                    search_mode="hybrid",
                    use_hybrid=self.kb.config.use_hybrid_retrieval,
                )
            else:
                results = await self.kb.retrieve_by_hypothetical_questions(
                    queries=all_queries,
                    top_k=self.top_k,
                    per_paper_limit=3,
                    search_mode="hybrid",
                    use_hybrid=self.kb.config.use_hybrid_retrieval,
                )
            
            state["retrieval_results"] = results
            state["rewrite_fallback_used"] = False
            
            # Check degradation: compare multi-query results vs original single-query
            if not post_research and len(all_queries) > 1 and results:
                original_results = await self.kb.retrieve_by_hypothetical_questions(
                    query=user_query,
                    top_k=self.top_k,
                    per_paper_limit=3,
                    search_mode="hybrid",
                )
                
                degraded = self._check_rewrite_degradation(
                    original_results, results, delta_threshold=fallback_delta
                )
                
                if degraded:
                    logger.warning(
                        "Retrieval Agent: Multi-query rewrite degraded, falling back to original query results"
                    )
                    state["retrieval_results"] = original_results
                    state["rewrite_fallback_used"] = True
                    results = original_results
                else:
                    logger.info("Retrieval Agent: Multi-query rewrite performed better or equal to original")
            
            # Assess quality
            if not results:
                state["retrieval_quality"] = "insufficient"
                state["retrieval_results"] = []
                logger.warning("Retrieval Agent: No results found")
            else:
                # ---- Relevance filter: drop chunks below 30% of best score ----
                best_score = max(r.get("score", 0) for r in results)
                min_score = best_score * 0.3
                filtered = [r for r in results if r.get("score", 0) >= min_score]
                if len(filtered) < len(results):
                    logger.info(
                        "Retrieval Agent: Filtered {} low-relevance chunks (best={:.3f}, cutoff={:.3f})",
                        len(results) - len(filtered), best_score, min_score,
                    )
                results = filtered
                state["retrieval_results"] = results
                
                if not results:
                    state["retrieval_quality"] = "insufficient"
                    logger.warning("Retrieval Agent: All results filtered out (below relevance threshold)")
                    return state
                
                best_score = max(r.get("score", 0) for r in results)
                # ---- End relevance filter ----
                try:
                    margin = float(state.get("retrieval_judge_margin", 0.02) or 0.02)
                except (TypeError, ValueError):
                    margin = 0.02
                low = self.similarity_threshold - margin
                high = self.similarity_threshold + margin

                if best_score <= low:
                    state["retrieval_quality"] = "insufficient"
                    logger.warning(
                        "Retrieval Agent: Deterministic insufficient (best_score={:.3f}, low={:.3f})",
                        best_score,
                        low,
                    )
                    await self._emit_progress(
                        f"📚 知识库检索完成: {len(results)} 条结果，匹配度不足 (best={best_score:.3f})，将转至外部搜索"
                    )
                elif best_score >= high:
                    state["retrieval_quality"] = "sufficient"
                    logger.info(
                        "Retrieval Agent: Deterministic sufficient (best_score={:.3f}, high={:.3f})",
                        best_score,
                        high,
                    )
                    await self._emit_progress(
                        f"📚 知识库检索完成: {len(results)} 条结果 (最高匹配度={best_score:.3f})"
                    )
                else:
                    try:
                        # Build prompt
                        prompt = RETRIEVAL_USER_TEMPLATE.format(
                            user_query=user_query,
                            recent_dialog_context=recent_dialog,
                            long_term_memory_context=long_term_memory,
                            user_profile_context=user_profile,
                            retrieval_history_context=retrieval_history_context,
                            results=self._results_preview(results),
                        )
                        response = await self.provider.chat_with_retry(
                            model=self.provider.get_default_model(),
                            messages=[
                                {"role": "system", "content": self.prompts.retrieval},
                                {"role": "user", "content": prompt},
                            ],
                            tools=None,
                            tool_choice=None,
                        )
                        
                        content = response.content or ""
                        
                        # Extract JSON from response
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0]
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0]
                        
                        result = json.loads(content.strip())
                        
                        state["retrieval_quality"] = result.get("quality", "sufficient")

                        if state["retrieval_quality"] == "sufficient":
                            logger.info(
                                "Retrieval Agent: Found {} results (best_score={:.3f})",
                                len(results),
                                best_score,
                            )
                            await self._emit_progress(
                                f"📚 知识库检索完成: {len(results)} 条结果 (最高匹配度={best_score:.3f})"
                            )
                        
                    except Exception as e:
                        logger.error("Retrieval Agent LLM assessment failed: {}", e)
                        # Default to sufficient
                        state["retrieval_quality"] = "sufficient"
                        logger.info(
                            "Retrieval Agent: Found {} results (best_score={:.3f})",
                            len(results),
                            best_score,
                        )
                        await self._emit_progress(
                            f"📚 知识库检索完成: {len(results)} 条结果 (best={best_score:.3f})"
                        )
            
        except Exception as e:
            logger.error("Retrieval Agent failed: {}", e)
            state["retrieval_results"] = []
            state["retrieval_quality"] = "insufficient"
            state["error_message"] = str(e)
        
        return state
    
    async def research_node(self, state: MultiAgentState) -> MultiAgentState:
        """Research Agent: Search and optionally ingest external papers.
        
        Decoupled phases:
        1. search: Execute search, store results, wait for user selection
        2. ingest: Ingest user-selected papers
        3. complete: Finish (user skipped or nothing to ingest)
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with external_papers and ingested_papers
        """
        phase = state.get("research_phase", "search")
        
        if phase == "search":
            return await self._research_search_phase(state)
        elif phase == "ingest":
            return await self._research_ingest_phase(state)
        else:  # complete or unknown
            return await self._research_complete_phase(state)

    async def _research_search_phase(self, state: MultiAgentState) -> MultiAgentState:
        """Phase 1: Execute search and prepare results for user selection."""
        logger.info("Research Agent [search phase]: Searching external sources")
        await self._emit_progress("🔍 正在搜索 arXiv 外部论文...")
        
        user_query = state.get("user_query", "")
        already_ingested = set(state.get("ingested_papers", []))
        
        try:
            search_tool = self.tools.get("paper_search")
            if not search_tool:
                logger.error("Research Agent: Search tool not available")
                state["error_message"] = "Paper search tool not available"
                state["research_phase"] = "complete"
                return state
            
            # Prepare queries (shared rewrite + decomposition, cached on first call)
            await self._emit_progress("✏️ 优化搜索关键词...")
            if not state.get("rewritten_queries"):
                all_queries = await self._prepare_queries(user_query, state)
            else:
                all_queries = state.get("rewritten_queries", [user_query])
            
            # ---- Extract keywords and time_filter from sub_queries_detail ----
            sub_queries_detail = state.get("sub_queries_detail", [])
            all_keywords: list[list[str]] = []
            time_filters: list[str] = []
            if sub_queries_detail and isinstance(sub_queries_detail, list):
                for sq in sub_queries_detail:
                    if isinstance(sq, dict):
                        kw = sq.get("keywords", [])
                        if isinstance(kw, list):
                            all_keywords.append([str(k).strip() for k in kw if str(k).strip()])
                        tf = sq.get("time_filter")
                        if tf and str(tf).strip():
                            time_filters.append(str(tf).strip())
            time_filter = time_filters[0] if time_filters else None
            
            if all_keywords:
                logger.info(
                    "Research Agent: Using {} keywords from sub_queries_detail: {}",
                    len(all_keywords),
                    all_keywords[:10],
                )
            # ---- End keywords/time_filter extraction ----
            
            search_queries = all_queries if all_queries else [user_query]
            
            # Execute search
            logger.info(
                "Research Agent: Searching with {} prepared queries: {}",
                len(search_queries),
                [q[:50] for q in search_queries],
            )
            sr = await search_tool.execute(
                query=user_query,
                candidate_queries=search_queries,
                keywords=all_keywords if all_keywords else None,
                source="arxiv",
                search_topk=60,
                recall_top_k=20,
                rerank_top_k=max(10, self.ingest_limit * 3),
            )
            search_data = json.loads(sr)
            all_papers = search_data.get("results", [])
            
            # Fallback to original query if multi-query returns nothing
            if not all_papers and len(search_queries) > 1:
                logger.warning("Research Agent: Multi-query search returned 0 results, falling back")
                fallback_sr = await search_tool.execute(
                    query=user_query,
                    source="arxiv",
                    search_topk=60,
                    recall_top_k=20,
                    rerank_top_k=self.ingest_limit * 2,
                )
                fallback_data = json.loads(fallback_sr)
                all_papers = fallback_data.get("results", [])
                state["rewrite_fallback_used"] = True
            
            # Filter out already ingested papers
            if already_ingested:
                all_papers = [p for p in all_papers if str(p.get("paper_id", "")) not in already_ingested]
            
            # Apply time filter
            if time_filter:
                try:
                    filter_year = int(time_filter)
                    before = len(all_papers)
                    all_papers = [p for p in all_papers if p.get("year") and int(p.get("year", 0)) >= filter_year]
                    logger.info(
                        "Research Agent: time_filter={} filtered {} -> {} papers",
                        filter_year, before, len(all_papers),
                    )
                except (ValueError, TypeError):
                    logger.warning("Research Agent: Invalid time_filter '{}', skipping", time_filter)
            
            if not all_papers:
                logger.warning("Research Agent: No papers found")
                state["papers_for_selection"] = []
                state["external_papers"] = []
                state["search_completed"] = True
                state["research_phase"] = "complete"
                await self._emit_progress("❌ 未找到相关论文")
                return state
            
            # Store results and wait for user selection
            state["papers_for_selection"] = all_papers[:20]  # Limit to 20 for selection
            state["external_papers"] = all_papers  # Keep full list for compatibility
            state["search_completed"] = True
            state["research_phase"] = "select"  # Waiting for user selection
            
            # Format selection info for user
            selection_info = self._format_papers_for_selection(all_papers)
            state["draft_answer"] = selection_info
            
            await self._emit_progress(f"📋 找到 {len(all_papers)} 篇论文，等待用户选择...")
            
            logger.info(
                "Research Agent [search phase]: Found {} papers, waiting for user selection",
                len(all_papers),
            )
            
        except Exception as e:
            logger.error("Research search phase failed: {}", e)
            state["error_message"] = str(e)
            state["research_phase"] = "complete"
        
        return state

    async def _research_ingest_phase(self, state: MultiAgentState) -> MultiAgentState:
        """Phase 2: Ingest user-selected papers."""
        logger.info("Research Agent [ingest phase]: Ingesting selected papers")
        
        selected_ids = state.get("user_selected_papers", [])
        papers_for_selection = state.get("papers_for_selection", [])
        
        # Check if user skipped ingest
        if state.get("user_skip_ingest") or not selected_ids:
            logger.info("Research Agent: User skipped ingest or no papers selected")
            await self._emit_progress("⏭️ 用户跳过论文摄取")
            state["research_phase"] = "complete"
            return await self._research_complete_phase(state)
        
        # Filter selected papers
        papers_to_ingest = [
            p for p in papers_for_selection
            if str(p.get("paper_id", "")) in selected_ids
        ]
        
        if not papers_to_ingest:
            logger.warning("Research Agent: No matching papers found for selected IDs")
            state["research_phase"] = "complete"
            return await self._research_complete_phase(state)
        
        # Deduplicate against KB
        orig_count = len(papers_to_ingest)
        try:
            existing_docs = self.kb._read_jsonl(self.kb.docs_file)
            existing_pids = {d.get("paper_id", "") for d in existing_docs}
            papers_to_ingest = [
                p for p in papers_to_ingest
                if str(p.get("paper_id", "")) not in existing_pids
            ]
            if len(papers_to_ingest) < orig_count:
                logger.info(
                    "Research Agent: Skipped {} already-in-KB papers",
                    orig_count - len(papers_to_ingest),
                )
        except Exception as e:
            logger.warning("Research Agent: KB dedup check failed: {}", e)
        
        if not papers_to_ingest:
            logger.info("Research Agent: All selected papers already in KB")
            await self._emit_progress("✓ 所选论文已在知识库中")
            state["research_phase"] = "complete"
            return await self._research_complete_phase(state)
        
        # Execute ingest
        try:
            ingest_tool = self.tools.get("paper_ingest")
            if not ingest_tool:
                logger.error("Research Agent: Ingest tool not available")
                state["error_message"] = "Paper ingest tool not available"
                state["research_phase"] = "complete"
                return await self._research_complete_phase(state)
            
            await self._emit_progress(f"📥 正在下载解析 {len(papers_to_ingest)} 篇论文...")
            
            ingest_result = await ingest_tool.execute(
                papers=papers_to_ingest,
                parse_mode="auto",
                concurrency=2,
                summarize=True,
            )
            
            ingest_data = json.loads(ingest_result)
            ingested = [
                r.get("paper_id", "")
                for r in ingest_data.get("results", [])
                if r.get("status") == "ok"
            ]
            
            state["ingested_papers"] = ingested
            state["post_research_retrieval"] = True
            
            await self._emit_progress(f"✅ 成功入库 {len(ingested)} 篇论文")
            
            logger.info(
                "Research Agent [ingest phase]: Ingested {} papers",
                len(ingested),
            )
            
        except Exception as e:
            logger.error("Research ingest phase failed: {}", e)
            state["error_message"] = str(e)
        
        state["research_phase"] = "complete"
        return await self._research_complete_phase(state)

    async def _research_complete_phase(self, state: MultiAgentState) -> MultiAgentState:
        """Phase 3: Finalize research and set loop guard."""
        state["research_phase"] = "complete"
        current_guard = int(state.get("loop_guard_count", 0) or 0)
        state["loop_guard_count"] = current_guard + 1
        return state

    def _format_papers_for_selection(self, papers: list[dict[str, Any]]) -> str:
        """Format papers for user selection display.
        
        Args:
            papers: List of paper dicts from search results
            
        Returns:
            Formatted markdown string for user selection
        """
        lines = [
            "## 📚 搜索到的论文\n",
            "请选择要摄取到知识库的论文（输入 paper_id 列表，或回复指令）：\n",
        ]
        
        for i, p in enumerate(papers, 1):
            paper_id = p.get("paper_id", "unknown")
            title = p.get("title", "Unknown Title")
            year = p.get("year", "N/A")
            authors = p.get("authors", [])
            authors_str = ", ".join(authors[:3]) if isinstance(authors, list) else str(authors)
            if len(authors) > 3:
                authors_str += ", etc."
            abstract = p.get("abstract", "")
            abstract = abstract.split()
            if len(abstract) > 100:
                abstract = " ".join(abstract[:100]) + "..."
            else:
                abstract = " ".join(abstract)
            
            lines.append(f"\n### [{i}] {title}")
            lines.append(f"- **ID**: `{paper_id}`")
            lines.append(f"- **年份**: {year}")
            lines.append(f"- **作者**: {authors_str}")
            lines.append(f"- **摘要**: {abstract}\n")
        
        lines.extend([
            "\n---",
            "\n**操作说明**：",
            f"- 回复论文 ID 列表（如：`{', '.join([p.get('paper_id') for p in papers[:2] if p.get('paper_id')])}`）或者标号（如：`1, 3`）选择要摄取的论文",
            "- 回复 `skip` 跳过摄取",
            "- 回复 `all` 摄取全部",
        ])
        
        return "\n".join(lines)

    async def _parse_selection_response(
        self,
        user_input: str,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Parse user's selection response into structured selection action.

        Supports:
        - Skip commands: "skip", "跳过", "不用", etc.
        - Select all: "all", "全部", "所有"
        - Direct arXiv IDs: "2401.12345v1, 2402.67890v1"
        - Numeric indices: "1, 3, 5" (1-based, referring to displayed paper order)
        - Free text: Falls back to LLM-based fuzzy matching

        Args:
            user_input: Raw user response string
            papers: List of paper dicts (from papers_for_selection)

        Returns:
            {"user_skip_ingest": bool, "user_selected_papers": list[str]}
        """
        cleaned = user_input.strip()
        lower_input = cleaned.lower()

        # --- 1. Skip commands ---
        skip_keywords = {"skip", "跳过", "不用", "不需要", "不", "no", "none", "pass", "略过"}
        if lower_input in skip_keywords:
            logger.info("_parse_selection_response: user chose to skip ingest")
            return {"user_skip_ingest": True, "user_selected_papers": []}

        # --- 2. Select all ---
        all_keywords = {"all", "全部", "所有", "全部论文", "全部摄取", "全部引入"}
        if lower_input in all_keywords:
            all_ids = [str(p.get("paper_id", "")) for p in papers if p.get("paper_id")]
            logger.info("_parse_selection_response: user selected ALL {} papers", len(all_ids))
            return {"user_skip_ingest": False, "user_selected_papers": all_ids}

        # --- 3. Direct arXiv IDs ---
        arxiv_ids = re.findall(r'\b(\d{4}\.\d{4,5}v?\d*)\b', cleaned)
        if arxiv_ids:
            # Validate: at least one matches a paper in the list
            valid_ids = {str(p.get("paper_id", "")) for p in papers}
            matched = [pid for pid in arxiv_ids if pid in valid_ids]
            if matched:
                logger.info(
                    "_parse_selection_response: user selected {} arXiv IDs ({} matched)",
                    len(arxiv_ids), len(matched),
                )
                return {"user_skip_ingest": False, "user_selected_papers": matched}
            # All IDs are invalid — fall through to numeric/LLM

        # --- 4. Numeric indices (1-based) ---
        # Match patterns like "1", "1,3,5", "1 3 5", "1、3、5"
        index_tokens = re.findall(r'\b(\d+)\b', cleaned)
        if index_tokens:
            indices = []
            all_valid = True
            for token in index_tokens:
                idx = int(token)
                if 1 <= idx <= len(papers):
                    indices.append(idx)
                else:
                    all_valid = False
                    break
            if all_valid and indices:
                selected_ids = [str(papers[i - 1].get("paper_id", "")) for i in indices]
                logger.info(
                    "_parse_selection_response: user selected indices {} → {} paper(s)",
                    indices, len(selected_ids),
                )
                return {"user_skip_ingest": False, "user_selected_papers": selected_ids}

        # --- 5. LLM fallback for fuzzy matching ---
        logger.info(
            "_parse_selection_response: falling back to LLM parsing for input='{}'",
            cleaned[:100],
        )
        return await self._llm_parse_selection(cleaned, papers)

    async def _llm_parse_selection(
        self,
        user_input: str,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """LLM-based fallback for fuzzy selection parsing.

        Handles cases like partial titles, informal references, etc.
        """
        paper_briefs = []
        for i, p in enumerate(papers[:20], 1):
            paper_briefs.append(
                f"{i}. id={p.get('paper_id','')}; title={str(p.get('title',''))[:120]}"
            )

        prompt = (
            "The user was shown the following papers and asked to select which ones to ingest.\n"
            "Parse their response and return strict JSON.\n\n"
            "Papers:\n" + "\n".join(paper_briefs) + "\n\n"
            f"User response: {user_input}\n\n"
            "Return JSON: {\"action\": \"skip\" | \"select\" | \"all\", "
            "\"selected_indices\": [1,3,...] (1-based, only for action=select), "
            "\"reasoning\": \"...\"}\n"
            "- action=skip: user wants to skip ingestion\n"
            "- action=all: user wants all papers\n"
            "- action=select: user selected specific papers (fill selected_indices)\n"
            "Return ONLY the JSON object."
        )

        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": "You are a precise paper selection parser."},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            content = (response.content or "").strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())
            action = data.get("action", "skip")

            if action == "skip":
                return {"user_skip_ingest": True, "user_selected_papers": []}
            elif action == "all":
                all_ids = [str(p.get("paper_id", "")) for p in papers if p.get("paper_id")]
                return {"user_skip_ingest": False, "user_selected_papers": all_ids}
            elif action == "select":
                indices = data.get("selected_indices", [])
                selected_ids = [
                    str(papers[int(i) - 1].get("paper_id", ""))
                    for i in indices
                    if isinstance(i, (int, str)) and str(i).isdigit()
                    and 1 <= int(i) <= len(papers)
                ]
                return {"user_skip_ingest": False, "user_selected_papers": selected_ids}
            else:
                return {"user_skip_ingest": True, "user_selected_papers": []}
        except Exception as e:
            logger.warning("LLM selection parsing failed: {}, defaulting to skip", e)
            return {"user_skip_ingest": True, "user_selected_papers": []}

    async def synthesis_node(self, state: MultiAgentState) -> MultiAgentState:
        """Synthesis Agent: Generate answer from sources.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with draft_answer
        """
        logger.info("Synthesis Agent: Generating answer")

        self._refresh_memory_context(state)

        # Early-return ONLY in initial select phase (not during resume/ingest)
        if (
            state.get("research_phase", "") == "select"
            and not state.get("resume_mode")
            and state.get("draft_answer", "")
        ):
            logger.info("Synthesis Agent: Using existing draft_answer: {}", state["draft_answer"][:100])
            return state
        
        user_query = state.get("user_query", "")
        retrieval_results = state.get("retrieval_results", [])
        external_papers = state.get("external_papers", [])
        routing_decision = state.get("routing_decision", "hybrid")
        rewritten_queries = state.get("rewritten_queries", [])
        rewrite_reasoning = state.get("rewrite_reasoning", "")
        recent_dialog = state.get("recent_dialog_context", "") or "(empty)"
        long_term_memory = state.get("long_term_memory_context", "") or "(empty)"
        user_profile = state.get("user_profile_context", "") or "(empty)"
        soul_context = state.get("soul_context", "") or "(empty)"
        
        # Load paper metadata and format sources
        docs_meta = self.kb.load_docs_meta() if hasattr(self, "kb") else {}
        sources_section = format_sources_section(
            retrieval_results,
            docs_meta=docs_meta,
        )

        logger.info(
            "Synthesis Agent: sources_section:\n{}",
            sources_section
        )
        
        # Build critic feedback section from previous iteration
        critic_feedback = ""
        if state.get("critic_verdict") == "needs_revision":
            issues = state.get("critic_issues", [])
            suggestion = state.get("critic_suggestion", "")
            critic_feedback = (
                "\n## 上一轮审查反馈 (MUST ADDRESS)\n"
                f"问题: {issues}\n"
                f"改进建议: {suggestion}\n"
                "请根据以上反馈修正答案，不要重复同样的错误。\n"
            )
            await self._emit_progress("✍️ 正在根据审查反馈修正答案...")
        else:
            await self._emit_progress("✍️ 正在综合信息生成答案...")
        
        # Build prompt
        rewrite_context_parts = []
        if rewritten_queries:
            rewrite_context_parts.append(f"All rewritten queries: {', '.join(rewritten_queries)}")
        if rewrite_reasoning:
            rewrite_context_parts.append(f"Rewrite reasoning: {rewrite_reasoning}")
        rewrite_context = "\n".join(rewrite_context_parts) if rewrite_context_parts else ""

        prompt = SYNTHESIS_USER_TEMPLATE.format(
            user_query=user_query,
            rewrite_context=rewrite_context,
            long_term_memory_context=long_term_memory,
            recent_dialog_context=recent_dialog,
            user_profile_context=user_profile,
            soul_context=soul_context,
            sources_section=sources_section,
            critic_feedback=critic_feedback,
        )
        
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": self.prompts.synthesis},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            
            content = response.content or ""
            state["draft_answer"] = content.strip()
            
            # # Try to parse JSON
            # try:
            #     if "```json" in content:
            #         json_content = content.split("```json")[1].split("```")[0]
            #     elif "```" in content:
            #         json_content = content.split("```")[1].split("```")[0]
            #     else:
            #         json_content = content
                
            #     result = json_repair.loads(json_content.strip())
            #     state["draft_answer"] = result.get("answer", content)
            #     state["citations"] = result.get("citations", [])
            # except json.JSONDecodeError:
            #     # Use raw content if JSON parsing fails
            #     logger.warning("Synthesis Agent: Failed to parse JSON, using raw content as answer")
            #     state["draft_answer"] = content
            #     state["citations"] = []
            
            logger.info("Synthesis Agent: Generated answer (length={})", len(state["draft_answer"]))
            
        except Exception as e:
            logger.error("Synthesis Agent failed: {}", e)
            state["draft_answer"] = f"Error generating answer: {e}"
            state["error_message"] = str(e)
        
        return state
    
    async def critic_node(self, state: MultiAgentState) -> MultiAgentState:
        """Critic Agent: Review answer quality.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with critic_verdict and feedback
        """
        logger.info("Critic Agent: Reviewing answer")

        self._refresh_memory_context(state)

        # Early-return ONLY in initial select phase (not during resume/ingest)
        if (
            state.get("research_phase", "") == "select"
            and not state.get("resume_mode")
            and state.get("draft_answer", "")
        ):
            state["final_answer"] = state["draft_answer"]
            state["critic_verdict"] = "passed"
            logger.info("Critic Agent: Skipping review in select phase, using draft_answer as final_answer")
            return state
        
        await self._emit_progress("✅ 正在审查答案质量...")
        
        user_query = state.get("user_query", "")
        draft_answer = state.get("draft_answer", "")
        retrieval_results = state.get("retrieval_results", [])
        external_papers = state.get("external_papers", [])
        rewritten_queries = state.get("rewritten_queries", [])
        rewrite_reasoning = state.get("rewrite_reasoning", "")
        critic_history_context = state.get("critic_history_context", "") or "(empty)"
        recent_dialog = state.get("recent_dialog_context", "") or "(empty)"
        long_term_memory = state.get("long_term_memory_context", "") or "(empty)"
        user_profile = state.get("user_profile_context", "") or "(empty)"
        soul_context = state.get("soul_context", "") or "(empty)"
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)

        if iteration_count >= max_iterations:
            # Max iterations reached, force completion
            state["is_complete"] = True
            state["final_answer"] = draft_answer
            logger.warning("Critic Agent: Max iterations reached, forcing completion")
            await self._emit_progress("⚠️ 达到最大迭代次数，强制输出答案")
        
        # Load paper metadata and format sources
        docs_meta = self.kb.load_docs_meta() if hasattr(self, "kb") else {}
        sources_section = format_sources_section(
            retrieval_results,
            docs_meta=docs_meta,
        )
        
        # Include previous critic context for progressive review
        prev_verdict = state.get("critic_verdict", "")
        prev_issues = state.get("critic_issues", [])
        prev_suggestion = state.get("critic_suggestion", "")
        critic_context_extra = ""
        if prev_verdict and prev_issues:
            critic_context_extra = (
                f"\n上一轮审查结果: verdict={prev_verdict}, issues={prev_issues}, suggestion={prev_suggestion}\n"
            )

        critic_history_context = critic_history_context + critic_context_extra

        rewrite_context_parts = []
        if rewritten_queries:
            rewrite_context_parts.append(f"All rewritten queries: {', '.join(rewritten_queries)}")
        if rewrite_reasoning:
            rewrite_context_parts.append(f"Rewrite reasoning: {rewrite_reasoning}")
        rewrite_context = "\n".join(rewrite_context_parts) if rewrite_context_parts else ""
        
        # Build prompt
        prompt = CRITIC_USER_TEMPLATE.format(
            user_query=user_query,
            rewrite_context=rewrite_context,
            recent_dialog_context=recent_dialog,
            long_term_memory_context=long_term_memory,
            user_profile_context=user_profile,
            soul_context=soul_context,
            critic_history_context=critic_history_context,
            draft_answer=draft_answer,
            sources_section=sources_section,
        )
        
        try:
            response = await self.provider.chat_with_retry(
                model=self.provider.get_default_model(),
                messages=[
                    {"role": "system", "content": self.prompts.critic},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            
            content = response.content or ""
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            verdict = result.get("verdict", "needs_revision")
            state["critic_verdict"] = verdict
            state["critic_issues"] = result.get("issues", [])
            state["critic_suggestion"] = result.get("suggestion", "")
            state["critic_feedback"] = result.get("feedback", "")
            
            # Determine completion
            if verdict == "passed":
                state["is_complete"] = True
                state["final_answer"] = draft_answer
                logger.info("Critic Agent: Answer approved")
                await self._emit_progress("✅ 审查通过 — 答案质量合格")
            else:
                state["is_complete"] = False
                state["iteration_count"] = iteration_count + 1
                logger.info(
                    "Critic Agent: Answer needs {} (iteration {}/{})",
                    verdict,
                    state["iteration_count"],
                    max_iterations,
                )
            
        except Exception as e:
            logger.error("Critic Agent failed: {}", e)
            # On failure, assume answer is acceptable to avoid loops
            state["is_complete"] = True
            state["final_answer"] = draft_answer
            state["critic_verdict"] = "passed"
            state["critic_issues"] = [f"Critic review failed: {e}"]
        
        return state
    
    async def hybrid_entry_node(self, state: MultiAgentState) -> MultiAgentState:
        """Hybrid entry: Set up state for hybrid mode.
        
        Hybrid mode runs retrieval first, then research if needed.
        
        Args:
            state: Current workflow state
            
        Returns:
            State ready for hybrid execution
        """
        logger.info("Hybrid Mode: Starting with internal retrieval")
        # Hybrid starts with retrieval, may continue to research
        return state


def create_nodes(
    provider: LLMProvider,
    kb: PaperKnowledgeBase,
    tools: Dict[str, Any],
    *,
    progress_callback: Optional[Callable[[str], None]] = None,
    **config,
) -> AgentNodes:
    """Factory function to create AgentNodes instance.
    
    Args:
        provider: LLM provider
        kb: Paper knowledge base
        tools: Dictionary of available tools
        progress_callback: Optional callback for real-time progress updates
        **config: Additional configuration
        
    Returns:
        Configured AgentNodes instance
    """
    return AgentNodes(
        provider=provider,
        kb=kb,
        tools=tools,
        similarity_threshold=config.get("similarity_threshold", 0.2),
        top_k=config.get("top_k", 5),
        ingest_limit=config.get("ingest_limit", 3),
        progress_callback=progress_callback,
        memory_store=config.get("memory_store"),
    )