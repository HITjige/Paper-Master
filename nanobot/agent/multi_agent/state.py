"""State definitions for multi-agent workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class RoutingDecision(str, Enum):
    """Routing decisions for the Router Agent."""
    
    INTERNAL = "internal"      # Query internal knowledge base
    EXTERNAL = "external"      # Search external sources
    HYBRID = "hybrid"          # Both internal and external
    DIRECT = "direct"          # Direct answer without retrieval


class RetrievalQuality(str, Enum):
    """Quality assessment for retrieval results."""
    
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class CriticVerdict(str, Enum):
    """Verdict from Critic Agent."""
    
    PASSED = "passed"
    NEEDS_REVISION = "needs_revision"
    NEEDS_MORE_INFO = "needs_more_info"


# TypedDict for LangGraph state
class MultiAgentState(TypedDict, total=False):
    """State dictionary for the multi-agent workflow.
    
    This defines the schema for state passed between nodes in the LangGraph.
    All fields are optional to allow incremental state building.
    """
    
    # === Input ===
    user_query: str
    session_id: str
    router_memory_short: str
    router_memory_long: str
    last_routing_decision: str
    routing_context: Dict[str, Any]
    orchestrator_decision: str
    orchestrator_reasoning: str
    orchestrator_confidence: float
    retrieval_history_context: str
    research_history_context: str
    critic_history_context: str
    retrieval_judge_margin: float
    query_rewrite_enabled: bool
    query_rewrite_use_llm: bool
    retrieval_rewrite_confidence_threshold: float
    research_query_decompose_count: int
    rewritten_queries: List[str]  # List of all rewritten queries (retrieval + research)
    rewrite_reasoning: str
    rewrite_confidence: float
    rewrite_fallback_used: bool
    rewrite_history_context: str
    rewrite_fallback_delta_threshold: float
    rewrite_context_chars: int
    extracted_entities: List[Dict[str, str]]  # Entities extracted from the query (for entity-aware retrieval)
    sub_queries_detail: List[Dict[str, Any]]  # Full sub_queries structure from unified rewrite (each has rewritten_queries, target_paper, keywords, time_filter)
    requires_clarification: bool  # True if query references are too ambiguous to resolve
    post_research_retrieval: bool  # True after research→retrieval loop, prevents infinite cycle
    referenced_papers: List[Dict[str, str]]  # Papers mentioned in conversation: [{"paper_id", "title"}]
    recent_dialog_context: str
    long_term_memory_context: str
    user_profile_context: str
    soul_context: str
    session_summary_context: str
    
    # === Router Decision ===
    routing_decision: str          # "internal" | "external" | "hybrid" | "direct"
    routing_reasoning: str         # Explanation for the decision
    
    # === Retrieval Results ===
    retrieval_results: List[Dict[str, Any]]
    retrieval_quality: str         # "sufficient" | "insufficient"
    
    # === External Research ===
    external_papers: List[Dict[str, Any]]
    ingested_papers: List[str]     # List of paper_ids successfully ingested
    
    # === Research Phase Control (decoupled search & ingest) ===
    research_phase: str            # "search" | "select" | "ingest" | "complete"
    search_completed: bool         # True when search is done and results are ready
    papers_for_selection: List[Dict[str, Any]]  # Papers found for user to select from
    user_selected_papers: List[str]  # Paper IDs selected by user for ingestion
    user_skip_ingest: bool         # True if user chooses to skip ingestion
    
    # === Synthesis ===
    draft_answer: str
    citations: List[str]
    
    # === Critic Review ===
    critic_verdict: str            # "passed" | "needs_revision" | "needs_more_info"
    critic_issues: List[str]
    critic_suggestion: str
    critic_feedback: str
    
    # === Output ===
    final_answer: str
    
    # === Control Flow ===
    iteration_count: int
    max_iterations: int
    is_complete: bool
    error_message: str
    
    # === Loop Guard (prevents infinite retrieval→research→retrieval loops) ===
    loop_guard_count: int

    # === Resume Mode (paused → resume workflow across turns) ===
    resume_mode: bool          # True when resuming from paused user-selection state
    resume_phase: str          # Phase to resume: "ingest" | ""


@dataclass
class AgentConfig:
    """Configuration for multi-agent system."""
    
    max_iterations: int = 3
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.2
    external_search_top_k: int = 60
    external_rerank_top_k: int = 5
    external_ingest_limit: int = 3
    
    # Quality thresholds
    min_citations_required: int = 1
    max_hallucination_score: float = 0.3
    
    # Query rewrite defaults
    query_rewrite_enabled: bool = True
    query_rewrite_use_llm: bool = True
    retrieval_rewrite_confidence_threshold: float = 0.6
    research_query_decompose_count: int = 3
    rewrite_fallback_delta_threshold: float = 0.05
    rewrite_context_chars: int = 60000
    mmr_lambda: float = 0.7  # relevance-diversity trade-off for MMR selection


def create_initial_state(
    user_query: str,
    session_id: str = "",
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> MultiAgentState:
    """Create initial state for the workflow.
    
    Args:
        user_query: The user's original question
        session_id: Optional session identifier
        config: Optional agent configuration
        
    Returns:
        Initial MultiAgentState with defaults
    """
    cfg = config or AgentConfig()
    
    return {
        "user_query": user_query,
        "session_id": session_id or "default",
        "router_memory_short": "",
        "router_memory_long": "",
        "last_routing_decision": "",
        "routing_context": {},
        "orchestrator_decision": "",
        "orchestrator_reasoning": "",
        "orchestrator_confidence": 0.0,
        "retrieval_history_context": "",
        "research_history_context": "",
        "critic_history_context": "",
        "retrieval_judge_margin": 0.02,
        "query_rewrite_enabled": True,
        "query_rewrite_use_llm": True,
        "retrieval_rewrite_confidence_threshold": 0.6,
        "research_query_decompose_count": 3,
        "rewrite_reasoning": "",
        "rewrite_confidence": 0.0,
        "rewrite_fallback_used": False,
        "rewrite_history_context": "",
        "rewrite_fallback_delta_threshold": cfg.rewrite_fallback_delta_threshold,
        "rewrite_context_chars": cfg.rewrite_context_chars,
        "sub_queries_detail": [],
        "requires_clarification": False,
        "post_research_retrieval": False,
        "recent_dialog_context": kwargs.get("recent_dialog_context", ""),
        "long_term_memory_context": kwargs.get("long_term_memory_context", ""),
        "user_profile_context": kwargs.get("user_profile_context", ""),
        "soul_context": kwargs.get("soul_context", ""),
        "session_summary_context": kwargs.get("session_summary_context", ""),
        "routing_decision": "",
        "routing_reasoning": "",
        "retrieval_results": [],
        "retrieval_quality": "",
        "external_papers": [],
        "ingested_papers": [],
        
        # Research Phase Control (decoupled search & ingest)
        "research_phase": "search",  # "search" | "select" | "ingest" | "complete"
        "search_completed": False,
        "papers_for_selection": [],
        "user_selected_papers": [],
        "user_skip_ingest": False,
        
        "draft_answer": "",
        "citations": [],
        "critic_verdict": "",
        "critic_issues": [],
        "critic_suggestion": "",
        "critic_feedback": "",
        "final_answer": "",
        "iteration_count": 0,
        "max_iterations": cfg.max_iterations,
        "is_complete": False,
        "error_message": "",
        "loop_guard_count": 0,
        "resume_mode": False,
        "resume_phase": "",
        "rewritten_queries": [],
        "extracted_entities": [],
        "referenced_papers": [],
    }


def get_state_summary(state: MultiAgentState) -> str:
    """Get a human-readable summary of current state.
    
    Useful for debugging and logging.
    """
    lines = [
        f"Query: {state.get('user_query', 'N/A')[:50]}...",
        f"Routing: {state.get('routing_decision', 'pending')}",
        f"Retrieval Quality: {state.get('retrieval_quality', 'N/A')}",
        f"Research Phase: {state.get('research_phase', 'N/A')}",
        f"External Papers: {len(state.get('external_papers', []))}",
        f"Iteration: {state.get('iteration_count', 0)}/{state.get('max_iterations', 3)}",
        f"Complete: {state.get('is_complete', False)}",
    ]
    return " | ".join(lines)