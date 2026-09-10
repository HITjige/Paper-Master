"""Conditional edge functions for LangGraph workflow.

These functions determine the flow of execution between nodes.
"""

from __future__ import annotations

from typing import Literal

from loguru import logger

from nanobot.agent.multi_agent.state import MultiAgentState


def router_conditional(state: MultiAgentState) -> Literal["retrieval", "research", "hybrid_entry", "synthesis"]:
    """Route from Router to appropriate next node.
    
    In resume mode, skip routing and go directly to research (for ingest phase).
    
    Args:
        state: Current workflow state with routing_decision set
        
    Returns:
        Next node name: "retrieval", "research", "hybrid_entry", or "synthesis"
    """
    # Resume mode: skip routing, direct to research for ingest continuation
    if state.get("resume_mode"):
        logger.debug("Router conditional: resume_mode=True, directing to research")
        return "research"

    decision = state.get("routing_decision", "hybrid")
    
    logger.debug("Router conditional: decision={}", decision)
    
    if decision == "internal":
        return "retrieval"
    elif decision == "external":
        return "research"
    elif decision == "hybrid":
        return "hybrid_entry"
    else:  # direct
        return "synthesis"


def retrieval_conditional(state: MultiAgentState) -> Literal["synthesis", "research"]:
    """Route from Retrieval based on quality assessment.
    
    If post_research_retrieval is set (just came back from research after 
    ingesting papers), go directly to synthesis regardless of quality.
    Otherwise, if retrieval quality is sufficient go to synthesis,
    if insufficient go to research for external search.
    
    Args:
        state: Current workflow state with retrieval_quality set
        
    Returns:
        Next node name: "synthesis" or "research"
    """
    if state.get("post_research_retrieval"):
        logger.debug("Retrieval conditional: post-research retrieval, forcing synthesis")
        return "synthesis"

    # An attempted external search is terminal for this retrieval/research
    # cycle even when arXiv returned no results (or the search tool failed).
    # Otherwise an insufficient local result would immediately trigger the
    # same external search again.
    if state.get("external_search_completed"):
        logger.debug("Retrieval conditional: external search already attempted, forcing synthesis")
        return "synthesis"

    quality = state.get("retrieval_quality", "insufficient")
    routing_decision = state.get("routing_decision", "")

    logger.debug("Retrieval conditional: quality={}", quality)

    # Hybrid means both sources: a sufficient local hit must not bypass the
    # first external search.
    if routing_decision == "hybrid" and not state.get("external_search_completed"):
        return "research"

    # If quality is sufficient, proceed to synthesis
    if quality == "sufficient":
        return "synthesis"
    
    return "research"


def research_phase_conditional(state: MultiAgentState) -> Literal[
    "wait_for_selection", "continue_ingest", "to_retrieval", "to_synthesis"
]:
    """Route from Research node based on current phase.
    
    Phases:
    - search: Just completed search, need to wait for user selection
    - select: Waiting for external input (user selection) - returns to synthesis with selection prompt
    - ingest: User made selection, continue with ingestion (self-loop in research node)
    - complete: Retrieve only after a successful ingest; otherwise synthesize
      from external results or report the terminal search outcome.
    
    In resume mode, skip directly to ingest phase if user has already made a selection.
    
    Args:
        state: Current workflow state with research_phase set
        
    Returns:
        Next action for selection, ingestion, retrieval, or synthesis
    """
    phase = state.get("research_phase", "search")
    search_completed = state.get("search_completed", False)
    resume_mode = state.get("resume_mode", False)
    user_made_selection = bool(state.get("user_selected_papers")) or state.get("user_skip_ingest", False)
    
    logger.debug(
        "Research phase conditional: phase={}, search_completed={}, resume_mode={}, user_made_selection={}",
        phase, search_completed, resume_mode, user_made_selection,
    )
    
    # Resume mode: skip search/select, go directly to ingest.
    # Only return "continue_ingest" when ingest is still in progress;
    # once complete, flow through to retrieval as normal.
    if resume_mode:
        if phase == "ingest":
            return "continue_ingest"
        if state.get("post_research_retrieval"):
            return "to_retrieval"
        return "to_synthesis"
    
    # After search phase completes and user hasn't made selection yet, 
    # go to synthesis to show selection prompt (wait_for_selection)
    if phase == "select" or (phase == "search" and search_completed and not user_made_selection):
        return "wait_for_selection"
    
    # Ingest phase - user made selection, continue with ingestion (self-loop)
    if phase == "ingest" or (user_made_selection and phase != "complete"):
        return "continue_ingest"
    
    if state.get("post_research_retrieval"):
        return "to_retrieval"
    return "to_synthesis"


def critic_conditional(state: MultiAgentState) -> Literal["complete", "rewrite", "research"]:
    """Route from Critic based on review verdict.
    
    - "passed": Complete the workflow
    - "needs_revision": Go back to synthesis for rewriting
    - "needs_more_info": Go to research for more information
    
    Args:
        state: Current workflow state with critic_verdict set
        
    Returns:
        Next action: "complete", "rewrite", or "research"
    """
    verdict = state.get("critic_verdict", "needs_revision")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    logger.debug(
        "Critic conditional: verdict={}, iteration={}/{}",
        verdict,
        iteration_count,
        max_iterations,
    )
    
    # Check if max iterations reached
    if iteration_count >= max_iterations:
        logger.warning("Max iterations reached, forcing completion")
        return "complete"
    
    # Route based on verdict
    if verdict == "passed":
        return "complete"
    elif verdict == "needs_more_info":
        return "research"
    else:  # needs_revision
        return "rewrite"


def should_continue(state: MultiAgentState) -> Literal["continue", "end"]:
    """Determine if workflow should continue or end.
    
    Used as a global check to prevent infinite loops.
    
    Args:
        state: Current workflow state
        
    Returns:
        "continue" or "end"
    """
    is_complete = state.get("is_complete", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if is_complete:
        return "end"
    
    if iteration_count >= max_iterations:
        logger.warning("Max iterations reached in should_continue")
        return "end"
    
    return "continue"


# Node name constants for clarity
NODE_ROUTER = "router"
NODE_RETRIEVAL = "retrieval"
NODE_RESEARCH = "research"
NODE_SYNTHESIS = "synthesis"
NODE_CRITIC = "critic"
NODE_HYBRID_ENTRY = "hybrid_entry"

# Edge target constants
EDGE_COMPLETE = "complete"
EDGE_REWRITE = "rewrite"
