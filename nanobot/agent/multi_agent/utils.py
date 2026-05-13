"""Utility functions for multi-agent system."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger


def extract_json_from_response(content: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response.
    
    Handles various formats:
    - Plain JSON
    - Markdown code blocks (```json ... ```)
    - Markdown generic blocks (``` ... ```)
    
    Args:
        content: Raw LLM response content
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    if not content:
        return None
    
    content = content.strip()
    
    # Try to extract from markdown code blocks
    if "```json" in content:
        try:
            json_str = content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except (IndexError, json.JSONDecodeError):
            pass
    
    if "```" in content:
        try:
            json_str = content.split("```")[1].split("```")[0]
            return json.loads(json_str.strip())
        except (IndexError, json.JSONDecodeError):
            pass
    
    # Try parsing the whole content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try finding JSON-like structure
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end+1])
    except json.JSONDecodeError:
        pass
    
    logger.warning("Failed to extract JSON from response: {}...", content[:100])
    return None


def format_paper_citation(paper: Dict[str, Any]) -> str:
    """Format a paper as a citation string.
    
    Args:
        paper: Paper metadata dict
        
    Returns:
        Formatted citation string
    """
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", [])
    year = paper.get("year", "")
    
    citation = f"[{paper_id}] {title}"
    
    if authors:
        if len(authors) > 2:
            citation += f" ({authors[0]} et al."
        else:
            citation += f" ({', '.join(authors)}"
        if year:
            citation += f", {year}"
        citation += ")"
    elif year:
        citation += f" ({year})"
    
    return citation


def format_retrieval_result(result: Dict[str, Any], max_text_length: int = 300) -> str:
    """Format a retrieval result for display.
    
    Args:
        result: Retrieval result dict
        max_text_length: Maximum text length to include
        
    Returns:
        Formatted string
    """
    paper_id = result.get("paper_id", "unknown")
    title = result.get("title", "Untitled")
    section = result.get("section", "")
    score = result.get("score", 0)
    text = result.get("text", "")[:max_text_length]
    
    lines = [
        f"**[{paper_id}]** {title}",
        f"Score: {score:.3f} | Section: {section}",
        f"> {text}..." if len(result.get("text", "")) > max_text_length else f"> {text}",
    ]
    
    return "\n".join(lines)


def merge_paper_results(
    internal_results: List[Dict[str, Any]],
    external_papers: List[Dict[str, Any]],
    max_total: int = 10,
) -> List[Dict[str, Any]]:
    """Merge and deduplicate paper results from internal and external sources.
    
    Args:
        internal_results: Results from internal KB
        external_papers: Papers from external search
        max_total: Maximum total results
        
    Returns:
        Merged and deduplicated list
    """
    seen_ids = set()
    merged = []
    
    # Add internal results first (higher priority)
    for result in internal_results:
        paper_id = result.get("paper_id", "")
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            merged.append({
                "source": "internal",
                **result,
            })
    
    # Add external papers
    for paper in external_papers:
        paper_id = paper.get("paper_id", "")
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            merged.append({
                "source": "external",
                **paper,
            })
    
    return merged[:max_total]


def calculate_retrieval_quality_score(results: List[Dict[str, Any]]) -> float:
    """Calculate overall quality score for retrieval results.
    
    Args:
        results: List of retrieval results
        
    Returns:
        Quality score between 0 and 1
    """
    if not results:
        return 0.0
    
    # Factors:
    # 1. Number of results (more is better, up to a point)
    # 2. Average similarity score
    # 3. Diversity (different papers)
    
    count_score = min(len(results) / 5, 1.0)  # Normalize to 5 results
    
    avg_similarity = sum(r.get("score", 0) for r in results) / len(results)
    
    unique_papers = len(set(r.get("paper_id", "") for r in results))
    diversity_score = min(unique_papers / 3, 1.0)  # Normalize to 3 papers
    
    # Weighted combination
    quality = 0.3 * count_score + 0.5 * avg_similarity + 0.2 * diversity_score
    
    return min(max(quality, 0.0), 1.0)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_workflow_result(state: Dict[str, Any]) -> str:
    """Format final workflow result for display.
    
    Args:
        state: Final workflow state
        
    Returns:
        Formatted result string
    """
    lines = []
    
    # Header
    lines.append("# Multi-Agent Workflow Result")
    lines.append("")
    
    # Query info
    lines.append(f"**Query:** {state.get('user_query', 'N/A')}")
    lines.append("")
    
    # Routing info
    lines.append(f"**Routing Decision:** {state.get('routing_decision', 'N/A')}")
    lines.append(f"**Reasoning:** {state.get('routing_reasoning', 'N/A')}")
    lines.append("")
    
    # Sources used
    sources = []
    if state.get('retrieval_results'):
        sources.append(f"Internal KB ({len(state['retrieval_results'])} chunks)")
    if state.get('external_papers'):
        sources.append(f"External Search ({len(state['external_papers'])} papers)")
    if sources:
        lines.append(f"**Sources:** {', '.join(sources)}")
        lines.append("")
    
    # Iterations
    lines.append(f"**Iterations:** {state.get('iteration_count', 0)}/{state.get('max_iterations', 3)}")
    lines.append("")
    
    # Answer
    lines.append("## Answer")
    lines.append("")
    lines.append(state.get('final_answer') or state.get('draft_answer', 'No answer generated.'))
    
    # Citations
    if state.get('citations'):
        lines.append("")
        lines.append("## Citations")
        for citation in state['citations']:
            lines.append(f"- {citation}")
    
    # Issues (if any)
    if state.get('critic_issues'):
        lines.append("")
        lines.append("## Review Notes")
        for issue in state['critic_issues']:
            lines.append(f"- ⚠️ {issue}")
    
    return "\n".join(lines)


def create_error_state(user_query: str, error_message: str) -> Dict[str, Any]:
    """Create an error state for failed workflows.
    
    Args:
        user_query: Original user query
        error_message: Error description
        
    Returns:
        Error state dict
    """
    return {
        "user_query": user_query,
        "routing_decision": "error",
        "routing_reasoning": f"Workflow failed: {error_message}",
        "final_answer": f"I encountered an error while processing your request: {error_message}",
        "is_complete": True,
        "error_message": error_message,
    }


class WorkflowLogger:
    """Structured logging for multi-agent workflow.
    
    Provides consistent logging format for workflow events.
    """
    
    def __init__(self, session_id: str = ""):
        self.session_id = session_id
    
    def _log(self, level: str, agent: str, message: str, **kwargs):
        """Internal logging method."""
        extra = {
            "session_id": self.session_id,
            "agent": agent,
            **kwargs,
        }
        getattr(logger, level)(f"[{agent}] {message}", **extra)
    
    def info(self, agent: str, message: str, **kwargs):
        """Log info message."""
        self._log("info", agent, message, **kwargs)
    
    def warning(self, agent: str, message: str, **kwargs):
        """Log warning message."""
        self._log("warning", agent, message, **kwargs)
    
    def error(self, agent: str, message: str, **kwargs):
        """Log error message."""
        self._log("error", agent, message, **kwargs)
    
    def debug(self, agent: str, message: str, **kwargs):
        """Log debug message."""
        self._log("debug", agent, message, **kwargs)
    
    def state_transition(
        self,
        from_node: str,
        to_node: str,
        reason: str = "",
    ):
        """Log state transition."""
        msg = f"Transition: {from_node} -> {to_node}"
        if reason:
            msg += f" ({reason})"
        self.info("workflow", msg, from_node=from_node, to_node=to_node)