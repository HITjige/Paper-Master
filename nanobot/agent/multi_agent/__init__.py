"""Multi-agent system for paper retrieval and analysis.

This package implements a LangGraph-based multi-agent workflow with 5 specialized agents:
- Router Agent: Decides execution path based on user intent
- Retrieval Agent: Queries internal knowledge base
- Research Agent: Searches and ingests external papers
- Synthesis Agent: Generates answers from retrieved information
- Critic Agent: Reviews and validates answer quality

Example usage:
    >>> from nanobot.agent.multi_agent import build_multi_agent_graph
    >>> graph = build_multi_agent_graph(provider, kb, tools)
    >>> result = await graph.run("What are the latest Mamba papers?")
    >>> print(result["final_answer"])
"""

from nanobot.agent.multi_agent.graph import build_multi_agent_graph, MultiAgentGraph
from nanobot.agent.multi_agent.state import (
    AgentConfig,
    CriticVerdict,
    MultiAgentState,
    RetrievalQuality,
    RoutingDecision,
    create_initial_state,
    get_state_summary,
)
from nanobot.agent.multi_agent.utils import (
    calculate_retrieval_quality_score,
    create_error_state,
    extract_json_from_response,
    format_paper_citation,
    format_retrieval_result,
    format_workflow_result,
    merge_paper_results,
)

__all__ = [
    # Graph
    "build_multi_agent_graph",
    "MultiAgentGraph",
    # State
    "MultiAgentState",
    "RoutingDecision",
    "RetrievalQuality",
    "CriticVerdict",
    "AgentConfig",
    "create_initial_state",
    "get_state_summary",
    # Utils
    "extract_json_from_response",
    "format_paper_citation",
    "format_retrieval_result",
    "format_workflow_result",
    "merge_paper_results",
    "calculate_retrieval_quality_score",
    "create_error_state",
]