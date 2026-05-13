"""LangGraph workflow definition for multi-agent system.

This module assembles the workflow graph with all nodes and edges.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from loguru import logger

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available. Multi-agent system will not function.")

from nanobot.agent.multi_agent.conditions import (
    NODE_CRITIC,
    NODE_RESEARCH,
    NODE_RETRIEVAL,
    NODE_ROUTER,
    NODE_SYNTHESIS,
    NODE_HYBRID_ENTRY,
    critic_conditional,
    hybrid_retrieval_conditional,
    research_phase_conditional,
    retrieval_conditional,
    router_conditional,
)
from nanobot.agent.multi_agent.nodes import AgentNodes, create_nodes
from nanobot.agent.multi_agent.state import MultiAgentState, create_initial_state

if TYPE_CHECKING:
    from nanobot.agent.paper_kb import PaperKnowledgeBase
    from nanobot.providers.base import LLMProvider


class MultiAgentGraph:
    """Multi-agent workflow graph manager.
    
    This class encapsulates the LangGraph workflow and provides
    a clean interface for running multi-agent tasks.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        kb: PaperKnowledgeBase,
        tools: Dict[str, Any],
        *,
        progress_callback: Callable[[str], None] | None = None,
        **config,
    ):
        """Initialize the multi-agent graph.
        
        Args:
            provider: LLM provider for agent reasoning
            kb: Paper knowledge base for retrieval
            tools: Dictionary of available tools
            progress_callback: Optional callback for real-time progress updates
            **config: Additional configuration options
        """
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "LangGraph is required for multi-agent system. "
                "Install with: pip install langgraph"
            )
        
        self.provider = provider
        self.kb = kb
        self.tools = tools
        self.config = config
        self._progress_callback = progress_callback
        
        # Create nodes (pass progress_callback)
        self.nodes = create_nodes(provider, kb, tools, progress_callback=progress_callback, **config)
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info("MultiAgentGraph initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Create graph builder
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node(NODE_ROUTER, self.nodes.router_node)
        workflow.add_node(NODE_RETRIEVAL, self.nodes.retrieval_node)
        workflow.add_node(NODE_RESEARCH, self.nodes.research_node)
        workflow.add_node(NODE_SYNTHESIS, self.nodes.synthesis_node)
        workflow.add_node(NODE_CRITIC, self.nodes.critic_node)
        workflow.add_node(NODE_HYBRID_ENTRY, self.nodes.hybrid_entry_node)
        
        # Set entry point
        workflow.set_entry_point(NODE_ROUTER)
        
        # Add conditional edges from Router
        workflow.add_conditional_edges(
            NODE_ROUTER,
            router_conditional,
            {
                "retrieval": NODE_RETRIEVAL,
                "research": NODE_RESEARCH,
                "hybrid_entry": NODE_HYBRID_ENTRY,
                "synthesis": NODE_SYNTHESIS,
            },
        )
        
        # Add conditional edges from Retrieval
        workflow.add_conditional_edges(
            NODE_RETRIEVAL,
            retrieval_conditional,
            {
                "synthesis": NODE_SYNTHESIS,
                "research": NODE_RESEARCH,
            },
        )
        
        # Add conditional edges from Hybrid Entry (goes to retrieval first)
        workflow.add_conditional_edges(
            NODE_HYBRID_ENTRY,
            lambda state: NODE_RETRIEVAL,  # Hybrid always starts with retrieval
            {NODE_RETRIEVAL: NODE_RETRIEVAL},
        )
        
        # After retrieval in hybrid mode, always go to research
        # Note: This is handled by the retrieval node checking routing_decision
        # We need to add a special check in retrieval_conditional for hybrid mode
        # Actually, let's modify the flow:
        # hybrid_entry -> retrieval -> research -> retrieval (post-research) -> synthesis
        
        # Research node uses conditional edges based on phase
        # - wait_for_selection: Go to synthesis to show selection prompt (user needs to respond)
        # - continue_ingest: Self-loop to continue with ingest phase within research node
        # - skip_to_retrieval: Go to retrieval for post-research search (ingest done or skipped)
        workflow.add_conditional_edges(
            NODE_RESEARCH,
            research_phase_conditional,
            {
                "wait_for_selection": NODE_SYNTHESIS,   # Go to synthesis to show selection UI
                "continue_ingest": NODE_RESEARCH,       # Self-loop to execute ingest phase
                "skip_to_retrieval": NODE_RETRIEVAL,    # Go to retrieval for post-research search
            },
        )
        
        # Synthesis always goes to critic
        workflow.add_edge(NODE_SYNTHESIS, NODE_CRITIC)
        
        # Add conditional edges from Critic
        workflow.add_conditional_edges(
            NODE_CRITIC,
            critic_conditional,
            {
                "complete": END,
                "rewrite": NODE_SYNTHESIS,
                "research": NODE_RESEARCH,
            },
        )
        
        # Compile graph
        compiled = workflow.compile()
        
        logger.debug("Multi-agent graph compiled successfully")
        
        return compiled
    
    async def run(
        self,
        user_query: str,
        session_id: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the multi-agent workflow.
        
        Args:
            user_query: The user's question
            session_id: Optional session identifier
            **kwargs: Additional state overrides. Special key:
                progress_callback: injected into AgentNodes._progress_callback
        
        Returns:
            Final state dictionary with results
        """
        # Extract progress_callback from kwargs so it doesn't pollute state
        progress_callback = kwargs.pop("progress_callback", None)
        if progress_callback is not None:
            self.nodes._progress_callback = progress_callback
        
        # Create initial state
        initial_state = create_initial_state(
            user_query=user_query,
            session_id=session_id,
            kwargs=kwargs,
        )
        
        # Apply any overrides (progress_callback already popped)
        initial_state.update(kwargs)
        
        logger.info(
            "Starting multi-agent workflow: query='{}...', session={}",
            user_query[:50],
            session_id,
        )
        
        try:
            # Run graph
            final_state = await self.graph.ainvoke(initial_state)
            
            logger.info(
                "Multi-agent workflow completed: iterations={}, complete={}",
                final_state.get("iteration_count", 0),
                final_state.get("is_complete", False),
            )
            
            return final_state
            
        except Exception as e:
            logger.error("Multi-agent workflow failed: {}", e)
            return {
                **initial_state,
                "error_message": str(e),
                "final_answer": f"Error: {e}",
                "is_complete": False,
            }

    async def resume(
        self,
        saved_state: Dict[str, Any],
        user_input: str,
    ) -> Dict[str, Any]:
        """Resume a paused workflow with user's paper selection response.

        Called when the workflow was paused at the "select" phase (waiting for
        user to choose papers for ingestion).  Parses the user's response,
        reconstructs continuation state, and runs the graph from the ingest
        phase through to completion.

        Args:
            saved_state: State dict saved from the previous pause. Must contain
                at minimum ``papers_for_selection`` and ``user_query``.
            user_input: The user's selection response — paper IDs, "skip",
                "all", numeric indices, or free text.

        Returns:
            Final state dictionary after completing the workflow (same shape
            as :meth:`run`).
        """
        # Validate that the saved state is sufficient to resume
        if not self._validate_saved_state(saved_state):
            logger.error(
                "Resume: invalid saved state — missing required keys. "
                "Falling back to fresh run."
            )
            return await self.run(
                user_query=saved_state.get("user_query", ""),
                session_id=saved_state.get("session_id", ""),
            )

        logger.info(
            "Resume: parsing user selection input='{}...' ({} papers available)",
            user_input[:80],
            len(saved_state.get("papers_for_selection", [])),
        )

        # 1. Parse user selection
        parsed = await self.nodes._parse_selection_response(
            user_input,
            saved_state.get("papers_for_selection", []),
        )

        # 2. Build continuation state: carry forward all context needed
        #    for the downstream nodes (synthesis, critic, retrieval)
        continuation = dict(saved_state)

        # Set resume mode flags so Router / Research skip ahead
        continuation["resume_mode"] = True
        continuation["resume_phase"] = "ingest"
        # Set the research phase to ingest so research_node picks it up
        continuation["research_phase"] = "ingest"

        # Inject parsed user selection
        continuation["user_selected_papers"] = parsed.get("user_selected_papers", [])
        continuation["user_skip_ingest"] = parsed.get("user_skip_ingest", False)

        # Ensure search_completed is set (needed by some conditional checks)
        continuation["search_completed"] = True

        logger.info(
            "Resume: user_selected_papers={}, user_skip_ingest={}",
            continuation["user_selected_papers"][:5],
            continuation["user_skip_ingest"],
        )

        # 3. Run the graph with pre-filled state.
        #    The graph will flow: Router (skips via resume_mode) → Research
        #    (ingest phase) → Retrieval (post-research) → Synthesis → Critic.
        try:
            final_state = await self.graph.ainvoke(continuation)

            logger.info(
                "Resume workflow completed: iterations={}, complete={}",
                final_state.get("iteration_count", 0),
                final_state.get("is_complete", False),
            )

            return final_state

        except Exception as e:
            logger.error("Resume workflow failed: {}", e)
            return {
                **continuation,
                "error_message": str(e),
                "final_answer": (
                    "An error occurred while processing your paper selection. "
                    f"Please try again.\n\nError: {e}"
                ),
                "is_complete": False,
            }

    @staticmethod
    def _validate_saved_state(state: Dict[str, Any]) -> bool:
        """Check whether a saved state dict is usable for resume.

        Args:
            state: The saved state dict to validate.

        Returns:
            True if the state has all required keys for resume.
        """
        required = {"papers_for_selection", "user_query", "research_phase"}
        if not required.issubset(state.keys()):
            missing = required - state.keys()
            logger.warning("Resume validation: missing keys: {}", missing)
            return False
        papers = state.get("papers_for_selection", [])
        if not isinstance(papers, list) or not papers:
            logger.warning("Resume validation: papers_for_selection is empty or not a list")
            return False
        return True

    def get_graph_visualization(self) -> Optional[str]:
        """Get Mermaid diagram of the workflow.
        
        Returns:
            Mermaid diagram string or None if not available
        """
        try:
            # This would require langgraph's visualization utilities
            # For now, return a static representation
            return """
graph TD
    START([Start]) --> ROUTER
    ROUTER -->|internal| RETRIEVAL
    ROUTER -->|external| RESEARCH
    ROUTER -->|hybrid| HYBRID_ENTRY
    ROUTER -->|direct| SYNTHESIS
    
    HYBRID_ENTRY --> RETRIEVAL
    RETRIEVAL -->|insufficient| RESEARCH
    RETRIEVAL -->|sufficient / post_research| SYNTHESIS
    
    RESEARCH -->|"post-ingest retrieval"| RETRIEVAL
    SYNTHESIS --> CRITIC
    
    CRITIC -->|passed| END([End])
    CRITIC -->|needs_revision| SYNTHESIS
    CRITIC -->|needs_more_info| RESEARCH
"""
        except Exception:
            return None


def build_multi_agent_graph(
    provider: LLMProvider,
    kb: PaperKnowledgeBase,
    tools: Dict[str, Any],
    *,
    progress_callback: Callable[[str], None] | None = None,
    **config,
) -> MultiAgentGraph:
    """Factory function to create a MultiAgentGraph instance.
    
    Args:
        provider: LLM provider
        kb: Paper knowledge base
        tools: Dictionary of available tools
        progress_callback: Optional callback for real-time progress updates
        **config: Additional configuration
        
    Returns:
        Configured MultiAgentGraph instance
        
    Example:
        >>> graph = build_multi_agent_graph(
        ...     provider=llm_provider,
        ...     kb=paper_kb,
        ...     tools={"paper_search": search_tool, ...},
        ...     max_iterations=3,
        ...     similarity_threshold=0.2,
        ... )
        >>> result = await graph.run("What are the latest Mamba papers?")
    """
    return MultiAgentGraph(provider, kb, tools, progress_callback=progress_callback, **config)


# Simple fallback for when LangGraph is not available
class SimpleMultiAgentRunner:
    """Simple fallback runner when LangGraph is not available.
    
    This implements a basic sequential workflow without graph structure.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        kb: PaperKnowledgeBase,
        tools: Dict[str, Any],
        **config,
    ):
        self.provider = provider
        self.kb = kb
        self.tools = tools
        self.config = config
        self.nodes = create_nodes(provider, kb, tools, **config)
    
    async def run(
        self,
        user_query: str,
        session_id: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run simple sequential workflow."""
        state = create_initial_state(user_query, session_id)
        state.update(kwargs)
        
        try:
            # Router
            state = await self.nodes.router_node(state)
            
            # Execute based on routing decision
            decision = state.get("routing_decision", "hybrid")
            
            if decision == "internal":
                state = await self.nodes.retrieval_node(state)
                if state.get("retrieval_quality") == "insufficient":
                    state = await self.nodes.research_node(state)
                    # Post-research retrieval to search newly ingested papers
                    state["post_research_retrieval"] = True
                    state = await self.nodes.retrieval_node(state)
            
            elif decision == "external":
                state = await self.nodes.research_node(state)
                # Post-research retrieval to search newly ingested papers
                state["post_research_retrieval"] = True
                state = await self.nodes.retrieval_node(state)
            
            elif decision == "hybrid":
                state = await self.nodes.retrieval_node(state)
                state = await self.nodes.research_node(state)
                # Post-research retrieval to search newly ingested papers
                state["post_research_retrieval"] = True
                state = await self.nodes.retrieval_node(state)
            
            # Synthesis
            state = await self.nodes.synthesis_node(state)
            
            # Critic (single pass)
            state = await self.nodes.critic_node(state)
            
            return state
            
        except Exception as e:
            logger.error("Simple workflow failed: {}", e)
            return {
                **state,
                "error_message": str(e),
                "final_answer": f"Error: {e}",
            }