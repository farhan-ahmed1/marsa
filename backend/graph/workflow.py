"""LangGraph workflow for the MARSA multi-agent research system.

This module assembles the research pipeline by connecting the Planner, Researcher,
Fact-Checker, and Synthesizer agents into a LangGraph StateGraph with conditional
routing and SQLite checkpointing.

The workflow supports:
- Sequential execution through all agents
- Conditional looping back to researcher if too many claims fail verification
- SQLite persistence for state inspection and resumption
- Human-in-the-loop checkpoints (optional)
"""

from typing import Literal, Optional

import structlog
from langgraph.graph import END, StateGraph

from agents.fact_checker import fact_check_node, should_loop_back
from agents.planner import planner_node
from agents.researcher import research_node
from agents.synthesizer import synthesize_node
from graph.checkpointer import create_checkpointer
from graph.state import AgentName, AgentState, PipelineStatus, TraceEvent

logger = structlog.get_logger(__name__)


def _add_planner_trace(state: AgentState) -> AgentState:
    """Add trace event for planner start."""
    agent_trace = state.get("agent_trace", []).copy()
    agent_trace.append(TraceEvent(
        agent=AgentName.PLANNER,
        action="start",
        detail="Starting query planning",
        metadata={"query": state.get("query", "")[:100]},
    ))
    return {**state, "agent_trace": agent_trace, "status": PipelineStatus.PLANNING.value}


async def planner_with_trace(state: AgentState) -> dict:
    """Planner node wrapper that adds trace events.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with plan and trace events.
    """
    agent_trace = state.get("agent_trace", []).copy()
    
    # Add starting trace
    agent_trace.append(TraceEvent(
        agent=AgentName.PLANNER,
        action="start",
        detail=f"Planning research for: {state.get('query', '')[:50]}...",
        metadata={"query": state.get("query", "")},
    ))
    
    # Execute planner
    result = await planner_node({**state, "agent_trace": agent_trace})
    
    # Add completion trace
    plan = result.get("plan")
    if plan:
        result_trace = result.get("agent_trace", agent_trace).copy()
        result_trace.append(TraceEvent(
            agent=AgentName.PLANNER,
            action="complete",
            detail=f"Created plan with {len(plan.sub_queries)} sub-queries",
            metadata={
                "query_type": plan.query_type.value,
                "sub_query_count": len(plan.sub_queries),
                "parallel": plan.parallel,
                "search_strategy": plan.search_strategy.value,
            },
        ))
        result["agent_trace"] = result_trace
        result["status"] = PipelineStatus.RESEARCHING.value
    else:
        result_trace = result.get("agent_trace", agent_trace).copy()
        result_trace.append(TraceEvent(
            agent=AgentName.PLANNER,
            action="error",
            detail="Failed to create query plan",
        ))
        result["agent_trace"] = result_trace
        result["status"] = PipelineStatus.FAILED.value
    
    return result


async def researcher_with_status(state: AgentState) -> dict:
    """Researcher node wrapper that updates status.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with research results.
    """
    # Update status before research
    updated_state = {**state, "status": PipelineStatus.RESEARCHING.value}
    result = await research_node(updated_state)
    return result


async def fact_checker_with_status(state: AgentState) -> dict:
    """Fact-checker node wrapper that updates status.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with verification results.
    """
    # Update status before fact-checking
    updated_state = {**state, "status": PipelineStatus.FACT_CHECKING.value}
    result = await fact_check_node(updated_state)
    return result


async def synthesizer_with_status(state: AgentState) -> dict:
    """Synthesizer node wrapper that updates status.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with final report.
    """
    # Update status before synthesis
    updated_state = {**state, "status": PipelineStatus.SYNTHESIZING.value}
    result = await synthesize_node(updated_state)
    return result


def route_after_fact_check(state: AgentState) -> Literal["researcher", "synthesizer"]:
    """Route to researcher for re-research or synthesizer to finish.
    
    This conditional edge function determines whether to loop back
    to the researcher agent or proceed to synthesis based on the
    fact-checking results.
    
    Args:
        state: Current agent state with verification results.
        
    Returns:
        "researcher" to loop back, "synthesizer" to proceed.
    """
    route = should_loop_back(state)
    logger.info("workflow_routing", route=route)
    return route


def create_workflow(
    checkpointer_path: Optional[str] = None,
    enable_hitl: bool = False,
    use_memory_checkpointer: bool = True,
) -> StateGraph:
    """Create the MARSA research workflow graph.
    
    Builds and compiles the LangGraph StateGraph connecting all agents
    with appropriate edges and conditional routing.
    
    Args:
        checkpointer_path: Path for SQLite checkpointer. If None, uses default.
        enable_hitl: Whether to enable human-in-the-loop checkpoints.
        use_memory_checkpointer: If True, use InMemorySaver for simplicity.
                                 If False, requires async SQLite handling.
        
    Returns:
        Compiled StateGraph ready for invocation.
    """
    logger.info("creating_workflow", enable_hitl=enable_hitl)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_with_trace)
    workflow.add_node("researcher", researcher_with_status)
    workflow.add_node("fact_checker", fact_checker_with_status)
    workflow.add_node("synthesizer", synthesizer_with_status)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "fact_checker")
    
    # Conditional edge from fact_checker
    workflow.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        {
            "researcher": "researcher",
            "synthesizer": "synthesizer",
        },
    )
    
    # Synthesizer goes to END
    workflow.add_edge("synthesizer", END)
    
    # Create checkpointer (InMemorySaver by default for simplicity)
    checkpointer = create_checkpointer(
        db_path=checkpointer_path, 
        use_memory=use_memory_checkpointer,
    )
    
    # Compile with optional HITL
    compile_kwargs = {"checkpointer": checkpointer}
    
    if enable_hitl:
        # Pause after fact-checking for human review
        compile_kwargs["interrupt_after"] = ["fact_checker"]
        logger.info("hitl_enabled", interrupt_after="fact_checker")
    
    app = workflow.compile(**compile_kwargs)
    
    logger.info("workflow_created")
    return app


# Default workflow instance
_default_workflow: Optional[StateGraph] = None


def get_workflow(
    checkpointer_path: Optional[str] = None,
    enable_hitl: bool = False,
    force_new: bool = False,
) -> StateGraph:
    """Get or create the default workflow instance.
    
    Uses a module-level singleton for efficiency, but can create
    a new instance if needed.
    
    Args:
        checkpointer_path: Path for SQLite checkpointer.
        enable_hitl: Whether to enable human-in-the-loop.
        force_new: Force creation of a new workflow instance.
        
    Returns:
        Compiled StateGraph.
    """
    global _default_workflow
    
    if _default_workflow is None or force_new:
        _default_workflow = create_workflow(
            checkpointer_path=checkpointer_path,
            enable_hitl=enable_hitl,
        )
    
    return _default_workflow


async def run_research(
    query: str,
    thread_id: Optional[str] = None,
    enable_hitl: bool = False,
) -> AgentState:
    """Run a research query through the full pipeline.
    
    Convenience function that creates initial state and runs
    the workflow to completion.
    
    Args:
        query: The research query to investigate.
        thread_id: Optional thread ID for checkpointing.
        enable_hitl: Whether to enable human-in-the-loop.
        
    Returns:
        Final AgentState with the research report.
    """
    import uuid
    
    from graph.state import create_initial_state
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Get workflow
    workflow = get_workflow(enable_hitl=enable_hitl)
    
    # Create config for checkpointing
    config = {
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4()),
        }
    }
    
    logger.info(
        "running_research",
        query=query[:50],
        thread_id=config["configurable"]["thread_id"],
    )
    
    # Run the workflow
    result = await workflow.ainvoke(initial_state, config)
    
    logger.info(
        "research_complete",
        status=result.get("status"),
        report_length=len(result.get("report", "")),
    )
    
    return result
