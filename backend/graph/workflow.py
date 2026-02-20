"""LangGraph workflow for the MARSA multi-agent research system.

This module assembles the research pipeline by connecting the Planner, Researcher,
Fact-Checker, and Synthesizer agents into a LangGraph StateGraph with conditional
routing and SQLite checkpointing.

The workflow supports:
- Sequential execution through all agents
- Parallel sub-query execution via LangGraph's Send API
- Conditional looping back to researcher if too many claims fail verification
- SQLite persistence for state inspection and resumption
- Human-in-the-loop checkpoints (optional)
"""

from typing import Literal, Optional, Union

import structlog
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agents.fact_checker import fact_check_node, should_loop_back
from agents.planner import planner_node
from agents.researcher import (
    merge_research_node,
    research_node,
    research_sub_query_node,
)
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
    
    # Clear hitl_feedback after consuming it (prevents stale feedback in next HITL round)
    result["hitl_feedback"] = None
    
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
    # Status remains as FACT_CHECKING - routing logic will determine next step
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
    # Clear hitl_feedback after consuming it
    result["hitl_feedback"] = None
    return result


async def hitl_checkpoint_node(state: AgentState) -> dict:
    """HITL checkpoint node that processes human feedback.
    
    This node runs AFTER the interrupt (we use interrupt_before).
    At this point, the user has already provided feedback via update_state,
    so hitl_feedback should be present in the state.
    
    We do NOT clear hitl_feedback here because route_after_hitl_feedback
    needs to read it. The planner node clears it after consuming.
    
    Args:
        state: Current agent state with hitl_feedback from user.
        
    Returns:
        State with status updated (feedback preserved for routing).
    """
    feedback = state.get("hitl_feedback")
    if feedback:
        action = feedback.action if hasattr(feedback, 'action') else feedback.get('action', '')
        logger.info("hitl_checkpoint_processing", action=action, has_query=bool(state.get("query")))
    else:
        logger.info("hitl_checkpoint_no_feedback", has_query=bool(state.get("query")))
    
    # Return only status update - LangGraph will merge with existing state
    # This preserves query, plan, claims, etc.
    return {
        "status": PipelineStatus.AWAITING_FEEDBACK.value,
    }


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


def route_sub_queries(state: AgentState) -> Union[list[Send], Literal["research_sequential"]]:
    """Route sub-queries for parallel or sequential execution.
    
    Uses LangGraph's Send API to fan out sub-queries to parallel workers
    when the plan indicates parallel execution is appropriate.
    
    This is the Spool connection: "Just like Spool fans out tasks across
    workers, MARSA fans out research sub-queries across parallel agent instances."
    
    Args:
        state: Current agent state with plan containing sub-queries.
        
    Returns:
        List of Send objects for parallel execution, or "research_sequential" literal.
    """
    plan = state.get("plan")
    
    if not plan or not plan.sub_queries:
        logger.warning("route_sub_queries_no_plan")
        return "research_sequential"
    
    if plan.parallel and len(plan.sub_queries) >= 2:
        # Fan out to parallel workers
        logger.info(
            "routing_parallel",
            sub_query_count=len(plan.sub_queries),
            parallel=True,
        )
        return [
            Send("research_sub_query", {"sub_query": sq, "parent_state": state})
            for sq in plan.sub_queries
        ]
    
    # Sequential execution
    logger.info(
        "routing_sequential",
        sub_query_count=len(plan.sub_queries),
        parallel=False,
    )
    return "research_sequential"


def route_after_hitl_feedback(state: AgentState) -> Literal["planner", "synthesizer", "end"]:
    """Route based on human-in-the-loop feedback.
    
    This function is called AFTER the hitl_checkpoint node and interrupt,
    so hitl_feedback should always be present in the state.
    
    Routing logic:
    - 'approve' -> proceed to synthesizer
    - 'dig_deeper' -> loop back to planner with new query
    - 'abort' -> end the workflow
    
    Args:
        state: Current agent state with hitl_feedback.
        
    Returns:
        Next node to route to.
    """
    feedback = state.get("hitl_feedback")
    
    if not feedback:
        # This shouldn't happen - log error and proceed to synthesizer
        logger.error("hitl_feedback_missing_after_interrupt")
        return "synthesizer"
    
    action = feedback.action if hasattr(feedback, 'action') else feedback.get('action', '')
    
    if action == "abort":
        logger.info("hitl_abort")
        return "end"
    elif action == "dig_deeper":
        logger.info("hitl_dig_deeper", topic=getattr(feedback, 'topic', None))
        return "planner"  # Go back to planner for new plan
    else:  # "approve" or default
        logger.info("hitl_approve")
        return "synthesizer"


def create_workflow(
    checkpointer_path: Optional[str] = None,
    enable_hitl: bool = False,
    use_memory_checkpointer: bool = True,
    enable_parallel: bool = True,
) -> StateGraph:
    """Create the MARSA research workflow graph.
    
    Builds and compiles the LangGraph StateGraph connecting all agents
    with appropriate edges and conditional routing.
    
    The workflow supports two modes:
    1. Sequential: planner -> researcher -> fact_checker -> synthesizer
    2. Parallel: planner -> (fan-out sub-queries) -> merge -> fact_checker -> synthesizer
    
    Parallel mode uses LangGraph's Send API to fan out sub-queries across
    parallel workers, similar to how Spool fans out tasks across workers.
    
    Args:
        checkpointer_path: Path for SQLite checkpointer. If None, uses default.
        enable_hitl: Whether to enable human-in-the-loop checkpoints.
        use_memory_checkpointer: If True, use InMemorySaver for simplicity.
                                 If False, requires async SQLite handling.
        enable_parallel: Whether to enable parallel sub-query execution.
        
    Returns:
        Compiled StateGraph ready for invocation.
    """
    logger.info(
        "creating_workflow",
        enable_hitl=enable_hitl,
        enable_parallel=enable_parallel,
    )
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add core nodes
    workflow.add_node("planner", planner_with_trace)
    workflow.add_node("fact_checker", fact_checker_with_status)
    workflow.add_node("synthesizer", synthesizer_with_status)
    
    # Add HITL checkpoint node if enabled
    if enable_hitl:
        workflow.add_node("hitl_checkpoint", hitl_checkpoint_node)
    
    # Add research nodes (both parallel and sequential paths)
    workflow.add_node("research_sequential", researcher_with_status)
    workflow.add_node("research_sub_query", research_sub_query_node)
    workflow.add_node("merge_research", merge_research_with_status)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    if enable_parallel:
        # Conditional routing: parallel fan-out or sequential
        workflow.add_conditional_edges(
            "planner",
            route_sub_queries,
            {
                "research_sequential": "research_sequential",
                # Send API handles "research_sub_query" dynamically
            },
        )
        
        # Parallel sub-queries merge back
        workflow.add_edge("research_sub_query", "merge_research")
        
        # Both paths lead to fact_checker
        workflow.add_edge("research_sequential", "fact_checker")
        workflow.add_edge("merge_research", "fact_checker")
    else:
        # Simple sequential path
        workflow.add_edge("planner", "research_sequential")
        workflow.add_edge("research_sequential", "fact_checker")
    
    # Conditional edge from fact_checker (with HITL support)
    if enable_hitl:
        # Route to HITL checkpoint first
        workflow.add_edge("fact_checker", "hitl_checkpoint")
        
        # Then conditional routing based on feedback (evaluated AFTER interrupt)
        workflow.add_conditional_edges(
            "hitl_checkpoint",
            route_after_hitl_feedback,
            {
                "planner": "planner",  # Loop back for dig_deeper (will replan)
                "synthesizer": "synthesizer",
                "end": END,
            },
        )
    else:
        workflow.add_conditional_edges(
            "fact_checker",
            route_after_fact_check,
            {
                "researcher": "research_sequential",
                "synthesizer": "synthesizer",
            },
        )
    
    # Synthesizer goes to END
    workflow.add_edge("synthesizer", END)
    
    # Get checkpointer - use shared singleton for memory mode (CRITICAL for HITL)
    if use_memory_checkpointer:
        checkpointer = get_shared_checkpointer()
        logger.info("using_shared_memory_checkpointer")
    else:
        # Note: SQLite checkpointer requires async context manager handling
        checkpointer = create_checkpointer(
            db_path=checkpointer_path, 
            use_memory=False,
        )
    
    # Compile with optional HITL
    compile_kwargs = {"checkpointer": checkpointer}
    
    if enable_hitl:
        # Pause BEFORE hitl_checkpoint for human review.
        # This ensures the routing decision (which reads hitl_feedback)
        # happens AFTER the user provides feedback via update_state.
        compile_kwargs["interrupt_before"] = ["hitl_checkpoint"]
        logger.info("hitl_enabled", interrupt_before="hitl_checkpoint")
    
    app = workflow.compile(**compile_kwargs)
    
    logger.info("workflow_created", enable_parallel=enable_parallel)
    return app


async def merge_research_with_status(state: AgentState) -> dict:
    """Merge research node wrapper that updates status.
    
    Args:
        state: Current agent state with parallel results.
        
    Returns:
        Updated state with merged research results.
    """
    updated_state = {**state, "status": PipelineStatus.RESEARCHING.value}
    result = await merge_research_node(updated_state)
    return result


# Shared checkpointer instance - CRITICAL for HITL resume to work
# Each InMemorySaver must be the SAME instance across initial run and resume
_shared_memory_checkpointer: Optional[InMemorySaver] = None


def get_shared_checkpointer() -> InMemorySaver:
    """Get or create the shared memory checkpointer.
    
    CRITICAL: For HITL workflow resumption to work, the same checkpointer
    instance must be used for both the initial workflow run and the resume.
    Creating a new InMemorySaver for each workflow would lose all state.
    
    Returns:
        The shared InMemorySaver instance.
    """
    global _shared_memory_checkpointer
    if _shared_memory_checkpointer is None:
        _shared_memory_checkpointer = InMemorySaver()
        logger.info("shared_checkpointer_created")
    return _shared_memory_checkpointer


# Default workflow instance
_default_workflow: Optional[StateGraph] = None


def get_workflow(
    checkpointer_path: Optional[str] = None,
    enable_hitl: bool = False,
    enable_parallel: bool = True,
    force_new: bool = False,
) -> StateGraph:
    """Get or create the default workflow instance.
    
    Uses a module-level singleton for efficiency, but can create
    a new instance if needed.
    
    Args:
        checkpointer_path: Path for SQLite checkpointer.
        enable_hitl: Whether to enable human-in-the-loop.
        enable_parallel: Whether to enable parallel sub-query execution.
        force_new: Force creation of a new workflow instance.
        
    Returns:
        Compiled StateGraph.
    """
    global _default_workflow
    
    if _default_workflow is None or force_new:
        _default_workflow = create_workflow(
            checkpointer_path=checkpointer_path,
            enable_hitl=enable_hitl,
            enable_parallel=enable_parallel,
        )
    
    return _default_workflow


async def run_research(
    query: str,
    thread_id: Optional[str] = None,
    enable_hitl: bool = False,
    enable_parallel: bool = True,
) -> AgentState:
    """Run a research query through the full pipeline.
    
    Convenience function that creates initial state and runs
    the workflow to completion.
    
    Args:
        query: The research query to investigate.
        thread_id: Optional thread ID for checkpointing.
        enable_hitl: Whether to enable human-in-the-loop.
        enable_parallel: Whether to enable parallel sub-query execution.
        
    Returns:
        Final AgentState with the research report.
    """
    import uuid
    
    from graph.state import create_initial_state
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Get workflow (force new to ensure correct settings)
    workflow = get_workflow(
        enable_hitl=enable_hitl,
        enable_parallel=enable_parallel,
        force_new=True,
    )
    
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
        enable_parallel=enable_parallel,
    )
    
    # Run the workflow
    result = await workflow.ainvoke(initial_state, config)
    
    logger.info(
        "research_complete",
        status=result.get("status"),
        report_length=len(result.get("report", "")),
    )
    
    return result
