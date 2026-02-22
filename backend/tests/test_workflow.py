"""Unit tests for the workflow module routing and wrapper functions.

These tests exercise the planner/researcher/fact_checker/synthesizer
status wrappers, store_memory_node, and the hitl_checkpoint_node
without running the real LLM-backed agents.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from graph.state import (  # noqa: E402
    AgentName,
    AgentState,
    Claim,
    ComplexityLevel,
    ConfidenceLevel,
    HITLFeedback,
    PipelineStatus,
    QueryPlan,
    QueryType,
    SearchStrategy,
    TraceEvent,
    VerificationResult,
    VerificationVerdict,
    create_initial_state,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_plan(**overrides):
    defaults = dict(
        query_type=QueryType.FACTUAL,
        sub_queries=["sub-q-1", "sub-q-2"],
        parallel=True,
        needs_fact_check=True,
        search_strategy=SearchStrategy.WEB_ONLY,
        estimated_complexity=ComplexityLevel.MEDIUM,
    )
    defaults.update(overrides)
    return QueryPlan(**defaults)


def _make_state(**overrides):
    state = create_initial_state("Test query")
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# planner_with_trace
# ---------------------------------------------------------------------------


class TestPlannerWithTrace:
    """Tests for the planner trace wrapper."""

    async def test_planner_with_trace_success(self):
        from graph.workflow import planner_with_trace

        plan = _make_plan()
        mock_result = {
            "plan": plan,
            "agent_trace": [
                TraceEvent(agent=AgentName.PLANNER, action="start", detail="Planning"),
            ],
        }

        with (
            patch("graph.workflow.planner_node", new_callable=AsyncMock, return_value=mock_result),
            patch("graph.workflow.get_relevant_memories", return_value=""),
        ):
            state = _make_state()
            result = await planner_with_trace(state)

        assert result["plan"] == plan
        # Should have added completion trace
        traces = result["agent_trace"]
        assert any(t.action == "complete" for t in traces)
        assert result["status"] == PipelineStatus.RESEARCHING.value
        # hitl_feedback should be cleared
        assert result.get("hitl_feedback") is None

    async def test_planner_with_trace_failed_plan(self):
        from graph.workflow import planner_with_trace

        mock_result = {
            "plan": None,
            "agent_trace": [],
        }

        with (
            patch("graph.workflow.planner_node", new_callable=AsyncMock, return_value=mock_result),
            patch("graph.workflow.get_relevant_memories", return_value=""),
        ):
            result = await planner_with_trace(_make_state())

        assert result["status"] == PipelineStatus.FAILED.value
        traces = result["agent_trace"]
        assert any(t.action == "error" for t in traces)

    async def test_planner_with_memory_context(self):
        from graph.workflow import planner_with_trace

        plan = _make_plan()

        # planner_node receives state with pre-populated agent_trace;
        # it must carry those traces through in its return value for
        # planner_with_trace to append the "complete" event correctly.
        async def _fake_planner(state):
            return {
                "plan": plan,
                "agent_trace": state.get("agent_trace", []),
            }

        with (
            patch("graph.workflow.planner_node", side_effect=_fake_planner),
            patch("graph.workflow.get_relevant_memories", return_value="## Prior Research Context\nSome prior findings"),
        ):
            result = await planner_with_trace(_make_state())

        assert result["memory_context"] != ""
        traces = result["agent_trace"]
        assert any(t.action == "memory_retrieved" for t in traces)

    async def test_planner_memory_retrieval_failure(self):
        from graph.workflow import planner_with_trace

        plan = _make_plan()
        mock_result = {"plan": plan, "agent_trace": []}

        with (
            patch("graph.workflow.planner_node", new_callable=AsyncMock, return_value=mock_result),
            patch("graph.workflow.get_relevant_memories", side_effect=RuntimeError("DB down")),
        ):
            result = await planner_with_trace(_make_state())

        # Should still succeed with empty memory context
        assert result["memory_context"] == ""


# ---------------------------------------------------------------------------
# researcher_with_status
# ---------------------------------------------------------------------------


class TestResearcherWithStatus:
    """Tests for the researcher status wrapper."""

    async def test_wraps_research_node(self):
        from graph.workflow import researcher_with_status

        mock_result = {"research_results": [{"title": "r1"}]}
        with patch("graph.workflow.research_node", new_callable=AsyncMock, return_value=mock_result):
            result = await researcher_with_status(_make_state())

        assert result["research_results"] == [{"title": "r1"}]


# ---------------------------------------------------------------------------
# fact_checker_with_status
# ---------------------------------------------------------------------------


class TestFactCheckerWithStatus:
    """Tests for the fact checker status wrapper."""

    async def test_wraps_fact_check_node(self):
        from graph.workflow import fact_checker_with_status

        mock_result = {"verification_results": []}
        with patch("graph.workflow.fact_check_node", new_callable=AsyncMock, return_value=mock_result):
            result = await fact_checker_with_status(_make_state())

        assert "verification_results" in result


# ---------------------------------------------------------------------------
# synthesizer_with_status
# ---------------------------------------------------------------------------


class TestSynthesizerWithStatus:
    """Tests for the synthesizer status wrapper."""

    async def test_wraps_synthesize_node(self):
        from graph.workflow import synthesizer_with_status

        mock_result = {"report": "Final report text", "hitl_feedback": "stale"}
        with patch("graph.workflow.synthesize_node", new_callable=AsyncMock, return_value=mock_result):
            result = await synthesizer_with_status(_make_state())

        assert result["report"] == "Final report text"
        # hitl_feedback should be cleared
        assert result.get("hitl_feedback") is None


# ---------------------------------------------------------------------------
# store_memory_node
# ---------------------------------------------------------------------------


class TestStoreMemoryNode:
    """Tests for the store_memory_node wrapper."""

    async def test_stores_memory_successfully(self):
        from graph.workflow import store_memory_node

        state = _make_state()
        with patch("graph.workflow.store_session") as mock_store:
            result = await store_memory_node(state)

        mock_store.assert_called_once_with(state)
        traces = result.get("agent_trace", [])
        assert any(t.action == "memory_stored" for t in traces)

    async def test_handles_memory_store_failure(self):
        from graph.workflow import store_memory_node

        state = _make_state()
        with patch("graph.workflow.store_session", side_effect=RuntimeError("ChromaDB error")):
            result = await store_memory_node(state)

        # Should not raise, just return state unchanged
        assert result["query"] == "Test query"


# ---------------------------------------------------------------------------
# hitl_checkpoint_node
# ---------------------------------------------------------------------------


class TestHITLCheckpointNode:
    """Tests for the HITL checkpoint node."""

    async def test_with_feedback(self):
        from graph.workflow import hitl_checkpoint_node

        state = _make_state(
            hitl_feedback=HITLFeedback(action="approve"),
        )
        result = await hitl_checkpoint_node(state)
        assert result["status"] == PipelineStatus.AWAITING_FEEDBACK.value

    async def test_without_feedback(self):
        from graph.workflow import hitl_checkpoint_node

        state = _make_state(hitl_feedback=None)
        result = await hitl_checkpoint_node(state)
        assert result["status"] == PipelineStatus.AWAITING_FEEDBACK.value


# ---------------------------------------------------------------------------
# merge_research_with_status
# ---------------------------------------------------------------------------


class TestMergeResearchWithStatus:
    """Tests for the merge_research_with_status wrapper."""

    async def test_wraps_merge_research_node(self):
        from graph.workflow import merge_research_with_status

        mock_result = {"research_results": [{"title": "merged"}]}
        with patch("graph.workflow.merge_research_node", new_callable=AsyncMock, return_value=mock_result):
            result = await merge_research_with_status(_make_state())

        assert result["research_results"] == [{"title": "merged"}]


# ---------------------------------------------------------------------------
# get_workflow / get_shared_checkpointer
# ---------------------------------------------------------------------------


class TestGetWorkflow:
    """Tests for workflow singleton management."""

    def test_get_shared_checkpointer_singleton(self):
        from graph.workflow import get_shared_checkpointer
        cp1 = get_shared_checkpointer()
        cp2 = get_shared_checkpointer()
        assert cp1 is cp2

    def test_get_workflow_creates_once(self):
        from graph.workflow import get_workflow
        wf1 = get_workflow(force_new=True)
        wf2 = get_workflow()
        assert wf1 is wf2

    def test_get_workflow_force_new(self):
        from graph.workflow import get_workflow
        wf1 = get_workflow(force_new=True)
        wf2 = get_workflow(force_new=True)
        # Different instances when force_new
        assert wf1 is not wf2
