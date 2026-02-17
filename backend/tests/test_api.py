"""Tests for the MARSA API endpoints.

Tests the FastAPI server including query submission, SSE streaming,
report retrieval, HITL feedback, and health checks.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from api.main import app  # noqa: E402
from api.models import (  # noqa: E402
    QueryRequest,
    FeedbackRequest,
    SSEEvent,
    SSEEventType,
)
from api.streaming import event_queue_manager  # noqa: E402
from graph.state import (  # noqa: E402
    PipelineStatus,
    create_initial_state,
)


@pytest.fixture
def test_client():
    """Create an async test client for the API."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def mock_workflow():
    """Mock the workflow execution to avoid real LLM calls."""
    with patch("api.streaming.run_workflow_with_streaming") as mock:
        mock.return_value = create_initial_state("test query")
        yield mock


@pytest.fixture
def mock_create_workflow():
    """Mock workflow creation."""
    mock_workflow = MagicMock()
    mock_workflow.astream_events = AsyncMock(return_value=iter([]))
    mock_workflow.ainvoke = AsyncMock(return_value={})
    
    with patch("api.streaming.create_workflow") as mock:
        mock.return_value = mock_workflow
        yield mock


class TestHealthEndpoint:
    """Tests for the /api/health endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_check_returns_ok(self, test_client):
        """Test that health check returns OK status."""
        async with test_client as client:
            response = await client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert "dependencies" in data
    
    @pytest.mark.asyncio
    async def test_health_check_includes_dependencies(self, test_client):
        """Test that health check includes dependency status."""
        async with test_client as client:
            response = await client.get("/api/health")
        
        data = response.json()
        assert "langgraph" in data["dependencies"]
        assert "active_streams" in data["dependencies"]


class TestQuerySubmission:
    """Tests for the POST /api/query endpoint."""
    
    @pytest.mark.asyncio
    async def test_submit_valid_query(self, test_client, mock_create_workflow):
        """Test submitting a valid query returns 202 with stream_id."""
        async with test_client as client:
            response = await client.post(
                "/api/query",
                json={"query": "What is gRPC?"}
            )
        
        assert response.status_code == 202
        data = response.json()
        assert "stream_id" in data
        assert data["status"] == "accepted"
        assert len(data["stream_id"]) == 36  # UUID format
    
    @pytest.mark.asyncio
    async def test_submit_query_with_options(self, test_client, mock_create_workflow):
        """Test submitting a query with HITL and parallel options."""
        async with test_client as client:
            response = await client.post(
                "/api/query",
                json={
                    "query": "Compare Rust vs Go",
                    "enable_hitl": True,
                    "enable_parallel": False,
                }
            )
        
        assert response.status_code == 202
        data = response.json()
        assert "stream_id" in data
    
    @pytest.mark.asyncio
    async def test_submit_empty_query_fails(self, test_client):
        """Test that empty query returns 422 validation error."""
        async with test_client as client:
            response = await client.post(
                "/api/query",
                json={"query": ""}
            )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_submit_short_query_fails(self, test_client):
        """Test that too-short query returns 422 validation error."""
        async with test_client as client:
            response = await client.post(
                "/api/query",
                json={"query": "ab"}
            )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_submit_whitespace_only_query_fails(self, test_client):
        """Test that whitespace-only query returns 422 validation error."""
        async with test_client as client:
            response = await client.post(
                "/api/query",
                json={"query": "   "}
            )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_submit_missing_query_field_fails(self, test_client):
        """Test that missing query field returns 422 validation error."""
        async with test_client as client:
            response = await client.post(
                "/api/query",
                json={}
            )
        
        assert response.status_code == 422


class TestStreamEndpoint:
    """Tests for the GET /api/query/{stream_id}/stream endpoint."""
    
    @pytest.mark.asyncio
    async def test_stream_unknown_id_returns_404(self, test_client):
        """Test that unknown stream_id returns 404."""
        async with test_client as client:
            response = await client.get(
                "/api/query/unknown-stream-id/stream"
            )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_stream_returns_sse_content_type(self, test_client, mock_create_workflow):
        """Test that stream endpoint returns SSE content type."""
        # First create a stream
        async with test_client as client:
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query for SSE"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            # Small delay to allow stream setup
            await asyncio.sleep(0.1)
            
            # Connect to stream
            async with client.stream("GET", f"/api/query/{stream_id}/stream") as response:
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestReportEndpoint:
    """Tests for the GET /api/query/{stream_id}/report endpoint."""
    
    @pytest.mark.asyncio
    async def test_report_unknown_id_returns_404(self, test_client):
        """Test that unknown stream_id returns 404."""
        async with test_client as client:
            response = await client.get(
                "/api/query/unknown-stream-id/report"
            )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_report_before_completion_returns_425(self, test_client, mock_create_workflow):
        """Test that requesting report before completion returns 425."""
        async with test_client as client:
            # Submit query
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            # Small delay
            await asyncio.sleep(0.1)
            
            # Set state to planning (not complete)
            await event_queue_manager.update_state(
                stream_id,
                {"status": "planning", "report": None}
            )
            
            # Try to get report
            response = await client.get(f"/api/query/{stream_id}/report")
        
        assert response.status_code == 425
    
    @pytest.mark.asyncio
    async def test_report_after_completion_returns_200(self, test_client, mock_create_workflow):
        """Test that requesting report after completion returns 200."""
        async with test_client as client:
            # Submit query
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            # Small delay
            await asyncio.sleep(0.1)
            
            # Set state to completed with report
            await event_queue_manager.update_state(
                stream_id,
                {
                    "status": "completed",
                    "report": "Test report content",
                    "claims": [],
                    "verification_results": [],
                    "citations": [],
                    "agent_trace": [],
                }
            )
            
            # Get report
            response = await client.get(f"/api/query/{stream_id}/report")
        
        assert response.status_code == 200
        data = response.json()
        assert data["stream_id"] == stream_id
        assert data["status"] == "completed"
        assert data["raw_report"] == "Test report content"


class TestFeedbackEndpoint:
    """Tests for the POST /api/query/{stream_id}/feedback endpoint."""
    
    @pytest.mark.asyncio
    async def test_feedback_unknown_id_returns_404(self, test_client):
        """Test that unknown stream_id returns 404."""
        async with test_client as client:
            response = await client.post(
                "/api/query/unknown-stream-id/feedback",
                json={"action": "approve"}
            )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_feedback_invalid_action_returns_422(self, test_client, mock_create_workflow):
        """Test that invalid action returns 422."""
        async with test_client as client:
            # Submit query
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            await asyncio.sleep(0.1)
            
            # Set state to awaiting feedback
            await event_queue_manager.update_state(
                stream_id,
                {"status": PipelineStatus.AWAITING_FEEDBACK.value}
            )
            
            # Try invalid action
            response = await client.post(
                f"/api/query/{stream_id}/feedback",
                json={"action": "invalid_action"}
            )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_feedback_not_at_checkpoint_returns_400(self, test_client, mock_create_workflow):
        """Test that feedback when not at checkpoint returns 400."""
        async with test_client as client:
            # Submit query
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            await asyncio.sleep(0.1)
            
            # Set state to researching (not at checkpoint)
            await event_queue_manager.update_state(
                stream_id,
                {"status": "researching"}
            )
            
            # Try to submit feedback
            response = await client.post(
                f"/api/query/{stream_id}/feedback",
                json={"action": "approve"}
            )
        
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_feedback_dig_deeper_requires_topic(self, test_client, mock_create_workflow):
        """Test that dig_deeper action requires topic."""
        async with test_client as client:
            # Submit query
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            await asyncio.sleep(0.1)
            
            # Set state to awaiting feedback
            await event_queue_manager.update_state(
                stream_id,
                {"status": PipelineStatus.AWAITING_FEEDBACK.value}
            )
            
            # Try dig_deeper without topic
            response = await client.post(
                f"/api/query/{stream_id}/feedback",
                json={"action": "dig_deeper"}
            )
        
        assert response.status_code == 400
        assert "Topic required" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_feedback_correct_requires_correction(self, test_client, mock_create_workflow):
        """Test that correct action requires correction text."""
        async with test_client as client:
            # Submit query
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            await asyncio.sleep(0.1)
            
            # Set state to awaiting feedback
            await event_queue_manager.update_state(
                stream_id,
                {"status": PipelineStatus.AWAITING_FEEDBACK.value}
            )
            
            # Try correct without correction text
            response = await client.post(
                f"/api/query/{stream_id}/feedback",
                json={"action": "correct"}
            )
        
        assert response.status_code == 400
        assert "Correction text required" in response.json()["detail"]


class TestCheckpointEndpoint:
    """Tests for the GET /api/query/{stream_id}/checkpoint endpoint."""
    
    @pytest.mark.asyncio
    async def test_checkpoint_unknown_id_returns_404(self, test_client):
        """Test that unknown stream_id returns 404."""
        async with test_client as client:
            response = await client.get(
                "/api/query/unknown-stream-id/checkpoint"
            )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_checkpoint_not_at_checkpoint_returns_400(self, test_client, mock_create_workflow):
        """Test that checkpoint data when not at checkpoint returns 400."""
        async with test_client as client:
            # Submit query
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            await asyncio.sleep(0.1)
            
            # Set state to researching (not at checkpoint)
            await event_queue_manager.update_state(
                stream_id,
                {"status": "researching"}
            )
            
            # Try to get checkpoint
            response = await client.get(f"/api/query/{stream_id}/checkpoint")
        
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_checkpoint_at_checkpoint_returns_200(self, test_client, mock_create_workflow):
        """Test that checkpoint data at checkpoint returns 200."""
        async with test_client as client:
            # Submit query
            submit_response = await client.post(
                "/api/query",
                json={"query": "Test query"}
            )
            stream_id = submit_response.json()["stream_id"]
            
            await asyncio.sleep(0.1)
            
            # Set state to awaiting feedback
            await event_queue_manager.update_state(
                stream_id,
                {
                    "status": PipelineStatus.AWAITING_FEEDBACK.value,
                    "claims": [],
                    "verification_results": [],
                    "source_scores": {},
                }
            )
            
            # Get checkpoint
            response = await client.get(f"/api/query/{stream_id}/checkpoint")
        
        assert response.status_code == 200
        data = response.json()
        assert data["stream_id"] == stream_id
        assert data["status"] == PipelineStatus.AWAITING_FEEDBACK.value
        assert "summary" in data
        assert "claims_summary" in data
        assert "available_actions" in data


class TestSSEEventModel:
    """Tests for the SSEEvent Pydantic model."""
    
    def test_sse_event_to_sse_format(self):
        """Test SSE event formatting."""
        event = SSEEvent(
            type=SSEEventType.AGENT_STARTED,
            data={"agent": "planner", "action": "start"},
            stream_id="test-stream",
        )
        
        formatted = event.to_sse_format()
        
        assert formatted.startswith("data: ")
        assert formatted.endswith("\n\n")
        
        # Parse the JSON payload
        json_str = formatted[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        payload = json.loads(json_str)
        
        assert payload["type"] == "agent_started"
        assert payload["data"]["agent"] == "planner"
        assert payload["stream_id"] == "test-stream"
    
    def test_all_event_types_are_valid(self):
        """Test that all SSEEventType values work."""
        for event_type in SSEEventType:
            event = SSEEvent(type=event_type, data={})
            formatted = event.to_sse_format()
            assert "data: " in formatted


class TestQueryRequestValidation:
    """Tests for QueryRequest Pydantic model validation."""
    
    def test_valid_query_request(self):
        """Test creating a valid QueryRequest."""
        request = QueryRequest(query="What is gRPC?")
        
        assert request.query == "What is gRPC?"
        assert request.enable_hitl is False
        assert request.enable_parallel is True
    
    def test_query_whitespace_is_stripped(self):
        """Test that query whitespace is stripped."""
        request = QueryRequest(query="  What is gRPC?  ")
        
        assert request.query == "What is gRPC?"
    
    def test_query_too_short_raises(self):
        """Test that too-short query raises validation error."""
        with pytest.raises(ValueError):
            QueryRequest(query="ab")
    
    def test_query_max_length(self):
        """Test that oversized query raises validation error."""
        with pytest.raises(ValueError):
            QueryRequest(query="x" * 2001)


class TestFeedbackRequestValidation:
    """Tests for FeedbackRequest Pydantic model validation."""
    
    def test_valid_approve_feedback(self):
        """Test creating a valid approve feedback."""
        request = FeedbackRequest(action="approve")
        
        assert request.action == "approve"
        assert request.topic is None
    
    def test_valid_dig_deeper_feedback(self):
        """Test creating a valid dig_deeper feedback."""
        request = FeedbackRequest(action="dig_deeper", topic="performance")
        
        assert request.action == "dig_deeper"
        assert request.topic == "performance"
    
    def test_invalid_action_raises(self):
        """Test that invalid action raises validation error."""
        with pytest.raises(ValueError):
            FeedbackRequest(action="invalid")
    
    def test_action_is_normalized(self):
        """Test that action is lowercased."""
        request = FeedbackRequest(action="APPROVE")
        
        assert request.action == "approve"


class TestEventQueueManager:
    """Tests for the EventQueueManager class."""
    
    @pytest.mark.asyncio
    async def test_create_stream(self):
        """Test creating a new event stream."""
        from api.streaming import EventQueueManager
        
        manager = EventQueueManager()
        queue = await manager.create_stream("test-stream")
        
        assert queue is not None
        assert manager.stream_exists("test-stream")
    
    @pytest.mark.asyncio
    async def test_push_and_get_event(self):
        """Test pushing and retrieving events."""
        from api.streaming import EventQueueManager
        
        manager = EventQueueManager()
        await manager.create_stream("test-stream")
        
        event = SSEEvent(type=SSEEventType.AGENT_STARTED, data={"test": True})
        success = await manager.push_event("test-stream", event)
        
        assert success is True
        
        queue = await manager.get_queue("test-stream")
        received = await queue.get()
        
        assert received.type == SSEEventType.AGENT_STARTED
        assert received.data["test"] is True
        assert received.stream_id == "test-stream"
    
    @pytest.mark.asyncio
    async def test_push_to_unknown_stream_fails(self):
        """Test that pushing to unknown stream returns False."""
        from api.streaming import EventQueueManager
        
        manager = EventQueueManager()
        event = SSEEvent(type=SSEEventType.ERROR, data={})
        
        success = await manager.push_event("unknown-stream", event)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_state_management(self):
        """Test state storage and retrieval."""
        from api.streaming import EventQueueManager
        
        manager = EventQueueManager()
        await manager.create_stream("test-stream")
        
        state = {"status": "completed", "report": "Test report"}
        await manager.update_state("test-stream", state)
        
        retrieved = await manager.get_state("test-stream")
        
        assert retrieved["status"] == "completed"
        assert retrieved["report"] == "Test report"
    
    @pytest.mark.asyncio
    async def test_cleanup_stream(self):
        """Test stream cleanup."""
        from api.streaming import EventQueueManager
        
        manager = EventQueueManager()
        await manager.create_stream("test-stream")
        assert manager.stream_exists("test-stream")
        
        await manager.cleanup_stream("test-stream")
        
        # Queue should be gone, but state is preserved
        queue = await manager.get_queue("test-stream")
        assert queue is None
    
    @pytest.mark.asyncio
    async def test_get_active_streams(self):
        """Test getting list of active streams."""
        from api.streaming import EventQueueManager
        
        manager = EventQueueManager()
        await manager.create_stream("stream-1")
        await manager.create_stream("stream-2")
        
        active = manager.get_active_streams()
        
        assert "stream-1" in active
        assert "stream-2" in active
        assert len(active) == 2
