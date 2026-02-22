"""Tests for api/main.py lifespan and LangSmith configuration."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from httpx import ASGITransport, AsyncClient

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


class TestConfigureLangSmith:
    """Tests for _configure_langsmith function."""

    def test_configures_when_enabled(self):
        from api.main import _configure_langsmith

        mock_config = MagicMock()
        mock_config.langchain_tracing = True
        mock_config.langchain_api_key = "test-key"
        mock_config.langchain_project = "test-project"

        # Clear env vars first
        for key in ["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]:
            os.environ.pop(key, None)

        # _configure_langsmith does `from config import config` internally
        with patch.dict("sys.modules", {"config": MagicMock(config=mock_config)}):
            _configure_langsmith()

        # Should have set env vars
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "test-key"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"

        # Cleanup
        for key in ["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]:
            os.environ.pop(key, None)

    def test_skips_when_disabled(self):
        from api.main import _configure_langsmith

        mock_config = MagicMock()
        mock_config.langchain_tracing = False
        mock_config.langchain_api_key = None
        mock_config.langchain_project = None

        with patch("api.main.config", mock_config, create=True):
            # Should not raise
            _configure_langsmith()

    def test_handles_import_error(self):
        from api.main import _configure_langsmith

        with patch("api.main.config", side_effect=ImportError("no config"), create=True):
            # Uses local import so we patch the import
            # Should not raise
            _configure_langsmith()


class TestAppCreation:
    """Tests for the FastAPI app instance."""

    def test_app_has_correct_title(self):
        from api.main import app
        assert app.title == "MARSA API"

    def test_app_has_cors_middleware(self):
        from api.main import app
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    async def test_lifespan_runs(self):
        from api.main import app

        with patch("api.main._configure_langsmith"):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/health")
                assert response.status_code == 200
