"""Configuration module for MARSA backend.

Loads and validates environment variables required for the application.
"""

import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ConfigError(Exception):
    """Raised when a required configuration value is missing."""
    pass


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self, validate: bool = True):
        """Initialize configuration and validate required values.
        
        Args:
            validate: If False, allows missing env vars during testing.
                     Automatically set to False when pytest is detected.
        """
        # Auto-detect pytest environment
        if not validate or "pytest" in sys.modules:
            validate = False
        
        # API Keys
        self.anthropic_api_key = self._get_required("ANTHROPIC_API_KEY", validate)
        self.openai_api_key = self._get_required("OPENAI_API_KEY", validate)
        self.tavily_api_key = self._get_required("TAVILY_API_KEY", validate)

        # Optional: LangSmith tracing
        self.langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT", "marsa")

        # Application settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # API key auth (empty = disabled)
        self.marsa_api_key = os.getenv("MARSA_API_KEY", "")

        # CORS extra origins (comma-separated)
        self.cors_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")

        # Cache settings
        self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    def _get_required(self, key: str, validate: bool = True) -> str:
        """Get a required environment variable or raise an error.

        Args:
            key: The environment variable name.
            validate: If False, returns empty string instead of raising error.

        Returns:
            The environment variable value.

        Raises:
            ConfigError: If the environment variable is not set or is empty (when validate=True).
        """
        value = os.getenv(key, "")
        if not value and validate:
            raise ConfigError(
                f"Required environment variable '{key}' is not set. "
                f"Please set it in your .env file or environment."
            )
        return value

    def _get_optional(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an optional environment variable.

        Args:
            key: The environment variable name.
            default: The default value if not set.

        Returns:
            The environment variable value or default.
        """
        return os.getenv(key, default)


# Global config instance
config = Config()
