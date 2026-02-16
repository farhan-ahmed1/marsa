"""Configuration module for MARSA backend.

Loads and validates environment variables required for the application.
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ConfigError(Exception):
    """Raised when a required configuration value is missing."""
    pass


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self):
        """Initialize configuration and validate required values."""
        # API Keys
        self.anthropic_api_key = self._get_required("ANTHROPIC_API_KEY")
        self.openai_api_key = self._get_required("OPENAI_API_KEY")
        self.tavily_api_key = self._get_required("TAVILY_API_KEY")

        # Optional: LangSmith tracing
        self.langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT", "marsa")

        # Application settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

    def _get_required(self, key: str) -> str:
        """Get a required environment variable or raise an error.

        Args:
            key: The environment variable name.

        Returns:
            The environment variable value.

        Raises:
            ConfigError: If the environment variable is not set or is empty.
        """
        value = os.getenv(key)
        if not value:
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
