"""Tests for LLM Provider."""

import pytest
from unittest.mock import patch, MagicMock


class TestLLMProvider:
    """Test cases for LLM Provider."""

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test_api_key"})
    def test_get_llm_provider_with_api_key(self):
        """Test that get_llm_provider returns ChatGoogleGenerativeAI when API key is set."""
        from app.agents.provider import get_llm_provider

        provider = get_llm_provider()

        assert provider is not None
        # Check that the provider is a ChatGoogleGenerativeAI instance
        assert provider.__class__.__name__ == "ChatGoogleGenerativeAI"

    @patch.dict("os.environ", {}, clear=True)
    def test_get_llm_provider_without_api_key_raises_error(self):
        """Test that get_llm_provider raises ValueError when API key is not set."""
        from app.agents.provider import get_llm_provider

        with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable is not set"):
            get_llm_provider()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test_api_key"})
    def test_get_llm_provider_with_custom_model(self):
        """Test that get_llm_provider uses custom model name when provided."""
        from app.agents.provider import get_llm_provider

        provider = get_llm_provider(model="gemini-2.0-flash")

        assert provider is not None
        assert provider.model == "gemini-2.0-flash"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test_api_key"})
    def test_get_llm_provider_default_model(self):
        """Test that get_llm_provider uses default model when not specified."""
        from app.agents.provider import get_llm_provider

        provider = get_llm_provider()

        assert provider is not None
        assert provider.model == "gemini-2.0-flash-exp"
