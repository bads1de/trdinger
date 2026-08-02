"""Tests for LLM Provider (OpenRouter)."""

from unittest.mock import MagicMock, Mock, patch

import pytest


# Mock langchain before importing provider
@pytest.fixture(autouse=True)
def mock_langchain():
    """Mock langchain_openai to avoid import dependency."""
    with patch.dict("sys.modules", {"langchain_openai": MagicMock()}):
        with patch("app.agents.provider.ChatOpenAI") as mock_chat_openai:
            mock_instance = Mock()
            mock_chat_openai.return_value = mock_instance
            yield mock_chat_openai


class TestLLMProvider:
    """Test cases for LLM Provider."""

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_api_key"})
    def test_get_llm_provider_with_api_key(self, mock_langchain):
        """Test that get_llm_provider returns ChatOpenAI when API key is set."""
        from app.agents.provider import get_llm_provider

        provider = get_llm_provider()

        assert provider is not None
        mock_langchain.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_get_llm_provider_without_api_key_raises_error(self):
        """Test that get_llm_provider raises ValueError when API key is not set."""
        # Import with mocked langchain
        with patch.dict("sys.modules", {"langchain_openai": MagicMock()}):
            from app.agents.provider import get_llm_provider

            with pytest.raises(
                ValueError, match="OPENROUTER_API_KEY environment variable is not set"
            ):
                get_llm_provider()

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_api_key"})
    def test_get_llm_provider_with_custom_model(self, mock_langchain):
        """Test that get_llm_provider uses custom model name when provided."""
        from app.agents.provider import get_llm_provider

        mock_instance = Mock()
        mock_instance.model = "nvidia/nemotron-3-super-120b-a12b:free"
        mock_langchain.return_value = mock_instance

        provider = get_llm_provider(model="nvidia/nemotron-3-super-120b-a12b:free")

        assert provider is not None
        assert provider.model == "nvidia/nemotron-3-super-120b-a12b:free"

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_api_key"})
    def test_get_llm_provider_default_model(self, mock_langchain):
        """Test that get_llm_provider uses default model when not specified."""
        from app.agents.provider import get_llm_provider

        mock_instance = Mock()
        mock_instance.model = "nvidia/nemotron-3-super-120b-a12b:free"
        mock_langchain.return_value = mock_instance

        provider = get_llm_provider()

        assert provider is not None
        assert provider.model == "nvidia/nemotron-3-super-120b-a12b:free"

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_api_key"})
    def test_get_llm_provider_uses_openrouter_base_url(self, mock_langchain):
        """Test that get_llm_provider uses OpenRouter base URL."""
        from app.agents.provider import get_llm_provider

        get_llm_provider()

        mock_langchain.assert_called_once()
        kwargs = mock_langchain.call_args.kwargs
        assert kwargs.get("base_url") == "https://openrouter.ai/api/v1"
