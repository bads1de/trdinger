"""Tests for LLM Provider."""

import pytest
from unittest.mock import patch, MagicMock, Mock


# Mock langchain before importing provider
@pytest.fixture(autouse=True)
def mock_langchain():
    """Mock langchain_google_genai to avoid import dependency."""
    with patch.dict("sys.modules", {"langchain_google_genai": MagicMock()}):
        with patch("app.agents.provider.ChatGoogleGenerativeAI") as mock_chat_google:
            mock_instance = Mock()
            mock_chat_google.return_value = mock_instance
            yield mock_chat_google


class TestLLMProvider:
    """Test cases for LLM Provider."""

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test_api_key"})
    def test_get_llm_provider_with_api_key(self, mock_langchain):
        """Test that get_llm_provider returns ChatGoogleGenerativeAI when API key is set."""
        from app.agents.provider import get_llm_provider

        provider = get_llm_provider()

        assert provider is not None
        mock_langchain.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_get_llm_provider_without_api_key_raises_error(self):
        """Test that get_llm_provider raises ValueError when API key is not set."""
        # Import with mocked langchain
        with patch.dict("sys.modules", {"langchain_google_genai": MagicMock()}):
            from app.agents.provider import get_llm_provider

            with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable is not set"):
                get_llm_provider()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test_api_key"})
    def test_get_llm_provider_with_custom_model(self, mock_langchain):
        """Test that get_llm_provider uses custom model name when provided."""
        from app.agents.provider import get_llm_provider

        mock_instance = Mock()
        mock_instance.model = "gemini-2.0-flash"
        mock_langchain.return_value = mock_instance

        provider = get_llm_provider(model="gemini-2.0-flash")

        assert provider is not None
        assert provider.model == "gemini-2.0-flash"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test_api_key"})
    def test_get_llm_provider_default_model(self, mock_langchain):
        """Test that get_llm_provider uses default model when not specified."""
        from app.agents.provider import get_llm_provider

        mock_instance = Mock()
        mock_instance.model = "gemini-2.0-flash-exp"
        mock_langchain.return_value = mock_instance

        provider = get_llm_provider()

        assert provider is not None
        assert provider.model == "gemini-2.0-flash-exp"
