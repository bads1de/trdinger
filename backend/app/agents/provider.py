"""LLM Provider Configuration for AI Agent (OpenRouter)."""

import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# デフォルトで使用する OpenRouter モデル（:free は無料枠）
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# OpenRouter の OpenAI 互換エンドポイント
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm_provider(model: str = DEFAULT_OPENROUTER_MODEL) -> ChatOpenAI:
    """Get configured LLM provider for the agent (OpenRouter).

    Args:
        model: The model name to use on OpenRouter.
            Defaults to "nvidia/nemotron-3-super-120b-a12b:free".

    Returns:
        ChatOpenAI: Configured OpenAI-compatible LLM instance.

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )
