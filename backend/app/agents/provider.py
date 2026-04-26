"""LLM Provider Configuration for AI Agent."""

import os
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm_provider(model: str = "gemma-4-31b-it") -> ChatGoogleGenerativeAI:
    """Get configured LLM provider for the agent.

    Args:
        model: The model name to use. Defaults to "gemma-4-31b-it".

    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance.

    Raises:
        ValueError: If GOOGLE_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=0,
    )
