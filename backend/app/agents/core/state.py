"""Agent State Management for LangGraph."""

from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State for the Trading Research Agent.

    This state is passed between nodes in the LangGraph and maintains
    the conversation history, current execution step, tool results, and
    completion status.
    """

    messages: Annotated[List[Dict[str, Any]], add_messages]
    """Conversation history between user and agent."""

    current_step: str
    """Current step in the reasoning pipeline (planner, executor, analyzer, responder)."""

    tool_results: List[Dict[str, Any]]
    """Results from tool executions (backtest, optuna, data queries, etc.)."""

    is_complete: bool
    """Whether the agent has completed the task and is ready to respond."""
