"""Tests for Agent State."""

import pytest


class TestAgentState:
    """Test cases for AgentState."""

    def test_agent_state_initialization(self):
        """Test that AgentState can be initialized with default values."""
        from app.agents.core.state import AgentState

        state = AgentState(
            messages=[],
            current_step="planner",
            tool_results=[],
            is_complete=False,
        )

        assert state["messages"] == []
        assert state["current_step"] == "planner"
        assert state["tool_results"] == []
        assert state["is_complete"] is False

    def test_agent_state_with_values(self):
        """Test that AgentState can be initialized with custom values."""
        from app.agents.core.state import AgentState

        state = AgentState(
            messages=[{"role": "user", "content": "test"}],
            current_step="executor",
            tool_results=[{"tool": "backtest", "result": "success"}],
            is_complete=True,
        )

        assert state["messages"] == [{"role": "user", "content": "test"}]
        assert state["current_step"] == "executor"
        assert state["tool_results"] == [{"tool": "backtest", "result": "success"}]
        assert state["is_complete"] is True
