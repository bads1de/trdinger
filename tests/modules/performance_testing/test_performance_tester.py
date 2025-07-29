"""
Unit tests for PerformanceTester module.
Tests the performance testing functionality and validates performance benchmarks.
"""

import asyncio
import pytest
import time
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

from .performance_tester import (
    PerformanceTester,
    PerformanceTestResult,
    MockMarketDataProcessor,
    MockStrategyEngine,
    MockPortfolioManager,
)

try:
    from ...config.test_config import PerformanceTestConfig
    from ...orchestrator.test_orchestrator import TestStatus
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.test_config import PerformanceTestConfig
    from orchestrator.test_orchestrator import TestStatus


class TestPerformanceTester:
    """Test cases for PerformanceTester class."""

    @pytest.fixture
    def performance_config(self):
        """Create test performance configuration."""
        return PerformanceTestConfig(
            market_data_processing_target_ms=100,
            strategy_signal_generation_target_ms=500,
            portfolio_update_target_ms=1000,
            performance_test_iterations=3,  # Reduced for faster testing
        )

    @pytest.fixture
    def performance_tester(self, performance_config):
        """Create PerformanceTester instance for testing."""
        return PerformanceTester(performance_config)

    def test_performance_tester_initialization(self, performance_tester):
        """Test PerformanceTester initialization."""
        assert performance_tester.get_module_name() == "performance_testing"
        assert performance_tester.config.market_data_processing_target_ms == 100
        assert performance_tester.config.strategy_signal_generation_target_ms == 500
        assert performance_tester.config.portfolio_update_target_ms == 1000
        assert performance_tester.config.performance_test_iterations == 3

    @pytest.mark.asyncio
    async def test_measure_execution_time_success(self, performance_tester):
        """Test execution time measurement for successful function."""

        async def test_function():
            await asyncio.sleep(0.01)  # 10ms delay
            return "success"

        execution_time_ms, result, error_message = (
            await performance_tester._measure_execution_time(test_function)
        )

        assert result == "success"
        assert error_message is None
        assert execution_time_ms >= 10  # Should be at least 10ms
        assert execution_time_ms < 50  # Should be reasonable

    @pytest.mark.asyncio
    async def test_measure_execution_time_failure(self, performance_tester):
        """Test execution time measurement for failing function."""

        async def failing_function():
            raise ValueError("Test error")

        execution_time_ms, result, error_message = (
            await performance_tester._measure_execution_time(failing_function)
        )

        assert result is None
        assert error_message is not None
        assert "Test error" in error_message
        assert execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_market_data_processing_speed(self, performance_tester):
        """Test market data processing speed test."""
        result = await performance_tester.test_market_data_processing_speed()

        assert isinstance(result, PerformanceTestResult)
        assert result.test_name == "market_data_processing"
        assert result.target_time_ms == 100
        assert result.iterations_run == 3
        assert len(result.actual_times_ms) == 3
        assert result.average_time_ms > 0
        assert result.success_rate >= 0
        assert isinstance(result.test_passed, bool)

    @pytest.mark.asyncio
    async def test_strategy_signal_generation_speed(self, performance_tester):
        """Test strategy signal generation speed test."""
        result = await performance_tester.test_strategy_signal_generation_speed()

        assert isinstance(result, PerformanceTestResult)
        assert result.test_name == "strategy_signal_generation"
        assert result.target_time_ms == 500
        assert result.iterations_run == 3
        assert len(result.actual_times_ms) == 3
        assert result.average_time_ms > 0
        assert result.success_rate >= 0
        assert isinstance(result.test_passed, bool)

    @pytest.mark.asyncio
    async def test_portfolio_update_speed(self, performance_tester):
        """Test portfolio update speed test."""
        result = await performance_tester.test_portfolio_update_speed()

        assert isinstance(result, PerformanceTestResult)
        assert result.test_name == "portfolio_update"
        assert result.target_time_ms == 1000
        assert result.iterations_run == 3
        assert len(result.actual_times_ms) == 3
        assert result.average_time_ms > 0
        assert result.success_rate >= 0
        assert isinstance(result.test_passed, bool)

    @pytest.mark.asyncio
    async def test_run_tests_success(self, performance_tester):
        """Test complete test suite execution."""
        result = await performance_tester.run_tests()

        assert result.module_name == "performance_testing"
        assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
        assert result.tests_run == 4
        assert result.execution_time_seconds > 0
        assert isinstance(result.detailed_results, dict)
        assert "market_data_processing" in result.detailed_results
        assert "strategy_signal_generation" in result.detailed_results
        assert "portfolio_update" in result.detailed_results

    @pytest.mark.asyncio
    async def test_run_tests_with_failures(self, performance_tester):
        """Test test suite execution with simulated failures."""
        # Mock one of the test methods to fail
        with patch.object(
            performance_tester,
            "test_market_data_processing_speed",
            side_effect=Exception("Simulated failure"),
        ):
            result = await performance_tester.run_tests()

            assert result.module_name == "performance_testing"
            assert result.status == TestStatus.FAILED
            assert result.tests_run == 4
            assert result.tests_failed >= 1
            assert len(result.error_messages) > 0
            assert any("Simulated failure" in msg for msg in result.error_messages)


class TestMockMarketDataProcessor:
    """Test cases for MockMarketDataProcessor."""

    @pytest.fixture
    def processor(self):
        """Create MockMarketDataProcessor instance."""
        return MockMarketDataProcessor()

    @pytest.mark.asyncio
    async def test_process_market_data(self, processor):
        """Test market data processing."""
        result = await processor.process_market_data(data_size=100)

        assert isinstance(result, dict)
        assert "processed_count" in result
        assert "data" in result
        assert "total_volume" in result
        assert result["processed_count"] == 100
        assert len(result["data"]) <= 10  # Sample data
        assert isinstance(result["total_volume"], Decimal)

    @pytest.mark.asyncio
    async def test_process_market_data_empty(self, processor):
        """Test market data processing with zero data size."""
        result = await processor.process_market_data(data_size=0)

        assert isinstance(result, dict)
        assert result["processed_count"] == 0
        assert len(result["data"]) == 0
        assert result["total_volume"] == Decimal("0")


class TestMockStrategyEngine:
    """Test cases for MockStrategyEngine."""

    @pytest.fixture
    def strategy_engine(self):
        """Create MockStrategyEngine instance."""
        return MockStrategyEngine()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return {
            "processed_count": 10,
            "data": [
                {
                    "symbol": "BTC/USDT",
                    "processed_price": Decimal("50000.00000000"),
                    "volume_weighted_price": Decimal("5000000.00000000"),
                    "timestamp": time.time(),
                }
                for i in range(10)
            ],
            "total_volume": Decimal("1000.00000000"),
        }

    @pytest.mark.asyncio
    async def test_generate_trading_signal_with_data(
        self, strategy_engine, sample_market_data
    ):
        """Test trading signal generation with market data."""
        result = await strategy_engine.generate_trading_signal(sample_market_data)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "confidence" in result
        assert "price" in result
        assert "timestamp" in result
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= result["confidence"] <= 1
        assert isinstance(result["price"], Decimal)

    @pytest.mark.asyncio
    async def test_generate_trading_signal_no_data(self, strategy_engine):
        """Test trading signal generation with no market data."""
        result = await strategy_engine.generate_trading_signal({"data": []})

        assert isinstance(result, dict)
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0
        assert "reason" in result


class TestMockPortfolioManager:
    """Test cases for MockPortfolioManager."""

    @pytest.fixture
    def portfolio_manager(self):
        """Create MockPortfolioManager instance."""
        return MockPortfolioManager()

    @pytest.fixture
    def sample_signal_data(self):
        """Create sample signal data for testing."""
        return {
            "signal": "BUY",
            "confidence": 0.8,
            "price": Decimal("50000.00000000"),
            "timestamp": time.time(),
        }

    @pytest.mark.asyncio
    async def test_update_portfolio_buy_signal(
        self, portfolio_manager, sample_signal_data
    ):
        """Test portfolio update with buy signal."""
        result = await portfolio_manager.update_portfolio(sample_signal_data)

        assert isinstance(result, dict)
        assert "total_value" in result
        assert "cash_balance" in result
        assert "positions" in result
        assert "last_update" in result
        assert isinstance(result["total_value"], Decimal)
        assert isinstance(result["cash_balance"], Decimal)

    @pytest.mark.asyncio
    async def test_update_portfolio_sell_signal(self, portfolio_manager):
        """Test portfolio update with sell signal."""
        # First buy some BTC
        buy_signal = {
            "signal": "BUY",
            "confidence": 0.8,
            "price": Decimal("50000.00000000"),
            "timestamp": time.time(),
        }
        await portfolio_manager.update_portfolio(buy_signal)

        # Then sell
        sell_signal = {
            "signal": "SELL",
            "confidence": 0.8,
            "price": Decimal("51000.00000000"),
            "timestamp": time.time(),
        }
        result = await portfolio_manager.update_portfolio(sell_signal)

        assert isinstance(result, dict)
        assert "total_value" in result
        assert "cash_balance" in result
        assert "positions" in result

    @pytest.mark.asyncio
    async def test_update_portfolio_hold_signal(self, portfolio_manager):
        """Test portfolio update with hold signal."""
        hold_signal = {
            "signal": "HOLD",
            "confidence": 0.5,
            "price": Decimal("50000.00000000"),
            "timestamp": time.time(),
        }
        result = await portfolio_manager.update_portfolio(hold_signal)

        assert isinstance(result, dict)
        assert result["cash_balance"] == Decimal("10000.00000000")  # No change
        assert result["positions"] == {}  # No positions


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
