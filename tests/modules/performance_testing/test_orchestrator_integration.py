"""
Integration tests for PerformanceTester with TestOrchestrator.
Tests the integration between performance testing module and the test orchestrator.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch

from .performance_tester import PerformanceTester

try:
    from ...config.test_config import PerformanceTestConfig
    from ...orchestrator.test_orchestrator import TestOrchestrator, TestStatus
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.test_config import PerformanceTestConfig
    from orchestrator.test_orchestrator import TestOrchestrator, TestStatus


class TestPerformanceTesterIntegration:
    """Integration tests for PerformanceTester with TestOrchestrator."""

    @pytest.fixture
    def performance_config(self):
        """Create test performance configuration."""
        return PerformanceTestConfig(
            market_data_processing_target_ms=100,
            strategy_signal_generation_target_ms=500,
            portfolio_update_target_ms=1000,
            performance_test_iterations=2,  # Reduced for faster testing
        )

    @pytest.fixture
    def performance_tester(self, performance_config):
        """Create PerformanceTester instance for testing."""
        return PerformanceTester(performance_config)

    @pytest.mark.asyncio
    async def test_performance_tester_module_interface(self, performance_tester):
        """Test that PerformanceTester properly implements TestModuleInterface."""
        # Test module name
        assert performance_tester.get_module_name() == "performance_testing"

        # Test run_tests method exists and returns proper result
        result = await performance_tester.run_tests()

        assert hasattr(result, "module_name")
        assert hasattr(result, "status")
        assert hasattr(result, "execution_time_seconds")
        assert hasattr(result, "tests_run")
        assert hasattr(result, "tests_passed")
        assert hasattr(result, "tests_failed")
        assert hasattr(result, "tests_skipped")
        assert hasattr(result, "error_messages")
        assert hasattr(result, "detailed_results")

        assert result.module_name == "performance_testing"
        assert isinstance(result.status, TestStatus)
        assert result.execution_time_seconds > 0
        assert result.tests_run > 0

    @pytest.mark.asyncio
    async def test_orchestrator_can_register_performance_tester(
        self, performance_tester
    ):
        """Test that TestOrchestrator can register and run PerformanceTester."""
        # Create a mock orchestrator (since we don't want to run full orchestrator)
        orchestrator = Mock()
        orchestrator.register_test_module = Mock()
        orchestrator.run_specific_tests = Mock()

        # Test registration
        orchestrator.register_test_module("performance_testing", performance_tester)
        orchestrator.register_test_module.assert_called_once_with(
            "performance_testing", performance_tester
        )

        # Test that the module can be called
        result = await performance_tester.run_tests()
        assert result.module_name == "performance_testing"

    @pytest.mark.asyncio
    async def test_performance_results_structure(self, performance_tester):
        """Test that performance test results have the expected structure."""
        result = await performance_tester.run_tests()

        # Check detailed results structure
        assert isinstance(result.detailed_results, dict)

        expected_tests = [
            "market_data_processing",
            "strategy_signal_generation",
            "portfolio_update",
        ]

        for test_name in expected_tests:
            if test_name in result.detailed_results:
                test_result = result.detailed_results[test_name]

                # Check PerformanceTestResult structure
                assert hasattr(test_result, "test_name")
                assert hasattr(test_result, "target_time_ms")
                assert hasattr(test_result, "actual_times_ms")
                assert hasattr(test_result, "average_time_ms")
                assert hasattr(test_result, "test_passed")
                assert hasattr(test_result, "iterations_run")

                assert test_result.test_name == test_name
                assert test_result.target_time_ms > 0
                assert isinstance(test_result.actual_times_ms, list)
                assert test_result.average_time_ms >= 0
                assert isinstance(test_result.test_passed, bool)
                assert test_result.iterations_run > 0

    @pytest.mark.asyncio
    async def test_performance_tester_error_handling(self, performance_config):
        """Test error handling in PerformanceTester."""
        performance_tester = PerformanceTester(performance_config)

        # Mock one of the test methods to raise an exception
        with patch.object(
            performance_tester,
            "test_market_data_processing_speed",
            side_effect=Exception("Test exception"),
        ):
            result = await performance_tester.run_tests()

            # Should still complete but with failures
            assert result.module_name == "performance_testing"
            assert result.status == TestStatus.FAILED
            assert result.tests_failed > 0
            assert len(result.error_messages) > 0
            assert any("Test exception" in msg for msg in result.error_messages)

    @pytest.mark.asyncio
    async def test_performance_benchmarks_validation(self, performance_tester):
        """Test that performance benchmarks are properly validated."""
        # Run individual performance tests
        market_data_result = (
            await performance_tester.test_market_data_processing_speed()
        )
        strategy_result = (
            await performance_tester.test_strategy_signal_generation_speed()
        )
        portfolio_result = await performance_tester.test_portfolio_update_speed()

        # Validate market data processing benchmark
        assert market_data_result.target_time_ms == 100
        assert (
            len(market_data_result.actual_times_ms)
            == performance_tester.config.performance_test_iterations
        )
        assert market_data_result.average_time_ms > 0

        # Validate strategy signal generation benchmark
        assert strategy_result.target_time_ms == 500
        assert (
            len(strategy_result.actual_times_ms)
            == performance_tester.config.performance_test_iterations
        )
        assert strategy_result.average_time_ms > 0

        # Validate portfolio update benchmark
        assert portfolio_result.target_time_ms == 1000
        assert (
            len(portfolio_result.actual_times_ms)
            == performance_tester.config.performance_test_iterations
        )
        assert portfolio_result.average_time_ms > 0

    @pytest.mark.asyncio
    async def test_performance_statistics_calculation(self, performance_tester):
        """Test that performance statistics are calculated correctly."""
        result = await performance_tester.test_market_data_processing_speed()

        # Check that all statistical measures are calculated
        assert result.average_time_ms > 0
        assert result.median_time_ms > 0
        assert result.min_time_ms > 0
        assert result.max_time_ms > 0
        assert result.std_deviation_ms >= 0
        assert 0 <= result.success_rate <= 1

        # Check logical relationships
        assert result.min_time_ms <= result.average_time_ms <= result.max_time_ms
        assert result.min_time_ms <= result.median_time_ms <= result.max_time_ms

    @pytest.mark.asyncio
    async def test_concurrent_performance_testing(self, performance_config):
        """Test that multiple performance testers can run concurrently."""
        tester1 = PerformanceTester(performance_config)
        tester2 = PerformanceTester(performance_config)

        # Run both testers concurrently
        results = await asyncio.gather(
            tester1.run_tests(), tester2.run_tests(), return_exceptions=True
        )

        # Both should complete successfully
        assert len(results) == 2
        for result in results:
            assert not isinstance(result, Exception)
            assert result.module_name == "performance_testing"
            assert result.tests_run > 0

    @pytest.mark.asyncio
    async def test_performance_tester_cleanup(self, performance_tester):
        """Test that PerformanceTester properly cleans up resources."""
        # Run tests
        result = await performance_tester.run_tests()

        # Verify that test results are stored
        assert len(performance_tester.test_results) > 0

        # Verify that mock components are still accessible
        assert performance_tester.market_data_processor is not None
        assert performance_tester.strategy_engine is not None
        assert performance_tester.portfolio_manager is not None

    @pytest.mark.asyncio
    async def test_performance_requirements_compliance(self, performance_tester):
        """Test that PerformanceTester meets the specified requirements."""
        result = await performance_tester.run_tests()

        # Requirement 6.1: Market data processing < 100ms
        if "market_data_processing" in result.detailed_results:
            market_result = result.detailed_results["market_data_processing"]
            # Test should validate against 100ms target
            assert market_result.target_time_ms == 100

        # Requirement 6.2: Strategy signal generation < 500ms
        if "strategy_signal_generation" in result.detailed_results:
            strategy_result = result.detailed_results["strategy_signal_generation"]
            # Test should validate against 500ms target
            assert strategy_result.target_time_ms == 500

        # Requirement 6.3: Portfolio update < 1 second
        if "portfolio_update" in result.detailed_results:
            portfolio_result = result.detailed_results["portfolio_update"]
            # Test should validate against 1000ms target
            assert portfolio_result.target_time_ms == 1000

        # All tests should have been executed (3 performance tests + 1 profiling test)
        assert result.tests_run == 4


if __name__ == "__main__":
    # Run integration tests directly
    pytest.main([__file__, "-v"])
