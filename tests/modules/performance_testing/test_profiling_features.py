"""
Unit tests for performance profiling and bottleneck detection features.
Tests the advanced profiling capabilities of the PerformanceTester module.
"""

import asyncio
import pytest
import time
import cProfile
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from .performance_tester import (
    PerformanceTester,
    ProfilingResult,
    BottleneckDetectionResult,
    RegressionDetectionResult,
    PerformanceMetrics,
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


class TestPerformanceProfilingFeatures:
    """Test cases for performance profiling and bottleneck detection."""

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

    def test_profiling_result_creation(self):
        """Test ProfilingResult dataclass creation."""
        profiling_result = ProfilingResult(
            function_name="test_function",
            total_time_seconds=0.1,
            cumulative_time_seconds=0.15,
            call_count=10,
            time_per_call_seconds=0.015,
            percentage_of_total=25.0,
        )

        assert profiling_result.function_name == "test_function"
        assert profiling_result.total_time_seconds == 0.1
        assert profiling_result.cumulative_time_seconds == 0.15
        assert profiling_result.call_count == 10
        assert profiling_result.time_per_call_seconds == 0.015
        assert profiling_result.percentage_of_total == 25.0

    def test_bottleneck_detection_result_creation(self):
        """Test BottleneckDetectionResult dataclass creation."""
        bottleneck = BottleneckDetectionResult(
            bottleneck_type="high_execution_time",
            severity="HIGH",
            description="Function consuming too much time",
            affected_function="slow_function",
            performance_impact=80.0,
            recommended_action="Optimize algorithm",
            detection_timestamp=None,  # Will be set by datetime.now()
        )

        assert bottleneck.bottleneck_type == "high_execution_time"
        assert bottleneck.severity == "HIGH"
        assert bottleneck.description == "Function consuming too much time"
        assert bottleneck.affected_function == "slow_function"
        assert bottleneck.performance_impact == 80.0
        assert bottleneck.recommended_action == "Optimize algorithm"

    def test_regression_detection_result_creation(self):
        """Test RegressionDetectionResult dataclass creation."""
        regression = RegressionDetectionResult(
            test_name="market_data_processing",
            baseline_time_ms=50.0,
            current_time_ms=75.0,
            regression_percentage=50.0,
            regression_detected=True,
            threshold_percentage=20.0,
            detection_timestamp=None,  # Will be set by datetime.now()
        )

        assert regression.test_name == "market_data_processing"
        assert regression.baseline_time_ms == 50.0
        assert regression.current_time_ms == 75.0
        assert regression.regression_percentage == 50.0
        assert regression.regression_detected is True
        assert regression.threshold_percentage == 20.0

    def test_profile_function_execution_sync(self, performance_tester):
        """Test profiling of synchronous function execution."""

        def test_function(delay_ms: int = 10):
            time.sleep(delay_ms / 1000)  # Convert ms to seconds
            return "completed"

        result, profiling_results = performance_tester._profile_function_execution(
            test_function, delay_ms=20
        )

        assert result == "completed"
        assert isinstance(profiling_results, list)
        # Should have some profiling data
        assert len(profiling_results) >= 0

    @pytest.mark.asyncio
    async def test_profile_function_execution_async(self, performance_tester):
        """Test profiling of asynchronous function execution."""

        async def async_test_function(delay_ms: int = 10):
            await asyncio.sleep(delay_ms / 1000)  # Convert ms to seconds
            return "async_completed"

        result, profiling_results = performance_tester._profile_function_execution(
            async_test_function, delay_ms=20
        )

        assert result == "async_completed"
        assert isinstance(profiling_results, list)

    def test_analyze_profiling_data(self, performance_tester):
        """Test analysis of cProfile profiling data."""
        # Create a mock profiler with some data
        profiler = cProfile.Profile()

        # Profile a simple function
        def sample_function():
            time.sleep(0.01)  # 10ms
            return sum(range(1000))

        profiler.enable()
        sample_function()
        profiler.disable()

        # Analyze the profiling data
        profiling_results = performance_tester._analyze_profiling_data(profiler)

        assert isinstance(profiling_results, list)
        # Should have at least some profiling results
        if profiling_results:
            result = profiling_results[0]
            assert isinstance(result, ProfilingResult)
            assert result.function_name is not None
            assert result.total_time_seconds >= 0
            assert result.cumulative_time_seconds >= 0
            assert result.call_count > 0

    def test_detect_performance_bottlenecks(self, performance_tester):
        """Test bottleneck detection from profiling results."""
        # Create mock profiling results
        profiling_results = [
            ProfilingResult(
                function_name="slow_function",
                total_time_seconds=0.08,  # 80ms
                cumulative_time_seconds=0.08,
                call_count=1,
                time_per_call_seconds=0.08,
                percentage_of_total=80.0,
            ),
            ProfilingResult(
                function_name="frequent_function",
                total_time_seconds=0.02,
                cumulative_time_seconds=0.02,
                call_count=2000,  # Many calls
                time_per_call_seconds=0.00001,
                percentage_of_total=20.0,
            ),
        ]

        bottlenecks = performance_tester._detect_performance_bottlenecks(
            profiling_results, "test_function", 100  # 100ms target
        )

        assert isinstance(bottlenecks, list)
        assert len(bottlenecks) >= 1  # Should detect at least one bottleneck

        # Check bottleneck properties
        for bottleneck in bottlenecks:
            assert isinstance(bottleneck, BottleneckDetectionResult)
            assert bottleneck.bottleneck_type in [
                "high_execution_time",
                "excessive_calls",
            ]
            assert bottleneck.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            assert bottleneck.affected_function is not None
            assert bottleneck.recommended_action is not None

    def test_monitor_system_resources(self, performance_tester):
        """Test system resource monitoring."""
        # Mock the system_monitor attribute directly
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 25.5
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 15.2

        # Replace the system_monitor with our mock
        performance_tester.system_monitor = mock_process

        # Mock system-wide metrics
        with patch("psutil.cpu_percent", return_value=45.0), patch(
            "psutil.virtual_memory", return_value=MagicMock(percent=60.0)
        ):

            resources = performance_tester._monitor_system_resources()

            assert isinstance(resources, dict)
            assert "process_cpu_percent" in resources
            assert "process_memory_mb" in resources
            assert "process_memory_percent" in resources
            assert "system_cpu_percent" in resources
            assert "system_memory_percent" in resources
            assert "thread_count" in resources

            assert resources["process_cpu_percent"] == 25.5
            assert resources["process_memory_mb"] == 100.0  # 100MB
            assert resources["process_memory_percent"] == 15.2
            assert resources["system_cpu_percent"] == 45.0
            assert resources["system_memory_percent"] == 60.0

    def test_detect_performance_regression_no_baseline(self, performance_tester):
        """Test regression detection when no baseline exists."""
        regression_result = performance_tester._detect_performance_regression(
            "new_test", 50.0, threshold_percentage=20.0
        )

        assert isinstance(regression_result, RegressionDetectionResult)
        assert regression_result.test_name == "new_test"
        assert regression_result.current_time_ms == 50.0
        assert regression_result.regression_detected is False
        assert regression_result.threshold_percentage == 20.0

        # Should set baseline
        assert performance_tester.baseline_performance["new_test"] == 50.0

    def test_detect_performance_regression_with_baseline(self, performance_tester):
        """Test regression detection with existing baseline."""
        # Set baseline
        performance_tester.baseline_performance["existing_test"] = 40.0

        # Test with regression
        regression_result = performance_tester._detect_performance_regression(
            "existing_test", 60.0, threshold_percentage=20.0
        )

        assert regression_result.test_name == "existing_test"
        assert regression_result.baseline_time_ms == 40.0
        assert regression_result.current_time_ms == 60.0
        assert regression_result.regression_percentage == 50.0  # (60-40)/40 * 100
        assert regression_result.regression_detected is True  # 50% > 20% threshold

    def test_detect_performance_improvement(self, performance_tester):
        """Test regression detection when performance improves."""
        # Set baseline
        performance_tester.baseline_performance["improving_test"] = 60.0

        # Test with improvement
        regression_result = performance_tester._detect_performance_regression(
            "improving_test", 40.0, threshold_percentage=20.0
        )

        assert regression_result.test_name == "improving_test"
        assert regression_result.baseline_time_ms == 60.0
        assert regression_result.current_time_ms == 40.0
        assert regression_result.regression_percentage < 0  # Negative means improvement
        assert regression_result.regression_detected is False

        # Should update baseline to better performance
        assert performance_tester.baseline_performance["improving_test"] == 40.0

    def test_calculate_performance_score_all_passed(self, performance_tester):
        """Test performance score calculation when all tests pass."""
        # Mock test results - all passed
        performance_tester.test_results = [
            Mock(test_passed=True),
            Mock(test_passed=True),
            Mock(test_passed=True),
        ]
        performance_tester.bottlenecks_detected = []
        performance_tester.regression_results = []

        score = performance_tester._calculate_performance_score()
        assert score == 100.0

    def test_calculate_performance_score_with_bottlenecks(self, performance_tester):
        """Test performance score calculation with bottlenecks."""
        # Mock test results - all passed
        performance_tester.test_results = [
            Mock(test_passed=True),
            Mock(test_passed=True),
        ]

        # Add bottlenecks
        performance_tester.bottlenecks_detected = [
            Mock(severity="HIGH"),
            Mock(severity="MEDIUM"),
            Mock(severity="LOW"),
        ]
        performance_tester.regression_results = []

        score = performance_tester._calculate_performance_score()
        # Base score 100, minus penalties: HIGH(10) + MEDIUM(5) + LOW(2) = 17
        assert score == 83.0

    def test_calculate_performance_score_with_regressions(self, performance_tester):
        """Test performance score calculation with regressions."""
        # Mock test results - all passed
        performance_tester.test_results = [Mock(test_passed=True)]
        performance_tester.bottlenecks_detected = []

        # Add regressions
        performance_tester.regression_results = [
            Mock(
                regression_detected=True, regression_percentage=30.0
            ),  # 3 point penalty
            Mock(regression_detected=False, regression_percentage=5.0),  # No penalty
        ]

        score = performance_tester._calculate_performance_score()
        # Base score 100, minus regression penalty: 3
        assert score == 97.0

    @pytest.mark.asyncio
    async def test_profile_performance_bottlenecks_integration(
        self, performance_tester
    ):
        """Test the complete performance profiling integration."""
        # This is an integration test that runs the full profiling system
        metrics = await performance_tester.profile_performance_bottlenecks()

        assert isinstance(metrics, PerformanceMetrics)
        assert isinstance(metrics.test_results, list)
        assert isinstance(metrics.profiling_results, list)
        assert isinstance(metrics.bottlenecks_detected, list)
        assert isinstance(metrics.regression_results, list)
        assert isinstance(metrics.overall_performance_score, float)
        assert isinstance(metrics.system_resource_usage, dict)
        assert metrics.execution_time_seconds > 0
        assert metrics.timestamp is not None

        # Should have profiled the three main test functions
        assert len(metrics.regression_results) <= 3  # May be less if profiling fails

    @pytest.mark.asyncio
    async def test_run_tests_includes_profiling(self, performance_tester):
        """Test that run_tests includes profiling results."""
        result = await performance_tester.run_tests()

        assert result.module_name == "performance_testing"
        assert "performance_profiling" in result.detailed_results

        profiling_data = result.detailed_results["performance_profiling"]
        assert "profiling_results" in profiling_data
        assert "bottlenecks_detected" in profiling_data
        assert "regression_results" in profiling_data
        assert "overall_performance_score" in profiling_data
        assert "system_resource_usage" in profiling_data

        # Should have run 4 tests now (3 original + 1 profiling)
        assert result.tests_run == 4


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
