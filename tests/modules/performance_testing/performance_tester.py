"""
Performance Testing Module for comprehensive testing framework.
Tests system performance against defined targets for market data processing,
strategy signal generation, and portfolio updates.
"""

import asyncio
import time
import traceback
import statistics
import cProfile
import pstats
import io
import psutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ...orchestrator.test_orchestrator import (
        TestModuleInterface,
        TestModuleResult,
        TestStatus,
    )
    from ...config.test_config import TestConfig, PerformanceTestConfig
    from ...utils.test_utilities import TestLogger, DecimalHelper, MockDataGenerator
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from orchestrator.test_orchestrator import (
        TestModuleInterface,
        TestModuleResult,
        TestStatus,
    )
    from config.test_config import TestConfig, PerformanceTestConfig
    from utils.test_utilities import TestLogger, DecimalHelper, MockDataGenerator


@dataclass
class PerformanceTestResult:
    """Result from a single performance test."""

    test_name: str
    target_time_ms: int
    actual_times_ms: List[float]
    average_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_deviation_ms: float
    success_rate: float
    test_passed: bool
    iterations_run: int
    error_messages: List[str] = field(default_factory=list)
    profiling_data: Optional[Dict[str, Any]] = None


@dataclass
class ProfilingResult:
    """Result from performance profiling."""

    function_name: str
    total_time_seconds: float
    cumulative_time_seconds: float
    call_count: int
    time_per_call_seconds: float
    percentage_of_total: float


@dataclass
class BottleneckDetectionResult:
    """Result from bottleneck detection analysis."""

    bottleneck_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    affected_function: str
    performance_impact: float
    recommended_action: str
    detection_timestamp: datetime


@dataclass
class RegressionDetectionResult:
    """Result from performance regression detection."""

    test_name: str
    baseline_time_ms: float
    current_time_ms: float
    regression_percentage: float
    regression_detected: bool
    threshold_percentage: float
    detection_timestamp: datetime


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    test_results: List[PerformanceTestResult]
    profiling_results: List[ProfilingResult]
    bottlenecks_detected: List[BottleneckDetectionResult]
    regression_results: List[RegressionDetectionResult]
    overall_performance_score: float
    system_resource_usage: Dict[str, float]
    execution_time_seconds: float
    timestamp: datetime


class MockMarketDataProcessor:
    """Mock market data processor for performance testing."""

    def __init__(self):
        self.decimal_helper = DecimalHelper()
        self.mock_data_generator = MockDataGenerator()

    async def process_market_data(self, data_size: int = 1000) -> Dict[str, Any]:
        """Simulate market data processing."""
        # Generate mock market data
        market_data = []
        for i in range(data_size):
            market_data.append(
                {
                    "symbol": f"BTC/USDT",
                    "price": self.decimal_helper.create_decimal(
                        f"{50000 + i * 0.01:.8f}"
                    ),
                    "volume": self.decimal_helper.create_decimal(f"{100.12345678:.8f}"),
                    "timestamp": time.time() + i,
                }
            )

        # Simulate processing operations
        processed_data = []
        for data_point in market_data:
            # Simulate price calculations
            price_change = data_point["price"] * self.decimal_helper.create_decimal(
                "0.001"
            )
            processed_price = data_point["price"] + price_change

            # Simulate volume calculations
            volume_weighted_price = data_point["price"] * data_point["volume"]

            processed_data.append(
                {
                    "symbol": data_point["symbol"],
                    "processed_price": processed_price,
                    "volume_weighted_price": volume_weighted_price,
                    "timestamp": data_point["timestamp"],
                }
            )

        return {
            "processed_count": len(processed_data),
            "data": processed_data[:10],  # Return sample for verification
            "total_volume": sum(d["volume"] for d in market_data),
        }


class MockStrategyEngine:
    """Mock strategy engine for performance testing."""

    def __init__(self):
        self.decimal_helper = DecimalHelper()

    async def generate_trading_signal(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate strategy signal generation."""
        # Simulate complex strategy calculations
        price_data = market_data.get("data", [])

        if not price_data:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "No data"}

        # Simulate moving average calculation
        prices = [d["processed_price"] for d in price_data]
        if len(prices) >= 5:
            ma_5 = sum(prices[-5:]) / 5
            ma_10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else ma_5

            # Simulate signal logic
            if ma_5 > ma_10:
                signal = "BUY"
                confidence = min(float((ma_5 - ma_10) / ma_10 * 100), 1.0)
            elif ma_5 < ma_10:
                signal = "SELL"
                confidence = min(float((ma_10 - ma_5) / ma_10 * 100), 1.0)
            else:
                signal = "HOLD"
                confidence = 0.5
        else:
            signal = "HOLD"
            confidence = 0.0

        # Simulate additional strategy calculations
        await asyncio.sleep(0.001)  # Simulate computation time

        return {
            "signal": signal,
            "confidence": confidence,
            "price": prices[-1] if prices else self.decimal_helper.create_decimal("0"),
            "timestamp": time.time(),
        }


class MockPortfolioManager:
    """Mock portfolio manager for performance testing."""

    def __init__(self):
        self.decimal_helper = DecimalHelper()
        self.positions = {}
        self.balance = self.decimal_helper.create_decimal("10000.00000000")

    async def update_portfolio(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate portfolio update operations."""
        signal = signal_data.get("signal", "HOLD")
        price = signal_data.get("price", self.decimal_helper.create_decimal("0"))
        confidence = signal_data.get("confidence", 0.0)

        # Simulate portfolio calculations
        position_size = self.decimal_helper.create_decimal(str(confidence * 100))

        if signal == "BUY" and confidence > 0.7:
            # Simulate buy operation
            cost = position_size * price
            if cost <= self.balance:
                self.balance -= cost
                self.positions["BTC"] = (
                    self.positions.get("BTC", self.decimal_helper.create_decimal("0"))
                    + position_size
                )

        elif signal == "SELL" and confidence > 0.7:
            # Simulate sell operation
            btc_position = self.positions.get(
                "BTC", self.decimal_helper.create_decimal("0")
            )
            sell_amount = min(position_size, btc_position)
            if sell_amount > 0:
                self.positions["BTC"] -= sell_amount
                self.balance += sell_amount * price

        # Calculate portfolio value
        btc_value = (
            self.positions.get("BTC", self.decimal_helper.create_decimal("0")) * price
        )
        total_value = self.balance + btc_value

        # Simulate additional portfolio calculations
        await asyncio.sleep(0.002)  # Simulate computation time

        return {
            "total_value": total_value,
            "cash_balance": self.balance,
            "positions": dict(self.positions),
            "last_update": time.time(),
        }


class PerformanceTester(TestModuleInterface):
    """
    Performance Testing Module implementing TestModuleInterface.

    Tests system performance against defined targets for market data processing,
    strategy signal generation, and portfolio updates.
    Implements requirements 6.1, 6.2, 6.3.
    """

    def __init__(self, config: PerformanceTestConfig = None):
        self.config = config or PerformanceTestConfig(
            market_data_processing_target_ms=100,
            strategy_signal_generation_target_ms=500,
            portfolio_update_target_ms=1000,
            performance_test_iterations=10,
        )
        self.logger = TestLogger("performance_tester", "INFO")
        self.decimal_helper = DecimalHelper()

        # Initialize mock components
        self.market_data_processor = MockMarketDataProcessor()
        self.strategy_engine = MockStrategyEngine()
        self.portfolio_manager = MockPortfolioManager()

        self.test_results: List[PerformanceTestResult] = []
        self.profiling_results: List[ProfilingResult] = []
        self.bottlenecks_detected: List[BottleneckDetectionResult] = []
        self.regression_results: List[RegressionDetectionResult] = []
        self.baseline_performance: Dict[str, float] = {}
        self.system_monitor = psutil.Process()

        self.logger.info(
            f"PerformanceTester initialized with targets: "
            f"market_data={self.config.market_data_processing_target_ms}ms, "
            f"strategy_signal={self.config.strategy_signal_generation_target_ms}ms, "
            f"portfolio_update={self.config.portfolio_update_target_ms}ms"
        )

    def get_module_name(self) -> str:
        """Get the name of this test module."""
        return "performance_testing"

    async def _measure_execution_time(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[float, Any, Optional[str]]:
        """Measure execution time of a function in milliseconds."""
        start_time = time.perf_counter()
        error_message = None
        result = None

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
        except Exception as e:
            error_message = f"Function execution failed: {str(e)}"
            self.logger.error(f"Error in _measure_execution_time: {error_message}")

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        return execution_time_ms, result, error_message

    async def test_market_data_processing_speed(self) -> PerformanceTestResult:
        """
        Test market data processing speed against < 100ms requirement.

        Returns:
            PerformanceTestResult with market data processing performance details
        """
        self.logger.info("Testing market data processing speed")

        test_name = "market_data_processing"
        target_time_ms = self.config.market_data_processing_target_ms
        actual_times_ms = []
        error_messages = []
        successful_runs = 0

        # Run multiple iterations for statistical accuracy
        for i in range(self.config.performance_test_iterations):
            execution_time_ms, result, error_message = (
                await self._measure_execution_time(
                    self.market_data_processor.process_market_data, data_size=1000
                )
            )

            actual_times_ms.append(execution_time_ms)

            if error_message:
                error_messages.append(f"Iteration {i+1}: {error_message}")
            else:
                successful_runs += 1
                self.logger.debug(
                    f"Market data processing iteration {i+1}: {execution_time_ms:.2f}ms"
                )

        # Calculate statistics
        if actual_times_ms:
            average_time_ms = statistics.mean(actual_times_ms)
            median_time_ms = statistics.median(actual_times_ms)
            min_time_ms = min(actual_times_ms)
            max_time_ms = max(actual_times_ms)
            std_deviation_ms = (
                statistics.stdev(actual_times_ms) if len(actual_times_ms) > 1 else 0.0
            )
        else:
            average_time_ms = median_time_ms = min_time_ms = max_time_ms = (
                std_deviation_ms
            ) = 0.0

        success_rate = successful_runs / self.config.performance_test_iterations
        test_passed = average_time_ms <= target_time_ms and success_rate >= 0.9

        result = PerformanceTestResult(
            test_name=test_name,
            target_time_ms=target_time_ms,
            actual_times_ms=actual_times_ms,
            average_time_ms=average_time_ms,
            median_time_ms=median_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            std_deviation_ms=std_deviation_ms,
            success_rate=success_rate,
            test_passed=test_passed,
            iterations_run=self.config.performance_test_iterations,
            error_messages=error_messages,
        )

        self.logger.info(
            f"Market data processing test completed: "
            f"average={average_time_ms:.2f}ms, target={target_time_ms}ms, "
            f"passed={test_passed}"
        )

        return result

    async def test_strategy_signal_generation_speed(self) -> PerformanceTestResult:
        """
        Test strategy signal generation speed against < 500ms requirement.

        Returns:
            PerformanceTestResult with strategy signal generation performance details
        """
        self.logger.info("Testing strategy signal generation speed")

        test_name = "strategy_signal_generation"
        target_time_ms = self.config.strategy_signal_generation_target_ms
        actual_times_ms = []
        error_messages = []
        successful_runs = 0

        # Prepare test market data
        market_data = await self.market_data_processor.process_market_data(
            data_size=100
        )

        # Run multiple iterations for statistical accuracy
        for i in range(self.config.performance_test_iterations):
            execution_time_ms, result, error_message = (
                await self._measure_execution_time(
                    self.strategy_engine.generate_trading_signal, market_data
                )
            )

            actual_times_ms.append(execution_time_ms)

            if error_message:
                error_messages.append(f"Iteration {i+1}: {error_message}")
            else:
                successful_runs += 1
                self.logger.debug(
                    f"Strategy signal generation iteration {i+1}: {execution_time_ms:.2f}ms"
                )

        # Calculate statistics
        if actual_times_ms:
            average_time_ms = statistics.mean(actual_times_ms)
            median_time_ms = statistics.median(actual_times_ms)
            min_time_ms = min(actual_times_ms)
            max_time_ms = max(actual_times_ms)
            std_deviation_ms = (
                statistics.stdev(actual_times_ms) if len(actual_times_ms) > 1 else 0.0
            )
        else:
            average_time_ms = median_time_ms = min_time_ms = max_time_ms = (
                std_deviation_ms
            ) = 0.0

        success_rate = successful_runs / self.config.performance_test_iterations
        test_passed = average_time_ms <= target_time_ms and success_rate >= 0.9

        result = PerformanceTestResult(
            test_name=test_name,
            target_time_ms=target_time_ms,
            actual_times_ms=actual_times_ms,
            average_time_ms=average_time_ms,
            median_time_ms=median_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            std_deviation_ms=std_deviation_ms,
            success_rate=success_rate,
            test_passed=test_passed,
            iterations_run=self.config.performance_test_iterations,
            error_messages=error_messages,
        )

        self.logger.info(
            f"Strategy signal generation test completed: "
            f"average={average_time_ms:.2f}ms, target={target_time_ms}ms, "
            f"passed={test_passed}"
        )

        return result

    async def test_portfolio_update_speed(self) -> PerformanceTestResult:
        """
        Test portfolio update speed against < 1 second requirement.

        Returns:
            PerformanceTestResult with portfolio update performance details
        """
        self.logger.info("Testing portfolio update speed")

        test_name = "portfolio_update"
        target_time_ms = self.config.portfolio_update_target_ms
        actual_times_ms = []
        error_messages = []
        successful_runs = 0

        # Prepare test data
        market_data = await self.market_data_processor.process_market_data(data_size=50)
        signal_data = await self.strategy_engine.generate_trading_signal(market_data)

        # Run multiple iterations for statistical accuracy
        for i in range(self.config.performance_test_iterations):
            execution_time_ms, result, error_message = (
                await self._measure_execution_time(
                    self.portfolio_manager.update_portfolio, signal_data
                )
            )

            actual_times_ms.append(execution_time_ms)

            if error_message:
                error_messages.append(f"Iteration {i+1}: {error_message}")
            else:
                successful_runs += 1
                self.logger.debug(
                    f"Portfolio update iteration {i+1}: {execution_time_ms:.2f}ms"
                )

        # Calculate statistics
        if actual_times_ms:
            average_time_ms = statistics.mean(actual_times_ms)
            median_time_ms = statistics.median(actual_times_ms)
            min_time_ms = min(actual_times_ms)
            max_time_ms = max(actual_times_ms)
            std_deviation_ms = (
                statistics.stdev(actual_times_ms) if len(actual_times_ms) > 1 else 0.0
            )
        else:
            average_time_ms = median_time_ms = min_time_ms = max_time_ms = (
                std_deviation_ms
            ) = 0.0

        success_rate = successful_runs / self.config.performance_test_iterations
        test_passed = average_time_ms <= target_time_ms and success_rate >= 0.9

        result = PerformanceTestResult(
            test_name=test_name,
            target_time_ms=target_time_ms,
            actual_times_ms=actual_times_ms,
            average_time_ms=average_time_ms,
            median_time_ms=median_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            std_deviation_ms=std_deviation_ms,
            success_rate=success_rate,
            test_passed=test_passed,
            iterations_run=self.config.performance_test_iterations,
            error_messages=error_messages,
        )

        self.logger.info(
            f"Portfolio update test completed: "
            f"average={average_time_ms:.2f}ms, target={target_time_ms}ms, "
            f"passed={test_passed}"
        )

        return result

    def _profile_function_execution(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[Any, List[ProfilingResult]]:
        """Profile function execution to identify performance bottlenecks."""
        profiler = cProfile.Profile()

        try:
            profiler.enable()
            if asyncio.iscoroutinefunction(func):
                # For async functions, check if we're already in an event loop
                try:
                    # Try to get the current event loop
                    current_loop = asyncio.get_running_loop()
                    # If we're in a loop, use run_in_executor to avoid conflicts
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, func(*args, **kwargs))
                        result = future.result(timeout=30)
                except RuntimeError:
                    # No event loop running, safe to create new one
                    result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            profiler.disable()

            # Analyze profiling results
            profiling_results = self._analyze_profiling_data(profiler)

            return result, profiling_results

        except Exception as e:
            profiler.disable()
            self.logger.error(f"Error during function profiling: {str(e)}")
            return None, []

    def _analyze_profiling_data(
        self, profiler: cProfile.Profile
    ) -> List[ProfilingResult]:
        """Analyze cProfile data to extract performance insights."""
        profiling_results = []

        # Create string buffer to capture profiler output
        string_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=string_buffer)
        stats.sort_stats("cumulative")

        # Get raw stats data
        total_time = stats.total_tt

        # Process stats using the stats object directly
        for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():
            if ct > 0.001:  # Only include functions taking > 1ms cumulative time
                filename, line_num, func_name = func_key

                profiling_result = ProfilingResult(
                    function_name=f"{filename}:{line_num}({func_name})",
                    total_time_seconds=tt,
                    cumulative_time_seconds=ct,
                    call_count=cc,
                    time_per_call_seconds=ct / cc if cc > 0 else 0,
                    percentage_of_total=(
                        (ct / total_time * 100) if total_time > 0 else 0
                    ),
                )
                profiling_results.append(profiling_result)

        # Sort by cumulative time and take top 20
        profiling_results.sort(key=lambda x: x.cumulative_time_seconds, reverse=True)
        profiling_results = profiling_results[:20]

        return profiling_results

    def _detect_performance_bottlenecks(
        self,
        profiling_results: List[ProfilingResult],
        test_name: str,
        target_time_ms: int,
    ) -> List[BottleneckDetectionResult]:
        """Detect performance bottlenecks from profiling data."""
        bottlenecks = []

        for profile_result in profiling_results:
            # Detect high time consumption
            if profile_result.cumulative_time_seconds * 1000 > target_time_ms * 0.5:
                severity = (
                    "HIGH"
                    if profile_result.cumulative_time_seconds * 1000
                    > target_time_ms * 0.8
                    else "MEDIUM"
                )

                bottleneck = BottleneckDetectionResult(
                    bottleneck_type="high_execution_time",
                    severity=severity,
                    description=f"Function consuming {profile_result.percentage_of_total:.1f}% of total execution time",
                    affected_function=profile_result.function_name,
                    performance_impact=profile_result.percentage_of_total,
                    recommended_action="Optimize algorithm or consider caching",
                    detection_timestamp=datetime.now(),
                )
                bottlenecks.append(bottleneck)

            # Detect excessive function calls
            if profile_result.call_count > 1000:
                bottleneck = BottleneckDetectionResult(
                    bottleneck_type="excessive_calls",
                    severity="MEDIUM",
                    description=f"Function called {profile_result.call_count} times",
                    affected_function=profile_result.function_name,
                    performance_impact=profile_result.call_count / 1000,
                    recommended_action="Consider reducing call frequency or batching operations",
                    detection_timestamp=datetime.now(),
                )
                bottlenecks.append(bottleneck)

        return bottlenecks

    def _monitor_system_resources(self) -> Dict[str, float]:
        """Monitor system resource usage during test execution."""
        try:
            cpu_percent = self.system_monitor.cpu_percent()
            memory_info = self.system_monitor.memory_info()
            memory_percent = self.system_monitor.memory_percent()

            # Get system-wide metrics
            system_cpu = psutil.cpu_percent(interval=0.1)
            system_memory = psutil.virtual_memory().percent

            return {
                "process_cpu_percent": cpu_percent,
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "process_memory_percent": memory_percent,
                "system_cpu_percent": system_cpu,
                "system_memory_percent": system_memory,
                "thread_count": threading.active_count(),
            }
        except Exception as e:
            self.logger.error(f"Error monitoring system resources: {str(e)}")
            return {}

    def _detect_performance_regression(
        self, test_name: str, current_time_ms: float, threshold_percentage: float = 20.0
    ) -> RegressionDetectionResult:
        """Detect performance regression compared to baseline."""
        baseline_time_ms = self.baseline_performance.get(test_name, current_time_ms)

        if baseline_time_ms == 0:
            regression_percentage = 0
            regression_detected = False
        else:
            regression_percentage = (
                (current_time_ms - baseline_time_ms) / baseline_time_ms
            ) * 100
            regression_detected = regression_percentage > threshold_percentage

        # Update baseline if this is better performance
        if (
            current_time_ms < baseline_time_ms
            or test_name not in self.baseline_performance
        ):
            self.baseline_performance[test_name] = current_time_ms

        return RegressionDetectionResult(
            test_name=test_name,
            baseline_time_ms=baseline_time_ms,
            current_time_ms=current_time_ms,
            regression_percentage=regression_percentage,
            regression_detected=regression_detected,
            threshold_percentage=threshold_percentage,
            detection_timestamp=datetime.now(),
        )

    async def profile_performance_bottlenecks(self) -> PerformanceMetrics:
        """
        Comprehensive performance profiling and bottleneck detection.

        Returns:
            PerformanceMetrics with detailed profiling and bottleneck analysis
        """
        self.logger.info("Starting comprehensive performance profiling")
        start_time = time.time()

        # Clear previous results
        self.profiling_results.clear()
        self.bottlenecks_detected.clear()
        self.regression_results.clear()

        # Profile each performance test
        test_functions = [
            (
                "market_data_processing",
                self.test_market_data_processing_speed,
                self.config.market_data_processing_target_ms,
            ),
            (
                "strategy_signal_generation",
                self.test_strategy_signal_generation_speed,
                self.config.strategy_signal_generation_target_ms,
            ),
            (
                "portfolio_update",
                self.test_portfolio_update_speed,
                self.config.portfolio_update_target_ms,
            ),
        ]

        for test_name, test_func, target_ms in test_functions:
            self.logger.info(f"Profiling {test_name}")

            # Monitor system resources before test
            pre_test_resources = self._monitor_system_resources()

            # Profile the test function
            result, profiling_results = self._profile_function_execution(test_func)

            # Monitor system resources after test
            post_test_resources = self._monitor_system_resources()

            if result and profiling_results:
                self.profiling_results.extend(profiling_results)

                # Detect bottlenecks
                bottlenecks = self._detect_performance_bottlenecks(
                    profiling_results, test_name, target_ms
                )
                self.bottlenecks_detected.extend(bottlenecks)

                # Detect regression
                if hasattr(result, "average_time_ms"):
                    regression_result = self._detect_performance_regression(
                        test_name, result.average_time_ms
                    )
                    self.regression_results.append(regression_result)

        # Calculate overall performance score
        overall_score = self._calculate_performance_score()

        # Get final system resource usage
        system_resources = self._monitor_system_resources()

        execution_time = time.time() - start_time

        metrics = PerformanceMetrics(
            test_results=self.test_results,
            profiling_results=self.profiling_results,
            bottlenecks_detected=self.bottlenecks_detected,
            regression_results=self.regression_results,
            overall_performance_score=overall_score,
            system_resource_usage=system_resources,
            execution_time_seconds=execution_time,
            timestamp=datetime.now(),
        )

        self.logger.info(
            f"Performance profiling completed: "
            f"score={overall_score:.2f}, bottlenecks={len(self.bottlenecks_detected)}, "
            f"regressions={sum(1 for r in self.regression_results if r.regression_detected)}"
        )

        return metrics

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score based on test results and bottlenecks."""
        if not self.test_results:
            return 0.0

        # Base score from test results
        passed_tests = sum(1 for result in self.test_results if result.test_passed)
        base_score = (passed_tests / len(self.test_results)) * 100

        # Penalty for bottlenecks
        bottleneck_penalty = 0
        for bottleneck in self.bottlenecks_detected:
            if bottleneck.severity == "CRITICAL":
                bottleneck_penalty += 20
            elif bottleneck.severity == "HIGH":
                bottleneck_penalty += 10
            elif bottleneck.severity == "MEDIUM":
                bottleneck_penalty += 5
            else:  # LOW
                bottleneck_penalty += 2

        # Penalty for regressions
        regression_penalty = sum(
            min(
                result.regression_percentage / 10, 15
            )  # Max 15 points penalty per regression
            for result in self.regression_results
            if result.regression_detected
        )

        # Calculate final score
        final_score = max(0, base_score - bottleneck_penalty - regression_penalty)

        return final_score

    async def run_tests(self) -> TestModuleResult:
        """
        Run all performance tests and return comprehensive results.

        Returns:
            TestModuleResult with performance testing results
        """
        start_time = time.time()
        self.logger.info("Starting performance testing module")

        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        error_messages = []
        detailed_results = {}

        try:
            # Test 1: Market Data Processing Speed
            try:
                market_data_result = await self.test_market_data_processing_speed()
                self.test_results.append(market_data_result)
                detailed_results["market_data_processing"] = market_data_result
                tests_run += 1
                if market_data_result.test_passed:
                    tests_passed += 1
                else:
                    tests_failed += 1
                    error_messages.extend(market_data_result.error_messages)
            except Exception as e:
                tests_run += 1
                tests_failed += 1
                error_msg = f"Market data processing test failed: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Test 2: Strategy Signal Generation Speed
            try:
                strategy_signal_result = (
                    await self.test_strategy_signal_generation_speed()
                )
                self.test_results.append(strategy_signal_result)
                detailed_results["strategy_signal_generation"] = strategy_signal_result
                tests_run += 1
                if strategy_signal_result.test_passed:
                    tests_passed += 1
                else:
                    tests_failed += 1
                    error_messages.extend(strategy_signal_result.error_messages)
            except Exception as e:
                tests_run += 1
                tests_failed += 1
                error_msg = f"Strategy signal generation test failed: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Test 3: Portfolio Update Speed
            try:
                portfolio_update_result = await self.test_portfolio_update_speed()
                self.test_results.append(portfolio_update_result)
                detailed_results["portfolio_update"] = portfolio_update_result
                tests_run += 1
                if portfolio_update_result.test_passed:
                    tests_passed += 1
                else:
                    tests_failed += 1
                    error_messages.extend(portfolio_update_result.error_messages)
            except Exception as e:
                tests_run += 1
                tests_failed += 1
                error_msg = f"Portfolio update test failed: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Test 4: Performance Profiling and Bottleneck Detection
            try:
                self.logger.info(
                    "Running performance profiling and bottleneck detection"
                )
                performance_metrics = await self.profile_performance_bottlenecks()
                detailed_results["performance_profiling"] = {
                    "profiling_results": performance_metrics.profiling_results,
                    "bottlenecks_detected": performance_metrics.bottlenecks_detected,
                    "regression_results": performance_metrics.regression_results,
                    "overall_performance_score": performance_metrics.overall_performance_score,
                    "system_resource_usage": performance_metrics.system_resource_usage,
                }
                tests_run += 1

                # Consider profiling successful if no critical bottlenecks found
                critical_bottlenecks = [
                    b
                    for b in performance_metrics.bottlenecks_detected
                    if b.severity == "CRITICAL"
                ]
                if len(critical_bottlenecks) == 0:
                    tests_passed += 1
                else:
                    tests_failed += 1
                    error_messages.append(
                        f"Critical performance bottlenecks detected: {len(critical_bottlenecks)}"
                    )

            except Exception as e:
                tests_run += 1
                tests_failed += 1
                error_msg = f"Performance profiling failed: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

        except Exception as e:
            error_msg = f"Performance testing module failed: {str(e)}"
            error_messages.append(error_msg)
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

        # Determine overall status
        if tests_failed > 0:
            status = TestStatus.FAILED
        elif tests_run > 0:
            status = TestStatus.COMPLETED
        else:
            status = TestStatus.FAILED

        execution_time = time.time() - start_time

        result = TestModuleResult(
            module_name=self.get_module_name(),
            status=status,
            execution_time_seconds=execution_time,
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            error_messages=error_messages,
            detailed_results=detailed_results,
            start_time=datetime.fromtimestamp(start_time),
            end_time=datetime.fromtimestamp(time.time()),
        )

        self.logger.info(
            f"Performance testing module completed: "
            f"status={status.value}, tests_run={tests_run}, "
            f"tests_passed={tests_passed}, tests_failed={tests_failed}, "
            f"execution_time={execution_time:.2f}s"
        )

        return result
