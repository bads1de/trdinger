"""
Backtest Testing Module for comprehensive testing framework.
Tests backtest functionality accuracy, Sharpe ratio, maximum drawdown, win rate.
"""

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import statistics
from scipy import stats
import logging

try:
    from ...orchestrator.test_orchestrator import (
        TestModuleInterface,
        TestModuleResult,
        TestStatus,
    )
    from ...config.test_config import TestConfig, BacktestTestConfig
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
    from config.test_config import TestConfig, BacktestTestConfig
    from utils.test_utilities import TestLogger, DecimalHelper, MockDataGenerator


@dataclass
class BacktestMetricsResult:
    """Result from backtest metrics testing."""

    test_name: str
    expected_value: Decimal
    calculated_value: Decimal
    tolerance: Decimal
    passed: bool
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class BacktestTestResult:
    """Comprehensive result from backtest testing."""

    sharpe_ratio_test: BacktestMetricsResult
    max_drawdown_test: BacktestMetricsResult
    win_rate_test: BacktestMetricsResult
    total_return_test: BacktestMetricsResult
    statistical_validation: Dict[str, Any]
    known_cases_validation: List[Dict[str, Any]]
    overall_passed: bool
    execution_time_seconds: float
    timestamp: datetime


class BacktestTester(TestModuleInterface):
    """
    Comprehensive backtest testing module that validates:
    - Backtest calculation accuracy with Decimal precision
    - Sharpe ratio calculations with statistical validation
    - Maximum drawdown calculations with known test cases
    - Win rate calculations with accuracy tests
    - Portfolio performance metrics validation

    Implements TestModuleInterface for integration with TestOrchestrator.
    """

    def __init__(self, config: BacktestTestConfig = None):
        self.config = config or BacktestTestConfig(
            sharpe_ratio_tolerance=0.01,
            max_drawdown_tolerance=0.01,
            win_rate_tolerance=0.01,
            extreme_condition_scenarios=[
                "high_volatility",
                "market_crash",
                "flash_crash",
                "low_liquidity",
                "extreme_pump",
            ],
        )
        self.logger = TestLogger("backtest_tester")
        self.test_results: Dict[str, Any] = {}
        self.precision = DecimalHelper.PRECISION
        self.mock_data_generator = MockDataGenerator()

        # Known test cases for validation
        self.known_test_cases = self._initialize_known_test_cases()

    async def run_tests(self) -> TestModuleResult:
        """
        Run all backtest tests and return comprehensive results.
        Implements TestModuleInterface.run_tests method.
        """
        start_time = datetime.now()
        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        error_messages = []
        detailed_results = {}

        try:
            self.logger.info("Starting comprehensive backtest testing")

            # Test 1: Sharpe ratio calculation with known test cases
            sharpe_test_result = await self._test_sharpe_ratio_calculation()
            tests_run += 1
            if sharpe_test_result.passed:
                tests_passed += 1
            else:
                tests_failed += 1
                error_messages.append(
                    f"Sharpe ratio test failed: {sharpe_test_result.error_message}"
                )
            detailed_results["sharpe_ratio_test"] = sharpe_test_result

            # Test 2: Maximum drawdown calculation with validation
            drawdown_test_result = await self._test_max_drawdown_calculation()
            tests_run += 1
            if drawdown_test_result.passed:
                tests_passed += 1
            else:
                tests_failed += 1
                error_messages.append(
                    f"Max drawdown test failed: {drawdown_test_result.error_message}"
                )
            detailed_results["max_drawdown_test"] = drawdown_test_result

            # Test 3: Win rate calculation accuracy
            win_rate_test_result = await self._test_win_rate_calculation()
            tests_run += 1
            if win_rate_test_result.passed:
                tests_passed += 1
            else:
                tests_failed += 1
                error_messages.append(
                    f"Win rate test failed: {win_rate_test_result.error_message}"
                )
            detailed_results["win_rate_test"] = win_rate_test_result

            # Test 4: Total return calculation validation
            total_return_test_result = await self._test_total_return_calculation()
            tests_run += 1
            if total_return_test_result.passed:
                tests_passed += 1
            else:
                tests_failed += 1
                error_messages.append(
                    f"Total return test failed: {total_return_test_result.error_message}"
                )
            detailed_results["total_return_test"] = total_return_test_result

            # Test 5: Statistical validation of calculations
            statistical_validation = await self._test_statistical_validation()
            tests_run += 1
            if statistical_validation.get("passed", False):
                tests_passed += 1
            else:
                tests_failed += 1
                error_messages.append(
                    f"Statistical validation failed: {statistical_validation.get('error', 'Unknown error')}"
                )
            detailed_results["statistical_validation"] = statistical_validation

            # Test 6: Known reference data comparison
            known_cases_validation = await self._test_known_cases_validation()
            tests_run += len(known_cases_validation)
            for case in known_cases_validation:
                if case.get("passed", False):
                    tests_passed += 1
                else:
                    tests_failed += 1
                    error_messages.append(
                        f"Known case validation failed: {case.get('error', 'Unknown error')}"
                    )
            detailed_results["known_cases_validation"] = known_cases_validation

            # Test 7: Extreme market condition testing
            extreme_conditions_result = await self._test_extreme_market_conditions()
            tests_run += 1
            if extreme_conditions_result.get("passed", False):
                tests_passed += 1
            else:
                tests_failed += 1
                error_messages.append(
                    f"Extreme market conditions test failed: {extreme_conditions_result.get('error', 'Unknown error')}"
                )
            detailed_results["extreme_conditions_test"] = extreme_conditions_result

            # Test 8: Edge case handling validation
            edge_case_result = await self._test_edge_case_handling()
            tests_run += 1
            if edge_case_result.get("passed", False):
                tests_passed += 1
            else:
                tests_failed += 1
                error_messages.append(
                    f"Edge case handling test failed: {edge_case_result.get('error', 'Unknown error')}"
                )
            detailed_results["edge_case_handling_test"] = edge_case_result

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Determine overall status
            if tests_failed == 0:
                status = TestStatus.COMPLETED
                self.logger.info(
                    f"All backtest tests passed. {tests_passed}/{tests_run} successful"
                )
            else:
                status = TestStatus.FAILED
                self.logger.error(
                    f"Some backtest tests failed. {tests_passed}/{tests_run} successful"
                )

            return TestModuleResult(
                module_name=self.get_module_name(),
                status=status,
                execution_time_seconds=execution_time,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_skipped=0,
                error_messages=error_messages,
                detailed_results=detailed_results,
                start_time=start_time,
                end_time=end_time,
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            error_msg = f"Critical error in backtest testing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return TestModuleResult(
                module_name=self.get_module_name(),
                status=TestStatus.FAILED,
                execution_time_seconds=execution_time,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_run + 1,  # +1 for the critical error
                tests_skipped=0,
                error_messages=[error_msg],
                detailed_results=detailed_results,
                start_time=start_time,
                end_time=end_time,
                exception_details=traceback.format_exc(),
            )

    def get_module_name(self) -> str:
        """Get the name of this test module."""
        return "backtest_testing"

    def _initialize_known_test_cases(self) -> List[Dict[str, Any]]:
        """Initialize known test cases for validation."""
        return [
            {
                "name": "simple_profitable_strategy",
                "trades": [
                    {"pnl": DecimalHelper.create_decimal("100.00")},
                    {"pnl": DecimalHelper.create_decimal("50.00")},
                    {"pnl": DecimalHelper.create_decimal("-25.00")},
                    {"pnl": DecimalHelper.create_decimal("75.00")},
                ],
                "expected_total_return": DecimalHelper.create_decimal("200.00"),
                "expected_win_rate": DecimalHelper.create_decimal("75.00"),  # 3/4 = 75%
            },
            {
                "name": "losing_strategy",
                "trades": [
                    {"pnl": DecimalHelper.create_decimal("-50.00")},
                    {"pnl": DecimalHelper.create_decimal("-30.00")},
                    {"pnl": DecimalHelper.create_decimal("20.00")},
                    {"pnl": DecimalHelper.create_decimal("-40.00")},
                ],
                "expected_total_return": DecimalHelper.create_decimal("-100.00"),
                "expected_win_rate": DecimalHelper.create_decimal("25.00"),  # 1/4 = 25%
            },
            {
                "name": "high_drawdown_scenario",
                "equity_curve": [
                    DecimalHelper.create_decimal("10000.00"),
                    DecimalHelper.create_decimal("12000.00"),
                    DecimalHelper.create_decimal(
                        "8000.00"
                    ),  # 33.33% drawdown from peak
                    DecimalHelper.create_decimal("9000.00"),
                    DecimalHelper.create_decimal("11000.00"),
                ],
                "expected_max_drawdown": DecimalHelper.create_decimal(
                    "0.3333"
                ),  # 33.33%
            },
        ]

    async def _test_sharpe_ratio_calculation(self) -> BacktestMetricsResult:
        """Test Sharpe ratio calculation with known test cases."""
        start_time = time.time()

        try:
            # Generate test data with known statistical properties
            returns = [0.02, -0.01, 0.03, -0.005, 0.015, 0.01, -0.02, 0.025]

            # Calculate expected Sharpe ratio manually
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            expected_sharpe = DecimalHelper.create_decimal(
                str(mean_return / std_return)
            )

            # Test our calculation
            calculated_sharpe = self._calculate_sharpe_ratio(returns)

            # Check tolerance
            tolerance = DecimalHelper.create_decimal(
                str(self.config.sharpe_ratio_tolerance)
            )
            passed = DecimalHelper.compare_decimals(
                expected_sharpe, calculated_sharpe, tolerance
            )

            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            if passed:
                self.logger.info(
                    f"Sharpe ratio test passed: expected={expected_sharpe}, calculated={calculated_sharpe}"
                )
            else:
                self.logger.error(
                    f"Sharpe ratio test failed: expected={expected_sharpe}, calculated={calculated_sharpe}, tolerance={tolerance}"
                )

            return BacktestMetricsResult(
                test_name="sharpe_ratio_calculation",
                expected_value=expected_sharpe,
                calculated_value=calculated_sharpe,
                tolerance=tolerance,
                passed=passed,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Error in Sharpe ratio calculation test: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return BacktestMetricsResult(
                test_name="sharpe_ratio_calculation",
                expected_value=DecimalHelper.create_decimal("0"),
                calculated_value=DecimalHelper.create_decimal("0"),
                tolerance=DecimalHelper.create_decimal("0"),
                passed=False,
                error_message=error_msg,
                execution_time_ms=execution_time,
            )

    async def _test_max_drawdown_calculation(self) -> BacktestMetricsResult:
        """Test maximum drawdown calculation with known test cases."""
        start_time = time.time()

        try:
            # Use known test case
            test_case = self.known_test_cases[2]  # high_drawdown_scenario
            equity_curve = [float(val) for val in test_case["equity_curve"]]
            expected_drawdown = test_case["expected_max_drawdown"]

            # Test our calculation
            calculated_drawdown = self._calculate_max_drawdown(equity_curve)

            # Check tolerance
            tolerance = DecimalHelper.create_decimal(
                str(self.config.max_drawdown_tolerance)
            )
            passed = DecimalHelper.compare_decimals(
                expected_drawdown, calculated_drawdown, tolerance
            )

            execution_time = (time.time() - start_time) * 1000

            if passed:
                self.logger.info(
                    f"Max drawdown test passed: expected={expected_drawdown}, calculated={calculated_drawdown}"
                )
            else:
                self.logger.error(
                    f"Max drawdown test failed: expected={expected_drawdown}, calculated={calculated_drawdown}, tolerance={tolerance}"
                )

            return BacktestMetricsResult(
                test_name="max_drawdown_calculation",
                expected_value=expected_drawdown,
                calculated_value=calculated_drawdown,
                tolerance=tolerance,
                passed=passed,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Error in max drawdown calculation test: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return BacktestMetricsResult(
                test_name="max_drawdown_calculation",
                expected_value=DecimalHelper.create_decimal("0"),
                calculated_value=DecimalHelper.create_decimal("0"),
                tolerance=DecimalHelper.create_decimal("0"),
                passed=False,
                error_message=error_msg,
                execution_time_ms=execution_time,
            )

    async def _test_win_rate_calculation(self) -> BacktestMetricsResult:
        """Test win rate calculation accuracy with statistical validation."""
        start_time = time.time()

        try:
            # Use known test case
            test_case = self.known_test_cases[0]  # simple_profitable_strategy
            trades = test_case["trades"]
            expected_win_rate = test_case["expected_win_rate"]

            # Test our calculation
            calculated_win_rate = self._calculate_win_rate(trades)

            # Check tolerance
            tolerance = DecimalHelper.create_decimal(
                str(self.config.win_rate_tolerance)
            )
            passed = DecimalHelper.compare_decimals(
                expected_win_rate, calculated_win_rate, tolerance
            )

            execution_time = (time.time() - start_time) * 1000

            if passed:
                self.logger.info(
                    f"Win rate test passed: expected={expected_win_rate}%, calculated={calculated_win_rate}%"
                )
            else:
                self.logger.error(
                    f"Win rate test failed: expected={expected_win_rate}%, calculated={calculated_win_rate}%, tolerance={tolerance}"
                )

            return BacktestMetricsResult(
                test_name="win_rate_calculation",
                expected_value=expected_win_rate,
                calculated_value=calculated_win_rate,
                tolerance=tolerance,
                passed=passed,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Error in win rate calculation test: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return BacktestMetricsResult(
                test_name="win_rate_calculation",
                expected_value=DecimalHelper.create_decimal("0"),
                calculated_value=DecimalHelper.create_decimal("0"),
                tolerance=DecimalHelper.create_decimal("0"),
                passed=False,
                error_message=error_msg,
                execution_time_ms=execution_time,
            )

    async def _test_total_return_calculation(self) -> BacktestMetricsResult:
        """Test total return calculation validation."""
        start_time = time.time()

        try:
            # Use known test case
            test_case = self.known_test_cases[0]  # simple_profitable_strategy
            trades = test_case["trades"]
            expected_total_return = test_case["expected_total_return"]

            # Test our calculation
            calculated_total_return = self._calculate_total_return(trades)

            # Check exact match (should be precise with Decimal)
            tolerance = DecimalHelper.PRECISION
            passed = DecimalHelper.compare_decimals(
                expected_total_return, calculated_total_return, tolerance
            )

            execution_time = (time.time() - start_time) * 1000

            if passed:
                self.logger.info(
                    f"Total return test passed: expected={expected_total_return}, calculated={calculated_total_return}"
                )
            else:
                self.logger.error(
                    f"Total return test failed: expected={expected_total_return}, calculated={calculated_total_return}"
                )

            return BacktestMetricsResult(
                test_name="total_return_calculation",
                expected_value=expected_total_return,
                calculated_value=calculated_total_return,
                tolerance=tolerance,
                passed=passed,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Error in total return calculation test: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return BacktestMetricsResult(
                test_name="total_return_calculation",
                expected_value=DecimalHelper.create_decimal("0"),
                calculated_value=DecimalHelper.create_decimal("0"),
                tolerance=DecimalHelper.create_decimal("0"),
                passed=False,
                error_message=error_msg,
                execution_time_ms=execution_time,
            )

    async def _test_statistical_validation(self) -> Dict[str, Any]:
        """Test statistical validation of backtest calculations."""
        try:
            self.logger.info("Running statistical validation tests")

            # Generate multiple test scenarios
            test_scenarios = []
            for i in range(10):  # Run 10 different scenarios
                mock_data = self.mock_data_generator.generate_backtest_data(
                    strategy_name=f"test_strategy_{i}", n_trades=50, win_rate=0.6
                )
                test_scenarios.append(mock_data)

            # Validate consistency across scenarios
            sharpe_ratios = []
            max_drawdowns = []
            win_rates = []

            for scenario in test_scenarios:
                # Calculate metrics for each scenario
                returns = [float(trade["pnl_percent"]) for trade in scenario["trades"]]
                equity_curve = [
                    float(trade["portfolio_value"]) for trade in scenario["trades"]
                ]

                sharpe = self._calculate_sharpe_ratio(returns)
                drawdown = self._calculate_max_drawdown(equity_curve)
                win_rate = self._calculate_win_rate(scenario["trades"])

                sharpe_ratios.append(float(sharpe))
                max_drawdowns.append(float(drawdown))
                win_rates.append(float(win_rate))

            # Statistical validation
            sharpe_mean = statistics.mean(sharpe_ratios)
            sharpe_std = (
                statistics.stdev(sharpe_ratios) if len(sharpe_ratios) > 1 else 0
            )

            drawdown_mean = statistics.mean(max_drawdowns)
            drawdown_std = (
                statistics.stdev(max_drawdowns) if len(max_drawdowns) > 1 else 0
            )

            win_rate_mean = statistics.mean(win_rates)
            win_rate_std = statistics.stdev(win_rates) if len(win_rates) > 1 else 0

            # Validate that calculations are consistent (low standard deviation relative to mean)
            sharpe_cv = (
                sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else float("inf")
            )
            drawdown_cv = (
                drawdown_std / abs(drawdown_mean)
                if drawdown_mean != 0
                else float("inf")
            )
            win_rate_cv = (
                win_rate_std / abs(win_rate_mean)
                if win_rate_mean != 0
                else float("inf")
            )

            # Pass if coefficient of variation is reasonable (< 0.5 for most metrics)
            passed = sharpe_cv < 1.0 and drawdown_cv < 1.0 and win_rate_cv < 0.3

            return {
                "passed": passed,
                "sharpe_ratio_stats": {
                    "mean": sharpe_mean,
                    "std": sharpe_std,
                    "cv": sharpe_cv,
                },
                "max_drawdown_stats": {
                    "mean": drawdown_mean,
                    "std": drawdown_std,
                    "cv": drawdown_cv,
                },
                "win_rate_stats": {
                    "mean": win_rate_mean,
                    "std": win_rate_std,
                    "cv": win_rate_cv,
                },
                "scenarios_tested": len(test_scenarios),
            }

        except Exception as e:
            error_msg = f"Error in statistical validation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "passed": False,
                "error": error_msg,
            }

    async def _test_known_cases_validation(self) -> List[Dict[str, Any]]:
        """Test known reference data comparison tests."""
        results = []

        for test_case in self.known_test_cases:
            try:
                case_name = test_case["name"]
                self.logger.info(f"Testing known case: {case_name}")

                if "trades" in test_case:
                    # Test total return and win rate
                    calculated_total_return = self._calculate_total_return(
                        test_case["trades"]
                    )
                    calculated_win_rate = self._calculate_win_rate(test_case["trades"])

                    total_return_match = DecimalHelper.compare_decimals(
                        test_case["expected_total_return"],
                        calculated_total_return,
                        DecimalHelper.PRECISION,
                    )

                    win_rate_match = DecimalHelper.compare_decimals(
                        test_case["expected_win_rate"],
                        calculated_win_rate,
                        DecimalHelper.create_decimal("0.01"),
                    )

                    passed = total_return_match and win_rate_match

                    results.append(
                        {
                            "case_name": case_name,
                            "passed": passed,
                            "total_return_match": total_return_match,
                            "win_rate_match": win_rate_match,
                            "expected_total_return": test_case["expected_total_return"],
                            "calculated_total_return": calculated_total_return,
                            "expected_win_rate": test_case["expected_win_rate"],
                            "calculated_win_rate": calculated_win_rate,
                        }
                    )

                elif "equity_curve" in test_case:
                    # Test max drawdown
                    equity_curve = [float(val) for val in test_case["equity_curve"]]
                    calculated_drawdown = self._calculate_max_drawdown(equity_curve)

                    drawdown_match = DecimalHelper.compare_decimals(
                        test_case["expected_max_drawdown"],
                        calculated_drawdown,
                        DecimalHelper.create_decimal("0.01"),
                    )

                    results.append(
                        {
                            "case_name": case_name,
                            "passed": drawdown_match,
                            "drawdown_match": drawdown_match,
                            "expected_max_drawdown": test_case["expected_max_drawdown"],
                            "calculated_max_drawdown": calculated_drawdown,
                        }
                    )

            except Exception as e:
                error_msg = f"Error testing known case {test_case.get('name', 'unknown')}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                results.append(
                    {
                        "case_name": test_case.get("name", "unknown"),
                        "passed": False,
                        "error": error_msg,
                    }
                )

        return results

    async def _test_extreme_market_conditions(self) -> Dict[str, Any]:
        """Test extreme market volatility scenarios using synthetic data."""
        try:
            self.logger.info("Running extreme market condition tests")

            extreme_scenarios = {}
            overall_passed = True

            for scenario in self.config.extreme_condition_scenarios:
                self.logger.info(f"Testing extreme scenario: {scenario}")

                # Generate synthetic extreme market data
                extreme_data = self._generate_extreme_market_data(scenario)

                # Test backtest calculations under extreme conditions
                scenario_result = await self._test_scenario_calculations(
                    scenario, extreme_data
                )
                extreme_scenarios[scenario] = scenario_result

                if not scenario_result.get("passed", False):
                    overall_passed = False

            return {
                "passed": overall_passed,
                "scenarios_tested": extreme_scenarios,
                "total_scenarios": len(self.config.extreme_condition_scenarios),
            }

        except Exception as e:
            error_msg = f"Error in extreme market condition testing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "passed": False,
                "error": error_msg,
            }

    async def _test_edge_case_handling(self) -> Dict[str, Any]:
        """Create edge case handling validation for backtest calculations."""
        try:
            self.logger.info("Running edge case handling validation")

            edge_cases = [
                {
                    "name": "empty_trades",
                    "trades": [],
                    "expected_behavior": "handle_gracefully",
                },
                {
                    "name": "single_trade",
                    "trades": [{"pnl": DecimalHelper.create_decimal("100.00")}],
                    "expected_behavior": "calculate_correctly",
                },
                {
                    "name": "all_zero_pnl",
                    "trades": [
                        {"pnl": DecimalHelper.create_decimal("0.00")},
                        {"pnl": DecimalHelper.create_decimal("0.00")},
                        {"pnl": DecimalHelper.create_decimal("0.00")},
                    ],
                    "expected_behavior": "handle_zero_values",
                },
                {
                    "name": "extreme_values",
                    "trades": [
                        {"pnl": DecimalHelper.create_decimal("999999999.99999999")},
                        {"pnl": DecimalHelper.create_decimal("-999999999.99999999")},
                    ],
                    "expected_behavior": "handle_extreme_values",
                },
                {
                    "name": "flat_equity_curve",
                    "equity_curve": [10000.0, 10000.0, 10000.0, 10000.0],
                    "expected_behavior": "zero_drawdown",
                },
                {
                    "name": "single_point_equity",
                    "equity_curve": [10000.0],
                    "expected_behavior": "zero_drawdown",
                },
            ]

            edge_case_results = []
            overall_passed = True

            for case in edge_cases:
                case_result = await self._test_edge_case(case)
                edge_case_results.append(case_result)

                if not case_result.get("passed", False):
                    overall_passed = False

            return {
                "passed": overall_passed,
                "edge_cases_tested": edge_case_results,
                "total_edge_cases": len(edge_cases),
            }

        except Exception as e:
            error_msg = f"Error in edge case handling validation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "passed": False,
                "error": error_msg,
            }

    def _generate_extreme_market_data(self, scenario: str) -> Dict[str, Any]:
        """Generate synthetic extreme market data for testing."""
        base_price = 10000.0

        if scenario == "high_volatility":
            # Generate high volatility price movements
            price_changes = []
            current_price = base_price

            for i in range(100):
                # High volatility: ±10% price swings
                volatility = (
                    0.1 * (1 if i % 2 == 0 else -1) * (0.5 + np.random.random())
                )
                price_change = current_price * volatility
                current_price += price_change
                price_changes.append(price_change)

            return {
                "scenario": scenario,
                "price_changes": price_changes,
                "equity_curve": [
                    base_price + sum(price_changes[: i + 1])
                    for i in range(len(price_changes))
                ],
                "returns": [change / base_price for change in price_changes],
            }

        elif scenario == "market_crash":
            # Generate market crash scenario: 50% drop over 10 periods
            crash_magnitude = -0.5
            crash_periods = 10

            price_changes = []
            current_price = base_price

            # Crash phase
            for i in range(crash_periods):
                change = current_price * (crash_magnitude / crash_periods)
                current_price += change
                price_changes.append(change)

            # Recovery phase (partial)
            recovery_periods = 20
            recovery_magnitude = 0.3  # 30% recovery
            for i in range(recovery_periods):
                change = abs(current_price) * (recovery_magnitude / recovery_periods)
                current_price += change
                price_changes.append(change)

            return {
                "scenario": scenario,
                "price_changes": price_changes,
                "equity_curve": [
                    base_price + sum(price_changes[: i + 1])
                    for i in range(len(price_changes))
                ],
                "returns": [change / base_price for change in price_changes],
            }

        elif scenario == "flash_crash":
            # Generate flash crash: sudden 20% drop and immediate recovery
            normal_periods = 50
            crash_drop = -0.2
            recovery_boost = 0.25

            price_changes = []
            current_price = base_price

            # Normal trading
            for i in range(normal_periods):
                change = current_price * 0.001 * (1 if i % 2 == 0 else -1)
                current_price += change
                price_changes.append(change)

            # Flash crash
            crash_change = current_price * crash_drop
            current_price += crash_change
            price_changes.append(crash_change)

            # Immediate recovery
            recovery_change = abs(crash_change) * recovery_boost
            current_price += recovery_change
            price_changes.append(recovery_change)

            # Continue normal trading
            for i in range(normal_periods):
                change = current_price * 0.001 * (1 if i % 2 == 0 else -1)
                current_price += change
                price_changes.append(change)

            return {
                "scenario": scenario,
                "price_changes": price_changes,
                "equity_curve": [
                    base_price + sum(price_changes[: i + 1])
                    for i in range(len(price_changes))
                ],
                "returns": [change / base_price for change in price_changes],
            }

        elif scenario == "low_liquidity":
            # Generate low liquidity scenario: large gaps between prices
            gap_periods = 20
            price_changes = []
            current_price = base_price

            for i in range(gap_periods):
                # Large price gaps (±5% jumps)
                gap_size = 0.05 * (1 if i % 3 == 0 else -1)
                change = current_price * gap_size
                current_price += change
                price_changes.append(change)

                # Add some small movements between gaps
                for j in range(3):
                    small_change = current_price * 0.001 * (1 if j % 2 == 0 else -1)
                    current_price += small_change
                    price_changes.append(small_change)

            return {
                "scenario": scenario,
                "price_changes": price_changes,
                "equity_curve": [
                    base_price + sum(price_changes[: i + 1])
                    for i in range(len(price_changes))
                ],
                "returns": [change / base_price for change in price_changes],
            }

        elif scenario == "extreme_pump":
            # Generate extreme pump scenario: 200% increase over short period
            pump_magnitude = 2.0  # 200% increase
            pump_periods = 15

            price_changes = []
            current_price = base_price

            # Normal trading before pump
            for i in range(30):
                change = current_price * 0.002 * (1 if i % 2 == 0 else -1)
                current_price += change
                price_changes.append(change)

            # Extreme pump
            for i in range(pump_periods):
                change = current_price * (pump_magnitude / pump_periods)
                current_price += change
                price_changes.append(change)

            # Correction after pump (50% drop)
            correction_periods = 10
            correction_magnitude = -0.5
            for i in range(correction_periods):
                change = current_price * (correction_magnitude / correction_periods)
                current_price += change
                price_changes.append(change)

            return {
                "scenario": scenario,
                "price_changes": price_changes,
                "equity_curve": [
                    base_price + sum(price_changes[: i + 1])
                    for i in range(len(price_changes))
                ],
                "returns": [change / base_price for change in price_changes],
            }

        # Default case
        return {
            "scenario": scenario,
            "price_changes": [0.0],
            "equity_curve": [base_price],
            "returns": [0.0],
        }

    async def _test_scenario_calculations(
        self, scenario: str, extreme_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test backtest calculations under extreme market conditions."""
        try:
            equity_curve = extreme_data["equity_curve"]
            returns = extreme_data["returns"]

            # Generate trades from the extreme data
            trades = []
            for i, ret in enumerate(returns):
                pnl = DecimalHelper.create_decimal(
                    str(ret * 10000)
                )  # Scale for meaningful PnL
                trades.append({"pnl": pnl})

            # Test calculations
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            win_rate = self._calculate_win_rate(trades)
            total_return = self._calculate_total_return(trades)

            # Validate that calculations don't produce invalid results
            calculations_valid = True
            validation_errors = []

            # Check for NaN or infinite values
            if not self._is_valid_decimal(sharpe_ratio):
                calculations_valid = False
                validation_errors.append(f"Invalid Sharpe ratio: {sharpe_ratio}")

            if not self._is_valid_decimal(max_drawdown):
                calculations_valid = False
                validation_errors.append(f"Invalid max drawdown: {max_drawdown}")

            if not self._is_valid_decimal(win_rate):
                calculations_valid = False
                validation_errors.append(f"Invalid win rate: {win_rate}")

            if not self._is_valid_decimal(total_return):
                calculations_valid = False
                validation_errors.append(f"Invalid total return: {total_return}")

            # Check reasonable bounds
            if max_drawdown < 0 or max_drawdown > 1:
                calculations_valid = False
                validation_errors.append(
                    f"Max drawdown out of bounds [0,1]: {max_drawdown}"
                )

            if win_rate < 0 or win_rate > 100:
                calculations_valid = False
                validation_errors.append(f"Win rate out of bounds [0,100]: {win_rate}")

            return {
                "scenario": scenario,
                "passed": calculations_valid,
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "total_return": float(total_return),
                "validation_errors": validation_errors,
                "data_points": len(equity_curve),
            }

        except Exception as e:
            error_msg = f"Error testing scenario {scenario}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "scenario": scenario,
                "passed": False,
                "error": error_msg,
            }

    async def _test_edge_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual edge case."""
        try:
            case_name = case["name"]
            expected_behavior = case["expected_behavior"]

            if "trades" in case:
                trades = case["trades"]

                # Test calculations with edge case data
                total_return = self._calculate_total_return(trades)
                win_rate = self._calculate_win_rate(trades)

                # Validate based on expected behavior
                if expected_behavior == "handle_gracefully":
                    # Should not crash and return reasonable defaults
                    passed = (
                        self._is_valid_decimal(total_return)
                        and self._is_valid_decimal(win_rate)
                        and total_return == DecimalHelper.create_decimal("0")
                        and win_rate == DecimalHelper.create_decimal("0")
                    )
                elif expected_behavior == "calculate_correctly":
                    # Should calculate correctly for single trade
                    passed = (
                        self._is_valid_decimal(total_return)
                        and self._is_valid_decimal(win_rate)
                        and total_return == trades[0]["pnl"]
                        and win_rate == DecimalHelper.create_decimal("100.00")
                    )
                elif expected_behavior == "handle_zero_values":
                    # Should handle zero values correctly
                    passed = (
                        self._is_valid_decimal(total_return)
                        and self._is_valid_decimal(win_rate)
                        and total_return == DecimalHelper.create_decimal("0")
                        and win_rate == DecimalHelper.create_decimal("0")
                    )
                elif expected_behavior == "handle_extreme_values":
                    # Should handle extreme values without overflow
                    passed = self._is_valid_decimal(
                        total_return
                    ) and self._is_valid_decimal(win_rate)
                else:
                    passed = False

                return {
                    "case_name": case_name,
                    "passed": passed,
                    "total_return": float(total_return),
                    "win_rate": float(win_rate),
                    "expected_behavior": expected_behavior,
                }

            elif "equity_curve" in case:
                equity_curve = case["equity_curve"]

                # Test max drawdown calculation
                max_drawdown = self._calculate_max_drawdown(equity_curve)

                if expected_behavior == "zero_drawdown":
                    passed = max_drawdown == DecimalHelper.create_decimal("0")
                else:
                    passed = self._is_valid_decimal(max_drawdown)

                return {
                    "case_name": case_name,
                    "passed": passed,
                    "max_drawdown": float(max_drawdown),
                    "expected_behavior": expected_behavior,
                }

            return {
                "case_name": case_name,
                "passed": False,
                "error": "Unknown edge case type",
            }

        except Exception as e:
            error_msg = (
                f"Error testing edge case {case.get('name', 'unknown')}: {str(e)}"
            )
            self.logger.error(error_msg, exc_info=True)
            return {
                "case_name": case.get("name", "unknown"),
                "passed": False,
                "error": error_msg,
            }

    def _is_valid_decimal(self, value: Decimal) -> bool:
        """Check if a Decimal value is valid (not NaN or infinite)."""
        try:
            # Check if the value is finite
            float_val = float(value)
            return not (np.isnan(float_val) or np.isinf(float_val))
        except (ValueError, OverflowError):
            return False

    def _calculate_total_return(self, trades: List[Dict]) -> Decimal:
        """Calculate total return using Decimal for precision."""
        if not trades:
            return DecimalHelper.create_decimal("0")

        total_pnl = DecimalHelper.create_decimal("0")
        for trade in trades:
            pnl = trade.get("pnl", DecimalHelper.create_decimal("0"))
            if not isinstance(pnl, Decimal):
                pnl = DecimalHelper.create_decimal(str(pnl))
            total_pnl += pnl

        return total_pnl

    def _calculate_sharpe_ratio(self, returns: List[float]) -> Decimal:
        """Calculate Sharpe ratio with proper financial precision."""
        if not returns or len(returns) < 2:
            return DecimalHelper.create_decimal("0")

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return DecimalHelper.create_decimal("0")

        # Assuming risk-free rate of 0 for simplicity
        sharpe = mean_return / std_return
        return DecimalHelper.create_decimal(str(sharpe))

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> Decimal:
        """Calculate maximum drawdown using Decimal precision."""
        if not equity_curve:
            return DecimalHelper.create_decimal("0")

        peak = DecimalHelper.create_decimal(str(equity_curve[0]))
        max_dd = DecimalHelper.create_decimal("0")

        for value in equity_curve:
            current_value = DecimalHelper.create_decimal(str(value))
            if current_value > peak:
                peak = current_value

            drawdown = (
                (peak - current_value) / peak
                if peak > 0
                else DecimalHelper.create_decimal("0")
            )
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def _calculate_win_rate(self, trades: List[Dict]) -> Decimal:
        """Calculate win rate as percentage of profitable trades."""
        if not trades:
            return DecimalHelper.create_decimal("0")

        winning_trades = 0
        for trade in trades:
            pnl = trade.get("pnl", DecimalHelper.create_decimal("0"))
            if not isinstance(pnl, Decimal):
                pnl = DecimalHelper.create_decimal(str(pnl))

            if pnl > DecimalHelper.create_decimal("0"):
                winning_trades += 1

        win_rate = (
            DecimalHelper.create_decimal(str(winning_trades))
            / DecimalHelper.create_decimal(str(len(trades)))
            * DecimalHelper.create_decimal("100")
        )

        return win_rate.quantize(DecimalHelper.PRECISION, rounding=ROUND_HALF_UP)
