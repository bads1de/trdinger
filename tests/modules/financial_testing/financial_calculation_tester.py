"""
Financial Calculation Testing Module for comprehensive testing framework.
Tests Decimal type enforcement, precision validation, and ROUND_HALF_UP rounding.
"""

import ast
import asyncio
import inspect
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN, ROUND_UP, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import importlib.util
import sys

try:
    from ...orchestrator.test_orchestrator import (
        TestModuleInterface,
        TestModuleResult,
        TestStatus,
    )
    from ...config.test_config import TestConfig, FinancialTestConfig
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
    from config.test_config import TestConfig, FinancialTestConfig
    from utils.test_utilities import TestLogger, DecimalHelper, MockDataGenerator


@dataclass
class DecimalEnforcementResult:
    """Result from Decimal type enforcement validation."""

    module_path: str
    decimal_violations: List[Dict[str, Any]]
    total_financial_operations: int
    decimal_compliance_rate: Decimal
    test_passed: bool
    execution_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class PrecisionValidationResult:
    """Result from 8-digit precision validation tests."""

    test_name: str
    input_value: Union[str, Decimal]
    expected_precision: int
    actual_precision: int
    precision_correct: bool
    quantized_value: Decimal
    execution_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class RoundingValidationResult:
    """Result from ROUND_HALF_UP rounding verification tests."""

    test_case: str
    input_value: Decimal
    expected_result: Decimal
    actual_result: Decimal
    rounding_mode_used: str
    rounding_correct: bool
    execution_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class FloatDetectionResult:
    """Result from static code analysis for float usage detection."""

    file_path: str
    line_number: int
    column_number: int
    float_usage_type: str
    code_snippet: str
    severity: str  # 'critical', 'warning', 'info'
    suggestion: str


@dataclass
class PortfolioCalculationResult:
    """Result from portfolio value calculation accuracy tests."""

    test_scenario: str
    input_positions: List[Dict[str, Any]]
    expected_portfolio_value: Decimal
    calculated_portfolio_value: Decimal
    calculation_accurate: bool
    precision_maintained: bool
    execution_time_seconds: float
    detailed_breakdown: Dict[str, Any]
    error_message: Optional[str] = None


class FinancialCalculationTester(TestModuleInterface):
    """
    Financial Calculation Testing Module implementing TestModuleInterface.

    Tests Decimal type enforcement, 8-digit precision validation, and ROUND_HALF_UP rounding.
    Implements requirements 4.1, 4.2, 4.3.
    """

    def __init__(self, config: FinancialTestConfig = None):
        self.config = config
        self.logger = TestLogger("financial_calculation_tester", "INFO")
        self.decimal_helper = DecimalHelper()
        self.test_results: List[
            Union[
                DecimalEnforcementResult,
                PrecisionValidationResult,
                RoundingValidationResult,
            ]
        ] = []

        # Financial calculation precision settings
        self.required_precision = (
            self.config.decimal_precision if self.config else Decimal("0.00000001")
        )
        self.required_rounding_mode = (
            getattr(self.config, "rounding_mode", "ROUND_HALF_UP")
            if self.config
            else "ROUND_HALF_UP"
        )

        # Paths to scan for float usage
        self.float_detection_paths = (
            self.config.float_detection_paths
            if self.config
            else [
                "backend/app/core",
                "backend/app/api",
                "backend/models",
                "backend/app/services",
            ]
        )

        self.logger.info(
            f"FinancialCalculationTester initialized with precision: {self.required_precision}, "
            f"rounding: {self.required_rounding_mode}"
        )

    def get_module_name(self) -> str:
        """Get the name of this test module."""
        return "financial_testing"

    def _get_decimal_places(self, decimal_value: Decimal) -> int:
        """Get the number of decimal places in a Decimal value."""
        return -decimal_value.as_tuple().exponent

    def _analyze_python_file_for_floats(
        self, file_path: Path
    ) -> List[FloatDetectionResult]:
        """Analyze a Python file for float usage in financial contexts."""
        float_detections = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))

            # Financial context keywords that indicate financial calculations
            financial_keywords = {
                "price",
                "amount",
                "value",
                "balance",
                "pnl",
                "profit",
                "loss",
                "portfolio",
                "position",
                "trade",
                "order",
                "fee",
                "commission",
                "sharpe",
                "drawdown",
                "return",
                "yield",
                "interest",
                "rate",
            }

            class FloatVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.line_numbers = content.split("\n")

                def visit_Num(self, node):
                    # For Python < 3.8
                    if isinstance(node.n, float):
                        self._check_financial_context(node)
                    self.generic_visit(node)

                def visit_Constant(self, node):
                    # For Python >= 3.8
                    if isinstance(node.value, float):
                        self._check_financial_context(node)
                    self.generic_visit(node)

                def visit_Name(self, node):
                    # Check for float() calls
                    if node.id == "float":
                        self._check_financial_context(
                            node, usage_type="float_conversion"
                        )
                    self.generic_visit(node)

                def visit_Call(self, node):
                    # Check for float() function calls
                    if isinstance(node.func, ast.Name) and node.func.id == "float":
                        self._check_financial_context(
                            node, usage_type="float_function_call"
                        )
                    self.generic_visit(node)

                def _check_financial_context(self, node, usage_type="float_literal"):
                    line_num = node.lineno
                    col_num = node.col_offset

                    # Get surrounding context (3 lines before and after)
                    start_line = max(0, line_num - 4)
                    end_line = min(len(self.line_numbers), line_num + 3)
                    context_lines = self.line_numbers[start_line:end_line]
                    context = "\n".join(context_lines)

                    # Check if any financial keywords are in the context
                    context_lower = context.lower()
                    has_financial_context = any(
                        keyword in context_lower for keyword in financial_keywords
                    )

                    # Get the specific line
                    code_snippet = (
                        self.line_numbers[line_num - 1]
                        if line_num <= len(self.line_numbers)
                        else ""
                    )

                    # Determine severity
                    if has_financial_context:
                        severity = "critical"
                        suggestion = f"Replace float with Decimal type for financial calculation on line {line_num}"
                    elif "test" in str(file_path).lower():
                        severity = "warning"
                        suggestion = f"Consider using Decimal for consistency in test on line {line_num}"
                    else:
                        severity = "info"
                        suggestion = f"Float usage detected on line {line_num} - verify if financial calculation"

                    float_detections.append(
                        FloatDetectionResult(
                            file_path=str(file_path),
                            line_number=line_num,
                            column_number=col_num,
                            float_usage_type=usage_type,
                            code_snippet=code_snippet.strip(),
                            severity=severity,
                            suggestion=suggestion,
                        )
                    )

            visitor = FloatVisitor()
            visitor.visit(tree)

        except Exception as e:
            self.logger.error(f"Failed to analyze file {file_path}: {e}")

        return float_detections

    async def test_decimal_type_enforcement(self) -> DecimalEnforcementResult:
        """
        Test Decimal type enforcement validation across all financial code.

        Returns:
            DecimalEnforcementResult with enforcement validation details
        """
        start_time = time.time()

        try:
            self.logger.info("Testing Decimal type enforcement across financial code")

            all_violations = []
            total_operations = 0

            # Scan configured paths for float usage
            for path_str in self.float_detection_paths:
                path = Path(path_str)
                if not path.exists():
                    self.logger.warning(f"Path does not exist: {path}")
                    continue

                if path.is_file() and path.suffix == ".py":
                    violations = self._analyze_python_file_for_floats(path)
                    all_violations.extend(violations)
                    total_operations += 1
                elif path.is_dir():
                    for py_file in path.rglob("*.py"):
                        violations = self._analyze_python_file_for_floats(py_file)
                        all_violations.extend(violations)
                        total_operations += 1

            # Filter critical violations (financial context)
            critical_violations = [
                v for v in all_violations if v.severity == "critical"
            ]

            # Calculate compliance rate
            if total_operations > 0:
                compliance_rate = self.decimal_helper.create_decimal(
                    str(
                        max(
                            0,
                            (total_operations - len(critical_violations))
                            / total_operations,
                        )
                    )
                )
            else:
                compliance_rate = self.decimal_helper.create_decimal("1.0")

            test_passed = len(critical_violations) == 0
            execution_time = time.time() - start_time

            # Convert violations to dict format
            violation_dicts = []
            for violation in all_violations:
                violation_dicts.append(
                    {
                        "file_path": violation.file_path,
                        "line_number": violation.line_number,
                        "column_number": violation.column_number,
                        "usage_type": violation.float_usage_type,
                        "code_snippet": violation.code_snippet,
                        "severity": violation.severity,
                        "suggestion": violation.suggestion,
                    }
                )

            result = DecimalEnforcementResult(
                module_path="financial_code_analysis",
                decimal_violations=violation_dicts,
                total_financial_operations=total_operations,
                decimal_compliance_rate=compliance_rate,
                test_passed=test_passed,
                execution_time_seconds=execution_time,
            )

            self.logger.info(
                f"Decimal enforcement test completed: {len(critical_violations)} critical violations found, "
                f"compliance rate: {compliance_rate}"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Decimal enforcement test failed: {str(e)}"
            self.logger.error(error_msg)

            return DecimalEnforcementResult(
                module_path="financial_code_analysis",
                decimal_violations=[],
                total_financial_operations=0,
                decimal_compliance_rate=Decimal("0"),
                test_passed=False,
                execution_time_seconds=execution_time,
                error_message=error_msg,
            )

    async def test_8_digit_precision_validation(
        self,
    ) -> List[PrecisionValidationResult]:
        """
        Test 8-digit precision validation for cryptocurrency calculations.

        Returns:
            List of PrecisionValidationResult with precision validation details
        """
        self.logger.info("Testing 8-digit precision validation")

        # Test cases for 8-digit precision
        test_cases = [
            ("0.12345678", "exact_8_digits"),
            ("1.23456789", "more_than_8_digits"),
            ("0.1234567", "less_than_8_digits"),
            ("123.45678901", "large_number_with_precision"),
            ("0.00000001", "minimum_precision"),
            ("99999999.99999999", "maximum_precision"),
            ("0.123456785", "rounding_test_up"),
            ("0.123456784", "rounding_test_down"),
        ]

        results = []

        for input_str, test_name in test_cases:
            start_time = time.time()

            try:
                # Create Decimal with input
                input_decimal = Decimal(input_str)

                # Quantize to 8 decimal places with ROUND_HALF_UP
                quantized = input_decimal.quantize(
                    self.required_precision, rounding=ROUND_HALF_UP
                )

                # Check precision
                actual_precision = self._get_decimal_places(quantized)
                expected_precision = 8

                # For values that naturally have fewer decimal places, that's acceptable
                precision_correct = actual_precision <= expected_precision

                execution_time = time.time() - start_time

                result = PrecisionValidationResult(
                    test_name=test_name,
                    input_value=input_str,
                    expected_precision=expected_precision,
                    actual_precision=actual_precision,
                    precision_correct=precision_correct,
                    quantized_value=quantized,
                    execution_time_seconds=execution_time,
                )

                results.append(result)

                self.logger.debug(
                    f"Precision test '{test_name}': {input_str} -> {quantized} "
                    f"(precision: {actual_precision}, correct: {precision_correct})"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Precision validation failed for {test_name}: {str(e)}"
                self.logger.error(error_msg)

                result = PrecisionValidationResult(
                    test_name=test_name,
                    input_value=input_str,
                    expected_precision=8,
                    actual_precision=0,
                    precision_correct=False,
                    quantized_value=Decimal("0"),
                    execution_time_seconds=execution_time,
                    error_message=error_msg,
                )

                results.append(result)

        self.logger.info(
            f"8-digit precision validation completed: {len(results)} tests run"
        )
        return results

    async def test_round_half_up_verification(self) -> List[RoundingValidationResult]:
        """
        Test ROUND_HALF_UP rounding verification for financial calculations.

        Returns:
            List of RoundingValidationResult with rounding validation details
        """
        self.logger.info("Testing ROUND_HALF_UP rounding verification")

        # Test cases for ROUND_HALF_UP rounding
        test_cases = [
            ("0.123456785", "0.12345679", "round_half_up_case"),
            ("0.123456775", "0.12345678", "round_half_up_case_2"),
            ("0.123456784", "0.12345678", "round_down_case"),
            ("0.123456786", "0.12345679", "round_up_case"),
            ("1.999999995", "2.00000000", "large_round_up"),
            ("0.000000005", "0.00000001", "small_round_up"),
            ("0.000000004", "0.00000000", "small_round_down"),
            ("123.456789125", "123.45678913", "integer_part_rounding"),
        ]

        results = []

        for input_str, expected_str, test_case in test_cases:
            start_time = time.time()

            try:
                input_value = Decimal(input_str)
                expected_result = Decimal(expected_str)

                # Test ROUND_HALF_UP
                actual_result = input_value.quantize(
                    self.required_precision, rounding=ROUND_HALF_UP
                )
                rounding_correct = actual_result == expected_result

                execution_time = time.time() - start_time

                result = RoundingValidationResult(
                    test_case=test_case,
                    input_value=input_value,
                    expected_result=expected_result,
                    actual_result=actual_result,
                    rounding_mode_used="ROUND_HALF_UP",
                    rounding_correct=rounding_correct,
                    execution_time_seconds=execution_time,
                )

                results.append(result)

                self.logger.debug(
                    f"Rounding test '{test_case}': {input_value} -> {actual_result} "
                    f"(expected: {expected_result}, correct: {rounding_correct})"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Rounding validation failed for {test_case}: {str(e)}"
                self.logger.error(error_msg)

                result = RoundingValidationResult(
                    test_case=test_case,
                    input_value=Decimal(input_str),
                    expected_result=Decimal(expected_str),
                    actual_result=Decimal("0"),
                    rounding_mode_used="ROUND_HALF_UP",
                    rounding_correct=False,
                    execution_time_seconds=execution_time,
                    error_message=error_msg,
                )

                results.append(result)

        self.logger.info(
            f"ROUND_HALF_UP verification completed: {len(results)} tests run"
        )
        return results

    def _calculate_portfolio_value_with_decimal(
        self, positions: List[Dict[str, Any]]
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Calculate portfolio value using proper Decimal arithmetic.

        Args:
            positions: List of position dictionaries with symbol, quantity, price

        Returns:
            Tuple of (total_value, detailed_breakdown)
        """
        total_value = Decimal("0.00000000")
        breakdown = {
            "positions": [],
            "total_positions": len(positions),
            "calculation_method": "decimal_arithmetic",
        }

        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            quantity = self.decimal_helper.create_decimal(position.get("quantity", 0))
            price = self.decimal_helper.create_decimal(position.get("price", 0))

            # Calculate position value and quantize to required precision
            position_value = (quantity * price).quantize(
                self.required_precision, rounding=ROUND_HALF_UP
            )
            total_value += position_value

            breakdown["positions"].append(
                {
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "value": position_value,
                }
            )

        # Quantize final total to required precision
        total_value = total_value.quantize(
            self.required_precision, rounding=ROUND_HALF_UP
        )

        return total_value, breakdown

    async def test_portfolio_calculation_accuracy(
        self,
    ) -> List[PortfolioCalculationResult]:
        """
        Test portfolio value calculation accuracy with known results.

        Returns:
            List of PortfolioCalculationResult with calculation accuracy details
        """
        self.logger.info("Testing portfolio calculation accuracy")

        # Test scenarios with known expected results
        test_scenarios = [
            {
                "name": "single_position_btc",
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "quantity": "1.50000000",
                        "price": "50000.12345678",
                    }
                ],
                "expected_value": "75000.18518517",
            },
            {
                "name": "multiple_positions",
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "quantity": "0.50000000",
                        "price": "50000.00000000",
                    },
                    {
                        "symbol": "ETHUSDT",
                        "quantity": "10.00000000",
                        "price": "3000.12345678",
                    },
                    {
                        "symbol": "ADAUSDT",
                        "quantity": "1000.00000000",
                        "price": "0.45678901",
                    },
                ],
                "expected_value": "55458.02357780",
            },
            {"name": "zero_balance", "positions": [], "expected_value": "0.00000000"},
            {
                "name": "small_amounts",
                "positions": [
                    {
                        "symbol": "DOGEUSDT",
                        "quantity": "100000.00000000",
                        "price": "0.00012345",
                    }
                ],
                "expected_value": "12.34500000",
            },
            {
                "name": "extreme_precision",
                "positions": [
                    {
                        "symbol": "TESTUSDT",
                        "quantity": "0.00000001",
                        "price": "99999999.99999999",
                    }
                ],
                "expected_value": "0.99999999",
            },
        ]

        results = []

        for scenario in test_scenarios:
            start_time = time.time()

            try:
                scenario_name = scenario["name"]
                positions = scenario["positions"]
                expected_value = self.decimal_helper.create_decimal(
                    scenario["expected_value"]
                )

                # Calculate portfolio value
                calculated_value, breakdown = (
                    self._calculate_portfolio_value_with_decimal(positions)
                )

                # Check accuracy
                calculation_accurate = self.decimal_helper.compare_decimals(
                    calculated_value, expected_value, tolerance=Decimal("0.00000001")
                )

                # Check precision maintenance
                precision_maintained = self._get_decimal_places(calculated_value) <= 8

                execution_time = time.time() - start_time

                result = PortfolioCalculationResult(
                    test_scenario=scenario_name,
                    input_positions=positions,
                    expected_portfolio_value=expected_value,
                    calculated_portfolio_value=calculated_value,
                    calculation_accurate=calculation_accurate,
                    precision_maintained=precision_maintained,
                    execution_time_seconds=execution_time,
                    detailed_breakdown=breakdown,
                )

                results.append(result)

                self.logger.debug(
                    f"Portfolio test '{scenario_name}': calculated={calculated_value}, "
                    f"expected={expected_value}, accurate={calculation_accurate}"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Portfolio calculation test failed for {scenario['name']}: {str(e)}"
                self.logger.error(error_msg)

                result = PortfolioCalculationResult(
                    test_scenario=scenario["name"],
                    input_positions=scenario["positions"],
                    expected_portfolio_value=Decimal("0"),
                    calculated_portfolio_value=Decimal("0"),
                    calculation_accurate=False,
                    precision_maintained=False,
                    execution_time_seconds=execution_time,
                    detailed_breakdown={},
                    error_message=error_msg,
                )

                results.append(result)

        self.logger.info(
            f"Portfolio calculation accuracy tests completed: {len(results)} scenarios tested"
        )
        return results

    async def run_tests(self) -> TestModuleResult:
        """
        Run all financial calculation tests and return comprehensive results.

        Returns:
            TestModuleResult with all test outcomes
        """
        start_time = time.time()
        self.logger.info("Starting comprehensive financial calculation tests")

        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        error_messages = []
        detailed_results = {}

        try:
            # Test 1: Decimal type enforcement
            self.logger.info("Running Decimal type enforcement tests")
            decimal_enforcement_result = await self.test_decimal_type_enforcement()
            tests_run += 1
            if decimal_enforcement_result.test_passed:
                tests_passed += 1
            else:
                tests_failed += 1
                if decimal_enforcement_result.error_message:
                    error_messages.append(decimal_enforcement_result.error_message)

            detailed_results["decimal_enforcement"] = {
                "compliance_rate": float(
                    decimal_enforcement_result.decimal_compliance_rate
                ),
                "violations_count": len(decimal_enforcement_result.decimal_violations),
                "critical_violations": len(
                    [
                        v
                        for v in decimal_enforcement_result.decimal_violations
                        if v["severity"] == "critical"
                    ]
                ),
                "test_passed": decimal_enforcement_result.test_passed,
            }

            # Test 2: 8-digit precision validation
            self.logger.info("Running 8-digit precision validation tests")
            precision_results = await self.test_8_digit_precision_validation()
            tests_run += len(precision_results)
            precision_passed = sum(1 for r in precision_results if r.precision_correct)
            precision_failed = len(precision_results) - precision_passed
            tests_passed += precision_passed
            tests_failed += precision_failed

            for result in precision_results:
                if result.error_message:
                    error_messages.append(result.error_message)

            detailed_results["precision_validation"] = {
                "total_tests": len(precision_results),
                "tests_passed": precision_passed,
                "tests_failed": precision_failed,
                "test_results": [
                    {
                        "test_name": r.test_name,
                        "input_value": str(r.input_value),
                        "precision_correct": r.precision_correct,
                        "actual_precision": r.actual_precision,
                    }
                    for r in precision_results
                ],
            }

            # Test 3: ROUND_HALF_UP verification
            self.logger.info("Running ROUND_HALF_UP rounding verification tests")
            rounding_results = await self.test_round_half_up_verification()
            tests_run += len(rounding_results)
            rounding_passed = sum(1 for r in rounding_results if r.rounding_correct)
            rounding_failed = len(rounding_results) - rounding_passed
            tests_passed += rounding_passed
            tests_failed += rounding_failed

            for result in rounding_results:
                if result.error_message:
                    error_messages.append(result.error_message)

            detailed_results["rounding_validation"] = {
                "total_tests": len(rounding_results),
                "tests_passed": rounding_passed,
                "tests_failed": rounding_failed,
                "test_results": [
                    {
                        "test_case": r.test_case,
                        "input_value": str(r.input_value),
                        "expected_result": str(r.expected_result),
                        "actual_result": str(r.actual_result),
                        "rounding_correct": r.rounding_correct,
                    }
                    for r in rounding_results
                ],
            }

            # Test 4: Portfolio calculation accuracy
            self.logger.info("Running portfolio calculation accuracy tests")
            portfolio_results = await self.test_portfolio_calculation_accuracy()
            tests_run += len(portfolio_results)
            portfolio_passed = sum(
                1
                for r in portfolio_results
                if r.calculation_accurate and r.precision_maintained
            )
            portfolio_failed = len(portfolio_results) - portfolio_passed
            tests_passed += portfolio_passed
            tests_failed += portfolio_failed

            for result in portfolio_results:
                if result.error_message:
                    error_messages.append(result.error_message)

            detailed_results["portfolio_calculation"] = {
                "total_tests": len(portfolio_results),
                "tests_passed": portfolio_passed,
                "tests_failed": portfolio_failed,
                "test_results": [
                    {
                        "test_scenario": r.test_scenario,
                        "expected_value": str(r.expected_portfolio_value),
                        "calculated_value": str(r.calculated_portfolio_value),
                        "calculation_accurate": r.calculation_accurate,
                        "precision_maintained": r.precision_maintained,
                    }
                    for r in portfolio_results
                ],
            }

            # Test 5: Financial calculation edge cases
            self.logger.info("Running financial calculation edge case tests")
            edge_case_results = await self.test_financial_calculation_edge_cases()
            tests_run += len(edge_case_results)
            edge_case_passed = sum(
                1
                for r in edge_case_results
                if r.calculation_accurate and r.precision_maintained
            )
            edge_case_failed = len(edge_case_results) - edge_case_passed
            tests_passed += edge_case_passed
            tests_failed += edge_case_failed

            for result in edge_case_results:
                if result.error_message:
                    error_messages.append(result.error_message)

            detailed_results["edge_case_calculation"] = {
                "total_tests": len(edge_case_results),
                "tests_passed": edge_case_passed,
                "tests_failed": edge_case_failed,
                "test_results": [
                    {
                        "test_scenario": r.test_scenario,
                        "expected_value": str(r.expected_portfolio_value),
                        "calculated_value": str(r.calculated_portfolio_value),
                        "calculation_accurate": r.calculation_accurate,
                        "precision_maintained": r.precision_maintained,
                    }
                    for r in edge_case_results
                ],
            }

            # Test 6: Comprehensive static float detection
            self.logger.info("Running comprehensive static float detection")
            float_detection_result = (
                await self.test_static_float_detection_comprehensive()
            )
            tests_run += 1
            if (
                float_detection_result["scan_successful"]
                and float_detection_result["scan_summary"]["critical_violations"] == 0
            ):
                tests_passed += 1
            else:
                tests_failed += 1
                if "error_message" in float_detection_result:
                    error_messages.append(float_detection_result["error_message"])

            detailed_results["static_float_detection"] = float_detection_result

            # Determine overall status
            if tests_failed == 0:
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
                end_time=datetime.now(),
            )

            self.logger.info(
                f"Financial calculation tests completed: {tests_passed}/{tests_run} passed, "
                f"execution time: {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Financial calculation tests failed with exception: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return TestModuleResult(
                module_name=self.get_module_name(),
                status=TestStatus.FAILED,
                execution_time_seconds=execution_time,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_run - tests_passed + 1,  # +1 for the exception
                tests_skipped=tests_skipped,
                error_messages=error_messages + [error_msg],
                detailed_results=detailed_results,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                exception_details=traceback.format_exc(),
            )

    async def test_financial_calculation_edge_cases(
        self,
    ) -> List[PortfolioCalculationResult]:
        """
        Test comprehensive financial calculation edge cases.

        Returns:
            List of PortfolioCalculationResult with edge case test details
        """
        self.logger.info("Testing financial calculation edge cases")

        # Edge case scenarios
        edge_case_scenarios = [
            {
                "name": "negative_balance_scenario",
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "quantity": "-0.50000000",
                        "price": "50000.00000000",
                    }
                ],
                "expected_value": "-25000.00000000",
            },
            {
                "name": "mixed_positive_negative",
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "quantity": "1.00000000",
                        "price": "50000.00000000",
                    },
                    {
                        "symbol": "ETHUSDT",
                        "quantity": "-0.50000000",
                        "price": "3000.00000000",
                    },
                ],
                "expected_value": "48500.00000000",
            },
            {
                "name": "very_large_numbers",
                "positions": [
                    {
                        "symbol": "TESTUSDT",
                        "quantity": "999999999.99999999",
                        "price": "0.00000001",
                    }
                ],
                "expected_value": "9.99999999",
            },
            {
                "name": "very_small_numbers",
                "positions": [
                    {
                        "symbol": "MICROUSDT",
                        "quantity": "0.00000001",
                        "price": "0.00000001",
                    }
                ],
                "expected_value": "0.00000000",  # Should round to 0 due to precision limits
            },
            {
                "name": "rounding_boundary_test",
                "positions": [
                    {
                        "symbol": "ROUNDUSDT",
                        "quantity": "1.00000000",
                        "price": "0.123456785",
                    }
                ],
                "expected_value": "0.12345679",  # Should round up due to ROUND_HALF_UP
            },
            {
                "name": "precision_accumulation_test",
                "positions": [
                    {
                        "symbol": "PREC1USDT",
                        "quantity": "0.33333333",
                        "price": "3.00000000",
                    },
                    {
                        "symbol": "PREC2USDT",
                        "quantity": "0.33333333",
                        "price": "3.00000000",
                    },
                    {
                        "symbol": "PREC3USDT",
                        "quantity": "0.33333334",
                        "price": "3.00000000",
                    },
                ],
                "expected_value": "3.00000000",
            },
            {
                "name": "zero_price_scenario",
                "positions": [
                    {
                        "symbol": "ZEROUSDT",
                        "quantity": "1000000.00000000",
                        "price": "0.00000000",
                    }
                ],
                "expected_value": "0.00000000",
            },
            {
                "name": "zero_quantity_scenario",
                "positions": [
                    {
                        "symbol": "NOQUANTUSDT",
                        "quantity": "0.00000000",
                        "price": "99999999.99999999",
                    }
                ],
                "expected_value": "0.00000000",
            },
            {
                "name": "maximum_precision_stress_test",
                "positions": [
                    {
                        "symbol": "STRESS1USDT",
                        "quantity": "12345678.87654321",
                        "price": "0.12345678",
                    },
                    {
                        "symbol": "STRESS2USDT",
                        "quantity": "0.12345678",
                        "price": "87654321.12345678",
                    },
                ],
                "expected_value": "12345678.00000000",
            },
        ]

        results = []

        for scenario in edge_case_scenarios:
            start_time = time.time()

            try:
                scenario_name = scenario["name"]
                positions = scenario["positions"]
                expected_value = self.decimal_helper.create_decimal(
                    scenario["expected_value"]
                )

                # Calculate portfolio value
                calculated_value, breakdown = (
                    self._calculate_portfolio_value_with_decimal(positions)
                )

                # For edge cases, use a slightly more lenient tolerance
                tolerance = Decimal("0.00000001")
                calculation_accurate = self.decimal_helper.compare_decimals(
                    calculated_value, expected_value, tolerance=tolerance
                )

                # Check precision maintenance
                precision_maintained = self._get_decimal_places(calculated_value) <= 8

                execution_time = time.time() - start_time

                result = PortfolioCalculationResult(
                    test_scenario=scenario_name,
                    input_positions=positions,
                    expected_portfolio_value=expected_value,
                    calculated_portfolio_value=calculated_value,
                    calculation_accurate=calculation_accurate,
                    precision_maintained=precision_maintained,
                    execution_time_seconds=execution_time,
                    detailed_breakdown=breakdown,
                )

                results.append(result)

                self.logger.debug(
                    f"Edge case test '{scenario_name}': calculated={calculated_value}, "
                    f"expected={expected_value}, accurate={calculation_accurate}"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Edge case test failed for {scenario['name']}: {str(e)}"
                self.logger.error(error_msg)

                result = PortfolioCalculationResult(
                    test_scenario=scenario["name"],
                    input_positions=scenario["positions"],
                    expected_portfolio_value=Decimal("0"),
                    calculated_portfolio_value=Decimal("0"),
                    calculation_accurate=False,
                    precision_maintained=False,
                    execution_time_seconds=execution_time,
                    detailed_breakdown={},
                    error_message=error_msg,
                )

                results.append(result)

        self.logger.info(
            f"Financial calculation edge case tests completed: {len(results)} scenarios tested"
        )
        return results

    async def test_static_float_detection_comprehensive(self) -> Dict[str, Any]:
        """
        Comprehensive static code analysis to detect float usage in financial calculations.

        Returns:
            Dictionary with comprehensive float detection results
        """
        self.logger.info("Running comprehensive static float detection analysis")

        try:
            all_detections = []
            scanned_files = 0
            critical_violations = 0
            warning_violations = 0
            info_violations = 0

            # Scan all configured paths
            for path_str in self.float_detection_paths:
                path = Path(path_str)
                if not path.exists():
                    self.logger.warning(f"Configured path does not exist: {path}")
                    continue

                if path.is_file() and path.suffix == ".py":
                    detections = self._analyze_python_file_for_floats(path)
                    all_detections.extend(detections)
                    scanned_files += 1
                elif path.is_dir():
                    for py_file in path.rglob("*.py"):
                        try:
                            detections = self._analyze_python_file_for_floats(py_file)
                            all_detections.extend(detections)
                            scanned_files += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to scan file {py_file}: {e}")

            # Categorize violations by severity
            for detection in all_detections:
                if detection.severity == "critical":
                    critical_violations += 1
                elif detection.severity == "warning":
                    warning_violations += 1
                else:
                    info_violations += 1

            # Generate summary statistics
            total_violations = len(all_detections)
            compliance_score = 1.0 - (critical_violations / max(scanned_files, 1))

            # Group violations by file for better reporting
            violations_by_file = {}
            for detection in all_detections:
                file_path = detection.file_path
                if file_path not in violations_by_file:
                    violations_by_file[file_path] = []
                violations_by_file[file_path].append(
                    {
                        "line_number": detection.line_number,
                        "usage_type": detection.float_usage_type,
                        "code_snippet": detection.code_snippet,
                        "severity": detection.severity,
                        "suggestion": detection.suggestion,
                    }
                )

            # Generate recommendations
            recommendations = []
            if critical_violations > 0:
                recommendations.append(
                    f"CRITICAL: {critical_violations} float usage(s) detected in financial contexts. "
                    "Replace with Decimal type immediately."
                )
            if warning_violations > 0:
                recommendations.append(
                    f"WARNING: {warning_violations} potential float usage(s) in test or utility code. "
                    "Consider using Decimal for consistency."
                )
            if info_violations > 0:
                recommendations.append(
                    f"INFO: {info_violations} float usage(s) detected in non-financial contexts. "
                    "Review to ensure they are not used in financial calculations."
                )

            if total_violations == 0:
                recommendations.append(
                    "EXCELLENT: No float usage detected in financial code!"
                )

            result = {
                "scan_summary": {
                    "files_scanned": scanned_files,
                    "total_violations": total_violations,
                    "critical_violations": critical_violations,
                    "warning_violations": warning_violations,
                    "info_violations": info_violations,
                    "compliance_score": compliance_score,
                },
                "violations_by_file": violations_by_file,
                "recommendations": recommendations,
                "detailed_detections": [
                    {
                        "file_path": d.file_path,
                        "line_number": d.line_number,
                        "column_number": d.column_number,
                        "usage_type": d.float_usage_type,
                        "code_snippet": d.code_snippet,
                        "severity": d.severity,
                        "suggestion": d.suggestion,
                    }
                    for d in all_detections
                ],
                "scan_successful": True,
            }

            self.logger.info(
                f"Static float detection completed: {scanned_files} files scanned, "
                f"{total_violations} violations found ({critical_violations} critical)"
            )

            return result

        except Exception as e:
            error_msg = f"Static float detection analysis failed: {str(e)}"
            self.logger.error(error_msg)

            return {
                "scan_summary": {
                    "files_scanned": 0,
                    "total_violations": 0,
                    "critical_violations": 0,
                    "warning_violations": 0,
                    "info_violations": 0,
                    "compliance_score": 0.0,
                },
                "violations_by_file": {},
                "recommendations": [f"ERROR: {error_msg}"],
                "detailed_detections": [],
                "scan_successful": False,
                "error_message": error_msg,
            }
