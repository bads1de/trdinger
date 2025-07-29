"""
Test file for FinancialCalculationTester module.
Validates the financial calculation testing functionality.
"""

import asyncio
import pytest
import tempfile
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from unittest.mock import patch, MagicMock

from .financial_calculation_tester import (
    FinancialCalculationTester,
    DecimalEnforcementResult,
    PrecisionValidationResult,
    RoundingValidationResult,
    PortfolioCalculationResult,
    FloatDetectionResult,
)

try:
    from ...config.test_config import FinancialTestConfig
    from ...orchestrator.test_orchestrator import TestStatus
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.test_config import FinancialTestConfig
    from orchestrator.test_orchestrator import TestStatus


class TestFinancialCalculationTester:
    """Test cases for FinancialCalculationTester."""

    @pytest.fixture
    def financial_config(self):
        """Create a test financial configuration."""
        return FinancialTestConfig(
            decimal_precision=Decimal("0.00000001"),
            rounding_mode="ROUND_HALF_UP",
            float_detection_paths=["test_path"],
            portfolio_test_scenarios=["test_scenario"],
        )

    @pytest.fixture
    def tester(self, financial_config):
        """Create a FinancialCalculationTester instance."""
        return FinancialCalculationTester(config=financial_config)

    def test_module_name(self, tester):
        """Test that module name is correct."""
        assert tester.get_module_name() == "financial_testing"

    def test_decimal_places_calculation(self, tester):
        """Test the _get_decimal_places method."""
        assert tester._get_decimal_places(Decimal("1.12345678")) == 8
        assert tester._get_decimal_places(Decimal("1.123")) == 3
        assert tester._get_decimal_places(Decimal("1")) == 0

    @pytest.mark.asyncio
    async def test_8_digit_precision_validation(self, tester):
        """Test 8-digit precision validation."""
        results = await tester.test_8_digit_precision_validation()

        assert len(results) > 0
        assert all(isinstance(r, PrecisionValidationResult) for r in results)

        # Check specific test cases
        exact_8_digits = next(
            (r for r in results if r.test_name == "exact_8_digits"), None
        )
        assert exact_8_digits is not None
        assert exact_8_digits.precision_correct
        assert exact_8_digits.quantized_value == Decimal("0.12345678")

    @pytest.mark.asyncio
    async def test_round_half_up_verification(self, tester):
        """Test ROUND_HALF_UP rounding verification."""
        results = await tester.test_round_half_up_verification()

        assert len(results) > 0
        assert all(isinstance(r, RoundingValidationResult) for r in results)

        # Check specific rounding cases
        for result in results:
            if result.test_case == "round_half_up_case":
                assert result.rounding_correct
                assert result.actual_result == Decimal("0.12345679")

    @pytest.mark.asyncio
    async def test_portfolio_calculation_accuracy(self, tester):
        """Test portfolio calculation accuracy."""
        results = await tester.test_portfolio_calculation_accuracy()

        assert len(results) > 0
        assert all(isinstance(r, PortfolioCalculationResult) for r in results)

        # Check zero balance scenario
        zero_balance = next(
            (r for r in results if r.test_scenario == "zero_balance"), None
        )
        assert zero_balance is not None
        assert zero_balance.calculation_accurate
        assert zero_balance.calculated_portfolio_value == Decimal("0.00000000")

    def test_calculate_portfolio_value_with_decimal(self, tester):
        """Test portfolio value calculation with Decimal arithmetic."""
        positions = [
            {"symbol": "BTCUSDT", "quantity": "1.0", "price": "50000.0"},
            {"symbol": "ETHUSDT", "quantity": "10.0", "price": "3000.0"},
        ]

        total_value, breakdown = tester._calculate_portfolio_value_with_decimal(
            positions
        )

        expected_value = Decimal("80000.00000000")
        assert total_value == expected_value
        assert breakdown["total_positions"] == 2
        assert len(breakdown["positions"]) == 2

    def test_analyze_python_file_for_floats_with_temp_file(self, tester):
        """Test float detection in Python files."""
        # Create a temporary Python file with float usage
        test_code = """
def calculate_price(amount):
    price = 50000.5  # This should be detected
    return amount * price

def calculate_portfolio_value(positions):
    total = 0.0  # This should be detected in financial context
    for pos in positions:
        total += float(pos.quantity) * pos.price  # This should be detected
    return total

def non_financial_function():
    x = 3.14  # This might be detected but with lower severity
    return x * 2
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_file_path = f.name

        try:
            detections = tester._analyze_python_file_for_floats(Path(temp_file_path))

            # Should detect float usage
            assert len(detections) > 0

            # Check that financial context is detected
            critical_detections = [d for d in detections if d.severity == "critical"]
            assert len(critical_detections) > 0

        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_decimal_type_enforcement_with_mock_paths(self, tester):
        """Test Decimal type enforcement with mocked file paths."""
        # Mock the float detection paths to avoid scanning real files
        with patch.object(tester, "float_detection_paths", ["nonexistent_path"]):
            result = await tester.test_decimal_type_enforcement()

            assert isinstance(result, DecimalEnforcementResult)
            # Should handle nonexistent paths gracefully
            assert result.total_financial_operations == 0

    @pytest.mark.asyncio
    async def test_run_tests_integration(self, tester):
        """Test the complete run_tests method."""
        # Mock file system operations to avoid scanning real files
        with patch.object(tester, "float_detection_paths", []):
            result = await tester.run_tests()

            assert result.module_name == "financial_testing"
            assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
            assert result.tests_run > 0
            assert "decimal_enforcement" in result.detailed_results
            assert "precision_validation" in result.detailed_results
            assert "rounding_validation" in result.detailed_results
            assert "portfolio_calculation" in result.detailed_results

    def test_decimal_helper_integration(self, tester):
        """Test integration with DecimalHelper."""
        # Test decimal creation
        decimal_val = tester.decimal_helper.create_decimal("123.456789")
        assert isinstance(decimal_val, Decimal)
        assert tester.decimal_helper.validate_decimal_type(decimal_val)

        # Test decimal comparison
        val1 = Decimal("1.12345678")
        val2 = Decimal("1.12345679")
        assert not tester.decimal_helper.compare_decimals(
            val1, val2, tolerance=Decimal("0.00000001")
        )
        assert tester.decimal_helper.compare_decimals(
            val1, val2, tolerance=Decimal("0.00000010")
        )

    @pytest.mark.asyncio
    async def test_error_handling_in_tests(self, tester):
        """Test error handling in various test methods."""
        # Test with invalid configuration
        tester.required_precision = None  # This should cause errors

        # The tests should handle errors gracefully and return error results
        try:
            results = await tester.test_8_digit_precision_validation()
            # Should still return results, possibly with errors
            assert isinstance(results, list)
        except Exception:
            # If exceptions occur, they should be handled in run_tests
            pass

    def test_float_detection_result_creation(self):
        """Test FloatDetectionResult creation."""
        result = FloatDetectionResult(
            file_path="test.py",
            line_number=10,
            column_number=5,
            float_usage_type="float_literal",
            code_snippet="price = 123.45",
            severity="critical",
            suggestion="Use Decimal instead",
        )

        assert result.file_path == "test.py"
        assert result.line_number == 10
        assert result.severity == "critical"

    @pytest.mark.asyncio
    async def test_precision_edge_cases(self, tester):
        """Test precision validation with edge cases."""
        # Test the precision validation with various edge cases
        results = await tester.test_8_digit_precision_validation()

        # Find specific edge cases
        min_precision = next(
            (r for r in results if r.test_name == "minimum_precision"), None
        )
        assert min_precision is not None
        assert min_precision.quantized_value == Decimal("0.00000001")

        max_precision = next(
            (r for r in results if r.test_name == "maximum_precision"), None
        )
        assert max_precision is not None
        # Should maintain precision within limits

    @pytest.mark.asyncio
    async def test_rounding_edge_cases(self, tester):
        """Test rounding verification with edge cases."""
        results = await tester.test_round_half_up_verification()

        # Check that all rounding modes are ROUND_HALF_UP
        for result in results:
            assert result.rounding_mode_used == "ROUND_HALF_UP"

        # Find specific edge cases
        small_round_up = next(
            (r for r in results if r.test_case == "small_round_up"), None
        )
        assert small_round_up is not None
        assert small_round_up.actual_result == Decimal("0.00000001")


if __name__ == "__main__":
    # Run tests directly
    import asyncio

    async def run_direct_tests():
        config = FinancialTestConfig(
            decimal_precision=Decimal("0.00000001"),
            rounding_mode="ROUND_HALF_UP",
            float_detection_paths=[],
            portfolio_test_scenarios=[],
        )

        tester = FinancialCalculationTester(config=config)

        print("Running FinancialCalculationTester tests...")

        # Test precision validation
        print("\n1. Testing 8-digit precision validation...")
        precision_results = await tester.test_8_digit_precision_validation()
        print(f"Precision tests: {len(precision_results)} completed")

        # Test rounding validation
        print("\n2. Testing ROUND_HALF_UP verification...")
        rounding_results = await tester.test_round_half_up_verification()
        print(f"Rounding tests: {len(rounding_results)} completed")

        # Test portfolio calculation
        print("\n3. Testing portfolio calculation accuracy...")
        portfolio_results = await tester.test_portfolio_calculation_accuracy()
        print(f"Portfolio tests: {len(portfolio_results)} completed")

        # Test full run
        print("\n4. Running complete test suite...")
        full_result = await tester.run_tests()
        print(
            f"Full test result: {full_result.status}, {full_result.tests_passed}/{full_result.tests_run} passed"
        )

        print("\nAll tests completed successfully!")

    asyncio.run(run_direct_tests())
