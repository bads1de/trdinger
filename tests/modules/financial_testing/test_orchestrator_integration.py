"""
Integration test for FinancialCalculationTester with TestOrchestrator.
Verifies proper module registration and execution.
"""

import asyncio
import pytest
from decimal import Decimal

try:
    from ...orchestrator.test_orchestrator import TestOrchestrator, TestStatus
    from ...config.test_config import FinancialTestConfig, get_test_config
    from .financial_calculation_tester import FinancialCalculationTester
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from orchestrator.test_orchestrator import TestOrchestrator, TestStatus
    from config.test_config import FinancialTestConfig, get_test_config
    from modules.financial_testing.financial_calculation_tester import (
        FinancialCalculationTester,
    )


class TestFinancialCalculationTesterIntegration:
    """Integration tests for FinancialCalculationTester with TestOrchestrator."""

    @pytest.fixture
    def financial_config(self):
        """Create a test financial configuration."""
        return FinancialTestConfig(
            decimal_precision=Decimal("0.00000001"),
            rounding_mode="ROUND_HALF_UP",
            float_detection_paths=[],  # Empty to avoid scanning real files
            portfolio_test_scenarios=["test_scenario"],
        )

    @pytest.fixture
    def orchestrator(self):
        """Create a TestOrchestrator instance."""
        config = get_test_config()
        return TestOrchestrator(config=config)

    @pytest.fixture
    def financial_tester(self, financial_config):
        """Create a FinancialCalculationTester instance."""
        return FinancialCalculationTester(config=financial_config)

    def test_module_registration(self, orchestrator, financial_tester):
        """Test that FinancialCalculationTester can be registered with TestOrchestrator."""
        # Register the module
        orchestrator.register_test_module("financial_testing", financial_tester)

        # Verify registration
        assert orchestrator.test_modules["financial_testing"] is not None
        assert orchestrator.test_modules["financial_testing"] == financial_tester

    def test_module_unregistration(self, orchestrator, financial_tester):
        """Test that FinancialCalculationTester can be unregistered."""
        # Register first
        orchestrator.register_test_module("financial_testing", financial_tester)
        assert orchestrator.test_modules["financial_testing"] is not None

        # Unregister
        orchestrator.unregister_test_module("financial_testing")
        assert orchestrator.test_modules["financial_testing"] is None

    @pytest.mark.asyncio
    async def test_module_execution_through_orchestrator(
        self, orchestrator, financial_tester
    ):
        """Test that FinancialCalculationTester can be executed through TestOrchestrator."""
        # Register the module
        orchestrator.register_test_module("financial_testing", financial_tester)

        # Execute the module through orchestrator
        result = await orchestrator._execute_test_module(
            "financial_testing", financial_tester, timeout_seconds=60.0
        )

        # Verify result
        assert result.module_name == "financial_testing"
        assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
        assert result.tests_run > 0
        assert result.execution_time_seconds > 0
        assert result.start_time is not None
        assert result.end_time is not None

    @pytest.mark.asyncio
    async def test_direct_module_execution(self, financial_tester):
        """Test direct execution of FinancialCalculationTester."""
        result = await financial_tester.run_tests()

        # Verify basic result structure
        assert result.module_name == "financial_testing"
        assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
        assert result.tests_run > 0

        # Verify detailed results contain expected sections
        assert "decimal_enforcement" in result.detailed_results
        assert "precision_validation" in result.detailed_results
        assert "rounding_validation" in result.detailed_results
        assert "portfolio_calculation" in result.detailed_results
        assert "edge_case_calculation" in result.detailed_results
        assert "static_float_detection" in result.detailed_results

    def test_module_interface_compliance(self, financial_tester):
        """Test that FinancialCalculationTester properly implements TestModuleInterface."""
        # Test required methods exist
        assert hasattr(financial_tester, "run_tests")
        assert hasattr(financial_tester, "get_module_name")

        # Test method signatures
        assert callable(financial_tester.run_tests)
        assert callable(financial_tester.get_module_name)

        # Test module name
        assert financial_tester.get_module_name() == "financial_testing"

    @pytest.mark.asyncio
    async def test_error_handling_in_orchestrator(self, orchestrator):
        """Test error handling when module execution fails."""

        # Create a mock module that will fail
        class FailingModule:
            def get_module_name(self):
                return "failing_test"

            async def run_tests(self):
                raise Exception("Intentional test failure")

        failing_module = FailingModule()

        # Execute through orchestrator
        result = await orchestrator._execute_test_module(
            "failing_test", failing_module, timeout_seconds=10.0
        )

        # Verify error handling
        assert result.status == TestStatus.FAILED
        assert len(result.error_messages) > 0
        assert result.exception_details is not None

    @pytest.mark.asyncio
    async def test_timeout_handling_in_orchestrator(self, orchestrator):
        """Test timeout handling in orchestrator."""

        # Create a mock module that will timeout
        class TimeoutModule:
            def get_module_name(self):
                return "timeout_test"

            async def run_tests(self):
                await asyncio.sleep(10)  # Sleep longer than timeout
                return None

        timeout_module = TimeoutModule()

        # Execute with short timeout
        result = await orchestrator._execute_test_module(
            "timeout_test", timeout_module, timeout_seconds=1.0
        )

        # Verify timeout handling
        assert result.status == TestStatus.FAILED
        assert "timed out" in result.error_messages[0].lower()
        assert result.detailed_results.get("timeout") is True


if __name__ == "__main__":
    # Run integration tests directly
    async def run_integration_tests():
        print("Running FinancialCalculationTester integration tests...")

        # Create instances
        financial_config = FinancialTestConfig(
            decimal_precision=Decimal("0.00000001"),
            rounding_mode="ROUND_HALF_UP",
            float_detection_paths=[],
            portfolio_test_scenarios=[],
        )

        orchestrator = TestOrchestrator(config=get_test_config())
        financial_tester = FinancialCalculationTester(config=financial_config)

        # Test registration
        print("\n1. Testing module registration...")
        orchestrator.register_test_module("financial_testing", financial_tester)
        print("✓ Module registered successfully")

        # Test direct execution
        print("\n2. Testing direct module execution...")
        result = await financial_tester.run_tests()
        print(
            f"✓ Direct execution completed: {result.status}, {result.tests_passed}/{result.tests_run} passed"
        )

        # Test orchestrator execution
        print("\n3. Testing orchestrator execution...")
        orchestrator_result = await orchestrator._execute_test_module(
            "financial_testing", financial_tester, timeout_seconds=60.0
        )
        print(
            f"✓ Orchestrator execution completed: {orchestrator_result.status}, {orchestrator_result.tests_passed}/{orchestrator_result.tests_run} passed"
        )

        print("\nAll integration tests completed successfully!")

    asyncio.run(run_integration_tests())
