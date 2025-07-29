"""
Integration tests for ConcurrencyTester with TestOrchestrator.
Tests the integration between concurrency testing module and the test orchestrator.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch

from .concurrency_tester import ConcurrencyTester
from ...config.test_config import ConcurrencyTestConfig
from ...orchestrator.test_orchestrator import TestOrchestrator, TestStatus


class TestConcurrencyTesterOrchestrationIntegration:
    """Test ConcurrencyTester integration with TestOrchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ConcurrencyTestConfig(
            concurrent_operations_count=2,  # Reduced for faster testing
            race_condition_iterations=5,  # Reduced for faster testing
            deadlock_timeout_seconds=2,  # Reduced for faster testing
            circuit_breaker_test_scenarios=["test_scenario"],
        )
        self.tester = ConcurrencyTester(self.config)

    def test_module_interface_compliance(self):
        """Test that ConcurrencyTester properly implements TestModuleInterface."""
        # Test required interface methods
        assert hasattr(self.tester, "run_tests")
        assert hasattr(self.tester, "get_module_name")

        # Test module name
        assert self.tester.get_module_name() == "concurrency_testing"

        # Test that run_tests is async
        assert asyncio.iscoroutinefunction(self.tester.run_tests)

    @pytest.mark.asyncio
    async def test_basic_test_execution(self):
        """Test basic test execution without full integration."""
        # Run a minimal test to verify basic functionality
        result = await self.tester.run_tests()

        # Verify result structure
        assert result.module_name == "concurrency_testing"
        assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
        assert result.execution_time_seconds > 0
        assert result.tests_run >= 0
        assert result.tests_passed >= 0
        assert result.tests_failed >= 0
        assert isinstance(result.error_messages, list)
        assert isinstance(result.detailed_results, dict)

    @pytest.mark.asyncio
    async def test_concurrent_operations_minimal(self):
        """Test concurrent operations with minimal configuration."""
        results = await self.tester.test_concurrent_trading_operations()

        assert len(results) == self.config.concurrent_operations_count
        assert all(hasattr(r, "operation_id") for r in results)
        assert all(hasattr(r, "success") for r in results)
        assert all(hasattr(r, "data_integrity_maintained") for r in results)

    @pytest.mark.asyncio
    async def test_race_condition_detection_minimal(self):
        """Test race condition detection with minimal configuration."""
        result = await self.tester.test_race_condition_detection()

        assert hasattr(result, "test_scenario")
        assert hasattr(result, "test_passed")
        assert hasattr(result, "race_conditions_detected")
        assert hasattr(result, "execution_time_seconds")
        assert result.test_scenario == "concurrent_balance_updates"

    @pytest.mark.asyncio
    async def test_circuit_breaker_minimal(self):
        """Test circuit breaker behavior with minimal configuration."""
        results = await self.tester.test_circuit_breaker_behavior()

        assert len(results) == 2  # Two default scenarios
        assert all(hasattr(r, "test_scenario") for r in results)
        assert all(hasattr(r, "test_passed") for r in results)
        assert all(hasattr(r, "circuit_opened") for r in results)

    def test_mock_database_functionality(self):
        """Test that the mock database works correctly for testing."""
        db = self.tester.mock_db

        # Test basic operations
        account_id = "test_account"
        initial_balance = db.read_balance(account_id)
        assert initial_balance == 0

        # Test balance update
        from decimal import Decimal

        new_balance = Decimal("1000.00000000")
        success = db.update_balance(account_id, new_balance)
        assert success is True

        updated_balance = db.read_balance(account_id)
        assert updated_balance == new_balance

        # Test operation logging
        log = db.get_operation_log()
        assert len(log) >= 2  # At least read and update operations

    def test_error_handling_in_tests(self):
        """Test that the tester handles errors gracefully."""
        # Test with invalid configuration
        invalid_config = ConcurrencyTestConfig(
            concurrent_operations_count=0,
            race_condition_iterations=0,
            deadlock_timeout_seconds=0,
            circuit_breaker_test_scenarios=[],
        )

        invalid_tester = ConcurrencyTester(invalid_config)

        # Should not raise exceptions during initialization
        assert invalid_tester.get_module_name() == "concurrency_testing"
        assert invalid_tester.config.concurrent_operations_count == 0

    @pytest.mark.asyncio
    async def test_orchestrator_registration_simulation(self):
        """Simulate how the tester would be registered with TestOrchestrator."""
        # This simulates the registration process without actually running the full orchestrator

        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.register_test_module = Mock()

        # Simulate registration
        module_name = self.tester.get_module_name()
        mock_orchestrator.register_test_module(module_name, self.tester)

        # Verify registration was called
        mock_orchestrator.register_test_module.assert_called_once_with(
            "concurrency_testing", self.tester
        )

    def test_configuration_validation(self):
        """Test that configuration is properly validated and used."""
        # Test with custom configuration
        custom_config = ConcurrencyTestConfig(
            concurrent_operations_count=3,
            race_condition_iterations=10,
            deadlock_timeout_seconds=5,
            circuit_breaker_test_scenarios=["custom_scenario"],
        )

        custom_tester = ConcurrencyTester(custom_config)

        assert custom_tester.config.concurrent_operations_count == 3
        assert custom_tester.config.race_condition_iterations == 10
        assert custom_tester.config.deadlock_timeout_seconds == 5
        assert "custom_scenario" in custom_tester.config.circuit_breaker_test_scenarios

    @pytest.mark.asyncio
    async def test_detailed_results_structure(self):
        """Test that detailed results have the expected structure."""
        result = await self.tester.run_tests()

        # Check that all expected test categories are present
        expected_categories = [
            "concurrent_operations",
            "race_condition_detection",
            "deadlock_detection",
            "circuit_breaker",
            "api_rate_limiting",
            "data_consistency",
            "integration_test",
        ]

        for category in expected_categories:
            assert category in result.detailed_results, f"Missing category: {category}"

        # Check concurrent operations structure
        concurrent_ops = result.detailed_results["concurrent_operations"]
        assert "total_operations" in concurrent_ops
        assert "successful_operations" in concurrent_ops
        assert "failed_operations" in concurrent_ops
        assert "data_integrity_maintained" in concurrent_ops

        # Check race condition detection structure
        race_detection = result.detailed_results["race_condition_detection"]
        assert "test_passed" in race_detection
        assert "race_conditions_detected" in race_detection
        assert "data_consistency_violations" in race_detection


if __name__ == "__main__":
    pytest.main([__file__])
