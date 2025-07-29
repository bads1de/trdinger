"""
Unit tests for ConcurrencyTester module.
Tests the concurrent trading operations, race condition detection, and deadlock prevention.
"""

import asyncio
import pytest
import time
from decimal import Decimal
from unittest.mock import Mock, patch

from .concurrency_tester import (
    ConcurrencyTester,
    MockTradingDatabase,
    MockCircuitBreaker,
    ConcurrentOperationResult,
    RaceConditionResult,
    DeadlockDetectionResult,
    CircuitBreakerResult,
)
from ...config.test_config import ConcurrencyTestConfig
from ...orchestrator.test_orchestrator import TestStatus


class TestMockTradingDatabase:
    """Test the MockTradingDatabase functionality."""

    def test_balance_operations(self):
        """Test basic balance read/write operations."""
        db = MockTradingDatabase()
        account_id = "test_account"

        # Test initial balance
        balance = db.read_balance(account_id)
        assert balance == Decimal("0")

        # Test balance update
        new_balance = Decimal("1000.50000000")
        success = db.update_balance(account_id, new_balance)
        assert success is True

        # Test balance read after update
        balance = db.read_balance(account_id)
        assert balance == new_balance

    def test_order_creation(self):
        """Test order creation functionality."""
        db = MockTradingDatabase()
        order_id = "test_order_1"
        order_data = {
            "type": "buy",
            "amount": Decimal("100.00000000"),
            "price": Decimal("50000.00000000"),
        }

        # Test successful order creation
        success = db.create_order(order_id, order_data)
        assert success is True

        # Test duplicate order creation (should fail)
        success = db.create_order(order_id, order_data)
        assert success is False

    def test_operation_logging(self):
        """Test operation logging functionality."""
        db = MockTradingDatabase()
        account_id = "test_account"

        # Perform some operations
        db.read_balance(account_id)
        db.update_balance(account_id, Decimal("500.00000000"))

        # Check operation log
        log = db.get_operation_log()
        assert len(log) == 2
        assert log[0]["operation"] == "read_balance"
        assert log[1]["operation"] == "update_balance"

    def test_clear_data(self):
        """Test data clearing functionality."""
        db = MockTradingDatabase()

        # Add some data
        db.update_balance("account_1", Decimal("1000.00000000"))
        db.create_order("order_1", {"type": "buy"})

        # Clear data
        db.clear_data()

        # Verify data is cleared
        balance = db.read_balance("account_1")
        assert balance == Decimal("0")

        log = db.get_operation_log()
        assert len(log) == 1  # Only the read_balance from above


class TestMockCircuitBreaker:
    """Test the MockCircuitBreaker functionality."""

    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        cb = MockCircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Initial state should be CLOSED
        assert cb.get_state() == "CLOSED"

        def failing_function():
            raise Exception("Test failure")

        # Trigger failures to open circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.get_state() == "CLOSED"  # Still closed after 1 failure

        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.get_state() == "OPEN"  # Should be open after 2 failures

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = MockCircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        def failing_function():
            raise Exception("Test failure")

        def success_function():
            return "success"

        # Open the circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.get_state() == "OPEN"

        # Should fail immediately when open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(success_function)

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should succeed after timeout (moves to HALF_OPEN then CLOSED)
        result = cb.call(success_function)
        assert result == "success"
        assert cb.get_state() == "CLOSED"

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset functionality."""
        cb = MockCircuitBreaker(failure_threshold=1)

        def failing_function():
            raise Exception("Test failure")

        # Open the circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.get_state() == "OPEN"

        # Reset circuit breaker
        cb.reset()
        assert cb.get_state() == "CLOSED"


class TestConcurrencyTester:
    """Test the ConcurrencyTester class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ConcurrencyTestConfig(
            concurrent_operations_count=5,
            race_condition_iterations=10,
            deadlock_timeout_seconds=5,
            circuit_breaker_test_scenarios=["test_scenario"],
        )
        self.tester = ConcurrencyTester(self.config)

    def test_initialization(self):
        """Test ConcurrencyTester initialization."""
        assert self.tester.get_module_name() == "concurrency_testing"
        assert self.tester.config.concurrent_operations_count == 5
        assert isinstance(self.tester.mock_db, MockTradingDatabase)

    @pytest.mark.asyncio
    async def test_concurrent_trading_operations(self):
        """Test concurrent trading operations simulation."""
        results = await self.tester.test_concurrent_trading_operations()

        assert len(results) == self.config.concurrent_operations_count
        assert all(isinstance(r, ConcurrentOperationResult) for r in results)

        # Check that operations have valid data
        for result in results:
            assert result.operation_id.startswith("op_")
            assert result.operation_type in ["buy_order", "sell_order", "balance_check"]
            assert result.execution_time_seconds >= 0
            assert result.thread_id is not None

    @pytest.mark.asyncio
    async def test_race_condition_detection(self):
        """Test race condition detection functionality."""
        result = await self.tester.test_race_condition_detection()

        assert isinstance(result, RaceConditionResult)
        assert result.test_scenario == "concurrent_balance_updates"
        assert result.concurrent_operations == self.config.concurrent_operations_count
        assert result.iterations_run == self.config.race_condition_iterations
        assert result.execution_time_seconds > 0
        assert isinstance(result.test_passed, bool)

    @pytest.mark.asyncio
    async def test_deadlock_detection(self):
        """Test deadlock detection and prevention."""
        result = await self.tester.test_deadlock_detection()

        assert isinstance(result, DeadlockDetectionResult)
        assert result.test_scenario == "resource_lock_ordering"
        assert result.concurrent_threads == 2
        assert result.execution_time_seconds > 0
        assert isinstance(result.deadlock_detected, bool)
        assert isinstance(result.prevention_mechanism_effective, bool)

    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self):
        """Test circuit breaker behavior validation."""
        results = await self.tester.test_circuit_breaker_behavior()

        assert len(results) == 2  # Two test scenarios
        assert all(isinstance(r, CircuitBreakerResult) for r in results)

        for result in results:
            assert result.test_scenario in [
                "api_failure_simulation",
                "database_timeout_simulation",
            ]
            assert result.failure_threshold > 0
            assert result.execution_time_seconds > 0
            assert isinstance(result.test_passed, bool)

    @pytest.mark.asyncio
    async def test_run_tests_integration(self):
        """Test the complete test suite execution."""
        result = await self.tester.run_tests()

        assert result.module_name == "concurrency_testing"
        assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
        assert result.execution_time_seconds > 0
        assert result.tests_run > 0
        assert result.tests_passed >= 0
        assert result.tests_failed >= 0
        assert result.tests_skipped >= 0

        # Check detailed results structure
        assert "concurrent_operations" in result.detailed_results
        assert "race_condition_detection" in result.detailed_results
        assert "deadlock_detection" in result.detailed_results
        assert "circuit_breaker" in result.detailed_results

    def test_simulate_trading_operation(self):
        """Test individual trading operation simulation."""
        # Test buy order
        result = self.tester._simulate_trading_operation(
            "test_op_1", "buy_order", "test_account"
        )
        assert isinstance(result, ConcurrentOperationResult)
        assert result.operation_id == "test_op_1"
        assert result.operation_type == "buy_order"

        # Test sell order
        result = self.tester._simulate_trading_operation(
            "test_op_2", "sell_order", "test_account"
        )
        assert isinstance(result, ConcurrentOperationResult)
        assert result.operation_type == "sell_order"

        # Test balance check
        result = self.tester._simulate_trading_operation(
            "test_op_3", "balance_check", "test_account"
        )
        assert isinstance(result, ConcurrentOperationResult)
        assert result.operation_type == "balance_check"

    def test_analyze_overlapping_operations(self):
        """Test overlapping operations analysis."""
        # Create mock operation log with overlapping operations
        operation_log = [
            {
                "timestamp": 1000.0,
                "operation": "read_balance",
                "resource_id": "balance_account_1",
                "thread_id": 1,
            },
            {
                "timestamp": 1000.002,  # Very close in time
                "operation": "update_balance",
                "resource_id": "balance_account_1",
                "thread_id": 2,  # Different thread
            },
        ]

        overlapping = self.tester._analyze_overlapping_operations(operation_log)
        assert len(overlapping) == 1
        assert overlapping[0]["type"] == "overlapping_operations"
        assert overlapping[0]["resource_id"] == "balance_account_1"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in concurrent operations."""
        # Test with invalid configuration
        invalid_config = ConcurrencyTestConfig(
            concurrent_operations_count=0,
            race_condition_iterations=0,
            deadlock_timeout_seconds=0,
            circuit_breaker_test_scenarios=[],
        )

        tester = ConcurrencyTester(invalid_config)
        result = await tester.run_tests()

        # Should handle gracefully even with invalid config
        assert result.module_name == "concurrency_testing"
        assert isinstance(result.status, TestStatus)


if __name__ == "__main__":
    pytest.main([__file__])

    @pytest.mark.asyncio
    async def test_api_rate_limiting_and_timeout_handling(self):
        """Test API rate limiting and timeout handling functionality."""
        results = await self.tester.test_api_rate_limiting_and_timeout_handling()

        assert len(results) == 2  # Two test scenarios
        assert all(isinstance(r, dict) for r in results)

        for result in results:
            assert "scenario" in result
            assert "total_requests" in result
            assert "successful_requests" in result
            assert "rate_limited_requests" in result
            assert "circuit_breaker_activations" in result
            assert "test_passed" in result
            assert result["scenario"] in [
                "exchange_api_rate_limit",
                "market_data_api_limit",
            ]

    @pytest.mark.asyncio
    async def test_data_consistency_under_concurrent_access(self):
        """Test data consistency verification under concurrent access."""
        result = await self.tester.test_data_consistency_under_concurrent_access()

        assert isinstance(result, dict)
        assert "test_scenario" in result
        assert "test_passed" in result
        assert "consistency_violations" in result
        assert "data_integrity_issues" in result
        assert "successful_transfers" in result
        assert "total_transfers_attempted" in result
        assert "final_account_balances" in result

        assert result["test_scenario"] == "concurrent_transfer_operations"
        assert isinstance(result["test_passed"], bool)
        assert isinstance(result["consistency_violations"], list)
        assert isinstance(result["data_integrity_issues"], list)

    @pytest.mark.asyncio
    async def test_comprehensive_concurrency_integration(self):
        """Test comprehensive integration of all concurrency features."""
        result = await self.tester.test_comprehensive_concurrency_integration()

        assert isinstance(result, dict)
        assert "integration_test_passed" in result
        assert "total_execution_time_seconds" in result
        assert "individual_test_results" in result
        assert "summary" in result

        # Check that all individual tests were run
        individual_results = result["individual_test_results"]
        expected_tests = [
            "concurrent_operations",
            "race_condition_detection",
            "deadlock_detection",
            "circuit_breaker_behavior",
            "api_rate_limiting",
            "data_consistency",
        ]

        for test_name in expected_tests:
            assert test_name in individual_results
            assert individual_results[test_name] is not None

    def test_analyze_data_integrity_issues(self):
        """Test data integrity issues analysis."""
        # Create mock operation log
        operation_log = [
            {
                "timestamp": 1000.0,
                "operation": "read_balance",
                "resource_id": "balance_account_1",
                "thread_id": 1,
                "data": {"balance": Decimal("1000.00000000")},
            },
            {
                "timestamp": 1000.0005,  # Very close in time
                "operation": "update_balance",
                "resource_id": "balance_account_1",
                "thread_id": 2,  # Different thread
                "data": {"new_balance": Decimal("900.00000000")},
            },
        ]

        # Create mock transfer results
        transfer_results = [
            {
                "transfer_id": 1,
                "success": True,
                "from_balance_before": Decimal("1000.00000000"),
                "to_balance_before": Decimal("500.00000000"),
                "amount": Decimal("100.00000000"),
                "from_balance_after": Decimal("900.00000000"),
                "to_balance_after": Decimal("600.00000000"),
            }
        ]

        issues = self.tester._analyze_data_integrity_issues(
            operation_log, transfer_results
        )

        assert len(issues) == 1
        assert issues[0]["type"] == "rapid_concurrent_access"
        assert issues[0]["account"] == "balance_account_1"

    @pytest.mark.asyncio
    async def test_extended_run_tests_integration(self):
        """Test the complete extended test suite execution."""
        result = await self.tester.run_tests()

        assert result.module_name == "concurrency_testing"
        assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
        assert result.execution_time_seconds > 0
        assert result.tests_run > 0

        # Check that all new test categories are included
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
            assert category in result.detailed_results

    def test_mock_rate_limited_api(self):
        """Test the MockRateLimitedAPI functionality."""
        # This test would be inside the actual test method, but we can test the concept
        requests_per_second = 2

        class MockRateLimitedAPI:
            def __init__(self, requests_per_second: int):
                self.requests_per_second = requests_per_second
                self.request_times = []
                self.lock = threading.Lock()

            def make_request(self, request_data):
                with self.lock:
                    current_time = time.time()

                    # Remove old requests (older than 1 second)
                    self.request_times = [
                        t for t in self.request_times if current_time - t < 1.0
                    ]

                    # Check rate limit
                    if len(self.request_times) >= self.requests_per_second:
                        raise Exception("Rate limit exceeded")

                    # Add current request time
                    self.request_times.append(current_time)

                    return {"status": "success", "data": request_data}

        api = MockRateLimitedAPI(requests_per_second)

        # First two requests should succeed
        result1 = api.make_request({"test": "data1"})
        assert result1["status"] == "success"

        result2 = api.make_request({"test": "data2"})
        assert result2["status"] == "success"

        # Third request should fail due to rate limit
        with pytest.raises(Exception, match="Rate limit exceeded"):
            api.make_request({"test": "data3"})

    @pytest.mark.asyncio
    async def test_error_handling_in_extended_tests(self):
        """Test error handling in the extended concurrency tests."""
        # Test with a tester that has minimal configuration to trigger edge cases
        minimal_config = ConcurrencyTestConfig(
            concurrent_operations_count=1,
            race_condition_iterations=1,
            deadlock_timeout_seconds=1,
            circuit_breaker_test_scenarios=["minimal_test"],
        )

        minimal_tester = ConcurrencyTester(minimal_config)

        # Should handle gracefully even with minimal config
        rate_limiting_results = (
            await minimal_tester.test_api_rate_limiting_and_timeout_handling()
        )
        assert len(rate_limiting_results) == 2  # Still runs both scenarios

        data_consistency_result = (
            await minimal_tester.test_data_consistency_under_concurrent_access()
        )
        assert isinstance(data_consistency_result, dict)
        assert "test_passed" in data_consistency_result

        integration_result = (
            await minimal_tester.test_comprehensive_concurrency_integration()
        )
        assert isinstance(integration_result, dict)
        assert "integration_test_passed" in integration_result
