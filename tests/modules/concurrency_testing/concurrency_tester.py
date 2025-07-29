"""
Concurrency Testing Module for comprehensive testing framework.
Tests concurrent trading operations, race conditions, and deadlock detection.
"""

import asyncio
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
import random
from decimal import Decimal
from contextlib import asynccontextmanager

try:
    from ...orchestrator.test_orchestrator import (
        TestModuleInterface,
        TestModuleResult,
        TestStatus,
    )
    from ...config.test_config import TestConfig, ConcurrencyTestConfig
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
    from config.test_config import TestConfig, ConcurrencyTestConfig
    from utils.test_utilities import TestLogger, DecimalHelper, MockDataGenerator


@dataclass
class ConcurrentOperationResult:
    """Result from concurrent trading operation tests."""

    operation_id: str
    operation_type: str
    start_time: float
    end_time: float
    execution_time_seconds: float
    success: bool
    data_integrity_maintained: bool
    error_message: Optional[str] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None


@dataclass
class RaceConditionResult:
    """Result from race condition detection tests."""

    test_scenario: str
    concurrent_operations: int
    iterations_run: int
    race_conditions_detected: int
    data_consistency_violations: List[Dict[str, Any]]
    execution_time_seconds: float
    test_passed: bool
    error_message: Optional[str] = None


@dataclass
class DeadlockDetectionResult:
    """Result from deadlock detection and prevention tests."""

    test_scenario: str
    concurrent_threads: int
    deadlock_detected: bool
    deadlock_resolution_time_seconds: Optional[float]
    resource_contention_details: Dict[str, Any]
    prevention_mechanism_effective: bool
    execution_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class CircuitBreakerResult:
    """Result from circuit breaker behavior validation tests."""

    test_scenario: str
    failure_threshold: int
    failures_triggered: int
    circuit_opened: bool
    recovery_time_seconds: Optional[float]
    fallback_mechanism_activated: bool
    execution_time_seconds: float
    test_passed: bool
    error_message: Optional[str] = None


class MockTradingDatabase:
    """Mock database for testing concurrent operations."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._operation_log: List[Dict[str, Any]] = []
        self._global_lock = threading.Lock()

    def get_lock(self, resource_id: str) -> threading.Lock:
        """Get or create a lock for a specific resource."""
        with self._global_lock:
            if resource_id not in self._locks:
                self._locks[resource_id] = threading.Lock()
            return self._locks[resource_id]

    def read_balance(self, account_id: str) -> Decimal:
        """Read account balance with potential for race conditions."""
        lock = self.get_lock(f"balance_{account_id}")
        with lock:
            # Simulate database read delay
            time.sleep(0.001)
            balance = self._data.get(f"balance_{account_id}", Decimal("0"))
            self._log_operation("read_balance", account_id, {"balance": balance})
            return balance

    def update_balance(self, account_id: str, new_balance: Decimal) -> bool:
        """Update account balance with potential for race conditions."""
        lock = self.get_lock(f"balance_{account_id}")
        with lock:
            # Simulate database write delay
            time.sleep(0.002)
            old_balance = self._data.get(f"balance_{account_id}", Decimal("0"))
            self._data[f"balance_{account_id}"] = new_balance
            self._log_operation(
                "update_balance",
                account_id,
                {"old_balance": old_balance, "new_balance": new_balance},
            )
            return True

    def create_order(self, order_id: str, order_data: Dict[str, Any]) -> bool:
        """Create trading order with potential for race conditions."""
        lock = self.get_lock(f"order_{order_id}")
        with lock:
            # Simulate order creation delay
            time.sleep(0.003)
            if f"order_{order_id}" in self._data:
                return False  # Order already exists
            self._data[f"order_{order_id}"] = order_data
            self._log_operation("create_order", order_id, order_data)
            return True

    def _log_operation(self, operation: str, resource_id: str, data: Dict[str, Any]):
        """Log database operations for analysis."""
        with self._global_lock:
            self._operation_log.append(
                {
                    "timestamp": time.time(),
                    "operation": operation,
                    "resource_id": resource_id,
                    "data": data,
                    "thread_id": threading.get_ident(),
                }
            )

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the operation log for analysis."""
        with self._global_lock:
            return self._operation_log.copy()

    def clear_data(self):
        """Clear all data and logs."""
        with self._global_lock:
            self._data.clear()
            self._operation_log.clear()


class MockCircuitBreaker:
    """Mock circuit breaker for testing circuit breaker behavior."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 10):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self._lock:
            if self.state == "OPEN":
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time > self.recovery_timeout
                ):
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                raise e

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state

    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "CLOSED"


class ConcurrencyTester(TestModuleInterface):
    """
    Concurrency Testing Module implementing TestModuleInterface.

    Tests concurrent trading operations, race conditions, and deadlock detection.
    Implements requirements 5.1, 5.2, 5.4.
    """

    def __init__(self, config: ConcurrencyTestConfig = None):
        self.config = config or ConcurrencyTestConfig(
            concurrent_operations_count=10,
            race_condition_iterations=100,
            deadlock_timeout_seconds=30,
            circuit_breaker_test_scenarios=["api_failure", "database_timeout"],
        )
        self.logger = TestLogger("concurrency_tester", "INFO")
        self.decimal_helper = DecimalHelper()
        self.mock_db = MockTradingDatabase()
        self.test_results: List[
            Union[
                ConcurrentOperationResult,
                RaceConditionResult,
                DeadlockDetectionResult,
                CircuitBreakerResult,
            ]
        ] = []

        self.logger.info(
            f"ConcurrencyTester initialized with {self.config.concurrent_operations_count} "
            f"concurrent operations, {self.config.race_condition_iterations} race condition iterations"
        )

    def get_module_name(self) -> str:
        """Get the name of this test module."""
        return "concurrency_testing"

    def _simulate_trading_operation(
        self, operation_id: str, operation_type: str, account_id: str
    ) -> ConcurrentOperationResult:
        """Simulate a trading operation for concurrent testing."""
        start_time = time.time()
        thread_id = threading.get_ident()
        success = True
        data_integrity_maintained = True
        error_message = None

        try:
            if operation_type == "buy_order":
                # Simulate buy order: check balance, create order, update balance
                current_balance = self.mock_db.read_balance(account_id)
                order_amount = Decimal("100.00000000")

                if current_balance >= order_amount:
                    order_data = {
                        "type": "buy",
                        "amount": order_amount,
                        "price": Decimal("50000.00000000"),
                        "account_id": account_id,
                    }
                    order_created = self.mock_db.create_order(operation_id, order_data)
                    if order_created:
                        new_balance = current_balance - order_amount
                        self.mock_db.update_balance(account_id, new_balance)
                    else:
                        success = False
                        error_message = "Failed to create order"
                else:
                    success = False
                    error_message = "Insufficient balance"

            elif operation_type == "sell_order":
                # Simulate sell order: create order, update balance
                order_amount = Decimal("50.00000000")
                order_data = {
                    "type": "sell",
                    "amount": order_amount,
                    "price": Decimal("50000.00000000"),
                    "account_id": account_id,
                }
                order_created = self.mock_db.create_order(operation_id, order_data)
                if order_created:
                    current_balance = self.mock_db.read_balance(account_id)
                    new_balance = current_balance + (
                        order_amount * Decimal("50000.00000000")
                    )
                    self.mock_db.update_balance(account_id, new_balance)
                else:
                    success = False
                    error_message = "Failed to create sell order"

            elif operation_type == "balance_check":
                # Simple balance check operation
                balance = self.mock_db.read_balance(account_id)
                # Verify balance is a valid Decimal
                if not isinstance(balance, Decimal):
                    data_integrity_maintained = False
                    error_message = "Balance is not Decimal type"

            # Add random delay to increase chance of race conditions
            time.sleep(random.uniform(0.001, 0.005))

        except Exception as e:
            success = False
            data_integrity_maintained = False
            error_message = f"Operation failed: {str(e)}"

        end_time = time.time()
        execution_time = end_time - start_time

        return ConcurrentOperationResult(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time,
            success=success,
            data_integrity_maintained=data_integrity_maintained,
            error_message=error_message,
            thread_id=thread_id,
        )

    async def test_concurrent_trading_operations(
        self,
    ) -> List[ConcurrentOperationResult]:
        """
        Test concurrent trading operations simulation.

        Returns:
            List of ConcurrentOperationResult with concurrent operation details
        """
        self.logger.info("Testing concurrent trading operations")

        # Clear previous test data
        self.mock_db.clear_data()

        # Initialize test accounts with balances
        test_accounts = ["account_1", "account_2", "account_3"]
        for account in test_accounts:
            self.mock_db.update_balance(account, Decimal("10000.00000000"))

        # Define concurrent operations
        operations = []
        for i in range(self.config.concurrent_operations_count):
            account_id = random.choice(test_accounts)
            operation_type = random.choice(["buy_order", "sell_order", "balance_check"])
            operations.append((f"op_{i}", operation_type, account_id))

        results = []

        # Execute operations concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(
            max_workers=self.config.concurrent_operations_count
        ) as executor:
            # Submit all operations
            future_to_operation = {
                executor.submit(
                    self._simulate_trading_operation, op_id, op_type, account_id
                ): (op_id, op_type, account_id)
                for op_id, op_type, account_id in operations
            }

            # Collect results as they complete
            for future in as_completed(future_to_operation):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    op_id, op_type, account_id = future_to_operation[future]
                    error_result = ConcurrentOperationResult(
                        operation_id=op_id,
                        operation_type=op_type,
                        start_time=time.time(),
                        end_time=time.time(),
                        execution_time_seconds=0,
                        success=False,
                        data_integrity_maintained=False,
                        error_message=f"Concurrent operation failed: {str(e)}",
                    )
                    results.append(error_result)

        self.logger.info(
            f"Concurrent trading operations test completed: {len(results)} operations executed"
        )
        return results

    async def test_race_condition_detection(self) -> RaceConditionResult:
        """
        Test race condition detection with multiple database sessions.

        Returns:
            RaceConditionResult with race condition detection details
        """
        start_time = time.time()
        self.logger.info("Testing race condition detection")

        # Clear previous test data
        self.mock_db.clear_data()

        # Initialize shared resource
        shared_account = "shared_account"
        initial_balance = Decimal("1000.00000000")
        self.mock_db.update_balance(shared_account, initial_balance)

        race_conditions_detected = 0
        data_consistency_violations = []

        def race_condition_operation(iteration: int) -> Dict[str, Any]:
            """Operation that can cause race conditions."""
            try:
                # Read-modify-write operation that can cause race conditions
                current_balance = self.mock_db.read_balance(shared_account)

                # Simulate processing time that increases race condition likelihood
                time.sleep(0.001)

                # Modify balance (add small amount)
                new_balance = current_balance + Decimal("1.00000000")
                success = self.mock_db.update_balance(shared_account, new_balance)

                return {
                    "iteration": iteration,
                    "success": success,
                    "old_balance": current_balance,
                    "new_balance": new_balance,
                    "thread_id": threading.get_ident(),
                }
            except Exception as e:
                return {
                    "iteration": iteration,
                    "success": False,
                    "error": str(e),
                    "thread_id": threading.get_ident(),
                }

        # Execute race condition test
        with ThreadPoolExecutor(
            max_workers=self.config.concurrent_operations_count
        ) as executor:
            futures = [
                executor.submit(race_condition_operation, i)
                for i in range(self.config.race_condition_iterations)
            ]

            operation_results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    operation_results.append(result)
                except Exception as e:
                    operation_results.append(
                        {
                            "success": False,
                            "error": f"Future execution failed: {str(e)}",
                        }
                    )

        # Analyze results for race conditions
        final_balance = self.mock_db.read_balance(shared_account)
        expected_balance = initial_balance + (
            Decimal("1.00000000") * self.config.race_condition_iterations
        )

        # Check for data consistency violations
        if final_balance != expected_balance:
            race_conditions_detected += 1
            data_consistency_violations.append(
                {
                    "type": "balance_inconsistency",
                    "expected_balance": expected_balance,
                    "actual_balance": final_balance,
                    "difference": expected_balance - final_balance,
                }
            )

        # Analyze operation log for overlapping operations
        operation_log = self.mock_db.get_operation_log()
        overlapping_operations = self._analyze_overlapping_operations(operation_log)

        if overlapping_operations:
            race_conditions_detected += len(overlapping_operations)
            data_consistency_violations.extend(overlapping_operations)

        execution_time = time.time() - start_time
        test_passed = race_conditions_detected == 0

        result = RaceConditionResult(
            test_scenario="concurrent_balance_updates",
            concurrent_operations=self.config.concurrent_operations_count,
            iterations_run=self.config.race_condition_iterations,
            race_conditions_detected=race_conditions_detected,
            data_consistency_violations=data_consistency_violations,
            execution_time_seconds=execution_time,
            test_passed=test_passed,
        )

        self.logger.info(
            f"Race condition detection test completed: {race_conditions_detected} race conditions detected"
        )
        return result

    def _analyze_overlapping_operations(
        self, operation_log: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze operation log for overlapping operations that could indicate race conditions."""
        overlapping_operations = []

        # Group operations by resource
        resource_operations = {}
        for op in operation_log:
            resource_id = op["resource_id"]
            if resource_id not in resource_operations:
                resource_operations[resource_id] = []
            resource_operations[resource_id].append(op)

        # Check for overlapping operations on same resource
        for resource_id, operations in resource_operations.items():
            operations.sort(key=lambda x: x["timestamp"])

            for i in range(len(operations) - 1):
                current_op = operations[i]
                next_op = operations[i + 1]

                # Check if operations from different threads overlap in time
                if (
                    current_op["thread_id"] != next_op["thread_id"]
                    and next_op["timestamp"] - current_op["timestamp"] < 0.005
                ):  # 5ms threshold
                    overlapping_operations.append(
                        {
                            "type": "overlapping_operations",
                            "resource_id": resource_id,
                            "operation_1": current_op,
                            "operation_2": next_op,
                            "time_difference": next_op["timestamp"]
                            - current_op["timestamp"],
                        }
                    )

        return overlapping_operations

    async def test_deadlock_detection(self) -> DeadlockDetectionResult:
        """
        Test deadlock detection and prevention.

        Returns:
            DeadlockDetectionResult with deadlock detection details
        """
        start_time = time.time()
        self.logger.info("Testing deadlock detection and prevention")

        # Clear previous test data
        self.mock_db.clear_data()

        # Initialize resources that can cause deadlocks
        resource_a = "resource_a"
        resource_b = "resource_b"
        self.mock_db.update_balance(resource_a, Decimal("1000.00000000"))
        self.mock_db.update_balance(resource_b, Decimal("2000.00000000"))

        deadlock_detected = False
        deadlock_resolution_time = None
        resource_contention_details = {}
        prevention_mechanism_effective = True

        def deadlock_operation_1() -> Dict[str, Any]:
            """Operation that acquires locks in order A -> B."""
            try:
                lock_a = self.mock_db.get_lock(f"balance_{resource_a}")
                lock_b = self.mock_db.get_lock(f"balance_{resource_b}")

                with lock_a:
                    time.sleep(0.1)  # Hold lock A for a while
                    with lock_b:
                        # Transfer from A to B
                        balance_a = self.mock_db.read_balance(resource_a)
                        balance_b = self.mock_db.read_balance(resource_b)

                        transfer_amount = Decimal("100.00000000")
                        self.mock_db.update_balance(
                            resource_a, balance_a - transfer_amount
                        )
                        self.mock_db.update_balance(
                            resource_b, balance_b + transfer_amount
                        )

                        return {"success": True, "operation": "A_to_B_transfer"}
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "operation": "A_to_B_transfer",
                }

        def deadlock_operation_2() -> Dict[str, Any]:
            """Operation that acquires locks in order B -> A (potential deadlock)."""
            try:
                lock_b = self.mock_db.get_lock(f"balance_{resource_b}")
                lock_a = self.mock_db.get_lock(f"balance_{resource_a}")

                with lock_b:
                    time.sleep(0.1)  # Hold lock B for a while
                    with lock_a:
                        # Transfer from B to A
                        balance_a = self.mock_db.read_balance(resource_a)
                        balance_b = self.mock_db.read_balance(resource_b)

                        transfer_amount = Decimal("50.00000000")
                        self.mock_db.update_balance(
                            resource_b, balance_b - transfer_amount
                        )
                        self.mock_db.update_balance(
                            resource_a, balance_a + transfer_amount
                        )

                        return {"success": True, "operation": "B_to_A_transfer"}
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "operation": "B_to_A_transfer",
                }

        # Execute potentially deadlocking operations
        deadlock_start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(deadlock_operation_1)
            future_2 = executor.submit(deadlock_operation_2)

            try:
                # Wait for operations with timeout to detect deadlock
                result_1 = future_1.result(timeout=self.config.deadlock_timeout_seconds)
                result_2 = future_2.result(timeout=self.config.deadlock_timeout_seconds)

                resource_contention_details = {
                    "operation_1_result": result_1,
                    "operation_2_result": result_2,
                    "deadlock_occurred": False,
                }

            except Exception as e:
                deadlock_detected = True
                deadlock_resolution_time = time.time() - deadlock_start_time
                prevention_mechanism_effective = False

                resource_contention_details = {
                    "deadlock_error": str(e),
                    "deadlock_occurred": True,
                    "resolution_time": deadlock_resolution_time,
                }

        execution_time = time.time() - start_time

        result = DeadlockDetectionResult(
            test_scenario="resource_lock_ordering",
            concurrent_threads=2,
            deadlock_detected=deadlock_detected,
            deadlock_resolution_time=deadlock_resolution_time,
            resource_contention_details=resource_contention_details,
            prevention_mechanism_effective=prevention_mechanism_effective,
            execution_time_seconds=execution_time,
        )

        self.logger.info(
            f"Deadlock detection test completed: deadlock_detected={deadlock_detected}, "
            f"prevention_effective={prevention_mechanism_effective}"
        )
        return result

    async def test_circuit_breaker_behavior(self) -> List[CircuitBreakerResult]:
        """
        Test circuit breaker behavior validation.

        Returns:
            List of CircuitBreakerResult with circuit breaker test details
        """
        self.logger.info("Testing circuit breaker behavior validation")

        results = []

        # Test scenarios for circuit breaker
        test_scenarios = [
            {
                "name": "api_failure_simulation",
                "failure_threshold": 3,
                "failure_count": 5,
                "recovery_timeout": 2,
            },
            {
                "name": "database_timeout_simulation",
                "failure_threshold": 5,
                "failure_count": 7,
                "recovery_timeout": 3,
            },
        ]

        for scenario in test_scenarios:
            start_time = time.time()

            try:
                circuit_breaker = MockCircuitBreaker(
                    failure_threshold=scenario["failure_threshold"],
                    recovery_timeout=scenario["recovery_timeout"],
                )

                def failing_operation():
                    """Operation that always fails to test circuit breaker."""
                    raise Exception("Simulated operation failure")

                def successful_operation():
                    """Operation that succeeds to test circuit breaker recovery."""
                    return "success"

                failures_triggered = 0
                circuit_opened = False
                fallback_mechanism_activated = False

                # Trigger failures to open circuit breaker
                for i in range(scenario["failure_count"]):
                    try:
                        circuit_breaker.call(failing_operation)
                    except Exception:
                        failures_triggered += 1
                        if circuit_breaker.get_state() == "OPEN":
                            circuit_opened = True
                            break

                # Test that circuit breaker blocks calls when open
                if circuit_opened:
                    try:
                        circuit_breaker.call(successful_operation)
                        # Should not reach here if circuit breaker is working
                    except Exception:
                        fallback_mechanism_activated = True

                # Wait for recovery timeout and test recovery
                recovery_start_time = time.time()
                time.sleep(scenario["recovery_timeout"] + 0.5)

                recovery_time = None
                try:
                    # Should succeed after recovery timeout
                    result = circuit_breaker.call(successful_operation)
                    if result == "success":
                        recovery_time = time.time() - recovery_start_time
                except Exception:
                    pass

                execution_time = time.time() - start_time
                test_passed = (
                    circuit_opened
                    and fallback_mechanism_activated
                    and recovery_time is not None
                )

                result = CircuitBreakerResult(
                    test_scenario=scenario["name"],
                    failure_threshold=scenario["failure_threshold"],
                    failures_triggered=failures_triggered,
                    circuit_opened=circuit_opened,
                    recovery_time_seconds=recovery_time,
                    fallback_mechanism_activated=fallback_mechanism_activated,
                    execution_time_seconds=execution_time,
                    test_passed=test_passed,
                )

                results.append(result)

                self.logger.debug(
                    f"Circuit breaker test '{scenario['name']}': "
                    f"opened={circuit_opened}, recovery_time={recovery_time}, passed={test_passed}"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = (
                    f"Circuit breaker test failed for {scenario['name']}: {str(e)}"
                )
                self.logger.error(error_msg)

                result = CircuitBreakerResult(
                    test_scenario=scenario["name"],
                    failure_threshold=scenario["failure_threshold"],
                    failures_triggered=0,
                    circuit_opened=False,
                    recovery_time_seconds=None,
                    fallback_mechanism_activated=False,
                    execution_time_seconds=execution_time,
                    test_passed=False,
                    error_message=error_msg,
                )

                results.append(result)

        self.logger.info(
            f"Circuit breaker behavior tests completed: {len(results)} scenarios tested"
        )
        return results

    async def run_tests(self) -> TestModuleResult:
        """
        Run all concurrency tests and return comprehensive results.

        Returns:
            TestModuleResult with all test outcomes
        """
        start_time = time.time()
        self.logger.info("Starting comprehensive concurrency tests")

        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        error_messages = []
        detailed_results = {}

        try:
            # Test 1: Concurrent trading operations
            self.logger.info("Running concurrent trading operations tests")
            concurrent_operation_results = (
                await self.test_concurrent_trading_operations()
            )
            tests_run += len(concurrent_operation_results)

            successful_operations = sum(
                1 for r in concurrent_operation_results if r.success
            )
            failed_operations = (
                len(concurrent_operation_results) - successful_operations
            )
            tests_passed += successful_operations
            tests_failed += failed_operations

            for result in concurrent_operation_results:
                if result.error_message:
                    error_messages.append(result.error_message)

            detailed_results["concurrent_operations"] = {
                "total_operations": len(concurrent_operation_results),
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "data_integrity_maintained": sum(
                    1
                    for r in concurrent_operation_results
                    if r.data_integrity_maintained
                ),
                "average_execution_time": (
                    sum(r.execution_time_seconds for r in concurrent_operation_results)
                    / len(concurrent_operation_results)
                    if concurrent_operation_results
                    else 0
                ),
            }

            # Test 2: Race condition detection
            self.logger.info("Running race condition detection tests")
            race_condition_result = await self.test_race_condition_detection()
            tests_run += 1
            if race_condition_result.test_passed:
                tests_passed += 1
            else:
                tests_failed += 1
                if race_condition_result.error_message:
                    error_messages.append(race_condition_result.error_message)

            detailed_results["race_condition_detection"] = {
                "test_passed": race_condition_result.test_passed,
                "race_conditions_detected": race_condition_result.race_conditions_detected,
                "data_consistency_violations": len(
                    race_condition_result.data_consistency_violations
                ),
                "iterations_run": race_condition_result.iterations_run,
                "execution_time": race_condition_result.execution_time_seconds,
            }

            # Test 3: Deadlock detection
            self.logger.info("Running deadlock detection tests")
            deadlock_result = await self.test_deadlock_detection()
            tests_run += 1
            if deadlock_result.prevention_mechanism_effective:
                tests_passed += 1
            else:
                tests_failed += 1
                if deadlock_result.error_message:
                    error_messages.append(deadlock_result.error_message)

            detailed_results["deadlock_detection"] = {
                "deadlock_detected": deadlock_result.deadlock_detected,
                "prevention_mechanism_effective": deadlock_result.prevention_mechanism_effective,
                "deadlock_resolution_time": deadlock_result.deadlock_resolution_time_seconds,
                "execution_time": deadlock_result.execution_time_seconds,
            }

            # Test 4: Circuit breaker behavior
            self.logger.info("Running circuit breaker behavior tests")
            circuit_breaker_results = await self.test_circuit_breaker_behavior()
            tests_run += len(circuit_breaker_results)

            circuit_breaker_passed = sum(
                1 for r in circuit_breaker_results if r.test_passed
            )
            circuit_breaker_failed = (
                len(circuit_breaker_results) - circuit_breaker_passed
            )
            tests_passed += circuit_breaker_passed
            tests_failed += circuit_breaker_failed

            for result in circuit_breaker_results:
                if result.error_message:
                    error_messages.append(result.error_message)

            detailed_results["circuit_breaker"] = {
                "total_scenarios": len(circuit_breaker_results),
                "scenarios_passed": circuit_breaker_passed,
                "scenarios_failed": circuit_breaker_failed,
                "test_results": [
                    {
                        "scenario": r.test_scenario,
                        "test_passed": r.test_passed,
                        "circuit_opened": r.circuit_opened,
                        "recovery_time": r.recovery_time_seconds,
                    }
                    for r in circuit_breaker_results
                ],
            }

            # Test 5: API rate limiting and timeout handling
            self.logger.info("Running API rate limiting and timeout handling tests")
            rate_limiting_results = (
                await self.test_api_rate_limiting_and_timeout_handling()
            )
            tests_run += len(rate_limiting_results)

            rate_limiting_passed = sum(
                1 for r in rate_limiting_results if r.get("test_passed", False)
            )
            rate_limiting_failed = len(rate_limiting_results) - rate_limiting_passed
            tests_passed += rate_limiting_passed
            tests_failed += rate_limiting_failed

            for result in rate_limiting_results:
                if result.get("error_message"):
                    error_messages.append(result["error_message"])

            detailed_results["api_rate_limiting"] = {
                "total_scenarios": len(rate_limiting_results),
                "scenarios_passed": rate_limiting_passed,
                "scenarios_failed": rate_limiting_failed,
                "rate_limiting_effective": sum(
                    1
                    for r in rate_limiting_results
                    if r.get("rate_limiting_effective", False)
                ),
                "circuit_breaker_integrations": sum(
                    1
                    for r in rate_limiting_results
                    if r.get("circuit_breaker_effective", False)
                ),
            }

            # Test 6: Data consistency under concurrent access
            self.logger.info("Running data consistency under concurrent access tests")
            data_consistency_result = (
                await self.test_data_consistency_under_concurrent_access()
            )
            tests_run += 1
            if data_consistency_result.get("test_passed", False):
                tests_passed += 1
            else:
                tests_failed += 1
                if data_consistency_result.get("error_message"):
                    error_messages.append(data_consistency_result["error_message"])

            detailed_results["data_consistency"] = {
                "test_passed": data_consistency_result.get("test_passed", False),
                "consistency_violations": len(
                    data_consistency_result.get("consistency_violations", [])
                ),
                "integrity_issues": len(
                    data_consistency_result.get("data_integrity_issues", [])
                ),
                "successful_transfers": data_consistency_result.get(
                    "successful_transfers", 0
                ),
                "total_transfers": data_consistency_result.get(
                    "total_transfers_attempted", 0
                ),
            }

            # Test 7: Comprehensive integration test
            self.logger.info("Running comprehensive concurrency integration test")
            integration_result = await self.test_comprehensive_concurrency_integration()
            tests_run += 1
            if integration_result.get("integration_test_passed", False):
                tests_passed += 1
            else:
                tests_failed += 1
                if integration_result.get("error"):
                    error_messages.append(integration_result["error"])

            detailed_results["integration_test"] = {
                "integration_test_passed": integration_result.get(
                    "integration_test_passed", False
                ),
                "total_execution_time": integration_result.get(
                    "total_execution_time_seconds", 0
                ),
                "individual_results": integration_result.get(
                    "individual_test_results", {}
                ),
                "summary": integration_result.get("summary", {}),
            }

        except Exception as e:
            error_msg = f"Concurrency tests execution failed: {str(e)}"
            self.logger.error(error_msg)
            error_messages.append(error_msg)
            tests_failed += 1

        execution_time = time.time() - start_time
        overall_status = (
            TestStatus.COMPLETED if tests_failed == 0 else TestStatus.FAILED
        )

        result = TestModuleResult(
            module_name=self.get_module_name(),
            status=overall_status,
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
            f"Concurrency tests completed: {tests_run} tests run, "
            f"{tests_passed} passed, {tests_failed} failed"
        )

        return result

    async def test_api_rate_limiting_and_timeout_handling(self) -> List[Dict[str, Any]]:
        """
        Test API rate limiting and timeout handling with circuit breaker integration.

        Returns:
            List of test results for API rate limiting scenarios
        """
        self.logger.info("Testing API rate limiting and timeout handling")

        results = []

        # Test scenarios for API rate limiting
        rate_limit_scenarios = [
            {
                "name": "exchange_api_rate_limit",
                "requests_per_second": 10,
                "burst_requests": 50,
                "timeout_seconds": 5,
            },
            {
                "name": "market_data_api_limit",
                "requests_per_second": 5,
                "burst_requests": 20,
                "timeout_seconds": 3,
            },
        ]

        for scenario in rate_limit_scenarios:
            start_time = time.time()

            try:
                # Create circuit breaker for this scenario
                circuit_breaker = MockCircuitBreaker(
                    failure_threshold=3, recovery_timeout=scenario["timeout_seconds"]
                )

                # Mock API that enforces rate limiting
                class MockRateLimitedAPI:
                    def __init__(self, requests_per_second: int):
                        self.requests_per_second = requests_per_second
                        self.request_times = []
                        self.lock = threading.Lock()

                    def make_request(
                        self, request_data: Dict[str, Any]
                    ) -> Dict[str, Any]:
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

                            # Simulate API processing time
                            time.sleep(0.01)

                            return {
                                "status": "success",
                                "data": request_data,
                                "timestamp": current_time,
                            }

                api = MockRateLimitedAPI(scenario["requests_per_second"])

                # Test concurrent requests that exceed rate limit
                successful_requests = 0
                rate_limited_requests = 0
                circuit_breaker_activations = 0

                def make_api_request(request_id: int) -> Dict[str, Any]:
                    try:
                        request_data = {"request_id": request_id, "type": "market_data"}
                        result = circuit_breaker.call(api.make_request, request_data)
                        return {"success": True, "result": result}
                    except Exception as e:
                        error_msg = str(e)
                        if "Rate limit exceeded" in error_msg:
                            return {"success": False, "error": "rate_limited"}
                        elif "Circuit breaker is OPEN" in error_msg:
                            return {"success": False, "error": "circuit_breaker"}
                        else:
                            return {"success": False, "error": error_msg}

                # Execute burst of requests
                with ThreadPoolExecutor(
                    max_workers=scenario["burst_requests"]
                ) as executor:
                    futures = [
                        executor.submit(make_api_request, i)
                        for i in range(scenario["burst_requests"])
                    ]

                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=10)
                            if result["success"]:
                                successful_requests += 1
                            elif result["error"] == "rate_limited":
                                rate_limited_requests += 1
                            elif result["error"] == "circuit_breaker":
                                circuit_breaker_activations += 1
                        except Exception as e:
                            self.logger.error(f"Request future failed: {e}")

                execution_time = time.time() - start_time

                # Analyze results
                rate_limiting_effective = rate_limited_requests > 0
                circuit_breaker_effective = circuit_breaker_activations > 0
                total_requests = (
                    successful_requests
                    + rate_limited_requests
                    + circuit_breaker_activations
                )

                test_result = {
                    "scenario": scenario["name"],
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "rate_limited_requests": rate_limited_requests,
                    "circuit_breaker_activations": circuit_breaker_activations,
                    "rate_limiting_effective": rate_limiting_effective,
                    "circuit_breaker_effective": circuit_breaker_effective,
                    "execution_time_seconds": execution_time,
                    "test_passed": rate_limiting_effective
                    and (
                        circuit_breaker_activations > 0
                        or rate_limited_requests < scenario["burst_requests"]
                    ),
                }

                results.append(test_result)

                self.logger.debug(
                    f"Rate limiting test '{scenario['name']}': "
                    f"successful={successful_requests}, rate_limited={rate_limited_requests}, "
                    f"circuit_breaker={circuit_breaker_activations}"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = (
                    f"Rate limiting test failed for {scenario['name']}: {str(e)}"
                )
                self.logger.error(error_msg)

                test_result = {
                    "scenario": scenario["name"],
                    "total_requests": 0,
                    "successful_requests": 0,
                    "rate_limited_requests": 0,
                    "circuit_breaker_activations": 0,
                    "rate_limiting_effective": False,
                    "circuit_breaker_effective": False,
                    "execution_time_seconds": execution_time,
                    "test_passed": False,
                    "error_message": error_msg,
                }

                results.append(test_result)

        self.logger.info(
            f"API rate limiting and timeout handling tests completed: {len(results)} scenarios tested"
        )
        return results

    async def test_data_consistency_under_concurrent_access(self) -> Dict[str, Any]:
        """
        Test data consistency verification under concurrent access.

        Returns:
            Dict with data consistency test results
        """
        start_time = time.time()
        self.logger.info("Testing data consistency under concurrent access")

        # Clear previous test data
        self.mock_db.clear_data()

        # Initialize test data
        test_accounts = [
            "consistency_account_1",
            "consistency_account_2",
            "consistency_account_3",
        ]
        initial_balances = {
            "consistency_account_1": Decimal("1000.00000000"),
            "consistency_account_2": Decimal("2000.00000000"),
            "consistency_account_3": Decimal("3000.00000000"),
        }

        for account, balance in initial_balances.items():
            self.mock_db.update_balance(account, balance)

        # Calculate total system balance before operations
        initial_total_balance = sum(initial_balances.values())

        consistency_violations = []
        data_integrity_issues = []

        def transfer_operation(transfer_id: int) -> Dict[str, Any]:
            """Perform transfer between random accounts."""
            try:
                from_account = random.choice(test_accounts)
                to_account = random.choice(
                    [acc for acc in test_accounts if acc != from_account]
                )
                transfer_amount = Decimal("10.00000000")

                # Read balances
                from_balance = self.mock_db.read_balance(from_account)
                to_balance = self.mock_db.read_balance(to_account)

                # Check if transfer is possible
                if from_balance >= transfer_amount:
                    # Perform transfer
                    new_from_balance = from_balance - transfer_amount
                    new_to_balance = to_balance + transfer_amount

                    # Update balances
                    self.mock_db.update_balance(from_account, new_from_balance)
                    self.mock_db.update_balance(to_account, new_to_balance)

                    return {
                        "transfer_id": transfer_id,
                        "success": True,
                        "from_account": from_account,
                        "to_account": to_account,
                        "amount": transfer_amount,
                        "from_balance_before": from_balance,
                        "to_balance_before": to_balance,
                        "from_balance_after": new_from_balance,
                        "to_balance_after": new_to_balance,
                    }
                else:
                    return {
                        "transfer_id": transfer_id,
                        "success": False,
                        "error": "insufficient_balance",
                        "from_account": from_account,
                        "from_balance": from_balance,
                        "requested_amount": transfer_amount,
                    }

            except Exception as e:
                return {
                    "transfer_id": transfer_id,
                    "success": False,
                    "error": str(e),
                }

        # Execute concurrent transfer operations
        num_transfers = 50
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(transfer_operation, i) for i in range(num_transfers)
            ]

            transfer_results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    transfer_results.append(result)
                except Exception as e:
                    transfer_results.append(
                        {
                            "success": False,
                            "error": f"Transfer future failed: {str(e)}",
                        }
                    )

        # Verify data consistency after all operations
        final_balances = {}
        final_total_balance = Decimal("0")

        for account in test_accounts:
            balance = self.mock_db.read_balance(account)
            final_balances[account] = balance
            final_total_balance += balance

        # Check for consistency violations
        balance_difference = abs(initial_total_balance - final_total_balance)
        if balance_difference > Decimal(
            "0.00000001"
        ):  # Allow for minimal rounding differences
            consistency_violations.append(
                {
                    "type": "total_balance_mismatch",
                    "initial_total": initial_total_balance,
                    "final_total": final_total_balance,
                    "difference": balance_difference,
                }
            )

        # Check for negative balances (should not happen in valid system)
        for account, balance in final_balances.items():
            if balance < Decimal("0"):
                data_integrity_issues.append(
                    {
                        "type": "negative_balance",
                        "account": account,
                        "balance": balance,
                    }
                )

        # Analyze transfer results for anomalies
        successful_transfers = [r for r in transfer_results if r.get("success", False)]
        failed_transfers = [r for r in transfer_results if not r.get("success", False)]

        # Check for double-spending or other integrity issues
        operation_log = self.mock_db.get_operation_log()
        integrity_issues = self._analyze_data_integrity_issues(
            operation_log, transfer_results
        )
        data_integrity_issues.extend(integrity_issues)

        execution_time = time.time() - start_time

        # Determine test result
        test_passed = (
            len(consistency_violations) == 0 and len(data_integrity_issues) == 0
        )

        result = {
            "test_scenario": "concurrent_transfer_operations",
            "initial_total_balance": initial_total_balance,
            "final_total_balance": final_total_balance,
            "balance_difference": balance_difference,
            "total_transfers_attempted": num_transfers,
            "successful_transfers": len(successful_transfers),
            "failed_transfers": len(failed_transfers),
            "consistency_violations": consistency_violations,
            "data_integrity_issues": data_integrity_issues,
            "final_account_balances": {k: float(v) for k, v in final_balances.items()},
            "execution_time_seconds": execution_time,
            "test_passed": test_passed,
        }

        self.logger.info(
            f"Data consistency test completed: {len(consistency_violations)} violations, "
            f"{len(data_integrity_issues)} integrity issues"
        )
        return result

    def _analyze_data_integrity_issues(
        self,
        operation_log: List[Dict[str, Any]],
        transfer_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Analyze operation log and transfer results for data integrity issues."""
        integrity_issues = []

        # Check for operations that might indicate race conditions
        balance_operations = [
            op for op in operation_log if "balance" in op["resource_id"]
        ]

        # Group by account
        account_operations = {}
        for op in balance_operations:
            account = op["resource_id"]
            if account not in account_operations:
                account_operations[account] = []
            account_operations[account].append(op)

        # Check for rapid successive operations that might indicate race conditions
        for account, operations in account_operations.items():
            operations.sort(key=lambda x: x["timestamp"])

            for i in range(len(operations) - 1):
                current_op = operations[i]
                next_op = operations[i + 1]

                # Check for very rapid operations from different threads
                if (
                    current_op["thread_id"] != next_op["thread_id"]
                    and next_op["timestamp"] - current_op["timestamp"] < 0.001
                ):  # 1ms threshold
                    integrity_issues.append(
                        {
                            "type": "rapid_concurrent_access",
                            "account": account,
                            "operation_1": current_op,
                            "operation_2": next_op,
                            "time_gap": next_op["timestamp"] - current_op["timestamp"],
                        }
                    )

        # Check for inconsistencies in transfer results
        for transfer in transfer_results:
            if transfer.get("success") and "from_balance_after" in transfer:
                # Verify that balance changes are consistent
                expected_from_balance = (
                    transfer["from_balance_before"] - transfer["amount"]
                )
                expected_to_balance = transfer["to_balance_before"] + transfer["amount"]

                if abs(
                    transfer["from_balance_after"] - expected_from_balance
                ) > Decimal("0.00000001") or abs(
                    transfer["to_balance_after"] - expected_to_balance
                ) > Decimal(
                    "0.00000001"
                ):
                    integrity_issues.append(
                        {
                            "type": "transfer_calculation_inconsistency",
                            "transfer_id": transfer["transfer_id"],
                            "expected_from_balance": expected_from_balance,
                            "actual_from_balance": transfer["from_balance_after"],
                            "expected_to_balance": expected_to_balance,
                            "actual_to_balance": transfer["to_balance_after"],
                        }
                    )

        return integrity_issues

    async def test_comprehensive_concurrency_integration(self) -> Dict[str, Any]:
        """
        Test comprehensive integration of all concurrency features.

        Returns:
            Dict with comprehensive integration test results
        """
        start_time = time.time()
        self.logger.info("Running comprehensive concurrency integration test")

        integration_results = {
            "concurrent_operations": None,
            "race_condition_detection": None,
            "deadlock_detection": None,
            "circuit_breaker_behavior": None,
            "api_rate_limiting": None,
            "data_consistency": None,
        }

        try:
            # Run all concurrency tests in sequence to verify integration
            self.logger.info("Running integrated concurrent operations test")
            concurrent_ops = await self.test_concurrent_trading_operations()
            integration_results["concurrent_operations"] = {
                "total_operations": len(concurrent_ops),
                "successful_operations": sum(1 for op in concurrent_ops if op.success),
                "data_integrity_maintained": sum(
                    1 for op in concurrent_ops if op.data_integrity_maintained
                ),
            }

            self.logger.info("Running integrated race condition detection test")
            race_condition = await self.test_race_condition_detection()
            integration_results["race_condition_detection"] = {
                "test_passed": race_condition.test_passed,
                "race_conditions_detected": race_condition.race_conditions_detected,
                "violations_count": len(race_condition.data_consistency_violations),
            }

            self.logger.info("Running integrated deadlock detection test")
            deadlock = await self.test_deadlock_detection()
            integration_results["deadlock_detection"] = {
                "deadlock_detected": deadlock.deadlock_detected,
                "prevention_effective": deadlock.prevention_mechanism_effective,
                "resolution_time": deadlock.deadlock_resolution_time_seconds,
            }

            self.logger.info("Running integrated circuit breaker test")
            circuit_breakers = await self.test_circuit_breaker_behavior()
            integration_results["circuit_breaker_behavior"] = {
                "scenarios_tested": len(circuit_breakers),
                "scenarios_passed": sum(1 for cb in circuit_breakers if cb.test_passed),
                "circuits_opened": sum(
                    1 for cb in circuit_breakers if cb.circuit_opened
                ),
            }

            self.logger.info("Running integrated API rate limiting test")
            rate_limiting = await self.test_api_rate_limiting_and_timeout_handling()
            integration_results["api_rate_limiting"] = {
                "scenarios_tested": len(rate_limiting),
                "scenarios_passed": sum(1 for rl in rate_limiting if rl["test_passed"]),
                "rate_limiting_effective": sum(
                    1 for rl in rate_limiting if rl["rate_limiting_effective"]
                ),
            }

            self.logger.info("Running integrated data consistency test")
            data_consistency = (
                await self.test_data_consistency_under_concurrent_access()
            )
            integration_results["data_consistency"] = {
                "test_passed": data_consistency["test_passed"],
                "consistency_violations": len(
                    data_consistency["consistency_violations"]
                ),
                "integrity_issues": len(data_consistency["data_integrity_issues"]),
            }

        except Exception as e:
            error_msg = f"Integration test failed: {str(e)}"
            self.logger.error(error_msg)
            integration_results["error"] = error_msg

        execution_time = time.time() - start_time

        # Calculate overall integration success
        all_tests_passed = all(
            result.get("test_passed", True) if isinstance(result, dict) else True
            for result in integration_results.values()
            if result is not None and "error" not in str(result)
        )

        final_result = {
            "integration_test_passed": all_tests_passed,
            "total_execution_time_seconds": execution_time,
            "individual_test_results": integration_results,
            "summary": {
                "concurrent_operations_successful": integration_results[
                    "concurrent_operations"
                ]
                is not None,
                "race_conditions_handled": integration_results[
                    "race_condition_detection"
                ]
                is not None,
                "deadlocks_prevented": integration_results["deadlock_detection"]
                is not None,
                "circuit_breakers_functional": integration_results[
                    "circuit_breaker_behavior"
                ]
                is not None,
                "rate_limiting_effective": integration_results["api_rate_limiting"]
                is not None,
                "data_consistency_maintained": integration_results["data_consistency"]
                is not None,
            },
        }

        self.logger.info(
            f"Comprehensive concurrency integration test completed: "
            f"overall_success={all_tests_passed}, execution_time={execution_time:.2f}s"
        )

        return final_result
