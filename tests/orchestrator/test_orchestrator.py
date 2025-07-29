"""
Test orchestrator for comprehensive testing framework.
Coordinates all test modules and manages test execution flow.
"""

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable

try:
    from ..config.test_config import TestConfig, get_test_config
    from ..utils.test_utilities import TestLogger
    from ..utils.cleanup_manager import TestCleanupManager, CleanupReport
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.test_config import TestConfig, get_test_config
    from utils.test_utilities import TestLogger
    from utils.cleanup_manager import TestCleanupManager, CleanupReport


class TestStatus(Enum):
    """Test execution status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestModuleResult:
    """Result from a single test module."""

    module_name: str
    status: TestStatus
    execution_time_seconds: float
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    error_messages: List[str]
    detailed_results: Dict[str, Any]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exception_details: Optional[str] = None


@dataclass
class TestResults:
    """Comprehensive test results from all modules."""

    overall_status: TestStatus
    total_execution_time_seconds: float
    modules_results: Dict[str, TestModuleResult]
    cleanup_report: Optional[CleanupReport]
    timestamp: datetime
    configuration: TestConfig
    summary: Dict[str, int] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)


class TestModuleInterface:
    """Interface that all test modules must implement."""

    async def run_tests(self) -> TestModuleResult:
        """Run all tests for this module and return results."""
        raise NotImplementedError("Test modules must implement run_tests method")

    def get_module_name(self) -> str:
        """Get the name of this test module."""
        raise NotImplementedError("Test modules must implement get_module_name method")


class TestOrchestrator:
    """
    Main orchestrator for the comprehensive testing framework.
    Manages test execution, cleanup, and reporting.

    Requirements implemented:
    - 8.1: Async test execution with proper error handling
    - 8.2: Test result aggregation and status tracking
    - 8.3: Module coordination and management
    """

    def __init__(self, config: TestConfig = None):
        self.config = config or get_test_config()
        self.logger = TestLogger("test_orchestrator", self.config.log_level)
        self.cleanup_manager = TestCleanupManager()
        self.test_modules: Dict[str, Optional[TestModuleInterface]] = {}
        self.execution_lock = asyncio.Lock()
        self.current_execution_id: Optional[str] = None
        self._initialize_test_modules()

    def _initialize_test_modules(self):
        """Initialize all test modules based on configuration."""
        self.logger.info("Initializing test modules")

        # Initialize test module registry
        # These will be populated when actual test modules are implemented
        self.test_modules = {
            "ml_testing": None,  # Will be MLModelTester
            "backtest_testing": None,  # Will be BacktestTester
            "financial_testing": None,  # Will be FinancialCalculationTester
            "concurrency_testing": None,  # Will be ConcurrencyTester
            "performance_testing": None,  # Will be PerformanceTester
            "security_testing": None,  # Will be SecurityTester
        }

        self.logger.info(f"Initialized {len(self.test_modules)} test module slots")

    def register_test_module(
        self, module_name: str, module_instance: TestModuleInterface
    ):
        """Register a test module with the orchestrator."""
        if module_name not in self.test_modules:
            raise ValueError(f"Unknown test module: {module_name}")

        self.test_modules[module_name] = module_instance
        self.logger.info(f"Registered test module: {module_name}")

    def unregister_test_module(self, module_name: str):
        """Unregister a test module from the orchestrator."""
        if module_name in self.test_modules:
            self.test_modules[module_name] = None
            self.logger.info(f"Unregistered test module: {module_name}")

    async def _execute_test_module(
        self,
        module_name: str,
        module_instance: TestModuleInterface,
        timeout_seconds: float = 300.0,
    ) -> TestModuleResult:
        """
        Execute a single test module with proper error handling and timeout.

        Args:
            module_name: Name of the test module
            module_instance: Instance of the test module
            timeout_seconds: Maximum execution time before timeout

        Returns:
            TestModuleResult with execution details
        """
        start_time = datetime.now()
        self.logger.info(f"Starting execution of {module_name}")

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                module_instance.run_tests(), timeout=timeout_seconds
            )

            # Ensure result has proper timing information
            result.start_time = start_time
            result.end_time = datetime.now()
            result.execution_time_seconds = (
                result.end_time - start_time
            ).total_seconds()

            self.logger.info(
                f"Completed {module_name}: {result.status.value} "
                f"({result.tests_passed}/{result.tests_run} passed) "
                f"in {result.execution_time_seconds:.2f}s"
            )

            return result

        except asyncio.TimeoutError:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            error_msg = f"Test module {module_name} timed out after {timeout_seconds}s"

            self.logger.error(error_msg)

            return TestModuleResult(
                module_name=module_name,
                status=TestStatus.FAILED,
                execution_time_seconds=execution_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=1,
                tests_skipped=0,
                error_messages=[error_msg],
                detailed_results={"timeout": True, "timeout_seconds": timeout_seconds},
                start_time=start_time,
                end_time=end_time,
                exception_details=f"TimeoutError: {timeout_seconds}s",
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            error_msg = f"Test module {module_name} failed with exception: {str(e)}"
            exception_details = traceback.format_exc()

            self.logger.error(f"{error_msg}\n{exception_details}")

            return TestModuleResult(
                module_name=module_name,
                status=TestStatus.FAILED,
                execution_time_seconds=execution_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=1,
                tests_skipped=0,
                error_messages=[error_msg],
                detailed_results={"exception": str(e)},
                start_time=start_time,
                end_time=end_time,
                exception_details=exception_details,
            )

    async def _create_placeholder_result(self, module_name: str) -> TestModuleResult:
        """Create a placeholder result for unimplemented modules."""
        start_time = datetime.now()

        # Simulate brief execution time
        await asyncio.sleep(0.1)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return TestModuleResult(
            module_name=module_name,
            status=TestStatus.SKIPPED,
            execution_time_seconds=execution_time,
            tests_run=0,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=1,
            error_messages=[f"{module_name} module not yet implemented"],
            detailed_results={"placeholder": True},
            start_time=start_time,
            end_time=end_time,
        )

    def _calculate_overall_status(
        self, module_results: Dict[str, TestModuleResult]
    ) -> TestStatus:
        """Calculate overall test status based on module results."""
        if not module_results:
            return TestStatus.NOT_STARTED

        statuses = [result.status for result in module_results.values()]

        # If any module failed, overall status is failed
        if TestStatus.FAILED in statuses:
            return TestStatus.FAILED

        # If any module is in progress, overall status is in progress
        if TestStatus.IN_PROGRESS in statuses:
            return TestStatus.IN_PROGRESS

        # If all modules are skipped, overall status is skipped
        if all(status == TestStatus.SKIPPED for status in statuses):
            return TestStatus.SKIPPED

        # If we have completed modules, overall status is completed
        if TestStatus.COMPLETED in statuses:
            return TestStatus.COMPLETED

        return TestStatus.NOT_STARTED

    def _generate_test_summary(
        self, module_results: Dict[str, TestModuleResult]
    ) -> Dict[str, int]:
        """Generate summary statistics from module results."""
        summary = {
            "total_modules": len(module_results),
            "modules_completed": 0,
            "modules_failed": 0,
            "modules_skipped": 0,
            "total_tests_run": 0,
            "total_tests_passed": 0,
            "total_tests_failed": 0,
            "total_tests_skipped": 0,
        }

        for result in module_results.values():
            if result.status == TestStatus.COMPLETED:
                summary["modules_completed"] += 1
            elif result.status == TestStatus.FAILED:
                summary["modules_failed"] += 1
            elif result.status == TestStatus.SKIPPED:
                summary["modules_skipped"] += 1

            summary["total_tests_run"] += result.tests_run
            summary["total_tests_passed"] += result.tests_passed
            summary["total_tests_failed"] += result.tests_failed
            summary["total_tests_skipped"] += result.tests_skipped

        return summary

    async def setup_test_infrastructure(self) -> CleanupReport:
        """
        Set up the test infrastructure by cleaning up existing tests and creating new structure.
        This implements the main functionality for task 1.
        """
        async with self.execution_lock:
            self.logger.info("Setting up test infrastructure")

            # Clean up existing tests if configured
            cleanup_report = None
            if self.config.cleanup_existing_tests:
                self.logger.info("Cleaning up existing test files")

                # Run cleanup in executor to avoid blocking async loop
                loop = asyncio.get_event_loop()
                cleanup_report = await loop.run_in_executor(
                    None, self.cleanup_manager.cleanup_existing_tests
                )

                if cleanup_report.errors:
                    self.logger.error(
                        f"Cleanup completed with {len(cleanup_report.errors)} errors"
                    )
                    for error in cleanup_report.errors:
                        self.logger.error(f"Cleanup error: {error}")
                else:
                    self.logger.info(
                        f"Successfully cleaned up {cleanup_report.deleted_files_count} files "
                        f"and {cleanup_report.deleted_directories_count} directories"
                    )

            # Create new test structure
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.cleanup_manager.create_new_test_structure
            )

            self.logger.info("Test infrastructure setup completed")
            return cleanup_report

    async def run_all_tests(self, parallel: bool = True) -> TestResults:
        """
        Run all test modules and return comprehensive results.

        Args:
            parallel: Whether to run test modules in parallel (default: True)

        Returns:
            TestResults with comprehensive execution results
        """
        async with self.execution_lock:
            execution_id = f"all_tests_{int(time.time())}"
            self.current_execution_id = execution_id

            start_time = time.time()
            timestamp = datetime.now()

            self.logger.info(
                f"Starting comprehensive test execution (ID: {execution_id})"
            )

            try:
                # Setup infrastructure first
                cleanup_report = await self.setup_test_infrastructure()

                # Initialize results
                module_results = {}

                if parallel:
                    # Run test modules in parallel
                    self.logger.info("Running test modules in parallel")

                    # Create tasks for all modules
                    tasks = []
                    for module_name, module_instance in self.test_modules.items():
                        if module_instance is not None:
                            task = asyncio.create_task(
                                self._execute_test_module(module_name, module_instance),
                                name=f"test_module_{module_name}",
                            )
                        else:
                            task = asyncio.create_task(
                                self._create_placeholder_result(module_name),
                                name=f"placeholder_{module_name}",
                            )
                        tasks.append((module_name, task))

                    # Wait for all tasks to complete
                    for module_name, task in tasks:
                        try:
                            result = await task
                            module_results[module_name] = result
                        except Exception as e:
                            self.logger.error(f"Task for {module_name} failed: {e}")
                            # Create error result
                            module_results[module_name] = TestModuleResult(
                                module_name=module_name,
                                status=TestStatus.FAILED,
                                execution_time_seconds=0.0,
                                tests_run=0,
                                tests_passed=0,
                                tests_failed=1,
                                tests_skipped=0,
                                error_messages=[f"Task execution failed: {str(e)}"],
                                detailed_results={"task_error": True},
                                exception_details=traceback.format_exc(),
                            )

                else:
                    # Run test modules sequentially
                    self.logger.info("Running test modules sequentially")

                    for module_name, module_instance in self.test_modules.items():
                        if module_instance is not None:
                            result = await self._execute_test_module(
                                module_name, module_instance
                            )
                        else:
                            result = await self._create_placeholder_result(module_name)

                        module_results[module_name] = result

                # Calculate overall status and summary
                overall_status = self._calculate_overall_status(module_results)
                summary = self._generate_test_summary(module_results)
                total_execution_time = time.time() - start_time

                # Create execution metadata
                execution_metadata = {
                    "execution_id": execution_id,
                    "parallel_execution": parallel,
                    "modules_registered": len(
                        [m for m in self.test_modules.values() if m is not None]
                    ),
                    "modules_placeholder": len(
                        [m for m in self.test_modules.values() if m is None]
                    ),
                }

                # Create comprehensive results
                results = TestResults(
                    overall_status=overall_status,
                    total_execution_time_seconds=total_execution_time,
                    modules_results=module_results,
                    cleanup_report=cleanup_report,
                    timestamp=timestamp,
                    configuration=self.config,
                    summary=summary,
                    execution_metadata=execution_metadata,
                )

                self.logger.info(
                    f"Comprehensive test execution completed in {total_execution_time:.2f} seconds. "
                    f"Status: {overall_status.value}, "
                    f"Modules: {summary['modules_completed']} completed, "
                    f"{summary['modules_failed']} failed, "
                    f"{summary['modules_skipped']} skipped"
                )

                return results

            except Exception as e:
                self.logger.error(f"Test execution failed: {e}")
                raise
            finally:
                self.current_execution_id = None

    async def run_specific_tests(
        self, test_types: List[str], parallel: bool = True
    ) -> TestResults:
        """
        Run only specific test modules.

        Args:
            test_types: List of test module names to run
            parallel: Whether to run test modules in parallel (default: True)

        Returns:
            TestResults with execution results for specified modules
        """
        async with self.execution_lock:
            execution_id = f"specific_tests_{int(time.time())}"
            self.current_execution_id = execution_id

            start_time = time.time()
            timestamp = datetime.now()

            self.logger.info(
                f"Running specific tests: {test_types} (ID: {execution_id})"
            )

            # Validate test types
            invalid_types = [t for t in test_types if t not in self.test_modules]
            if invalid_types:
                raise ValueError(f"Invalid test types: {invalid_types}")

            try:
                # Setup infrastructure first
                cleanup_report = await self.setup_test_infrastructure()

                # Initialize results
                module_results = {}

                if parallel and len(test_types) > 1:
                    # Run specified test modules in parallel
                    self.logger.info("Running specified test modules in parallel")

                    # Create tasks for specified modules
                    tasks = []
                    for module_name in test_types:
                        module_instance = self.test_modules[module_name]
                        if module_instance is not None:
                            task = asyncio.create_task(
                                self._execute_test_module(module_name, module_instance),
                                name=f"test_module_{module_name}",
                            )
                        else:
                            task = asyncio.create_task(
                                self._create_placeholder_result(module_name),
                                name=f"placeholder_{module_name}",
                            )
                        tasks.append((module_name, task))

                    # Wait for all tasks to complete
                    for module_name, task in tasks:
                        try:
                            result = await task
                            module_results[module_name] = result
                        except Exception as e:
                            self.logger.error(f"Task for {module_name} failed: {e}")
                            # Create error result
                            module_results[module_name] = TestModuleResult(
                                module_name=module_name,
                                status=TestStatus.FAILED,
                                execution_time_seconds=0.0,
                                tests_run=0,
                                tests_passed=0,
                                tests_failed=1,
                                tests_skipped=0,
                                error_messages=[f"Task execution failed: {str(e)}"],
                                detailed_results={"task_error": True},
                                exception_details=traceback.format_exc(),
                            )

                else:
                    # Run specified test modules sequentially
                    self.logger.info("Running specified test modules sequentially")

                    for module_name in test_types:
                        module_instance = self.test_modules[module_name]
                        if module_instance is not None:
                            result = await self._execute_test_module(
                                module_name, module_instance
                            )
                        else:
                            result = await self._create_placeholder_result(module_name)

                        module_results[module_name] = result

                # Calculate overall status and summary
                overall_status = self._calculate_overall_status(module_results)
                summary = self._generate_test_summary(module_results)
                total_execution_time = time.time() - start_time

                # Create execution metadata
                execution_metadata = {
                    "execution_id": execution_id,
                    "parallel_execution": parallel,
                    "requested_modules": test_types,
                    "modules_registered": len(
                        [
                            m
                            for name, m in self.test_modules.items()
                            if name in test_types and m is not None
                        ]
                    ),
                    "modules_placeholder": len(
                        [
                            m
                            for name, m in self.test_modules.items()
                            if name in test_types and m is None
                        ]
                    ),
                }

                # Create comprehensive results
                results = TestResults(
                    overall_status=overall_status,
                    total_execution_time_seconds=total_execution_time,
                    modules_results=module_results,
                    cleanup_report=cleanup_report,
                    timestamp=timestamp,
                    configuration=self.config,
                    summary=summary,
                    execution_metadata=execution_metadata,
                )

                self.logger.info(
                    f"Specific test execution completed in {total_execution_time:.2f} seconds. "
                    f"Status: {overall_status.value}, "
                    f"Modules: {summary['modules_completed']} completed, "
                    f"{summary['modules_failed']} failed, "
                    f"{summary['modules_skipped']} skipped"
                )

                return results

            except Exception as e:
                self.logger.error(f"Specific test execution failed: {e}")
                raise
            finally:
                self.current_execution_id = None

    def get_test_status(self) -> Dict[str, Any]:
        """Get current status of the test system."""
        registered_modules = {
            name: instance is not None for name, instance in self.test_modules.items()
        }

        return {
            "test_modules": list(self.test_modules.keys()),
            "registered_modules": registered_modules,
            "modules_registered_count": len(
                [m for m in self.test_modules.values() if m is not None]
            ),
            "current_execution_id": self.current_execution_id,
            "execution_in_progress": self.current_execution_id is not None,
            "configuration": {
                "cleanup_existing_tests": self.config.cleanup_existing_tests,
                "test_data_directory": self.config.test_data_directory,
                "log_level": self.config.log_level,
                "report_formats": self.config.reporting_config.report_formats,
                "ci_integration_enabled": self.config.reporting_config.ci_integration_enabled,
            },
            "infrastructure_ready": self.validate_test_environment()["all_valid"],
        }

    def validate_test_environment(self) -> Dict[str, Any]:
        """Validate that the test environment is properly set up."""
        from pathlib import Path

        validation_results = {}

        # Check if test directories exist
        required_dirs = [
            "tests/config",
            "tests/utils",
            "tests/data",
            "tests/reports",
            "tests/modules",
            "tests/orchestrator",
            "tests/fixtures",
        ]

        for dir_path in required_dirs:
            validation_results[f"directory_{dir_path}"] = Path(dir_path).exists()

        # Check if configuration is valid
        validation_results["configuration_valid"] = self.config is not None

        # Check if cleanup manager is available
        validation_results["cleanup_manager_available"] = (
            self.cleanup_manager is not None
        )

        # Check if required config files exist
        config_files = [
            "tests/config/test_config.py",
            "tests/utils/test_utilities.py",
            "tests/utils/cleanup_manager.py",
            "tests/fixtures/test_fixtures.py",
        ]

        for config_file in config_files:
            validation_results[f"config_file_{config_file}"] = Path(
                config_file
            ).exists()

        # Overall validation
        validation_results["all_valid"] = all(validation_results.values())
        validation_results["validation_errors"] = [
            key
            for key, value in validation_results.items()
            if not value and key != "all_valid" and key != "validation_errors"
        ]

        return validation_results

    async def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status with detailed information."""
        return {
            "current_execution_id": self.current_execution_id,
            "execution_in_progress": self.current_execution_id is not None,
            "execution_lock_locked": self.execution_lock.locked(),
            "test_modules_status": {
                name: {
                    "registered": instance is not None,
                    "module_type": type(instance).__name__ if instance else None,
                }
                for name, instance in self.test_modules.items()
            },
            "system_status": self.get_test_status(),
        }

    async def stop_current_execution(self) -> bool:
        """
        Attempt to stop the current test execution.

        Returns:
            bool: True if execution was stopped, False if no execution was running
        """
        if self.current_execution_id is None:
            self.logger.info("No test execution currently running")
            return False

        self.logger.warning(
            f"Attempting to stop execution: {self.current_execution_id}"
        )

        # Note: This is a basic implementation. In a more sophisticated system,
        # we would need to track and cancel individual tasks
        execution_id = self.current_execution_id
        self.current_execution_id = None

        self.logger.info(f"Marked execution {execution_id} for stopping")
        return True

    def get_available_test_modules(self) -> List[str]:
        """Get list of available test module names."""
        return list(self.test_modules.keys())

    def get_registered_test_modules(self) -> List[str]:
        """Get list of currently registered test module names."""
        return [
            name for name, instance in self.test_modules.items() if instance is not None
        ]

    def get_unregistered_test_modules(self) -> List[str]:
        """Get list of unregistered test module names."""
        return [
            name for name, instance in self.test_modules.items() if instance is None
        ]
