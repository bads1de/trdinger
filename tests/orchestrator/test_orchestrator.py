"""
Test orchestrator for comprehensive testing framework.
Coordinates all test modules and manages test execution flow.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from ..config.test_config import TestConfig, get_test_config
from ..utils.test_utilities import TestLogger
from ..utils.cleanup_manager import TestCleanupManager, CleanupReport


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


@dataclass
class TestResults:
    """Comprehensive test results from all modules."""

    overall_status: TestStatus
    total_execution_time_seconds: float
    modules_results: Dict[str, TestModuleResult]
    cleanup_report: Optional[CleanupReport]
    timestamp: datetime
    configuration: TestConfig


class TestOrchestrator:
    """
    Main orchestrator for the comprehensive testing framework.
    Manages test execution, cleanup, and reporting.
    """

    def __init__(self, config: TestConfig = None):
        self.config = config or get_test_config()
        self.logger = TestLogger("test_orchestrator")
        self.cleanup_manager = TestCleanupManager()
        self.test_modules = {}
        self._initialize_test_modules()

    def _initialize_test_modules(self):
        """Initialize all test modules based on configuration."""
        self.logger.info("Initializing test modules")

        # Note: These will be implemented in subsequent tasks
        # For now, we're just setting up the structure
        self.test_modules = {
            "ml_testing": None,  # Will be MLModelTester
            "backtest_testing": None,  # Will be BacktestTester
            "financial_testing": None,  # Will be FinancialCalculationTester
            "concurrency_testing": None,  # Will be ConcurrencyTester
            "performance_testing": None,  # Will be PerformanceTester
            "security_testing": None,  # Will be SecurityTester
        }

        self.logger.info(f"Initialized {len(self.test_modules)} test modules")

    async def setup_test_infrastructure(self) -> CleanupReport:
        """
        Set up the test infrastructure by cleaning up existing tests and creating new structure.
        This implements the main functionality for task 1.
        """
        self.logger.info("Setting up test infrastructure")

        # Clean up existing tests if configured
        cleanup_report = None
        if self.config.cleanup_existing_tests:
            self.logger.info("Cleaning up existing test files")
            cleanup_report = self.cleanup_manager.cleanup_existing_tests()

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
        self.cleanup_manager.create_new_test_structure()

        self.logger.info("Test infrastructure setup completed")
        return cleanup_report

    async def run_all_tests(self) -> TestResults:
        """
        Run all test modules and return comprehensive results.
        This will be expanded in subsequent tasks.
        """
        start_time = time.time()
        self.logger.info("Starting comprehensive test execution")

        # Setup infrastructure first
        cleanup_report = await self.setup_test_infrastructure()

        # Initialize results
        module_results = {}
        overall_status = TestStatus.COMPLETED

        # Run each test module (placeholder for now)
        for module_name, module_instance in self.test_modules.items():
            self.logger.info(f"Running {module_name} tests")

            module_start_time = time.time()

            # Placeholder result - will be replaced with actual test execution
            module_result = TestModuleResult(
                module_name=module_name,
                status=TestStatus.SKIPPED,  # Skipped until modules are implemented
                execution_time_seconds=time.time() - module_start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=1,
                error_messages=[f"{module_name} module not yet implemented"],
                detailed_results={},
            )

            module_results[module_name] = module_result
            self.logger.info(
                f"Completed {module_name} tests: {module_result.status.value}"
            )

        total_execution_time = time.time() - start_time

        # Create comprehensive results
        results = TestResults(
            overall_status=overall_status,
            total_execution_time_seconds=total_execution_time,
            modules_results=module_results,
            cleanup_report=cleanup_report,
            timestamp=datetime.now(),
            configuration=self.config,
        )

        self.logger.info(
            f"Comprehensive test execution completed in {total_execution_time:.2f} seconds"
        )

        return results

    async def run_specific_tests(self, test_types: List[str]) -> TestResults:
        """
        Run only specific test modules.

        Args:
            test_types: List of test module names to run
        """
        start_time = time.time()
        self.logger.info(f"Running specific tests: {test_types}")

        # Validate test types
        invalid_types = [t for t in test_types if t not in self.test_modules]
        if invalid_types:
            raise ValueError(f"Invalid test types: {invalid_types}")

        # Setup infrastructure first
        cleanup_report = await self.setup_test_infrastructure()

        # Initialize results
        module_results = {}
        overall_status = TestStatus.COMPLETED

        # Run only specified test modules
        for module_name in test_types:
            if module_name in self.test_modules:
                self.logger.info(f"Running {module_name} tests")

                module_start_time = time.time()

                # Placeholder result - will be replaced with actual test execution
                module_result = TestModuleResult(
                    module_name=module_name,
                    status=TestStatus.SKIPPED,  # Skipped until modules are implemented
                    execution_time_seconds=time.time() - module_start_time,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    tests_skipped=1,
                    error_messages=[f"{module_name} module not yet implemented"],
                    detailed_results={},
                )

                module_results[module_name] = module_result
                self.logger.info(
                    f"Completed {module_name} tests: {module_result.status.value}"
                )

        total_execution_time = time.time() - start_time

        # Create comprehensive results
        results = TestResults(
            overall_status=overall_status,
            total_execution_time_seconds=total_execution_time,
            modules_results=module_results,
            cleanup_report=cleanup_report,
            timestamp=datetime.now(),
            configuration=self.config,
        )

        self.logger.info(
            f"Specific test execution completed in {total_execution_time:.2f} seconds"
        )

        return results

    def get_test_status(self) -> Dict[str, Any]:
        """Get current status of the test system."""
        return {
            "test_modules": list(self.test_modules.keys()),
            "configuration": {
                "cleanup_existing_tests": self.config.cleanup_existing_tests,
                "test_data_directory": self.config.test_data_directory,
                "log_level": self.config.log_level,
            },
            "infrastructure_ready": True,  # Will be more sophisticated later
        }

    def validate_test_environment(self) -> Dict[str, bool]:
        """Validate that the test environment is properly set up."""
        validation_results = {}

        # Check if test directories exist
        from pathlib import Path

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

        return validation_results
