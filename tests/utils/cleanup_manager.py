"""
Test cleanup manager for safely removing existing test files and initializing new test environment.
Implements requirements 1.1, 1.2, 1.3 for existing test cleanup and initialization.
"""

import os
import shutil
import subprocess
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from .test_utilities import TestLogger


@dataclass
class CleanupReport:
    """Report of cleanup operations performed."""

    deleted_files_count: int
    deleted_directories_count: int
    deleted_files: List[str]
    deleted_directories: List[str]
    errors: List[str]
    execution_time_seconds: float
    directory_structure_before: Dict[str, any]
    directory_structure_after: Dict[str, any]
    safety_checks_passed: bool = True
    running_processes_detected: List[str] = field(default_factory=list)
    initialization_completed: bool = False


@dataclass
class InitializationReport:
    """Report of test environment initialization operations."""

    created_directories: List[str]
    created_files: List[str]
    initialization_errors: List[str]
    execution_time_seconds: float
    directory_structure_created: Dict[str, any]
    success: bool


class TestCleanupManager:
    """
    Manages cleanup of existing test files and initialization of new test environment.

    Implements requirements:
    - 1.1: System SHALL delete all files from backend/tests and frontend/__tests__
    - 1.2: System SHALL report deleted file count and directory structure
    - 1.3: System SHALL require execution stop for safe deletion if test files are running
    """

    def __init__(self):
        self.logger = TestLogger("cleanup_manager")
        self.backend_test_path = Path("backend/tests")
        self.frontend_test_path = Path("frontend/__tests__")
        self.new_test_path = Path("tests")

        # Safety mechanisms
        self._cleanup_lock = threading.Lock()
        self._active_test_processes: Set[str] = set()
        self._initialization_completed = False

        # Test environment structure
        self._test_directory_structure = {
            "tests": {
                "config": ["__init__.py", "test_config.py"],
                "utils": ["__init__.py", "test_utilities.py", "cleanup_manager.py"],
                "data": ["__init__.py", "README.md"],
                "reports": ["__init__.py", "README.md"],
                "modules": {
                    "__init__.py": None,
                    "ml_testing": ["__init__.py"],
                    "backtest_testing": ["__init__.py"],
                    "financial_testing": ["__init__.py"],
                    "concurrency_testing": ["__init__.py"],
                    "performance_testing": ["__init__.py"],
                    "security_testing": ["__init__.py"],
                },
                "orchestrator": ["__init__.py", "test_orchestrator.py"],
                "fixtures": ["__init__.py", "test_fixtures.py"],
            }
        }

    def register_active_test_process(self, process_id: str):
        """Register an active test process to prevent accidental deletion."""
        with self._cleanup_lock:
            self._active_test_processes.add(process_id)
            self.logger.debug(f"Registered active test process: {process_id}")

    def unregister_active_test_process(self, process_id: str):
        """Unregister an active test process."""
        with self._cleanup_lock:
            self._active_test_processes.discard(process_id)
            self.logger.debug(f"Unregistered active test process: {process_id}")

    def get_active_test_processes(self) -> Set[str]:
        """Get currently registered active test processes."""
        with self._cleanup_lock:
            return self._active_test_processes.copy()

    def check_running_tests(self) -> Tuple[bool, List[str]]:
        """
        Check if any test processes are currently running.
        Requirement 1.3: System SHALL require execution stop for safe deletion if test files are running.
        """
        running_processes = []

        # Check registered active processes first
        with self._cleanup_lock:
            if self._active_test_processes:
                running_processes.extend(
                    [
                        f"Registered active process: {pid}"
                        for pid in self._active_test_processes
                    ]
                )

        # Check system processes
        try:
            # Check for pytest processes
            result = subprocess.run(
                ["pgrep", "-f", "pytest"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                running_processes.extend(
                    [f"pytest process (PID: {pid})" for pid in pids if pid]
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # pgrep might not be available on Windows, try alternative
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq python.exe"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if "pytest" in result.stdout.lower():
                    running_processes.extend(["pytest processes found (Windows)"])
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.warning("Could not check for running pytest processes")

        try:
            # Check for jest/npm test processes
            result = subprocess.run(
                ["pgrep", "-f", "jest"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                running_processes.extend(
                    [f"jest process (PID: {pid})" for pid in pids if pid]
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Alternative for Windows
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq node.exe"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if "jest" in result.stdout.lower() or "test" in result.stdout.lower():
                    running_processes.extend(
                        ["jest/node test processes found (Windows)"]
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.warning("Could not check for running jest processes")

        # Check for any Python processes that might be running tests
        try:
            result = subprocess.run(
                ["pgrep", "-f", "python.*test"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                running_processes.extend(
                    [f"python test process (PID: {pid})" for pid in pids if pid]
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return len(running_processes) > 0, running_processes

    def perform_safety_checks(self, force: bool = False) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive safety checks before cleanup.

        Args:
            force: Skip safety checks if True

        Returns:
            Tuple of (safety_passed, error_messages)
        """
        if force:
            self.logger.warning("Safety checks bypassed with force=True")
            return True, []

        safety_errors = []

        # Check for running test processes
        tests_running, running_processes = self.check_running_tests()
        if tests_running:
            safety_errors.extend(
                [
                    "Test processes are currently running:",
                    *[f"  - {process}" for process in running_processes],
                    "Please stop all test processes before cleanup or use force=True",
                ]
            )

        # Check if critical files are being accessed
        critical_paths = [self.backend_test_path, self.frontend_test_path]
        for path in critical_paths:
            if path.exists():
                try:
                    # Try to create a temporary file to check write access
                    temp_file = path / ".cleanup_test"
                    temp_file.touch()
                    temp_file.unlink()
                except (PermissionError, OSError) as e:
                    safety_errors.append(f"Cannot access {path} for cleanup: {e}")

        # Check available disk space (basic check)
        try:
            import shutil

            total, used, free = shutil.disk_usage(Path.cwd())
            if free < 100 * 1024 * 1024:  # Less than 100MB free
                safety_errors.append(f"Low disk space: {free / (1024*1024):.1f}MB free")
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")

        safety_passed = len(safety_errors) == 0

        if not safety_passed:
            self.logger.error("Safety checks failed:")
            for error in safety_errors:
                self.logger.error(f"  {error}")
        else:
            self.logger.info("All safety checks passed")

        return safety_passed, safety_errors

    def get_directory_structure(self, path: Path) -> Dict[str, any]:
        """Get the structure of a directory for reporting."""
        if not path.exists():
            return {}

        structure = {}
        try:
            for item in path.rglob("*"):
                relative_path = str(item.relative_to(path))
                if item.is_file():
                    structure[relative_path] = {
                        "type": "file",
                        "size": item.stat().st_size,
                    }
                elif item.is_dir():
                    structure[relative_path] = {"type": "directory"}
        except Exception as e:
            self.logger.error(f"Error getting directory structure for {path}: {e}")

        return structure

    def count_files_and_directories(self, path: Path) -> Tuple[int, int]:
        """Count files and directories in a path."""
        if not path.exists():
            return 0, 0

        file_count = 0
        dir_count = 0

        try:
            for item in path.rglob("*"):
                if item.is_file():
                    file_count += 1
                elif item.is_dir():
                    dir_count += 1
        except Exception as e:
            self.logger.error(f"Error counting files in {path}: {e}")

        return file_count, dir_count

    def safe_delete_directory(
        self, path: Path
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Safely delete a directory and return lists of deleted files, directories, and errors.
        """
        deleted_files = []
        deleted_directories = []
        errors = []

        if not path.exists():
            self.logger.info(f"Directory {path} does not exist, skipping")
            return deleted_files, deleted_directories, errors

        try:
            # First, collect all files and directories
            all_files = []
            all_dirs = []

            for item in path.rglob("*"):
                if item.is_file():
                    all_files.append(item)
                elif item.is_dir():
                    all_dirs.append(item)

            # Delete files first
            for file_path in all_files:
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    self.logger.debug(f"Deleted file: {file_path}")
                except Exception as e:
                    error_msg = f"Failed to delete file {file_path}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

            # Delete directories (in reverse order to handle nested directories)
            for dir_path in reversed(sorted(all_dirs)):
                try:
                    if (
                        dir_path.exists()
                    ):  # Check if still exists (might be deleted with parent)
                        dir_path.rmdir()
                        deleted_directories.append(str(dir_path))
                        self.logger.debug(f"Deleted directory: {dir_path}")
                except Exception as e:
                    error_msg = f"Failed to delete directory {dir_path}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

            # Finally, delete the root directory
            try:
                if path.exists():
                    path.rmdir()
                    deleted_directories.append(str(path))
                    self.logger.info(f"Deleted root directory: {path}")
            except Exception as e:
                error_msg = f"Failed to delete root directory {path}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error during directory deletion {path}: {e}"
            errors.append(error_msg)
            self.logger.error(error_msg)

        return deleted_files, deleted_directories, errors

    def cleanup_existing_tests(self, force: bool = False) -> CleanupReport:
        """
        Clean up existing test files and directories with enhanced safety checks.

        Requirements:
        1.1: System SHALL delete all files from backend/tests and frontend/__tests__
        1.2: System SHALL report deleted file count and directory structure
        1.3: System SHALL require execution stop for safe deletion if test files are running

        Args:
            force: Skip safety checks if True

        Returns:
            CleanupReport with detailed cleanup results
        """
        with self._cleanup_lock:
            start_time = time.time()
            self.logger.info("Starting cleanup of existing test files")

            # Perform safety checks (Requirement 1.3)
            safety_passed, safety_errors = self.perform_safety_checks(force)
            tests_running, running_processes = self.check_running_tests()

            if not safety_passed:
                error_msg = "Safety checks failed. Cannot proceed with cleanup."
                self.logger.error(error_msg)
                return CleanupReport(
                    deleted_files_count=0,
                    deleted_directories_count=0,
                    deleted_files=[],
                    deleted_directories=[],
                    errors=[error_msg] + safety_errors,
                    execution_time_seconds=time.time() - start_time,
                    directory_structure_before={},
                    directory_structure_after={},
                    safety_checks_passed=False,
                    running_processes_detected=running_processes,
                    initialization_completed=False,
                )

            # Get directory structures before deletion (Requirement 1.2)
            backend_structure_before = self.get_directory_structure(
                self.backend_test_path
            )
            frontend_structure_before = self.get_directory_structure(
                self.frontend_test_path
            )

            all_deleted_files = []
            all_deleted_directories = []
            all_errors = []

            # Clean up backend tests (Requirement 1.1)
            if self.backend_test_path.exists():
                self.logger.info(
                    f"Cleaning up backend tests at {self.backend_test_path}"
                )
                backend_files, backend_dirs, backend_errors = (
                    self.safe_delete_directory(self.backend_test_path)
                )
                all_deleted_files.extend(backend_files)
                all_deleted_directories.extend(backend_dirs)
                all_errors.extend(backend_errors)

            # Clean up frontend tests (Requirement 1.1)
            if self.frontend_test_path.exists():
                self.logger.info(
                    f"Cleaning up frontend tests at {self.frontend_test_path}"
                )
                frontend_files, frontend_dirs, frontend_errors = (
                    self.safe_delete_directory(self.frontend_test_path)
                )
                all_deleted_files.extend(frontend_files)
                all_deleted_directories.extend(frontend_dirs)
                all_errors.extend(frontend_errors)

            # Get directory structures after deletion (Requirement 1.2)
            backend_structure_after = self.get_directory_structure(
                self.backend_test_path
            )
            frontend_structure_after = self.get_directory_structure(
                self.frontend_test_path
            )

            # Initialize new test environment after cleanup
            initialization_report = self.initialize_new_test_environment()
            if not initialization_report.success:
                all_errors.extend(initialization_report.initialization_errors)

            # Update execution time to include initialization
            execution_time = time.time() - start_time

            # Create cleanup report (Requirement 1.2)
            report = CleanupReport(
                deleted_files_count=len(all_deleted_files),
                deleted_directories_count=len(all_deleted_directories),
                deleted_files=all_deleted_files,
                deleted_directories=all_deleted_directories,
                errors=all_errors,
                execution_time_seconds=execution_time,
                directory_structure_before={
                    "backend/tests": backend_structure_before,
                    "frontend/__tests__": frontend_structure_before,
                },
                directory_structure_after={
                    "backend/tests": backend_structure_after,
                    "frontend/__tests__": frontend_structure_after,
                },
                safety_checks_passed=True,
                running_processes_detected=running_processes,
                initialization_completed=initialization_report.success,
            )

            # Log summary (Requirement 1.2)
            self.logger.info(
                f"Cleanup completed in {execution_time:.2f} seconds. "
                f"Deleted {report.deleted_files_count} files and "
                f"{report.deleted_directories_count} directories."
            )

            if report.errors:
                self.logger.warning(
                    f"Cleanup completed with {len(report.errors)} errors"
                )
                for error in report.errors:
                    self.logger.error(f"Cleanup error: {error}")
            else:
                self.logger.info("Cleanup completed successfully with no errors")

            return report

    def _create_directory_structure(
        self,
        base_path: Path,
        structure: Dict,
        created_dirs: List[str],
        created_files: List[str],
        errors: List[str],
    ):
        """Recursively create directory structure from configuration."""
        for name, content in structure.items():
            current_path = base_path / name

            if isinstance(content, dict):
                # It's a directory
                try:
                    current_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(current_path))
                    self.logger.debug(f"Created directory: {current_path}")

                    # Recursively create subdirectories
                    self._create_directory_structure(
                        current_path, content, created_dirs, created_files, errors
                    )
                except Exception as e:
                    error_msg = f"Failed to create directory {current_path}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

            elif isinstance(content, list):
                # It's a directory with files
                try:
                    current_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(current_path))
                    self.logger.debug(f"Created directory: {current_path}")

                    # Create files in this directory
                    for file_name in content:
                        file_path = current_path / file_name
                        try:
                            if not file_path.exists():
                                if file_name == "__init__.py":
                                    file_path.write_text(
                                        f"# {current_path.name} test module\n"
                                    )
                                elif file_name == "README.md":
                                    file_path.write_text(
                                        f"# {current_path.name}\n\nTest data and resources.\n"
                                    )
                                else:
                                    file_path.touch()

                                created_files.append(str(file_path))
                                self.logger.debug(f"Created file: {file_path}")
                        except Exception as e:
                            error_msg = f"Failed to create file {file_path}: {e}"
                            errors.append(error_msg)
                            self.logger.error(error_msg)

                except Exception as e:
                    error_msg = f"Failed to create directory {current_path}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

            elif content is None:
                # It's a single file
                try:
                    if not current_path.exists():
                        if current_path.suffix == ".py":
                            current_path.write_text(f"# {current_path.stem} module\n")
                        else:
                            current_path.touch()

                        created_files.append(str(current_path))
                        self.logger.debug(f"Created file: {current_path}")
                except Exception as e:
                    error_msg = f"Failed to create file {current_path}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

    def initialize_new_test_environment(self) -> InitializationReport:
        """
        Initialize the new test environment with proper structure.

        Creates the comprehensive test directory structure and required files.

        Returns:
            InitializationReport with initialization results
        """
        start_time = time.time()
        self.logger.info("Initializing new test environment")

        created_directories = []
        created_files = []
        initialization_errors = []

        try:
            # Create the main test directory structure
            self._create_directory_structure(
                Path("."),
                self._test_directory_structure,
                created_directories,
                created_files,
                initialization_errors,
            )

            # Create additional configuration files
            self._create_configuration_files(created_files, initialization_errors)

            # Verify the structure was created correctly
            verification_errors = self._verify_test_structure()
            initialization_errors.extend(verification_errors)

            execution_time = time.time() - start_time
            success = len(initialization_errors) == 0

            if success:
                self._initialization_completed = True
                self.logger.info(
                    f"Test environment initialization completed successfully in {execution_time:.2f} seconds. "
                    f"Created {len(created_directories)} directories and {len(created_files)} files."
                )
            else:
                self.logger.error(
                    f"Test environment initialization completed with {len(initialization_errors)} errors"
                )
                for error in initialization_errors:
                    self.logger.error(f"Initialization error: {error}")

            return InitializationReport(
                created_directories=created_directories,
                created_files=created_files,
                initialization_errors=initialization_errors,
                execution_time_seconds=execution_time,
                directory_structure_created=self.get_directory_structure(
                    self.new_test_path
                ),
                success=success,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Critical error during test environment initialization: {e}"
            self.logger.error(error_msg)

            return InitializationReport(
                created_directories=created_directories,
                created_files=created_files,
                initialization_errors=[error_msg],
                execution_time_seconds=execution_time,
                directory_structure_created={},
                success=False,
            )

    def _create_configuration_files(self, created_files: List[str], errors: List[str]):
        """Create additional configuration files for the test environment."""
        config_files = {
            "tests/pytest.ini": """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    security: Security tests
    ml: Machine learning tests
    financial: Financial calculation tests
    concurrency: Concurrency tests
    backtest: Backtest tests
""",
            "tests/.gitignore": """# Test artifacts
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.coverage
.pytest_cache/
.tox/
htmlcov/
.cache/
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Test data
/data/temp/
/data/cache/
/reports/temp/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
""",
            "tests/conftest.py": """# Global pytest configuration and fixtures
import pytest
import asyncio
from pathlib import Path

from fixtures.test_fixtures import cleanup_test_fixtures


@pytest.fixture(scope="session")
def event_loop():
    \"\"\"Create an instance of the default event loop for the test session.\"\"\"
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True, scope="session")
def cleanup_after_tests():
    \"\"\"Cleanup test fixtures after all tests complete.\"\"\"
    yield
    cleanup_test_fixtures()


@pytest.fixture
def test_data_dir():
    \"\"\"Provide path to test data directory.\"\"\"
    return Path("tests/data")


@pytest.fixture
def test_reports_dir():
    \"\"\"Provide path to test reports directory.\"\"\"
    return Path("tests/reports")
""",
        }

        for file_path, content in config_files.items():
            try:
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)

                if not path.exists():
                    path.write_text(content)
                    created_files.append(str(path))
                    self.logger.debug(f"Created configuration file: {path}")
            except Exception as e:
                error_msg = f"Failed to create configuration file {file_path}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)

    def _verify_test_structure(self) -> List[str]:
        """Verify that the test structure was created correctly."""
        verification_errors = []

        # Check critical directories
        critical_dirs = [
            "tests/config",
            "tests/utils",
            "tests/modules",
            "tests/orchestrator",
            "tests/fixtures",
            "tests/data",
            "tests/reports",
        ]

        for dir_path in critical_dirs:
            path = Path(dir_path)
            if not path.exists():
                verification_errors.append(f"Critical directory missing: {dir_path}")
            elif not path.is_dir():
                verification_errors.append(
                    f"Path exists but is not a directory: {dir_path}"
                )

        # Check critical files
        critical_files = [
            "tests/__init__.py",
            "tests/config/__init__.py",
            "tests/utils/__init__.py",
            "tests/orchestrator/__init__.py",
            "tests/fixtures/__init__.py",
        ]

        for file_path in critical_files:
            path = Path(file_path)
            if not path.exists():
                verification_errors.append(f"Critical file missing: {file_path}")
            elif not path.is_file():
                verification_errors.append(
                    f"Path exists but is not a file: {file_path}"
                )

        return verification_errors

    def create_new_test_structure(self) -> InitializationReport:
        """
        Create the new test directory structure.

        This method is kept for backward compatibility but now uses the enhanced
        initialization system.
        """
        return self.initialize_new_test_environment()

    def is_initialization_completed(self) -> bool:
        """Check if test environment initialization has been completed."""
        return self._initialization_completed

    def reset_initialization_status(self):
        """Reset the initialization status (for testing purposes)."""
        self._initialization_completed = False
        self.logger.debug("Reset initialization status")
