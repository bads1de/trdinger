"""
Test cleanup manager for safely removing existing test files.
Implements requirements 1.1, 1.2, 1.3 for existing test cleanup.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class TestCleanupManager:
    """Manages cleanup of existing test files and directories."""

    def __init__(self):
        self.logger = TestLogger("cleanup_manager")
        self.backend_test_path = Path("backend/tests")
        self.frontend_test_path = Path("frontend/__tests__")

    def check_running_tests(self) -> Tuple[bool, List[str]]:
        """
        Check if any test processes are currently running.
        Requirement 1.3: System SHALL require execution stop for safe deletion if test files are running.
        """
        running_processes = []

        try:
            # Check for pytest processes
            result = subprocess.run(
                ["pgrep", "-f", "pytest"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                running_processes.extend(["pytest processes found"])
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
                self.logger.warning("Could not check for running test processes")

        try:
            # Check for jest/npm test processes
            result = subprocess.run(
                ["pgrep", "-f", "jest"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                running_processes.extend(["jest processes found"])
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

        return len(running_processes) > 0, running_processes

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
        Clean up existing test files and directories.

        Requirements:
        1.1: System SHALL delete all files from backend/tests and frontend/__tests__
        1.2: System SHALL report deleted file count and directory structure
        1.3: System SHALL require execution stop for safe deletion if test files are running
        """
        start_time = time.time()

        # Check for running tests (Requirement 1.3)
        if not force:
            tests_running, running_processes = self.check_running_tests()
            if tests_running:
                error_msg = (
                    f"Cannot safely delete test files while tests are running. "
                    f"Running processes: {running_processes}. "
                    f"Please stop all test processes and try again, or use force=True."
                )
                self.logger.error(error_msg)
                return CleanupReport(
                    deleted_files_count=0,
                    deleted_directories_count=0,
                    deleted_files=[],
                    deleted_directories=[],
                    errors=[error_msg],
                    execution_time_seconds=time.time() - start_time,
                    directory_structure_before={},
                    directory_structure_after={},
                )

        # Get directory structures before deletion (Requirement 1.2)
        backend_structure_before = self.get_directory_structure(self.backend_test_path)
        frontend_structure_before = self.get_directory_structure(
            self.frontend_test_path
        )

        self.logger.info("Starting cleanup of existing test files")

        all_deleted_files = []
        all_deleted_directories = []
        all_errors = []

        # Clean up backend tests (Requirement 1.1)
        if self.backend_test_path.exists():
            self.logger.info(f"Cleaning up backend tests at {self.backend_test_path}")
            backend_files, backend_dirs, backend_errors = self.safe_delete_directory(
                self.backend_test_path
            )
            all_deleted_files.extend(backend_files)
            all_deleted_directories.extend(backend_dirs)
            all_errors.extend(backend_errors)

        # Clean up frontend tests (Requirement 1.1)
        if self.frontend_test_path.exists():
            self.logger.info(f"Cleaning up frontend tests at {self.frontend_test_path}")
            frontend_files, frontend_dirs, frontend_errors = self.safe_delete_directory(
                self.frontend_test_path
            )
            all_deleted_files.extend(frontend_files)
            all_deleted_directories.extend(frontend_dirs)
            all_errors.extend(frontend_errors)

        # Get directory structures after deletion (Requirement 1.2)
        backend_structure_after = self.get_directory_structure(self.backend_test_path)
        frontend_structure_after = self.get_directory_structure(self.frontend_test_path)

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
        )

        # Log summary (Requirement 1.2)
        self.logger.info(
            f"Cleanup completed in {execution_time:.2f} seconds. "
            f"Deleted {report.deleted_files_count} files and "
            f"{report.deleted_directories_count} directories."
        )

        if report.errors:
            self.logger.warning(f"Cleanup completed with {len(report.errors)} errors")
            for error in report.errors:
                self.logger.error(f"Cleanup error: {error}")

        return report

    def create_new_test_structure(self) -> None:
        """Create the new test directory structure."""
        self.logger.info("Creating new test directory structure")

        # Create main test directories
        test_dirs = [
            "tests/config",
            "tests/utils",
            "tests/data",
            "tests/reports",
            "tests/modules/ml_testing",
            "tests/modules/backtest_testing",
            "tests/modules/financial_testing",
            "tests/modules/concurrency_testing",
            "tests/modules/performance_testing",
            "tests/modules/security_testing",
            "tests/orchestrator",
            "tests/fixtures",
        ]

        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

            # Create __init__.py files for Python packages
            init_file = Path(dir_path) / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Test module\n")

        self.logger.info("New test directory structure created successfully")
