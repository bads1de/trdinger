#!/usr/bin/env python3
"""
Test infrastructure setup script.
This script implements task 1 of the comprehensive testing overhaul.

Usage:
    python tests/setup_test_infrastructure.py [--force] [--no-cleanup]
    
Options:
    --force: Force cleanup even if tests are running
    --no-cleanup: Skip cleanup of existing tests, only create new structure
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.orchestrator.test_orchestrator import TestOrchestrator
from tests.config.test_config import get_test_config
from tests.utils.test_utilities import TestLogger


async def main():
    """Main entry point for test infrastructure setup."""
    parser = argparse.ArgumentParser(
        description="Set up comprehensive testing infrastructure"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force cleanup even if tests are running"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup of existing tests, only create new structure",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = TestLogger("setup_infrastructure", log_level)

    logger.info("Starting test infrastructure setup")
    logger.info(f"Force cleanup: {args.force}")
    logger.info(f"Skip cleanup: {args.no_cleanup}")

    try:
        # Get configuration
        config = get_test_config()

        # Override cleanup setting if --no-cleanup is specified
        if args.no_cleanup:
            config.cleanup_existing_tests = False
            logger.info("Cleanup disabled by --no-cleanup flag")

        # Create orchestrator
        orchestrator = TestOrchestrator(config)

        # Set up infrastructure
        if args.no_cleanup:
            # Just create new structure without cleanup
            orchestrator.cleanup_manager.create_new_test_structure()
            logger.info("New test structure created successfully")
            cleanup_report = None
        else:
            # Full setup including cleanup
            if args.force:
                # Force cleanup by directly calling with force=True
                cleanup_report = orchestrator.cleanup_manager.cleanup_existing_tests(
                    force=True
                )
                orchestrator.cleanup_manager.create_new_test_structure()
            else:
                # Normal setup
                cleanup_report = await orchestrator.setup_test_infrastructure()

        # Report results
        if cleanup_report:
            logger.info("=== Cleanup Report ===")
            logger.info(f"Files deleted: {cleanup_report.deleted_files_count}")
            logger.info(
                f"Directories deleted: {cleanup_report.deleted_directories_count}"
            )
            logger.info(
                f"Execution time: {cleanup_report.execution_time_seconds:.2f} seconds"
            )

            if cleanup_report.errors:
                logger.warning(
                    f"Cleanup completed with {len(cleanup_report.errors)} errors:"
                )
                for error in cleanup_report.errors:
                    logger.error(f"  - {error}")
            else:
                logger.info("Cleanup completed successfully with no errors")

        # Validate environment
        validation_results = orchestrator.validate_test_environment()
        logger.info("=== Environment Validation ===")

        all_valid = True
        for check, result in validation_results.items():
            status = "✓" if result else "✗"
            logger.info(f"{status} {check}: {result}")
            if not result:
                all_valid = False

        if all_valid:
            logger.info("✓ Test environment validation passed")
            logger.info("Test infrastructure setup completed successfully!")
            return 0
        else:
            logger.error("✗ Test environment validation failed")
            return 1

    except Exception as e:
        logger.error(f"Test infrastructure setup failed: {e}")
        if args.verbose:
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
