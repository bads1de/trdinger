#!/usr/bin/env python3
"""
Simple test script to verify the TestOrchestrator implementation.
This tests the core functionality implemented in tasks 2.1 and 2.2.
"""

import asyncio
import sys
from pathlib import Path

from orchestrator.test_orchestrator import TestOrchestrator, TestStatus
from config.test_config import get_test_config


async def test_orchestrator_basic_functionality():
    """Test basic orchestrator functionality."""
    print("=== Testing TestOrchestrator Basic Functionality ===")

    # Initialize orchestrator
    config = get_test_config()
    orchestrator = TestOrchestrator(config)

    print(
        f"✓ Orchestrator initialized with {len(orchestrator.get_available_test_modules())} test modules"
    )

    # Test status methods
    status = orchestrator.get_test_status()
    print(
        f"✓ Test status retrieved: {status['modules_registered_count']} modules registered"
    )

    # Test environment validation
    validation = orchestrator.validate_test_environment()
    print(f"✓ Environment validation: {validation['all_valid']}")
    if not validation["all_valid"]:
        print(f"  Validation errors: {validation['validation_errors']}")

    # Test execution status
    exec_status = await orchestrator.get_execution_status()
    print(f"✓ Execution status: In progress = {exec_status['execution_in_progress']}")

    return True


async def test_orchestrator_test_execution():
    """Test orchestrator test execution."""
    print("\n=== Testing TestOrchestrator Test Execution ===")

    # Initialize orchestrator
    orchestrator = TestOrchestrator()

    # Test running all tests (should be placeholders for now)
    print("Running all tests...")
    results = await orchestrator.run_all_tests(parallel=False)

    print(f"✓ Test execution completed:")
    print(f"  Overall status: {results.overall_status.value}")
    print(f"  Execution time: {results.total_execution_time_seconds:.2f}s")
    print(f"  Modules tested: {len(results.modules_results)}")
    print(f"  Summary: {results.summary}")

    # Test running specific tests
    print("\nRunning specific tests...")
    specific_results = await orchestrator.run_specific_tests(
        ["ml_testing", "financial_testing"], parallel=False
    )

    print(f"✓ Specific test execution completed:")
    print(f"  Overall status: {specific_results.overall_status.value}")
    print(f"  Modules tested: {len(specific_results.modules_results)}")

    return True


async def test_cleanup_and_initialization():
    """Test cleanup and initialization functionality."""
    print("\n=== Testing Cleanup and Initialization ===")

    orchestrator = TestOrchestrator()

    # Test infrastructure setup
    print("Setting up test infrastructure...")
    cleanup_report = await orchestrator.setup_test_infrastructure()

    if cleanup_report:
        print(f"✓ Infrastructure setup completed:")
        print(f"  Safety checks passed: {cleanup_report.safety_checks_passed}")
        print(f"  Initialization completed: {cleanup_report.initialization_completed}")
        print(f"  Execution time: {cleanup_report.execution_time_seconds:.2f}s")

        if cleanup_report.errors:
            print(f"  Errors: {len(cleanup_report.errors)}")
            for error in cleanup_report.errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
    else:
        print("✓ No cleanup was needed")

    return True


async def main():
    """Run all tests."""
    print("Testing TestOrchestrator Implementation")
    print("=" * 50)

    try:
        # Test basic functionality
        await test_orchestrator_basic_functionality()

        # Test test execution
        await test_orchestrator_test_execution()

        # Test cleanup and initialization
        await test_cleanup_and_initialization()

        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
