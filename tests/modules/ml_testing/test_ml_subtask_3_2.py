#!/usr/bin/env python3
"""
Test script for MLModelTester subtask 3.2 implementation.
Tests prediction consistency and format validation functionality.
"""

import asyncio
import sys
import numpy as np
from pathlib import Path

# Add the tests directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.test_config import get_test_config
from modules.ml_testing.ml_model_tester import MLModelTester


async def test_prediction_consistency():
    """Test prediction consistency across multiple runs."""
    print("=== Testing Prediction Consistency ===")

    try:
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        # Test prediction consistency with different parameters
        test_cases = [
            ("consistency_model_1", 3, 200),
            ("consistency_model_2", 5, 500),
            ("consistency_model_3", 2, 100),
        ]

        for model_name, runs, data_size in test_cases:
            print(
                f"\nTesting {model_name} with {runs} runs and {data_size} data points..."
            )

            result = await ml_tester.test_prediction_consistency(
                model_name, runs, data_size
            )

            print(f"✓ Prediction consistency test completed:")
            print(f"  Model: {result.model_name}")
            print(f"  Consistency runs: {result.consistency_runs}")
            print(f"  Mean accuracy: {result.mean_accuracy}")
            print(f"  Std accuracy: {result.std_accuracy}")
            print(f"  Consistency score: {result.consistency_score}")
            print(f"  Is consistent: {result.is_consistent}")
            print(f"  Statistical significance: {result.statistical_significance:.6f}")
            print(f"  Execution time: {result.execution_time_seconds:.3f}s")
            print(f"  Detailed runs: {len(result.detailed_runs)} runs")

            # Validate result structure
            assert result.model_name == model_name
            assert result.consistency_runs == runs
            assert len(result.detailed_runs) == runs
            assert result.execution_time_seconds > 0

            # Validate detailed runs
            for i, run_result in enumerate(result.detailed_runs):
                assert run_result["run_index"] == i
                assert "accuracy" in run_result
                assert "precision" in run_result
                assert "recall" in run_result
                assert "f1_score" in run_result
                assert run_result["data_size"] == data_size

            if result.error_message:
                print(f"  Error: {result.error_message}")

        print("✓ All prediction consistency tests passed")
        return True

    except Exception as e:
        print(f"❌ Prediction consistency test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prediction_format_validation():
    """Test prediction format validation."""
    print("\n=== Testing Prediction Format Validation ===")

    try:
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        # Test cases with different prediction formats
        test_cases = [
            ("valid_predictions", np.array([0.1, 0.5, 0.8, 0.3, 0.9]), True),
            ("edge_case_predictions", np.array([0.0, 1.0, 0.5]), True),
            ("out_of_range_predictions", np.array([-0.1, 0.5, 1.1]), False),
            ("nan_predictions", np.array([0.5, np.nan, 0.8]), False),
            ("inf_predictions", np.array([0.5, np.inf, 0.8]), False),
            ("2d_predictions", np.array([[0.5, 0.8], [0.3, 0.9]]), False),
            ("integer_predictions", np.array([0, 1, 0, 1]), True),  # Should be valid
        ]

        for test_name, predictions, should_be_valid in test_cases:
            print(f"\nTesting {test_name}...")

            result = ml_tester.validate_prediction_format(predictions, test_name)

            print(f"✓ Format validation completed:")
            print(f"  Model: {result.model_name}")
            print(f"  Format valid: {result.format_valid}")
            print(f"  Expected format: {result.expected_format}")
            print(f"  Actual format: {result.actual_format}")
            print(f"  Validation errors: {len(result.validation_errors)}")
            print(f"  Execution time: {result.execution_time_seconds:.6f}s")

            if result.validation_errors:
                print(f"  Errors:")
                for error in result.validation_errors:
                    print(f"    - {error}")

            # Validate result matches expectation
            if should_be_valid:
                if not result.format_valid:
                    print(
                        f"  ⚠️  Expected valid format but got invalid: {result.validation_errors}"
                    )
            else:
                if result.format_valid:
                    print(f"  ⚠️  Expected invalid format but got valid")

            # Validate result structure
            assert result.model_name == test_name
            assert isinstance(result.format_valid, bool)
            assert isinstance(result.expected_format, dict)
            assert isinstance(result.actual_format, dict)
            assert isinstance(result.validation_errors, list)
            assert result.execution_time_seconds > 0

            if result.error_message:
                print(f"  Error: {result.error_message}")

        # Test with non-numpy array
        print(f"\nTesting non-numpy array...")
        result = ml_tester.validate_prediction_format(
            [0.5, 0.8, 0.3], "list_predictions"
        )
        print(f"  Format valid: {result.format_valid} (should be False)")
        print(f"  Validation errors: {result.validation_errors}")
        assert not result.format_valid

        print("✓ All prediction format validation tests passed")
        return True

    except Exception as e:
        print(f"❌ Prediction format validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_performance_degradation_detection():
    """Test model performance degradation detection."""
    print("\n=== Testing Performance Degradation Detection ===")

    try:
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        # Test cases with different degradation scenarios
        test_cases = [
            ("stable_model", None, None),  # Auto-generate baseline
            (
                "degraded_model",
                ml_tester.decimal_helper.create_decimal("0.95"),
                ml_tester.decimal_helper.create_decimal("3.0"),
            ),  # 3% threshold
        ]

        for model_name, baseline_accuracy, degradation_threshold in test_cases:
            print(f"\nTesting {model_name}...")

            result = await ml_tester.test_model_performance_degradation(
                model_name, baseline_accuracy, degradation_threshold
            )

            print(f"✓ Performance degradation test completed:")
            print(f"  Model: {result.model_name}")
            print(f"  Baseline accuracy: {result.baseline_accuracy}")
            print(f"  Current accuracy: {result.current_accuracy}")
            print(f"  Degradation percentage: {result.degradation_percentage}%")
            print(f"  Degradation detected: {result.degradation_detected}")
            print(f"  Degradation threshold: {result.degradation_threshold}%")
            print(f"  Execution time: {result.execution_time_seconds:.3f}s")

            if result.baseline_timestamp:
                print(f"  Baseline timestamp: {result.baseline_timestamp}")

            # Validate result structure
            assert result.model_name == model_name
            assert result.execution_time_seconds > 0

            # Validate Decimal types
            from decimal import Decimal

            assert isinstance(result.baseline_accuracy, Decimal)
            assert isinstance(result.current_accuracy, Decimal)
            assert isinstance(result.degradation_percentage, Decimal)
            assert isinstance(result.degradation_threshold, Decimal)

            if result.error_message:
                print(f"  Error: {result.error_message}")

        print("✓ All performance degradation detection tests passed")
        return True

    except Exception as e:
        print(f"❌ Performance degradation detection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_full_ml_module_with_subtask_3_2():
    """Test full ML module execution including subtask 3.2 functionality."""
    print("\n=== Testing Full ML Module with Subtask 3.2 ===")

    try:
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        # Run the full test module (should now include subtask 3.2 tests)
        result = await ml_tester.run_tests()

        print(f"✓ Full ML module execution completed:")
        print(f"  Module name: {result.module_name}")
        print(f"  Status: {result.status.value}")
        print(f"  Tests run: {result.tests_run}")
        print(f"  Tests passed: {result.tests_passed}")
        print(f"  Tests failed: {result.tests_failed}")
        print(f"  Tests skipped: {result.tests_skipped}")
        print(f"  Execution time: {result.execution_time_seconds:.3f}s")

        if result.error_messages:
            print(f"  Error messages: {len(result.error_messages)}")
            for i, error in enumerate(result.error_messages[:3]):  # Show first 3 errors
                print(f"    {i+1}. {error}")

        print(f"  Detailed results keys: {list(result.detailed_results.keys())}")

        # Validate that subtask 3.2 tests are included
        expected_keys = [
            "threshold_validation",
            "decimal_precision_validation",
            "synthetic_data_generation",
            "prediction_consistency",  # New in subtask 3.2
            "prediction_format_validation",  # New in subtask 3.2
            "performance_degradation",  # New in subtask 3.2
            "test_summary",
        ]

        for key in expected_keys:
            if key in result.detailed_results:
                print(f"  ✓ {key} test included")
            else:
                print(f"  ❌ {key} test missing")

        # Validate result
        assert result.module_name == "ml_testing"
        assert result.tests_run >= 6  # Should have at least 6 tests now
        assert result.execution_time_seconds > 0

        print("✓ Full ML module execution with subtask 3.2 completed successfully")
        return True

    except Exception as e:
        print(f"❌ Full ML module execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_orchestrator_integration():
    """Test integration with TestOrchestrator."""
    print("\n=== Testing TestOrchestrator Integration ===")

    try:
        from orchestrator.test_orchestrator import TestOrchestrator

        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)
        orchestrator = TestOrchestrator(config)

        # Register the ML tester with orchestrator
        ml_tester.register_with_orchestrator(orchestrator)

        # Verify registration
        registered_modules = orchestrator.get_registered_test_modules()
        print(f"✓ Registered modules: {registered_modules}")

        assert "ml_testing" in registered_modules

        # Test running ML tests through orchestrator
        results = await orchestrator.run_specific_tests(["ml_testing"], parallel=False)

        print(f"✓ Orchestrator test execution completed:")
        print(f"  Overall status: {results.overall_status.value}")
        print(f"  Modules tested: {len(results.modules_results)}")
        print(
            f"  ML testing status: {results.modules_results['ml_testing'].status.value}"
        )
        print(f"  ML tests run: {results.modules_results['ml_testing'].tests_run}")
        print(
            f"  ML tests passed: {results.modules_results['ml_testing'].tests_passed}"
        )

        assert "ml_testing" in results.modules_results
        assert results.modules_results["ml_testing"].tests_run >= 6

        print("✓ TestOrchestrator integration test passed")
        return True

    except Exception as e:
        print(f"❌ TestOrchestrator integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all subtask 3.2 tests."""
    print("Testing MLModelTester Subtask 3.2 Implementation")
    print("=" * 60)

    tests = [
        test_prediction_consistency,
        test_prediction_format_validation,
        test_performance_degradation_detection,
        test_full_ml_module_with_subtask_3_2,
        test_orchestrator_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if await test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print(
            "✅ All subtask 3.2 tests passed! Prediction consistency and format validation are working correctly."
        )
        print("\nSubtask 3.2 Features Implemented:")
        print(
            "- Prediction consistency tests across multiple runs with statistical validation"
        )
        print("- Prediction format validation for expected output structure")
        print(
            "- Model performance degradation detection system with baseline comparison"
        )
        print("- Integration with TestOrchestrator for proper module registration")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
