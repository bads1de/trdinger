#!/usr/bin/env python3
"""
Test script for MLModelTester implementation.
Tests the functionality implemented in subtask 3.1.
"""

import asyncio
import sys
from pathlib import Path

# Add the tests directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.test_config import get_test_config
from modules.ml_testing.ml_model_tester import MLModelTester


async def test_ml_model_tester_basic():
    """Test basic MLModelTester functionality."""
    print("=== Testing MLModelTester Basic Functionality ===")

    try:
        # Initialize with test config
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        print(f"✓ MLModelTester initialized successfully")
        print(f"  Module name: {ml_tester.get_module_name()}")
        print(f"  Accuracy thresholds: {ml_tester.accuracy_thresholds}")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_synthetic_data_generation():
    """Test synthetic data generation."""
    print("\n=== Testing Synthetic Data Generation ===")

    try:
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        # Test different data sizes
        for size in [100, 500, 1000]:
            synthetic_data = ml_tester.generate_synthetic_test_data(size)

            print(f"✓ Generated synthetic data with {size} samples:")
            print(f"  Features shape: {synthetic_data.features.shape}")
            print(f"  Labels shape: {synthetic_data.labels.shape}")
            print(f"  Predictions shape: {synthetic_data.predictions.shape}")
            print(f"  Generation method: {synthetic_data.generation_method}")

            # Validate data
            assert synthetic_data.data_size == size
            assert synthetic_data.features.shape[0] == size
            assert synthetic_data.labels.shape[0] == size
            assert synthetic_data.predictions.shape[0] == size
            assert set(synthetic_data.labels) <= {0, 1}  # Binary labels

        print("✓ All synthetic data generation tests passed")
        return True

    except Exception as e:
        print(f"❌ Synthetic data generation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_ml_metrics_calculation():
    """Test ML metrics calculation with Decimal precision."""
    print("\n=== Testing ML Metrics Calculation ===")

    try:
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        # Generate test data
        synthetic_data = ml_tester.generate_synthetic_test_data(200)

        # Calculate metrics
        precision, recall, f1_score, accuracy = (
            ml_tester.calculate_ml_metrics_with_decimal_precision(
                synthetic_data.labels, synthetic_data.predictions
            )
        )

        print(f"✓ ML metrics calculated successfully:")
        print(f"  Precision: {precision} (type: {type(precision).__name__})")
        print(f"  Recall: {recall} (type: {type(recall).__name__})")
        print(f"  F1-score: {f1_score} (type: {type(f1_score).__name__})")
        print(f"  Accuracy: {accuracy} (type: {type(accuracy).__name__})")

        # Validate Decimal types
        from decimal import Decimal

        assert isinstance(
            precision, Decimal
        ), f"Precision should be Decimal, got {type(precision)}"
        assert isinstance(
            recall, Decimal
        ), f"Recall should be Decimal, got {type(recall)}"
        assert isinstance(
            f1_score, Decimal
        ), f"F1-score should be Decimal, got {type(f1_score)}"
        assert isinstance(
            accuracy, Decimal
        ), f"Accuracy should be Decimal, got {type(accuracy)}"

        print("✓ All metrics are properly using Decimal type")
        return True

    except Exception as e:
        print(f"❌ ML metrics calculation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_model_accuracy_testing():
    """Test model accuracy testing functionality."""
    print("\n=== Testing Model Accuracy Testing ===")

    try:
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        # Test model accuracy with synthetic data
        result = await ml_tester.test_model_accuracy_with_synthetic_data(
            "test_model", 500
        )

        print(f"✓ Model accuracy test completed:")
        print(f"  Model name: {result.model_name}")
        print(f"  Precision: {result.precision}")
        print(f"  Recall: {result.recall}")
        print(f"  F1-score: {result.f1_score}")
        print(f"  Accuracy: {result.accuracy}")
        print(f"  Threshold met: {result.threshold_met}")
        print(f"  Test data size: {result.test_data_size}")
        print(f"  Execution time: {result.execution_time_seconds:.3f}s")

        # Validate result
        assert result.model_name == "test_model"
        assert result.test_data_size == 500
        assert result.execution_time_seconds > 0

        if result.error_message:
            print(f"  Error message: {result.error_message}")

        print("✓ Model accuracy testing completed successfully")
        return True

    except Exception as e:
        print(f"❌ Model accuracy testing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_full_ml_module_execution():
    """Test full ML module execution."""
    print("\n=== Testing Full ML Module Execution ===")

    try:
        config = get_test_config()
        ml_tester = MLModelTester(config.ml_config)

        # Run the full test module
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

        # Validate result
        assert result.module_name == "ml_testing"
        assert result.tests_run > 0
        assert result.execution_time_seconds > 0

        print("✓ Full ML module execution completed successfully")
        return True

    except Exception as e:
        print(f"❌ Full ML module execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all ML model tester tests."""
    print("Testing MLModelTester Implementation (Subtask 3.1)")
    print("=" * 60)

    tests = [
        test_ml_model_tester_basic,
        test_synthetic_data_generation,
        test_ml_metrics_calculation,
        test_model_accuracy_testing,
        test_full_ml_module_execution,
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
        print("✅ All MLModelTester tests passed! Subtask 3.1 is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
