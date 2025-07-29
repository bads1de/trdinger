"""
ML Model Testing Module for comprehensive testing framework.
Tests ML model accuracy, precision, recall, F1-score with Decimal precision.
"""

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import statistics
from scipy import stats

try:
    from ...orchestrator.test_orchestrator import (
        TestModuleInterface,
        TestModuleResult,
        TestStatus,
    )
    from ...config.test_config import TestConfig, MLTestConfig
    from ...utils.test_utilities import TestLogger, DecimalHelper
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
    from config.test_config import TestConfig, MLTestConfig
    from utils.test_utilities import TestLogger, DecimalHelper


@dataclass
class MLModelAccuracyResult:
    """Result from ML model accuracy testing."""

    model_name: str
    precision: Decimal
    recall: Decimal
    f1_score: Decimal
    accuracy: Decimal
    threshold_met: bool
    test_data_size: int
    execution_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class SyntheticTestData:
    """Synthetic test data for ML model validation."""

    features: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    data_size: int
    generation_method: str


@dataclass
class PredictionConsistencyResult:
    """Result from prediction consistency testing."""

    model_name: str
    consistency_runs: int
    mean_accuracy: Decimal
    std_accuracy: Decimal
    consistency_score: Decimal
    is_consistent: bool
    statistical_significance: float
    execution_time_seconds: float
    detailed_runs: List[Dict[str, Any]]
    error_message: Optional[str] = None


@dataclass
class PredictionFormatResult:
    """Result from prediction format validation."""

    model_name: str
    format_valid: bool
    expected_format: Dict[str, Any]
    actual_format: Dict[str, Any]
    validation_errors: List[str]
    execution_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class PerformanceDegradationResult:
    """Result from model performance degradation detection."""

    model_name: str
    baseline_accuracy: Decimal
    current_accuracy: Decimal
    degradation_percentage: Decimal
    degradation_detected: bool
    degradation_threshold: Decimal
    execution_time_seconds: float
    baseline_timestamp: Optional[datetime] = None
    error_message: Optional[str] = None


class MLModelTester(TestModuleInterface):
    """
    ML Model Testing Module implementing TestModuleInterface.

    Tests ML model accuracy, precision, recall, F1-score with Decimal precision.
    Implements requirements 2.1, 2.2, 2.3.
    """

    def __init__(self, config: MLTestConfig = None):
        self.config = config
        self.logger = TestLogger("ml_model_tester", "INFO")
        self.decimal_helper = DecimalHelper()
        self.test_results: List[MLModelAccuracyResult] = []

        # ML accuracy thresholds with Decimal precision
        self.accuracy_thresholds = {
            "precision": self.decimal_helper.create_decimal(
                self.config.accuracy_thresholds.get("precision", 0.7)
                if self.config
                else 0.7
            ),
            "recall": self.decimal_helper.create_decimal(
                self.config.accuracy_thresholds.get("recall", 0.6)
                if self.config
                else 0.6
            ),
            "f1_score": self.decimal_helper.create_decimal(
                self.config.accuracy_thresholds.get("f1_score", 0.65)
                if self.config
                else 0.65
            ),
        }

        self.logger.info(
            f"MLModelTester initialized with thresholds: {self.accuracy_thresholds}"
        )

    def get_module_name(self) -> str:
        """Get the name of this test module."""
        return "ml_testing"

    def generate_synthetic_test_data(self, data_size: int = 1000) -> SyntheticTestData:
        """
        Generate synthetic test data for model validation.

        Args:
            data_size: Number of samples to generate

        Returns:
            SyntheticTestData with features, labels, and mock predictions
        """
        try:
            self.logger.info(f"Generating synthetic test data with {data_size} samples")

            # Generate synthetic features (trading indicators)
            np.random.seed(42)  # For reproducible results

            # Features: price_change, volume, rsi, macd, bollinger_position
            features = np.random.randn(data_size, 5)

            # Normalize features to realistic trading ranges
            features[:, 0] = features[:, 0] * 0.05  # price_change: -15% to +15%
            features[:, 1] = np.abs(features[:, 1]) * 1000000  # volume: positive values
            features[:, 2] = (features[:, 2] + 1) * 50  # RSI: 0-100 range
            features[:, 3] = features[:, 3] * 0.01  # MACD: small values
            features[:, 4] = np.clip(
                features[:, 4], -1, 1
            )  # Bollinger position: -1 to 1

            # Generate labels based on simple trading logic
            # Buy (1) if price_change > 0 and RSI < 70, Sell (0) otherwise
            labels = ((features[:, 0] > 0) & (features[:, 2] < 70)).astype(int)

            # Generate mock predictions with some noise
            # Add slight randomness to simulate model uncertainty
            prediction_noise = np.random.randn(data_size) * 0.1
            predictions = labels.astype(float) + prediction_noise
            predictions = np.clip(predictions, 0, 1)  # Keep in [0,1] range

            synthetic_data = SyntheticTestData(
                features=features,
                labels=labels,
                predictions=predictions,
                data_size=data_size,
                generation_method="trading_indicators_simulation",
            )

            self.logger.info(
                f"Generated synthetic data: {data_size} samples, "
                f"{np.sum(labels)} positive labels ({np.mean(labels)*100:.1f}%)"
            )

            return synthetic_data

        except Exception as e:
            self.logger.error(f"Failed to generate synthetic test data: {e}")
            raise

    def calculate_ml_metrics_with_decimal_precision(
        self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Calculate ML metrics (precision, recall, F1-score, accuracy) with Decimal precision.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities or scores
            threshold: Classification threshold

        Returns:
            Tuple of (precision, recall, f1_score, accuracy) as Decimal values
        """
        try:
            # Convert predictions to binary classifications
            y_pred_binary = (y_pred >= threshold).astype(int)

            # Calculate metrics using sklearn
            precision = precision_score(
                y_true, y_pred_binary, average="binary", zero_division=0
            )
            recall = recall_score(
                y_true, y_pred_binary, average="binary", zero_division=0
            )
            f1 = f1_score(y_true, y_pred_binary, average="binary", zero_division=0)
            accuracy = accuracy_score(y_true, y_pred_binary)

            # Convert to Decimal with proper precision
            precision_decimal = self.decimal_helper.create_decimal(precision)
            recall_decimal = self.decimal_helper.create_decimal(recall)
            f1_decimal = self.decimal_helper.create_decimal(f1)
            accuracy_decimal = self.decimal_helper.create_decimal(accuracy)

            self.logger.debug(
                f"Calculated metrics - Precision: {precision_decimal}, "
                f"Recall: {recall_decimal}, F1: {f1_decimal}, Accuracy: {accuracy_decimal}"
            )

            return precision_decimal, recall_decimal, f1_decimal, accuracy_decimal

        except Exception as e:
            self.logger.error(f"Failed to calculate ML metrics: {e}")
            raise

    async def test_model_accuracy_with_synthetic_data(
        self, model_name: str = "test_model", data_size: int = None
    ) -> MLModelAccuracyResult:
        """
        Test ML model accuracy using synthetic data.

        Args:
            model_name: Name of the model being tested
            data_size: Size of test data to generate

        Returns:
            MLModelAccuracyResult with detailed accuracy metrics
        """
        start_time = time.time()

        try:
            self.logger.info(f"Testing model accuracy for {model_name}")

            # Use configured data size or default
            test_data_size = data_size or (
                self.config.model_test_data_size if self.config else 1000
            )

            # Generate synthetic test data
            synthetic_data = self.generate_synthetic_test_data(test_data_size)

            # Calculate metrics with Decimal precision
            precision, recall, f1_score, accuracy = (
                self.calculate_ml_metrics_with_decimal_precision(
                    synthetic_data.labels, synthetic_data.predictions
                )
            )

            # Check if thresholds are met
            threshold_met = (
                precision >= self.accuracy_thresholds["precision"]
                and recall >= self.accuracy_thresholds["recall"]
                and f1_score >= self.accuracy_thresholds["f1_score"]
            )

            execution_time = time.time() - start_time

            result = MLModelAccuracyResult(
                model_name=model_name,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                accuracy=accuracy,
                threshold_met=threshold_met,
                test_data_size=test_data_size,
                execution_time_seconds=execution_time,
            )

            self.logger.info(
                f"Model accuracy test completed for {model_name}: "
                f"Precision={precision}, Recall={recall}, F1={f1_score}, "
                f"Accuracy={accuracy}, Threshold met={threshold_met}"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Model accuracy test failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)

            return MLModelAccuracyResult(
                model_name=model_name,
                precision=Decimal("0"),
                recall=Decimal("0"),
                f1_score=Decimal("0"),
                accuracy=Decimal("0"),
                threshold_met=False,
                test_data_size=0,
                execution_time_seconds=execution_time,
                error_message=error_msg,
            )

    async def validate_model_accuracy_thresholds(self) -> Dict[str, Any]:
        """
        Validate that model accuracy meets configured thresholds.

        Returns:
            Dictionary with validation results
        """
        try:
            self.logger.info("Validating model accuracy thresholds")

            # Test multiple models with different configurations
            test_models = [
                ("primary_trading_model", 1000),
                ("secondary_trading_model", 500),
                ("validation_model", 200),
            ]

            validation_results = {
                "models_tested": 0,
                "models_passed": 0,
                "models_failed": 0,
                "threshold_violations": [],
                "detailed_results": {},
            }

            for model_name, data_size in test_models:
                try:
                    result = await self.test_model_accuracy_with_synthetic_data(
                        model_name, data_size
                    )
                    self.test_results.append(result)

                    validation_results["models_tested"] += 1
                    validation_results["detailed_results"][model_name] = {
                        "precision": float(result.precision),
                        "recall": float(result.recall),
                        "f1_score": float(result.f1_score),
                        "accuracy": float(result.accuracy),
                        "threshold_met": result.threshold_met,
                        "execution_time": result.execution_time_seconds,
                    }

                    if result.threshold_met:
                        validation_results["models_passed"] += 1
                    else:
                        validation_results["models_failed"] += 1
                        validation_results["threshold_violations"].append(
                            {
                                "model": model_name,
                                "precision": float(result.precision),
                                "recall": float(result.recall),
                                "f1_score": float(result.f1_score),
                                "required_precision": float(
                                    self.accuracy_thresholds["precision"]
                                ),
                                "required_recall": float(
                                    self.accuracy_thresholds["recall"]
                                ),
                                "required_f1_score": float(
                                    self.accuracy_thresholds["f1_score"]
                                ),
                            }
                        )

                except Exception as e:
                    self.logger.error(f"Failed to test model {model_name}: {e}")
                    validation_results["models_failed"] += 1
                    validation_results["threshold_violations"].append(
                        {"model": model_name, "error": str(e)}
                    )

            validation_results["overall_success"] = (
                validation_results["models_failed"] == 0
            )

            self.logger.info(
                f"Model accuracy threshold validation completed: "
                f"{validation_results['models_passed']}/{validation_results['models_tested']} passed"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Model accuracy threshold validation failed: {e}")
            return {
                "models_tested": 0,
                "models_passed": 0,
                "models_failed": 1,
                "threshold_violations": [{"error": str(e)}],
                "overall_success": False,
                "detailed_results": {},
            }

    async def test_prediction_consistency(
        self,
        model_name: str = "test_model",
        consistency_runs: int = None,
        data_size: int = 500,
    ) -> PredictionConsistencyResult:
        """
        Test prediction consistency across multiple runs with statistical validation.

        Args:
            model_name: Name of the model being tested
            consistency_runs: Number of consistency runs to perform
            data_size: Size of test data for each run

        Returns:
            PredictionConsistencyResult with consistency analysis
        """
        start_time = time.time()

        try:
            self.logger.info(f"Testing prediction consistency for {model_name}")

            # Use configured consistency runs or default
            runs = consistency_runs or (
                self.config.prediction_consistency_runs if self.config else 5
            )

            detailed_runs = []
            accuracies = []

            # Run multiple consistency tests
            for run_idx in range(runs):
                self.logger.debug(f"Running consistency test {run_idx + 1}/{runs}")

                # Generate fresh synthetic data for each run
                synthetic_data = self.generate_synthetic_test_data(data_size)

                # Calculate metrics for this run
                precision, recall, f1_score, accuracy = (
                    self.calculate_ml_metrics_with_decimal_precision(
                        synthetic_data.labels, synthetic_data.predictions
                    )
                )

                run_result = {
                    "run_index": run_idx,
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                    "data_size": data_size,
                }

                detailed_runs.append(run_result)
                accuracies.append(float(accuracy))

            # Calculate statistical measures
            mean_accuracy = self.decimal_helper.create_decimal(
                statistics.mean(accuracies)
            )
            std_accuracy = self.decimal_helper.create_decimal(
                statistics.stdev(accuracies) if len(accuracies) > 1 else 0
            )

            # Calculate consistency score (inverse of coefficient of variation)
            if mean_accuracy > 0:
                cv = std_accuracy / mean_accuracy
                consistency_score = self.decimal_helper.create_decimal(1.0) - cv
            else:
                consistency_score = self.decimal_helper.create_decimal(0.0)

            # Statistical significance test (one-sample t-test against expected accuracy)
            expected_accuracy = 0.8  # Expected baseline accuracy
            if len(accuracies) > 1:
                t_stat, p_value = stats.ttest_1samp(accuracies, expected_accuracy)
                statistical_significance = p_value
            else:
                statistical_significance = 1.0

            # Determine if predictions are consistent (low standard deviation)
            consistency_threshold = self.decimal_helper.create_decimal(
                0.05
            )  # 5% threshold
            is_consistent = std_accuracy <= consistency_threshold

            execution_time = time.time() - start_time

            result = PredictionConsistencyResult(
                model_name=model_name,
                consistency_runs=runs,
                mean_accuracy=mean_accuracy,
                std_accuracy=std_accuracy,
                consistency_score=consistency_score,
                is_consistent=is_consistent,
                statistical_significance=statistical_significance,
                execution_time_seconds=execution_time,
                detailed_runs=detailed_runs,
            )

            self.logger.info(
                f"Prediction consistency test completed for {model_name}: "
                f"Mean accuracy={mean_accuracy}, Std={std_accuracy}, "
                f"Consistent={is_consistent}"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Prediction consistency test failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)

            return PredictionConsistencyResult(
                model_name=model_name,
                consistency_runs=0,
                mean_accuracy=Decimal("0"),
                std_accuracy=Decimal("0"),
                consistency_score=Decimal("0"),
                is_consistent=False,
                statistical_significance=1.0,
                execution_time_seconds=execution_time,
                detailed_runs=[],
                error_message=error_msg,
            )

    def validate_prediction_format(
        self, predictions: np.ndarray, model_name: str = "test_model"
    ) -> PredictionFormatResult:
        """
        Validate prediction format for expected output structure.

        Args:
            predictions: Model predictions to validate
            model_name: Name of the model being tested

        Returns:
            PredictionFormatResult with format validation details
        """
        start_time = time.time()

        try:
            self.logger.info(f"Validating prediction format for {model_name}")

            # Define expected format for trading model predictions
            expected_format = {
                "type": "numpy.ndarray",
                "dtype": "float64",
                "shape_dimensions": 1,
                "value_range": (0.0, 1.0),
                "required_properties": ["finite_values", "no_nan", "no_inf"],
            }

            validation_errors = []

            # Validate type
            actual_type = type(predictions).__name__
            if not isinstance(predictions, np.ndarray):
                validation_errors.append(f"Expected numpy.ndarray, got {actual_type}")

            # Validate shape
            if isinstance(predictions, np.ndarray):
                if len(predictions.shape) != expected_format["shape_dimensions"]:
                    validation_errors.append(
                        f"Expected {expected_format['shape_dimensions']}D array, "
                        f"got {len(predictions.shape)}D array with shape {predictions.shape}"
                    )

                # Validate data type
                actual_dtype = str(predictions.dtype)
                if predictions.dtype.kind not in ["f", "i"]:  # float or int
                    validation_errors.append(
                        f"Expected numeric dtype, got {actual_dtype}"
                    )

                # Validate value range
                if predictions.size > 0:
                    min_val, max_val = float(np.min(predictions)), float(
                        np.max(predictions)
                    )
                    expected_min, expected_max = expected_format["value_range"]

                    if min_val < expected_min or max_val > expected_max:
                        validation_errors.append(
                            f"Values out of expected range {expected_format['value_range']}: "
                            f"actual range ({min_val:.6f}, {max_val:.6f})"
                        )

                    # Validate finite values
                    if not np.all(np.isfinite(predictions)):
                        nan_count = np.sum(np.isnan(predictions))
                        inf_count = np.sum(np.isinf(predictions))
                        validation_errors.append(
                            f"Non-finite values detected: {nan_count} NaN, {inf_count} Inf"
                        )

            # Create actual format description
            actual_format = {
                "type": actual_type,
                "dtype": (
                    str(predictions.dtype)
                    if isinstance(predictions, np.ndarray)
                    else "unknown"
                ),
                "shape": (
                    predictions.shape
                    if isinstance(predictions, np.ndarray)
                    else "unknown"
                ),
                "shape_dimensions": (
                    len(predictions.shape) if isinstance(predictions, np.ndarray) else 0
                ),
                "value_range": (
                    (float(np.min(predictions)), float(np.max(predictions)))
                    if isinstance(predictions, np.ndarray) and predictions.size > 0
                    else (0, 0)
                ),
                "size": predictions.size if isinstance(predictions, np.ndarray) else 0,
                "has_nan": (
                    bool(np.any(np.isnan(predictions)))
                    if isinstance(predictions, np.ndarray)
                    else False
                ),
                "has_inf": (
                    bool(np.any(np.isinf(predictions)))
                    if isinstance(predictions, np.ndarray)
                    else False
                ),
            }

            format_valid = len(validation_errors) == 0
            execution_time = max(
                time.time() - start_time, 0.000001
            )  # Ensure non-zero time

            result = PredictionFormatResult(
                model_name=model_name,
                format_valid=format_valid,
                expected_format=expected_format,
                actual_format=actual_format,
                validation_errors=validation_errors,
                execution_time_seconds=execution_time,
            )

            self.logger.info(
                f"Prediction format validation completed for {model_name}: "
                f"Valid={format_valid}, Errors={len(validation_errors)}"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = (
                f"Prediction format validation failed for {model_name}: {str(e)}"
            )
            self.logger.error(error_msg)

            return PredictionFormatResult(
                model_name=model_name,
                format_valid=False,
                expected_format={},
                actual_format={},
                validation_errors=[error_msg],
                execution_time_seconds=execution_time,
                error_message=error_msg,
            )

    async def test_model_performance_degradation(
        self,
        model_name: str = "test_model",
        baseline_accuracy: Decimal = None,
        degradation_threshold: Decimal = None,
    ) -> PerformanceDegradationResult:
        """
        Test for model performance degradation by comparing against baseline.

        Args:
            model_name: Name of the model being tested
            baseline_accuracy: Baseline accuracy to compare against
            degradation_threshold: Threshold for detecting degradation

        Returns:
            PerformanceDegradationResult with degradation analysis
        """
        start_time = time.time()

        try:
            self.logger.info(f"Testing performance degradation for {model_name}")

            # Use provided baseline or generate one
            if baseline_accuracy is None:
                # Generate baseline by running a reference test
                baseline_result = await self.test_model_accuracy_with_synthetic_data(
                    f"{model_name}_baseline", 1000
                )
                baseline_accuracy = baseline_result.accuracy
                baseline_timestamp = datetime.now()
            else:
                baseline_timestamp = None

            # Test current model performance
            current_result = await self.test_model_accuracy_with_synthetic_data(
                model_name, 1000
            )
            current_accuracy = current_result.accuracy

            # Calculate degradation
            degradation_percentage = (
                (baseline_accuracy - current_accuracy) / baseline_accuracy
            ) * 100

            # Use configured threshold or default (5% degradation)
            threshold = degradation_threshold or (
                self.decimal_helper.create_decimal(
                    self.config.performance_degradation_threshold
                    if self.config
                    else 0.05
                )
                * 100
            )

            degradation_detected = degradation_percentage > threshold

            execution_time = time.time() - start_time

            result = PerformanceDegradationResult(
                model_name=model_name,
                baseline_accuracy=baseline_accuracy,
                current_accuracy=current_accuracy,
                degradation_percentage=degradation_percentage,
                degradation_detected=degradation_detected,
                degradation_threshold=threshold,
                execution_time_seconds=execution_time,
                baseline_timestamp=baseline_timestamp,
            )

            self.logger.info(
                f"Performance degradation test completed for {model_name}: "
                f"Baseline={baseline_accuracy}, Current={current_accuracy}, "
                f"Degradation={degradation_percentage}%, Detected={degradation_detected}"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = (
                f"Performance degradation test failed for {model_name}: {str(e)}"
            )
            self.logger.error(error_msg)

            return PerformanceDegradationResult(
                model_name=model_name,
                baseline_accuracy=Decimal("0"),
                current_accuracy=Decimal("0"),
                degradation_percentage=Decimal("0"),
                degradation_detected=False,
                degradation_threshold=Decimal("5"),
                execution_time_seconds=execution_time,
                error_message=error_msg,
            )

    async def run_tests(self) -> TestModuleResult:
        """
        Run all ML model tests and return comprehensive results.

        Returns:
            TestModuleResult with execution details
        """
        start_time = datetime.now()
        module_name = self.get_module_name()

        self.logger.info(f"Starting {module_name} test execution")

        try:
            # Initialize test counters
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            tests_skipped = 0
            error_messages = []
            detailed_results = {}

            # Test 1: Model accuracy threshold validation
            self.logger.info("Running model accuracy threshold validation tests")
            tests_run += 1

            try:
                threshold_validation = await self.validate_model_accuracy_thresholds()
                detailed_results["threshold_validation"] = threshold_validation

                if threshold_validation["overall_success"]:
                    tests_passed += 1
                    self.logger.info("✓ Model accuracy threshold validation passed")
                else:
                    tests_failed += 1
                    error_msg = f"Model accuracy threshold validation failed: {len(threshold_validation['threshold_violations'])} violations"
                    error_messages.append(error_msg)
                    self.logger.error(error_msg)

            except Exception as e:
                tests_failed += 1
                error_msg = f"Model accuracy threshold validation error: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Test 2: Decimal precision validation
            self.logger.info("Running Decimal precision validation tests")
            tests_run += 1

            try:
                # Test that all metrics are properly calculated with Decimal precision
                test_data = self.generate_synthetic_test_data(100)
                precision, recall, f1, accuracy = (
                    self.calculate_ml_metrics_with_decimal_precision(
                        test_data.labels, test_data.predictions
                    )
                )

                # Verify all results are Decimal type
                decimal_validation = {
                    "precision_is_decimal": isinstance(precision, Decimal),
                    "recall_is_decimal": isinstance(recall, Decimal),
                    "f1_is_decimal": isinstance(f1, Decimal),
                    "accuracy_is_decimal": isinstance(accuracy, Decimal),
                    "precision_value": float(precision),
                    "recall_value": float(recall),
                    "f1_value": float(f1),
                    "accuracy_value": float(accuracy),
                }

                detailed_results["decimal_precision_validation"] = decimal_validation

                if all(
                    [
                        decimal_validation["precision_is_decimal"],
                        decimal_validation["recall_is_decimal"],
                        decimal_validation["f1_is_decimal"],
                        decimal_validation["accuracy_is_decimal"],
                    ]
                ):
                    tests_passed += 1
                    self.logger.info("✓ Decimal precision validation passed")
                else:
                    tests_failed += 1
                    error_msg = "Decimal precision validation failed: Some metrics not using Decimal type"
                    error_messages.append(error_msg)
                    self.logger.error(error_msg)

            except Exception as e:
                tests_failed += 1
                error_msg = f"Decimal precision validation error: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Test 3: Synthetic data generation validation
            self.logger.info("Running synthetic data generation validation tests")
            tests_run += 1

            try:
                # Test different data sizes
                data_sizes = [100, 500, 1000]
                data_generation_results = {}

                for size in data_sizes:
                    test_data = self.generate_synthetic_test_data(size)
                    data_generation_results[f"size_{size}"] = {
                        "features_shape": test_data.features.shape,
                        "labels_shape": test_data.labels.shape,
                        "predictions_shape": test_data.predictions.shape,
                        "data_size_match": test_data.data_size == size,
                        "labels_binary": set(np.unique(test_data.labels)) <= {0, 1},
                        "predictions_range": (
                            float(np.min(test_data.predictions)),
                            float(np.max(test_data.predictions)),
                        ),
                    }

                detailed_results["synthetic_data_generation"] = data_generation_results

                # Validate all data generation was successful
                all_valid = all(
                    result["data_size_match"] and result["labels_binary"]
                    for result in data_generation_results.values()
                )

                if all_valid:
                    tests_passed += 1
                    self.logger.info("✓ Synthetic data generation validation passed")
                else:
                    tests_failed += 1
                    error_msg = "Synthetic data generation validation failed"
                    error_messages.append(error_msg)
                    self.logger.error(error_msg)

            except Exception as e:
                tests_failed += 1
                error_msg = f"Synthetic data generation validation error: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Test 4: Prediction consistency validation
            self.logger.info("Running prediction consistency validation tests")
            tests_run += 1

            try:
                # Test prediction consistency across multiple runs
                consistency_result = await self.test_prediction_consistency(
                    "consistency_test_model", 3, 200
                )
                detailed_results["prediction_consistency"] = {
                    "model_name": consistency_result.model_name,
                    "consistency_runs": consistency_result.consistency_runs,
                    "mean_accuracy": float(consistency_result.mean_accuracy),
                    "std_accuracy": float(consistency_result.std_accuracy),
                    "consistency_score": float(consistency_result.consistency_score),
                    "is_consistent": consistency_result.is_consistent,
                    "statistical_significance": consistency_result.statistical_significance,
                    "execution_time": consistency_result.execution_time_seconds,
                    "detailed_runs": consistency_result.detailed_runs,
                }

                if (
                    consistency_result.is_consistent
                    and not consistency_result.error_message
                ):
                    tests_passed += 1
                    self.logger.info("✓ Prediction consistency validation passed")
                else:
                    tests_failed += 1
                    error_msg = f"Prediction consistency validation failed: {consistency_result.error_message or 'Inconsistent predictions'}"
                    error_messages.append(error_msg)
                    self.logger.error(error_msg)

            except Exception as e:
                tests_failed += 1
                error_msg = f"Prediction consistency validation error: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Test 5: Prediction format validation
            self.logger.info("Running prediction format validation tests")
            tests_run += 1

            try:
                # Test prediction format validation with different data types
                test_data = self.generate_synthetic_test_data(100)
                format_result = self.validate_prediction_format(
                    test_data.predictions, "format_test_model"
                )

                detailed_results["prediction_format_validation"] = {
                    "model_name": format_result.model_name,
                    "format_valid": format_result.format_valid,
                    "expected_format": format_result.expected_format,
                    "actual_format": format_result.actual_format,
                    "validation_errors": format_result.validation_errors,
                    "execution_time": format_result.execution_time_seconds,
                }

                if format_result.format_valid and not format_result.error_message:
                    tests_passed += 1
                    self.logger.info("✓ Prediction format validation passed")
                else:
                    tests_failed += 1
                    error_msg = f"Prediction format validation failed: {len(format_result.validation_errors)} errors"
                    error_messages.append(error_msg)
                    self.logger.error(error_msg)

            except Exception as e:
                tests_failed += 1
                error_msg = f"Prediction format validation error: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Test 6: Model performance degradation detection
            self.logger.info("Running model performance degradation detection tests")
            tests_run += 1

            try:
                # Test performance degradation detection
                degradation_result = await self.test_model_performance_degradation(
                    "degradation_test_model"
                )

                detailed_results["performance_degradation"] = {
                    "model_name": degradation_result.model_name,
                    "baseline_accuracy": float(degradation_result.baseline_accuracy),
                    "current_accuracy": float(degradation_result.current_accuracy),
                    "degradation_percentage": float(
                        degradation_result.degradation_percentage
                    ),
                    "degradation_detected": degradation_result.degradation_detected,
                    "degradation_threshold": float(
                        degradation_result.degradation_threshold
                    ),
                    "execution_time": degradation_result.execution_time_seconds,
                }

                if not degradation_result.error_message:
                    tests_passed += 1
                    self.logger.info("✓ Model performance degradation detection passed")
                else:
                    tests_failed += 1
                    error_msg = f"Model performance degradation detection failed: {degradation_result.error_message}"
                    error_messages.append(error_msg)
                    self.logger.error(error_msg)

            except Exception as e:
                tests_failed += 1
                error_msg = f"Model performance degradation detection error: {str(e)}"
                error_messages.append(error_msg)
                self.logger.error(error_msg)

            # Determine overall status
            if tests_failed > 0:
                status = TestStatus.FAILED
            elif tests_passed > 0:
                status = TestStatus.COMPLETED
            else:
                status = TestStatus.SKIPPED

            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Add summary to detailed results
            detailed_results["test_summary"] = {
                "total_models_tested": len(self.test_results),
                "accuracy_thresholds": {
                    k: float(v) for k, v in self.accuracy_thresholds.items()
                },
                "test_execution_time": execution_time,
            }

            result = TestModuleResult(
                module_name=module_name,
                status=status,
                execution_time_seconds=execution_time,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_skipped=tests_skipped,
                error_messages=error_messages,
                detailed_results=detailed_results,
                start_time=start_time,
                end_time=end_time,
            )

            self.logger.info(
                f"ML model testing completed: {status.value} "
                f"({tests_passed}/{tests_run} passed) in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            error_msg = f"ML model testing failed with exception: {str(e)}"
            exception_details = traceback.format_exc()

            self.logger.error(f"{error_msg}\n{exception_details}")

            return TestModuleResult(
                module_name=module_name,
                status=TestStatus.FAILED,
                execution_time_seconds=execution_time,
                tests_run=1,
                tests_passed=0,
                tests_failed=1,
                tests_skipped=0,
                error_messages=[error_msg],
                detailed_results={"exception": str(e)},
                start_time=start_time,
                end_time=end_time,
                exception_details=exception_details,
            )

    def register_with_orchestrator(self, orchestrator):
        """Register this test module with the TestOrchestrator."""
        try:
            orchestrator.register_test_module("ml_testing", self)
            self.logger.info("MLModelTester registered with TestOrchestrator")
        except Exception as e:
            self.logger.error(
                f"Failed to register MLModelTester with orchestrator: {e}"
            )
            raise
