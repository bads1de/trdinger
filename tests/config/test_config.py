"""
Test configuration management system for comprehensive testing overhaul.
Provides centralized configuration for all test modules.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal
import os
from pathlib import Path


@dataclass
class MLTestConfig:
    """Configuration for ML model testing."""

    accuracy_thresholds: Dict[str, float]
    model_test_data_size: int
    prediction_consistency_runs: int
    performance_degradation_threshold: float


@dataclass
class BacktestTestConfig:
    """Configuration for backtest testing."""

    sharpe_ratio_tolerance: float
    max_drawdown_tolerance: float
    win_rate_tolerance: float
    extreme_condition_scenarios: List[str]


@dataclass
class FinancialTestConfig:
    """Configuration for financial calculation testing."""

    decimal_precision: Decimal
    rounding_mode: str
    float_detection_paths: List[str]
    portfolio_test_scenarios: List[str]


@dataclass
class ConcurrencyTestConfig:
    """Configuration for concurrency testing."""

    concurrent_operations_count: int
    race_condition_iterations: int
    deadlock_timeout_seconds: int
    circuit_breaker_test_scenarios: List[str]


@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing."""

    market_data_processing_target_ms: int
    strategy_signal_generation_target_ms: int
    portfolio_update_target_ms: int
    performance_test_iterations: int


@dataclass
class SecurityTestConfig:
    """Configuration for security testing."""

    sensitive_patterns: List[str]
    log_scan_paths: List[str]
    input_validation_test_cases: List[str]
    encryption_test_scenarios: List[str]


@dataclass
class ReportingTestConfig:
    """Configuration for test reporting."""

    report_formats: List[str]
    output_directory: str
    ci_integration_enabled: bool
    metrics_collection_enabled: bool


@dataclass
class TestConfig:
    """Main test configuration class."""

    ml_config: MLTestConfig
    backtest_config: BacktestTestConfig
    financial_config: FinancialTestConfig
    concurrency_config: ConcurrencyTestConfig
    performance_config: PerformanceTestConfig
    security_config: SecurityTestConfig
    reporting_config: ReportingTestConfig
    cleanup_existing_tests: bool
    test_data_directory: str
    log_level: str


def load_test_config() -> TestConfig:
    """Load test configuration from environment variables and defaults."""

    # ML Testing Configuration
    ml_config = MLTestConfig(
        accuracy_thresholds={
            "precision": float(os.getenv("ML_PRECISION_THRESHOLD", "0.7")),
            "recall": float(os.getenv("ML_RECALL_THRESHOLD", "0.6")),
            "f1_score": float(os.getenv("ML_F1_THRESHOLD", "0.65")),
        },
        model_test_data_size=int(os.getenv("ML_TEST_DATA_SIZE", "1000")),
        prediction_consistency_runs=int(os.getenv("ML_CONSISTENCY_RUNS", "5")),
        performance_degradation_threshold=float(
            os.getenv("ML_DEGRADATION_THRESHOLD", "0.05")
        ),
    )

    # Backtest Testing Configuration
    backtest_config = BacktestTestConfig(
        sharpe_ratio_tolerance=float(os.getenv("BACKTEST_SHARPE_TOLERANCE", "0.01")),
        max_drawdown_tolerance=float(os.getenv("BACKTEST_DRAWDOWN_TOLERANCE", "0.01")),
        win_rate_tolerance=float(os.getenv("BACKTEST_WINRATE_TOLERANCE", "0.01")),
        extreme_condition_scenarios=[
            "high_volatility",
            "market_crash",
            "flash_crash",
            "low_liquidity",
            "extreme_pump",
        ],
    )

    # Financial Testing Configuration
    financial_config = FinancialTestConfig(
        decimal_precision=Decimal("0.00000001"),  # 8 decimal places
        rounding_mode="ROUND_HALF_UP",
        float_detection_paths=["backend/app/core", "backend/app/api", "backend/models"],
        portfolio_test_scenarios=[
            "single_position",
            "multiple_positions",
            "zero_balance",
            "negative_balance",
            "extreme_values",
        ],
    )

    # Concurrency Testing Configuration
    concurrency_config = ConcurrencyTestConfig(
        concurrent_operations_count=int(os.getenv("CONCURRENCY_OPS_COUNT", "10")),
        race_condition_iterations=int(os.getenv("RACE_CONDITION_ITERATIONS", "100")),
        deadlock_timeout_seconds=int(os.getenv("DEADLOCK_TIMEOUT", "30")),
        circuit_breaker_test_scenarios=[
            "api_rate_limit",
            "database_timeout",
            "external_service_failure",
        ],
    )

    # Performance Testing Configuration
    performance_config = PerformanceTestConfig(
        market_data_processing_target_ms=int(os.getenv("PERF_MARKET_DATA_MS", "100")),
        strategy_signal_generation_target_ms=int(
            os.getenv("PERF_STRATEGY_SIGNAL_MS", "500")
        ),
        portfolio_update_target_ms=int(os.getenv("PERF_PORTFOLIO_UPDATE_MS", "1000")),
        performance_test_iterations=int(os.getenv("PERF_TEST_ITERATIONS", "10")),
    )

    # Security Testing Configuration
    security_config = SecurityTestConfig(
        sensitive_patterns=[
            r"api[_-]?key",
            r"secret[_-]?key",
            r"password",
            r"token",
            r"private[_-]?key",
        ],
        log_scan_paths=["backend/logs", "frontend/logs", "/tmp", "/var/log"],
        input_validation_test_cases=[
            "sql_injection",
            "xss_attack",
            "command_injection",
            "path_traversal",
            "buffer_overflow",
        ],
        encryption_test_scenarios=[
            "data_at_rest",
            "data_in_transit",
            "api_keys",
            "user_data",
        ],
    )

    # Reporting Configuration
    reporting_config = ReportingTestConfig(
        report_formats=["json", "html", "junit"],
        output_directory=os.getenv("TEST_REPORT_DIR", "tests/reports"),
        ci_integration_enabled=os.getenv("CI_INTEGRATION", "true").lower() == "true",
        metrics_collection_enabled=os.getenv("METRICS_COLLECTION", "true").lower()
        == "true",
    )

    return TestConfig(
        ml_config=ml_config,
        backtest_config=backtest_config,
        financial_config=financial_config,
        concurrency_config=concurrency_config,
        performance_config=performance_config,
        security_config=security_config,
        reporting_config=reporting_config,
        cleanup_existing_tests=os.getenv("CLEANUP_EXISTING_TESTS", "true").lower()
        == "true",
        test_data_directory=os.getenv("TEST_DATA_DIR", "tests/data"),
        log_level=os.getenv("TEST_LOG_LEVEL", "INFO"),
    )


def get_test_config() -> TestConfig:
    """Get the global test configuration instance."""
    return load_test_config()
