"""
Test fixtures for comprehensive testing framework.
Provides reusable test data and setup/teardown functionality.
"""

import asyncio
import tempfile
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config.test_config import TestConfig, get_test_config
from ..utils.test_utilities import DecimalHelper, MockDataGenerator, TestLogger


class TestFixtures:
    """Central fixture management for all test modules."""

    def __init__(self):
        self.config = get_test_config()
        self.logger = TestLogger("test_fixtures")
        self.temp_directories = []
        self.mock_objects = {}

    @contextmanager
    def temporary_test_directory(self):
        """Create a temporary directory for test operations."""
        temp_dir = tempfile.mkdtemp(prefix="trdinger_test_")
        temp_path = Path(temp_dir)
        self.temp_directories.append(temp_path)

        try:
            yield temp_path
        finally:
            # Cleanup will be handled by cleanup_all_fixtures
            pass

    def cleanup_all_fixtures(self):
        """Clean up all temporary resources created by fixtures."""
        self.logger.info("Cleaning up test fixtures")

        # Clean up temporary directories
        import shutil

        for temp_dir in self.temp_directories:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                self.logger.error(f"Failed to clean up {temp_dir}: {e}")

        self.temp_directories.clear()
        self.mock_objects.clear()

        self.logger.info("Test fixtures cleanup completed")


# Global fixture instance
_test_fixtures = TestFixtures()


# ML Testing Fixtures
def get_ml_test_data(
    n_samples: int = 1000, n_features: int = 20
) -> tuple[pd.DataFrame, pd.Series]:
    """Get ML test data fixture."""
    return MockDataGenerator.generate_ml_features(
        n_samples=n_samples, n_features=n_features, target_correlation=0.3
    )


def get_ml_model_mock() -> MagicMock:
    """Get a mock ML model for testing."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1, 0, 1, 1]
    mock_model.predict_proba.return_value = [
        [0.7, 0.3],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.3, 0.7],
        [0.1, 0.9],
    ]
    mock_model.score.return_value = 0.75

    _test_fixtures.mock_objects["ml_model"] = mock_model
    return mock_model


# Backtest Testing Fixtures
def get_backtest_test_data(
    strategy_name: str = "test_strategy", n_trades: int = 100, win_rate: float = 0.6
) -> Dict[str, Any]:
    """Get backtest test data fixture."""
    return MockDataGenerator.generate_backtest_data(
        strategy_name=strategy_name, n_trades=n_trades, win_rate=win_rate
    )


def get_ohlcv_test_data(
    symbol: str = "BTCUSDT", days: int = 30, interval_minutes: int = 60
) -> pd.DataFrame:
    """Get OHLCV test data fixture."""
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()

    return MockDataGenerator.generate_ohlcv_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval_minutes=interval_minutes,
    )


# Financial Testing Fixtures
def get_financial_test_scenarios() -> List[Dict[str, Any]]:
    """Get financial calculation test scenarios."""
    return [
        {
            "name": "basic_calculation",
            "price": DecimalHelper.create_decimal("50000.12345678"),
            "quantity": DecimalHelper.create_decimal("0.5"),
            "expected_value": DecimalHelper.create_decimal("25000.06172839"),
        },
        {
            "name": "high_precision",
            "price": DecimalHelper.create_decimal("0.00000001"),
            "quantity": DecimalHelper.create_decimal("1000000"),
            "expected_value": DecimalHelper.create_decimal("0.01"),
        },
        {
            "name": "rounding_test",
            "price": DecimalHelper.create_decimal("123.456789125"),
            "quantity": DecimalHelper.create_decimal("1"),
            "expected_value": DecimalHelper.create_decimal(
                "123.45678913"
            ),  # ROUND_HALF_UP
        },
        {
            "name": "zero_value",
            "price": DecimalHelper.create_decimal("0"),
            "quantity": DecimalHelper.create_decimal("100"),
            "expected_value": DecimalHelper.create_decimal("0"),
        },
        {
            "name": "large_numbers",
            "price": DecimalHelper.create_decimal("999999.99999999"),
            "quantity": DecimalHelper.create_decimal("1000"),
            "expected_value": DecimalHelper.create_decimal("999999999.99999"),
        },
    ]


def get_portfolio_test_data() -> Dict[str, Any]:
    """Get portfolio test data fixture."""
    return {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "quantity": DecimalHelper.create_decimal("0.5"),
                "entry_price": DecimalHelper.create_decimal("50000.00"),
                "current_price": DecimalHelper.create_decimal("52000.00"),
                "pnl": DecimalHelper.create_decimal("1000.00"),
            },
            {
                "symbol": "ETHUSDT",
                "quantity": DecimalHelper.create_decimal("2.0"),
                "entry_price": DecimalHelper.create_decimal("3000.00"),
                "current_price": DecimalHelper.create_decimal("2900.00"),
                "pnl": DecimalHelper.create_decimal("-200.00"),
            },
        ],
        "total_value": DecimalHelper.create_decimal("107800.00"),
        "total_pnl": DecimalHelper.create_decimal("800.00"),
        "cash_balance": DecimalHelper.create_decimal("5000.00"),
    }


# Concurrency Testing Fixtures
@asynccontextmanager
async def get_async_test_context():
    """Get async test context for concurrency testing."""
    # Setup
    tasks = []
    try:
        yield tasks
    finally:
        # Cleanup - cancel any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


def get_database_session_mock():
    """Get a mock database session for testing."""
    mock_session = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.rollback = MagicMock()
    mock_session.close = MagicMock()
    mock_session.query = MagicMock()

    _test_fixtures.mock_objects["db_session"] = mock_session
    return mock_session


# Performance Testing Fixtures
def get_performance_test_data() -> Dict[str, Any]:
    """Get performance test data and targets."""
    return {
        "market_data_processing_target_ms": _test_fixtures.config.performance_config.market_data_processing_target_ms,
        "strategy_signal_generation_target_ms": _test_fixtures.config.performance_config.strategy_signal_generation_target_ms,
        "portfolio_update_target_ms": _test_fixtures.config.performance_config.portfolio_update_target_ms,
        "test_iterations": _test_fixtures.config.performance_config.performance_test_iterations,
        "sample_market_data": get_ohlcv_test_data(
            days=1, interval_minutes=1
        ),  # 1 day of minute data
    }


# Security Testing Fixtures
def get_security_test_scenarios() -> Dict[str, Any]:
    """Get security test scenarios and data."""
    return {
        "sensitive_patterns": _test_fixtures.config.security_config.sensitive_patterns,
        "malicious_inputs": [
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "../../../etc/passwd",  # Path traversal
            "$(rm -rf /)",  # Command injection
            "A" * 10000,  # Buffer overflow attempt
        ],
        "test_log_content": """
        2024-01-01 10:00:00 INFO Processing trade
        2024-01-01 10:00:01 DEBUG API key: sk-1234567890abcdef
        2024-01-01 10:00:02 INFO Trade completed
        2024-01-01 10:00:03 ERROR Failed to connect with password: secret123
        """,
        "encryption_test_data": {
            "plaintext": "sensitive trading data",
            "key": "test_encryption_key_32_bytes_long",
            "expected_encrypted_length": 64,  # Example
        },
    }


# Test Data Management Fixtures
def get_test_data_directory() -> Path:
    """Get test data directory fixture."""
    data_dir = Path(_test_fixtures.config.test_data_directory)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_test_reports_directory() -> Path:
    """Get test reports directory fixture."""
    reports_dir = Path(_test_fixtures.config.reporting_config.output_directory)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


# Configuration Fixtures
def get_test_config_fixture() -> TestConfig:
    """Get test configuration fixture."""
    return _test_fixtures.config


# Cleanup fixture
def cleanup_test_fixtures():
    """Clean up all test fixtures."""
    _test_fixtures.cleanup_all_fixtures()


# Pytest fixtures (if using pytest)
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment for the entire test session."""
    _test_fixtures.logger.info("Setting up test environment")
    yield
    _test_fixtures.logger.info("Tearing down test environment")
    cleanup_test_fixtures()


@pytest.fixture
def ml_test_data():
    """Pytest fixture for ML test data."""
    return get_ml_test_data()


@pytest.fixture
def backtest_test_data():
    """Pytest fixture for backtest test data."""
    return get_backtest_test_data()


@pytest.fixture
def financial_test_scenarios():
    """Pytest fixture for financial test scenarios."""
    return get_financial_test_scenarios()


@pytest.fixture
def portfolio_test_data():
    """Pytest fixture for portfolio test data."""
    return get_portfolio_test_data()


@pytest.fixture
def performance_test_data():
    """Pytest fixture for performance test data."""
    return get_performance_test_data()


@pytest.fixture
def security_test_scenarios():
    """Pytest fixture for security test scenarios."""
    return get_security_test_scenarios()


@pytest.fixture
def test_config():
    """Pytest fixture for test configuration."""
    return get_test_config_fixture()
