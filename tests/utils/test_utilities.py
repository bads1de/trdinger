"""
Test utilities and helper functions for comprehensive testing framework.
Provides common functionality used across all test modules.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class TestLogger:
    """Centralized logging for test execution."""

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(f"test.{name}")
        self.logger.setLevel(getattr(logging, level.upper()))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)


class DecimalHelper:
    """Helper functions for financial calculations with Decimal precision."""

    PRECISION = Decimal("0.00000001")  # 8 decimal places
    ROUNDING = ROUND_HALF_UP

    @classmethod
    def create_decimal(cls, value: Union[str, int, float]) -> Decimal:
        """Create a Decimal with proper precision."""
        return Decimal(str(value)).quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def validate_decimal_type(cls, value: Any) -> bool:
        """Validate that a value is a Decimal type."""
        return isinstance(value, Decimal)

    @classmethod
    def compare_decimals(
        cls, a: Decimal, b: Decimal, tolerance: Decimal = None
    ) -> bool:
        """Compare two Decimal values with optional tolerance."""
        if tolerance is None:
            tolerance = cls.PRECISION
        return abs(a - b) <= tolerance

    @classmethod
    def detect_float_usage(cls, obj: Any) -> List[str]:
        """Detect float usage in an object (for validation)."""
        float_usages = []

        if isinstance(obj, float):
            float_usages.append(f"Direct float value: {obj}")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, float):
                    float_usages.append(f"Float in dict key '{key}': {value}")
                elif hasattr(value, "__dict__"):
                    float_usages.extend(cls.detect_float_usage(value))
        elif hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                if isinstance(attr_value, float):
                    float_usages.append(
                        f"Float in attribute '{attr_name}': {attr_value}"
                    )

        return float_usages


class MockDataGenerator:
    """Generate mock data for testing purposes."""

    @staticmethod
    def generate_ohlcv_data(
        symbol: str = "BTCUSDT",
        start_date: datetime = None,
        end_date: datetime = None,
        interval_minutes: int = 60,
        base_price: Decimal = None,
    ) -> pd.DataFrame:
        """Generate mock OHLCV data."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        if base_price is None:
            base_price = DecimalHelper.create_decimal("50000.00")

        # Generate time series
        time_range = pd.date_range(
            start=start_date, end=end_date, freq=f"{interval_minutes}min"
        )

        data = []
        current_price = base_price

        for timestamp in time_range:
            # Generate realistic price movement
            change_percent = Decimal(str(np.random.normal(0, 0.02)))  # 2% volatility
            price_change = current_price * change_percent

            open_price = current_price
            close_price = current_price + price_change
            high_price = max(open_price, close_price) * DecimalHelper.create_decimal(
                "1.01"
            )
            low_price = min(open_price, close_price) * DecimalHelper.create_decimal(
                "0.99"
            )
            volume = DecimalHelper.create_decimal(str(np.random.uniform(100, 1000)))

            data.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

            current_price = close_price

        return pd.DataFrame(data)

    @staticmethod
    def generate_ml_features(
        n_samples: int = 1000, n_features: int = 20, target_correlation: float = 0.3
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate mock ML features and target data."""
        # Generate features
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Generate target with some correlation to features
        target_base = features.iloc[:, :5].sum(axis=1) * target_correlation
        noise = pd.Series(np.random.randn(n_samples) * 0.5)
        target = (target_base + noise > 0).astype(int)

        return features, target

    @staticmethod
    def generate_backtest_data(
        strategy_name: str = "test_strategy", n_trades: int = 100, win_rate: float = 0.6
    ) -> Dict[str, Any]:
        """Generate mock backtest results."""
        # Generate trade results
        trades = []
        portfolio_value = DecimalHelper.create_decimal("10000.00")

        for i in range(n_trades):
            is_win = np.random.random() < win_rate

            if is_win:
                pnl_percent = DecimalHelper.create_decimal(
                    str(np.random.uniform(0.01, 0.05))
                )
            else:
                pnl_percent = DecimalHelper.create_decimal(
                    str(np.random.uniform(-0.03, -0.01))
                )

            pnl = portfolio_value * pnl_percent
            portfolio_value += pnl

            trades.append(
                {
                    "trade_id": i,
                    "entry_time": datetime.now() - timedelta(days=n_trades - i),
                    "exit_time": datetime.now() - timedelta(days=n_trades - i - 1),
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "portfolio_value": portfolio_value,
                }
            )

        # Calculate metrics
        total_pnl = sum(trade["pnl"] for trade in trades)
        returns = [trade["pnl_percent"] for trade in trades]

        sharpe_ratio = DecimalHelper.create_decimal(
            str(
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0
                else 0
            )
        )

        # Calculate max drawdown
        peak = DecimalHelper.create_decimal("10000.00")
        max_drawdown = DecimalHelper.create_decimal("0.00")

        for trade in trades:
            if trade["portfolio_value"] > peak:
                peak = trade["portfolio_value"]
            drawdown = (peak - trade["portfolio_value"]) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "strategy_name": strategy_name,
            "trades": trades,
            "total_pnl": total_pnl,
            "win_rate": DecimalHelper.create_decimal(str(win_rate)),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": n_trades,
        }


class DatabaseTestHelper:
    """Helper for database testing operations."""

    def __init__(self, test_db_url: str = None):
        if test_db_url is None:
            test_db_url = "sqlite:///:memory:"

        self.engine = create_engine(test_db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

    @contextmanager
    def get_test_session(self):
        """Get a test database session."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def create_tables(self, base):
        """Create all tables for testing."""
        base.metadata.create_all(bind=self.engine)

    def drop_tables(self, base):
        """Drop all tables after testing."""
        base.metadata.drop_all(bind=self.engine)


class FileSystemTestHelper:
    """Helper for file system operations in tests."""

    @staticmethod
    @contextmanager
    def temporary_directory():
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        try:
            yield Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def count_files_in_directory(directory: Path, pattern: str = "*") -> int:
        """Count files in a directory matching a pattern."""
        if not directory.exists():
            return 0
        return len(list(directory.glob(pattern)))

    @staticmethod
    def get_directory_structure(directory: Path) -> Dict[str, Any]:
        """Get the structure of a directory as a nested dict."""
        if not directory.exists():
            return {}

        structure = {}
        for item in directory.iterdir():
            if item.is_file():
                structure[item.name] = "file"
            elif item.is_dir():
                structure[item.name] = FileSystemTestHelper.get_directory_structure(
                    item
                )

        return structure


class AsyncTestHelper:
    """Helper for async testing operations."""

    @staticmethod
    async def run_with_timeout(coro, timeout_seconds: float = 30.0):
        """Run an async operation with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

    @staticmethod
    @asynccontextmanager
    async def async_test_context():
        """Async context manager for test setup/teardown."""
        # Setup
        try:
            yield
        finally:
            # Teardown
            pass


class PerformanceTestHelper:
    """Helper for performance testing operations."""

    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> tuple[Any, float]:
        """Measure execution time of a function."""
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        return result, execution_time

    @staticmethod
    async def measure_async_execution_time(coro) -> tuple[Any, float]:
        """Measure execution time of an async function."""
        start_time = datetime.now()
        result = await coro
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        return result, execution_time


class SecurityTestHelper:
    """Helper for security testing operations."""

    @staticmethod
    def scan_text_for_secrets(text: str, patterns: List[str]) -> List[Dict[str, str]]:
        """Scan text for potential secrets using regex patterns."""
        import re

        findings = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append(
                    {
                        "pattern": pattern,
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        return findings

    @staticmethod
    def generate_malicious_inputs() -> List[str]:
        """Generate common malicious input patterns for testing."""
        return [
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "../../../etc/passwd",  # Path traversal
            "$(rm -rf /)",  # Command injection
            "A" * 10000,  # Buffer overflow attempt
        ]
