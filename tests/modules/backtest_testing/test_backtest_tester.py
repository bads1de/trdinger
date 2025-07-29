"""
Test file for BacktestTester class to validate implementation.
"""

import asyncio
import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest_tester import BacktestTester, BacktestMetricsResult
from config.test_config import BacktestTestConfig
from orchestrator.test_orchestrator import TestStatus


def test_backtest_tester_initialization():
    """Test BacktestTester initialization."""
    config = BacktestTestConfig(
        sharpe_ratio_tolerance=0.01,
        max_drawdown_tolerance=0.01,
        win_rate_tolerance=0.01,
        extreme_condition_scenarios=["high_volatility", "market_crash"],
    )
    tester = BacktestTester(config)

    assert tester.get_module_name() == "backtest_testing"
    assert tester.config is not None
    assert len(tester.known_test_cases) > 0
    print("✓ BacktestTester initialization test passed")


def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation with known values."""
    tester = BacktestTester()

    # Test with known returns
    returns = [0.02, -0.01, 0.03, -0.005, 0.015]
    result = tester._calculate_sharpe_ratio(returns)

    assert isinstance(result, Decimal)
    assert result != Decimal("0")  # Should have a non-zero Sharpe ratio
    print(f"✓ Sharpe ratio calculation test passed: {result}")


def test_max_drawdown_calculation():
    """Test maximum drawdown calculation with known values."""
    tester = BacktestTester()

    # Test with known equity curve
    equity_curve = [10000.0, 12000.0, 8000.0, 9000.0, 11000.0]
    result = tester._calculate_max_drawdown(equity_curve)

    assert isinstance(result, Decimal)
    # Should be approximately 33.33% (4000/12000)
    expected = Decimal("0.3333")
    assert abs(result - expected) < Decimal("0.01")
    print(f"✓ Max drawdown calculation test passed: {result}")


def test_win_rate_calculation():
    """Test win rate calculation with known values."""
    tester = BacktestTester()

    # Test with known trades
    trades = [
        {"pnl": Decimal("100.00")},
        {"pnl": Decimal("-50.00")},
        {"pnl": Decimal("75.00")},
        {"pnl": Decimal("-25.00")},
    ]
    result = tester._calculate_win_rate(trades)

    assert isinstance(result, Decimal)
    # Should be 50% (2 winning out of 4 trades)
    expected = Decimal("50.00")
    assert result == expected
    print(f"✓ Win rate calculation test passed: {result}%")


def test_total_return_calculation():
    """Test total return calculation with known values."""
    tester = BacktestTester()

    # Test with known trades
    trades = [
        {"pnl": Decimal("100.00")},
        {"pnl": Decimal("-50.00")},
        {"pnl": Decimal("75.00")},
        {"pnl": Decimal("-25.00")},
    ]
    result = tester._calculate_total_return(trades)

    assert isinstance(result, Decimal)
    # Should be 100.00 (100 - 50 + 75 - 25)
    expected = Decimal("100.00")
    assert result == expected
    print(f"✓ Total return calculation test passed: {result}")


async def test_run_tests_method():
    """Test the main run_tests method."""
    tester = BacktestTester()
    result = await tester.run_tests()

    assert result.module_name == "backtest_testing"
    assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
    assert result.tests_run > 0
    assert result.execution_time_seconds > 0
    assert result.start_time is not None
    assert result.end_time is not None

    print(f"✓ Run tests method passed:")
    print(f"  - Status: {result.status}")
    print(f"  - Tests run: {result.tests_run}")
    print(f"  - Tests passed: {result.tests_passed}")
    print(f"  - Tests failed: {result.tests_failed}")
    print(f"  - Execution time: {result.execution_time_seconds:.2f}s")

    if result.error_messages:
        print(f"  - Errors: {result.error_messages}")

    return result


if __name__ == "__main__":
    # Run all tests
    print("Running BacktestTester validation tests...")

    try:
        test_backtest_tester_initialization()
        test_sharpe_ratio_calculation()
        test_max_drawdown_calculation()
        test_win_rate_calculation()
        test_total_return_calculation()

        # Run async test
        async def main():
            result = await test_run_tests_method()
            return result

        result = asyncio.run(main())

        print("\n" + "=" * 50)
        print("All BacktestTester validation tests completed!")
        print(f"Overall result: {result.status}")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
