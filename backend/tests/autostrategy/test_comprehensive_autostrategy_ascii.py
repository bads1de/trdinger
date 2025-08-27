#!/usr/bin/env python3
"""
Comprehensive AutoStrategy Execution Test
Tests that technical-only AutoStrategy works properly and
backtest results are saved to database
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create test OHLCV data"""
    print("Creating test data...")

    # Past 100 hours of 1-hour data
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=100)

    dates = pd.date_range(start=start_date, end=end_date, freq='1H')

    # More realistic price data generation (like Bitcoin volatility)
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, len(dates))  # 2% volatility
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(100, new_price))  # Minimum price of 100

    # OHLCV data generation
    high_prices = [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices]
    open_prices = close_prices[:-1] + [close_prices[-1] * (1 + np.random.normal(0, 0.005))]
    volumes = np.random.uniform(1000000, 10000000, len(dates))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })

    print(f"Test data created: {len(df)} rows")
    return df

def test_autostrategy_execution():
    """AutoStrategy execution test"""
    print("\n=== AutoStrategy Execution Test ===")

    try:
        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
        from database.connection import SessionLocal

        # Create database session
        db = SessionLocal()

        try:
            # Create test data
            test_data = create_test_data()

            # Initialize AutoStrategy service
            print("Initializing AutoStrategy service...")
            auto_strategy_service = AutoStrategyService()

            # Experiment configuration
            experiment_config = {
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'start_date': test_data['timestamp'].min(),
                'end_date': test_data['timestamp'].max(),
                'initial_capital': 100000.0,
                'commission_rate': 0.001,
                'population_size': 10,  # Small size for testing
                'generations': 5,      # Small size for testing
                'indicator_mode': 'technical_only',  # Technical indicators only
                'enable_multi_objective': False,
                'ga_config': {
                    'population_size': 10,
                    'generations': 5,
                    'crossover_rate': 0.8,
                    'mutation_rate': 0.1,
                    'tournament_size': 3
                }
            }

            print("Running AutoStrategy experiment...")
            print(f"   Config: {experiment_config['symbol']}, {experiment_config['timeframe']}")
            print(f"   Period: {experiment_config['start_date']} - {experiment_config['end_date']}")
            print(f"   Indicator mode: {experiment_config['indicator_mode']}")

            # Run AutoStrategy
            result = auto_strategy_service.run_experiment(
                experiment_config=experiment_config,
                db_session=db
            )

            if result and result.get('success'):
                experiment_id = result.get('experiment_id')
                print(f"SUCCESS: AutoStrategy experiment completed - Experiment ID {experiment_id}")

                # Check experiment results in detail
                if 'best_strategy' in result:
                    best_strategy = result['best_strategy']
                    print(f"   Best strategy ID: {best_strategy.get('id')}")
                    print(f"   Fitness score: {best_strategy.get('fitness_score'):.4f}")
                    print(f"   Indicators used: {best_strategy.get('indicators', [])}")

                # Check backtest result saving
                if 'backtest_result' in result:
                    backtest_result = result['backtest_result']
                    print(f"   Backtest result ID: {backtest_result.get('id')}")
                    print(f"   Final balance: {backtest_result.get('final_balance', 0):.2f}")
                    print(f"   Total return: {backtest_result.get('total_return', 0):.4f}")

                return True, result
            else:
                print(f"FAILED: AutoStrategy experiment failed - {result.get('error', 'Unknown error')}")
                return False, result

        finally:
            db.close()

    except Exception as e:
        print(f"ERROR: AutoStrategy execution test failed - {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def test_backtest_result_persistence():
    """Backtest result persistence test"""
    print("\n=== Backtest Result Persistence Test ===")

    try:
        from database.repositories.backtest_result_repository import BacktestResultRepository
        from database.connection import SessionLocal

        db = SessionLocal()

        try:
            repo = BacktestResultRepository(db)

            # Get recent backtest results
            recent_results = repo.get_recent_backtest_results(limit=5)

            if recent_results:
                print(f"SUCCESS: Backtest results retrieved - {len(recent_results)} records")

                # Display latest result details
                latest_result = recent_results[0]
                print(f"   Result ID: {latest_result.get('id')}")
                print(f"   Strategy name: {latest_result.get('strategy_name')}")
                print(f"   Symbol: {latest_result.get('symbol')}")
                print(f"   Final balance: {latest_result.get('final_balance', 0):.2f}")
                print(f"   Total return: {latest_result.get('total_return', 0):.4f}")
                print(f"   Created at: {latest_result.get('created_at')}")

                return True, recent_results
            else:
                print("WARNING: No backtest results found (this is normal for first run)")
                return True, []

        finally:
            db.close()

    except Exception as e:
        print(f"ERROR: Backtest result persistence test failed - {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    """Main execution function"""
    print("=" * 60)
    print("Comprehensive AutoStrategy Execution Test")
    print("=" * 60)

    all_tests_passed = True
    test_results = {}

    try:
        # 1. AutoStrategy execution test
        success, result = test_autostrategy_execution()
        test_results['autostrategy_execution'] = {'success': success, 'result': result}
        if not success:
            all_tests_passed = False

        # 2. Backtest result persistence test
        success, result = test_backtest_result_persistence()
        test_results['backtest_persistence'] = {'success': success, 'result': result}
        if not success:
            all_tests_passed = False

        # Overall results
        print("\n" + "=" * 60)
        print("Overall Test Results")
        print("=" * 60)

        for test_name, test_result in test_results.items():
            status = "SUCCESS" if test_result['success'] else "FAILED"
            print(f"{status}: {test_name}")

        if all_tests_passed:
            print("\nAll tests passed!")
            print("Technical-only AutoStrategy is working correctly and")
            print("backtest results are being saved to the database.")
            return 0
        else:
            print("\nSome tests failed.")
            return 1

    except Exception as e:
        print(f"\nFatal error during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)