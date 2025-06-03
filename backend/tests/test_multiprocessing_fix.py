"""
マルチプロセシング問題修正のテストスクリプト

戦略クラスの修正とマルチプロセシング有効化のテスト
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from datetime import datetime, timedelta
import time


def test_simple_backtest():
    """シンプルなバックテストのテスト"""
    print("=== シンプルなバックテストのテスト ===")

    service = EnhancedBacktestService()

    # 基本設定
    config = {
        "strategy_name": "SMA_Cross_Test",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": 10000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "SMA_CROSS",
            "parameters": {"n1": 20, "n2": 50},
        },
    }

    try:
        result = service.run_backtest(config)
        print("✅ シンプルなバックテスト成功")
        print(f"総リターン: {result['performance_metrics']['total_return']:.2f}%")
        print(f"シャープレシオ: {result['performance_metrics']['sharpe_ratio']:.3f}")
        return True
    except Exception as e:
        print(f"❌ シンプルなバックテスト失敗: {e}")
        return False


def test_optimization_with_multiprocessing():
    """マルチプロセシング有効での最適化テスト"""
    print("\n=== マルチプロセシング最適化テスト ===")

    service = EnhancedBacktestService()

    # 基本設定
    config = {
        "strategy_name": "SMA_Cross_Optimization",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-15",  # 短期間でテスト
        "initial_capital": 10000,
        "commission_rate": 0.001,
        "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
    }

    # 最適化パラメータ（小さな範囲でテスト）
    optimization_params = {
        "method": "grid",
        "maximize": "Sharpe Ratio",
        "parameters": {
            "n1": range(10, 21, 5),  # [10, 15, 20]
            "n2": range(30, 41, 5),  # [30, 35, 40]
        },
    }

    try:
        start_time = time.time()
        result = service.optimize_strategy_enhanced(config, optimization_params)
        execution_time = time.time() - start_time

        print("✅ マルチプロセシング最適化成功")
        print(f"実行時間: {execution_time:.2f}秒")
        print(f"最適パラメータ: {result.get('optimized_parameters', {})}")
        print(
            f"最適シャープレシオ: {result['performance_metrics']['sharpe_ratio']:.3f}"
        )
        return True
    except Exception as e:
        print(f"❌ マルチプロセシング最適化失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_rsi_strategy():
    """RSI戦略のテスト"""
    print("\n=== RSI戦略テスト ===")

    service = EnhancedBacktestService()

    config = {
        "strategy_name": "RSI_Test",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-15",
        "initial_capital": 10000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "RSI",
            "parameters": {"period": 14, "oversold": 30, "overbought": 70},
        },
    }

    try:
        result = service.run_backtest(config)
        print("✅ RSI戦略テスト成功")
        print(f"総リターン: {result['performance_metrics']['total_return']:.2f}%")
        return True
    except Exception as e:
        print(f"❌ RSI戦略テスト失敗: {e}")
        return False


def test_macd_strategy():
    """MACD戦略のテスト"""
    print("\n=== MACD戦略テスト ===")

    service = EnhancedBacktestService()

    config = {
        "strategy_name": "MACD_Test",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-15",
        "initial_capital": 10000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "MACD",
            "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        },
    }

    try:
        result = service.run_backtest(config)
        print("✅ MACD戦略テスト成功")
        print(f"総リターン: {result['performance_metrics']['total_return']:.2f}%")
        return True
    except Exception as e:
        print(f"❌ MACD戦略テスト失敗: {e}")
        return False


def main():
    """メインテスト実行"""
    print("マルチプロセシング問題修正テスト開始")
    print("=" * 50)

    tests = [
        test_simple_backtest,
        test_rsi_strategy,
        test_macd_strategy,
        test_optimization_with_multiprocessing,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"テスト結果: {passed}/{total} 成功")

    if passed == total:
        print("🎉 すべてのテストが成功しました！")
        print("マルチプロセシング問題が解決されました。")
    else:
        print("⚠️  一部のテストが失敗しました。")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
