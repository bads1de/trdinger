#!/usr/bin/env python3
"""
拡張バックテスト最適化機能のデモテスト

実際のデータを使用して拡張最適化機能をテストします。
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from app.core.services.backtest_data_service import BacktestDataService
from unittest.mock import Mock


def create_sample_data():
    """サンプルOHLCVデータを作成"""
    print("サンプルデータを作成中...")
    
    # 1年間の日次データを生成
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    # 現実的なBTC価格データを生成
    base_price = 50000
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # High >= max(Open, Close), Low <= min(Open, Close) を保証
    df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
    df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
    
    print(f"データ作成完了: {len(df)}行")
    return df


def test_basic_enhanced_optimization():
    """基本的な拡張最適化テスト"""
    print("\n=== 基本的な拡張最適化テスト ===")
    
    # サンプルデータを作成
    sample_data = create_sample_data()
    
    # モックデータサービスを作成
    mock_data_service = Mock(spec=BacktestDataService)
    mock_data_service.get_ohlcv_for_backtest.return_value = sample_data
    
    # 拡張バックテストサービスを初期化
    enhanced_service = EnhancedBacktestService(data_service=mock_data_service)
    
    # 設定
    config = {
        "strategy_name": "SMA_CROSS_ENHANCED_TEST",
        "symbol": "BTC/USDT",
        "timeframe": "1d",
        "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
        "initial_capital": 100000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "SMA_CROSS",
            "parameters": {"n1": 20, "n2": 50}
        }
    }
    
    # 最適化パラメータ（小さな範囲でテスト）
    optimization_params = {
        "method": "grid",  # 高速化のためgridを使用
        "maximize": "Sharpe Ratio",
        "return_heatmap": True,
        "constraint": "sma_cross",
        "parameters": {
            "n1": range(10, 25, 5),  # [10, 15, 20]
            "n2": range(30, 55, 10)  # [30, 40, 50]
        }
    }
    
    try:
        print("最適化実行中...")
        result = enhanced_service.optimize_strategy_enhanced(config, optimization_params)
        
        print("✅ 最適化成功!")
        print(f"戦略名: {result['strategy_name']}")
        print(f"最適化されたパラメータ: {result.get('optimized_parameters', {})}")
        
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print(f"総リターン: {metrics.get('total_return', 0):.2f}%")
            print(f"シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"最大ドローダウン: {metrics.get('max_drawdown', 0):.2f}%")
        
        if 'heatmap_summary' in result:
            heatmap = result['heatmap_summary']
            print(f"最適な組み合わせ: {heatmap.get('best_combination')}")
            print(f"最適値: {heatmap.get('best_value', 0):.3f}")
            print(f"テストした組み合わせ数: {heatmap.get('total_combinations', 0)}")
        
        if 'optimization_metadata' in result:
            metadata = result['optimization_metadata']
            print(f"最適化手法: {metadata.get('method')}")
            print(f"パラメータ空間サイズ: {metadata.get('parameter_space_size')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 最適化エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_objective_optimization():
    """マルチ目的最適化テスト"""
    print("\n=== マルチ目的最適化テスト ===")
    
    # サンプルデータを作成
    sample_data = create_sample_data()
    
    # モックデータサービスを作成
    mock_data_service = Mock(spec=BacktestDataService)
    mock_data_service.get_ohlcv_for_backtest.return_value = sample_data
    
    # 拡張バックテストサービスを初期化
    enhanced_service = EnhancedBacktestService(data_service=mock_data_service)
    
    # 設定
    config = {
        "strategy_name": "SMA_CROSS_MULTI_TEST",
        "symbol": "BTC/USDT",
        "timeframe": "1d",
        "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
        "initial_capital": 100000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "SMA_CROSS",
            "parameters": {}
        }
    }
    
    # マルチ目的最適化
    objectives = ['Sharpe Ratio', 'Return [%]', '-Max. Drawdown [%]']
    weights = [0.4, 0.4, 0.2]
    optimization_params = {
        "method": "grid",
        "parameters": {
            "n1": range(10, 25, 5),
            "n2": range(30, 55, 10)
        }
    }
    
    try:
        print("マルチ目的最適化実行中...")
        result = enhanced_service.multi_objective_optimization(
            config, objectives, weights, optimization_params
        )
        
        print("✅ マルチ目的最適化成功!")
        print(f"目的関数: {objectives}")
        print(f"重み: {weights}")
        
        if 'multi_objective_details' in result:
            details = result['multi_objective_details']
            print("個別スコア:")
            for obj, score in details.get('individual_scores', {}).items():
                print(f"  {obj}: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ マルチ目的最適化エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_robustness_test():
    """ロバストネステスト"""
    print("\n=== ロバストネステスト ===")
    
    # サンプルデータを作成
    sample_data = create_sample_data()
    
    # モックデータサービスを作成
    mock_data_service = Mock(spec=BacktestDataService)
    mock_data_service.get_ohlcv_for_backtest.return_value = sample_data
    
    # 拡張バックテストサービスを初期化
    enhanced_service = EnhancedBacktestService(data_service=mock_data_service)
    
    # 設定
    config = {
        "strategy_name": "SMA_CROSS_ROBUST_TEST",
        "symbol": "BTC/USDT",
        "timeframe": "1d",
        "initial_capital": 100000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "SMA_CROSS",
            "parameters": {}
        }
    }
    
    # テスト期間
    test_periods = [
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31")
    ]
    
    optimization_params = {
        "method": "grid",
        "maximize": "Sharpe Ratio",
        "parameters": {
            "n1": [10, 20],
            "n2": [30, 50]
        }
    }
    
    try:
        print("ロバストネステスト実行中...")
        result = enhanced_service.robustness_test(
            config, test_periods, optimization_params
        )
        
        print("✅ ロバストネステスト成功!")
        print(f"テスト期間数: {result['total_periods']}")
        
        if 'robustness_analysis' in result:
            analysis = result['robustness_analysis']
            print(f"ロバストネススコア: {analysis.get('robustness_score', 0):.3f}")
            print(f"成功期間: {analysis.get('successful_periods', 0)}")
            print(f"失敗期間: {analysis.get('failed_periods', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ロバストネステストエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("拡張バックテスト最適化機能のデモテスト開始")
    print("=" * 60)
    
    tests = [
        ("基本的な拡張最適化", test_basic_enhanced_optimization),
        ("マルチ目的最適化", test_multi_objective_optimization),
        ("ロバストネステスト", test_robustness_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name}でエラー: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー:")
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    print(f"\n成功: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("🎉 全てのテストが成功しました！")
    else:
        print("⚠️ 一部のテストが失敗しました。")


if __name__ == "__main__":
    main()
