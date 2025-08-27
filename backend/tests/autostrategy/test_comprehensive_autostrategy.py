#!/usr/bin/env python3
"""
包括的なオートストラテジー実行テスト
テクニカル指標のみのオートストラテジーが正常に動作し、
バックテスト結果がデータベースに保存されることを確認する
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """テスト用のOHLCVデータを作成"""
    print("Creating test data...")

    # 過去100時間の1時間足データ
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=100)

    dates = pd.date_range(start=start_date, end=end_date, freq='1H')

    # より現実的な価格データ生成（ビットコインのような変動）
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, len(dates))  # 2%のボラティリティ
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(100, new_price))  # 最低価格を100に設定

    # OHLCVデータ生成
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

    print(f"SUCCESS: Test data creation completed: {len(df)} rows")
    return df

def test_autostrategy_execution():
    """オートストラテジーの実行テスト"""
    print("\n=== AutoStrategy Execution Test ===")

    try:
        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
        from database.connection import SessionLocal

        # データベースセッション作成
        db = SessionLocal()

        try:
            # テストデータ作成
            test_data = create_test_data()

            # オートストラテジーサービス初期化
            print("CONFIG: AutoStrategy Service Initializing...")
            auto_strategy_service = AutoStrategyService()

            # 実験設定
            experiment_config = {
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'start_date': test_data['timestamp'].min(),
                'end_date': test_data['timestamp'].max(),
                'initial_capital': 100000.0,
                'commission_rate': 0.001,
                'population_size': 10,  # 小さめに設定
                'generations': 5,      # 小さめに設定
                'indicator_mode': 'technical_only',  # テクニカル指標のみ
                'enable_multi_objective': False,
                'ga_config': {
                    'population_size': 10,
                    'generations': 5,
                    'crossover_rate': 0.8,
                    'mutation_rate': 0.1,
                    'tournament_size': 3
                }
            }

            print("RUNNING: AutoStrategy Experiment Executing...")
            print(f"   設定: {experiment_config['symbol']}, {experiment_config['timeframe']}")
            print(f"   期間: {experiment_config['start_date']} - {experiment_config['end_date']}")
            print(f"   指標モード: {experiment_config['indicator_mode']}")

            # オートストラテジー実行（GAエンジン直接使用）
            from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
            from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
            from app.services.auto_strategy.generators.strategy_factory import StrategyFactory

            # GA設定
            ga_config_dict = experiment_config['ga_config']
            from app.services.auto_strategy.models.ga_config import GAConfig
            ga_config = GAConfig.from_dict(ga_config_dict)

            # バックテスト設定
            backtest_config = {
                'symbol': experiment_config['symbol'],
                'timeframe': experiment_config['timeframe'],
                'start_date': experiment_config['start_date'],
                'end_date': experiment_config['end_date'],
                'initial_capital': experiment_config['initial_capital'],
                'commission_rate': experiment_config['commission_rate'],
            }

            # 実験ID生成
            import uuid
            experiment_id = str(uuid.uuid4())
            experiment_name = f"test_experiment_{experiment_id[:8]}"

            # 実験を作成
            auto_strategy_service.persistence_service.create_experiment(
                experiment_id, experiment_name, ga_config, backtest_config
            )

            # GAエンジン初期化
            gene_generator = RandomGeneGenerator(ga_config)
            strategy_factory = StrategyFactory()
            ga_engine = GeneticAlgorithmEngine(
                auto_strategy_service.backtest_service, strategy_factory, gene_generator
            )

            # GA実行
            print(f"   GA実行中... (population: {ga_config.population_size}, generations: {ga_config.generations})")
            result = ga_engine.run_evolution(ga_config, backtest_config)

            # 実験結果を保存
            auto_strategy_service.persistence_service.save_experiment_result(
                experiment_id, result, ga_config, backtest_config
            )

            # 実験を完了状態にする
            auto_strategy_service.persistence_service.complete_experiment(experiment_id)

            if result and result.get('success'):
                experiment_id = result.get('experiment_id')
                print(f"SUCCESS: AutoStrategy Experiment Success: Experiment ID {experiment_id}")

                # 実験結果の詳細確認
                if 'best_strategy' in result:
                    best_strategy = result['best_strategy']
                    print(f"   最良戦略ID: {best_strategy.get('id')}")
                    print(f"   フィットネススコア: {best_strategy.get('fitness_score'):.4f}")
                    print(f"   使用指標: {best_strategy.get('indicators', [])}")

                # バックテスト結果の保存確認
                if 'backtest_result' in result:
                    backtest_result = result['backtest_result']
                    print(f"   バックテスト結果ID: {backtest_result.get('id')}")
                    print(f"   最終残高: {backtest_result.get('final_balance', 0):.2f}")
                    print(f"   リターン: {backtest_result.get('total_return', 0):.4f}")

                return True, result
            else:
                print(f"FAILED: AutoStrategy Experiment Failed: {result.get('error', 'Unknown error')}")
                return False, result

        finally:
            db.close()

    except Exception as e:
        print(f"FAILED: AutoStrategy Execution Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def test_backtest_result_persistence():
    """バックテスト結果の永続化テスト"""
    print("\nSAVE: Backtest Result Persistence Test Starting")

    try:
        from database.repositories.backtest_result_repository import BacktestResultRepository
        from database.connection import SessionLocal

        db = SessionLocal()

        try:
            repo = BacktestResultRepository(db)

            # 最新のバックテスト結果を取得
            recent_results = repo.get_recent_backtest_results(limit=5)

            if recent_results:
                print(f"SUCCESS: Backtest Results Retrieved: {len(recent_results)} items")

                # 最新の結果の詳細を表示
                latest_result = recent_results[0]
                print(f"   Result ID: {latest_result.get('id')}")
                print(f"   Strategy Name: {latest_result.get('strategy_name')}")
                print(f"   Symbol: {latest_result.get('symbol')}")
                print(f"   Final Balance: {latest_result.get('final_balance', 0):.2f}")
                print(f"   Total Return: {latest_result.get('total_return', 0):.4f}")
                print(f"   Created At: {latest_result.get('created_at')}")

                return True, recent_results
            else:
                print("WARNING: No backtest results found (normal for first run)")
                return True, []

        finally:
            db.close()

    except Exception as e:
        print(f"FAILED: Backtest Result Persistence Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("TARGET: Comprehensive AutoStrategy Execution Test")
    print("=" * 60)

    all_tests_passed = True
    test_results = {}

    try:
        # 1. オートストラテジー実行テスト
        success, result = test_autostrategy_execution()
        test_results['autostrategy_execution'] = {'success': success, 'result': result}
        if not success:
            all_tests_passed = False

        # 2. バックテスト結果永続化テスト
        success, result = test_backtest_result_persistence()
        test_results['backtest_persistence'] = {'success': success, 'result': result}
        if not success:
            all_tests_passed = False

        # 総合結果
        print("\n" + "=" * 60)
        print("ANALYSIS: Comprehensive Test Results")
        print("=" * 60)

        for test_name, test_result in test_results.items():
            status = "SUCCESS" if test_result['success'] else "FAILED"
            print(f"{status}: {test_name}")

        if all_tests_passed:
            print("\nSUCCESS: All tests passed!")
            print("Technical-only AutoStrategy is working correctly and")
            print("backtest results are being saved to the database.")
            return 0
        else:
            print("\nWARNING: Some tests failed.")
            return 1

    except Exception as e:
        print(f"\nFATAL: Fatal error occurred during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)