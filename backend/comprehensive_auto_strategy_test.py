#!/usr/bin/env python3
"""
オートストラテジー機能の包括的テストスクリプト
"""

import time
import json
from datetime import datetime
from app.core.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.core.services.auto_strategy.models.ga_config import GAConfig
from database.connection import SessionLocal
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from database.repositories.ga_experiment_repository import GAExperimentRepository


def test_1_basic_auto_strategy():
    """基本的なオートストラテジー機能のテスト"""
    print("=== テスト1: 基本的なオートストラテジー機能 ===")

    try:
        service = AutoStrategyService()

        ga_config = GAConfig(
            population_size=3,
            generations=1,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=1,
            fitness_weights={
                "total_return": 0.4,
                "sharpe_ratio": 0.3,
                "max_drawdown": 0.2,
                "win_rate": 0.1,
            },
        )

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-04-09",
            "initial_capital": 100000,
            "commission_rate": 0.001,
        }

        experiment_id = service.start_strategy_generation(
            experiment_name="Comprehensive_Test_1",
            ga_config=ga_config,
            backtest_config=backtest_config,
        )

        print(f"✅ 実験開始成功: {experiment_id}")

        # 実験完了まで待機
        max_wait = 120  # 2分に延長
        start_time = time.time()

        while time.time() - start_time < max_wait:
            progress = service.get_experiment_progress(experiment_id)
            print(
                f"進捗: {progress.status} - 世代 {progress.current_generation}/{progress.total_generations}"
            )

            if progress.status == "completed":
                print("✅ 実験完了")
                return experiment_id
            elif progress.status == "failed":
                print("❌ 実験失敗")
                return None

            time.sleep(10)  # チェック間隔を延長

        print("❌ タイムアウト")
        return None

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_2_database_storage(experiment_id):
    """データベース保存の確認"""
    print("\n=== テスト2: データベース保存確認 ===")

    try:
        db = SessionLocal()

        # 1. generated_strategiesテーブルの確認
        gen_repo = GeneratedStrategyRepository(db)
        strategies = gen_repo.get_strategies_by_experiment(experiment_id)
        print(f"✅ generated_strategies保存確認: {len(strategies)}件")

        # 2. backtest_resultsテーブルの確認
        bt_repo = BacktestResultRepository(db)
        results = bt_repo.get_backtest_results(limit=10)

        auto_strategy_results = [
            r for r in results if "AUTO_STRATEGY" in r.get("strategy_name", "")
        ]
        print(
            f"✅ backtest_results保存確認: {len(auto_strategy_results)}件のオートストラテジー結果"
        )

        if auto_strategy_results:
            latest = auto_strategy_results[0]
            print(f"最新結果: {latest.get('strategy_name')}")

            # 詳細データの確認
            metrics = latest.get("performance_metrics", {})
            trade_history = latest.get("trade_history", [])
            equity_curve = latest.get("equity_curve", [])

            print(f"  パフォーマンス指標: {len(metrics)}項目")
            print(f"  取引履歴: {len(trade_history)}件")
            print(f"  資産曲線: {len(equity_curve)}ポイント")

            # 必須指標の確認
            required_metrics = [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "total_trades",
            ]
            missing_metrics = [m for m in required_metrics if m not in metrics]

            if missing_metrics:
                print(f"❌ 不足している指標: {missing_metrics}")
                return False
            else:
                print("✅ 必須指標すべて存在")

            return True
        else:
            print("❌ オートストラテジー結果が見つかりません")
            return False

    except Exception as e:
        print(f"❌ データベース確認エラー: {e}")
        return False
    finally:
        db.close()


def test_3_multiple_experiments():
    """複数実験の実行テスト"""
    print("\n=== テスト3: 複数実験実行 ===")

    try:
        service = AutoStrategyService()
        experiment_ids = []

        # 3つの異なる設定で実験実行
        configs = [
            {
                "name": "Multi_Test_A",
                "population": 3,
                "generations": 1,
                "weights": {
                    "total_return": 0.5,
                    "sharpe_ratio": 0.3,
                    "max_drawdown": 0.2,
                    "win_rate": 0.0,
                },
            },
            {
                "name": "Multi_Test_B",
                "population": 4,
                "generations": 1,
                "weights": {
                    "total_return": 0.3,
                    "sharpe_ratio": 0.4,
                    "max_drawdown": 0.2,
                    "win_rate": 0.1,
                },
            },
        ]

        for config in configs:
            ga_config = GAConfig(
                population_size=config["population"],
                generations=config["generations"],
                crossover_rate=0.8,
                mutation_rate=0.2,
                elite_size=1,
                fitness_weights=config["weights"],
            )

            backtest_config = {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-04-09",
                "initial_capital": 100000,
                "commission_rate": 0.001,
            }

            experiment_id = service.start_strategy_generation(
                experiment_name=config["name"],
                ga_config=ga_config,
                backtest_config=backtest_config,
            )

            experiment_ids.append(experiment_id)
            print(f"✅ 実験開始: {config['name']} -> {experiment_id}")

            # 少し待機
            time.sleep(2)

        # すべての実験完了を待機
        print("実験完了を待機中...")
        max_wait = 120
        start_time = time.time()

        while time.time() - start_time < max_wait:
            completed = 0
            for exp_id in experiment_ids:
                progress = service.get_experiment_progress(exp_id)
                if progress.status == "completed":
                    completed += 1

            print(f"完了済み: {completed}/{len(experiment_ids)}")

            if completed == len(experiment_ids):
                print("✅ すべての実験完了")
                return experiment_ids

            time.sleep(10)

        print("❌ 一部実験がタイムアウト")
        return experiment_ids

    except Exception as e:
        print(f"❌ 複数実験エラー: {e}")
        return []


def test_4_data_consistency():
    """データ整合性の確認"""
    print("\n=== テスト4: データ整合性確認 ===")

    try:
        db = SessionLocal()

        # 1. 実験データの整合性確認
        exp_repo = GAExperimentRepository(db)
        experiments = exp_repo.get_recent_experiments(limit=10)

        print(f"実験数: {len(experiments)}")

        for exp in experiments:
            if "Test" in exp.name:  # テスト実験のみ
                # generated_strategiesとの整合性
                gen_repo = GeneratedStrategyRepository(db)
                strategies = gen_repo.get_strategies_by_experiment(exp.id)

                # backtest_resultsとの整合性
                bt_repo = BacktestResultRepository(db)
                results = bt_repo.get_backtest_results(limit=50)

                auto_results = [
                    r for r in results if exp.name in r.get("strategy_name", "")
                ]

                print(f"実験 {exp.name}:")
                print(f"  生成戦略数: {len(strategies)}")
                print(f"  バックテスト結果数: {len(auto_results)}")

                # 最良戦略がbacktest_resultsに保存されているか確認
                if strategies and not auto_results:
                    print(f"  ❌ 最良戦略のバックテスト結果が見つかりません")
                    return False
                elif strategies and auto_results:
                    print(f"  ✅ データ整合性OK")

        return True

    except Exception as e:
        print(f"❌ データ整合性確認エラー: {e}")
        return False
    finally:
        db.close()


def test_5_frontend_compatibility():
    """フロントエンド互換性の確認"""
    print("\n=== テスト5: フロントエンド互換性確認 ===")

    try:
        db = SessionLocal()
        bt_repo = BacktestResultRepository(db)

        # 最新のオートストラテジー結果を取得
        results = bt_repo.get_backtest_results(limit=20)
        auto_results = [
            r for r in results if "AUTO_STRATEGY" in r.get("strategy_name", "")
        ]

        if not auto_results:
            print("❌ オートストラテジー結果が見つかりません")
            return False

        result = auto_results[0]
        print(f"確認対象: {result.get('strategy_name')}")

        # フロントエンドで期待される形式の確認
        required_fields = [
            "strategy_name",
            "symbol",
            "timeframe",
            "start_date",
            "end_date",
            "initial_capital",
            "performance_metrics",
            "equity_curve",
            "trade_history",
        ]

        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            print(f"❌ 不足フィールド: {missing_fields}")
            return False

        # performance_metricsの詳細確認
        metrics = result.get("performance_metrics", {})
        required_metrics = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "winning_trades",
            "losing_trades",
        ]

        missing_metrics = [m for m in required_metrics if m not in metrics]

        if missing_metrics:
            print(f"❌ 不足指標: {missing_metrics}")
            return False

        # データ型の確認
        if not isinstance(result.get("trade_history"), list):
            print("❌ trade_historyがリスト形式ではありません")
            return False

        if not isinstance(result.get("equity_curve"), list):
            print("❌ equity_curveがリスト形式ではありません")
            return False

        print("✅ フロントエンド互換性OK")
        print(f"  取引履歴: {len(result.get('trade_history', []))}件")
        print(f"  資産曲線: {len(result.get('equity_curve', []))}ポイント")
        print(f"  リターン: {metrics.get('total_return', 0):.2f}%")
        print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")

        return True

    except Exception as e:
        print(f"❌ フロントエンド互換性確認エラー: {e}")
        return False
    finally:
        db.close()


def main():
    """包括的テストの実行"""
    print("🚀 オートストラテジー機能 包括的テスト開始")
    print("=" * 50)

    results = {}

    # テスト1: 基本機能
    experiment_id = test_1_basic_auto_strategy()
    results["basic_functionality"] = experiment_id is not None

    if experiment_id:
        # テスト2: データベース保存
        results["database_storage"] = test_2_database_storage(experiment_id)
    else:
        results["database_storage"] = False

    # テスト3: 複数実験
    multi_experiments = test_3_multiple_experiments()
    results["multiple_experiments"] = len(multi_experiments) > 0

    # テスト4: データ整合性
    results["data_consistency"] = test_4_data_consistency()

    # テスト5: フロントエンド互換性
    results["frontend_compatibility"] = test_5_frontend_compatibility()

    # 結果サマリー
    print("\n" + "=" * 50)
    print("📊 テスト結果サマリー")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    print(f"\n総合結果: {passed_tests}/{total_tests} テスト通過")

    if passed_tests == total_tests:
        print("🎉 すべてのテストが成功しました！")
        print("オートストラテジー機能は完全に動作しています。")
        return True
    else:
        print("⚠️  一部のテストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
