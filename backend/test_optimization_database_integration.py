#!/usr/bin/env python3
"""
バックテスト最適化のデータベース統合テスト

最適化結果の保存と取得を確認します。
"""

import sys
import os
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.backtest_result_repository import BacktestResultRepository
from app.core.services.backtest_data_service import BacktestDataService


def test_optimization_result_saving():
    """最適化結果の保存テスト"""
    print("=== 最適化結果の保存テスト ===")

    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)
        backtest_repo = BacktestResultRepository(db)

        # 最適化実行
        config = {
            "strategy_name": "DB_INTEGRATION_TEST",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 1000000,
            "commission_rate": 0.001,
            "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
        }

        optimization_params = {
            "method": "grid",
            "maximize": "Sharpe Ratio",
            "parameters": {"n1": [10, 15], "n2": [30, 40]},
        }

        print("最適化実行中...")
        result = enhanced_service.optimize_strategy_enhanced(
            config, optimization_params
        )

        # 結果をデータベースに保存
        print("結果をデータベースに保存中...")

        # BacktestResultオブジェクトを作成
        backtest_result_data = {
            "strategy_name": config["strategy_name"],
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "start_date": datetime.fromisoformat(config["start_date"]).replace(
                tzinfo=timezone.utc
            ),
            "end_date": datetime.fromisoformat(config["end_date"]).replace(
                tzinfo=timezone.utc
            ),
            "initial_capital": config["initial_capital"],
            "commission_rate": config["commission_rate"],
            "config_json": config,
            "results_json": result,
            "total_return": result.get("performance_metrics", {}).get(
                "total_return", 0
            ),
            "sharpe_ratio": result.get("performance_metrics", {}).get(
                "sharpe_ratio", 0
            ),
            "max_drawdown": result.get("performance_metrics", {}).get(
                "max_drawdown", 0
            ),
            "total_trades": result.get("performance_metrics", {}).get(
                "total_trades", 0
            ),
            "win_rate": result.get("performance_metrics", {}).get("win_rate", 0),
            "profit_factor": result.get("performance_metrics", {}).get(
                "profit_factor", 0
            ),
        }

        saved_result = backtest_repo.save_backtest_result(backtest_result_data)
        print(f"✅ 結果が保存されました。ID: {saved_result.id}")

        # 保存された結果を取得して確認
        print("保存された結果を取得中...")
        retrieved_result = backtest_repo.get_backtest_result_by_id(saved_result.id)

        if retrieved_result:
            print("✅ 結果の取得成功")
            print(f"  戦略名: {retrieved_result.strategy_name}")
            print(f"  シンボル: {retrieved_result.symbol}")

            # パフォーマンス指標から値を取得
            performance_metrics = retrieved_result.performance_metrics or {}
            print(f"  総リターン: {performance_metrics.get('total_return', 0.0):.2f}%")
            print(
                f"  シャープレシオ: {performance_metrics.get('sharpe_ratio', 0.0):.3f}"
            )
            print(
                f"  最大ドローダウン: {performance_metrics.get('max_drawdown', 0.0):.2f}%"
            )
            print(f"  総取引数: {performance_metrics.get('total_trades', 0)}")
        else:
            print("❌ 結果の取得に失敗")

        # 結果一覧の取得テスト
        print("\n結果一覧の取得テスト...")
        results_list = backtest_repo.get_backtest_results(limit=5)
        print(f"✅ {len(results_list)} 件の結果を取得")

        # 最適化結果の詳細確認
        print("\n最適化結果の詳細:")
        if "optimization_results" in result:
            opt_results = result["optimization_results"]
            print(f"  最適パラメータ: {opt_results.get('best_parameters', {})}")
            print(f"  最適値: {opt_results.get('best_value', 'N/A')}")
            print(
                f"  テスト組み合わせ数: {opt_results.get('total_combinations', 'N/A')}"
            )

        # クリーンアップ（テストデータの削除）
        print(f"\nテストデータをクリーンアップ中...")
        backtest_repo.delete_backtest_result(saved_result.id)
        print("✅ テストデータを削除しました")

    except Exception as e:
        print(f"❌ テスト中にエラーが発生: {e}")
        raise
    finally:
        db.close()


def test_multiple_optimization_results():
    """複数の最適化結果の管理テスト"""
    print("\n=== 複数の最適化結果の管理テスト ===")

    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)
        backtest_repo = BacktestResultRepository(db)

        saved_ids = []

        # 複数の最適化を実行
        test_configs = [
            {
                "strategy_name": "MULTI_TEST_1",
                "params": {"n1": [10, 15], "n2": [30, 40]},
                "period": ("2024-01-01", "2024-01-31"),
            },
            {
                "strategy_name": "MULTI_TEST_2",
                "params": {"n1": [20, 25], "n2": [50, 60]},
                "period": ("2024-02-01", "2024-02-29"),
            },
        ]

        for i, test_config in enumerate(test_configs, 1):
            print(f"\n{i}. {test_config['strategy_name']} の最適化実行中...")

            config = {
                "strategy_name": test_config["strategy_name"],
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": test_config["period"][0],
                "end_date": test_config["period"][1],
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            }

            optimization_params = {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": test_config["params"],
            }

            result = enhanced_service.optimize_strategy_enhanced(
                config, optimization_params
            )

            # 結果を保存
            backtest_result_data = {
                "strategy_name": config["strategy_name"],
                "symbol": config["symbol"],
                "timeframe": config["timeframe"],
                "start_date": datetime.fromisoformat(config["start_date"]).replace(
                    tzinfo=timezone.utc
                ),
                "end_date": datetime.fromisoformat(config["end_date"]).replace(
                    tzinfo=timezone.utc
                ),
                "initial_capital": config["initial_capital"],
                "commission_rate": config["commission_rate"],
                "config_json": config,
                "results_json": result,
                "total_return": result.get("performance_metrics", {}).get(
                    "total_return", 0
                ),
                "sharpe_ratio": result.get("performance_metrics", {}).get(
                    "sharpe_ratio", 0
                ),
                "max_drawdown": result.get("performance_metrics", {}).get(
                    "max_drawdown", 0
                ),
                "total_trades": result.get("performance_metrics", {}).get(
                    "total_trades", 0
                ),
                "win_rate": result.get("performance_metrics", {}).get("win_rate", 0),
                "profit_factor": result.get("performance_metrics", {}).get(
                    "profit_factor", 0
                ),
            }

            saved_result = backtest_repo.save_backtest_result(backtest_result_data)
            saved_ids.append(saved_result.id)
            print(f"✅ {test_config['strategy_name']} 保存完了 (ID: {saved_result.id})")

        # 保存された結果の確認
        print(f"\n保存された結果の確認:")
        for result_id in saved_ids:
            result = backtest_repo.get_backtest_result_by_id(result_id)
            performance_metrics = result.performance_metrics or {}
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
            print(
                f"  ID {result_id}: {result.strategy_name} - Sharpe: {sharpe_ratio:.3f}"
            )

        # フィルタリングテスト
        print(f"\n戦略名でのフィルタリングテスト:")
        filtered_results = backtest_repo.get_backtest_results(
            strategy_name="MULTI_TEST_1", limit=10
        )
        print(f"✅ MULTI_TEST_1 で {len(filtered_results)} 件取得")

        # クリーンアップ
        print(f"\nテストデータをクリーンアップ中...")
        for result_id in saved_ids:
            backtest_repo.delete_backtest_result(result_id)
        print("✅ 全てのテストデータを削除しました")

    except Exception as e:
        print(f"❌ テスト中にエラーが発生: {e}")
        # エラーが発生した場合もクリーンアップを試行
        try:
            for result_id in saved_ids:
                backtest_repo.delete_backtest_result(result_id)
        except:
            pass
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("バックテスト最適化データベース統合テスト開始")
    print("=" * 80)

    test_optimization_result_saving()
    test_multiple_optimization_results()

    print("\n" + "=" * 80)
    print("データベース統合テスト完了")
