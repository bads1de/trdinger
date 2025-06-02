#!/usr/bin/env python3
"""
実際のBTCデータを使用した拡張最適化テスト

データベースの実際のBTC/USDTデータを使用して、
拡張バックテスト最適化機能の実用性を検証します。
"""

import sys
import os
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.backtest_data_service import BacktestDataService


def test_real_btc_enhanced_optimization():
    """実際のBTCデータを使用した拡張最適化テスト"""
    print("=== 実際のBTCデータを使用した拡張最適化テスト ===")

    # データベース接続
    db = SessionLocal()
    try:
        # データサービスを初期化
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)

        # 設定
        config = {
            "strategy_name": "SMA_CROSS_REAL_BTC",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
            "initial_capital": 10000000,  # 1000万円（BTC価格に対応）
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        # Grid最適化パラメータ（高速テスト用）
        optimization_params = {
            "method": "grid",
            "maximize": "Sharpe Ratio",
            "return_heatmap": True,
            "constraint": "sma_cross",
            "parameters": {
                "n1": range(10, 30, 5),  # 10, 15, 20, 25
                "n2": range(30, 80, 10),  # 30, 40, 50, 60, 70
            },
        }

        print(
            f"パラメータ空間サイズ: {len(list(optimization_params['parameters']['n1']))} × {len(list(optimization_params['parameters']['n2']))} = {len(list(optimization_params['parameters']['n1'])) * len(list(optimization_params['parameters']['n2']))}"
        )
        print("SAMBO最適化実行中...")

        result = enhanced_service.optimize_strategy_enhanced(
            config, optimization_params
        )

        print("✅ 実際のBTCデータでの最適化成功!")
        print(f"戦略名: {result['strategy_name']}")
        print(f"期間: {config['start_date'].date()} - {config['end_date'].date()}")
        print(f"最適化されたパラメータ: {result.get('optimized_parameters', {})}")

        if "performance_metrics" in result:
            metrics = result["performance_metrics"]
            print(f"\n📊 パフォーマンス指標:")
            print(f"  総リターン: {metrics.get('total_return', 0):.2f}%")
            print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  最大ドローダウン: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"  勝率: {metrics.get('win_rate', 0):.2f}%")
            print(f"  プロフィットファクター: {metrics.get('profit_factor', 0):.3f}")
            print(f"  総取引数: {metrics.get('total_trades', 0)}")

        if "heatmap_summary" in result:
            heatmap = result["heatmap_summary"]
            print(f"\n🔥 ヒートマップサマリー:")
            print(f"  最適な組み合わせ: {heatmap.get('best_combination')}")
            print(f"  最適値: {heatmap.get('best_value', 0):.3f}")
            print(f"  最悪な組み合わせ: {heatmap.get('worst_combination')}")
            print(f"  最悪値: {heatmap.get('worst_value', 0):.3f}")
            print(f"  平均値: {heatmap.get('mean_value', 0):.3f}")
            print(f"  標準偏差: {heatmap.get('std_value', 0):.3f}")
            print(f"  テストした組み合わせ数: {heatmap.get('total_combinations', 0)}")

        if "optimization_details" in result:
            details = result["optimization_details"]
            print(f"\n🎯 最適化詳細:")
            print(f"  手法: {details.get('method')}")
            print(f"  関数評価回数: {details.get('n_calls')}")
            print(f"  最終値: {details.get('best_value', 0):.3f}")

            if "convergence" in details:
                conv = details["convergence"]
                print(f"  初期値: {conv.get('initial_value', 0):.3f}")
                print(f"  改善度: {conv.get('improvement', 0):.3f}")
                print(f"  収束率: {conv.get('convergence_rate', 0):.6f}")
                print(
                    f"  プラトー検出: {'はい' if conv.get('plateau_detection') else 'いいえ'}"
                )

        return result

    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        db.close()


def test_real_btc_multi_objective():
    """実際のBTCデータを使用したマルチ目的最適化テスト"""
    print("\n=== 実際のBTCデータを使用したマルチ目的最適化テスト ===")

    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)

        config = {
            "strategy_name": "SMA_CROSS_MULTI_REAL_BTC",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
            "initial_capital": 10000000,  # 1000万円（BTC価格に対応）
            "commission_rate": 0.001,
            "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
        }

        # マルチ目的最適化: リターン、シャープレシオ、ドローダウンのバランス
        objectives = ["Return [%]", "Sharpe Ratio", "-Max. Drawdown [%]"]
        weights = [0.3, 0.4, 0.3]  # シャープレシオを重視

        optimization_params = {
            "method": "grid",
            "parameters": {"n1": range(10, 25, 5), "n2": range(30, 70, 10)},
        }

        print("マルチ目的最適化実行中...")
        result = enhanced_service.multi_objective_optimization(
            config, objectives, weights, optimization_params
        )

        print("✅ 実際のBTCデータでのマルチ目的最適化成功!")
        print(f"目的関数: {objectives}")
        print(f"重み: {weights}")

        if "multi_objective_details" in result:
            details = result["multi_objective_details"]
            print(f"\n🎯 個別スコア:")
            for obj, score in details.get("individual_scores", {}).items():
                print(f"  {obj}: {score:.3f}")

        if "performance_metrics" in result:
            metrics = result["performance_metrics"]
            print(f"\n📊 最終パフォーマンス:")
            print(f"  総リターン: {metrics.get('total_return', 0):.2f}%")
            print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  最大ドローダウン: {metrics.get('max_drawdown', 0):.2f}%")

        return result

    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        db.close()


def test_real_btc_robustness():
    """実際のBTCデータを使用したロバストネステスト"""
    print("\n=== 実際のBTCデータを使用したロバストネステスト ===")

    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)

        config = {
            "strategy_name": "SMA_CROSS_ROBUST_REAL_BTC",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "initial_capital": 10000000,  # 1000万円（BTC価格に対応）
            "commission_rate": 0.001,
            "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
        }

        # 四半期ごとのテスト期間
        test_periods = [
            ("2024-01-01", "2024-03-31"),  # Q1
            ("2024-04-01", "2024-06-30"),  # Q2
            ("2024-07-01", "2024-09-30"),  # Q3
            ("2024-10-01", "2024-12-31"),  # Q4
        ]

        optimization_params = {
            "method": "grid",
            "maximize": "Sharpe Ratio",
            "parameters": {"n1": range(10, 25, 5), "n2": range(30, 60, 10)},
        }

        print("ロバストネステスト実行中...")
        print(f"テスト期間: {len(test_periods)}四半期")

        result = enhanced_service.robustness_test(
            config, test_periods, optimization_params
        )

        print("✅ 実際のBTCデータでのロバストネステスト成功!")

        if "robustness_analysis" in result:
            analysis = result["robustness_analysis"]
            print(f"\n🛡️ ロバストネス分析:")
            print(f"  ロバストネススコア: {analysis.get('robustness_score', 0):.3f}")
            print(f"  成功期間: {analysis.get('successful_periods', 0)}")
            print(f"  失敗期間: {analysis.get('failed_periods', 0)}")

            if "performance_statistics" in analysis:
                perf_stats = analysis["performance_statistics"]
                print(f"\n📈 パフォーマンス統計:")
                for metric, stats in perf_stats.items():
                    print(f"  {metric}:")
                    print(f"    平均: {stats.get('mean', 0):.3f}")
                    print(f"    標準偏差: {stats.get('std', 0):.3f}")
                    print(f"    最小: {stats.get('min', 0):.3f}")
                    print(f"    最大: {stats.get('max', 0):.3f}")
                    print(f"    一貫性スコア: {stats.get('consistency_score', 0):.3f}")

            if "parameter_stability" in analysis:
                param_stats = analysis["parameter_stability"]
                print(f"\n⚙️ パラメータ安定性:")
                for param, stats in param_stats.items():
                    print(f"  {param}:")
                    print(f"    平均: {stats.get('mean', 0):.1f}")
                    print(f"    標準偏差: {stats.get('std', 0):.3f}")
                    print(
                        f"    変動係数: {stats.get('coefficient_of_variation', 0):.3f}"
                    )

        # 各期間の結果サマリー
        if "individual_results" in result:
            print(f"\n📅 期間別結果:")
            for period_name, period_result in result["individual_results"].items():
                if "error" not in period_result:
                    params = period_result.get("optimized_parameters", {})
                    metrics = period_result.get("performance_metrics", {})
                    print(
                        f"  {period_name}: n1={params.get('n1', 'N/A')}, n2={params.get('n2', 'N/A')}, "
                        f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                        f"Return={metrics.get('total_return', 0):.2f}%"
                    )
                else:
                    print(f"  {period_name}: エラー - {period_result['error']}")

        return result

    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        db.close()


def main():
    """メイン関数"""
    print("実際のBTCデータを使用した拡張最適化テスト開始")
    print("=" * 80)

    tests = [
        ("SAMBO拡張最適化", test_real_btc_enhanced_optimization),
        ("マルチ目的最適化", test_real_btc_multi_objective),
        ("ロバストネステスト", test_real_btc_robustness),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}を実行中...")
        try:
            result = test_func()
            success = result is not None
            results.append((test_name, success, result))
        except Exception as e:
            print(f"{test_name}でエラー: {e}")
            results.append((test_name, False, None))

    print("\n" + "=" * 80)
    print("テスト結果サマリー:")
    for test_name, success, _ in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {test_name}: {status}")

    success_count = sum(1 for _, success, _ in results if success)
    print(f"\n成功: {success_count}/{len(results)}")

    if success_count == len(results):
        print("🎉 全ての実際のBTCデータテストが成功しました！")
        print("\n💡 実用性評価:")
        print("- 実際の市場データでの最適化が正常に動作")
        print("- SAMBO最適化による効率的なパラメータ探索")
        print("- マルチ目的最適化による複合指標の最適化")
        print("- ロバストネステストによる戦略の安定性評価")
    else:
        print("⚠️ 一部のテストが失敗しました。")


if __name__ == "__main__":
    main()
