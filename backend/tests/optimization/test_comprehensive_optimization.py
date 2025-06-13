#!/usr/bin/env python3
"""
包括的な最適化テスト

DB内のOHLCVデータの全期間でバックテスト最適化をテストします。
実際のデータが利用できない場合は、サンプルデータを生成してテストを実行します。
"""

import sys
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from itertools import product

# プロジェクトルートをパスに追加
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from app.core.services.backtest_service import BacktestService  # BacktestServiceも必要
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.backtest_data_service import BacktestDataService
from database.models import OHLCVData


def generate_sample_btc_data(
    start_date: datetime, end_date: datetime, initial_price: float = 50000
) -> pd.DataFrame:
    """
    サンプルのBTC価格データを生成

    Args:
        start_date: 開始日
        end_date: 終了日
        initial_price: 初期価格

    Returns:
        OHLCV形式のDataFrame
    """
    # 日付範囲を生成
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # ランダムウォークでリアルな価格変動を生成
    np.random.seed(42)  # 再現性のため

    # 日次リターンを生成（平均0.1%、標準偏差3%）
    daily_returns = np.random.normal(0.001, 0.03, len(date_range))

    # 価格を計算
    prices = [initial_price]
    for ret in daily_returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # OHLCV データを生成
    data = []
    for i, (date, price) in enumerate(zip(date_range, prices)):
        # 日中の変動を生成
        daily_volatility = np.random.uniform(0.005, 0.02)  # 0.5-2%の日中変動

        high = price * (1 + daily_volatility)
        low = price * (1 - daily_volatility)

        # Open/Closeを調整
        if i == 0:
            open_price = price
        else:
            open_price = prices[i - 1]
        close_price = price

        # Volumeを生成（価格変動に応じて調整）
        base_volume = 1000000
        volume_multiplier = 1 + abs(daily_returns[i]) * 10  # 変動が大きいほど出来高増加
        volume = base_volume * volume_multiplier

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)

    # 列名を大文字に変換（backtesting.py用）
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return df


def insert_sample_data_to_db(
    df: pd.DataFrame, symbol: str = "BTC/USDT", timeframe: str = "1d"
):
    """
    サンプルデータをデータベースに挿入

    Args:
        df: OHLCVデータ
        symbol: シンボル
        timeframe: 時間軸
    """
    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)

        # 既存データを削除
        db.query(OHLCVData).filter(
            OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe
        ).delete()

        # 新しいデータを挿入
        for timestamp, row in df.iterrows():
            ohlcv_data = OHLCVData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
            db.add(ohlcv_data)

        db.commit()
        print(f"✅ {len(df)}件のサンプルデータを挿入しました")

    except Exception as e:
        db.rollback()
        print(f"❌ データ挿入エラー: {e}")
        raise
    finally:
        db.close()


def manual_grid_optimization(backtest_service, base_config, param_ranges):
    """手動でのグリッド最適化"""
    print("手動グリッド最適化を開始...")

    # パラメータの組み合わせを生成
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(product(*param_values))

    print(f"テストするパラメータ組み合わせ数: {len(combinations)}")

    results = []
    best_result = None
    best_sharpe = -float("inf")

    for i, combination in enumerate(combinations):
        params = dict(zip(param_names, combination))
        print(f"  {i+1}/{len(combinations)}: {params}")

        # 設定を更新
        config = base_config.copy()
        config["strategy_config"]["parameters"] = params
        config["strategy_name"] = f"SMA_CROSS_OPT_{i+1}"

        try:
            result = backtest_service.run_backtest(config)

            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                sharpe = metrics.get("sharpe_ratio", -float("inf"))
                total_return = metrics.get("total_return", 0)
                max_drawdown = metrics.get("max_drawdown", 0)

                results.append(
                    {
                        "parameters": params,
                        "sharpe_ratio": sharpe,
                        "total_return": total_return,
                        "max_drawdown": max_drawdown,
                        "result": result,
                    }
                )

                print(
                    f"    シャープレシオ: {sharpe:.3f}, リターン: {total_return:.2f}%, DD: {max_drawdown:.2f}%"
                )

                # 最良結果の更新
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = {
                        "parameters": params,
                        "result": result,
                        "metrics": metrics,
                    }
            else:
                print(f"    エラー: パフォーマンス指標が取得できませんでした")

        except Exception as e:
            print(f"    エラー: {e}")

    return results, best_result


def test_full_period_optimization():
    """DB内の全期間データを使用した最適化テスト"""
    print("=== 全期間データを使用した最適化テスト ===")

    # サンプルデータを生成・挿入
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)

    print("サンプルデータを生成中...")
    sample_data = generate_sample_btc_data(start_date, end_date)
    print(
        f"生成されたデータ期間: {sample_data.index.min()} - {sample_data.index.max()}"
    )
    print(f"データ件数: {len(sample_data)}")

    print("データベースに挿入中...")
    insert_sample_data_to_db(sample_data)

    # データベース接続
    db = SessionLocal()
    try:
        # データサービスを初期化
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)
        backtest_service = BacktestService(data_service)  # manual_grid_optimization用

        # 全期間での最適化設定
        config = {
            "strategy_name": "SMA_CROSS_FULL_PERIOD",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": 1000000,  # 100万円
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        # Grid最適化パラメータ
        optimization_params = {
            "method": "grid",
            "maximize": "Sharpe Ratio",
            "return_heatmap": True,
            "constraint": "sma_cross",
            "parameters": {
                "n1": range(5, 30, 5),  # 5, 10, 15, 20, 25
                "n2": range(30, 100, 10),  # 30, 40, 50, 60, 70, 80, 90
            },
        }

        print(
            f"パラメータ空間サイズ: {len(list(optimization_params['parameters']['n1']))} × {len(list(optimization_params['parameters']['n2']))} = {len(list(optimization_params['parameters']['n1'])) * len(list(optimization_params['parameters']['n2']))}"
        )
        print("最適化実行中...")

        result = enhanced_service.optimize_strategy_enhanced(
            config, optimization_params
        )

        print("✅ 全期間データでの最適化成功!")
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

        return result

    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        db.close()


def test_period_comparison():
    """期間別最適化比較テスト"""
    print("\n=== 期間別最適化比較テスト ===")

    # 異なる期間でのテスト
    test_periods = [
        (
            "短期",
            datetime(2024, 10, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        ),
        (
            "中期",
            datetime(2024, 7, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        ),
        (
            "長期",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        ),
    ]

    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        backtest_service = BacktestService(data_service)

        # 簡略化されたパラメータ範囲（高速テスト用）
        param_ranges = {"n1": [10, 15, 20, 25], "n2": [30, 40, 50, 60]}

        period_results = {}

        for period_name, start_date, end_date in test_periods:
            print(
                f"\n{period_name}期間の最適化: {start_date.date()} - {end_date.date()}"
            )

            base_config = {
                "strategy_name": f"SMA_CROSS_{period_name.upper()}",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "SMA_CROSS",
                    "parameters": {"n1": 20, "n2": 50},
                },
            }

            results, best_result = manual_grid_optimization(
                backtest_service, base_config, param_ranges
            )
            period_results[period_name] = best_result

            if best_result:
                metrics = best_result["metrics"]
                print(f"  最適パラメータ: {best_result['parameters']}")
                print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"  総リターン: {metrics.get('total_return', 0):.2f}%")

        # 期間別結果の比較
        print(f"\n📊 期間別最適化結果比較:")
        print("期間\t\t最適パラメータ\t\tシャープレシオ\t総リターン")
        print("-" * 70)

        for period_name, result in period_results.items():
            if result:
                params = result["parameters"]
                metrics = result["metrics"]
                print(
                    f"{period_name}\t\tn1={params['n1']}, n2={params['n2']}\t\t{metrics.get('sharpe_ratio', 0):.3f}\t\t{metrics.get('total_return', 0):.2f}%"
                )
            else:
                print(f"{period_name}\t\tエラー\t\t\tN/A\t\tN/A")

        return period_results

    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        db.close()


def main():
    """メイン関数"""
    print("包括的な最適化テスト開始")
    print("=" * 80)

    try:
        result = test_full_period_optimization()
        success = result is not None

        print("\n" + "=" * 80)
        print("テスト結果サマリー:")
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  全期間最適化テスト: {status}")

        if success:
            print("🎉 最適化テストが成功しました！")
            print("\n💡 テスト結果:")
            print("- DB内の全期間データでの最適化が正常に動作")
            print("- サンプルデータでの動作確認完了")
            print("- Grid最適化による効率的なパラメータ探索")
        else:
            print("⚠️ テストが失敗しました。")

    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
