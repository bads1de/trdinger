#!/usr/bin/env python3
"""
バックテスト結果コンバーターのデバッグスクリプト
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ログレベルを設定
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def debug_backtest_converter():
    """バックテスト結果コンバーターをデバッグ"""
    print("=" * 60)
    print("Backtest Result Converter Debug")
    print("=" * 60)

    try:
        from app.services.backtest.conversion.backtest_result_converter import (
            BacktestResultConverter,
        )

        # モックのstatsオブジェクトを作成
        class MockStats:
            def __init__(self):
                # backtesting.pyのような統計情報を模擬
                self._data = {
                    "Return [%]": 181.11,
                    "# Trades": 0,  # 問題の原因: これが0になっている
                    "Win Rate [%]": 0.0,
                    "Profit Factor": 0.0,
                    "Sharpe Ratio": 1.8147,
                    "Max. Drawdown [%]": -13.47,
                    "Best Trade [%]": 0.0,
                    "Worst Trade [%]": 0.0,
                    "Avg. Trade [%]": 0.0,
                    "Equity Final [$]": 18210734.9,
                    "Equity Peak [$]": 20000000.0,
                    "Buy & Hold Return [%]": 150.0,
                }

                # モックの取引データフレーム
                import pandas as pd

                self._trades = pd.DataFrame(
                    {
                        "EntryTime": [
                            datetime(2024, 1, 1, 10, 0),
                            datetime(2024, 1, 2, 14, 0),
                            datetime(2024, 1, 3, 9, 0),
                        ],
                        "ExitTime": [
                            datetime(2024, 1, 1, 16, 0),
                            datetime(2024, 1, 2, 18, 0),
                            datetime(2024, 1, 3, 15, 0),
                        ],
                        "EntryPrice": [50000.0, 51000.0, 49000.0],
                        "ExitPrice": [52000.0, 50500.0, 50000.0],
                        "Size": [1.0, 1.0, 1.0],
                        "PnL": [2000.0, -500.0, 1000.0],  # 2勝1敗
                        "ReturnPct": [4.0, -0.98, 2.04],
                        "Duration": [6, 4, 6],
                    }
                )

                # モックのエクイティカーブ
                self._equity_curve = pd.DataFrame(
                    {
                        "Equity": [
                            10000000.0,
                            12000000.0,
                            11500000.0,
                            12500000.0,
                            18210734.9,
                        ]
                    },
                    index=pd.date_range("2024-01-01", periods=5, freq="D"),
                )

            def get(self, key, default=None):
                return self._data.get(key, default)

            def keys(self):
                return self._data.keys()

            @property
            def index(self):
                return list(self._data.keys())

            @property
            def values(self):
                return list(self._data.values())

        # コンバーターをテスト
        converter = BacktestResultConverter()
        mock_stats = MockStats()

        print(f"モック統計データ: {dict(mock_stats._data)}")
        print(f"モック取引データ: {len(mock_stats._trades)} 件")
        print(f"取引データの詳細:")
        for i, (_, trade) in enumerate(mock_stats._trades.iterrows()):
            print(f"  取引{i+1}: PnL={trade['PnL']}, Size={trade['Size']}")

        # 統計情報を抽出
        print("\n--- 統計情報抽出テスト ---")
        extracted_stats = converter._extract_statistics(mock_stats)

        print(f"\n抽出された統計情報:")
        for key, value in extracted_stats.items():
            print(f"  {key}: {value}")

        # 特に重要な指標をチェック
        print(f"\n重要指標チェック:")
        print(f"  total_trades: {extracted_stats.get('total_trades', 'N/A')}")
        print(f"  win_rate: {extracted_stats.get('win_rate', 'N/A')}")
        print(f"  profit_factor: {extracted_stats.get('profit_factor', 'N/A')}")
        print(f"  avg_win: {extracted_stats.get('avg_win', 'N/A')}")
        print(f"  avg_loss: {extracted_stats.get('avg_loss', 'N/A')}")

        # 期待値と比較
        expected_total_trades = 3
        expected_win_rate = 66.67  # 2勝1敗
        expected_profit_factor = 3000.0 / 500.0  # 利益3000 / 損失500

        print(f"\n期待値との比較:")
        print(
            f"  total_trades: 期待={expected_total_trades}, 実際={extracted_stats.get('total_trades', 'N/A')}"
        )
        print(
            f"  win_rate: 期待={expected_win_rate:.2f}%, 実際={extracted_stats.get('win_rate', 'N/A')}"
        )
        print(
            f"  profit_factor: 期待={expected_profit_factor:.2f}, 実際={extracted_stats.get('profit_factor', 'N/A')}"
        )

        # 完全な変換テスト
        print("\n--- 完全変換テスト ---")
        config_json = {"commission_rate": 0.001}

        result = converter.convert_backtest_results(
            stats=mock_stats,
            strategy_name="Debug_Test_Strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=10000000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            config_json=config_json,
        )

        print(f"変換結果のperformance_metrics:")
        metrics = result.get("performance_metrics", {})
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"デバッグ中にエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_backtest_converter()
