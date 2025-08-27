#!/usr/bin/env python3
"""
シンプルなオートストラテジーテスト
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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_autostrategy_simple():
    """シンプルなオートストラテジーテスト"""
    print("=" * 60)
    print("Simple AutoStrategy Test")
    print("=" * 60)

    try:
        # 必要なモジュールをインポート
        from app.services.auto_strategy.generators.strategy_factory import (
            StrategyFactory,
        )
        from app.services.auto_strategy.models.gene_strategy import (
            StrategyGene,
            IndicatorGene,
            Condition,
        )
        from app.services.backtest.conversion.backtest_result_converter import (
            BacktestResultConverter,
        )
        from backtesting import Backtest
        import pandas as pd
        import numpy as np
        import uuid

        # テストデータを作成
        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        np.random.seed(42)

        # より変動の大きいランダムウォークでOHLCVデータを生成
        price_changes = np.random.randn(200) * 300
        close_prices = 50000 + np.cumsum(price_changes)
        high_prices = close_prices + np.random.rand(200) * 200
        low_prices = close_prices - np.random.rand(200) * 200
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        volume = np.random.randint(1000, 10000, 200)

        data = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volume,
            },
            index=dates,
        )

        print(f"テストデータ作成完了: {len(data)}行")
        print(f"価格範囲: {data['Close'].min():.2f} - {data['Close'].max():.2f}")

        # シンプルな戦略遺伝子を作成
        strategy_gene = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[
                Condition(
                    left_operand="close", operator=">", right_operand="SMA"
                ),  # 価格 > SMA（より簡単な条件）
            ],
            exit_conditions=[
                Condition(
                    left_operand="RSI", operator=">", right_operand=70
                )  # RSI > 70 (売られすぎ)
            ],
            risk_management={
                "position_size": 0.01,  # 非常に小さなポジションサイズ
                "stop_loss": 0.02,
                "take_profit": 0.04,
            },
        )

        print("戦略遺伝子作成完了")
        print(f"指標数: {len(strategy_gene.indicators)}")
        print(f"エントリー条件数: {len(strategy_gene.entry_conditions)}")
        print(f"ポジションサイズ: {strategy_gene.risk_management['position_size']}")

        # 戦略ファクトリーで戦略クラスを生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)

        print("戦略クラス生成完了")

        # バックテスト実行
        print("バックテスト実行中...")
        bt = Backtest(
            data,
            strategy_class,
            cash=1000000,  # 100万ドル
            commission=0.001,
            exclusive_orders=True,
            trade_on_close=True,
            hedging=False,
            margin=1.0,
        )

        # 戦略パラメータを設定
        stats = bt.run(strategy_gene=strategy_gene)

        print("バックテスト完了!")
        print(f"statsの型: {type(stats)}")

        # 基本統計情報を表示
        print(f"\n基本統計情報:")
        print(f"  Return [%]: {stats.get('Return [%]', 'N/A')}")
        print(f"  # Trades: {stats.get('# Trades', 'N/A')}")
        print(f"  Win Rate [%]: {stats.get('Win Rate [%]', 'N/A')}")
        print(f"  Profit Factor: {stats.get('Profit Factor', 'N/A')}")
        print(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
        print(f"  Max Drawdown [%]: {stats.get('Max. Drawdown [%]', 'N/A')}")

        # _trades属性を確認
        print(f"\n_trades属性確認:")
        if hasattr(stats, "_trades"):
            trades_df = getattr(stats, "_trades")
            print(f"  _tradesの型: {type(trades_df)}")
            if trades_df is not None:
                print(f"  _tradesの長さ: {len(trades_df)}")
                if len(trades_df) > 0:
                    print(f"  最初の取引:")
                    for col in trades_df.columns:
                        print(f"    {col}: {trades_df.iloc[0][col]}")

        # コンバーターでテスト
        print(f"\nコンバーターテスト:")
        converter = BacktestResultConverter()

        config_json = {"commission_rate": 0.001}
        result = converter.convert_backtest_results(
            stats=stats,
            strategy_name="Simple_AutoStrategy_Test",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=1000000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 9),
            config_json=config_json,
        )

        print(f"変換結果:")
        metrics = result.get("performance_metrics", {})
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        trade_history = result.get("trade_history", [])
        print(f"\n取引履歴: {len(trade_history)}件")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_autostrategy_simple()
