#!/usr/bin/env python3
"""
戦略実行のデバッグスクリプト

GENERATED_TEST戦略の実行時の条件評価を詳しく調査します。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from app.core.services.backtest_service import BacktestService
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.connection import SessionLocal


def debug_strategy_execution():
    """戦略実行をデバッグして問題を特定"""

    print("🔍 戦略実行デバッグ開始")
    print("=" * 50)

    # MACD戦略を手動で作成
    strategy_gene = StrategyGene(
        id="debug_macd_execution",
        indicators=[
            IndicatorGene(
                type="MACD",
                parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
                enabled=True,
            )
        ],
        entry_conditions=[
            Condition(left_operand="MACD", operator=">", right_operand=0.0)
        ],
        exit_conditions=[
            Condition(left_operand="MACD", operator="<", right_operand=0.0)
        ],
        risk_management={"stop_loss": 0.03, "take_profit": 0.15, "position_size": 0.1},
    )

    print("📊 テスト戦略:")
    print(f"  指標: MACD(12,26,9)")
    print(f"  エントリー: MACD > 0")
    print(f"  エグジット: MACD < 0")
    print()

    try:
        # データベース接続
        db = SessionLocal()

        # データサービスとバックテストサービス
        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        data_service = BacktestDataService(ohlcv_repo, oi_repo, fr_repo)
        backtest_service = BacktestService(data_service)

        # 戦略ファクトリーで戦略クラスを作成
        print("🏭 戦略クラス作成中...")
        from app.core.services.auto_strategy.factories.strategy_factory import (
            StrategyFactory,
        )

        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)

        print(f"  ✅ 戦略クラス作成成功: {strategy_class.__name__}")

        # データを取得
        print("📊 データ取得中...")
        data = data_service.get_data_for_backtest(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime.strptime("2024-12-01", "%Y-%m-%d"),
            end_date=datetime.strptime("2024-12-31", "%Y-%m-%d"),
        )

        print(f"  データポイント数: {len(data)}")
        print(f"  データカラム: {list(data.columns)}")
        print(f"  データ期間: {data.index[0]} - {data.index[-1]}")

        # 戦略インスタンスを作成してテスト
        print("🧪 戦略インスタンステスト...")

        # backtesting.pyのStrategyクラスを継承した戦略を手動でテスト
        from backtesting import Strategy

        # テスト用の簡単な戦略クラスを作成
        class TestStrategy(Strategy):
            def init(self):
                # 指標を初期化
                from app.core.services.indicators.adapters.momentum_adapter import (
                    MomentumAdapter,
                )

                close_prices = pd.Series(self.data.Close)
                macd_result = MomentumAdapter.macd(
                    close_prices, fast=12, slow=26, signal=9
                )

                self.macd_line = macd_result["macd_line"]
                self.signal_line = macd_result["signal_line"]
                self.histogram = macd_result["histogram"]

                print(f"  📊 MACD指標初期化完了:")
                print(f"    MACD Line: {len(self.macd_line)} 値")
                print(f"    Signal Line: {len(self.signal_line)} 値")
                print(f"    Histogram: {len(self.histogram)} 値")

                # 最初の10個の値を表示
                print(f"  📋 最初の10個のMACD値:")
                for i in range(min(10, len(self.macd_line))):
                    macd_val = (
                        self.macd_line.iloc[i]
                        if not pd.isna(self.macd_line.iloc[i])
                        else "NaN"
                    )
                    signal_val = (
                        self.signal_line.iloc[i]
                        if not pd.isna(self.signal_line.iloc[i])
                        else "NaN"
                    )
                    print(f"    {i+1:2d}: MACD={macd_val}, Signal={signal_val}")

                # 最後の10個の値を表示
                print(f"  📋 最後の10個のMACD値:")
                for i in range(max(0, len(self.macd_line) - 10), len(self.macd_line)):
                    macd_val = (
                        self.macd_line.iloc[i]
                        if not pd.isna(self.macd_line.iloc[i])
                        else "NaN"
                    )
                    signal_val = (
                        self.signal_line.iloc[i]
                        if not pd.isna(self.signal_line.iloc[i])
                        else "NaN"
                    )
                    entry_signal = (
                        "✅ ENTRY"
                        if isinstance(macd_val, (int, float)) and macd_val > 0
                        else "❌"
                    )
                    print(
                        f"    {i+1:2d}: MACD={macd_val}, Signal={signal_val} {entry_signal}"
                    )

            def next(self):
                # 現在のインデックスを取得
                current_index = len(self.data) - 1

                # エントリー条件: MACD > 0
                if current_index < len(self.macd_line):
                    current_macd = self.macd_line.iloc[current_index]

                    if not pd.isna(current_macd) and current_macd > 0:
                        if not self.position:
                            print(
                                f"  🟢 エントリーシグナル: MACD={current_macd:.6f} > 0"
                            )
                            self.buy()

                    # エグジット条件: MACD < 0
                    elif not pd.isna(current_macd) and current_macd < 0:
                        if self.position:
                            print(
                                f"  🔴 エグジットシグナル: MACD={current_macd:.6f} < 0"
                            )
                            self.sell()

        # バックテストを実行
        print("⚡ 手動バックテスト実行中...")
        from backtesting import Backtest

        bt = Backtest(data, TestStrategy, cash=100000, commission=0.001)
        result = bt.run()

        print("📈 手動バックテスト結果:")
        print(f"  総取引数: {result['# Trades']}")
        print(f"  総リターン: {result['Return [%]']:.4f}%")
        print(f"  最終資産: {result['Equity Final [$]']:,.0f}")
        print(f"  勝率: {result['Win Rate [%]']:.2f}%")
        print(f"  最大ドローダウン: {result['Max. Drawdown [%]']:.4f}%")

        # 取引履歴を表示
        trades = result._trades
        if len(trades) > 0:
            print(f"📋 取引履歴 ({len(trades)}件):")
            for i, trade in trades.iterrows():
                print(f"  取引 {i+1}:")
                print(
                    f"    エントリー: {trade['EntryTime']} @ {trade['EntryPrice']:.2f}"
                )
                print(f"    エグジット: {trade['ExitTime']} @ {trade['ExitPrice']:.2f}")
                print(f"    P&L: {trade['PnL']:.2f}")
                print(f"    リターン: {trade['ReturnPct']:.2f}%")
        else:
            print("  ❌ 取引履歴なし")

        db.close()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_strategy_execution()
