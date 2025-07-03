#!/usr/bin/env python3
"""
現実的な戦略のデバッグスクリプト

より現実的な条件とより長い期間でテストします。
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


def debug_realistic_strategy():
    """現実的な戦略でテスト"""

    print("🔍 現実的な戦略テスト開始")
    print("=" * 50)

    # より現実的なRSI戦略を作成
    strategy_gene = StrategyGene(
        id="debug_realistic_rsi",
        indicators=[IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)],
        entry_conditions=[
            Condition(
                left_operand="RSI", operator="<", right_operand=30.0  # 売られすぎ条件
            )
        ],
        exit_conditions=[
            Condition(
                left_operand="RSI", operator=">", right_operand=70.0  # 買われすぎ条件
            )
        ],
        risk_management={"stop_loss": 0.03, "take_profit": 0.15, "position_size": 0.1},
    )

    print("📊 テスト戦略:")
    print(f"  指標: RSI(14)")
    print(f"  エントリー: RSI < 30 (売られすぎ)")
    print(f"  エグジット: RSI > 70 (買われすぎ)")
    print()

    try:
        # データベース接続
        db = SessionLocal()

        # データサービス
        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        data_service = BacktestDataService(ohlcv_repo, oi_repo, fr_repo)

        # より長い期間でテスト
        print("📊 データ取得中...")
        data = data_service.get_data_for_backtest(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime.strptime("2024-01-01", "%Y-%m-%d"),
            end_date=datetime.strptime("2024-12-31", "%Y-%m-%d"),
        )

        print(f"  データポイント数: {len(data)}")
        print(f"  データ期間: {data.index[0]} - {data.index[-1]}")

        # RSI戦略をテスト
        from backtesting import Strategy, Backtest

        class TestRSIStrategy(Strategy):
            def init(self):
                # RSI指標を初期化（新しいシステムを使用）
                from app.core.services.indicators.momentum import MomentumIndicators

                close_prices = pd.Series(self.data.Close)
                rsi_result = MomentumIndicators.rsi(close_prices.values, period=14)
                rsi_result = pd.Series(rsi_result, index=close_prices.index)

                self.rsi = rsi_result

                print(f"  📊 RSI指標初期化完了: {len(self.rsi)} 値")

                # RSI統計
                valid_rsi = [x for x in self.rsi if not pd.isna(x)]
                if valid_rsi:
                    oversold_count = sum(1 for x in valid_rsi if x < 30)
                    overbought_count = sum(1 for x in valid_rsi if x > 70)

                    print(f"  📊 RSI統計:")
                    print(f"    有効値数: {len(valid_rsi)}")
                    print(f"    最小値: {min(valid_rsi):.2f}")
                    print(f"    最大値: {max(valid_rsi):.2f}")
                    print(f"    平均値: {sum(valid_rsi)/len(valid_rsi):.2f}")
                    print(
                        f"    売られすぎ (RSI < 30): {oversold_count}/{len(valid_rsi)} ({oversold_count/len(valid_rsi)*100:.1f}%)"
                    )
                    print(
                        f"    買われすぎ (RSI > 70): {overbought_count}/{len(valid_rsi)} ({overbought_count/len(valid_rsi)*100:.1f}%)"
                    )

            def next(self):
                # 現在のインデックスを取得
                current_index = len(self.data) - 1

                # RSI条件をチェック
                if current_index < len(self.rsi):
                    current_rsi = self.rsi.iloc[current_index]

                    # エントリー条件: RSI < 30 (売られすぎ)
                    if not pd.isna(current_rsi) and current_rsi < 30:
                        if not self.position:
                            print(
                                f"  🟢 エントリーシグナル: RSI={current_rsi:.2f} < 30"
                            )
                            self.buy()

                    # エグジット条件: RSI > 70 (買われすぎ)
                    elif not pd.isna(current_rsi) and current_rsi > 70:
                        if self.position:
                            print(
                                f"  🔴 エグジットシグナル: RSI={current_rsi:.2f} > 70"
                            )
                            self.sell()

        # バックテストを実行
        print("⚡ RSI戦略バックテスト実行中...")
        bt = Backtest(data, TestRSIStrategy, cash=100000, commission=0.001)
        result = bt.run()

        print("📈 RSI戦略バックテスト結果:")
        print(f"  総取引数: {result['# Trades']}")
        print(f"  総リターン: {result['Return [%]']:.4f}%")
        print(f"  最終資産: {result['Equity Final [$]']:,.0f}")
        print(f"  勝率: {result['Win Rate [%]']:.2f}%")
        print(f"  最大ドローダウン: {result['Max. Drawdown [%]']:.4f}%")

        # 取引履歴を表示
        trades = result._trades
        if len(trades) > 0:
            print(f"📋 取引履歴 ({len(trades)}件):")
            for i, trade in trades.head(10).iterrows():  # 最初の10件を表示
                print(f"  取引 {i+1}:")
                print(
                    f"    エントリー: {trade['EntryTime']} @ {trade['EntryPrice']:.2f}"
                )
                print(f"    エグジット: {trade['ExitTime']} @ {trade['ExitPrice']:.2f}")
                print(f"    P&L: {trade['PnL']:.2f}")
                print(f"    リターン: {trade['ReturnPct']:.2f}%")
        else:
            print("  ❌ 取引履歴なし")

        print()
        print("🧪 GENERATED_TEST戦略との比較テスト...")

        # 同じ戦略をGENERATED_TEST形式でテスト
        backtest_service = BacktestService(data_service)

        backtest_config = {
            "strategy_name": "DEBUG_RSI_GENERATED_TEST",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "GENERATED_TEST",
                "parameters": {"strategy_gene": strategy_gene.to_dict()},
            },
        }

        generated_result = backtest_service.run_backtest(backtest_config)

        print("📈 GENERATED_TEST戦略結果:")
        print(f"  総取引数: {generated_result['performance_metrics']['total_trades']}")
        print(
            f"  総リターン: {generated_result['performance_metrics']['total_return']:.4f}"
        )
        print(
            f"  最終資産: {generated_result['performance_metrics']['equity_final']:,.0f}"
        )
        print(f"  勝率: {generated_result['performance_metrics']['win_rate']}")
        print(
            f"  最大ドローダウン: {generated_result['performance_metrics']['max_drawdown']:.4f}"
        )

        # 比較結果
        print()
        print("🔍 比較結果:")
        manual_trades = result["# Trades"]
        generated_trades = generated_result["performance_metrics"]["total_trades"]

        if manual_trades > 0 and generated_trades == 0:
            print(
                "  ❌ 手動実装では取引が発生するが、GENERATED_TEST戦略では取引が発生しない"
            )
            print("  → GENERATED_TEST戦略の実装に問題がある可能性")
        elif manual_trades == 0 and generated_trades == 0:
            print("  ⚠️ 両方とも取引が発生しない - 条件または期間の問題")
        elif manual_trades > 0 and generated_trades > 0:
            print("  ✅ 両方で取引が発生 - GENERATED_TEST戦略は正常に動作")
        else:
            print("  🤔 予期しない結果")

        db.close()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_realistic_strategy()
