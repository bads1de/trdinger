#!/usr/bin/env python3
"""
取引生成のデバッグスクリプト

MACD条件で取引が発生しない理由を調査します。
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
from database.connection import SessionLocal


def debug_trade_generation():
    """取引生成をデバッグして問題を特定"""

    print("🔍 取引生成デバッグ開始")
    print("=" * 50)

    # MACD戦略を手動で作成
    strategy_gene = StrategyGene(
        id="debug_macd",
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
    print(f"  ストップロス: 3%")
    print(f"  テイクプロフィット: 15%")
    print()

    # バックテスト設定
    backtest_config = {
        "strategy_name": "DEBUG_MACD_TRADE_TEST",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2024-12-01",
        "end_date": "2024-12-31",
        "initial_capital": 100000.0,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "GENERATED_TEST",
            "parameters": {"strategy_gene": strategy_gene.to_dict()},
        },
    }

    print("🧪 バックテスト実行:")
    print(f"  期間: {backtest_config['start_date']} - {backtest_config['end_date']}")
    print(f"  シンボル: {backtest_config['symbol']}")
    print(f"  時間軸: {backtest_config['timeframe']}")
    print()

    try:
        # データベース接続
        db = SessionLocal()

        # データサービスとバックテストサービス
        from app.core.services.backtest_data_service import BacktestDataService
        from database.repositories.ohlcv_repository import OHLCVRepository
        from database.repositories.open_interest_repository import (
            OpenInterestRepository,
        )
        from database.repositories.funding_rate_repository import FundingRateRepository

        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        data_service = BacktestDataService(ohlcv_repo, oi_repo, fr_repo)
        backtest_service = BacktestService(data_service)

        # バックテスト実行
        print("⚡ バックテスト実行中...")
        result = backtest_service.run_backtest(backtest_config)

        print("📈 バックテスト結果:")
        print(f"  総取引数: {result['performance_metrics']['total_trades']}")
        print(f"  総リターン: {result['performance_metrics']['total_return']:.4f}")
        print(f"  最終資産: {result['performance_metrics']['equity_final']:,.0f}")
        print(f"  勝率: {result['performance_metrics']['win_rate']}")
        print(
            f"  最大ドローダウン: {result['performance_metrics']['max_drawdown']:.4f}"
        )
        print()

        # 取引履歴の詳細分析
        trade_history = result.get("trade_history", [])
        print(f"📋 取引履歴詳細 ({len(trade_history)}件):")

        if trade_history:
            for i, trade in enumerate(trade_history[:5], 1):  # 最初の5件を表示
                print(f"  取引 {i}:")
                print(
                    f"    エントリー: {trade.get('entry_time')} @ {trade.get('entry_price')}"
                )
                print(
                    f"    エグジット: {trade.get('exit_time')} @ {trade.get('exit_price')}"
                )
                print(f"    P&L: {trade.get('pnl', 0):.4f}")
                print(f"    タイプ: {trade.get('trade_type', 'N/A')}")
        else:
            print("  ❌ 取引履歴なし")

        print()

        # MACD値の分析
        print("🔍 MACD値の詳細分析:")

        # データを取得してMACDを計算（既存のdata_serviceを使用）

        # OHLCVリポジトリから直接データを取得
        from datetime import datetime

        ohlcv_data = ohlcv_repo.get_ohlcv_data(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_time=datetime.strptime("2024-12-01", "%Y-%m-%d"),
            end_time=datetime.strptime("2024-12-31", "%Y-%m-%d"),
        )

        print(f"  データポイント数: {len(ohlcv_data)}")

        if len(ohlcv_data) > 0:
            # MACD計算
            from app.core.services.indicators.adapters.momentum_adapter import (
                MomentumAdapter,
            )

            close_prices = pd.Series([float(d.close) for d in ohlcv_data])
            macd_result = MomentumAdapter.macd(close_prices, fast=12, slow=26, signal=9)

            macd_line = macd_result["macd_line"]
            signal_line = macd_result["signal_line"]
            histogram = macd_result["histogram"]

            # 統計情報
            valid_macd = [x for x in macd_line if x is not None and not pd.isna(x)]
            if valid_macd:
                print(f"  MACD統計:")
                print(f"    有効値数: {len(valid_macd)}")
                print(f"    最小値: {min(valid_macd):.6f}")
                print(f"    最大値: {max(valid_macd):.6f}")
                print(f"    平均値: {sum(valid_macd)/len(valid_macd):.6f}")

                # 正の値の割合
                positive_count = sum(1 for x in valid_macd if x > 0)
                positive_ratio = positive_count / len(valid_macd)
                print(
                    f"    正の値: {positive_count}/{len(valid_macd)} ({positive_ratio:.1%})"
                )

                # エントリー条件を満たす期間
                print(f"  エントリー条件 (MACD > 0) を満たす期間: {positive_ratio:.1%}")

                if positive_ratio == 0:
                    print(
                        "  ❌ エントリー条件を満たす期間が0% - これが取引が発生しない理由です"
                    )
                elif positive_ratio < 0.1:
                    print("  ⚠️ エントリー条件を満たす期間が非常に少ない")
                else:
                    print("  ✅ エントリー条件を満たす期間は十分")

                # 最近の値を表示
                print(f"  最近のMACD値 (最後の10個):")
                for i, (macd_val, signal_val) in enumerate(
                    zip(macd_line[-10:], signal_line[-10:]), 1
                ):
                    if macd_val is not None and signal_val is not None:
                        entry_signal = "✅ ENTRY" if macd_val > 0 else "❌"
                        exit_signal = "✅ EXIT" if macd_val < 0 else "❌"
                        print(
                            f"    {i:2d}: MACD={macd_val:8.6f}, Signal={signal_val:8.6f} {entry_signal} {exit_signal}"
                        )
            else:
                print("  ❌ 有効なMACD値が計算されませんでした")

        db.close()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_trade_generation()
