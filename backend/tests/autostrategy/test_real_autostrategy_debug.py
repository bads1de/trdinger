#!/usr/bin/env python3
"""
実際のオートストラテジーでバックテストを実行してデバッグ
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


def test_real_autostrategy_debug():
    """実際のオートストラテジーでバックテストを実行してデバッグ"""
    print("=" * 60)
    print("Real AutoStrategy Backtest Debug")
    print("=" * 60)

    try:
        from app.services.auto_strategy.generators.strategy_factory import (
            StrategyFactory,
        )
        from app.services.auto_strategy.models.gene_strategy import (
            StrategyGene,
            IndicatorGene,
            Condition,
        )
        from app.services.backtest.backtest_service import BacktestService
        from app.services.backtest.backtest_data_service import BacktestDataService
        from database.connection import SessionLocal
        from database.repositories.ohlcv_repository import OHLCVRepository
        from database.repositories.open_interest_repository import (
            OpenInterestRepository,
        )
        from database.repositories.funding_rate_repository import FundingRateRepository

        # データベースセッションとリポジトリを初期化
        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)
            oi_repo = OpenInterestRepository(db)
            fr_repo = FundingRateRepository(db)

            data_service = BacktestDataService(
                ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
            )
            backtest_service = BacktestService(data_service)

            # シンプルな戦略遺伝子を作成
            print("シンプルな戦略遺伝子を作成中...")

            # RSIとMACDを使用したシンプルな戦略
            indicators = [
                IndicatorGene(
                    type="RSI",
                    parameters={"period": 14, "overbought": 70, "oversold": 30},
                    enabled=True,
                ),
                IndicatorGene(
                    type="MACD",
                    parameters={
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                    },
                    enabled=True,
                ),
            ]

            # シンプルな条件を作成
            entry_conditions = [
                Condition(left_operand="RSI", operator="<", right_operand=30.0),
                Condition(
                    left_operand="MACD", operator=">", right_operand="MACD_SIGNAL"
                ),
            ]

            exit_conditions = [
                Condition(left_operand="RSI", operator=">", right_operand=70.0),
                Condition(
                    left_operand="MACD", operator="<", right_operand="MACD_SIGNAL"
                ),
            ]

            strategy_gene = StrategyGene(
                id="test_strategy_001",
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
            )

            print(f"生成された戦略: 指標数={len(strategy_gene.indicators)}")
            for i, ind in enumerate(strategy_gene.indicators):
                print(f"  指標{i+1}: {ind.type}, enabled={ind.enabled}")

            # StrategyFactoryで戦略クラスを生成
            strategy_factory = StrategyFactory()
            strategy_class = strategy_factory.create_strategy_class(strategy_gene)

            # バックテスト設定
            config = {
                "strategy_name": "Debug_AutoStrategy",
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": datetime.now() - timedelta(days=30),
                "end_date": datetime.now() - timedelta(days=1),
                "initial_capital": 100000.0,
                "commission_rate": 0.001,
                "strategy_class": strategy_class,
                "strategy_config": {
                    "strategy_gene": {
                        "id": strategy_gene.id,
                        "indicators": [
                            {
                                "type": ind.type,
                                "parameters": ind.parameters,
                                "enabled": ind.enabled,
                            }
                            for ind in strategy_gene.indicators
                        ],
                        "entry_conditions": [
                            {
                                "left_operand": cond.left_operand,
                                "operator": cond.operator,
                                "right_operand": cond.right_operand,
                            }
                            for cond in strategy_gene.entry_conditions
                        ],
                        "exit_conditions": [
                            {
                                "left_operand": cond.left_operand,
                                "operator": cond.operator,
                                "right_operand": cond.right_operand,
                            }
                            for cond in strategy_gene.exit_conditions
                        ],
                    }
                },
            }

            print("バックテスト実行中...")
            try:
                result = backtest_service.run_backtest(config)

                print("バックテスト完了!")
                print(f"結果の型: {type(result)}")

                # パフォーマンス指標を確認
                metrics = result.get("performance_metrics", {})
                print(f"\nパフォーマンス指標:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")

                # 取引履歴を確認
                trade_history = result.get("trade_history", [])
                print(f"\n取引履歴: {len(trade_history)}件")
                if trade_history:
                    print("最初の取引:")
                    for key, value in trade_history[0].items():
                        print(f"  {key}: {value}")

                # バックテストオブジェクトの詳細情報を確認
                if hasattr(result, "backtest_obj") and result["backtest_obj"]:
                    bt = result["backtest_obj"]
                    print(f"\nバックテストオブジェクト詳細:")
                    print(
                        f"  _trades: {len(bt._trades) if hasattr(bt, '_trades') else 'N/A'}"
                    )
                    print(
                        f"  _results: {type(bt._results) if hasattr(bt, '_results') else 'N/A'}"
                    )

                    if hasattr(bt, "_trades") and len(bt._trades) > 0:
                        print(f"  取引データの最初の行:")
                        print(f"    {bt._trades.iloc[0].to_dict()}")

            except Exception as e:
                print(f"バックテスト実行エラー: {e}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_real_autostrategy_debug()
