#!/usr/bin/env python3
"""
ショート条件が正しく動作するかテストするスクリプト
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


def test_short_conditions():
    """ショート条件のテスト"""
    print("=" * 80)
    print("ショート条件のテスト")
    print("=" * 80)

    try:
        from app.services.auto_strategy.models.gene_strategy import (
            StrategyGene,
            IndicatorGene,
            Condition,
        )
        from app.services.auto_strategy.models.gene_tpsl import TPSLGene
        from app.services.auto_strategy.models.gene_position_sizing import (
            PositionSizingGene,
        )

        # テスト用の戦略遺伝子を作成（ショート条件を含む）
        print("🧬 テスト用戦略遺伝子を作成...")

        # 指標を作成
        indicators = [
            IndicatorGene(
                type="AROONOSC",
                parameters={"period": 14},
                enabled=True,
            ),
            IndicatorGene(
                type="SMA",
                parameters={"period": 20},
                enabled=True,
            ),
        ]

        # ロング条件: AROONOSC > 0 AND close > SMA
        long_entry_conditions = [
            Condition(
                left_operand="AROONOSC",
                operator=">",
                right_operand=0.0,
            ),
            Condition(
                left_operand="close",
                operator=">",
                right_operand="SMA",
            ),
        ]

        # ショート条件: AROONOSC < 0 AND close < SMA
        short_entry_conditions = [
            Condition(
                left_operand="AROONOSC",
                operator="<",
                right_operand=0.0,
            ),
            Condition(
                left_operand="close",
                operator="<",
                right_operand="SMA",
            ),
        ]

        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method="fixed",
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            enabled=True,
        )

        # ポジションサイジング遺伝子を作成
        position_sizing_gene = PositionSizingGene(
            method="fixed",
            enabled=True,
        )

        # 戦略遺伝子を作成
        strategy_gene = StrategyGene(
            id="test_short_strategy",
            indicators=indicators,
            entry_conditions=[],  # 空のまま（ロング・ショート分離のため）
            exit_conditions=[],  # 空のまま（TP/SLで管理）
            long_entry_conditions=long_entry_conditions,
            short_entry_conditions=short_entry_conditions,
            tpsl_gene=tpsl_gene,
            position_sizing_gene=position_sizing_gene,
            risk_management={},
        )

        print(f"✅ 戦略遺伝子作成完了:")
        print(f"   ID: {strategy_gene.id}")
        print(f"   指標数: {len(strategy_gene.indicators)}")
        print(f"   ロングエントリー条件数: {len(strategy_gene.long_entry_conditions)}")
        print(
            f"   ショートエントリー条件数: {len(strategy_gene.short_entry_conditions)}"
        )
        print(
            f"   TP/SL遺伝子: {strategy_gene.tpsl_gene.enabled if strategy_gene.tpsl_gene else False}"
        )

        # 戦略クラス作成テスト
        print(f"\n🚀 戦略クラス作成テスト...")

        from app.services.auto_strategy.generators.strategy_factory import (
            StrategyFactory,
        )

        strategy_factory = StrategyFactory()
        strategy_class = strategy_factory.create_strategy_class(strategy_gene)
        print(f"✅ 戦略クラス生成成功: {strategy_class.__name__}")

        # バックテストを実行
        print(f"\n📊 バックテスト実行...")

        from database.connection import SessionLocal
        from app.services.backtest.backtest_service import BacktestService
        from app.services.backtest.backtest_data_service import (
            BacktestDataService,
        )
        from database.repositories.ohlcv_repository import OHLCVRepository
        from database.repositories.open_interest_repository import (
            OpenInterestRepository,
        )
        from database.repositories.funding_rate_repository import (
            FundingRateRepository,
        )

        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)
            oi_repo = OpenInterestRepository(db)
            fr_repo = FundingRateRepository(db)

            data_service = BacktestDataService(
                ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
            )
            backtest_service = BacktestService(data_service)

            # バックテスト設定（より長い期間でテスト）
            config = {
                "strategy_name": "Test_Short_Strategy",
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": datetime.now() - timedelta(days=14),  # 2週間
                "end_date": datetime.now() - timedelta(days=1),
                "initial_capital": 10000000.0,  # 1000万円
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
                        "entry_conditions": [],
                        "exit_conditions": [],
                        "long_entry_conditions": [
                            {
                                "left_operand": cond.left_operand,
                                "operator": cond.operator,
                                "right_operand": cond.right_operand,
                            }
                            for cond in strategy_gene.long_entry_conditions
                        ],
                        "short_entry_conditions": [
                            {
                                "left_operand": cond.left_operand,
                                "operator": cond.operator,
                                "right_operand": cond.right_operand,
                            }
                            for cond in strategy_gene.short_entry_conditions
                        ],
                    }
                },
            }

            print("⏳ バックテスト実行中...")
            result = backtest_service.run_backtest(config)

            # 結果の表示
            metrics = result.get("performance_metrics", {})
            trade_history = result.get("trade_history", [])

            print(f"\n📊 バックテスト結果:")
            print(f"   総取引数: {metrics.get('total_trades', 0)}")
            print(f"   最終資産: {metrics.get('final_equity', 0):,.0f}円")
            print(f"   利益率: {metrics.get('profit_factor', 0):.4f}")
            print(f"   勝率: {metrics.get('win_rate', 0):.2f}%")
            print(f"   取引履歴: {len(trade_history)}件")

            # 取引の詳細を分析
            if len(trade_history) > 0:
                long_trades = [t for t in trade_history if t.get("size", 0) > 0]
                short_trades = [t for t in trade_history if t.get("size", 0) < 0]

                print(f"\n📈 取引分析:")
                print(f"   ロング取引数: {len(long_trades)}")
                print(f"   ショート取引数: {len(short_trades)}")

                if len(short_trades) > 0:
                    print(f"🎉 ショート取引が発生しました！")
                    for i, trade in enumerate(short_trades[:3]):  # 最初の3つを表示
                        print(
                            f"   ショート取引 {i+1}: サイズ={trade.get('size', 0)}, エントリー価格={trade.get('entry_price', 0)}"
                        )
                else:
                    print(f"⚠️  ショート取引が発生しませんでした")

                return_pct = (
                    (metrics.get("final_equity", 10000000) / 10000000) - 1
                ) * 100
                print(f"   リターン: {return_pct:.2f}%")
            else:
                print(f"⚠️  取引が発生しませんでした")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_short_conditions()
