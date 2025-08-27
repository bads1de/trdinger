#!/usr/bin/env python3
"""
実際にUIで作成された戦略をデバッグするスクリプト
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


def debug_real_strategy():
    """実際のUIで作成された戦略をデバッグ"""
    print("=" * 80)
    print("実際のUIで作成された戦略のデバッグ")
    print("=" * 80)

    try:
        from database.connection import SessionLocal
        from database.repositories.generated_strategy_repository import (
            GeneratedStrategyRepository,
        )
        from app.services.auto_strategy.generators.strategy_factory import (
            StrategyFactory,
        )
        from app.services.backtest.backtest_service import BacktestService
        from app.services.backtest.backtest_data_service import BacktestDataService
        from database.repositories.ohlcv_repository import OHLCVRepository
        from database.repositories.open_interest_repository import (
            OpenInterestRepository,
        )
        from database.repositories.funding_rate_repository import FundingRateRepository

        # データベースから最新の戦略を取得
        with SessionLocal() as db:
            strategy_repo = GeneratedStrategyRepository(db)
            strategies = strategy_repo.get_filtered_data(
                limit=10, order_by_column="created_at", order_asc=False
            )

            if not strategies:
                print("❌ データベースに戦略が見つかりません")
                return

            # 最新の戦略を取得
            latest_strategy = strategies[-1]
            print(f"📋 最新戦略: {latest_strategy.name} (ID: {latest_strategy.id})")
            print(f"   作成日時: {latest_strategy.created_at}")
            print(
                f"   指標数: {len(latest_strategy.indicators) if latest_strategy.indicators else 0}"
            )
            print(
                f"   エントリー条件数: {len(latest_strategy.entry_conditions) if latest_strategy.entry_conditions else 0}"
            )
            print(
                f"   イグジット条件数: {len(latest_strategy.exit_conditions) if latest_strategy.exit_conditions else 0}"
            )

            # 戦略の詳細を表示
            print("\n📊 戦略詳細:")
            if latest_strategy.indicators:
                print("  指標:")
                for i, indicator in enumerate(latest_strategy.indicators):
                    print(f"    {i+1}. {indicator.type} - enabled: {indicator.enabled}")
                    if indicator.parameters:
                        print(f"       パラメータ: {indicator.parameters}")

            if latest_strategy.entry_conditions:
                print("  エントリー条件:")
                for i, condition in enumerate(latest_strategy.entry_conditions):
                    print(
                        f"    {i+1}. {condition.left_operand} {condition.operator} {condition.right_operand}"
                    )

            if latest_strategy.exit_conditions:
                print("  イグジット条件:")
                for i, condition in enumerate(latest_strategy.exit_conditions):
                    print(
                        f"    {i+1}. {condition.left_operand} {condition.operator} {condition.right_operand}"
                    )

            # 戦略をバックテストで実行
            print(f"\n🚀 戦略 '{latest_strategy.name}' のバックテスト実行...")

            # バックテストサービスの初期化
            ohlcv_repo = OHLCVRepository(db)
            oi_repo = OpenInterestRepository(db)
            fr_repo = FundingRateRepository(db)

            data_service = BacktestDataService(
                ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
            )
            backtest_service = BacktestService(data_service)

            # StrategyFactoryで戦略クラスを生成
            strategy_factory = StrategyFactory()

            # 戦略遺伝子を作成
            from app.services.auto_strategy.models.gene_strategy import StrategyGene

            strategy_gene = StrategyGene.from_db_model(latest_strategy)

            print(f"📋 戦略遺伝子作成完了: {strategy_gene.id}")
            print(f"   指標数: {len(strategy_gene.indicators)}")
            print(f"   エントリー条件数: {len(strategy_gene.entry_conditions)}")
            print(f"   イグジット条件数: {len(strategy_gene.exit_conditions)}")

            # 戦略クラスを生成
            strategy_class = strategy_factory.create_strategy_class(strategy_gene)
            print(f"✅ 戦略クラス生成完了: {strategy_class.__name__}")

            # バックテスト設定
            config = {
                "strategy_name": latest_strategy.name,
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": datetime.now() - timedelta(days=7),
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

            if metrics.get("total_trades", 0) == 0:
                print("\n⚠️  取引が発生していません。原因を調査します...")

                # 戦略の条件を詳しく調査
                print("\n🔍 条件詳細調査:")
                for i, condition in enumerate(strategy_gene.entry_conditions):
                    print(
                        f"   エントリー条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand}"
                    )

                for i, condition in enumerate(strategy_gene.exit_conditions):
                    print(
                        f"   イグジット条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand}"
                    )

                # 指標の詳細
                print("\n📈 指標詳細:")
                for i, indicator in enumerate(strategy_gene.indicators):
                    print(
                        f"   指標{i+1}: {indicator.type} (enabled: {indicator.enabled})"
                    )
                    if indicator.parameters:
                        print(f"      パラメータ: {indicator.parameters}")
            else:
                print(f"\n✅ 取引が正常に発生しました！")
                if trade_history:
                    print("最初の取引:")
                    first_trade = trade_history[0]
                    for key, value in first_trade.items():
                        print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_real_strategy()
