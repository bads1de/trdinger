#!/usr/bin/env python3
"""
実際の戦略でTP/SL遺伝子を含めてテストするスクリプト
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


def test_real_strategy_with_tpsl():
    """実際の戦略でTP/SL遺伝子を含めてテスト"""
    print("=" * 80)
    print("実際の戦略でTP/SL遺伝子を含めてテスト")
    print("=" * 80)

    try:
        from database.connection import SessionLocal
        from database.repositories.generated_strategy_repository import (
            GeneratedStrategyRepository,
        )
        from app.services.auto_strategy.models.gene_strategy import (
            StrategyGene,
            IndicatorGene,
            Condition,
        )
        from app.services.auto_strategy.models.gene_tpsl import TPSLGene
        from app.services.auto_strategy.models.gene_position_sizing import (
            PositionSizingGene,
        )

        # データベースから最新の戦略を取得
        with SessionLocal() as db:
            strategy_repo = GeneratedStrategyRepository(db)
            strategies = strategy_repo.get_filtered_data(
                limit=1, order_by_column="created_at", order_asc=False
            )

            if not strategies:
                print("❌ データベースに戦略が見つかりません")
                return

            # 最新の戦略を取得
            latest_strategy = strategies[0]
            gene_data = latest_strategy.gene_data

            print(f"📋 最新戦略: ID {latest_strategy.id}")
            print(
                f"   TP/SL遺伝子: {gene_data.get('tpsl_gene', {}).get('enabled', False)}"
            )
            print(f"   イグジット条件数: {len(gene_data.get('exit_conditions', []))}")

            # 戦略遺伝子を完全に再構築
            print(f"\n🧬 戦略遺伝子の完全再構築...")

            # 指標を作成
            indicators = []
            for ind_data in gene_data.get("indicators", []):
                indicator = IndicatorGene(
                    type=ind_data.get("type", ""),
                    parameters=ind_data.get("parameters", {}),
                    enabled=ind_data.get("enabled", True),
                )
                indicators.append(indicator)

            # エントリー条件を作成
            entry_conditions = []
            for cond_data in gene_data.get("entry_conditions", []):
                condition = Condition(
                    left_operand=cond_data.get("left_operand", ""),
                    operator=cond_data.get("operator", ""),
                    right_operand=cond_data.get("right_operand", 0),
                )
                entry_conditions.append(condition)

            # ロング・ショートエントリー条件を作成
            long_entry_conditions = []
            for cond_data in gene_data.get("long_entry_conditions", []):
                condition = Condition(
                    left_operand=cond_data.get("left_operand", ""),
                    operator=cond_data.get("operator", ""),
                    right_operand=cond_data.get("right_operand", 0),
                )
                long_entry_conditions.append(condition)

            short_entry_conditions = []
            for cond_data in gene_data.get("short_entry_conditions", []):
                condition = Condition(
                    left_operand=cond_data.get("left_operand", ""),
                    operator=cond_data.get("operator", ""),
                    right_operand=cond_data.get("right_operand", 0),
                )
                short_entry_conditions.append(condition)

            # TP/SL遺伝子を作成
            tpsl_gene = None
            if gene_data.get("tpsl_gene"):
                tpsl_data = gene_data["tpsl_gene"]
                tpsl_gene = TPSLGene(
                    method=tpsl_data.get("method", "fixed"),
                    stop_loss_pct=tpsl_data.get("stop_loss_pct", 0.02),
                    take_profit_pct=tpsl_data.get("take_profit_pct", 0.04),
                    enabled=tpsl_data.get("enabled", False),
                )

            # ポジションサイジング遺伝子を作成
            position_sizing_gene = None
            if gene_data.get("position_sizing_gene"):
                ps_data = gene_data["position_sizing_gene"]
                position_sizing_gene = PositionSizingGene(
                    method=ps_data.get("method", "fixed"),
                    enabled=ps_data.get("enabled", False),
                )

            # リスク管理を作成（辞書形式で保持）
            risk_management = gene_data.get("risk_management", {})

            # 戦略遺伝子を作成
            strategy_gene = StrategyGene(
                id=gene_data.get("id", f"strategy_{latest_strategy.id}"),
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=[],  # 空のまま（TP/SLで管理）
                long_entry_conditions=long_entry_conditions,
                short_entry_conditions=short_entry_conditions,
                tpsl_gene=tpsl_gene,
                position_sizing_gene=position_sizing_gene,
                risk_management=risk_management,
            )

            print(f"✅ 戦略遺伝子作成完了:")
            print(f"   ID: {strategy_gene.id}")
            print(f"   指標数: {len(strategy_gene.indicators)}")
            print(f"   エントリー条件数: {len(strategy_gene.entry_conditions)}")
            print(
                f"   ロングエントリー条件数: {len(strategy_gene.long_entry_conditions)}"
            )
            print(
                f"   ショートエントリー条件数: {len(strategy_gene.short_entry_conditions)}"
            )
            print(f"   イグジット条件数: {len(strategy_gene.exit_conditions)}")
            print(
                f"   TP/SL遺伝子: {strategy_gene.tpsl_gene.enabled if strategy_gene.tpsl_gene else False}"
            )
            print(
                f"   ポジションサイジング遺伝子: {strategy_gene.position_sizing_gene.enabled if strategy_gene.position_sizing_gene else False}"
            )

            # 戦略クラス作成テスト
            print(f"\n🚀 戦略クラス作成テスト...")

            from app.services.auto_strategy.generators.strategy_factory import (
                StrategyFactory,
            )

            try:
                strategy_factory = StrategyFactory()
                strategy_class = strategy_factory.create_strategy_class(strategy_gene)
                print(f"✅ 戦略クラス生成成功: {strategy_class.__name__}")

                # 簡単なバックテストを実行
                print(f"\n📊 簡単なバックテスト実行...")

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

                ohlcv_repo = OHLCVRepository(db)
                oi_repo = OpenInterestRepository(db)
                fr_repo = FundingRateRepository(db)

                data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                backtest_service = BacktestService(data_service)

                # バックテスト設定
                config = {
                    "strategy_name": f"Real_Strategy_{latest_strategy.id}",
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
                            "exit_conditions": [],  # 空のまま
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

                if metrics.get("total_trades", 0) > 0:
                    print(f"🎉 実際の戦略で取引が発生しました！")
                    return_pct = (
                        (metrics.get("final_equity", 10000000) / 10000000) - 1
                    ) * 100
                    print(f"   リターン: {return_pct:.2f}%")
                else:
                    print(f"⚠️  実際の戦略でも取引が発生しませんでした")

            except Exception as e:
                print(f"❌ 戦略クラス生成エラー: {e}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_real_strategy_with_tpsl()
