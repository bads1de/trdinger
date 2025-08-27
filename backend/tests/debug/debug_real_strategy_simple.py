#!/usr/bin/env python3
"""
実際にUIで作成された戦略をデバッグするスクリプト（簡易版）
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
            latest_strategy = strategies[0]
            gene_data = latest_strategy.gene_data

            print(f"📋 最新戦略: ID {latest_strategy.id}")
            print(f"   作成日時: {latest_strategy.created_at}")
            print(f"   実験ID: {latest_strategy.experiment_id}")
            print(f"   世代: {latest_strategy.generation}")
            print(f"   フィットネス: {latest_strategy.fitness_score}")

            # 戦略の詳細を表示
            print(f"\n📊 戦略詳細 (gene_data):")
            print(
                f"   gene_data keys: {list(gene_data.keys()) if gene_data else 'None'}"
            )

            if gene_data and "indicators" in gene_data:
                indicators = gene_data["indicators"]
                print(f"  指標数: {len(indicators)}")
                for i, indicator in enumerate(indicators):
                    print(
                        f"    {i+1}. {indicator.get('type', 'Unknown')} - enabled: {indicator.get('enabled', False)}"
                    )
                    if indicator.get("parameters"):
                        print(f"       パラメータ: {indicator['parameters']}")

            if gene_data and "entry_conditions" in gene_data:
                entry_conditions = gene_data["entry_conditions"]
                print(f"  エントリー条件数: {len(entry_conditions)}")
                for i, condition in enumerate(entry_conditions):
                    print(
                        f"    {i+1}. {condition.get('left_operand')} {condition.get('operator')} {condition.get('right_operand')}"
                    )

            if gene_data and "exit_conditions" in gene_data:
                exit_conditions = gene_data["exit_conditions"]
                print(f"  イグジット条件数: {len(exit_conditions)}")
                for i, condition in enumerate(exit_conditions):
                    print(
                        f"    {i+1}. {condition.get('left_operand')} {condition.get('operator')} {condition.get('right_operand')}"
                    )

            # 戦略遺伝子を作成してテスト
            print(f"\n🧬 戦略遺伝子の作成テスト...")

            from app.services.auto_strategy.models.gene_strategy import StrategyGene

            # gene_dataから直接StrategyGeneを作成
            strategy_gene = StrategyGene(
                id=gene_data.get("id", f"strategy_{latest_strategy.id}"),
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
            )

            # 指標を追加
            if gene_data and "indicators" in gene_data:
                from app.services.auto_strategy.models.gene_strategy import (
                    IndicatorGene,
                )

                for ind_data in gene_data["indicators"]:
                    indicator = IndicatorGene(
                        type=ind_data.get("type", ""),
                        parameters=ind_data.get("parameters", {}),
                        enabled=ind_data.get("enabled", True),
                    )
                    strategy_gene.indicators.append(indicator)

            # 条件を追加
            if gene_data and "entry_conditions" in gene_data:
                from app.services.auto_strategy.models.gene_strategy import Condition

                for cond_data in gene_data["entry_conditions"]:
                    condition = Condition(
                        left_operand=cond_data.get("left_operand", ""),
                        operator=cond_data.get("operator", ""),
                        right_operand=cond_data.get("right_operand", 0),
                    )
                    strategy_gene.entry_conditions.append(condition)

            if gene_data and "exit_conditions" in gene_data:
                from app.services.auto_strategy.models.gene_strategy import Condition

                for cond_data in gene_data["exit_conditions"]:
                    condition = Condition(
                        left_operand=cond_data.get("left_operand", ""),
                        operator=cond_data.get("operator", ""),
                        right_operand=cond_data.get("right_operand", 0),
                    )
                    strategy_gene.exit_conditions.append(condition)

            print(f"✅ 戦略遺伝子作成完了: {strategy_gene.id}")
            print(f"   指標数: {len(strategy_gene.indicators)}")
            print(f"   エントリー条件数: {len(strategy_gene.entry_conditions)}")
            print(f"   イグジット条件数: {len(strategy_gene.exit_conditions)}")

            # 簡単なバックテストを実行
            print(f"\n🚀 簡易バックテスト実行...")

            from app.services.auto_strategy.generators.strategy_factory import (
                StrategyFactory,
            )

            try:
                strategy_factory = StrategyFactory()
                strategy_class = strategy_factory.create_strategy_class(strategy_gene)
                print(f"✅ 戦略クラス生成成功: {strategy_class.__name__}")

                # 実際のバックテストは省略し、戦略クラスが正常に作成されることを確認
                print("✅ 戦略は正常に作成可能です")

            except Exception as e:
                print(f"❌ 戦略クラス生成エラー: {e}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_real_strategy()
