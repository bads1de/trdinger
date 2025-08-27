#!/usr/bin/env python3
"""
gene_dataの詳細を確認するスクリプト
"""

import sys
import os
import json

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def debug_gene_data():
    """gene_dataの詳細を確認"""
    print("=" * 80)
    print("gene_dataの詳細確認")
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
                limit=5, order_by_column="created_at", order_asc=False
            )

            if not strategies:
                print("❌ データベースに戦略が見つかりません")
                return

            for i, strategy in enumerate(strategies):
                print(f"\n📋 戦略 {i+1}: ID {strategy.id}")
                print(f"   作成日時: {strategy.created_at}")
                print(f"   実験ID: {strategy.experiment_id}")
                print(f"   世代: {strategy.generation}")

                gene_data = strategy.gene_data
                print(f"\n📊 gene_data構造:")
                print(json.dumps(gene_data, indent=2, ensure_ascii=False))

                # 各セクションの詳細確認
                if gene_data:
                    print(f"\n🔍 詳細分析:")

                    # indicators
                    if "indicators" in gene_data:
                        indicators = gene_data["indicators"]
                        print(f"   indicators: {len(indicators)}個")
                        for j, ind in enumerate(indicators):
                            print(f"     {j+1}. {ind}")

                    # entry_conditions
                    if "entry_conditions" in gene_data:
                        entry_conditions = gene_data["entry_conditions"]
                        print(f"   entry_conditions: {len(entry_conditions)}個")
                        for j, cond in enumerate(entry_conditions):
                            print(f"     {j+1}. {cond}")

                    # exit_conditions
                    if "exit_conditions" in gene_data:
                        exit_conditions = gene_data["exit_conditions"]
                        print(f"   exit_conditions: {len(exit_conditions)}個")
                        for j, cond in enumerate(exit_conditions):
                            print(f"     {j+1}. {cond}")
                    else:
                        print(f"   ❌ exit_conditions: 存在しません")

                    # long_entry_conditions
                    if "long_entry_conditions" in gene_data:
                        long_entry_conditions = gene_data["long_entry_conditions"]
                        print(
                            f"   long_entry_conditions: {len(long_entry_conditions)}個"
                        )
                        for j, cond in enumerate(long_entry_conditions):
                            print(f"     {j+1}. {cond}")

                    # short_entry_conditions
                    if "short_entry_conditions" in gene_data:
                        short_entry_conditions = gene_data["short_entry_conditions"]
                        print(
                            f"   short_entry_conditions: {len(short_entry_conditions)}個"
                        )
                        for j, cond in enumerate(short_entry_conditions):
                            print(f"     {j+1}. {cond}")

                print("-" * 60)

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_gene_data()
