#!/usr/bin/env python3
"""
統合されたショート条件テストスクリプト
シンプルと複雑なショート条件の両方をテスト
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


def test_simple_short_conditions():
    """シンプルで発生しやすいショート条件のテスト"""
    print("=" * 80)
    print("シンプルで発生しやすいショート条件のテスト")
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

        # テスト用の戦略遺伝子を作成（シンプルなショート条件）
        print("🧬 テスト用戦略遺伝子を作成...")

        # 指標を作成
        indicators = [
            IndicatorGene(
                type="SMA",
                parameters={"period": 20},
                enabled=True,
            )
        ]

        # ロング条件: close > SMA (上昇トレンド)
        long_entry_conditions = [
            Condition(
                left_operand="close",
                operator=">",
                right_operand="SMA",
            )
        ]

        # ショート条件: close < SMA (下降トレンド)
        short_entry_conditions = [
            Condition(
                left_operand="close",
                operator="<",
                right_operand="SMA",
            )
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
            id="test_simple_short_strategy",
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

        # シンプルなテストなので、バックテストをスキップ
        print("\n📊 バックテストはオプション（実行をスキップ）")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_complex_short_conditions():
    """複雑なショート条件のテスト"""
    print("=" * 80)
    print("複雑なショート条件のテスト")
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
            id="test_complex_short_strategy",
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

        print("\n📊 詳細なバックテストはオプション（実行をスキップ）")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ショート条件テスト開始")
    test_simple_short_conditions()
    print("\n" + "="*50 + "\n")
    test_complex_short_conditions()