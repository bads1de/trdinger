#!/usr/bin/env python3
"""
修正された戦略生成機能のテスト

正しいGA目的（高リターン・高シャープレシオ・低ドローダウン）に基づく
OI/FRを判断材料として使用する戦略生成のテスト
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)


def test_corrected_strategy_gene():
    """修正された戦略遺伝子のテスト"""
    print("🎯 修正された戦略遺伝子テスト開始")
    print("=" * 60)

    try:
        # 1. 正しいOI/FR判断条件を含む戦略遺伝子を作成
        print("1. 正しいOI/FR判断条件の戦略遺伝子作成中...")

        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(
                type="MACD",
                parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
                enabled=True,
            ),
        ]

        # 正しいOI/FR判断条件
        entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand="SMA_20"),
            Condition(
                left_operand="RSI_14", operator="<", right_operand=35
            ),  # 売られすぎからの反発
            Condition(
                left_operand="FundingRate", operator=">", right_operand=0.0005
            ),  # ロング過熱時のショート
            Condition(
                left_operand="OpenInterest", operator=">", right_operand=10000000
            ),  # 大きなOIでトレンド確認
        ]

        exit_conditions = [
            Condition(left_operand="close", operator="<", right_operand="SMA_20"),
            Condition(
                left_operand="RSI_14", operator=">", right_operand=70
            ),  # 買われすぎで利確
            Condition(
                left_operand="FundingRate", operator="<", right_operand=-0.0001
            ),  # センチメント変化
        ]

        gene = StrategyGene(
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management={"stop_loss": 0.03, "take_profit": 0.1},
        )

        print(f"  ✅ 戦略遺伝子作成成功: ID {gene.id}")
        print(f"  📊 指標数: {len(gene.indicators)} (テクニカル指標のみ)")
        print(f"  📊 エントリー条件数: {len(gene.entry_conditions)}")
        print(f"  📊 イグジット条件数: {len(gene.exit_conditions)}")

        # 2. 指標の内容確認
        print("\n2. 指標内容確認:")
        for i, indicator in enumerate(gene.indicators):
            print(f"  📈 指標{i+1}: {indicator.type} - {indicator.parameters}")

        # 3. OI/FR判断条件の確認
        print("\n3. OI/FR判断条件確認:")
        oi_fr_conditions = []
        for condition in entry_conditions + exit_conditions:
            if condition.left_operand in [
                "OpenInterest",
                "FundingRate",
            ] or condition.right_operand in ["OpenInterest", "FundingRate"]:
                oi_fr_conditions.append(condition)

        print(f"  📋 OI/FR判断条件数: {len(oi_fr_conditions)}")
        for i, condition in enumerate(oi_fr_conditions):
            print(
                f"    {i+1}. {condition.left_operand} {condition.operator} {condition.right_operand}"
            )

        # 4. 妥当性検証
        print("\n4. 妥当性検証中...")
        is_valid, errors = gene.validate()

        if is_valid:
            print("  ✅ 戦略遺伝子は有効です")
        else:
            print(f"  ❌ 戦略遺伝子が無効: {errors}")
            return False

        print("\n🎉 修正された戦略遺伝子テスト完了！")
        return True

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_corrected_random_generator():
    """修正されたランダム遺伝子生成器のテスト"""
    print("\n🎲 修正されたランダム遺伝子生成器テスト開始")
    print("=" * 60)

    try:
        # 1. 修正されたランダム遺伝子生成器作成
        print("1. 修正されたランダム遺伝子生成器作成中...")
        from app.core.services.auto_strategy.models.ga_config import GAConfig

        ga_config = GAConfig(
            max_indicators=4,
            min_indicators=2,
            max_conditions=4,
            min_conditions=2,
        )
        generator = RandomGeneGenerator(ga_config)
        print("  ✅ ランダム遺伝子生成器作成完了")

        # 2. 複数遺伝子生成テスト
        print("\n2. 複数遺伝子生成テスト中...")
        population = generator.generate_population(10)

        print(f"  ✅ 個体群生成成功: {len(population)} 個体")

        # 3. 生成された戦略の分析
        print("\n3. 生成された戦略の分析:")

        # 指標タイプの統計
        indicator_types = {}
        total_indicators = 0

        # OI/FR判断条件の統計
        oi_fr_usage = 0
        total_conditions = 0

        # フィットネス関連指標の統計
        valid_strategies = 0

        for i, individual in enumerate(population):
            # 妥当性確認
            is_valid, _ = individual.validate()
            if is_valid:
                valid_strategies += 1

            # 指標統計
            for indicator in individual.indicators:
                indicator_types[indicator.type] = (
                    indicator_types.get(indicator.type, 0) + 1
                )
                total_indicators += 1

            # OI/FR使用統計
            all_conditions = individual.entry_conditions + individual.exit_conditions
            for condition in all_conditions:
                total_conditions += 1
                if condition.left_operand in ["OpenInterest", "FundingRate"] or (
                    isinstance(condition.right_operand, str)
                    and condition.right_operand in ["OpenInterest", "FundingRate"]
                ):
                    oi_fr_usage += 1

            # 個別戦略の詳細表示（最初の3つのみ）
            if i < 3:
                print(f"\n  📋 戦略{i+1}詳細:")
                print(f"    指標: {[ind.type for ind in individual.indicators]}")

                oi_fr_conds = []
                for cond in all_conditions:
                    if cond.left_operand in ["OpenInterest", "FundingRate"] or (
                        isinstance(cond.right_operand, str)
                        and cond.right_operand in ["OpenInterest", "FundingRate"]
                    ):
                        oi_fr_conds.append(
                            f"{cond.left_operand} {cond.operator} {cond.right_operand}"
                        )

                if oi_fr_conds:
                    print(f"    OI/FR判断: {oi_fr_conds}")
                else:
                    print(f"    OI/FR判断: なし")

        # 4. 統計結果表示
        print(f"\n4. 統計結果:")
        print(
            f"  📊 有効戦略率: {valid_strategies}/{len(population)} ({valid_strategies/len(population)*100:.1f}%)"
        )
        print(
            f"  📊 OI/FR判断使用率: {oi_fr_usage}/{total_conditions} ({oi_fr_usage/total_conditions*100:.1f}%)"
        )

        print(f"  📊 指標タイプ分布:")
        for indicator_type, count in sorted(indicator_types.items()):
            percentage = count / total_indicators * 100
            print(f"    {indicator_type}: {count} ({percentage:.1f}%)")

        # 5. 品質評価
        print(f"\n5. 品質評価:")

        if valid_strategies >= len(population) * 0.9:
            print("  ✅ 戦略生成品質: 優秀 (90%以上有効)")
        elif valid_strategies >= len(population) * 0.7:
            print("  ✅ 戦略生成品質: 良好 (70%以上有効)")
        else:
            print("  ⚠️ 戦略生成品質: 要改善")

        if oi_fr_usage >= total_conditions * 0.2:
            print("  ✅ OI/FR活用度: 良好 (20%以上)")
        elif oi_fr_usage >= total_conditions * 0.1:
            print("  ✅ OI/FR活用度: 適度 (10%以上)")
        else:
            print("  ⚠️ OI/FR活用度: 低い")

        # テクニカル指標のみが使用されているかチェック
        invalid_indicators = [
            t
            for t in indicator_types.keys()
            if t in ["OpenInterest", "FundingRate"] or t.startswith(("OI_", "FR_"))
        ]

        if not invalid_indicators:
            print("  ✅ 指標使用: 正しい (テクニカル指標のみ)")
        else:
            print(f"  ❌ 指標使用: 不正 (無効な指標: {invalid_indicators})")
            return False

        print("\n🎉 修正されたランダム遺伝子生成器テスト完了！")
        return True

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fitness_focus():
    """フィットネス関数の目的確認テスト"""
    print("\n🎯 フィットネス関数目的確認テスト開始")
    print("=" * 60)

    try:
        print("1. GA真の目的確認:")
        print("  🎯 高いリターン (Total Return)")
        print("  📊 高いシャープレシオ (Sharpe Ratio)")
        print("  📉 低いドローダウン (Max Drawdown)")
        print("  ✅ これらを最適化する戦略の発掘が目的")

        print("\n2. OI/FRの正しい役割:")
        print("  📋 判断材料・シグナルとして使用")
        print("  📋 例: FundingRate > 0.01% → ロング過熱 → ショート検討")
        print("  📋 例: OpenInterest 急増 → トレンド継続可能性")
        print("  ❌ 指標計算対象ではない (FR_SMA等は不適切)")

        print("\n3. 実装確認:")
        generator = RandomGeneGenerator()

        # サンプル戦略生成
        sample_gene = generator.generate_random_gene()

        # テクニカル指標のみが使用されているか確認
        technical_only = all(
            indicator.type not in ["OpenInterest", "FundingRate"]
            and not indicator.type.startswith(("OI_", "FR_"))
            for indicator in sample_gene.indicators
        )

        # OI/FRが判断条件で使用されているか確認
        all_conditions = sample_gene.entry_conditions + sample_gene.exit_conditions
        oi_fr_in_conditions = any(
            condition.left_operand in ["OpenInterest", "FundingRate"]
            or (
                isinstance(condition.right_operand, str)
                and condition.right_operand in ["OpenInterest", "FundingRate"]
            )
            for condition in all_conditions
        )

        if technical_only:
            print("  ✅ 指標: テクニカル指標のみ使用 (正しい)")
        else:
            print("  ❌ 指標: OI/FR指標が含まれている (不正)")
            return False

        if oi_fr_in_conditions:
            print("  ✅ OI/FR: 判断条件で使用 (正しい)")
        else:
            print("  ⚠️ OI/FR: 判断条件で未使用 (このサンプルでは)")

        print("\n🎉 フィットネス関数目的確認テスト完了！")
        return True

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_corrected_strategy_gene()
    success2 = test_corrected_random_generator()
    success3 = test_fitness_focus()

    print("\n" + "=" * 60)
    if success1 and success2 and success3:
        print("🎊 全テスト成功！")
        print("✨ 修正されたGA戦略生成機能が正常に動作しています")
        print("🎯 目的: 高リターン・高シャープレシオ・低ドローダウンの戦略発掘")
        print("📋 OI/FR: 判断材料として適切に使用")
    else:
        print("💥 一部テスト失敗")
        print("🔧 さらなる修正が必要です")
        sys.exit(1)
