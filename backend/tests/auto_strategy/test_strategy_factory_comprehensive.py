"""
戦略ファクトリー機能の包括的テスト
"""

import pytest
import asyncio
import time
import json
import random
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import sys
import os

# パスを追加
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
    encode_gene_to_list,
    decode_list_to_gene,
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory


class TestStrategyFactoryComprehensive:
    """戦略ファクトリーの包括的テスト"""

    def test_factory_with_all_indicators(self):
        """全指標対応テスト"""
        print("\n=== 全指標対応テスト ===")

        factory = StrategyFactory()
        all_indicators = list(factory.indicator_adapters.keys())

        successful_strategies = 0
        failed_strategies = 0

        for indicator in all_indicators:
            try:
                gene = StrategyGene(
                    indicators=[
                        IndicatorGene(type=indicator, parameters={"period": 20})
                    ],
                    entry_conditions=[Condition("price", ">", 100)],
                    exit_conditions=[Condition("price", "<", 90)],
                )

                is_valid, errors = factory.validate_gene(gene)
                if is_valid:
                    strategy_class = factory.create_strategy_class(gene)
                    successful_strategies += 1
                    print(f"✅ {indicator}: 戦略クラス生成成功")
                else:
                    failed_strategies += 1
                    print(f"❌ {indicator}: 妥当性検証失敗: {errors}")

            except Exception as e:
                failed_strategies += 1
                print(f"❌ {indicator}: 戦略生成エラー: {e}")

        print(f"✅ 成功: {successful_strategies}, 失敗: {failed_strategies}")
        success_rate = successful_strategies / len(all_indicators)
        assert success_rate > 0.8, f"成功率が低すぎます: {success_rate:.2%}"

    def test_complex_strategy_generation(self):
        """複雑な戦略生成テスト"""
        print("\n=== 複雑な戦略生成テスト ===")

        factory = StrategyFactory()

        # 複雑な戦略遺伝子
        complex_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}),
                IndicatorGene(type="EMA", parameters={"period": 12}),
                IndicatorGene(type="RSI", parameters={"period": 14}),
                IndicatorGene(
                    type="MACD", parameters={"fast": 12, "slow": 26, "signal": 9}
                ),
                IndicatorGene(type="BB", parameters={"period": 20, "std": 2}),
            ],
            entry_conditions=[
                Condition("RSI_14", "<", 30),
                Condition("SMA_20", "cross_above", "EMA_12"),
                Condition("price", ">", "BB_lower"),
            ],
            exit_conditions=[
                Condition("RSI_14", ">", 70),
                Condition("SMA_20", "cross_below", "EMA_12"),
                Condition("price", "<", "BB_upper"),
            ],
            risk_management={"stop_loss": 0.02, "take_profit": 0.05},
        )

        try:
            is_valid, errors = factory.validate_gene(complex_gene)
            print(f"✅ 複雑な戦略の妥当性: {is_valid}")
            if not is_valid:
                print(f"   エラー: {errors}")

            if is_valid:
                strategy_class = factory.create_strategy_class(complex_gene)
                strategy_instance = strategy_class()
                print(f"✅ 複雑な戦略クラス生成成功")
                print(f"   指標数: {len(complex_gene.indicators)}")
                print(f"   エントリー条件数: {len(complex_gene.entry_conditions)}")
                print(f"   イグジット条件数: {len(complex_gene.exit_conditions)}")

        except Exception as e:
            print(f"❌ 複雑な戦略生成エラー: {e}")
            raise


if __name__ == "__main__":
    print("戦略ファクトリー包括的テスト実行")
    test_suite = TestStrategyFactoryComprehensive()
    test_suite.test_factory_with_all_indicators()
    test_suite.test_complex_strategy_generation()
