"""
戦略遺伝子機能の包括的テスト
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


class TestStrategyGeneComprehensive:
    """戦略遺伝子の包括的テスト"""

    def test_large_scale_gene_creation(self):
        """大量の戦略遺伝子作成テスト"""
        print("\n=== 大量戦略遺伝子作成テスト ===")

        indicators = [
            "SMA",
            "EMA",
            "RSI",
            "MACD",
            "BB",
            "STOCH",
            "CCI",
            "WILLIAMS",
            "ADX",
        ]
        operators = [">", "<", ">=", "<=", "cross_above", "cross_below"]

        genes = []
        start_time = time.time()

        for i in range(1000):
            # ランダムな指標を選択
            num_indicators = random.randint(1, 5)
            selected_indicators = random.sample(indicators, num_indicators)

            indicator_genes = []
            for ind_type in selected_indicators:
                period = random.randint(5, 200)
                indicator_genes.append(
                    IndicatorGene(
                        type=ind_type, parameters={"period": period}, enabled=True
                    )
                )

            # ランダムな条件を生成
            entry_conditions = []
            exit_conditions = []

            for _ in range(random.randint(1, 3)):
                left_operand = (
                    f"{random.choice(selected_indicators)}_{random.randint(5, 50)}"
                )
                operator = random.choice(operators)
                right_operand = random.choice(
                    [
                        f"{random.choice(selected_indicators)}_{random.randint(5, 50)}",
                        random.uniform(10, 90),
                    ]
                )

                entry_conditions.append(
                    Condition(
                        left_operand=left_operand,
                        operator=operator,
                        right_operand=right_operand,
                    )
                )

                exit_conditions.append(
                    Condition(
                        left_operand=left_operand,
                        operator=operator,
                        right_operand=right_operand,
                    )
                )

            gene = StrategyGene(
                indicators=indicator_genes,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management={
                    "stop_loss": random.uniform(0.01, 0.05),
                    "take_profit": random.uniform(0.02, 0.10),
                },
            )

            genes.append(gene)

        creation_time = time.time() - start_time
        print(f"✅ 1000個の戦略遺伝子作成完了: {creation_time:.2f}秒")

        # 妥当性検証
        valid_count = 0
        validation_start = time.time()

        for gene in genes:
            is_valid, _ = gene.validate()
            if is_valid:
                valid_count += 1

        validation_time = time.time() - validation_start
        print(f"✅ 妥当性検証完了: {valid_count}/1000 有効 ({validation_time:.2f}秒)")

        assert valid_count > 800, f"有効な遺伝子が少なすぎます: {valid_count}/1000"

        return genes

    def test_serialization_performance(self):
        """シリアライゼーション性能テスト"""
        print("\n=== シリアライゼーション性能テスト ===")

        # テスト用遺伝子を作成
        gene = StrategyGene(
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
                Condition("SMA_20", ">", "EMA_12"),
            ],
            exit_conditions=[
                Condition("RSI_14", ">", 70),
                Condition("SMA_20", "<", "EMA_12"),
            ],
        )

        # JSON シリアライゼーション性能
        start_time = time.time()
        for _ in range(10000):
            json_str = gene.to_json()
            restored_gene = StrategyGene.from_json(json_str)
        json_time = time.time() - start_time
        print(f"✅ JSON シリアライゼーション (10000回): {json_time:.2f}秒")

        # エンコード/デコード性能
        start_time = time.time()
        for _ in range(10000):
            encoded = encode_gene_to_list(gene)
            decoded_gene = decode_list_to_gene(encoded)
        encode_time = time.time() - start_time
        print(f"✅ エンコード/デコード (10000回): {encode_time:.2f}秒")

        assert json_time < 10.0, f"JSON処理が遅すぎます: {json_time}秒"
        assert encode_time < 5.0, f"エンコード処理が遅すぎます: {encode_time}秒"


if __name__ == "__main__":
    print("戦略遺伝子包括的テスト実行")
    test_suite = TestStrategyGeneComprehensive()
    test_suite.test_large_scale_gene_creation()
    test_suite.test_serialization_performance()
