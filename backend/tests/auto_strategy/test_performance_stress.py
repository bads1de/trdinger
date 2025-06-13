"""
自動戦略生成機能のパフォーマンステスト
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


def run_stress_test():
    """ストレステスト"""
    print("\n" + "=" * 60)
    print("🔥 ストレステスト開始")
    print("=" * 60)

    # 大量の戦略遺伝子生成・処理
    start_time = time.time()

    factory = StrategyFactory()
    genes = []

    # 1000個の戦略遺伝子を生成
    for i in range(1000):
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": random.randint(5, 50)}),
                IndicatorGene(
                    type="RSI", parameters={"period": random.randint(10, 30)}
                ),
            ],
            entry_conditions=[Condition("RSI_14", "<", random.randint(20, 40))],
            exit_conditions=[Condition("RSI_14", ">", random.randint(60, 80))],
        )
        genes.append(gene)

    generation_time = time.time() - start_time
    print(f"✅ 1000個の戦略遺伝子生成: {generation_time:.2f}秒")

    # 妥当性検証
    validation_start = time.time()
    valid_count = 0

    for gene in genes:
        is_valid, _ = factory.validate_gene(gene)
        if is_valid:
            valid_count += 1

    validation_time = time.time() - validation_start
    print(f"✅ 妥当性検証: {valid_count}/1000 有効 ({validation_time:.2f}秒)")

    # エンコード/デコード性能
    encode_start = time.time()
    for gene in genes[:100]:  # 100個でテスト
        encoded = encode_gene_to_list(gene)
        decoded = decode_list_to_gene(encoded)

    encode_time = time.time() - encode_start
    print(f"✅ エンコード/デコード (100個): {encode_time:.2f}秒")

    total_time = time.time() - start_time
    print(f"🎯 ストレステスト完了: {total_time:.2f}秒")

    # パフォーマンス基準
    assert generation_time < 5.0, f"遺伝子生成が遅すぎます: {generation_time}秒"
    assert validation_time < 2.0, f"妥当性検証が遅すぎます: {validation_time}秒"
    assert encode_time < 1.0, f"エンコード処理が遅すぎます: {encode_time}秒"
    assert valid_count > 950, f"有効な遺伝子が少なすぎます: {valid_count}/1000"

    print("🎉 ストレステスト全て成功！")


if __name__ == "__main__":
    run_stress_test()
