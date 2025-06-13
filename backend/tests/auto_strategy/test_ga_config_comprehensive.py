"""
GA設定機能の包括的テスト
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

from app.core.services.auto_strategy.models.ga_config import GAConfig, GAProgress


class TestGAConfigComprehensive:
    """GA設定の包括的テスト"""

    def test_config_variations(self):
        """設定バリエーションテスト"""
        print("\n=== GA設定バリエーションテスト ===")

        test_configs = [
            # 小規模設定
            {"population_size": 10, "generations": 5},
            # 中規模設定
            {"population_size": 50, "generations": 30},
            # 大規模設定
            {"population_size": 200, "generations": 100},
            # 極端な設定
            {"population_size": 500, "generations": 200},
        ]

        valid_configs = 0
        for i, config_params in enumerate(test_configs):
            try:
                config = GAConfig(**config_params)
                is_valid, errors = config.validate()

                if is_valid:
                    valid_configs += 1
                    print(f"✅ 設定{i+1}: {config_params} - 有効")
                else:
                    print(f"❌ 設定{i+1}: {config_params} - 無効: {errors}")

            except Exception as e:
                print(f"❌ 設定{i+1}: {config_params} - エラー: {e}")

        print(f"✅ 有効な設定: {valid_configs}/{len(test_configs)}")
        assert valid_configs >= len(test_configs) - 1, "設定の妥当性に問題があります"

    def test_fitness_weight_combinations(self):
        """フィットネス重み組み合わせテスト"""
        print("\n=== フィットネス重み組み合わせテスト ===")

        weight_combinations = [
            {
                "total_return": 1.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            },
            {
                "total_return": 0.0,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            },
            {
                "total_return": 0.25,
                "sharpe_ratio": 0.25,
                "max_drawdown": 0.25,
                "win_rate": 0.25,
            },
            {
                "total_return": 0.4,
                "sharpe_ratio": 0.3,
                "max_drawdown": 0.2,
                "win_rate": 0.1,
            },
            {
                "total_return": 0.1,
                "sharpe_ratio": 0.6,
                "max_drawdown": 0.2,
                "win_rate": 0.1,
            },
        ]

        valid_weights = 0
        for i, weights in enumerate(weight_combinations):
            config = GAConfig(fitness_weights=weights)
            is_valid, errors = config.validate()

            if is_valid:
                valid_weights += 1
                print(f"✅ 重み{i+1}: 有効")
            else:
                print(f"❌ 重み{i+1}: 無効: {errors}")

        print(f"✅ 有効な重み設定: {valid_weights}/{len(weight_combinations)}")
        assert valid_weights == len(
            weight_combinations
        ), "重み設定の妥当性に問題があります"


if __name__ == "__main__":
    print("GA設定包括的テスト実行")
    test_suite = TestGAConfigComprehensive()
    test_suite.test_config_variations()
    test_suite.test_fitness_weight_combinations()
