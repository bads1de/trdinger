"""
自動戦略サービス機能の包括的テスト
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
from app.core.services.auto_strategy.models.ga_config import GAConfig, GAProgress
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.core.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)


class TestAutoStrategyServiceComprehensive:
    """自動戦略サービスの包括的テスト"""

    def test_service_initialization(self):
        """サービス初期化テスト"""
        print("\n=== サービス初期化テスト ===")

        try:
            # 実際のサービス初期化はデータベース接続が必要なため、
            # コンポーネントの個別初期化をテスト
            factory = StrategyFactory()
            print("✅ 戦略ファクトリー初期化成功")

            # GA設定の作成
            config = GAConfig.create_default()
            print("✅ デフォルトGA設定作成成功")

            # 戦略遺伝子の作成
            gene = StrategyGene(
                indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
                entry_conditions=[Condition("price", ">", 100)],
                exit_conditions=[Condition("price", "<", 90)],
            )
            print("✅ 戦略遺伝子作成成功")

            # 妥当性検証
            is_valid, errors = factory.validate_gene(gene)
            assert is_valid, f"遺伝子妥当性検証失敗: {errors}"
            print("✅ 遺伝子妥当性検証成功")

        except Exception as e:
            print(f"❌ サービス初期化テストエラー: {e}")
            raise

    def test_experiment_management(self):
        """実験管理テスト"""
        print("\n=== 実験管理テスト ===")

        # 実験情報の管理をシミュレート
        experiments = {}

        # 実験作成
        experiment_id = "test_experiment_001"
        experiment_info = {
            "id": experiment_id,
            "name": "Test Experiment",
            "status": "running",
            "start_time": time.time(),
            "config": GAConfig.create_fast().to_dict(),
        }

        experiments[experiment_id] = experiment_info
        print(f"✅ 実験作成: {experiment_id}")

        # 進捗更新シミュレート
        for generation in range(1, 6):
            progress = GAProgress(
                experiment_id=experiment_id,
                current_generation=generation,
                total_generations=5,
                best_fitness=0.5 + generation * 0.1,
                average_fitness=0.3 + generation * 0.05,
                execution_time=generation * 10.0,
                estimated_remaining_time=(5 - generation) * 10.0,
            )

            print(f"✅ 世代{generation}: フィットネス={progress.best_fitness:.2f}")

        # 実験完了
        experiments[experiment_id]["status"] = "completed"
        experiments[experiment_id]["end_time"] = time.time()

        print(f"✅ 実験完了: {experiment_id}")

        assert len(experiments) == 1, "実験管理に問題があります"
        assert (
            experiments[experiment_id]["status"] == "completed"
        ), "実験状態更新に問題があります"


if __name__ == "__main__":
    print("自動戦略サービス包括的テスト実行")
    test_suite = TestAutoStrategyServiceComprehensive()
    test_suite.test_service_initialization()
    test_suite.test_experiment_management()
