"""
エンドツーエンド完全ワークフローテスト

オートストラテジー機能の完全なユーザーワークフローを検証します。
実際のユーザーシナリオに基づいた包括的なテストを実行します。
"""

import pytest
import sys
import os
import time
import json
import threading
import psutil
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# テスト対象のモジュールをインポートするためのパス設定
backend_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, backend_path)

from app.core.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene
from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.core.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
from app.core.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
)
from app.core.services.indicators import TechnicalIndicatorService

from tests.utils.helpers import (
    TestExecutionHelper,
    performance_monitor,
    assert_financial_precision,
    ConcurrencyTestHelper,
)
from tests.utils.data_generators import TestDataGenerator, PerformanceTestHelper


@pytest.mark.e2e
class TestCompleteWorkflow:
    """エンドツーエンド完全ワークフローテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.config = GAConfig(
            population_size=5,
            generations=2,
            max_indicators=3,
            min_indicators=1,
            max_conditions=2,
            min_conditions=1,
        )
        self.generator = RandomGeneGenerator(self.config)
        self.indicator_service = TechnicalIndicatorService()
        self.data_generator = TestDataGenerator()
        self.performance_helper = PerformanceTestHelper()
        self.test_results = []

    @pytest.mark.slow
    def test_end_to_end_workflow(self):
        """完全なエンドツーエンドワークフローテスト"""
        with performance_monitor("完全ワークフロー"):
            # 1. 戦略生成
            strategy_gene = self._test_strategy_generation()

            # 2. インジケータ統合確認
            self._test_indicator_integration()

            # 3. データ品質検証
            self._test_data_quality_and_integrity()

            # 4. パフォーマンス検証
            self._test_performance_requirements()

            # 5. 並行処理テスト
            self._test_concurrent_operations()

            # 6. エラー回復テスト
            self._test_error_recovery_mechanisms()

    def _test_strategy_generation(self) -> StrategyGene:
        """戦略生成テスト"""
        print("📊 戦略生成テスト開始")

        strategy_gene = self.generator.generate_random_gene()

        # 基本的な検証
        assert strategy_gene is not None
        assert len(strategy_gene.indicators) >= self.config.min_indicators
        assert len(strategy_gene.indicators) <= self.config.max_indicators
        assert len(strategy_gene.conditions) >= self.config.min_conditions
        assert len(strategy_gene.conditions) <= self.config.max_conditions

        print(f"✓ 戦略生成成功: {len(strategy_gene.indicators)}個のインジケータ")
        return strategy_gene

    def _test_indicator_integration(self):
        """インジケータ統合テスト"""
        print("📊 インジケータ統合テスト開始")

        supported_indicators = self.indicator_service.get_supported_indicators()

        # 新しいカテゴリのインジケータが含まれていることを確認
        new_indicators = [
            "HT_DCPERIOD",
            "HT_DCPHASE",
            "HT_SINE",  # サイクル系
            "BETA",
            "CORREL",
            "STDDEV",
            "VAR",  # 統計系
            "ACOS",
            "ASIN",
            "COS",
            "SIN",
            "SQRT",  # 数学変換系
            "ADD",
            "SUB",
            "MULT",
            "DIV",  # 数学演算子系
            "CDL_DOJI",
            "CDL_HAMMER",
            "CDL_HANGING_MAN",  # パターン認識系
        ]

        found_indicators = [
            ind for ind in new_indicators if ind in supported_indicators
        ]
        assert len(found_indicators) > 0, "新しいインジケータが見つかりません"

        print(f"✓ インジケータ統合成功: {len(found_indicators)}個の新インジケータ確認")

    def _test_data_quality_and_integrity(self):
        """データ品質と整合性テスト"""
        print("📊 データ品質テスト開始")

        # 様々な市場シナリオでのテスト
        market_scenarios = self.data_generator.generate_market_scenarios()

        for scenario_name, data in market_scenarios.items():
            # データ品質チェック
            assert not data.empty, f"{scenario_name}: データが空です"
            assert len(data) > 50, f"{scenario_name}: データが不十分です"

            # 価格データの論理的整合性チェック
            assert (data["high"] >= data["low"]).all(), f"{scenario_name}: 高値 < 安値"
            assert (data["high"] >= data["open"]).all(), f"{scenario_name}: 高値 < 始値"
            assert (
                data["high"] >= data["close"]
            ).all(), f"{scenario_name}: 高値 < 終値"
            assert (data["low"] <= data["open"]).all(), f"{scenario_name}: 安値 > 始値"
            assert (data["low"] <= data["close"]).all(), f"{scenario_name}: 安値 > 終値"

        print("✓ データ品質テスト成功")

    def _test_performance_requirements(self):
        """パフォーマンス要件テスト"""
        print("📊 パフォーマンステスト開始")

        # 市場データ処理パフォーマンス（< 100ms）
        test_data = self.data_generator.generate_ohlcv_data(100)

        def process_market_data():
            return self.indicator_service.calculate_indicators(
                test_data, ["SMA", "RSI"]
            )

        result, execution_time = self.performance_helper.measure_execution_time(
            process_market_data
        )
        assert (
            execution_time < 0.1
        ), f"市場データ処理が遅すぎます: {execution_time:.3f}秒"

        # 戦略シグナル生成パフォーマンス（< 500ms）
        def generate_strategy_signal():
            return self.generator.generate_random_gene()

        result, execution_time = self.performance_helper.measure_execution_time(
            generate_strategy_signal
        )
        assert (
            execution_time < 0.5
        ), f"戦略シグナル生成が遅すぎます: {execution_time:.3f}秒"

        print("✓ パフォーマンステスト成功")

    def _test_concurrent_operations(self):
        """並行処理テスト"""
        print("📊 並行処理テスト開始")

        def concurrent_strategy_generation():
            return self.generator.generate_random_gene()

        # 5つの並行処理で戦略生成
        results = ConcurrencyTestHelper.run_concurrent_operations(
            concurrent_strategy_generation, num_threads=5
        )

        assert len(results) == 5, "並行処理で期待される結果数が得られませんでした"

        # 各結果が有効であることを確認
        for result in results:
            assert result is not None
            assert hasattr(result, "indicators")
            assert hasattr(result, "conditions")

        print("✓ 並行処理テスト成功")

    def _test_error_recovery_mechanisms(self):
        """エラー回復メカニズムテスト"""
        print("📊 エラー回復テスト開始")

        # 無効なデータでのテスト
        invalid_data = self.data_generator.generate_ohlcv_data(5)  # 少なすぎるデータ

        try:
            # エラーが適切にハンドリングされることを確認
            result = self.indicator_service.calculate_indicators(invalid_data, ["SMA"])
            # エラーが発生しない場合は、適切にハンドリングされている
        except Exception as e:
            # エラーが発生した場合は、適切なエラーメッセージであることを確認
            assert (
                "insufficient data" in str(e).lower() or "not enough" in str(e).lower()
            )

        print("✓ エラー回復テスト成功")

    @pytest.mark.slow
    def test_scalability_and_performance(self):
        """スケーラビリティとパフォーマンステスト"""
        with performance_monitor("スケーラビリティテスト"):
            # 大量データでのテスト
            large_dataset = self.data_generator.generate_ohlcv_data(1000)

            def process_large_dataset():
                return self.indicator_service.calculate_indicators(
                    large_dataset, ["SMA", "EMA", "RSI", "MACD"]
                )

            result, execution_time = self.performance_helper.measure_execution_time(
                process_large_dataset
            )
            result, memory_used = self.performance_helper.measure_memory_usage(
                process_large_dataset
            )

            # パフォーマンス要件の確認
            assert (
                execution_time < 5.0
            ), f"大量データ処理が遅すぎます: {execution_time:.2f}秒"
            assert memory_used < 200, f"メモリ使用量が多すぎます: {memory_used:.2f}MB"

    @pytest.mark.slow
    def test_security_and_robustness(self):
        """セキュリティと堅牢性テスト"""
        with performance_monitor("セキュリティテスト"):
            # 極端な市場条件でのテスト
            extreme_conditions = (
                self.data_generator.generate_extreme_market_conditions()
            )

            for condition_name, data in extreme_conditions.items():
                try:
                    # 極端な条件でも適切に動作することを確認
                    result = self.indicator_service.calculate_indicators(
                        data, ["SMA", "RSI"]
                    )

                    # 結果が有効であることを確認
                    assert result is not None

                except Exception as e:
                    # エラーが発生した場合は、適切にハンドリングされていることを確認
                    assert "invalid" in str(e).lower() or "error" in str(e).lower()

    def teardown_method(self):
        """テスト後処理"""
        # テスト結果のサマリー出力
        if self.test_results:
            TestExecutionHelper.print_test_results(
                {
                    "passed": len(
                        [r for r in self.test_results if r.get("status") == "PASSED"]
                    ),
                    "failed": len(
                        [r for r in self.test_results if r.get("status") == "FAILED"]
                    ),
                    "total": len(self.test_results),
                    "details": self.test_results,
                }
            )
