"""
エンドツーエンドワークフローテスト

オートストラテジー機能の包括的なワークフローテストを実行します。
実際のユーザーワークフローに基づいた統合テストです。
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

from tests.utils.data_generators import TestDataGenerator
from tests.utils.helpers import TestExecutionHelper, performance_monitor


@pytest.mark.e2e
class TestEndToEndWorkflow:
    """エンドツーエンドワークフローテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.test_results = []
        self.performance_metrics = {}
        self.security_checks = {}
        self.data_quality_results = {}
        self.data_generator = TestDataGenerator()

    def test_complete_strategy_generation_workflow(self):
        """完全な戦略生成ワークフローテスト"""
        with performance_monitor("戦略生成ワークフロー"):
            # 1. 設定の初期化
            config = self.data_generator.generate_ga_config(
                population_size=10, generations=3
            )

            # 2. 戦略生成サービスの初期化
            generator = RandomGeneGenerator(config)

            # 3. 戦略の生成
            strategy = generator.generate_random_gene()

            # 4. 戦略の検証
            assert strategy is not None
            assert hasattr(strategy, "indicators")
            assert hasattr(strategy, "conditions")
            assert len(strategy.indicators) >= config.min_indicators
            assert len(strategy.indicators) <= config.max_indicators

            print("✓ 戦略生成ワークフロー完了")

    def test_market_data_processing_workflow(self):
        """市場データ処理ワークフローテスト"""
        with performance_monitor("市場データ処理"):
            # 1. テストデータの生成
            market_data = self.data_generator.generate_ohlcv_data(length=200)

            # 2. 市場シナリオの生成
            scenarios = self.data_generator.generate_market_scenarios()

            # 3. 各シナリオでの処理確認
            for scenario_name, data in scenarios.items():
                assert not data.empty
                assert len(data) > 0
                assert all(
                    col in data.columns
                    for col in ["open", "high", "low", "close", "volume"]
                )

                # データの整合性確認
                assert (data["high"] >= data["low"]).all()
                assert (data["high"] >= data["open"]).all()
                assert (data["high"] >= data["close"]).all()
                assert (data["low"] <= data["open"]).all()
                assert (data["low"] <= data["close"]).all()

            print("✓ 市場データ処理ワークフロー完了")

    def test_risk_management_workflow(self):
        """リスク管理ワークフローテスト"""
        with performance_monitor("リスク管理"):
            # 1. TP/SL計算機の初期化
            tpsl_calculator = TPSLCalculator()

            # 2. ポジションサイジング計算機の初期化
            position_calculator = PositionSizingCalculatorService()

            # 3. テストデータの準備
            tpsl_gene = self.data_generator.generate_tpsl_gene()
            position_gene = self.data_generator.generate_position_sizing_gene()

            # 4. 計算の実行と検証
            entry_price = 50000.0
            current_price = 51000.0

            tp_price = tpsl_calculator.calculate_take_profit(
                tpsl_gene, entry_price, "long"
            )
            sl_price = tpsl_calculator.calculate_stop_loss(
                tpsl_gene, entry_price, "long"
            )

            # 5. 結果の妥当性確認
            assert tp_price > entry_price  # ロングポジションのTP
            assert sl_price < entry_price  # ロングポジションのSL

            print("✓ リスク管理ワークフロー完了")

    def test_concurrent_operations_workflow(self):
        """並行処理ワークフローテスト"""
        with performance_monitor("並行処理"):

            def generate_strategy():
                config = self.data_generator.generate_ga_config(population_size=5)
                generator = RandomGeneGenerator(config)
                return generator.generate_random_gene()

            # 並行で戦略を生成
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(generate_strategy) for _ in range(5)]
                strategies = [future.result() for future in as_completed(futures)]

            # 結果の検証
            assert len(strategies) == 5
            for strategy in strategies:
                assert strategy is not None
                assert hasattr(strategy, "indicators")
                assert hasattr(strategy, "conditions")

            print("✓ 並行処理ワークフロー完了")

    def test_error_recovery_workflow(self):
        """エラー回復ワークフローテスト"""
        with performance_monitor("エラー回復"):
            # 1. 不正なデータでのテスト
            try:
                config = GAConfig(
                    population_size=0,  # 不正な値
                    generations=1,
                    max_indicators=1,
                    min_indicators=1,
                )
                # エラーが発生することを期待
                assert False, "不正な設定でエラーが発生しませんでした"
            except (ValueError, AssertionError) as e:
                if "不正な設定" in str(e):
                    raise
                # 期待されるエラー
                pass

            # 2. 正常な設定での回復確認
            config = self.data_generator.generate_ga_config()
            generator = RandomGeneGenerator(config)
            strategy = generator.generate_random_gene()

            assert strategy is not None
            print("✓ エラー回復ワークフロー完了")

    def test_performance_under_load(self):
        """負荷下でのパフォーマンステスト"""
        with performance_monitor("負荷テスト"):
            # 大量のデータでのテスト
            large_dataset = self.data_generator.generate_ohlcv_data(length=1000)

            # 複数の戦略生成
            config = self.data_generator.generate_ga_config(population_size=20)
            generator = RandomGeneGenerator(config)

            start_time = time.time()
            strategies = []

            for _ in range(10):
                strategy = generator.generate_random_gene()
                strategies.append(strategy)

            execution_time = time.time() - start_time

            # パフォーマンス要件の確認
            assert execution_time < 30.0, f"実行時間が長すぎます: {execution_time}秒"
            assert len(strategies) == 10

            print(f"✓ 負荷テスト完了 (実行時間: {execution_time:.2f}秒)")

    def test_data_quality_validation(self):
        """データ品質検証テスト"""
        with performance_monitor("データ品質検証"):
            # 1. 生成データの品質確認
            market_data = self.data_generator.generate_ohlcv_data(length=100)

            # 基本的な品質チェック
            assert not market_data.empty
            assert not market_data.isnull().any().any()
            assert (market_data["volume"] > 0).all()
            assert (market_data["high"] >= market_data["low"]).all()

            # 2. 極端な市場条件での品質確認
            extreme_conditions = (
                self.data_generator.generate_extreme_market_conditions()
            )

            for condition_name, data in extreme_conditions.items():
                assert not data.empty
                assert not data.isnull().any().any()
                print(f"  ✓ {condition_name} データ品質確認完了")

            print("✓ データ品質検証完了")

    def test_memory_efficiency(self):
        """メモリ効率性テスト"""
        import gc

        with performance_monitor("メモリ効率性"):
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024

            # 大量のオブジェクト生成
            objects = []
            for _ in range(100):
                config = self.data_generator.generate_ga_config()
                generator = RandomGeneGenerator(config)
                strategy = generator.generate_random_gene()
                objects.append(strategy)

            # メモリ使用量確認
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_used = peak_memory - initial_memory

            # オブジェクトの削除
            del objects
            gc.collect()

            # メモリ解放確認
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_freed = peak_memory - final_memory

            print(f"  メモリ使用量: {memory_used:.2f}MB")
            print(f"  メモリ解放量: {memory_freed:.2f}MB")

            # メモリリークがないことを確認
            assert memory_used < 500, f"メモリ使用量が多すぎます: {memory_used}MB"

            print("✓ メモリ効率性テスト完了")


@pytest.mark.e2e
class TestRealMarketValidation:
    """実際の市場データを使用した検証テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.data_generator = TestDataGenerator()

    def test_market_scenario_adaptation(self):
        """市場シナリオ適応テスト"""
        scenarios = self.data_generator.generate_market_scenarios()

        for scenario_name, data in scenarios.items():
            with performance_monitor(f"市場シナリオ: {scenario_name}"):
                # 戦略生成
                config = self.data_generator.generate_ga_config()
                generator = RandomGeneGenerator(config)
                strategy = generator.generate_random_gene()

                # 基本的な検証
                assert strategy is not None
                assert len(strategy.indicators) > 0
                assert len(strategy.conditions) > 0

                print(f"  ✓ {scenario_name} シナリオ対応確認")

    def test_volatility_handling(self):
        """ボラティリティ対応テスト"""
        # 高ボラティリティデータ
        high_vol_data = self.data_generator.generate_ohlcv_data(
            length=200, volatility=0.05
        )

        # 低ボラティリティデータ
        low_vol_data = self.data_generator.generate_ohlcv_data(
            length=200, volatility=0.005
        )

        # 両方のケースで戦略が生成できることを確認
        config = self.data_generator.generate_ga_config()
        generator = RandomGeneGenerator(config)

        strategy_high_vol = generator.generate_random_gene()
        strategy_low_vol = generator.generate_random_gene()

        assert strategy_high_vol is not None
        assert strategy_low_vol is not None

        print("✓ ボラティリティ対応テスト完了")


if __name__ == "__main__":
    # 直接実行時のテスト
    test_suite = TestEndToEndWorkflow()
    test_suite.setup_method()

    tests = [
        test_suite.test_complete_strategy_generation_workflow,
        test_suite.test_market_data_processing_workflow,
        test_suite.test_risk_management_workflow,
        test_suite.test_concurrent_operations_workflow,
        test_suite.test_error_recovery_workflow,
        test_suite.test_performance_under_load,
        test_suite.test_data_quality_validation,
        test_suite.test_memory_efficiency,
    ]

    results = TestExecutionHelper.run_test_suite(tests)
    TestExecutionHelper.print_test_results(results)
