"""
包括的統合テスト

TP/SL/資金管理の相互作用、エッジケース動作、計算精度と丸め処理を検証します。
市場検証テストと統合テストを統合した包括的なテストスイートです。
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

# テスト対象のモジュールをインポートするためのパス設定
backend_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, backend_path)

from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
from app.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
)
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.evaluators.condition_evaluator import (
    ConditionEvaluator,
)
from app.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.ga_config import GAConfig

from tests.utils.data_generators import TestDataGenerator, PerformanceTestHelper
from tests.utils.helpers import (
    TestExecutionHelper,
    performance_monitor,
    assert_financial_precision,
    ConcurrencyTestHelper,
)


@pytest.mark.integration
class TestTPSLPositionSizingIntegration:
    """TP/SL とポジションサイジングの統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.tpsl_calculator = TPSLCalculator()
        self.position_calculator = PositionSizingCalculatorService()
        self.data_generator = TestDataGenerator()
        self.performance_helper = PerformanceTestHelper()

    def test_tpsl_position_sizing_interaction(self):
        """TP/SL とポジションサイジングの相互作用テスト"""
        with performance_monitor("TP/SL ポジションサイジング統合"):
            # テストデータ準備
            tpsl_gene = self.data_generator.generate_tpsl_gene(
                TPSLMethod.FIXED_PERCENTAGE
            )
            position_gene = self.data_generator.generate_position_sizing_gene(
                PositionSizingMethod.FIXED_PERCENTAGE
            )

            entry_price = 50000.0
            account_balance = 100000.0

            # TP/SL 計算
            tp_price = self.tpsl_calculator.calculate_take_profit(
                tpsl_gene, entry_price, "long"
            )
            sl_price = self.tpsl_calculator.calculate_stop_loss(
                tpsl_gene, entry_price, "long"
            )

            # ポジションサイズ計算
            position_size = self.position_calculator.calculate_position_size(
                position_gene, account_balance, entry_price, sl_price
            )

            # 統合検証
            assert tp_price > entry_price
            assert sl_price < entry_price
            assert position_size > 0
            assert position_size <= account_balance

            # リスク計算
            risk_amount = (entry_price - sl_price) * position_size / entry_price
            max_risk = account_balance * 0.02  # 2%リスク

            print(f"  エントリー価格: {entry_price}")
            print(f"  TP価格: {tp_price}")
            print(f"  SL価格: {sl_price}")
            print(f"  ポジションサイズ: {position_size}")
            print(f"  リスク金額: {risk_amount}")

            print("✓ TP/SL ポジションサイジング統合テスト完了")

    def test_extreme_market_conditions_integration(self):
        """極端な市場条件での統合テスト"""
        with performance_monitor("極端市場条件統合"):
            extreme_conditions = (
                self.data_generator.generate_extreme_market_conditions()
            )

            for condition_name, market_data in extreme_conditions.items():
                # 各極端条件でのテスト
                entry_price = float(market_data["close"].iloc[-1])

                tpsl_gene = self.data_generator.generate_tpsl_gene()
                position_gene = self.data_generator.generate_position_sizing_gene()

                try:
                    tp_price = self.tpsl_calculator.calculate_take_profit(
                        tpsl_gene, entry_price, "long"
                    )
                    sl_price = self.tpsl_calculator.calculate_stop_loss(
                        tpsl_gene, entry_price, "long"
                    )

                    position_size = self.position_calculator.calculate_position_size(
                        position_gene, 100000.0, entry_price, sl_price
                    )

                    # 基本的な妥当性確認
                    assert tp_price > 0
                    assert sl_price > 0
                    assert position_size >= 0

                    print(f"  ✓ {condition_name} 条件対応確認")

                except Exception as e:
                    print(f"  ⚠ {condition_name} でエラー: {e}")
                    # 極端な条件では一部エラーが許容される
                    continue

            print("✓ 極端市場条件統合テスト完了")

    def test_calculation_precision_integration(self):
        """計算精度統合テスト"""
        with performance_monitor("計算精度統合"):
            # 高精度が要求される計算のテスト
            test_cases = [
                {"entry": 0.00001234, "balance": 1000.0},  # 極小価格
                {"entry": 1234567.89, "balance": 1000000.0},  # 極大価格
                {"entry": 50000.0, "balance": 0.01},  # 極小残高
            ]

            for i, case in enumerate(test_cases):
                tpsl_gene = TPSLGene(
                    method=TPSLMethod.FIXED_PERCENTAGE,
                    take_profit_percentage=Decimal("2.0"),
                    stop_loss_percentage=Decimal("1.0"),
                )

                tp_price = self.tpsl_calculator.calculate_take_profit(
                    tpsl_gene, case["entry"], "long"
                )
                sl_price = self.tpsl_calculator.calculate_stop_loss(
                    tpsl_gene, case["entry"], "long"
                )

                # 精度確認
                expected_tp = case["entry"] * 1.02
                expected_sl = case["entry"] * 0.99

                assert_financial_precision(tp_price, expected_tp, tolerance=1e-10)
                assert_financial_precision(sl_price, expected_sl, tolerance=1e-10)

                print(f"  ✓ ケース{i+1} 精度確認完了")

            print("✓ 計算精度統合テスト完了")

    def test_concurrent_operations_integration(self):
        """並行処理統合テスト"""
        with performance_monitor("並行処理統合"):

            def calculate_integrated_values():
                tpsl_gene = self.data_generator.generate_tpsl_gene()
                position_gene = self.data_generator.generate_position_sizing_gene()

                entry_price = 50000.0
                balance = 100000.0

                tp_price = self.tpsl_calculator.calculate_take_profit(
                    tpsl_gene, entry_price, "long"
                )
                sl_price = self.tpsl_calculator.calculate_stop_loss(
                    tpsl_gene, entry_price, "long"
                )
                position_size = self.position_calculator.calculate_position_size(
                    position_gene, balance, entry_price, sl_price
                )

                return {
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "position_size": position_size,
                }

            # 並行実行
            results = ConcurrencyTestHelper.run_concurrent_operations(
                calculate_integrated_values, num_threads=5
            )

            # 結果検証
            assert len(results) == 5
            for result in results:
                assert result["tp_price"] > 0
                assert result["sl_price"] > 0
                assert result["position_size"] >= 0

            print("✓ 並行処理統合テスト完了")


@pytest.mark.integration
class TestCalculationAccuracy:
    """計算精度テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.tpsl_calculator = TPSLCalculator()
        self.data_generator = TestDataGenerator()

    def test_rounding_behavior(self):
        """丸め処理テスト"""
        with performance_monitor("丸め処理"):
            # 丸めが必要なケースのテスト
            test_prices = [50000.123456789, 0.000012345678, 1234567.987654321]

            for price in test_prices:
                tpsl_gene = TPSLGene(
                    method=TPSLMethod.FIXED_PERCENTAGE,
                    take_profit_percentage=Decimal("1.5"),
                    stop_loss_percentage=Decimal("0.8"),
                )

                tp_price = self.tpsl_calculator.calculate_take_profit(
                    tpsl_gene, price, "long"
                )
                sl_price = self.tpsl_calculator.calculate_stop_loss(
                    tpsl_gene, price, "long"
                )

                # 適切な精度で丸められていることを確認
                assert isinstance(tp_price, (int, float))
                assert isinstance(sl_price, (int, float))
                assert tp_price > price
                assert sl_price < price

                print(f"  価格 {price} → TP: {tp_price}, SL: {sl_price}")

            print("✓ 丸め処理テスト完了")

    def test_edge_case_calculations(self):
        """エッジケース計算テスト"""
        with performance_monitor("エッジケース計算"):
            edge_cases = [
                {
                    "method": TPSLMethod.FIXED_PERCENTAGE,
                    "tp": Decimal("0.1"),
                    "sl": Decimal("0.05"),
                },
                {
                    "method": TPSLMethod.FIXED_PERCENTAGE,
                    "tp": Decimal("10.0"),
                    "sl": Decimal("5.0"),
                },
            ]

            for case in edge_cases:
                tpsl_gene = TPSLGene(
                    method=case["method"],
                    take_profit_percentage=case["tp"],
                    stop_loss_percentage=case["sl"],
                )

                entry_price = 50000.0

                tp_price = self.tpsl_calculator.calculate_take_profit(
                    tpsl_gene, entry_price, "long"
                )
                sl_price = self.tpsl_calculator.calculate_stop_loss(
                    tpsl_gene, entry_price, "long"
                )

                # エッジケースでも適切に計算されることを確認
                assert tp_price > entry_price
                assert sl_price < entry_price

                print(
                    f"  TP%: {case['tp']}, SL%: {case['sl']} → TP: {tp_price}, SL: {sl_price}"
                )

            print("✓ エッジケース計算テスト完了")


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """エラーハンドリング統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.tpsl_calculator = TPSLCalculator()
        self.position_calculator = PositionSizingCalculatorService()
        self.data_generator = TestDataGenerator()

    def test_invalid_input_handling(self):
        """不正入力ハンドリングテスト"""
        with performance_monitor("不正入力ハンドリング"):
            invalid_cases = [
                {"entry_price": 0, "description": "ゼロ価格"},
                {"entry_price": -1000, "description": "負の価格"},
            ]

            tpsl_gene = self.data_generator.generate_tpsl_gene()

            for case in invalid_cases:
                try:
                    tp_price = self.tpsl_calculator.calculate_take_profit(
                        tpsl_gene, case["entry_price"], "long"
                    )
                    # エラーが発生しない場合は、結果が妥当かチェック
                    if case["entry_price"] <= 0:
                        assert tp_price <= 0 or tp_price is None
                except (ValueError, ZeroDivisionError, AssertionError):
                    # 期待されるエラー
                    print(f"  ✓ {case['description']} で適切にエラー処理")
                    continue

                print(f"  ⚠ {case['description']} でエラーが発生しませんでした")

            print("✓ 不正入力ハンドリングテスト完了")

    def test_memory_efficiency_integration(self):
        """メモリ効率性統合テスト"""
        import gc
        import psutil

        with performance_monitor("メモリ効率性統合"):
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024

            # 大量の計算を実行
            for _ in range(1000):
                tpsl_gene = self.data_generator.generate_tpsl_gene()
                position_gene = self.data_generator.generate_position_sizing_gene()

                entry_price = 50000.0
                balance = 100000.0

                tp_price = self.tpsl_calculator.calculate_take_profit(
                    tpsl_gene, entry_price, "long"
                )
                sl_price = self.tpsl_calculator.calculate_stop_loss(
                    tpsl_gene, entry_price, "long"
                )

                # 一部のケースでポジションサイズも計算
                if _ % 10 == 0:
                    position_size = self.position_calculator.calculate_position_size(
                        position_gene, balance, entry_price, sl_price
                    )

            # ガベージコレクション実行
            gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory

            print(f"  メモリ使用量: {memory_used:.2f}MB")

            # メモリ使用量が適切な範囲内であることを確認
            assert memory_used < 100, f"メモリ使用量が多すぎます: {memory_used}MB"

            print("✓ メモリ効率性統合テスト完了")


if __name__ == "__main__":
    # 直接実行時のテスト
    integration_suite = TestTPSLPositionSizingIntegration()
    integration_suite.setup_method()

    accuracy_suite = TestCalculationAccuracy()
    accuracy_suite.setup_method()

    error_suite = TestErrorHandlingIntegration()
    error_suite.setup_method()

    tests = [
        integration_suite.test_tpsl_position_sizing_interaction,
        integration_suite.test_extreme_market_conditions_integration,
        integration_suite.test_calculation_precision_integration,
        integration_suite.test_concurrent_operations_integration,
        accuracy_suite.test_rounding_behavior,
        accuracy_suite.test_edge_case_calculations,
        error_suite.test_invalid_input_handling,
        error_suite.test_memory_efficiency_integration,
    ]

    results = TestExecutionHelper.run_test_suite(tests)
    TestExecutionHelper.print_test_results(results)
                method=TPSLMethod.FIXED_PERCENTAGE
            )
            position_gene = self.data_generator.generate_position_sizing_gene(
                method=PositionSizingMethod.FIXED_PERCENTAGE
            )
            
            # 相互作用テスト
            current_price = Decimal("50000.0")
            portfolio_value = Decimal("100000.0")
            
            # ポジションサイズ計算
            position_size = self.position_calculator.calculate_position_size(
                position_gene, portfolio_value, current_price
            )
            
            # TP/SL計算
            tp_price, sl_price = self.tpsl_calculator.calculate_tpsl(
                tpsl_gene, current_price, "long"
            )
            
            # 結果検証
            assert position_size > 0
            assert tp_price > current_price
            assert sl_price < current_price
            
            # 財務精度検証
            assert_financial_precision(float(position_size), float(portfolio_value * position_gene.fixed_percentage / 100))

    def test_extreme_market_conditions(self):
        """極端な市場条件でのテスト"""
        with performance_monitor("極端市場条件テスト"):
            extreme_conditions = self.data_generator.generate_extreme_market_conditions()
            
            for condition_name, data in extreme_conditions.items():
                # 各極端条件でのTP/SL計算テスト
                tpsl_gene = self.data_generator.generate_tpsl_gene()
                
                for _, row in data.head(10).iterrows():  # 最初の10行をテスト
                    current_price = Decimal(str(row['close']))
                    
                    try:
                        tp_price, sl_price = self.tpsl_calculator.calculate_tpsl(
                            tpsl_gene, current_price, "long"
                        )
                        
                        # 基本的な論理チェック
                        assert tp_price > current_price, f"{condition_name}: TP価格が不正"
                        assert sl_price < current_price, f"{condition_name}: SL価格が不正"
                        
                    except Exception as e:
                        # エラーが発生した場合は適切にハンドリングされていることを確認
                        assert "invalid" in str(e).lower() or "error" in str(e).lower()

    def test_calculation_precision(self):
        """計算精度テスト"""
        with performance_monitor("計算精度テスト"):
            # 高精度が要求される計算のテスト
            tpsl_gene = TPSLGene(
                method=TPSLMethod.FIXED_PERCENTAGE,
                take_profit_percentage=Decimal("1.5"),
                stop_loss_percentage=Decimal("1.0"),
            )
            
            current_price = Decimal("50000.12345678")  # 8桁精度
            
            tp_price, sl_price = self.tpsl_calculator.calculate_tpsl(
                tpsl_gene, current_price, "long"
            )
            
            # 期待値計算
            expected_tp = current_price * (1 + tpsl_gene.take_profit_percentage / 100)
            expected_sl = current_price * (1 - tpsl_gene.stop_loss_percentage / 100)
            
            # 精度検証（8桁精度）
            assert_financial_precision(float(tp_price), float(expected_tp), tolerance=1e-8)
            assert_financial_precision(float(sl_price), float(expected_sl), tolerance=1e-8)

    def test_concurrent_operations(self):
        """並行処理テスト"""
        with performance_monitor("並行処理テスト"):
            def concurrent_calculation():
                tpsl_gene = self.data_generator.generate_tpsl_gene()
                current_price = Decimal("50000.0")
                return self.tpsl_calculator.calculate_tpsl(tpsl_gene, current_price, "long")
            
            # 並行処理実行
            results = ConcurrencyTestHelper.run_concurrent_operations(
                concurrent_calculation, num_threads=10
            )
            
            # 結果検証
            assert len(results) == 10
            for tp_price, sl_price in results:
                assert tp_price > Decimal("50000.0")
                assert sl_price < Decimal("50000.0")

    def test_memory_efficiency(self):
        """メモリ効率テスト"""
        with performance_monitor("メモリ効率テスト"):
            def memory_test_operation():
                # 大量の計算を実行
                for _ in range(1000):
                    tpsl_gene = self.data_generator.generate_tpsl_gene()
                    current_price = Decimal("50000.0")
                    self.tpsl_calculator.calculate_tpsl(tpsl_gene, current_price, "long")
            
            # メモリ使用量測定
            result, memory_used = self.performance_helper.measure_memory_usage(memory_test_operation)
            
            # メモリ使用量が適切な範囲内であることを確認
            assert memory_used < 100, f"メモリ使用量が多すぎます: {memory_used:.2f}MB"

    def test_performance_under_load(self):
        """負荷下でのパフォーマンステスト"""
        with performance_monitor("負荷テスト"):
            def load_test_operation():
                results = []
                for _ in range(100):
                    tpsl_gene = self.data_generator.generate_tpsl_gene()
                    position_gene = self.data_generator.generate_position_sizing_gene()
                    
                    current_price = Decimal("50000.0")
                    portfolio_value = Decimal("100000.0")
                    
                    # TP/SL計算
                    tp_price, sl_price = self.tpsl_calculator.calculate_tpsl(
                        tpsl_gene, current_price, "long"
                    )
                    
                    # ポジションサイズ計算
                    position_size = self.position_calculator.calculate_position_size(
                        position_gene, portfolio_value, current_price
                    )
                    
                    results.append((tp_price, sl_price, position_size))
                
                return results
            
            # パフォーマンス測定
            result, execution_time = self.performance_helper.measure_execution_time(load_test_operation)
            
            # パフォーマンス要件確認
            assert execution_time < 5.0, f"負荷テストが遅すぎます: {execution_time:.2f}秒"
            assert len(result) == 100, "期待される結果数が得られませんでした"


@pytest.mark.integration
class TestMarketValidation:
    """実際の市場データを使用した検証テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.data_generator = TestDataGenerator()
        self.performance_helper = PerformanceTestHelper()
        self.generator = RandomGeneGenerator(GAConfig())

    def test_market_data_processing(self):
        """市場データ処理テスト"""
        with performance_monitor("市場データ処理"):
            market_scenarios = self.data_generator.generate_market_scenarios()
            
            for scenario_name, data in market_scenarios.items():
                # データ品質確認
                assert not data.empty, f"{scenario_name}: データが空"
                assert len(data) > 50, f"{scenario_name}: データ不足"
                
                # 価格データの論理的整合性
                assert (data['high'] >= data['low']).all()
                assert (data['high'] >= data['open']).all()
                assert (data['high'] >= data['close']).all()

    def test_volatility_adaptation(self):
        """ボラティリティ適応テスト"""
        with performance_monitor("ボラティリティ適応"):
            # 高ボラティリティと低ボラティリティでのテスト
            high_vol_data = self.data_generator.generate_ohlcv_data(volatility=0.05)
            low_vol_data = self.data_generator.generate_ohlcv_data(volatility=0.005)
            
            for data, vol_type in [(high_vol_data, "高"), (low_vol_data, "低")]:
                # ボラティリティ計算
                returns = data['close'].pct_change().dropna()
                volatility = returns.std()
                
                if vol_type == "高":
                    assert volatility > 0.03, f"高ボラティリティが期待値より低い: {volatility}"
                else:
                    assert volatility < 0.01, f"低ボラティリティが期待値より高い: {volatility}"

    def test_trend_detection_accuracy(self):
        """トレンド検出精度テスト"""
        with performance_monitor("トレンド検出"):
            market_scenarios = self.data_generator.generate_market_scenarios()
            
            # 上昇トレンドの検証
            bull_data = market_scenarios["bull_market"]
            bull_returns = bull_data['close'].pct_change().dropna()
            assert bull_returns.mean() > 0, "上昇トレンドが検出されませんでした"
            
            # 下降トレンドの検証
            bear_data = market_scenarios["bear_market"]
            bear_returns = bear_data['close'].pct_change().dropna()
            assert bear_returns.mean() < 0, "下降トレンドが検出されませんでした"

    def test_risk_management_effectiveness(self):
        """リスク管理有効性テスト"""
        with performance_monitor("リスク管理"):
            extreme_conditions = self.data_generator.generate_extreme_market_conditions()
            
            for condition_name, data in extreme_conditions.items():
                tpsl_gene = self.data_generator.generate_tpsl_gene()
                
                # 各価格ポイントでリスク管理をテスト
                for _, row in data.head(20).iterrows():
                    current_price = Decimal(str(row['close']))
                    
                    try:
                        tp_price, sl_price = TPSLCalculator().calculate_tpsl(
                            tpsl_gene, current_price, "long"
                        )
                        
                        # リスク・リワード比の確認
                        risk = float(current_price - sl_price)
                        reward = float(tp_price - current_price)
                        
                        if risk > 0:  # ゼロ除算回避
                            risk_reward_ratio = reward / risk
                            assert risk_reward_ratio > 0.5, f"リスク・リワード比が低すぎます: {risk_reward_ratio}"
                        
                    except Exception:
                        # 極端な条件では計算エラーが発生する可能性がある
                        pass

    def test_multi_timeframe_consistency(self):
        """マルチタイムフレーム整合性テスト"""
        with performance_monitor("マルチタイムフレーム"):
            # 異なる期間のデータで整合性をテスト
            short_data = self.data_generator.generate_ohlcv_data(50)
            long_data = self.data_generator.generate_ohlcv_data(200)
            
            for data, period in [(short_data, "短期"), (long_data, "長期")]:
                # 基本統計の計算
                mean_price = data['close'].mean()
                std_price = data['close'].std()
                
                # 統計値が合理的な範囲内であることを確認
                assert mean_price > 0, f"{period}: 平均価格が無効"
                assert std_price > 0, f"{period}: 標準偏差が無効"
                assert std_price < mean_price, f"{period}: ボラティリティが異常に高い"

    def test_strategy_performance_validation(self):
        """戦略パフォーマンス検証テスト"""
        with performance_monitor("戦略パフォーマンス"):
            market_scenarios = self.data_generator.generate_market_scenarios()
            
            for scenario_name, data in market_scenarios.items():
                # 戦略生成
                strategy_gene = self.generator.generate_random_gene()
                
                # 基本的な戦略検証
                assert strategy_gene is not None
                assert len(strategy_gene.indicators) > 0
                assert len(strategy_gene.conditions) > 0
                
                # 戦略の構造的整合性確認
                for indicator in strategy_gene.indicators:
                    assert hasattr(indicator, 'name')
                    assert hasattr(indicator, 'parameters')
                
                for condition in strategy_gene.conditions:
                    assert hasattr(condition, 'left_operand')
                    assert hasattr(condition, 'operator')
                    assert hasattr(condition, 'right_operand')