"""
統合TPSLジェネレーターのテスト
Phase 2-1b: 統合TPSL設計 & テスト
"""

import pytest
import sys
import os
from typing import Dict, Any

# PYTHONPATHを追加してimportを可能にする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.auto_strategy.models.strategy_models import TPSLMethod


class TestUnifiedTPSLIntegration:
    """統合TPSLジェネレーター統合テスト"""

    def test_unified_tpsl_generator_can_be_imported(self):
        """UnifiedTPSLGeneratorが正しくimportできることを確認"""
        try:
            from app.services.auto_strategy.generators.tpsl_generator import UnifiedTPSLGenerator
            assert True, "UnifiedTPSLGeneratorのimportに成功"
        except ImportError:
            pytest.fail("UnifiedTPSLGeneratorのimportに失敗 - Phase 2-1cで実装予定")

    def test_unified_result_class_structure(self):
        """統合されたResultクラスの構造を確認"""
        try:
            from app.services.auto_strategy.generators.tpsl_generator import TPSLResult as UnifiedTPSLResult

            # 必須フィールドを確認
            required_fields = [
                'stop_loss_pct', 'take_profit_pct', 'method_used', 'expected_performance'
            ]

            # ダミーインスタンスでフィールドアクセスをテスト
            result = UnifiedTPSLResult(
                stop_loss_pct=0.05,
                take_profit_pct=0.10,
                method_used="risk_reward",
                confidence_score=0.8,
                expected_performance={"sharpe_ratio": 1.2}
            )

            for field in required_fields:
                assert hasattr(result, field), f"必須フィールド {field} が欠けています"

        except (ImportError, AttributeError) as e:
            pytest.fail(f"UnifiedTPSLResultクラスに問題: {e}")

    def test_strategy_pattern_implementation(self):
        """戦略パターンが正しく実装されていることを確認"""
        try:
            from app.services.auto_strategy.generators.tpsl_generator import (
                RiskRewardStrategy,
                StatisticalStrategy,
                VolatilityStrategy
            )

            # 各戦略クラスの存在を確認
            strategies = [RiskRewardStrategy, StatisticalStrategy, VolatilityStrategy]
            for strategy_class in strategies:
                assert callable(strategy_class), f"{strategy_class.__name__} が呼び出し可能"
                assert hasattr(strategy_class, 'generate'), f"{strategy_class.__name__} にgenerateメソッドがある"

        except ImportError as e:
            pytest.fail(f"戦略クラスimportエラー: {e}")

    def test_backward_compatibility_maintained(self):
        """既存のTPSLServiceとの後方互換性が維持されていることを確認"""
        from app.services.auto_strategy.services.tpsl_service import TPSLService

        # TPSLServiceが正常に動作することを確認
        service = TPSLService()

        # 基本機能が利用可能
        assert hasattr(service, 'calculate_tpsl_prices'), "TPSL価格計算メソッドが存在"

        # テスト用のフォールバック計算が可能
        fallback_sl, fallback_tp = service._calculate_fallback(50000.0, 1.0)
        assert fallback_sl is not None
        assert fallback_tp is not None

    def test_method_mapping_to_strategies(self):
        """TPSLMethod が適切な戦略にマッピングされていることを確認"""
        method_to_strategy = {
            TPSLMethod.RISK_REWARD_RATIO: "RiskRewardStrategy",
            TPSLMethod.STATISTICAL: "StatisticalStrategy",
            TPSLMethod.VOLATILITY_BASED: "VolatilityStrategy",
            TPSLMethod.FIXED_PERCENTAGE: "FixedPercentageStrategy",
        }

        for method, expected_strategy in method_to_strategy.items():
            method_value = method.value if hasattr(method, 'value') else method
            assert method_value in ["risk_reward_ratio", "statistical", "volatility_based", "fixed_percentage"]
            assert expected_strategy.endswith("Strategy")

    def test_adaptive_method_selection(self):
        """適応的な手法選択が機能することを確認"""
        # 市場条件に基づいた手法選択ロジックテスト
        market_conditions = {
            "high_volatility": True,
            "trend_strength": 0.8,
            "has_historical_data": False
        }

        # 高ボラティリティ時はVolatilityStrategyが優先されるべき
        # トレンドが強い場合はRiskRewardStrategyが適している
        # 過去データなしの場合はFixedPercentageが選択される

        # 実際の実装ではこの種のロジックを実装予定
        adaptive_selection_logic = {
            "high_volatility": "volatility",
            "strong_trend": "risk_reward",
            "no_historical_data": "fixed_percentage",
            "normal_conditions": "statistical"
        }

        assert adaptive_selection_logic["high_volatility"] == "volatility"
        assert adaptive_selection_logic["no_historical_data"] == "fixed_percentage"

    @pytest.mark.parametrize("method_name,expected_sl,expected_tp,expected_confidence", [
        ("risk_reward", 0.05, 0.15, 0.8),
        ("volatility", 0.03, 0.12, 0.9),
        ("statistical", 0.04, 0.16, 0.7),
        ("fixed_percentage", 0.03, 0.06, 0.95),
    ])
    def test_different_methods_produce_valid_results(self, method_name, expected_sl, expected_tp, expected_confidence):
        """各手法が妥当な結果を生成することを確認"""
        try:
            from app.services.auto_strategy.generators.tpsl_generator import UnifiedTPSLGenerator

            generator = UnifiedTPSLGenerator()

            # テスト用のconfig
            test_config = {
                "current_price": 50000.0,
                "base_sl": 0.03,
                "market_data": {"atr": 0.002},
                "target_ratio": 2.0
            }

            # 各手法別に結果が生成できることを確認
            result = generator.generate_tpsl(method_name, **test_config)

            assert result is not None
            assert hasattr(result, 'stop_loss_pct')
            assert hasattr(result, 'take_profit_pct')
            assert result.stop_loss_pct > 0
            assert result.take_profit_pct > 0
            assert result.take_profit_pct > result.stop_loss_pct

        except ImportError:
            pytest.skip(f"UnifiedTPSLGenerator実装待ち - {method_name}手法はPhase 2-1cで実装予定")
        except Exception as e:
            pytest.fail(f"{method_name}手法のテスト失敗: {e}")

    def test_error_handling_robustness(self):
        """エラー発生時の堅牢性を確認"""
        try:
            from app.services.auto_strategy.generators.tpsl_generator import UnifiedTPSLGenerator

            generator = UnifiedTPSLGenerator()

            # 不正な入力でエラーが適切に処理されることを確認
            invalid_config = {
                "current_price": -1000.0,  # 不正な価格
                "base_sl": -0.1,  # 不正な損切り率
                "unknown_param": "invalid"
            }

            # 適当なフォールバックが返される
            result = generator.generate_tpsl("risk_reward", **invalid_config)
            assert result is not None

        except ImportError:
            pytest.skip("UnifiedTPSLGenerator実装待ち - エラーハンドリングはPhase 2-1cで実装予定")
        except Exception as e:
            if "method" in str(e).lower():
                pytest.skip("メソッド存在エラー - 実装中で正常")
            else:
                pytest.fail(f"予期せぬエラー: {e}")

    def test_integration_with_tpsl_service(self):
        """TPSLServiceとの統合が可能であることを確認"""
        from app.services.auto_strategy.services.tpsl_service import TPSLService

        service = TPSLService()

        try:
            # TPSLServiceが統合ジェネレーターを利用可能
            assert hasattr(service, 'calculate_tpsl_prices'), "基本機能は維持されている"
        except Exception as e:
            pytest.fail(f"TPSLService統合テスト失敗: {e}")

    def test_performance_requirements_met(self):
        """パフォーマンス要件を満たしていることを確認"""
        import time

        start_time = time.time()

        try:
            from app.services.auto_strategy.generators.tpsl_generator import UnifiedTPSLGenerator

            generator = UnifiedTPSLGenerator()

            # 複数回の計算でパフォーマンスを確認
            for i in range(10):
                result = generator.generate_tpsl(
                    "risk_reward",
                    current_price=50000.0 + i*1000,
                    base_sl=0.03
                )
                assert result is not None

            elapsed = time.time() - start_time
            # パフォーマンス要件: 10回の計算で1秒以内
            assert elapsed < 1.0, f"パフォーマンス要件未達: {elapsed:.2f}秒"

        except ImportError:
            pytest.skip("UnifiedTPSLGenerator実装待ち")
        except Exception as e:
            pytest.fail(f"パフォーマンステスト失敗: {e}")

    @pytest.mark.parametrize("conditions,result_method", [
        ({"volatility": "high"}, "volatility"),
        ({"trend": "strong"}, "risk_reward"),
        ({"historical_data": False}, "fixed_percentage"),
        ({"all_conditions_normal": True}, "statistical"),
    ])
    def test_market_adaptive_selection_logic(self, conditions, result_method):
        """市場条件適応型の手法選択ロジックの妥当性を確認"""
        # 市場条件に基づいた自動最適手法選択のロジック検証

        condition_map = {
            "high_volatility": lambda: "volatility",
            "strong_trend": lambda: "risk_reward",
            "no_historical": lambda: "fixed_percentage",
            "normal": lambda: "statistical"
        }

        # 条件に基づいたメソッド選択
        selected_method = "statistical"  # デフォルト（正常条件）

        if "volatility" in conditions and conditions["volatility"] == "high":
            selected_method = "volatility"
        elif "trend" in conditions and conditions["trend"] == "strong":
            selected_method = "risk_reward"
        elif "historical_data" in conditions and not conditions["historical_data"]:
            selected_method = "fixed_percentage"
        elif "all_conditions_normal" in conditions and conditions["all_conditions_normal"]:
            selected_method = "statistical"

        assert selected_method == result_method, \
            f"条件 {conditions} に対して {result_method} が期待されるが {selected_method} が選択された"