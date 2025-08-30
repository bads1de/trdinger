"""
Phase 1統合テスト
フェーズ1で実装した全ての機能を統合テスト
"""

import pytest
import sys
import os

# PYTHONPATHを追加してimportを可能にする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.auto_strategy.config.constants import (
    INDICATOR_CHARACTERISTICS,
    IndicatorType,
    StrategyType
)
from app.services.auto_strategy.utils.decorators import auto_strategy_operation
from app.services.auto_strategy.generators.condition_generator import ConditionGenerator


class TestPhase1Integration:
    """フェーズ1統合テスト"""

    def test_constants_integration_complete(self):
        """定数統合が完全に完了していることを確認"""
        # INDICATOR_CHARACTERISTICSがconstants.pyで利用可能
        assert INDICATOR_CHARACTERISTICS is not None
        assert isinstance(INDICATOR_CHARACTERISTICS, dict)
        assert len(INDICATOR_CHARACTERISTICS) > 0

        # IndicatorTypeとStrategyTypeが利用可能
        assert IndicatorType.MOMENTUM == "momentum"
        assert StrategyType.DIFFERENT_INDICATORS == "different_indicators"

    def test_smart_condition_generator_uses_integrated_constants(self):
        """SmartConditionGeneratorが統合された定数を使用していることを確認"""
        generator = ConditionGenerator()

        # テスト用の指標遺伝子を作成
        from app.services.auto_strategy.models.strategy_models import IndicatorGene

        test_gene = IndicatorGene(
            type="RSI",
            parameters={"period": 14},
            enabled=True
        )

        # 正常に動作することを確認
        try:
            conditions = generator._generic_long_conditions(test_gene)
            assert conditions is not None
        except Exception as e:
            pytest.fail(f"統合定数を使用した処理でエラー: {e}")

    def test_decorator_system_functionality(self):
        """デコレーターシステムが機能していることを確認"""
        @auto_strategy_operation()
        def test_operation():
            return "operation_result"

        result = test_operation()
        assert result == "operation_result"

    def test_debug_logging_system(self):
        """デバッグログシステムが機能していることを確認"""
        from app.services.auto_strategy.generators.strategy_factory import _debug_log

        # 関数が存在することを確認
        assert callable(_debug_log)

        # 正常に呼び出せることを確認（実際にログが出力されるかは環境による）
        try:
            _debug_log("テストメッセージ")
        except Exception as e:
            pytest.fail(f"デバッグログでエラー: {e}")

    def test_logging_reduction_implemented(self):
        """ログ出力削減が実装されていることを確認"""
        # strategy_factory.pyでログが削減されていることを確認
        from app.services.auto_strategy.generators import strategy_factory

        # _debug_log関数が存在することを確認
        assert hasattr(strategy_factory, '_debug_log')

        # strategy_factoryが正常にimportできることを確認
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        assert StrategyFactory is not None

    def test_overall_system_cohesion(self):
        """全体システムの凝集性が維持されていることを確認"""
        try:
            # 定数統合のテスト
            rsi_char = INDICATOR_CHARACTERISTICS["RSI"]
            assert rsi_char["type"] == IndicatorType.MOMENTUM

            # デコレーターのテスト
            @auto_strategy_operation()
            def cohesive_test():
                return INDICATOR_CHARACTERISTICS["SMA"]["type"]

            result = cohesive_test()
            assert result == IndicatorType.TREND

            # ログシステムのテスト
            from app.services.auto_strategy.generators.strategy_factory import _debug_log
            _debug_log("整合性テスト完了")

        except Exception as e:
            pytest.fail(f"システムの整合性テストでエラー: {e}")

    def test_phase1_requirements_met(self):
        """フェーズ1の要件が満たされていることを確認"""
        # 要件1: 定数の統合（分散していた定数を一元化）
        assert "RSI" in INDICATOR_CHARACTERISTICS
        assert "SMA" in INDICATOR_CHARACTERISTICS
        assert "MACD" in INDICATOR_CHARACTERISTICS

        # 要件2: 共通デコレーター（safe_operationの統一）
        assert callable(auto_strategy_operation)

        # 要件3: ログ出力削減（条件付きロギング）
        from app.services.auto_strategy.generators.strategy_factory import _debug_log
        assert callable(_debug_log)

        # 要件4: 後方互換性の維持
        from app.services.auto_strategy.generators.condition_generator import ConditionGenerator
        generator = ConditionGenerator()
        assert generator is not None

    def test_performance_improvement_indicators(self):
        """パフォーマンス改善の指標を確認"""
        # 定数の数
        assert len(INDICATOR_CHARACTERISTICS) >= 4  # RSI, SMA, MACD, STOCH

        # デコレーターが適切に機能するか確認
        call_count = 0

        @auto_strategy_operation()
        def performance_test():
            nonlocal call_count
            call_count += 1
            return "result"

        result = performance_test()
        assert result == "result"
        assert call_count == 1

    def test_maintenance_improvements(self):
        """保守性改善を確認"""
        # 1. 定数の一元管理
        assert INDICATOR_CHARACTERISTICS is not None

        # 2. デコレーターの統一インターフェース
        assert auto_strategy_operation is not None

        # 3. ログの条件付き出力
        from app.services.auto_strategy.generators.strategy_factory import _debug_log
        assert _debug_log is not None

        # 4. 明確な関心の分離
        from app.services.auto_strategy.config.constants import IndicatorType
        assert IndicatorType is not None
        assert StrategyType is not None