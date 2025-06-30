"""
パラメータ管理システム統合テスト

新しいIndicatorParameterManagerシステムと既存システムの統合をテストします。
"""

import pytest
from unittest.mock import patch, Mock

from app.core.services.indicators.parameter_manager import IndicatorParameterManager
from app.core.services.indicators.config.indicator_config import (
    IndicatorConfig,
    ParameterConfig,
    IndicatorResultType,
    indicator_registry,
)
from app.core.services.auto_strategy.utils.parameter_generators import (
    generate_indicator_parameters,
)
from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder


class TestParameterSystemIntegration:
    """パラメータ管理システム統合テストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.manager = IndicatorParameterManager()
        self.encoder = GeneEncoder()

    def test_end_to_end_parameter_generation(self):
        """エンドツーエンドのパラメータ生成テスト"""
        # RSI設定を作成してレジストリに登録
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(
                name="period",
                default_value=14,
                min_value=5,
                max_value=50,
                description="RSI計算期間",
            )
        )

        # 一時的にレジストリに登録
        original_config = indicator_registry.get_config("RSI")
        indicator_registry.register(rsi_config)

        try:
            # 1. IndicatorParameterManagerによる直接生成
            direct_params = self.manager.generate_parameters("RSI", rsi_config)
            assert "period" in direct_params
            assert 5 <= direct_params["period"] <= 50

            # 2. generate_indicator_parameters関数による生成
            function_params = generate_indicator_parameters("RSI")
            assert "period" in function_params
            assert 5 <= function_params["period"] <= 50

            # 3. GeneEncoderによる生成（フォールバック）
            encoder_params = self.encoder._generate_indicator_parameters("RSI", 0.5)
            assert "period" in encoder_params
            assert isinstance(encoder_params["period"], int)

        finally:
            # レジストリを元に戻す
            if original_config:
                indicator_registry.register(original_config)

    def test_parameter_consistency_across_systems(self):
        """システム間でのパラメータ一貫性テスト"""
        # MACD設定を作成
        macd_config = IndicatorConfig(
            indicator_name="MACD",
            required_data=["close"],
            result_type=IndicatorResultType.COMPLEX,
        )
        macd_config.add_parameter(
            ParameterConfig(name="fast_period", default_value=12, min_value=5, max_value=20)
        )
        macd_config.add_parameter(
            ParameterConfig(name="slow_period", default_value=26, min_value=20, max_value=50)
        )
        macd_config.add_parameter(
            ParameterConfig(name="signal_period", default_value=9, min_value=5, max_value=15)
        )

        # 新システムでの生成
        new_params = self.manager.generate_parameters("MACD", macd_config)

        # 従来システムでの生成（フォールバック）
        with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
            mock_registry.get_config.return_value = None
            legacy_params = generate_indicator_parameters("MACD")

        # 両方とも必要なパラメータを持っていることを確認
        required_params = ["fast_period", "slow_period", "signal_period"]
        for param in required_params:
            assert param in new_params
            assert param in legacy_params

        # 新システムの制約が守られていることを確認
        assert new_params["fast_period"] < new_params["slow_period"]
        assert 5 <= new_params["fast_period"] <= 20
        assert 20 <= new_params["slow_period"] <= 50
        assert 5 <= new_params["signal_period"] <= 15

    def test_backward_compatibility_with_auto_strategy(self):
        """オートストラテジーとの後方互換性テスト"""
        # オートストラテジーで使用される10個の指標をテスト
        auto_strategy_indicators = [
            "SMA", "EMA", "RSI", "CCI", "ADX", "ATR", "MACD", "BB", "STOCH", "OBV"
        ]

        for indicator_type in auto_strategy_indicators:
            # パラメータ生成が成功することを確認
            params = generate_indicator_parameters(indicator_type)
            assert isinstance(params, dict)

            # OBV以外は何らかのパラメータを持つことを確認
            if indicator_type == "OBV":
                assert params == {}
            else:
                assert len(params) > 0

    def test_error_handling_and_fallback(self):
        """エラーハンドリングとフォールバック機能のテスト"""
        # 存在しない指標での生成
        params = generate_indicator_parameters("UNKNOWN_INDICATOR")
        assert isinstance(params, dict)
        # フォールバックで期間パラメータが生成される
        assert "period" in params

        # GeneEncoderでのエラーハンドリング
        encoder_params = self.encoder._generate_indicator_parameters("UNKNOWN_INDICATOR", 0.5)
        assert isinstance(encoder_params, dict)
        # デフォルト値が返される
        assert "period" in encoder_params

    def test_parameter_validation_integration(self):
        """パラメータバリデーション統合テスト"""
        # RSI設定を作成
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(
                name="period",
                default_value=14,
                min_value=2,
                max_value=100,
                description="RSI計算期間",
            )
        )

        # 複数回パラメータを生成してすべて有効であることを確認
        for _ in range(20):
            params = self.manager.generate_parameters("RSI", rsi_config)
            assert self.manager.validate_parameters("RSI", params, rsi_config)
            assert rsi_config.validate_parameters(params)

    def test_performance_and_memory_usage(self):
        """パフォーマンスとメモリ使用量のテスト"""
        import time
        import gc

        # 大量のパラメータ生成でのパフォーマンステスト
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(name="period", default_value=14, min_value=2, max_value=100)
        )

        start_time = time.time()
        
        # 1000回のパラメータ生成
        for _ in range(1000):
            params = self.manager.generate_parameters("RSI", rsi_config)
            assert "period" in params

        end_time = time.time()
        generation_time = end_time - start_time

        # パフォーマンス要件: 1000回の生成が1秒以内
        assert generation_time < 1.0, f"Parameter generation too slow: {generation_time:.3f}s"

        # メモリクリーンアップ
        gc.collect()

    def test_registry_integration(self):
        """レジストリ統合テスト"""
        # レジストリの基本機能をテスト
        test_config = IndicatorConfig(
            indicator_name="TEST_INDICATOR",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        test_config.add_parameter(
            ParameterConfig(name="test_param", default_value=10, min_value=1, max_value=20)
        )

        # 登録
        indicator_registry.register(test_config)

        # 取得
        retrieved_config = indicator_registry.get_config("TEST_INDICATOR")
        assert retrieved_config is not None
        assert retrieved_config.indicator_name == "TEST_INDICATOR"

        # パラメータ生成
        params = self.manager.generate_parameters("TEST_INDICATOR", retrieved_config)
        assert "test_param" in params
        assert 1 <= params["test_param"] <= 20

    def test_json_config_integration(self):
        """JSON設定統合テスト"""
        # IndicatorConfigのJSON機能をテスト
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(name="period", default_value=14, min_value=2, max_value=100)
        )

        # JSON変換
        json_str = rsi_config.to_json()
        assert isinstance(json_str, str)
        assert "RSI" in json_str

        # JSON復元
        restored_config = IndicatorConfig.from_json(json_str)
        assert restored_config.indicator_name == "RSI"
        assert "period" in restored_config.parameters

        # 復元された設定でのパラメータ生成
        params = self.manager.generate_parameters("RSI", restored_config)
        assert "period" in params
        assert 2 <= params["period"] <= 100
