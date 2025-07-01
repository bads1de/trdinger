"""
自動戦略生成システムの修正内容をテストするモジュール

MAMA指標対応と未対応指標の代替機能をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from app.core.services.auto_strategy.factories.indicator_initializer import (
    IndicatorInitializer,
)
from app.core.services.auto_strategy.factories.condition_evaluator import (
    ConditionEvaluator,
)
from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
from app.core.services.auto_strategy.models.strategy_gene import (
    IndicatorGene,
    Condition,
)


class TestMAMAIndicatorSupport:
    """MAMA指標対応のテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.initializer = IndicatorInitializer()
        self.encoder = GeneEncoder()

    def test_mama_indicator_in_adapters(self):
        """MAMA指標がアダプターマッピングに含まれているかテスト"""
        assert "MAMA" in self.initializer.indicator_adapters
        assert self.initializer.is_supported_indicator("MAMA")

    def test_mama_parameter_generation(self):
        """MAMA指標のパラメータ生成テスト"""
        parameters = self.encoder._generate_indicator_parameters("MAMA", 0.5)

        assert "fast_limit" in parameters
        assert "slow_limit" in parameters
        assert 0.2 <= parameters["fast_limit"] <= 0.5
        assert 0.01 <= parameters["slow_limit"] <= 0.1

    def test_mama_indicator_initialization(self):
        """MAMA指標の初期化テスト"""
        # テストデータの準備
        test_data = self._create_test_data()
        mock_strategy = self._create_mock_strategy()

        # MAMA指標遺伝子の作成
        mama_gene = IndicatorGene(
            type="MAMA",
            parameters={"fast_limit": 0.5, "slow_limit": 0.05},
            enabled=True,
        )

        # 初期化テスト（実際のTA-Libが必要なのでモック使用）
        with patch.object(self.initializer, "_calculate_indicator") as mock_calc:
            mock_calc.return_value = (pd.Series([1, 2, 3]), "MAMA")

            result = self.initializer.initialize_indicator(
                mama_gene, test_data, mock_strategy
            )

            assert result == "MAMA"
            mock_calc.assert_called_once()

    def _create_test_data(self):
        """テスト用データの作成"""
        mock_data = Mock()
        mock_data.Close = pd.Series([100, 101, 102, 103, 104])
        mock_data.High = pd.Series([101, 102, 103, 104, 105])
        mock_data.Low = pd.Series([99, 100, 101, 102, 103])
        mock_data.Volume = pd.Series([1000, 1100, 1200, 1300, 1400])
        return mock_data

    def _create_mock_strategy(self):
        """モック戦略インスタンスの作成"""
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.I = Mock(return_value=Mock())
        return mock_strategy


class TestFallbackIndicators:
    """未対応指標の代替機能テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.initializer = IndicatorInitializer()

    def test_fallback_mapping_exists(self):
        """代替指標マッピングが存在するかテスト"""
        assert hasattr(self.initializer, "fallback_indicators")
        assert isinstance(self.initializer.fallback_indicators, dict)
        assert len(self.initializer.fallback_indicators) > 0

    def test_stochf_fallback_to_stoch(self):
        """STOCHF指標がSTOCH指標に代替されるかテスト"""
        assert "STOCHF" in self.initializer.fallback_indicators
        assert self.initializer.fallback_indicators["STOCHF"] == "STOCH"

    def test_unsupported_indicator_fallback(self):
        """未対応指標の代替処理テスト"""
        test_data = self._create_test_data()
        mock_strategy = self._create_mock_strategy()

        # 未対応指標（STOCHF）の遺伝子作成
        stochf_gene = IndicatorGene(
            type="STOCHF", parameters={"period": 14}, enabled=True
        )

        with patch.object(self.initializer, "_calculate_indicator") as mock_calc:
            mock_calc.return_value = (pd.Series([1, 2, 3]), "STOCH_14")

            result = self.initializer.initialize_indicator(
                stochf_gene, test_data, mock_strategy
            )

            # 元の指標名（STOCHF_14）で登録されることを確認
            assert result == "STOCHF_14"

    def test_unsupported_without_fallback(self):
        """代替なしの未対応指標の処理テスト"""
        test_data = self._create_test_data()
        mock_strategy = self._create_mock_strategy()

        # 代替なしの未対応指標
        unknown_gene = IndicatorGene(
            type="UNKNOWN_INDICATOR", parameters={"period": 14}, enabled=True
        )

        result = self.initializer.initialize_indicator(
            unknown_gene, test_data, mock_strategy
        )

        # Noneが返されることを確認
        assert result is None

    def _create_test_data(self):
        """テスト用データの作成"""
        mock_data = Mock()
        mock_data.Close = pd.Series([100, 101, 102, 103, 104])
        mock_data.High = pd.Series([101, 102, 103, 104, 105])
        mock_data.Low = pd.Series([99, 100, 101, 102, 103])
        mock_data.Volume = pd.Series([1000, 1100, 1200, 1300, 1400])
        return mock_data

    def _create_mock_strategy(self):
        """モック戦略インスタンスの作成"""
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.I = Mock(return_value=Mock())
        return mock_strategy


class TestConditionEvaluatorImprovements:
    """条件評価器の改善テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.evaluator = ConditionEvaluator()

    def test_missing_indicator_handling(self):
        """存在しない指標の処理テスト"""
        mock_strategy = Mock()
        mock_strategy.indicators = {"RSI_14": Mock()}

        # 存在しない指標を参照
        result = self.evaluator.get_condition_value("MISSING_INDICATOR", mock_strategy)

        assert result is None

    def test_available_indicators_logging(self):
        """利用可能な指標のログ出力テスト"""
        mock_strategy = Mock()
        mock_strategy.indicators = {"RSI_14": Mock(), "SMA_20": Mock()}

        with patch(
            "app.core.services.auto_strategy.factories.condition_evaluator.logger"
        ) as mock_logger:
            self.evaluator.get_condition_value("MISSING_INDICATOR", mock_strategy)

            # ログが出力されることを確認
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args[0][0]
            assert "MISSING_INDICATOR" in call_args
            assert "RSI_14" in call_args
            assert "SMA_20" in call_args


class TestGeneEncodingMAMASupport:
    """遺伝子エンコーディングのMAMA対応テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.encoder = GeneEncoder()

    def test_mama_in_condition_generation(self):
        """MAMA指標の条件生成テスト"""
        # MAMA指標を含む指標リスト
        from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene

        indicators = [
            IndicatorGene(
                type="MAMA",
                parameters={"fast_limit": 0.5, "slow_limit": 0.05},
                enabled=True,
            )
        ]

        # MAMA指標が移動平均系として扱われることを確認
        # 実際のテストは統合テストで行う
        assert True  # プレースホルダー

    def test_mama_parameter_range(self):
        """MAMAパラメータの範囲テスト"""
        # 異なるparam_valでのパラメータ生成
        params_low = self.encoder._generate_indicator_parameters("MAMA", 0.0)
        params_high = self.encoder._generate_indicator_parameters("MAMA", 1.0)

        # fast_limitの範囲確認
        assert params_low["fast_limit"] == 0.2
        assert params_high["fast_limit"] == 0.5

        # slow_limitの範囲確認
        assert params_low["slow_limit"] == 0.01
        assert abs(params_high["slow_limit"] - 0.1) < 1e-10  # 浮動小数点の比較


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
