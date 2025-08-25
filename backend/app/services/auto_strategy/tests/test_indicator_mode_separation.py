"""
テスト: Auto Strategyの指標モード分離機能

テクニカル指標とML指標が完全に分離された指標モード機能を検証します。
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.gene_strategy import IndicatorGene


class TestIndicatorModeSeparation(unittest.TestCase):
    """指標モード分離機能のテスト"""

    def setUp(self):
        """テスト前処理"""
        # テスト用の基本データフレームを作成
        self.test_df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

    @patch(
        "app.services.auto_strategy.generators.random_gene_generator.TechnicalIndicatorService"
    )
    def test_technical_only_mode(self, mock_indicator_service):
        """テクニカルオンリー指標モードのテスト"""
        # モックの設定
        mock_indicator_service.return_value.get_supported_indicators.return_value = {
            "SMA": {},
            "RSI": {},
            "MACD": {},
        }

        # GA設定の作成（テクニカルオンリー）
        config = GAConfig(indicator_mode="technical_only")

        # ランダム遺伝子生成器の作成
        generator = RandomGeneGenerator(config)

        # 利用可能な指標を確認
        available_indicators = generator.available_indicators

        # テクニカル指標のみが含まれていることを確認
        expected_technical_indicators = {"SMA", "RSI", "MACD"}
        self.assertTrue(
            set(available_indicators).issubset(expected_technical_indicators),
            f"テクニカル指標モードにML指標が含まれています: {available_indicators}",
        )

        # ML指標が含まれていないことを確認
        ml_indicators = {"ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"}
        self.assertFalse(
            any(indicator in available_indicators for indicator in ml_indicators),
            f"テクニカル指標モードにML指標が含まれています: {available_indicators}",
        )

    @patch(
        "app.services.auto_strategy.generators.random_gene_generator.TechnicalIndicatorService"
    )
    def test_ml_only_mode(self, mock_indicator_service):
        """MLオンリー指標モードのテスト"""
        # モックの設定
        mock_indicator_service.return_value.get_supported_indicators.return_value = {
            "SMA": {},
            "RSI": {},
            "MACD": {},
        }

        # GA設定の作成（MLオンリー）
        config = GAConfig(indicator_mode="ml_only")

        # ランダム遺伝子生成器の作成
        generator = RandomGeneGenerator(config)

        # 利用可能な指標を確認
        available_indicators = generator.available_indicators

        # ML指標のみが含まれていることを確認
        expected_ml_indicators = {"ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"}
        self.assertTrue(
            set(available_indicators).issubset(expected_ml_indicators),
            f"ML指標モードにテクニカル指標が含まれています: {available_indicators}",
        )

        # テクニカル指標が含まれていないことを確認
        technical_indicators = {"SMA", "RSI", "MACD"}
        self.assertFalse(
            any(
                indicator in available_indicators for indicator in technical_indicators
            ),
            f"ML指標モードにテクニカル指標が含まれています: {available_indicators}",
        )

    @patch(
        "app.services.auto_strategy.generators.random_gene_generator.TechnicalIndicatorService"
    )
    @patch(
        "app.services.auto_strategy.generators.random_gene_generator.indicator_registry"
    )
    def test_gene_generation_technical_only(
        self, mock_indicator_registry, mock_indicator_service
    ):
        """テクニカルオンリーでの遺伝子生成テスト"""
        # モックの設定
        mock_indicator_service.return_value.get_supported_indicators.return_value = {
            "SMA": {},
            "RSI": {},
        }
        mock_indicator_registry.generate_parameters_for_indicator.return_value = {
            "period": 14
        }

        # GA設定の作成（テクニカルオンリー）
        config = GAConfig(indicator_mode="technical_only", max_indicators=2)

        # ランダム遺伝子生成器の作成
        generator = RandomGeneGenerator(config)

        # 戦略遺伝子の生成
        strategy_gene = generator.generate_random_gene()

        # 生成された指標を確認
        generated_indicators = [
            indicator.type for indicator in strategy_gene.indicators
        ]

        # テクニカル指標のみが生成されていることを確認
        technical_indicators = {"SMA", "RSI"}
        self.assertTrue(
            all(
                indicator in technical_indicators for indicator in generated_indicators
            ),
            f"テクニカルオンリーでML指標が生成されました: {generated_indicators}",
        )

    @patch(
        "app.services.auto_strategy.generators.random_gene_generator.TechnicalIndicatorService"
    )
    @patch(
        "app.services.auto_strategy.generators.random_gene_generator.indicator_registry"
    )
    def test_gene_generation_ml_only(
        self, mock_indicator_registry, mock_indicator_service
    ):
        """MLオンリーでの遺伝子生成テスト"""
        # モックの設定
        mock_indicator_service.return_value.get_supported_indicators.return_value = {
            "SMA": {},
            "RSI": {},
        }
        mock_indicator_registry.generate_parameters_for_indicator.return_value = {
            "period": 14
        }

        # GA設定の作成（MLオンリー）
        config = GAConfig(indicator_mode="ml_only", max_indicators=2)

        # ランダム遺伝子生成器の作成
        generator = RandomGeneGenerator(config)

        # 戦略遺伝子の生成
        strategy_gene = generator.generate_random_gene()

        # 生成された指標を確認
        generated_indicators = [
            indicator.type for indicator in strategy_gene.indicators
        ]

        # ML指標のみが生成されていることを確認
        ml_indicators = {"ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"}
        self.assertTrue(
            all(indicator in ml_indicators for indicator in generated_indicators),
            f"MLオンリーでテクニカル指標が生成されました: {generated_indicators}",
        )


if __name__ == "__main__":
    unittest.main()
