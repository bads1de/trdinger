"""
IndicatorCompositionService のテスト
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from app.services.auto_strategy.generators.indicator_composition_service import (
    IndicatorCompositionService,
)
from app.services.auto_strategy.models.strategy_models import IndicatorGene
from app.services.auto_strategy.config.constants import MOVING_AVERAGE_INDICATORS


class TestIndicatorCompositionService:
    """IndicatorCompositionService のテストクラス"""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.max_indicators = 5
        return config

    @pytest.fixture
    def service(self, mock_config):
        return IndicatorCompositionService(config=mock_config)

    def test_enhance_with_trend_indicators_no_change(self, service):
        """トレンド指標強制追加が削除されたことの確認"""
        indicators = []
        available = ["SMA", "RSI"]
        result = service.enhance_with_trend_indicators(indicators, available)
        assert result == []

    @patch("random.random")
    @patch("random.sample")
    def test_enhance_with_ma_cross_strategy(
        self, mock_sample, mock_random, service, mock_config
    ):
        """MAクロス戦略追加のテスト"""
        # 確率ヒットさせる
        mock_random.return_value = 0.1  # < 0.25

        # 既存指標: SMA (period=20) 1つ
        existing_indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]

        available_indicators = ["SMA", "EMA", "RSI"]

        # random.sample のモック: 利用可能なMAを返す
        # _choose_ma_with_unique_period 内で random.sample(candidates) が呼ばれる
        # candidates -> preferential logic or pool. "SMA" is usually preferred.
        # Let's say it picks "EMA"
        mock_sample.side_effect = lambda pop, k: ["EMA"] if "EMA" in pop else pop

        # 実行
        result = service.enhance_with_ma_cross_strategy(
            existing_indicators, available_indicators
        )

        # 結果検証: 元のSMA + 新しいMA が増えているはず
        assert len(result) == 2
        assert result[0].type == "SMA"
        assert result[1].type == "EMA"  # sampleで返させたもの

    @patch("random.random")
    def test_enhance_with_ma_cross_strategy_limit_reached(
        self, mock_random, service, mock_config
    ):
        """MAクロス追加時の上限チェックテスト"""
        mock_random.return_value = 0.1
        mock_config.max_indicators = 2

        # 既に非MAで埋まっている
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 20}, enabled=True),
        ]
        available = ["SMA"]

        # `_choose_ma_with_unique_period` will likely pick SMA
        # We need to mock correct behavior for `_remove_non_ma_indicator` to be triggered

        # 実行
        result = service.enhance_with_ma_cross_strategy(indicators, available)

        # RSI 2つ + SMA 1つ = 3つ -> max 2 なので RSIが1つ消されて SMAが残るはず
        # Result should be length 2 (or kept at max)
        # Note: logic says `if len > max: remove_non_ma`.
        # So 2 -> 3 -> remove 1 -> 2.
        assert len(result) == 2
        # Check that we have at least one MA now
        types = [ind.type for ind in result]
        assert "SMA" in types
        # One RSI should remain
        assert "RSI" in types
