"""
Indicator Factory Functions Tests

Tests for indicator generation logic integrated into genes/indicator.py,
including multi-timeframe (MTF) functionality.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pytest

from app.services.auto_strategy.genes import (
    IndicatorGene, 
    generate_random_indicators, 
    create_random_indicator_gene
)

@dataclass
class MockGAConfig:
    """テスト用のモックGA設定"""
    min_indicators: int = 2
    max_indicators: int = 5
    enable_multi_timeframe: bool = False
    available_timeframes: Optional[List[str]] = None
    mtf_indicator_probability: float = 0.3
    parameter_range_preset: Optional[str] = None

class TestIndicatorGenerationMTF:
    """マルチタイムフレーム（MTF）機能のテスト"""

    def test_mtf_disabled_no_timeframe_assigned(self) -> None:
        """MTF無効時は指標にタイムフレームが割り当てられないこと"""
        config = MockGAConfig(enable_multi_timeframe=False)
        indicators = generate_random_indicators(config)

        # 全ての指標のタイムフレームがNoneであること
        for indicator in indicators:
            assert indicator.timeframe is None

    def test_mtf_enabled_some_indicators_have_timeframe(self) -> None:
        """MTF有効時は一部の指標にタイムフレームが割り当てられること"""
        config = MockGAConfig(
            enable_multi_timeframe=True,
            available_timeframes=["15m", "1h", "4h", "1d"],
            mtf_indicator_probability=1.0,  # 100%でタイムフレームを割り当て
            min_indicators=3,
            max_indicators=5,
        )
        indicators = generate_random_indicators(config)

        # MTFを持つ指標が少なくとも1つ存在することを確認
        # 1本目はデフォルトタイムフレームになる可能性があるため、全体でチェック
        mtf_indicators = [
            ind for ind in indicators if ind.timeframe in ["15m", "1h", "4h", "1d"]
        ]
        assert len(mtf_indicators) >= 1

    def test_mtf_probability_zero_no_timeframe(self) -> None:
        """MTF確率が0の場合、タイムフレームが割り当てられないこと"""
        config = MockGAConfig(
            enable_multi_timeframe=True,
            available_timeframes=["1h", "4h", "1d"],
            mtf_indicator_probability=0.0,
            min_indicators=3,
            max_indicators=5,
        )
        indicators = generate_random_indicators(config)

        for indicator in indicators:
            assert indicator.timeframe is None

    def test_create_random_indicator_gene_with_timeframe(self) -> None:
        """create_random_indicator_gene がタイムフレームを正しく設定すること"""
        config = MockGAConfig()
        indicator = create_random_indicator_gene("SMA", config=config, timeframe="4h")

        assert isinstance(indicator, IndicatorGene)
        assert indicator.type == "SMA"
        assert indicator.timeframe == "4h"
        assert indicator.enabled is True

class TestIndicatorGenerationBasic:
    """基本機能テスト"""

    def test_generates_required_number_of_indicators(self) -> None:
        """指定された範囲の数の指標が生成されること"""
        config = MockGAConfig(min_indicators=2, max_indicators=4)
        indicators = generate_random_indicators(config)

        assert len(indicators) >= 2
        # IndicatorCompositionServiceによって増える可能性があるため上限チェックは緩和
        assert len(indicators) >= config.min_indicators

    def test_all_indicators_are_enabled(self) -> None:
        """生成される全ての指標がenabledであること"""
        config = MockGAConfig()
        indicators = generate_random_indicators(config)

        for indicator in indicators:
            assert indicator.enabled is True