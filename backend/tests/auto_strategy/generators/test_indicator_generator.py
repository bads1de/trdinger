"""
IndicatorGenerator テスト

マルチタイムフレーム（MTF）機能を含む指標生成のテスト
"""

from dataclasses import dataclass
from typing import List, Optional


from app.services.auto_strategy.generators.random.indicator_generator import (
    IndicatorGenerator,
)
from app.services.auto_strategy.models import IndicatorGene


@dataclass
class MockGAConfig:
    """テスト用のモックGA設定"""

    min_indicators: int = 2
    max_indicators: int = 5
    enable_multi_timeframe: bool = False
    available_timeframes: Optional[List[str]] = None
    mtf_indicator_probability: float = 0.3


class TestIndicatorGeneratorMTF:
    """マルチタイムフレーム（MTF）機能のテスト"""

    def test_mtf_disabled_no_timeframe_assigned(self) -> None:
        """MTF無効時は指標にタイムフレームが割り当てられないこと"""
        config = MockGAConfig(enable_multi_timeframe=False)
        generator = IndicatorGenerator(config)

        indicators = generator.generate_random_indicators()

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
        generator = IndicatorGenerator(config)

        indicators = generator.generate_random_indicators()

        # MTFを持つ指標が少なくとも1つ存在することを確認
        # 注: enhance_with_* メソッドで追加された指標はデフォルトタイムフレームを持つ
        assert len(indicators) >= 2
        mtf_indicators = [
            ind for ind in indicators if ind.timeframe in ["15m", "1h", "4h", "1d"]
        ]
        assert len(mtf_indicators) >= 1, (
            f"少なくとも1つはMTFタイムフレームを持つべき。"
            f"指標: {[(ind.type, ind.timeframe) for ind in indicators]}"
        )

    def test_mtf_enabled_first_indicator_uses_default_timeframe(self) -> None:
        """MTF有効時でも少なくとも1つはデフォルトタイムフレーム（None）を使用"""
        config = MockGAConfig(
            enable_multi_timeframe=True,
            available_timeframes=["1h", "4h", "1d"],
            mtf_indicator_probability=1.0,
            min_indicators=2,
            max_indicators=2,
        )
        generator = IndicatorGenerator(config)

        indicators = generator.generate_random_indicators()

        # 少なくとも1つはデフォルトタイムフレーム（None）を持つ
        # 注: enhance_with_* メソッドで追加された指標もデフォルトタイムフレームを持つ
        default_tf_indicators = [ind for ind in indicators if ind.timeframe is None]
        assert len(default_tf_indicators) >= 1, (
            f"少なくとも1つはデフォルトタイムフレームを持つべき。"
            f"指標: {[(ind.type, ind.timeframe) for ind in indicators]}"
        )

    def test_mtf_probability_zero_no_timeframe(self) -> None:
        """MTF確率が0の場合、タイムフレームが割り当てられないこと"""
        config = MockGAConfig(
            enable_multi_timeframe=True,
            available_timeframes=["1h", "4h", "1d"],
            mtf_indicator_probability=0.0,  # 0%
            min_indicators=3,
            max_indicators=5,
        )
        generator = IndicatorGenerator(config)

        indicators = generator.generate_random_indicators()

        # 全ての指標のタイムフレームがNoneであること
        for indicator in indicators:
            assert indicator.timeframe is None

    def test_available_timeframes_custom(self) -> None:
        """カスタムタイムフレームリストが正しく使用されること"""
        config = MockGAConfig(
            enable_multi_timeframe=True,
            available_timeframes=["5m", "15m"],  # カスタム
            mtf_indicator_probability=1.0,
            min_indicators=3,
            max_indicators=3,
        )
        generator = IndicatorGenerator(config)

        indicators = generator.generate_random_indicators()

        # 指標構成サービスにより追加の指標が生成される可能性がある
        # MTFが有効な場合、タイムフレームが設定されている指標があることを確認
        mtf_indicators = [ind for ind in indicators if ind.timeframe is not None]

        # 少なくとも1つはMTFタイムフレームを持つ（2つ目以降で生成される）
        # 注: enhance_with_* メソッドで追加された指標はタイムフレームなしで生成される
        assert len(mtf_indicators) >= 1 or len(indicators) == 1

        # タイムフレームが設定されている指標はカスタムリストからのもの
        for ind in mtf_indicators:
            assert ind.timeframe in ["5m", "15m"]

    def test_create_indicator_gene_with_timeframe(self) -> None:
        """_create_indicator_gene がタイムフレームを正しく設定すること"""
        config = MockGAConfig()
        generator = IndicatorGenerator(config)

        # タイムフレーム付きで作成
        indicator = generator._create_indicator_gene("SMA", timeframe="4h")

        assert isinstance(indicator, IndicatorGene)
        assert indicator.type == "SMA"
        assert indicator.timeframe == "4h"
        assert indicator.enabled is True

    def test_create_indicator_gene_without_timeframe(self) -> None:
        """_create_indicator_gene がタイムフレームなしで作成できること"""
        config = MockGAConfig()
        generator = IndicatorGenerator(config)

        # タイムフレームなしで作成
        indicator = generator._create_indicator_gene("RSI")

        assert isinstance(indicator, IndicatorGene)
        assert indicator.type == "RSI"
        assert indicator.timeframe is None


class TestIndicatorGeneratorBasic:
    """IndicatorGenerator の基本機能テスト"""

    def test_generates_required_number_of_indicators(self) -> None:
        """指定された範囲の数の指標が生成されること"""
        config = MockGAConfig(min_indicators=2, max_indicators=4)
        generator = IndicatorGenerator(config)

        indicators = generator.generate_random_indicators()

        assert len(indicators) >= 2
        assert len(indicators) <= 4

    def test_all_indicators_are_enabled(self) -> None:
        """生成される全ての指標がenabledであること"""
        config = MockGAConfig()
        generator = IndicatorGenerator(config)

        indicators = generator.generate_random_indicators()

        for indicator in indicators:
            assert indicator.enabled is True

    def test_indicators_have_valid_types(self) -> None:
        """生成される指標が有効なタイプを持つこと"""
        config = MockGAConfig()
        generator = IndicatorGenerator(config)

        indicators = generator.generate_random_indicators()

        for indicator in indicators:
            assert indicator.type is not None
            assert isinstance(indicator.type, str)
            assert len(indicator.type) > 0
