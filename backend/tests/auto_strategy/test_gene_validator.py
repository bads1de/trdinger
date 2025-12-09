"""
GeneValidator テスト

指標遺伝子のタイムフレームバリデーションを含むテスト
"""

from backend.app.services.auto_strategy.models.strategy_models import IndicatorGene
from backend.app.services.auto_strategy.models.validator import GeneValidator


class TestGeneValidatorTimeframe:
    """タイムフレームバリデーションのテスト"""

    def test_valid_timeframe_accepted(self) -> None:
        """有効なタイムフレームが受け入れられること"""
        validator = GeneValidator()

        indicator = IndicatorGene(
            type="SMA", parameters={"period": 20}, enabled=True, timeframe="1h"
        )

        assert validator.validate_indicator_gene(indicator) is True

    def test_invalid_timeframe_rejected(self) -> None:
        """無効なタイムフレームが拒否されること"""
        validator = GeneValidator()

        indicator = IndicatorGene(
            type="SMA", parameters={"period": 20}, enabled=True, timeframe="invalid_tf"
        )

        assert validator.validate_indicator_gene(indicator) is False

    def test_none_timeframe_accepted(self) -> None:
        """Noneタイムフレームが受け入れられること（デフォルトTF使用）"""
        validator = GeneValidator()

        indicator = IndicatorGene(
            type="SMA", parameters={"period": 20}, enabled=True, timeframe=None
        )

        assert validator.validate_indicator_gene(indicator) is True

    def test_all_supported_timeframes_accepted(self) -> None:
        """全てのサポートされるタイムフレームが受け入れられること"""
        from backend.app.services.auto_strategy.config.constants import (
            SUPPORTED_TIMEFRAMES,
        )

        validator = GeneValidator()

        for tf in SUPPORTED_TIMEFRAMES:
            indicator = IndicatorGene(
                type="SMA", parameters={"period": 20}, enabled=True, timeframe=tf
            )
            assert (
                validator.validate_indicator_gene(indicator) is True
            ), f"Timeframe {tf} should be valid"


class TestGeneValidatorBasic:
    """GeneValidator の基本機能テスト"""

    def test_valid_indicator_gene(self) -> None:
        """有効な指標遺伝子が受け入れられること"""
        validator = GeneValidator()

        indicator = IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)

        assert validator.validate_indicator_gene(indicator) is True

    def test_invalid_indicator_type(self) -> None:
        """無効な指標タイプが拒否されること"""
        validator = GeneValidator()

        indicator = IndicatorGene(
            type="INVALID_INDICATOR", parameters={"period": 20}, enabled=True
        )

        assert validator.validate_indicator_gene(indicator) is False

    def test_empty_indicator_type(self) -> None:
        """空の指標タイプが拒否されること"""
        validator = GeneValidator()

        indicator = IndicatorGene(type="", parameters={"period": 20}, enabled=True)

        assert validator.validate_indicator_gene(indicator) is False

    def test_invalid_period_parameter(self) -> None:
        """無効な期間パラメータが拒否されること"""
        validator = GeneValidator()

        indicator = IndicatorGene(
            type="SMA", parameters={"period": -5}, enabled=True  # 負の期間
        )

        assert validator.validate_indicator_gene(indicator) is False

    def test_zero_period_parameter(self) -> None:
        """0の期間パラメータが拒否されること"""
        validator = GeneValidator()

        indicator = IndicatorGene(type="SMA", parameters={"period": 0}, enabled=True)

        assert validator.validate_indicator_gene(indicator) is False
