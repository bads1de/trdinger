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


class TestGeneValidatorTPSLSplit:
    """ロング/ショート別TPSL遺伝子のバリデーションテスト"""

    def _create_base_strategy(self, long_tpsl=None, short_tpsl=None):
        from backend.app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene, Condition
        
        # 最小限の有効な戦略
        return StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)],
            entry_conditions=[Condition("close", ">", "SMA_20")],
            exit_conditions=[], # TPSLがあれば空でもOK
            long_tpsl_gene=long_tpsl,
            short_tpsl_gene=short_tpsl,
        )

    def test_valid_split_tpsl(self):
        """有効なロング/ショートTPSL設定が通過すること"""
        from backend.app.services.auto_strategy.models.strategy_models import TPSLGene, TPSLMethod
        
        validator = GeneValidator()
        
        long_tpsl = TPSLGene(enabled=True, method=TPSLMethod.FIXED_PERCENTAGE, stop_loss_pct=0.01)
        short_tpsl = TPSLGene(enabled=True, method=TPSLMethod.FIXED_PERCENTAGE, stop_loss_pct=0.01)
        
        gene = self._create_base_strategy(long_tpsl, short_tpsl)
        is_valid, errors = validator.validate_strategy_gene(gene)
        
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_long_tpsl(self):
        """無効なロングTPSL設定（負のSL）が拒否されること"""
        from backend.app.services.auto_strategy.models.strategy_models import TPSLGene, TPSLMethod
        
        validator = GeneValidator()
        
        # 無効な設定: stop_loss_pct < 0
        long_tpsl = TPSLGene(enabled=True, method=TPSLMethod.FIXED_PERCENTAGE, stop_loss_pct=-0.01)
        short_tpsl = TPSLGene(enabled=True, method=TPSLMethod.FIXED_PERCENTAGE, stop_loss_pct=0.01)
        
        gene = self._create_base_strategy(long_tpsl, short_tpsl)
        is_valid, errors = validator.validate_strategy_gene(gene)
        
        assert is_valid is False
        assert any("long_tpsl_gene" in err for err in errors) or any("stop_loss_pct" in err for err in errors)

    def test_invalid_short_tpsl(self):
        """無効なショートTPSL設定（負のTP）が拒否されること"""
        from backend.app.services.auto_strategy.models.strategy_models import TPSLGene, TPSLMethod
        
        validator = GeneValidator()
        
        long_tpsl = TPSLGene(enabled=True, method=TPSLMethod.FIXED_PERCENTAGE, stop_loss_pct=0.01)
        # 無効な設定: take_profit_pct < 0
        short_tpsl = TPSLGene(enabled=True, method=TPSLMethod.FIXED_PERCENTAGE, take_profit_pct=-0.01)
        
        gene = self._create_base_strategy(long_tpsl, short_tpsl)
        is_valid, errors = validator.validate_strategy_gene(gene)
        
        assert is_valid is False
        assert any("short_tpsl_gene" in err for err in errors) or any("take_profit_pct" in err for err in errors)


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
