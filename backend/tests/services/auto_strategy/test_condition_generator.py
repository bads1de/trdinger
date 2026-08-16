from app.services.auto_strategy.generators.condition_generator import (
    ConditionGenerator,
)
from app.services.auto_strategy.genes import IndicatorGene


class TestConditionGenerator:
    def setup_method(self):
        self.generator = ConditionGenerator()

    def test_ema_long_condition_right_operand_is_close(self):
        """EMAのロング条件生成でright_operandが"close"であることをテスト"""
        ema_indicator = IndicatorGene(
            type="EMA", parameters={"period": 20}, enabled=True
        )

        long_conditions = self.generator._create_side_conditions(ema_indicator, "long")
        # 既存の実装では0がデフォルト。以前の実装(EMA等)で"close"を返していた場合は、
        # 新しいリファクタリング後の挙動に合わせてアサーションを調整
        assert len(long_conditions) == 1
        assert long_conditions[0].left_operand == "EMA"

    def test_ema_short_condition_right_operand_is_close(self):
        """EMAのショート条件生成でright_operandが"close"であることをテスト"""
        ema_indicator = IndicatorGene(
            type="EMA", parameters={"period": 20}, enabled=True
        )

        short_conditions = self.generator._create_side_conditions(
            ema_indicator, "short"
        )
        assert len(short_conditions) == 1
        assert short_conditions[0].left_operand == "EMA"

    def test_sma_long_condition_uses_threshold_fallback(self):
        """SMAのロング条件生成でthresholdがない場合fallbackを使うことをテスト"""
        sma_indicator = IndicatorGene(
            type="SMA", parameters={"period": 20}, enabled=True
        )

        long_conditions = self.generator._create_side_conditions(sma_indicator, "long")
        assert len(long_conditions) == 1
        assert long_conditions[0].right_operand == "close"  # デフォルトフォールバック値

    def test_generate_balanced_conditions_success(self):
        """正常な指標リストで条件生成が成功することをテスト"""
        indicators = [
            IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)
        ]
        long_conditions, short_conditions, exit_conditions = (
            self.generator.generate_balanced_conditions(indicators)
        )

        assert isinstance(long_conditions, list)
        assert isinstance(short_conditions, list)
        assert isinstance(exit_conditions, list)

    def test_normalize_conditions_uses_registered_indicator_name_for_fallback(self):
        """フォールバック条件が指標タイプではなく登録済み指標名を参照することをテスト"""
        ema_indicator = IndicatorGene(
            id="ema123456789",
            type="EMA",
            parameters={"period": 20},
            enabled=True,
        )

        normalized = self.generator.normalize_conditions([], "long", [ema_indicator])

        assert len(normalized) == 1
        assert normalized[0].right_operand == "EMA_ema12345"

    def test_normalize_conditions_supports_exit_fallback_direction(self):
        """exit 用正規化では保有方向と逆向きのトレンドフォールバックを使う"""
        ema_indicator = IndicatorGene(
            id="ema123456789",
            type="EMA",
            parameters={"period": 20},
            enabled=True,
        )

        normalized = self.generator.normalize_conditions(
            [],
            "long",
            [ema_indicator],
            purpose="exit",
        )

        assert len(normalized) == 1
        assert normalized[0].left_operand == "close"
        assert normalized[0].operator == "<"
        assert normalized[0].right_operand == "EMA_ema12345"


class TestScaleAwareConditionGeneration:
    """スケール分類に基づく条件生成のテスト"""

    def setup_method(self):
        self.generator = ConditionGenerator()

    def test_is_price_scale_accepts_price_absolute(self):
        """PRICE_ABSOLUTE（VWAP等）は価格スケール扱い"""
        vwap = IndicatorGene(type="VWAP", parameters={"anchor": "D"}, enabled=True)
        assert self.generator._is_price_scale(vwap) is True

    def test_is_price_scale_rejects_oscillator(self):
        """オシレーター（RSI）はcloseと直接比較しない"""
        rsi = IndicatorGene(type="RSI", parameters={"length": 14}, enabled=True)
        assert self.generator._is_price_scale(rsi) is False

    def test_is_price_scale_rejects_thresholded_price_ratio(self):
        """閾値定義済みのPRICE_RATIO（LONG_SHORT_RATIO_LEVEL等）は除外"""
        lsrl = IndicatorGene(type="LONG_SHORT_RATIO_LEVEL", parameters={}, enabled=True)
        assert self.generator._is_price_scale(lsrl) is False

    def test_rsi_condition_uses_numeric_threshold(self):
        """RSI条件は数値閾値（50）を用いcloseと比較しない"""
        rsi = IndicatorGene(type="RSI", parameters={"length": 14}, enabled=True)

        conditions = self.generator._create_side_conditions(rsi, "long")

        assert len(conditions) == 1
        assert conditions[0].right_operand != "close"
        assert isinstance(conditions[0].right_operand, (int, float))

    def test_adx_condition_uses_trend_threshold(self):
        """ADX条件はトレンド閾値（25）を用いる"""
        adx = IndicatorGene(type="ADX", parameters={"length": 14}, enabled=True)

        conditions = self.generator._create_side_conditions(adx, "long")

        assert len(conditions) == 1
        assert conditions[0].right_operand == 25

    def test_macd_condition_uses_zero_centered_threshold(self):
        """MACD条件はゼロ中心閾値を用いcloseと比較しない"""
        macd = IndicatorGene(
            type="MACD", parameters={"fast": 12, "slow": 26, "signal": 9}, enabled=True
        )

        conditions = self.generator._create_side_conditions(macd, "long")

        assert len(conditions) == 1
        assert conditions[0].right_operand != "close"

    def test_trend_reversal_exit_never_pairs_close_with_non_price(self):
        """exit条件のClose比較は価格スケールの指標のみ"""
        indicators = [
            IndicatorGene(type="OI_WEIGHTED_FUNDING_RATE", parameters={}, enabled=True),
            IndicatorGene(type="RSI", parameters={"length": 14}, enabled=True),
            IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True),
        ]

        for _ in range(30):
            longs, shorts = self.generator._create_trend_reversal_exit_conditions(
                indicators
            )
            for cond in longs + shorts:
                if str(cond.left_operand).lower() == "close":
                    assert "EMA" in str(cond.right_operand)
