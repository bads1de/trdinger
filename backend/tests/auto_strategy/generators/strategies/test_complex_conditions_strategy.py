import unittest
from unittest.mock import Mock, MagicMock
from app.services.auto_strategy.generators.complex_conditions_strategy import (
    ComplexConditionsStrategy,
)
from app.services.auto_strategy.genes import IndicatorGene, Condition, ConditionGroup
from app.services.auto_strategy.config.constants import IndicatorType


class TestComplexConditionsStrategy(unittest.TestCase):
    def setUp(self):
        # ConditionGeneratorのモックを作成
        self.mock_generator = Mock()
        self.mock_generator.logger = Mock()
        self.mock_generator._get_indicator_type = self._mock_get_indicator_type

        # モックの戻り値を設定
        dummy_cond = Condition(left_operand="Dummy", operator="<", right_operand=0)
        self.mock_generator._generic_short_conditions.return_value = [dummy_cond]
        self.mock_generator._generic_long_conditions.return_value = [dummy_cond]

        # テスト対象の戦略クラス
        self.strategy = ComplexConditionsStrategy(self.mock_generator)

    def _mock_get_indicator_type(self, indicator):
        # 簡易的な型判定
        name = indicator.type.upper()
        if "SMA" in name or "EMA" in name:
            return IndicatorType.TREND
        if "RSI" in name or "MACD" in name:
            return IndicatorType.MOMENTUM
        if "BB" in name or "ATR" in name:
            return IndicatorType.VOLATILITY
        return IndicatorType.TREND

    def test_generate_trend_pullback_pattern(self):
        """トレンド押し目買いパターンの生成テスト"""
        # SMA(トレンド) と RSI(オシレーター) を用意
        sma = IndicatorGene(
            type="SMA", parameters={"length": 50}, enabled=True, id="sma_50"
        )
        rsi = IndicatorGene(
            type="RSI", parameters={"length": 14}, enabled=True, id="rsi_14"
        )

        # 条件生成
        long_result, short_result, _ = self.strategy.generate_conditions([sma, rsi])

        # 期待: SMAによるトレンドフィルタ と RSIによる逆張りトリガー が生成されていること
        # 例: (Close > SMA) AND (RSI < 30)

        # 少なくとも2つの条件、あるいは1つのConditionGroupが生成されるはず
        self.assertTrue(len(long_result) > 0)

        # 条件の中身を検査（簡易チェック）
        has_sma_condition = False
        has_rsi_condition = False

        def check_cond(cond):
            nonlocal has_sma_condition, has_rsi_condition
            if isinstance(cond, ConditionGroup):
                for c in cond.conditions:
                    check_cond(c)
            elif isinstance(cond, Condition):
                # ID付きの名前になっているはず
                if "SMA" in str(cond.left_operand) or "SMA" in str(cond.right_operand):
                    has_sma_condition = True
                if "RSI" in str(cond.left_operand):
                    has_rsi_condition = True
                # Price > SMA の場合、leftが"close"になる可能性も考慮
                if str(cond.left_operand).lower() == "close" and "SMA" in str(
                    cond.right_operand
                ):
                    has_sma_condition = True

        for cond in long_result:
            check_cond(cond)

        # 実装後は確実に通るようにする
        self.assertTrue(has_sma_condition and has_rsi_condition)

    def test_generate_moving_average_cross_pattern(self):
        """移動平均線クロスパターンの生成テスト"""
        # 短期SMAと長期SMAを用意
        sma_short = IndicatorGene(
            type="SMA", parameters={"length": 10}, enabled=True, id="sma_10"
        )
        sma_long = IndicatorGene(
            type="SMA", parameters={"length": 50}, enabled=True, id="sma_50"
        )

        # 条件生成
        long_result, short_result, _ = self.strategy.generate_conditions(
            [sma_short, sma_long]
        )

        # 期待: SMA同士の比較条件 (SMA_10 > SMA_50)
        has_cross_condition = False

        def check_cond(cond):
            nonlocal has_cross_condition
            if isinstance(cond, Condition):
                # 左右のオペランドが共にSMAであるか、あるいはIDで比較
                if "SMA" in str(cond.left_operand) and "SMA" in str(cond.right_operand):
                    has_cross_condition = True

        for cond in long_result:
            check_cond(cond)

        self.assertTrue(has_cross_condition)

    def test_generate_volatility_breakout_pattern(self):
        """ボラティリティブレイクアウトパターンの生成テスト"""
        # BBを用意
        bb = IndicatorGene(
            type="BB", parameters={"length": 20, "std": 2.0}, enabled=True, id="bb_20"
        )

        long_result, _, _ = self.strategy.generate_conditions([bb])

        # 期待: Close > BB_Upper
        has_breakout = False

        def check_cond(cond):
            nonlocal has_breakout
            if isinstance(cond, Condition):
                # leftがclose, rightがBB系
                if str(cond.left_operand).lower() == "close" and "BB" in str(
                    cond.right_operand
                ):
                    has_breakout = True

        for cond in long_result:
            check_cond(cond)

        self.assertTrue(has_breakout)


if __name__ == "__main__":
    unittest.main()
