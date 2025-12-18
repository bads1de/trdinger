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
        self.mock_generator = MagicMock()
        self.mock_generator.logger = Mock()
        
        # 内部ヘルパーの振る舞いを設定
        self.mock_generator._get_indicator_name.side_effect = lambda i: i.id or i.type
        self.mock_generator._get_band_names.side_effect = lambda i: (f"{i.id}_up", f"{i.id}_low")
        self.mock_generator._is_price_scale.return_value = True
        self.mock_generator._is_band_indicator.side_effect = lambda i: "BB" in i.type
        self.mock_generator._classify_indicators.side_effect = self._mock_classify
        self.mock_generator._structure_conditions.side_effect = lambda x: x
        
        # サイド別条件の生成モック
        def _mock_side_cond(ind, side, name=None):
            target = name or ind.type
            return Condition(left_operand=target, operator=">" if side=="long" else "<", right_operand=0)
        self.mock_generator._create_side_condition.side_effect = _mock_side_cond
        self.mock_generator.generate_fallback_conditions.return_value = ([], [], [])

        # テスト対象の戦略クラス
        self.strategy = ComplexConditionsStrategy(self.mock_generator)

    def _mock_classify(self, indicators):
        res = {IndicatorType.TREND: [], IndicatorType.MOMENTUM: [], IndicatorType.VOLATILITY: []}
        for ind in indicators:
            name = ind.type.upper()
            if "SMA" in name: res[IndicatorType.TREND].append(ind)
            elif "RSI" in name: res[IndicatorType.MOMENTUM].append(ind)
            elif "BB" in name: res[IndicatorType.VOLATILITY].append(ind)
        return res

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

                    # 大文字小文字を区別せずにチェック

                    l_op, r_op = str(cond.left_operand).upper(), str(cond.right_operand).upper()

                    if "SMA" in l_op or "SMA" in r_op:

                        has_sma_condition = True

                    if "RSI" in l_op:

                        has_rsi_condition = True

                    if l_op == "CLOSE" and "SMA" in r_op:

                        has_sma_condition = True

        

            for cond in long_result:

                check_cond(cond)

        

            # 実装後は確実に通るようにする

            self.assertTrue(has_sma_condition and has_rsi_condition)

        

        def test_generate_moving_average_cross_pattern(self):

            """移動平均線クロスパターンの生成テスト"""

            # 短期SMAと長期SMAを用意（期間を変える）

            sma_short = IndicatorGene(

                type="SMA", parameters={"period": 10}, enabled=True, id="sma_10"

            )

            sma_long = IndicatorGene(

                type="SMA", parameters={"period": 50}, enabled=True, id="sma_50"

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

                    l_op, r_op = str(cond.left_operand).upper(), str(cond.right_operand).upper()

                    if "SMA" in l_op and "SMA" in r_op:

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
                # 大文字小文字を区別せずにチェック
                l_op, r_op = str(cond.left_operand).upper(), str(cond.right_operand).upper()
                if l_op == "CLOSE" and "BB" in r_op:
                    has_breakout = True

        for cond in long_result:
            check_cond(cond)

        self.assertTrue(has_breakout)


if __name__ == "__main__":
    unittest.main()
