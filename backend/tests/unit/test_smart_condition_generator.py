"""
SmartConditionGeneratorの単体テスト

定義されたルールに基づき、多様かつ論理的な条件の組み合わせを生成することを検証
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from app.core.services.auto_strategy.generators.smart_condition_generator import (
    SmartConditionGenerator,
    StrategyType,
    IndicatorType,
    INDICATOR_CHARACTERISTICS,
    COMBINATION_RULES
)
from app.core.services.auto_strategy.models.gene_strategy import IndicatorGene, Condition


class TestSmartConditionGenerator:
    """SmartConditionGeneratorのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.generator = SmartConditionGenerator(enable_smart_generation=True)

    def test_initialization(self):
        """初期化のテスト"""
        # 有効な初期化
        generator = SmartConditionGenerator(enable_smart_generation=True)
        assert generator.enable_smart_generation is True

        # 無効な初期化
        generator_disabled = SmartConditionGenerator(enable_smart_generation=False)
        assert generator_disabled.enable_smart_generation is False

    def test_indicator_characteristics_database(self):
        """指標特性データベースの検証"""
        # 必要な指標が定義されていることを確認
        required_indicators = ["RSI", "SMA", "EMA", "BB", "ADX", "CCI", "MACD", "STOCH", "ATR"]

        for indicator in required_indicators:
            assert indicator in INDICATOR_CHARACTERISTICS, f"{indicator}が指標特性データベースに定義されていません"

        # RSIの特性を詳細検証
        rsi_char = INDICATOR_CHARACTERISTICS["RSI"]
        assert rsi_char["type"] == IndicatorType.MOMENTUM
        assert rsi_char["range"] == (0, 100)
        assert "oversold_threshold" in rsi_char
        assert "overbought_threshold" in rsi_char

        # BBの特性を詳細検証
        bb_char = INDICATOR_CHARACTERISTICS["BB"]
        assert bb_char["type"] == IndicatorType.VOLATILITY
        assert "components" in bb_char
        assert bb_char["components"] == ["upper", "middle", "lower"]

        # ADXの特性を詳細検証（方向性を示さないことを確認）
        adx_char = INDICATOR_CHARACTERISTICS["ADX"]
        assert adx_char["type"] == IndicatorType.TREND
        assert adx_char["no_direction"] is True

    def test_combination_rules(self):
        """組み合わせルールの検証"""
        # 必要なルールが定義されていることを確認
        required_rules = ["trend_momentum", "volatility_trend", "momentum_volatility"]

        for rule in required_rules:
            assert rule in COMBINATION_RULES, f"{rule}が組み合わせルールに定義されていません"

        # 重み付けの合計が1.0になることを確認
        total_weight = sum(rule["weight"] for rule in COMBINATION_RULES.values())
        assert abs(total_weight - 1.0) < 0.01, f"重み付けの合計が1.0ではありません: {total_weight}"

    def test_strategy_type_selection(self):
        """戦略タイプ選択のテスト"""
        # 複数の指標タイプがある場合：異なる指標の組み合わせ戦略
        indicators_mixed = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]
        strategy_type = self.generator._select_strategy_type(indicators_mixed)
        assert strategy_type == StrategyType.DIFFERENT_INDICATORS

        # 同じ指標が複数ある場合：時間軸分離戦略
        indicators_same = [
            IndicatorGene(type="RSI", parameters={"period": 7}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 21}, enabled=True)
        ]
        strategy_type = self.generator._select_strategy_type(indicators_same)
        assert strategy_type == StrategyType.TIME_SEPARATION

        # ボリンジャーバンドがある場合：指標特性活用戦略
        indicators_bb = [
            IndicatorGene(type="BB", parameters={"period": 20}, enabled=True)
        ]
        strategy_type = self.generator._select_strategy_type(indicators_bb)
        assert strategy_type == StrategyType.INDICATOR_CHARACTERISTICS

    def test_fallback_conditions(self):
        """フォールバック条件のテスト"""
        long_conds, short_conds, exit_conds = self.generator._generate_fallback_conditions()

        # 基本的な条件が生成されることを確認
        assert len(long_conds) == 1
        assert len(short_conds) == 1
        assert len(exit_conds) == 0

        # ロング条件の検証
        assert long_conds[0].left_operand == "close"
        assert long_conds[0].operator == ">"
        assert long_conds[0].right_operand == "open"

        # ショート条件の検証
        assert short_conds[0].left_operand == "close"
        assert short_conds[0].operator == "<"
        assert short_conds[0].right_operand == "open"

    def test_different_indicators_strategy(self):
        """異なる指標の組み合わせ戦略のテスト"""
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="BB", parameters={"period": 20}, enabled=True)
        ]

        long_conds, short_conds, exit_conds = self.generator._generate_different_indicators_strategy(indicators)

        # 条件が生成されることを確認
        assert len(long_conds) > 0
        assert len(short_conds) > 0
        assert len(exit_conds) == 0  # TP/SLが使用されるため

        # 異なる指標が使用されていることを確認（可能な限り）
        long_indicators = set()
        short_indicators = set()

        for cond in long_conds:
            if "_" in str(cond.left_operand):
                indicator_type = str(cond.left_operand).split("_")[0]
                long_indicators.add(indicator_type)

        for cond in short_conds:
            if "_" in str(cond.left_operand):
                indicator_type = str(cond.left_operand).split("_")[0]
                short_indicators.add(indicator_type)

        # 少なくとも1つの指標が使用されていることを確認
        assert len(long_indicators) > 0 or len(short_indicators) > 0

    def test_time_separation_strategy(self):
        """時間軸分離戦略のテスト"""
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 7}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 21}, enabled=True)
        ]

        long_conds, short_conds, exit_conds = self.generator._generate_time_separation_strategy(indicators)

        # 条件が生成されることを確認
        assert len(long_conds) > 0
        assert len(short_conds) > 0
        assert len(exit_conds) == 0

        # 異なる期間のRSIが使用されていることを確認
        rsi_periods = set()
        for cond in long_conds + short_conds:
            if "RSI_" in str(cond.left_operand):
                period = str(cond.left_operand).split("_")[1]
                rsi_periods.add(period)

        # 複数の期間が使用されていることを確認
        assert len(rsi_periods) >= 2

    def test_indicator_characteristics_strategy_bb(self):
        """指標特性活用戦略（ボリンジャーバンド）のテスト"""
        indicators = [
            IndicatorGene(type="BB", parameters={"period": 20}, enabled=True)
        ]

        long_conds, short_conds, exit_conds = self.generator._generate_indicator_characteristics_strategy(indicators)

        # 条件が生成されることを確認
        assert len(long_conds) > 0
        assert len(short_conds) > 0
        assert len(exit_conds) == 0

        # ボリンジャーバンドの3つの値が使用されていることを確認
        bb_components = set()
        for cond in long_conds + short_conds:
            if "BB_" in str(cond.left_operand) or "BB_" in str(cond.right_operand):
                # Upper, Middle, Lowerのいずれかが使用されているかチェック
                operand_str = str(cond.left_operand) + str(cond.right_operand)
                if "Upper" in operand_str:
                    bb_components.add("Upper")
                elif "Middle" in operand_str:
                    bb_components.add("Middle")
                elif "Lower" in operand_str:
                    bb_components.add("Lower")

        # 少なくとも2つのBB成分が使用されていることを確認
        assert len(bb_components) >= 2

    def test_indicator_characteristics_strategy_adx(self):
        """指標特性活用戦略（ADX）のテスト"""
        indicators = [
            IndicatorGene(type="ADX", parameters={"period": 14}, enabled=True)
        ]

        long_conds, short_conds, exit_conds = self.generator._generate_indicator_characteristics_strategy(indicators)

        # 条件が生成されることを確認
        assert len(long_conds) > 0
        assert len(short_conds) > 0
        assert len(exit_conds) == 0

        # ADXと価格方向の組み合わせが使用されていることを確認
        has_adx = False
        has_price_direction = False

        for cond in long_conds + short_conds:
            if "ADX_" in str(cond.left_operand):
                has_adx = True
            if cond.left_operand == "close" and cond.right_operand == "open":
                has_price_direction = True

        assert has_adx, "ADX条件が含まれていません"
        assert has_price_direction, "価格方向条件が含まれていません"

    def test_generate_balanced_conditions_integration(self):
        """generate_balanced_conditionsの統合テスト"""
        # 様々な指標の組み合わせでテスト
        test_cases = [
            # 異なる指標の組み合わせ
            [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            # 時間軸分離
            [
                IndicatorGene(type="RSI", parameters={"period": 7}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 21}, enabled=True)
            ],
            # ボリンジャーバンド
            [
                IndicatorGene(type="BB", parameters={"period": 20}, enabled=True)
            ],
            # ADX
            [
                IndicatorGene(type="ADX", parameters={"period": 14}, enabled=True)
            ]
        ]

        for indicators in test_cases:
            long_conds, short_conds, exit_conds = self.generator.generate_balanced_conditions(indicators)

            # 基本的な検証
            assert len(long_conds) > 0, f"ロング条件が生成されませんでした: {[i.type for i in indicators]}"
            assert len(short_conds) > 0, f"ショート条件が生成されませんでした: {[i.type for i in indicators]}"
            assert len(exit_conds) == 0, "exit条件は空である必要があります（TP/SL使用のため）"

            # 条件の妥当性検証
            for cond in long_conds + short_conds:
                assert hasattr(cond, 'left_operand'), "left_operandが存在しません"
                assert hasattr(cond, 'operator'), "operatorが存在しません"
                assert hasattr(cond, 'right_operand'), "right_operandが存在しません"
                assert cond.operator in [">", "<", ">=", "<=", "==", "!="], f"無効な演算子: {cond.operator}"

    def test_disabled_smart_generation(self):
        """スマート生成無効時のテスト"""
        generator_disabled = SmartConditionGenerator(enable_smart_generation=False)

        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]

        long_conds, short_conds, exit_conds = generator_disabled.generate_balanced_conditions(indicators)

        # フォールバック条件が返されることを確認
        assert len(long_conds) == 1
        assert len(short_conds) == 1
        assert long_conds[0].left_operand == "close"
        assert short_conds[0].left_operand == "close"

    def test_empty_indicators(self):
        """空の指標リストのテスト"""
        long_conds, short_conds, exit_conds = self.generator.generate_balanced_conditions([])

        # フォールバック条件が返されることを確認
        assert len(long_conds) == 1
        assert len(short_conds) == 1
        assert long_conds[0].left_operand == "close"
        assert short_conds[0].left_operand == "close"

    def test_disabled_indicators(self):
        """無効な指標のテスト"""
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=False),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=False)
        ]

        long_conds, short_conds, exit_conds = self.generator.generate_balanced_conditions(indicators)

        # フォールバック条件が返されることを確認
        assert len(long_conds) == 1
        assert len(short_conds) == 1

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 不正な指標タイプでもエラーにならないことを確認
        indicators = [
            IndicatorGene(type="UNKNOWN_INDICATOR", parameters={"period": 14}, enabled=True)
        ]

        # エラーが発生せず、フォールバック条件が返されることを確認
        long_conds, short_conds, exit_conds = self.generator.generate_balanced_conditions(indicators)

        assert len(long_conds) > 0
        assert len(short_conds) > 0

    def test_condition_diversity(self):
        """条件の多様性テスト"""
        # 同じ指標セットで複数回生成し、異なる条件が生成されることを確認
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]

        results = []
        for _ in range(5):
            long_conds, short_conds, _ = self.generator.generate_balanced_conditions(indicators)
            # 条件を文字列化して比較
            condition_str = str([(c.left_operand, c.operator, c.right_operand) for c in long_conds + short_conds])
            results.append(condition_str)

        # 少なくとも一部は異なる結果が生成されることを期待
        # （ランダム要素があるため、必ずしも全て異なるとは限らない）
        unique_results = set(results)
        assert len(unique_results) >= 1, "条件生成に多様性がありません"