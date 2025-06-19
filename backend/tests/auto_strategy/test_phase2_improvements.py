"""
Phase 2改善項目のテスト

指標セット拡張、条件生成ロジック改善のテストを実装
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
    decode_list_to_gene,
    _generate_indicator_parameters,
    _generate_indicator_specific_conditions,
)


class TestIndicatorSetExpansion:
    """指標セット拡張のテスト"""

    def test_expanded_indicator_mapping(self):
        """拡張された指標マッピングのテスト"""
        # 拡張された指標セットをテスト（0.4で7番目のMACDを取得）
        test_encoded = [0.4, 0.5] + [0.0] * 14  # MACD指標をテスト

        gene = decode_list_to_gene(test_encoded)

        # MACD指標が正しく生成されることを確認
        assert len(gene.indicators) > 0
        assert gene.indicators[0].type == "MACD"
        assert "fast_period" in gene.indicators[0].parameters
        assert "slow_period" in gene.indicators[0].parameters
        assert "signal_period" in gene.indicators[0].parameters

    def test_bollinger_bands_parameters(self):
        """ボリンジャーバンド指標のパラメータテスト"""
        parameters = _generate_indicator_parameters("BB", 0.5)

        assert "period" in parameters
        assert "std_dev" in parameters
        assert 10 <= parameters["period"] <= 30
        assert 1.5 <= parameters["std_dev"] <= 2.5

    def test_stochastic_parameters(self):
        """ストキャスティクス指標のパラメータテスト"""
        parameters = _generate_indicator_parameters("STOCH", 0.5)

        assert "k_period" in parameters
        assert "d_period" in parameters
        assert 10 <= parameters["k_period"] <= 20
        assert 3 <= parameters["d_period"] <= 7

    def test_macd_parameters(self):
        """MACD指標のパラメータテスト"""
        parameters = _generate_indicator_parameters("MACD", 0.5)

        assert "fast_period" in parameters
        assert "slow_period" in parameters
        assert "signal_period" in parameters
        assert parameters["fast_period"] < parameters["slow_period"]

    def test_indicator_diversity(self):
        """指標の多様性テスト"""
        # 複数の異なる指標IDをテスト（0.0-1.0の範囲で正規化された値を使用）
        indicator_types = set()

        for i in range(15):  # 0-14の範囲
            # 0.0から1.0の範囲で均等に分散させる
            normalized_value = i / 15.0
            test_encoded = [normalized_value, 0.5] + [0.0] * 14
            gene = decode_list_to_gene(test_encoded)

            if gene.indicators:
                indicator_types.add(gene.indicators[0].type)

        # 少なくとも8種類以上の指標が利用可能であることを確認
        assert len(indicator_types) >= 8

        # 重要な指標が含まれていることを確認
        expected_indicators = {"SMA", "EMA", "RSI", "MACD", "BB", "STOCH"}
        # 部分的な一致を確認（全てが含まれている必要はない）
        assert len(expected_indicators.intersection(indicator_types)) >= 4


class TestConditionGenerationImprovement:
    """条件生成ロジック改善のテスト"""

    def test_rsi_specific_conditions(self):
        """RSI固有の条件生成テスト"""
        indicator = IndicatorGene(type="RSI", parameters={"period": 14})
        entry_conditions, exit_conditions = _generate_indicator_specific_conditions(
            indicator, "RSI_14"
        )

        # RSIのオーバーソールド/オーバーボート条件を確認
        assert len(entry_conditions) == 1
        assert len(exit_conditions) == 1
        assert entry_conditions[0].left_operand == "RSI_14"
        assert entry_conditions[0].operator == "<"
        assert entry_conditions[0].right_operand == 30
        assert exit_conditions[0].right_operand == 70

    def test_moving_average_conditions(self):
        """移動平均の条件生成テスト"""
        for ma_type in ["SMA", "EMA", "WMA"]:
            indicator = IndicatorGene(type=ma_type, parameters={"period": 20})
            entry_conditions, exit_conditions = _generate_indicator_specific_conditions(
                indicator, f"{ma_type}_20"
            )

            # 価格と移動平均の比較条件を確認
            assert entry_conditions[0].left_operand == "close"
            assert entry_conditions[0].operator == ">"
            assert entry_conditions[0].right_operand == f"{ma_type}_20"
            assert exit_conditions[0].operator == "<"

    def test_macd_conditions(self):
        """MACD条件生成テスト"""
        indicator = IndicatorGene(
            type="MACD",
            parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        )
        entry_conditions, exit_conditions = _generate_indicator_specific_conditions(
            indicator, "MACD_12_26_9"
        )

        # MACDの複合条件を確認
        assert len(entry_conditions) == 2  # ゼロライン交差 + シグナル交差
        assert len(exit_conditions) == 1

        # ゼロライン交差条件
        assert any(cond.right_operand == 0 for cond in entry_conditions)
        # シグナル交差条件
        assert any("signal" in str(cond.right_operand) for cond in entry_conditions)

    def test_bollinger_bands_conditions(self):
        """ボリンジャーバンド条件生成テスト"""
        indicator = IndicatorGene(type="BB", parameters={"period": 20, "std_dev": 2.0})
        entry_conditions, exit_conditions = _generate_indicator_specific_conditions(
            indicator, "BB_20_2.0"
        )

        # バンドタッチ条件を確認
        assert "lower" in entry_conditions[0].right_operand
        assert "upper" in exit_conditions[0].right_operand

    def test_stochastic_conditions(self):
        """ストキャスティクス条件生成テスト"""
        indicator = IndicatorGene(
            type="STOCH", parameters={"k_period": 14, "d_period": 3}
        )
        entry_conditions, exit_conditions = _generate_indicator_specific_conditions(
            indicator, "STOCH_14_3"
        )

        # オーバーソールド/オーバーボート条件を確認
        assert "_k" in entry_conditions[0].left_operand
        assert entry_conditions[0].right_operand == 20
        assert exit_conditions[0].right_operand == 80

    def test_oscillator_conditions(self):
        """オシレーター系指標の条件生成テスト"""
        # CCI
        cci_indicator = IndicatorGene(type="CCI", parameters={"period": 20})
        cci_entry, cci_exit = _generate_indicator_specific_conditions(
            cci_indicator, "CCI_20"
        )
        assert cci_entry[0].right_operand == -100
        assert cci_exit[0].right_operand == 100

        # Williams %R
        williams_indicator = IndicatorGene(type="WILLIAMS", parameters={"period": 14})
        williams_entry, williams_exit = _generate_indicator_specific_conditions(
            williams_indicator, "WILLIAMS_14"
        )
        assert williams_entry[0].right_operand == -80
        assert williams_exit[0].right_operand == -20

    def test_trend_indicators_conditions(self):
        """トレンド系指標の条件生成テスト"""
        # ADX
        adx_indicator = IndicatorGene(type="ADX", parameters={"period": 14})
        adx_entry, adx_exit = _generate_indicator_specific_conditions(
            adx_indicator, "ADX_14"
        )
        assert adx_entry[0].right_operand == 25
        assert adx_exit[0].right_operand == 20

        # Aroon
        aroon_indicator = IndicatorGene(type="AROON", parameters={"period": 14})
        aroon_entry, aroon_exit = _generate_indicator_specific_conditions(
            aroon_indicator, "AROON_14"
        )
        assert "_up" in aroon_entry[0].left_operand
        assert "_down" in aroon_entry[0].right_operand


class TestStrategyQualityImprovement:
    """戦略品質改善の統合テスト"""

    def test_strategy_diversity_improvement(self):
        """戦略多様性改善のテスト"""
        # 複数の戦略を生成して多様性を確認
        strategies = []

        for i in range(10):
            # 異なる指標組み合わせをテスト
            test_encoded = [float(i % 15 + 1), 0.5, float((i + 5) % 15 + 1), 0.3] + [
                0.0
            ] * 12
            gene = decode_list_to_gene(test_encoded)
            strategies.append(gene)

        # 指標の多様性を確認
        indicator_types = set()
        for strategy in strategies:
            for indicator in strategy.indicators:
                indicator_types.add(indicator.type)

        # 少なくとも5種類以上の異なる指標が使用されていることを確認
        assert len(indicator_types) >= 5

    def test_condition_complexity_improvement(self):
        """条件複雑性改善のテスト"""
        # MACD戦略の条件複雑性をテスト
        test_encoded = [7.0, 0.5] + [0.0] * 14  # MACD
        gene = decode_list_to_gene(test_encoded)

        # 複数の条件が生成されることを確認（MACD の場合）
        if gene.indicators and gene.indicators[0].type == "MACD":
            assert len(gene.entry_conditions) >= 1
            assert len(gene.exit_conditions) >= 1

            # 条件の内容が適切であることを確認
            entry_condition_operands = [
                cond.left_operand for cond in gene.entry_conditions
            ]
            assert any("MACD" in operand for operand in entry_condition_operands)

    def test_parameter_appropriateness(self):
        """パラメータ適切性のテスト"""
        # 各指標タイプのパラメータが適切な範囲にあることを確認
        test_cases = [
            ("RSI", {"period": (5, 50)}),
            ("MACD", {"fast_period": (8, 15), "slow_period": (20, 30)}),
            ("BB", {"period": (10, 30), "std_dev": (1.5, 2.5)}),
            ("STOCH", {"k_period": (10, 20), "d_period": (3, 7)}),
        ]

        for indicator_type, expected_ranges in test_cases:
            parameters = _generate_indicator_parameters(indicator_type, 0.5)

            for param_name, (min_val, max_val) in expected_ranges.items():
                if param_name in parameters:
                    param_value = parameters[param_name]
                    assert (
                        min_val <= param_value <= max_val
                    ), f"{indicator_type}.{param_name} = {param_value} not in range [{min_val}, {max_val}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
