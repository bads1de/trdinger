"""
StrategyParameterSpace のテスト
"""

import pytest

from app.services.auto_strategy.models.conditions import Condition, ConditionGroup
from app.services.auto_strategy.models.indicator_gene import IndicatorGene
from app.services.auto_strategy.models.strategy_gene import StrategyGene
from app.services.auto_strategy.models.tpsl_gene import TPSLGene
from app.services.auto_strategy.optimization.strategy_parameter_space import (
    StrategyParameterSpace,
)
from app.services.ml.optimization.optuna_optimizer import ParameterSpace


class TestStrategyParameterSpace:
    """StrategyParameterSpace のテスト"""

    @pytest.fixture
    def parameter_space(self):
        """テスト用の StrategyParameterSpace インスタンス"""
        return StrategyParameterSpace()

    @pytest.fixture
    def sample_gene(self):
        """テスト用のサンプル StrategyGene"""
        return StrategyGene(
            id="test-gene-001",
            indicators=[
                IndicatorGene(type="RSI", parameters={"length": 14}),
                IndicatorGene(type="SMA", parameters={"length": 20}),
            ],
            long_entry_conditions=[
                Condition(
                    left_operand={"indicator": "RSI", "output": "RSI"},
                    operator="<",
                    right_operand=30.0,
                )
            ],
            short_entry_conditions=[
                Condition(
                    left_operand={"indicator": "RSI", "output": "RSI"},
                    operator=">",
                    right_operand=70.0,
                )
            ],
            tpsl_gene=TPSLGene(
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
            ),
        )

    @pytest.fixture
    def empty_gene(self):
        """空の StrategyGene"""
        return StrategyGene(id="empty-gene")

    def test_init(self, parameter_space):
        """初期化のテスト"""
        assert parameter_space is not None

    def test_build_parameter_space_with_indicators(self, parameter_space, sample_gene):
        """インジケーターパラメータ空間の構築テスト"""
        result = parameter_space.build_parameter_space(
            sample_gene, include_indicators=True, include_tpsl=False
        )

        assert isinstance(result, dict)
        # RSI の length パラメータ
        assert "ind_0_length" in result
        assert isinstance(result["ind_0_length"], ParameterSpace)
        # SMA の length パラメータ
        assert "ind_1_length" in result

    def test_build_parameter_space_with_tpsl(self, parameter_space, sample_gene):
        """TPSL パラメータ空間の構築テスト"""
        result = parameter_space.build_parameter_space(
            sample_gene, include_indicators=False, include_tpsl=True
        )

        assert isinstance(result, dict)
        # TPSL パラメータ
        assert "tpsl_stop_loss_pct" in result
        assert "tpsl_take_profit_pct" in result
        assert isinstance(result["tpsl_stop_loss_pct"], ParameterSpace)

    def test_build_parameter_space_with_thresholds(self, parameter_space, sample_gene):
        """閾値パラメータ空間の構築テスト"""
        result = parameter_space.build_parameter_space(
            sample_gene,
            include_indicators=False,
            include_tpsl=False,
            include_thresholds=True,
        )

        assert isinstance(result, dict)
        # 閾値パラメータ（right_operand が数値の場合）
        assert "long_thresh_0" in result
        assert "short_thresh_0" in result

    def test_build_parameter_space_empty_gene(self, parameter_space, empty_gene):
        """空の遺伝子に対するパラメータ空間構築テスト"""
        result = parameter_space.build_parameter_space(empty_gene)

        # 空の遺伝子でも辞書が返される
        assert isinstance(result, dict)
        # インジケーターがないのでパラメータはない
        assert len([k for k in result.keys() if k.startswith("ind_")]) == 0

    def test_apply_params_to_gene_indicators(self, parameter_space, sample_gene):
        """インジケーターパラメータの適用テスト"""
        params = {
            "ind_0_length": 21,  # RSI の length を変更
            "ind_1_length": 50,  # SMA の length を変更
        }

        result = parameter_space.apply_params_to_gene(sample_gene, params)

        # 元の遺伝子は変更されない
        assert sample_gene.indicators[0].parameters["length"] == 14
        assert sample_gene.indicators[1].parameters["length"] == 20

        # 新しい遺伝子にはパラメータが適用される
        assert result.indicators[0].parameters["length"] == 21
        assert result.indicators[1].parameters["length"] == 50

    def test_apply_params_to_gene_tpsl(self, parameter_space, sample_gene):
        """TPSL パラメータの適用テスト"""
        params = {
            "tpsl_stop_loss_pct": 0.05,
            "tpsl_take_profit_pct": 0.10,
        }

        result = parameter_space.apply_params_to_gene(sample_gene, params)

        # 新しい遺伝子にはパラメータが適用される
        assert result.tpsl_gene.stop_loss_pct == 0.05
        assert result.tpsl_gene.take_profit_pct == 0.10

    def test_apply_params_to_gene_thresholds(self, parameter_space, sample_gene):
        """閾値パラメータの適用テスト"""
        params = {
            "long_thresh_0": 25.0,
            "short_thresh_0": 75.0,
        }

        result = parameter_space.apply_params_to_gene(sample_gene, params)

        # 閾値が更新される
        assert result.long_entry_conditions[0].right_operand == 25.0
        assert result.short_entry_conditions[0].right_operand == 75.0

    def test_apply_params_preserves_type(self, parameter_space, sample_gene):
        """パラメータ適用時の型保持テスト"""
        params = {
            "ind_0_length": 21.0,  # 浮動小数点で渡す
        }

        result = parameter_space.apply_params_to_gene(sample_gene, params)

        # 元の型（整数）が保持される
        assert isinstance(result.indicators[0].parameters["length"], int)
        assert result.indicators[0].parameters["length"] == 21

    def test_build_parameter_space_disabled_indicator(self, parameter_space):
        """無効なインジケーターはパラメータ空間に含まれないテスト"""
        gene = StrategyGene(
            id="test-gene",
            indicators=[
                IndicatorGene(type="RSI", parameters={"length": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"length": 20}, enabled=False),
            ],
        )

        result = parameter_space.build_parameter_space(
            gene, include_indicators=True, include_tpsl=False
        )

        # 有効なインジケーターのパラメータのみ含まれる
        assert "ind_0_length" in result
        assert "ind_1_length" not in result

    def test_build_parameter_space_condition_group(self, parameter_space):
        """ConditionGroup を含む場合のテスト"""
        gene = StrategyGene(
            id="test-gene",
            long_entry_conditions=[
                ConditionGroup(
                    operator="OR",
                    conditions=[
                        Condition(
                            left_operand={"indicator": "RSI"},
                            operator="<",
                            right_operand=30.0,
                        ),
                        Condition(
                            left_operand={"indicator": "RSI"},
                            operator="<",
                            right_operand=35.0,
                        ),
                    ],
                )
            ],
        )

        result = parameter_space.build_parameter_space(
            gene,
            include_indicators=False,
            include_tpsl=False,
            include_thresholds=True,
        )

        # ネストされた条件の閾値も含まれる
        assert isinstance(result, dict)
        # グループ名付きのキーが生成される
        assert any("grp" in k for k in result.keys())

    def test_long_short_tpsl_separation(self, parameter_space):
        """ロング・ショート別TPSL のテスト"""
        gene = StrategyGene(
            id="test-gene",
            long_tpsl_gene=TPSLGene(stop_loss_pct=0.02, take_profit_pct=0.04),
            short_tpsl_gene=TPSLGene(stop_loss_pct=0.03, take_profit_pct=0.06),
        )

        result = parameter_space.build_parameter_space(
            gene, include_indicators=False, include_tpsl=True
        )

        # ロングとショート別々のパラメータが含まれる
        assert "long_tpsl_stop_loss_pct" in result
        assert "short_tpsl_stop_loss_pct" in result


