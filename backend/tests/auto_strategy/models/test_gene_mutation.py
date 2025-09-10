"""
gene_mutation.py のユニットテスト
"""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import replace

from backend.app.services.auto_strategy.models.gene_mutation import (
    mutate_position_sizing_gene,
    mutate_tpsl_gene
)
from backend.app.services.auto_strategy.models.tpsl_gene import TPSLGene
from backend.app.services.auto_strategy.models.position_sizing_gene import PositionSizingGene
from backend.app.services.auto_strategy.models.enums import TPSLMethod, PositionSizingMethod


class TestPositionSizingGeneMutation:
    """PositionSizingGene 突然変異テスト"""

    def test_mutate_position_sizing_gene_basic(self):
        """基本的な突然変異機能をテスト"""
        # 元の遺伝子を作成
        original_gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=100,
            optimal_f_multiplier=0.5,
            atr_period=14,
            atr_multiplier=2.0,
            risk_per_trade=0.02,
            fixed_ratio=0.1,
            fixed_quantity=1.0,
            min_position_size=0.01,
            max_position_size=10.0,
            enabled=True,
            priority=1.0
        )

        # 突然変異適用
        mutated_gene = mutate_position_sizing_gene(original_gene, mutation_rate=1.0)

        # 型チェック
        assert isinstance(mutated_gene, PositionSizingGene)

        # 範囲内チェック
        assert 50 <= mutated_gene.lookback_period <= 200
        assert 0.25 <= mutated_gene.optimal_f_multiplier <= 0.75
        assert 0.1 <= mutated_gene.atr_multiplier <= 5.0
        assert 0.001 <= mutated_gene.risk_per_trade <= 0.1
        assert 0.001 <= mutated_gene.fixed_ratio <= 1.0
        assert 0.1 <= mutated_gene.fixed_quantity <= 10.0
        assert 0.001 <= mutated_gene.min_position_size <= 0.1
        assert 5.0 <= mutated_gene.max_position_size <= 50.0
        assert 0.5 <= mutated_gene.priority <= 1.5
        assert 10 <= mutated_gene.atr_period <= 30

    def test_mutate_position_sizing_gene_zero_mutation_rate(self):
        """突然変異率0%のテスト"""
        original_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_QUANTITY,
            lookback_period=100,
            optimal_f_multiplier=0.5,
            fixed_quantity=1.0
        )

        mutated_gene = mutate_position_sizing_gene(original_gene, mutation_rate=0.0)

        # 突然変異率0%なので値が変化しないはず
        assert mutated_gene.lookback_period == original_gene.lookback_period
        assert mutated_gene.optimal_f_multiplier == original_gene.optimal_f_multiplier
        assert mutated_gene.fixed_quantity == original_gene.fixed_quantity

    def test_mutate_position_sizing_gene_enum_mutation(self):
        """Enumフィールドの突然変異テスト"""
        original_gene = PositionSizingGene(method=PositionSizingMethod.FIXED_QUANTITY)

        # 突然変異を必ず発生させる
        mutated_gene = mutate_position_sizing_gene(original_gene, mutation_rate=1.0)

        # Enum型は変更される可能性がある
        assert isinstance(mutated_gene.method, PositionSizingMethod)

    def test_mutate_position_sizing_gene_boundary_values(self):
        """境界値テスト"""
        # 最小値設定の遺伝子
        min_gene = PositionSizingGene(
            lookback_period=50,
            optimal_f_multiplier=0.25,
            atr_multiplier=0.1,
            risk_per_trade=0.001,
            fixed_ratio=0.001,
            min_position_size=0.001,
            max_position_size=5.0,
            priority=0.5,
            atr_period=10
        )

        # 最大値設定の遺伝子
        max_gene = PositionSizingGene(
            lookback_period=200,
            optimal_f_multiplier=0.75,
            atr_multiplier=5.0,
            risk_per_trade=0.1,
            fixed_ratio=1.0,
            min_position_size=0.1,
            max_position_size=50.0,
            priority=1.5,
            atr_period=30
        )

        # 突然変異適用
        mutated_min = mutate_position_sizing_gene(min_gene, mutation_rate=1.0)
        mutated_max = mutate_position_sizing_gene(max_gene, mutation_rate=1.0)

        # 範囲内制約が機能するか確認
        assert 50 <= mutated_min.lookback_period <= 200
        assert 0.25 <= mutated_min.optimal_f_multiplier <= 0.75
        assert 50 <= mutated_max.lookback_period <= 200
        assert 0.25 <= mutated_max.optimal_f_multiplier <= 0.75


class TestTPSLGeneMutation:
    """TPSLGene 突然変異テスト"""

    def test_mutate_tpsl_gene_basic(self):
        """基本的なTPSL突然変異機能をテスト"""
        # 元のTPSL遺伝子を作成
        original_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            risk_reward_ratio=2.0,
            base_stop_loss=0.03,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=3.0,
            confidence_threshold=0.7,
            priority=1.0,
            lookback_period=100,
            atr_period=14,
            enabled=True
        )

        # 突然変異適用
        mutated_gene = mutate_tpsl_gene(original_gene, mutation_rate=1.0)

        # 型チェック
        assert isinstance(mutated_gene, TPSLGene)

        # 範囲内チェック
        assert 0.005 <= mutated_gene.stop_loss_pct <= 0.15
        assert 0.01 <= mutated_gene.take_profit_pct <= 0.3
        assert 1.0 <= mutated_gene.risk_reward_ratio <= 10.0
        assert 0.01 <= mutated_gene.base_stop_loss <= 0.06
        assert 0.5 <= mutated_gene.atr_multiplier_sl <= 3.0
        assert 1.0 <= mutated_gene.atr_multiplier_tp <= 5.0
        assert 0.1 <= mutated_gene.confidence_threshold <= 0.9
        assert 0.5 <= mutated_gene.priority <= 1.5
        assert 50 <= mutated_gene.lookback_period <= 200
        assert 10 <= mutated_gene.atr_period <= 30

    def test_mutate_tpsl_gene_method_weights_mutation(self):
        """method_weights突然変異テスト"""
        original_gene = TPSLGene(
            method_weights={"fixed": 0.25, "risk_reward": 0.35, "volatility": 0.25, "statistical": 0.15}
        )

        # 突然変異率1.0で必ず突然変異が発生
        mutated_gene = mutate_tpsl_gene(original_gene, mutation_rate=1.0)

        # method_weightsの合計が1.0になっているか確認
        total_weight = sum(mutated_gene.method_weights.values())
        assert abs(total_weight - 1.0) < 0.01  # 浮動小数点誤差考慮

        # すべてのウェイトが0以上1以下か確認
        for weight in mutated_gene.method_weights.values():
            assert 0.0 <= weight <= 1.0

    def test_mutate_tpsl_gene_zero_mutation_rate(self):
        """突然変異率0%のテスト"""
        original_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
            risk_reward_ratio=2.0
        )

        mutated_gene = mutate_tpsl_gene(original_gene, mutation_rate=0.0)

        # 突然変異率0%なので値が変化しないはず
        assert mutated_gene.stop_loss_pct == original_gene.stop_loss_pct
        assert mutated_gene.take_profit_pct == original_gene.take_profit_pct
        assert mutated_gene.risk_reward_ratio == original_gene.risk_reward_ratio

    def test_mutate_tpsl_gene_enum_mutation(self):
        """Enumフィールドの突然変異テスト"""
        original_gene = TPSLGene(method=TPSLMethod.FIXED_PERCENTAGE)

        mutated_gene = mutate_tpsl_gene(original_gene, mutation_rate=1.0)

        # Enum型は変更される可能性がある
        assert isinstance(mutated_gene.method, TPSLMethod)

    def test_mutate_tpsl_gene_boundary_values(self):
        """境界値テスト"""
        # 最小値設定の遺伝子
        min_gene = TPSLGene(
            stop_loss_pct=0.005,
            take_profit_pct=0.01,
            risk_reward_ratio=1.0,
            base_stop_loss=0.01,
            atr_multiplier_sl=0.5,
            atr_multiplier_tp=1.0,
            confidence_threshold=0.1,
            priority=0.5,
            lookback_period=50,
            atr_period=10
        )

        # 最大値設定の遺伝子
        max_gene = TPSLGene(
            stop_loss_pct=0.15,
            take_profit_pct=0.3,
            risk_reward_ratio=10.0,
            base_stop_loss=0.06,
            atr_multiplier_sl=3.0,
            atr_multiplier_tp=5.0,
            confidence_threshold=0.9,
            priority=1.5,
            lookback_period=200,
            atr_period=30
        )

        # 突然変異適用
        mutated_min = mutate_tpsl_gene(min_gene, mutation_rate=1.0)
        mutated_max = mutate_tpsl_gene(max_gene, mutation_rate=1.0)

        # 範囲内制約が機能するか確認
        assert 0.005 <= mutated_min.stop_loss_pct <= 0.15
        assert 0.01 <= mutated_max.take_profit_pct <= 0.3
        assert 1.0 <= mutated_min.risk_reward_ratio <= 10.0
        assert 1.0 <= mutated_max.risk_reward_ratio <= 10.0


class TestGeneMutationEdgeCases:
    """エッジケーステスト"""

    def test_mutation_with_invalid_numeric_values(self):
        """無効な数値での突然変異テスト"""
        # 不正な値を持つ遺伝子を作成
        invalid_gene = PositionSizingGene(
            lookback_period=-10,  # 負の値
            optimal_f_multiplier=10.0,  # 範囲外
            risk_per_trade=-0.5  # 負の値
        )

        # 突然変異適用
        mutated_gene = mutate_position_sizing_gene(invalid_gene, mutation_rate=1.0)

        # 範囲内に修正されているか確認
        assert 50 <= mutated_gene.lookback_period <= 200
        assert 0.25 <= mutated_gene.optimal_f_multiplier <= 0.75
        assert 0.001 <= mutated_gene.risk_per_trade <= 0.1

    def test_mutation_with_float_precision(self):
        """浮動小数点精度テスト"""
        original_gene = TPSLGene(
            method_weights={
                "fixed": 0.249999999,
                "risk_reward": 0.35,
                "volatility": 0.25,
                "statistical": 0.150000001
            },
            confidence_threshold=0.7000000001
        )

        # method_weightsの合計が1を超える場合もテスト
        high_total_gene = TPSLGene(
            method_weights={
                "fixed": 0.4,
                "risk_reward": 0.4,
                "volatility": 0.4,
                "statistical": 0.3  # 合計1.5（意図的に無効）
            }
        )

        mutated_gene = mutate_tpsl_gene(original_gene, mutation_rate=0.5)
        mutated_high = mutate_tpsl_gene(high_total_gene, mutation_rate=1.0)

        # 正規化が機能するか確認
        total_weight = sum(mutated_gene.method_weights.values())
        assert abs(total_weight - 1.0) < 0.01

        total_high = sum(mutated_high.method_weights.values())
        assert abs(total_high - 1.0) < 0.01