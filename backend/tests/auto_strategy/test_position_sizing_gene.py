"""
PositionSizingGeneクラスのテスト
"""

import pytest
import sys
import os

# テスト対象のモジュールをインポートするためのパス設定
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "app",
        "core",
        "services",
        "auto_strategy",
        "models",
    )
)

from position_sizing_gene import (
    PositionSizingGene,
    PositionSizingMethod,
    create_random_position_sizing_gene,
    crossover_position_sizing_genes,
    mutate_position_sizing_gene,
)


class TestPositionSizingGene:
    """PositionSizingGeneクラスのテスト"""

    def test_default_initialization(self):
        """デフォルト初期化のテスト"""
        gene = PositionSizingGene()
        
        assert gene.method == PositionSizingMethod.FIXED_RATIO
        assert gene.lookback_period == 100
        assert gene.optimal_f_multiplier == 0.5
        assert gene.atr_period == 14
        assert gene.atr_multiplier == 2.0
        assert gene.risk_per_trade == 0.02
        assert gene.fixed_ratio == 0.1
        assert gene.fixed_quantity == 1.0
        assert gene.min_position_size == 0.01
        assert gene.max_position_size == 1.0
        assert gene.enabled is True
        assert gene.priority == 1.0

    def test_custom_initialization(self):
        """カスタム初期化のテスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=150,
            optimal_f_multiplier=0.3,
            atr_period=20,
            atr_multiplier=3.0,
            risk_per_trade=0.03,
            fixed_ratio=0.15,
            fixed_quantity=2.0,
            min_position_size=0.02,
            max_position_size=1.5,
            enabled=False,
            priority=0.8,
        )
        
        assert gene.method == PositionSizingMethod.VOLATILITY_BASED
        assert gene.lookback_period == 150
        assert gene.optimal_f_multiplier == 0.3
        assert gene.atr_period == 20
        assert gene.atr_multiplier == 3.0
        assert gene.risk_per_trade == 0.03
        assert gene.fixed_ratio == 0.15
        assert gene.fixed_quantity == 2.0
        assert gene.min_position_size == 0.02
        assert gene.max_position_size == 1.5
        assert gene.enabled is False
        assert gene.priority == 0.8

    def test_to_dict(self):
        """辞書変換のテスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.HALF_OPTIMAL_F,
            lookback_period=120,
            optimal_f_multiplier=0.4,
        )
        
        result = gene.to_dict()
        
        assert result["method"] == "half_optimal_f"
        assert result["lookback_period"] == 120
        assert result["optimal_f_multiplier"] == 0.4
        assert result["atr_period"] == 14  # デフォルト値
        assert result["enabled"] is True

    def test_from_dict(self):
        """辞書からの復元のテスト"""
        data = {
            "method": "volatility_based",
            "lookback_period": 80,
            "optimal_f_multiplier": 0.6,
            "atr_period": 25,
            "atr_multiplier": 2.5,
            "risk_per_trade": 0.025,
            "fixed_ratio": 0.12,
            "fixed_quantity": 1.5,
            "min_position_size": 0.015,
            "max_position_size": 1.2,
            "enabled": False,
            "priority": 1.1,
        }
        
        gene = PositionSizingGene.from_dict(data)
        
        assert gene.method == PositionSizingMethod.VOLATILITY_BASED
        assert gene.lookback_period == 80
        assert gene.optimal_f_multiplier == 0.6
        assert gene.atr_period == 25
        assert gene.atr_multiplier == 2.5
        assert gene.risk_per_trade == 0.025
        assert gene.fixed_ratio == 0.12
        assert gene.fixed_quantity == 1.5
        assert gene.min_position_size == 0.015
        assert gene.max_position_size == 1.2
        assert gene.enabled is False
        assert gene.priority == 1.1

    def test_validate_valid_gene(self):
        """有効な遺伝子のバリデーションテスト"""
        gene = PositionSizingGene()
        is_valid, errors = gene.validate()
        
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_parameters(self):
        """無効なパラメータのバリデーションテスト"""
        gene = PositionSizingGene(
            lookback_period=5,  # 範囲外（10未満）
            optimal_f_multiplier=1.5,  # 範囲外（1.0超過）
            atr_period=60,  # 範囲外（50超過）
            atr_multiplier=15.0,  # 範囲外（10.0超過）
            risk_per_trade=0.15,  # 範囲外（0.1超過）
            fixed_ratio=0.6,  # 範囲外（0.5超過）
            fixed_quantity=15.0,  # 範囲外（10.0超過）
            min_position_size=1.5,  # 範囲外（1.0超過）
            max_position_size=0.005,  # min_position_size未満
            priority=3.0,  # 範囲外（2.0超過）
        )
        
        is_valid, errors = gene.validate()
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("lookback_period" in error for error in errors)
        assert any("optimal_f_multiplier" in error for error in errors)
        assert any("atr_period" in error for error in errors)
        assert any("atr_multiplier" in error for error in errors)
        assert any("risk_per_trade" in error for error in errors)
        assert any("fixed_ratio" in error for error in errors)
        assert any("fixed_quantity" in error for error in errors)
        assert any("min_position_size" in error for error in errors)
        assert any("max_position_size" in error for error in errors)
        assert any("priority" in error for error in errors)

    def test_calculate_position_size_disabled(self):
        """無効化された遺伝子のポジションサイズ計算テスト"""
        gene = PositionSizingGene(enabled=False, min_position_size=0.05)
        
        result = gene.calculate_position_size(
            account_balance=10000.0,
            current_price=50000.0
        )
        
        assert result == 0.05  # min_position_sizeが返される

    def test_calculate_position_size_fixed_ratio(self):
        """固定比率方式のポジションサイズ計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.2,
            min_position_size=0.01,
            max_position_size=5.0,
        )
        
        result = gene.calculate_position_size(
            account_balance=10000.0,
            current_price=50000.0
        )
        
        expected = 10000.0 * 0.2  # 2000.0
        assert result == expected

    def test_calculate_position_size_fixed_quantity(self):
        """固定枚数方式のポジションサイズ計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_QUANTITY,
            fixed_quantity=3.0,
            min_position_size=0.01,
            max_position_size=5.0,
        )
        
        result = gene.calculate_position_size(
            account_balance=10000.0,
            current_price=50000.0
        )
        
        assert result == 3.0

    def test_calculate_position_size_volatility_based(self):
        """ボラティリティベース方式のポジションサイズ計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            atr_multiplier=2.0,
            risk_per_trade=0.02,
            min_position_size=0.01,
            max_position_size=5.0,
        )
        
        market_data = {"atr": 1000.0}  # ATR値
        
        result = gene.calculate_position_size(
            account_balance=10000.0,
            current_price=50000.0,
            market_data=market_data
        )
        
        # risk_amount = 10000 * 0.02 = 200
        # position_size = 200 / (1000 * 2.0) = 0.1
        expected = 200.0 / (1000.0 * 2.0)
        assert result == expected

    def test_calculate_position_size_half_optimal_f_insufficient_data(self):
        """ハーフオプティマルF方式（データ不足）のポジションサイズ計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.HALF_OPTIMAL_F,
            fixed_ratio=0.1,  # フォールバック用
            min_position_size=0.01,
            max_position_size=5.0,
        )
        
        # データ不足の場合
        trade_history = []
        
        result = gene.calculate_position_size(
            account_balance=10000.0,
            current_price=50000.0,
            trade_history=trade_history
        )
        
        # データ不足時は固定比率にフォールバック
        expected = 10000.0 * 0.1  # 1000.0
        assert result == expected

    def test_calculate_position_size_half_optimal_f_with_data(self):
        """ハーフオプティマルF方式（データあり）のポジションサイズ計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.HALF_OPTIMAL_F,
            optimal_f_multiplier=0.5,
            min_position_size=0.01,
            max_position_size=5.0,
        )
        
        # 勝率60%、平均利益100、平均損失50のサンプルデータ
        trade_history = [
            {"pnl": 100}, {"pnl": -50}, {"pnl": 100}, {"pnl": 100}, {"pnl": -50},
            {"pnl": 100}, {"pnl": -50}, {"pnl": 100}, {"pnl": -50}, {"pnl": 100},
        ]
        
        result = gene.calculate_position_size(
            account_balance=10000.0,
            current_price=50000.0,
            trade_history=trade_history
        )
        
        # 勝率 = 6/10 = 0.6
        # 平均利益 = 100, 平均損失 = 50
        # optimal_f = (0.6 * 100 - 0.4 * 50) / 100 = (60 - 20) / 100 = 0.4
        # half_optimal_f = 0.4 * 0.5 = 0.2
        # position_size = 10000 * 0.2 = 2000
        assert result > 0  # 正の値が返されることを確認

    def test_apply_size_limits(self):
        """サイズ制限の適用テスト"""
        gene = PositionSizingGene(
            min_position_size=0.1,
            max_position_size=2.0,
        )
        
        # 最小値未満
        result1 = gene._apply_size_limits(0.05)
        assert result1 == 0.1
        
        # 範囲内
        result2 = gene._apply_size_limits(1.0)
        assert result2 == 1.0
        
        # 最大値超過
        result3 = gene._apply_size_limits(3.0)
        assert result3 == 2.0


class TestPositionSizingGeneUtilities:
    """ポジションサイジング遺伝子のユーティリティ関数のテスト"""

    def test_create_random_position_sizing_gene(self):
        """ランダム遺伝子生成のテスト"""
        gene = create_random_position_sizing_gene()
        
        assert isinstance(gene, PositionSizingGene)
        assert gene.method in list(PositionSizingMethod)
        assert 50 <= gene.lookback_period <= 200
        assert 0.25 <= gene.optimal_f_multiplier <= 0.75
        assert 10 <= gene.atr_period <= 30
        assert 1.0 <= gene.atr_multiplier <= 4.0
        assert 0.01 <= gene.risk_per_trade <= 0.05
        assert 0.05 <= gene.fixed_ratio <= 0.3
        assert 0.1 <= gene.fixed_quantity <= 5.0
        assert 0.01 <= gene.min_position_size <= 0.05
        assert 0.5 <= gene.max_position_size <= 2.0
        assert gene.enabled is True
        assert 0.5 <= gene.priority <= 1.5

    def test_crossover_position_sizing_genes(self):
        """交叉操作のテスト"""
        parent1 = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            lookback_period=100,
            optimal_f_multiplier=0.3,
            fixed_ratio=0.1,
        )
        
        parent2 = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=150,
            optimal_f_multiplier=0.7,
            fixed_ratio=0.2,
        )
        
        child1, child2 = crossover_position_sizing_genes(parent1, parent2)
        
        assert isinstance(child1, PositionSizingGene)
        assert isinstance(child2, PositionSizingGene)
        
        # 方式は親のいずれかから選択される
        assert child1.method in [parent1.method, parent2.method]
        assert child2.method in [parent1.method, parent2.method]
        
        # 数値パラメータは平均値になる
        expected_optimal_f = (parent1.optimal_f_multiplier + parent2.optimal_f_multiplier) / 2
        assert child1.optimal_f_multiplier == expected_optimal_f
        assert child2.optimal_f_multiplier == expected_optimal_f
        
        expected_fixed_ratio = (parent1.fixed_ratio + parent2.fixed_ratio) / 2
        assert child1.fixed_ratio == expected_fixed_ratio
        assert child2.fixed_ratio == expected_fixed_ratio

    def test_mutate_position_sizing_gene(self):
        """突然変異操作のテスト"""
        original = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            lookback_period=100,
            optimal_f_multiplier=0.5,
            atr_period=14,
            atr_multiplier=2.0,
            risk_per_trade=0.02,
            fixed_ratio=0.1,
            fixed_quantity=1.0,
            min_position_size=0.01,
            max_position_size=1.0,
            priority=1.0,
        )
        
        # 突然変異率100%でテスト
        mutated = mutate_position_sizing_gene(original, mutation_rate=1.0)
        
        assert isinstance(mutated, PositionSizingGene)
        
        # 元の遺伝子とは異なるオブジェクトであることを確認
        assert mutated is not original
        
        # パラメータが範囲内に収まっていることを確認
        assert 50 <= mutated.lookback_period <= 200
        assert 0.25 <= mutated.optimal_f_multiplier <= 0.75
        assert 10 <= mutated.atr_period <= 30
        assert 1.0 <= mutated.atr_multiplier <= 4.0
        assert 0.01 <= mutated.risk_per_trade <= 0.05
        assert 0.05 <= mutated.fixed_ratio <= 0.3
        assert 0.1 <= mutated.fixed_quantity <= 5.0
        assert 0.01 <= mutated.min_position_size <= 0.1
        assert mutated.min_position_size <= mutated.max_position_size <= 2.0
        assert 0.5 <= mutated.priority <= 1.5

    def test_mutate_position_sizing_gene_low_rate(self):
        """低い突然変異率でのテスト"""
        original = PositionSizingGene()
        
        # 突然変異率0%でテスト
        mutated = mutate_position_sizing_gene(original, mutation_rate=0.0)
        
        # パラメータが変更されていないことを確認
        assert mutated.method == original.method
        assert mutated.lookback_period == original.lookback_period
        assert mutated.optimal_f_multiplier == original.optimal_f_multiplier
        assert mutated.atr_period == original.atr_period
        assert mutated.atr_multiplier == original.atr_multiplier
        assert mutated.risk_per_trade == original.risk_per_trade
        assert mutated.fixed_ratio == original.fixed_ratio
        assert mutated.fixed_quantity == original.fixed_quantity
        assert mutated.min_position_size == original.min_position_size
        assert mutated.max_position_size == original.max_position_size
        assert mutated.priority == original.priority


if __name__ == "__main__":
    pytest.main([__file__])
