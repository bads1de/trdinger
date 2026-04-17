"""
TPSLGene Factory Function Tests

Test logic for create_random_tpsl_gene
"""

from app.services.auto_strategy.genes import (
    TPSLGene,
    create_random_tpsl_gene,
)


class TestCreateRandomTPSLGene:
    """create_random_tpsl_geneのテスト"""

    def test_returns_tpsl_gene(self):
        """引数なしで正常に動作する"""
        result = create_random_tpsl_gene()
        assert isinstance(result, TPSLGene)

    def test_generates_valid_values(self):
        """有効な値を生成する"""
        result = create_random_tpsl_gene()

        # 値が有効な範囲内であることを確認
        assert 0.01 <= result.stop_loss_pct <= 0.08
        assert 0.02 <= result.take_profit_pct <= 0.15
        assert 1.2 <= result.risk_reward_ratio <= 4.0
        assert 0.01 <= result.base_stop_loss <= 0.06
        assert 1.0 <= result.atr_multiplier_sl <= 3.0
        assert 2.0 <= result.atr_multiplier_tp <= 5.0
        assert 10 <= result.atr_period <= 30
        assert 50 <= result.lookback_period <= 200
        assert 0.5 <= result.confidence_threshold <= 0.9
        assert 0.5 <= result.priority <= 1.5
