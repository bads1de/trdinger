"""
TPSLGene Factory Function Tests

Test logic for create_random_tpsl_gene with constraints
"""

import pytest
from unittest.mock import Mock
from app.services.auto_strategy.genes import TPSLGene, TPSLMethod, create_random_tpsl_gene

# テスト用の単純なConfigクラス
class Config:
    pass

class TestCreateRandomTPSLGene:
    """create_random_tpsl_geneのテスト"""

    def test_returns_tpsl_gene(self):
        """引数なしで正常に動作する"""
        result = create_random_tpsl_gene()
        assert isinstance(result, TPSLGene)

    def test_applies_method_constraints(self):
        """メソッド制約を適用"""
        config = Config()
        config.tpsl_method_constraints = ["risk_reward_ratio", "volatility_based"]
        
        # 複数回実行して制約が適用されることを確認
        methods_used = set()
        for _ in range(50):
            result = create_random_tpsl_gene(config)
            methods_used.add(result.method)

        # 使用されたメソッドが制約内であることを確認
        for method in methods_used:
            assert method.value in ["risk_reward_ratio", "volatility_based"]

    def test_applies_sl_range_constraints(self):
        """SL範囲制約を適用"""
        config = Config()
        config.tpsl_sl_range = (0.01, 0.02)
        
        result = create_random_tpsl_gene(config)

        # SL値が範囲内であることを確認
        assert 0.01 <= result.stop_loss_pct <= 0.02
        assert 0.01 <= result.base_stop_loss <= 0.02

    def test_applies_tp_range_constraints(self):
        """TP範囲制約を適用"""
        config = Config()
        config.tpsl_tp_range = (0.03, 0.05)
        
        result = create_random_tpsl_gene(config)

        # TP値が範囲内であることを確認
        assert 0.03 <= result.take_profit_pct <= 0.05

    def test_applies_rr_range_constraints(self):
        """リスクリワード比範囲制約を適用"""
        config = Config()
        config.tpsl_rr_range = (1.5, 3.0)
        
        result = create_random_tpsl_gene(config)

        # RR値が範囲内であることを確認
        assert 1.5 <= result.risk_reward_ratio <= 3.0

    def test_applies_atr_multiplier_range_constraints(self):
        """ATR倍率範囲制約を適用"""
        config = Config()
        config.tpsl_atr_multiplier_range = (1.0, 2.0)
        
        result = create_random_tpsl_gene(config)

        # ATR倍率SLが範囲内であることを確認
        assert 1.0 <= result.atr_multiplier_sl <= 2.0
        # ATR倍率TP範囲（1.5x ~ 2.0x）
        assert 1.5 <= result.atr_multiplier_tp <= 4.0

    def test_all_constraints_applied(self):
        """全ての制約が同時に適用される"""
        config = Config()
        config.tpsl_method_constraints = ["risk_reward_ratio"]
        config.tpsl_sl_range = (0.01, 0.015)
        config.tpsl_tp_range = (0.04, 0.05)
        config.tpsl_rr_range = (2.5, 3.5)
        config.tpsl_atr_multiplier_range = (1.5, 2.5)

        result = create_random_tpsl_gene(config)

        # メソッド制約
        assert result.method == TPSLMethod.RISK_REWARD_RATIO
        # SL範囲制約
        assert 0.01 <= result.stop_loss_pct <= 0.015
        assert 0.01 <= result.base_stop_loss <= 0.015
        # TP範囲制約
        assert 0.04 <= result.take_profit_pct <= 0.05
        # RR範囲制約
        assert 2.5 <= result.risk_reward_ratio <= 3.5
        # ATR倍率範囲制約
        assert 1.5 <= result.atr_multiplier_sl <= 2.5
        assert 2.25 <= result.atr_multiplier_tp <= 5.0

    def test_hasattr_check_for_missing_attributes(self):
        """属性が存在しない場合のhasattrチェック"""
        config = Config()
        # 属性を設定しない
        
        # 属性がなくてもエラーにならないことを確認
        result = create_random_tpsl_gene(config)
        assert result is not None
