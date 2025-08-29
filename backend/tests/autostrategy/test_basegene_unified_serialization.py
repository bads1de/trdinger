"""
BaseGene/BaseConfig 統合シリアライズ機能テスト

BaseGeneとBaseConfigの統一されたシリアライズ機能をテストします。
"""

import pytest
from typing import Dict, Any
from dataclasses import dataclass
from app.services.auto_strategy.utils.common_utils import BaseGene
from app.services.auto_strategy.models.strategy_models import (
    PositionSizingGene,
    TPSLGene,
    PositionSizingMethod,
    TPSLMethod
)
from app.services.auto_strategy.config.auto_strategy_config import BaseConfig


class TestGene(BaseGene):
    """テスト用のGeneクラス"""
    name: str = "test"
    value: int = 42
    enabled: bool = True
    method: str = "test_method"

    def _validate_parameters(self, errors):
        pass


class TestConfig(BaseConfig):
    """テスト用のConfigクラス"""
    name: str = "test_config"
    value: int = 100
    enabled: bool = True

    def get_default_values(self) -> Dict[str, Any]:
        return {
            "name": "default_config",
            "value": 50,
            "enabled": False,
        }


class TestBaseGeneUnifiedSerialization:
    """BaseGene統合シリアライズ機能テスト"""

    def test_position_sizing_gene_unified_from_dict(self):
        """PositionSizingGeneのBaseGene統一from_dictテスト"""
        # 既存のPositionSizingGeneデータを準備（Enum値として文字列を使用）
        data = {
            "method": "volatility_based",  # Enumを文字列で表現
            "lookback_period": 150,
            "optimal_f_multiplier": 0.6,
            "atr_period": 20,
            "atr_multiplier": 2.5,
            "risk_per_trade": 0.03,
            "fixed_ratio": 0.15,
            "fixed_quantity": 1.5,
            "min_position_size": 0.02,
            "max_position_size": 50.0,
            "enabled": True,
            "priority": 1.2,
        }

        # BaseGene.from_dictを使用して復元
        restored = PositionSizingGene.from_dict(data)

        # 検証
        assert isinstance(restored, PositionSizingGene)
        assert restored.method == PositionSizingMethod.VOLATILITY_BASED
        assert restored.lookback_period == 150
        assert restored.optimal_f_multiplier == 0.6
        assert restored.atr_period == 20
        assert restored.atr_multiplier == 2.5
        assert restored.risk_per_trade == 0.03
        assert restored.fixed_ratio == 0.15
        assert restored.fixed_quantity == 1.5
        assert restored.min_position_size == 0.02
        assert restored.max_position_size == 50.0
        assert restored.enabled == True
        assert restored.priority == 1.2

    def test_tpsl_gene_unified_from_dict(self):
        """TPSLGeneのBaseGene統一from_dictテスト"""
        data = {
            "method": "risk_reward_ratio",
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.075,
            "risk_reward_ratio": 3.0,
            "base_stop_loss": 0.03,
            "atr_multiplier_sl": 2.2,
            "atr_multiplier_tp": 3.5,
            "atr_period": 18,
            "lookback_period": 120,
            "confidence_threshold": 0.75,
            "method_weights": {
                "fixed": 0.2,
                "risk_reward": 0.4,
                "volatility": 0.3,
                "statistical": 0.1,
            },
            "enabled": True,
            "priority": 0.9,
        }

        # BaseGene.from_dictを使用して復元
        restored = TPSLGene.from_dict(data)

        # 検証
        assert isinstance(restored, TPSLGene)
        assert restored.method == TPSLMethod.RISK_REWARD_RATIO
        assert restored.stop_loss_pct == 0.025
        assert restored.take_profit_pct == 0.075
        assert restored.risk_reward_ratio == 3.0
        assert restored.base_stop_loss == 0.03
        assert restored.atr_multiplier_sl == 2.2
        assert restored.atr_multiplier_tp == 3.5
        assert restored.atr_period == 18
        assert restored.lookback_period == 120
        assert restored.confidence_threshold == 0.75
        assert restored.method_weights == {
            "fixed": 0.2,
            "risk_reward": 0.4,
            "volatility": 0.3,
            "statistical": 0.1,
        }
        assert restored.enabled == True
        assert restored.priority == 0.9

    def test_position_sizing_gene_to_dict_roundtrip(self):
        """PositionSizingGeneのto_dict→from_dictラウンドトリップテスト"""
        # オリジナルインスタンス作成
        original = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=150,
            optimal_f_multiplier=0.6,
            enabled=True,
            priority=1.2,
        )

        # to_dict → from_dict
        data = original.to_dict()
        restored = PositionSizingGene.from_dict(data)

        # 検証
        assert isinstance(restored, PositionSizingGene)
        assert restored.method == original.method
        assert restored.lookback_period == original.lookback_period
        assert restored.optimal_f_multiplier == original.optimal_f_multiplier
        assert restored.enabled == original.enabled
        assert restored.priority == original.priority

    def test_tpsl_gene_to_dict_roundtrip(self):
        """TPSLGeneのto_dict→from_dictラウンドトリップテスト"""
        # オリジナルインスタンス作成
        original = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.025,
            take_profit_pct=0.075,
            enabled=True,
            priority=0.9,
        )

        # to_dict → from_dict
        data = original.to_dict()
        restored = TPSLGene.from_dict(data)

        # 検証
        assert isinstance(restored, TPSLGene)
        assert restored.method == original.method
        assert restored.stop_loss_pct == original.stop_loss_pct
        assert restored.take_profit_pct == original.take_profit_pct
        assert restored.enabled == original.enabled
        assert restored.priority == original.priority

    def test_position_sizing_gene_partial_data_handling(self):
        """PositionSizingGeneの不完全データ処理テスト"""
        # 一部のフィールドのみを含むデータ
        partial_data = {
            "method": "fixed_ratio",
            "enabled": False,
            # 他のフィールドは含まない
        }

        restored = PositionSizingGene.from_dict(partial_data)

        # 部分データでもインスタンスが作成される
        assert isinstance(restored, PositionSizingGene)
        assert restored.method == PositionSizingMethod.FIXED_RATIO
        assert restored.enabled == False
        # 他のフィールドはデフォルト値となる
        assert restored.lookback_period == 100  # デフォルト値

    def test_enum_conversion_error_handling(self):
        """Enum変換エラーハンドリングテスト"""
        data = {
            "method": "invalid_method",  # 存在しないEnum値
            "enabled": True,
        }

        restored = PositionSizingGene.from_dict(data)

        # 無効なEnum値は無視され、デフォルト値が使用される
        assert isinstance(restored, PositionSizingGene)
        assert restored.method == PositionSizingMethod.VOLATILITY_BASED  # デフォルト値
        assert restored.enabled == True

    def test_validate_function_still_works(self):
        """validation機能が依然として動作することを確認"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            risk_per_trade=0.5,  # 範囲外の値
        )

        is_valid, errors = gene.validate()

        # validationが機能していることを確認
        assert is_valid == False
        assert len(errors) > 0


if __name__ == "__main__":
    # 簡易テスト実行
    print("=== BaseGene統合シリアライズ機能テスト ===")

    test = TestBaseGeneUnifiedSerialization()

    try:
        test.test_base_gene_to_dict_from_dict_roundtrip()
        print("✓ BaseGeneラウンドトリップテスト - OK")

        test.test_position_sizing_gene_unified_from_dict()
        print("✓ PositionSizingGene統合from_dictテスト - OK")

        test.test_tpsl_gene_unified_from_dict()
        print("✓ TPSLGene統合from_dictテスト - OK")

        test.test_position_sizing_gene_partial_data_handling()
        print("✓ 不完全データ処理テスト - OK")

        test.test_enum_conversion_error_handling()
        print("✓ Enum変換エラーハンドリングテスト - OK")

        print("\n=== 全テスト成功！ ===")
        print("BaseGene/BaseConfigの統合シリアライズ機能が正常に動作しています。")

    except Exception as e:
        print(f"✗ テスト失敗: {e}")
        raise