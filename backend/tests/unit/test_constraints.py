"""
制約システムのテスト
"""

import pytest
from app.services.indicators.constraints import (
    OrderConstraint,
    RangeConstraint,
    DependencyConstraint,
    ConstraintEngine,
    constraint_engine,
)


class TestOrderConstraint:
    """順序制約のテスト"""

    def test_order_constraint_less_than(self):
        """< 制約のテスト"""
        constraint = OrderConstraint("fast_period", "slow_period", "<")
        
        # 制約違反のケース
        params = {"fast_period": 26, "slow_period": 12}
        result = constraint.apply(params)
        
        assert result["fast_period"] == 26
        assert result["slow_period"] == 27  # fast_period + margin(1)
        assert constraint.validate(result)

    def test_order_constraint_validation(self):
        """順序制約の検証テスト"""
        constraint = OrderConstraint("fast_period", "slow_period", "<")
        
        # 有効なケース
        valid_params = {"fast_period": 12, "slow_period": 26}
        assert constraint.validate(valid_params)
        
        # 無効なケース
        invalid_params = {"fast_period": 26, "slow_period": 12}
        assert not constraint.validate(invalid_params)

    def test_order_constraint_missing_params(self):
        """パラメータが不足している場合のテスト"""
        constraint = OrderConstraint("fast_period", "slow_period", "<")
        
        # パラメータが不足している場合は何もしない
        params = {"fast_period": 12}
        result = constraint.apply(params)
        assert result == params
        assert constraint.validate(params)  # 不足している場合は有効とみなす


class TestRangeConstraint:
    """値域制約のテスト"""

    def test_range_constraint_apply(self):
        """値域制約の適用テスト"""
        constraint = RangeConstraint("matype", 0, 8)
        
        params = {"matype": 15}  # 範囲外の値
        result = constraint.apply(params)
        
        assert 0 <= result["matype"] <= 8

    def test_range_constraint_validation(self):
        """値域制約の検証テスト"""
        constraint = RangeConstraint("matype", 0, 8)
        
        # 有効なケース
        valid_params = {"matype": 5}
        assert constraint.validate(valid_params)
        
        # 無効なケース
        invalid_params = {"matype": 15}
        assert not constraint.validate(invalid_params)

    def test_range_constraint_missing_param(self):
        """パラメータが不足している場合のテスト"""
        constraint = RangeConstraint("matype", 0, 8)
        
        params = {"other_param": 10}
        result = constraint.apply(params)
        assert result == params
        assert constraint.validate(params)


class TestDependencyConstraint:
    """依存関係制約のテスト"""

    def test_dependency_constraint(self):
        """依存関係制約のテスト"""
        # source_paramの2倍をtarget_paramに設定する制約
        constraint = DependencyConstraint(
            "source_param", 
            "target_param", 
            lambda x: x * 2
        )
        
        params = {"source_param": 10}
        result = constraint.apply(params)
        
        assert result["target_param"] == 20
        assert constraint.validate(result)


class TestConstraintEngine:
    """制約エンジンのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.engine = ConstraintEngine()

    def test_register_and_apply_constraints(self):
        """制約の登録と適用テスト"""
        constraints = [
            OrderConstraint("fast_period", "slow_period", "<"),
            RangeConstraint("signal_period", 2, 50)
        ]
        
        self.engine.register_constraints("TEST_MACD", constraints)
        
        params = {
            "fast_period": 26,
            "slow_period": 12,
            "signal_period": 100
        }
        
        result = self.engine.apply_constraints("TEST_MACD", params)
        
        # 順序制約が適用されている
        assert result["fast_period"] < result["slow_period"]
        # 値域制約が適用されている
        assert 2 <= result["signal_period"] <= 50

    def test_validate_constraints(self):
        """制約検証のテスト"""
        constraints = [
            OrderConstraint("fast_period", "slow_period", "<")
        ]
        
        self.engine.register_constraints("TEST_MACD", constraints)
        
        # 有効なパラメータ
        valid_params = {"fast_period": 12, "slow_period": 26}
        assert self.engine.validate_constraints("TEST_MACD", valid_params)
        
        # 無効なパラメータ
        invalid_params = {"fast_period": 26, "slow_period": 12}
        assert not self.engine.validate_constraints("TEST_MACD", invalid_params)

    def test_unknown_indicator(self):
        """未知のインディケーターのテスト"""
        params = {"period": 14}
        
        # 制約が登録されていないインディケーターは何もしない
        result = self.engine.apply_constraints("UNKNOWN", params)
        assert result == params
        assert self.engine.validate_constraints("UNKNOWN", params)

    def test_get_constraints(self):
        """制約取得のテスト"""
        constraints = [OrderConstraint("fast_period", "slow_period", "<")]
        self.engine.register_constraints("TEST", constraints)
        
        retrieved = self.engine.get_constraints("TEST")
        assert len(retrieved) == 1
        assert isinstance(retrieved[0], OrderConstraint)

    def test_list_indicators(self):
        """インディケーター一覧取得のテスト"""
        self.engine.register_constraints("TEST1", [])
        self.engine.register_constraints("TEST2", [])
        
        indicators = self.engine.list_indicators()
        assert "TEST1" in indicators
        assert "TEST2" in indicators


class TestDefaultConstraints:
    """デフォルト制約のテスト"""

    def test_macd_constraints(self):
        """MACD制約のテスト"""
        params = {"fast_period": 26, "slow_period": 12, "signal_period": 9}
        
        result = constraint_engine.apply_constraints("MACD", params)
        
        # fast_period < slow_period が保証されている
        assert result["fast_period"] < result["slow_period"]
        assert constraint_engine.validate_constraints("MACD", result)

    def test_stoch_constraints(self):
        """Stochastic制約のテスト"""
        params = {
            "fastk_period": 5,
            "slowk_period": 3,
            "slowk_matype": 15,  # 範囲外
            "slowd_period": 3,
            "slowd_matype": 20   # 範囲外
        }
        
        result = constraint_engine.apply_constraints("STOCH", params)
        
        # matype が 0-8 の範囲に制限されている
        assert 0 <= result["slowk_matype"] <= 8
        assert 0 <= result["slowd_matype"] <= 8
        assert constraint_engine.validate_constraints("STOCH", result)

    def test_macdext_constraints(self):
        """MACDEXT制約のテスト"""
        params = {
            "fast_period": 26,
            "slow_period": 12,
            "signal_period": 9,
            "fast_ma_type": 15,
            "slow_ma_type": 20,
            "signal_ma_type": 25
        }
        
        result = constraint_engine.apply_constraints("MACDEXT", params)
        
        # 順序制約
        assert result["fast_period"] < result["slow_period"]
        # MA種別制約
        assert 0 <= result["fast_ma_type"] <= 8
        assert 0 <= result["slow_ma_type"] <= 8
        assert 0 <= result["signal_ma_type"] <= 8
        assert constraint_engine.validate_constraints("MACDEXT", result)
