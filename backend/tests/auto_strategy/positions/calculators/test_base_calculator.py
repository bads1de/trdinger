import pytest
from unittest.mock import MagicMock
from app.services.auto_strategy.positions.calculators.base_calculator import BaseCalculator

class MockCalculator(BaseCalculator):
    """テスト用の具象クラス"""
    def calculate(self, gene, account_balance, current_price, **kwargs):
        # 単純に1を返すダミー実装
        return self._create_calculation_result(1.0, {"method": "mock"}, [], gene)

class TestBaseCalculator:
    @pytest.fixture
    def calculator(self):
        return MockCalculator()

    def test_get_param(self, calculator):
        gene = MagicMock()
        gene.exists = 10
        # 存在する属性
        assert calculator._get_param(gene, "exists", 0) == 10
        
        # 存在しない属性へのアクセスでdefaultが返ることを確認
        # MagicMockはデフォルトですべての属性を生成するため、明示的に削除するか
        # 属性リストを指定したMockを使用する
        del gene.not_exists
        assert calculator._get_param(gene, "not_exists", 5) == 5

    def test_get_risk_params(self, calculator):
        # 必要な属性だけをセットしたプレーンなオブジェクト
        class DummyGene:
            pass
        gene = DummyGene()
        gene.var_confidence = 0.99
        # 他の属性（max_var_ratioなど）は定義されていない
        
        params = calculator._get_risk_params(gene)
        assert params["var_confidence"] == 0.99
        assert params["max_var_ratio"] == 0.0 # デフォルト値が返ることを期待

    def test_apply_size_limits(self, calculator):
        gene = MagicMock()
        gene.min_position_size = 0.1
        gene.max_position_size = 10.0
        
        # 最小値以下
        res = calculator._apply_size_limits_and_finalize(0.05, {}, [], gene)
        assert res["position_size"] == 0.1
        
        # 最大値以上
        res = calculator._apply_size_limits_and_finalize(15.0, {}, [], gene)
        assert res["position_size"] == 10.0
        
        # 範囲内
        res = calculator._apply_size_limits_and_finalize(5.0, {}, [], gene)
        assert res["position_size"] == 5.0

    def test_safe_calculate_with_price_check(self, calculator):
        calc_fn = MagicMock(return_value=100.0)
        
        # 正常価格
        res = calculator._safe_calculate_with_price_check(calc_fn, 50000.0)
        assert res == 100.0
        calc_fn.assert_called_once()
        
        # 不正価格 (0)
        warnings = []
        res = calculator._safe_calculate_with_price_check(
            calc_fn, 0.0, fallback_value=0.0, warnings_list=warnings
        )
        assert res == 0.0
        assert len(warnings) > 0
