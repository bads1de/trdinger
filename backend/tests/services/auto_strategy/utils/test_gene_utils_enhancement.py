import pytest
from unittest.mock import MagicMock
from app.services.auto_strategy.utils.gene_utils import GeneUtils, normalize_parameter, create_default_strategy_gene

class TestGeneUtilsEnhancement:
    def test_normalize_parameter_basic(self):
        # 範囲内: 100 in [1, 200] -> 約0.5
        assert normalize_parameter(100, 1, 200) == pytest.approx(0.497, 0.01)
        
        # 正確な計算: (50-0)/(100-0) = 0.5
        assert normalize_parameter(50, 0, 100) == 0.5

    def test_normalize_parameter_bounds(self):
        # 最小値
        assert normalize_parameter(1, 1, 200) == 0.0
        # 最大値
        assert normalize_parameter(200, 1, 200) == 1.0

    def test_normalize_parameter_out_of_bounds(self):
        # 最小値未満 -> 0.0
        assert normalize_parameter(0, 1, 200) == 0.0
        # 最大値超過 -> 1.0
        assert normalize_parameter(300, 1, 200) == 1.0

    def test_normalize_parameter_invalid_type(self):
        # 文字列などを渡すと警告が出てデフォルト値0.1が返る
        assert normalize_parameter("invalid") == 0.1
        assert normalize_parameter(None) == 0.1

    def test_create_default_strategy_gene(self):
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_class.create_default.return_value = mock_instance
        
        result = create_default_strategy_gene(mock_class)
        
        assert result == mock_instance
        mock_class.create_default.assert_called_once()

    def test_class_method_access(self):
        # GeneUtilsクラス経由でのアクセスも確認
        assert GeneUtils.normalize_parameter(50, 0, 100) == 0.5
