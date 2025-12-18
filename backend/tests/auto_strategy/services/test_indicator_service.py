"""
IndicatorServiceのユニットテスト
"""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

from app.services.auto_strategy.services.indicator_service import IndicatorCalculator
from app.services.auto_strategy.genes.indicator import IndicatorGene


class TestIndicatorCalculator:
    """IndicatorCalculatorのテストクラス"""

    @pytest.fixture
    def mock_indicator_service(self):
        service = Mock()
        # SMAの計算結果をシミュレート
        service.calculate_indicator.return_value = pd.Series([10, 11, 12])
        return service

    @pytest.fixture
    def calculator(self, mock_indicator_service):
        return IndicatorCalculator(technical_indicator_service=mock_indicator_service)

    def test_calculate_indicator_basic(self, calculator, mock_indicator_service):
        """基本的な指標計算の委譲テスト"""
        # データの準備
        mock_data = Mock()
        mock_data.df = pd.DataFrame({
            "Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0, 1, 2],
            "Close": [1.5, 2.5, 3.5], "Volume": [100, 200, 300]
        })
        
        result = calculator.calculate_indicator(mock_data, "SMA", {"period": 20})
        
        assert isinstance(result, pd.Series)
        mock_indicator_service.calculate_indicator.assert_called_once()

    def test_init_indicator_registers_on_strategy(self, calculator):
        """戦略インスタンスへの指標登録テスト"""
        # 遺伝子の準備
        gene = IndicatorGene(type="SMA", parameters={"period": 10}, id="test_id_123")
        
        # 戦略インスタンスのモック
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        # データの準備（空だとエラーになるため）
        mock_strategy.data = Mock()
        mock_strategy.data.df = pd.DataFrame({
            "Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0, 1, 2],
            "Close": [1.5, 2.5, 3.5], "Volume": [100, 200, 300]
        })
        
        calculator.init_indicator(gene, mock_strategy)
        
        # 名前形式: Type_id[:8]
        expected_name_base = "SMA_"
        
        # indicators辞書に登録されているはず
        keys = list(mock_strategy.indicators.keys())
        assert any(expected_name_base in k for k in keys)
        
        # インスタンス属性としてもアクセスできるはず
        found_attr = False
        for k in keys:
            if hasattr(mock_strategy, k):
                found_attr = True
                break
        assert found_attr

    def test_init_indicator_multiple_outputs(self, calculator, mock_indicator_service):
        """複数出力がある指標（MACD等）の登録テスト"""
        # タプルを返すようにモックを設定
        mock_indicator_service.calculate_indicator.return_value = (
            pd.Series([1, 2]), pd.Series([3, 4]), pd.Series([5, 6])
        )
        
        gene = IndicatorGene(type="MACD", id="macd_id")
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.data = Mock()
        mock_strategy.data.df = pd.DataFrame({
            "Open": [1, 2], "High": [2, 3], "Low": [0, 1],
            "Close": [1.5, 2.5], "Volume": [100, 200]
        })
        
        calculator.init_indicator(gene, mock_strategy)
        
        # 各出力（_0, _1, _2）が登録されているか確認
        expected_base = "MACD_macd_id_"
        keys = list(mock_strategy.indicators.keys())
        assert any(f"{expected_base}0" in k for k in keys)
        assert any(f"{expected_base}1" in k for k in keys)
        assert any(f"{expected_base}2" in k for k in keys)