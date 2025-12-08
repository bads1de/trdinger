"""
IndicatorService のテスト
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from app.services.auto_strategy.services.indicator_service import IndicatorCalculator
from app.services.auto_strategy.models.strategy_models import IndicatorGene


class TestIndicatorCalculator:
    """IndicatorCalculator のテストクラス"""

    @pytest.fixture
    def mock_technical_indicator_service(self):
        """TechnicalIndicatorService のモック"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def calculator(self, mock_technical_indicator_service):
        """IndicatorCalculator のインスタンス"""
        return IndicatorCalculator(
            technical_indicator_service=mock_technical_indicator_service
        )

    @pytest.fixture
    def mock_data(self):
        """backtesting.py Data オブジェクトのモック"""
        mock = MagicMock()
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [102, 103, 104],
                "Volume": [1000, 1100, 1200],
            }
        )
        mock.df = df
        return mock

    def test_calculate_indicator_success(
        self, calculator, mock_technical_indicator_service, mock_data
    ):
        """calculate_indicator 成功時のテスト"""
        # モックの設定
        expected_result = pd.Series([1, 2, 3])
        mock_technical_indicator_service.calculate_indicator.return_value = (
            expected_result
        )

        indicator_type = "SMA"
        parameters = {"period": 20}

        result = calculator.calculate_indicator(mock_data, indicator_type, parameters)

        # サービスが正しい引数で呼ばれたか確認
        mock_technical_indicator_service.calculate_indicator.assert_called_once_with(
            mock_data.df, indicator_type, parameters
        )
        assert result.equals(expected_result)

    def test_calculate_indicator_validation_error(self, calculator):
        """calculate_indicator 検証エラーテスト"""
        # Data None
        with pytest.raises(ValueError, match="データオブジェクトがNone"):
            calculator.calculate_indicator(None, "SMA", {})

        # Empty Data
        mock_empty_data = MagicMock()
        mock_empty_data.df = pd.DataFrame()
        with pytest.raises(ValueError, match="データが空"):
            calculator.calculate_indicator(mock_empty_data, "SMA", {})

    def test_init_indicator_single_output(
        self, calculator, mock_technical_indicator_service, mock_data
    ):
        """init_indicator 単一出力指標のテスト"""
        # モック設定
        expected_result = pd.Series([10, 20, 30])
        mock_technical_indicator_service.calculate_indicator.return_value = (
            expected_result
        )

        # 戦略インスタンスモック
        strategy_instance = MagicMock()
        strategy_instance.data = mock_data

        # __dict__ をシミュレートするために spec を設定しない、または dict を使う
        # MagicMock はデフォルトで属性設定可能だが、 __dict__ 操作をテストするため実オブジェクトに近いものを使う
        class MockStrategy:
            pass

        strategy_instance = MockStrategy()
        strategy_instance.data = mock_data

        indicator_gene = IndicatorGene(
            type="SMA", parameters={"period": 20}, enabled=True
        )

        calculator.init_indicator(indicator_gene, strategy_instance)

        # 属性としてセットされたか確認
        assert hasattr(strategy_instance, "SMA")
        assert getattr(strategy_instance, "SMA").equals(expected_result)
        # indicators 辞書にも入っているか (IndicatorCalculator が作成する)
        assert hasattr(strategy_instance, "indicators")
        assert strategy_instance.indicators["SMA"].equals(expected_result)

    def test_init_indicator_multi_output(
        self, calculator, mock_technical_indicator_service, mock_data
    ):
        """init_indicator 複数出力指標のテスト"""
        # モック設定 (MACDなど tuple を返す場合)
        res1 = pd.Series([1, 2, 3])
        res2 = pd.Series([4, 5, 6])
        mock_technical_indicator_service.calculate_indicator.return_value = (res1, res2)

        class MockStrategy:
            pass

        strategy_instance = MockStrategy()
        strategy_instance.data = mock_data

        indicator_gene = IndicatorGene(type="MACD", parameters={}, enabled=True)

        calculator.init_indicator(indicator_gene, strategy_instance)

        # 複数の属性としてセットされたか (MACD_0, MACD_1)
        assert hasattr(strategy_instance, "MACD_0")
        assert hasattr(strategy_instance, "MACD_1")
        assert getattr(strategy_instance, "MACD_0").equals(res1)
        assert getattr(strategy_instance, "MACD_1").equals(res2)

        assert hasattr(strategy_instance, "indicators")
        assert strategy_instance.indicators["MACD_0"].equals(res1)

    def test_init_indicator_strategy_none(self, calculator):
        """init_indicator 戦略インスタンスNoneエラー"""
        gene = IndicatorGene(type="SMA", parameters={}, enabled=True)
        with pytest.raises(ValueError, match="戦略インスタンスがNone"):
            calculator.init_indicator(gene, None)
