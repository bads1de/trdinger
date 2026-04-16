"""
IndicatorServiceのユニットテスト
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.genes.indicator import IndicatorGene
from app.services.auto_strategy.services.indicator_service import IndicatorCalculator
from app.services.auto_strategy.services.mtf_data_provider import (
    MultiTimeframeDataProvider,
)


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
        mock_data.df = pd.DataFrame(
            {
                "Open": [1, 2, 3],
                "High": [2, 3, 4],
                "Low": [0, 1, 2],
                "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            }
        )

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
        mock_strategy.data.df = pd.DataFrame(
            {
                "Open": [1, 2, 3],
                "High": [2, 3, 4],
                "Low": [0, 1, 2],
                "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            }
        )

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
            pd.Series([1, 2]),
            pd.Series([3, 4]),
            pd.Series([5, 6]),
        )

        gene = IndicatorGene(type="MACD", id="macd_id")
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.data = Mock()
        mock_strategy.data.df = pd.DataFrame(
            {
                "Open": [1, 2],
                "High": [2, 3],
                "Low": [0, 1],
                "Close": [1.5, 2.5],
                "Volume": [100, 200],
            }
        )

        calculator.init_indicator(gene, mock_strategy)

        # 各出力（_0, _1, _2）が登録されているか確認
        expected_base = "MACD_macd_id_"
        keys = list(mock_strategy.indicators.keys())
        assert any(f"{expected_base}0" in k for k in keys)
        assert any(f"{expected_base}1" in k for k in keys)
        assert any(f"{expected_base}2" in k for k in keys)

    def test_init_indicator_uses_mtf_runtime_reference_name(self, calculator):
        """MTF指標はタイムフレーム込みの参照名で登録されること"""
        gene = IndicatorGene(
            type="SMA",
            parameters={"period": 10},
            timeframe="4h",
            id="testid123456",
        )

        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.data = Mock()
        mock_strategy.data.df = pd.DataFrame(
            {
                "Open": [1, 2, 3],
                "High": [2, 3, 4],
                "Low": [0, 1, 2],
                "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            }
        )

        calculator.init_indicator(gene, mock_strategy)

        assert "SMA_4h_testid12" in mock_strategy.indicators


class TestMTFIntegration:
    """MTFデータプロバイダーと指標計算の統合テスト"""

    @pytest.fixture
    def sample_hourly_data(self) -> pd.DataFrame:
        """1時間足のテストデータ"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        periods = 168  # 7日分
        dates = [start + timedelta(hours=i) for i in range(periods)]
        return pd.DataFrame(
            {
                "Open": [50000 + i * 10 for i in range(periods)],
                "High": [50100 + i * 10 for i in range(periods)],
                "Low": [49900 + i * 10 for i in range(periods)],
                "Close": [50050 + i * 10 for i in range(periods)],
                "Volume": [1000 + i for i in range(periods)],
            },
            index=pd.DatetimeIndex(dates),
        )

    def test_mtf_data_provider_resamples_to_4h(self, sample_hourly_data):
        """MTFデータプロバイダーが1h→4hに正しくリサンプリングすること"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_hourly_data,
            base_timeframe="1h",
        )

        data_4h = provider.get_data("4h")

        # 168時間 / 4時間 = 42バー
        assert len(data_4h) == 42
        assert "Open" in data_4h.columns
        assert "Close" in data_4h.columns

    def test_mtf_data_provider_resamples_to_1d(self, sample_hourly_data):
        """MTFデータプロバイダーが1h→1dに正しくリサンプリングすること"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_hourly_data,
            base_timeframe="1h",
        )

        data_1d = provider.get_data("1d")

        # 7日分
        assert len(data_1d) == 7

    def test_indicator_calculator_with_mtf_provider(self, sample_hourly_data):
        """MTFプロバイダー経由で指標計算ができること"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_hourly_data,
            base_timeframe="1h",
        )

        # SMAの計算結果を返すモックサービス
        mock_service = Mock()
        mock_service.calculate_indicator.return_value = pd.Series(
            [50000.0] * 42,
            index=provider.get_data("4h").index,
        )

        calculator = IndicatorCalculator(
            technical_indicator_service=mock_service,
            mtf_data_provider=provider,
        )

        # MTF指標を持つ遺伝子
        gene = IndicatorGene(type="SMA", parameters={"period": 10}, timeframe="4h")

        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.data = Mock()
        mock_strategy.data.df = sample_hourly_data

        calculator.init_indicator(gene, mock_strategy)

        # MTFデータプロバイダーが呼び出されたことを確認
        mock_service.calculate_indicator.assert_called_once()
        call_args = mock_service.calculate_indicator.call_args
        called_df = call_args[0][0]

        # 4hデータが渡されていることを確認（168時間 → 42バー）
        assert len(called_df) == 42

    def test_mtf_gene_has_timeframe_attribute(self):
        """MTF指標遺伝子がtimeframe属性を持つこと"""
        gene = IndicatorGene(
            type="SMA",
            parameters={"period": 20},
            timeframe="4h",
            id="mtf_sma_001",
        )

        assert gene.timeframe == "4h"
        assert gene.type == "SMA"
