"""
Trend indicators (SAR, etc.) のテスト
"""
import pytest
import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config.indicator_config import indicator_registry


class TestTrendIndicators:
    """Trend indicators のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用OHLCVデータの準備"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        high = np.random.randn(100).cumsum() + 110
        low = np.random.randn(100).cumsum() + 90
        close = np.random.randn(100).cumsum() + 100

        # 高値は安値より常に高いことを保証
        for i in range(len(high)):
            if high[i] <= low[i]:
                high[i] = low[i] + np.random.rand() * 10

        df = pd.DataFrame({
            'High': high,
            'Low': low,
            'Close': close,
        }, index=dates)
        return df

    @pytest.fixture
    def indicator_service(self):
        """インジケーターサービスの準備"""
        return TechnicalIndicatorService()

    def test_sar_registration(self):
        """SAR指標がレジストリに登録されていることをテスト"""
        config = indicator_registry.get_indicator_config('SAR')
        assert config is not None
        assert config.indicator_name == 'SAR'
        assert config.category == 'trend'
        assert config.required_data == ['high', 'low']

    def test_sar_calculation(self, indicator_service, sample_data):
        """SAR指標が正常に計算できることをテスト"""
        result = indicator_service.calculate_indicator(sample_data, 'SAR', {})

        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))
        assert len(result) == len(sample_data)

    def test_sar_with_custom_params(self, indicator_service, sample_data):
        """カスタムパラメータでSARを計算できることをテスト"""
        params = {
            'acceleration': 0.03,
            'maximum_acceleration': 0.3
        }

        result = indicator_service.calculate_indicator(sample_data, 'SAR', params)

        assert result is not None
        assert len(result) == len(sample_data)

    def test_indicator_configuration_loaded(self):
        """指標設定が正しく読み込まれていることをテスト"""
        # pandas-ta設定にSARが存在することを確認
        from app.services.indicators.config.indicator_definitions import PANDAS_TA_CONFIG

        assert 'SAR' in PANDAS_TA_CONFIG

        sar_config = PANDAS_TA_CONFIG['SAR']
        assert sar_config['function'] == 'psar'
        assert 'acceleration' in sar_config['params']