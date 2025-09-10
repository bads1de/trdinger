"""
SQUEEZEとMFI指標のテスト

新規に追加された指標の設定をテストします。
"""
import pytest
import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config.indicator_config import indicator_registry


class TestSqueezeMFIIndicators:
    """SQUEEZEとMFI指標のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用OHLCVデータの準備"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        high = np.random.randn(100).cumsum() + 110
        low = np.random.randn(100).cumsum() + 90
        close = np.random.randn(100).cumsum() + 100
        volume = np.random.randint(1000, 10000, 100)

        # 高値は安値より常に高いことを保証
        for i in range(len(high)):
            if high[i] <= low[i]:
                high[i] = low[i] + np.random.rand() * 10

        df = pd.DataFrame({
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        return df

    @pytest.fixture
    def indicator_service(self):
        """インジケーターサービスの準備"""
        return TechnicalIndicatorService()

    def test_squeeze_registration(self):
        """SQUEEZE指標がレジストリに登録されていることをテスト"""
        config = indicator_registry.get_indicator_config('SQUEEZE')
        assert config is not None
        assert config.indicator_name == 'SQUEEZE'
        assert config.category == 'momentum'
        assert config.required_data == ['high', 'low', 'close']

    def test_squeeze_calculation(self, indicator_service, sample_data):
        """SQUEEZE指標が正常に計算できることをテスト"""
        result = indicator_service.calculate_indicator(sample_data, 'SQUEEZE', {})

        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))
        assert len(result) == len(sample_data)

    def test_squeeze_with_custom_params(self, indicator_service, sample_data):
        """カスタムパラメータでSQUEEZEを計算できることをテスト"""
        params = {
            'bb_length': 25,
            'bb_std': 2.5,
            'kc_length': 15,
            'kc_scalar': 2.0,
            'mom_length': 10,
            'mom_smooth': 5,
            'use_tr': True
        }

        result = indicator_service.calculate_indicator(sample_data, 'SQUEEZE', params)

        assert result is not None
        assert len(result) == len(sample_data)

    def test_mfi_registration(self):
        """MFI指標がレジストリに登録されていることをテスト"""
        config = indicator_registry.get_indicator_config('MFI')
        assert config is not None
        assert config.indicator_name == 'MFI'
        assert config.category == 'volume'
        assert config.required_data == ['high', 'low', 'close', 'volume']

    def test_mfi_calculation(self, indicator_service, sample_data):
        """MFI指標が正常に計算できることをテスト"""
        result = indicator_service.calculate_indicator(sample_data, 'MFI', {})

        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))
        assert len(result) == len(sample_data)

        # MFIは0-100の範囲
        if len(result) > 0 and not pd.isna(result).all():
            valid_values = result[~pd.isna(result)]
            if len(valid_values) > 0:
                assert (valid_values >= 0).all()
                assert (valid_values <= 100).all()

    def test_mfi_with_custom_params(self, indicator_service, sample_data):
        """カスタムパラメータでMFIを計算できることをテスト"""
        params = {
            'length': 20,
            'drift': 2
        }

        result = indicator_service.calculate_indicator(sample_data, 'MFI', params)

        assert result is not None
        assert len(result) == len(sample_data)

    def test_indicator_configuration_loaded(self):
        """指標設定が正しく読み込まれていることをテスト"""
        # pandas-ta設定にSQUEEZEとMFIが存在することを確認
        from app.services.indicators.config.indicator_definitions import PANDAS_TA_CONFIG

        assert 'SQUEEZE' in PANDAS_TA_CONFIG
        assert 'MFI' in PANDAS_TA_CONFIG

        squeeze_config = PANDAS_TA_CONFIG['SQUEEZE']
        assert squeeze_config['function'] == 'squeeze'
        assert 'bb_length' in squeeze_config['params']

        mfi_config = PANDAS_TA_CONFIG['MFI']
        assert mfi_config['function'] == 'mfi'
        assert 'length' in mfi_config['params']
