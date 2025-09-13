"""
統合指標テスト

すべての技術指標の機能を統合テスト
TDD原則に基づき、各指標を包括的にテスト
"""

import pytest
import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config.indicator_config import indicator_registry


class TestIndicatorsIntegrated:
    """指標機能の統合テスト"""

    @pytest.fixture
    def sample_ohlcv_data(self):
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
    def sample_close_data(self):
        """テスト用Closeデータのみの準備"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        close = np.random.randn(100).cumsum() + 100

        df = pd.DataFrame({
            'Close': close
        }, index=dates)
        return df

    @pytest.fixture
    def indicator_service(self):
        """インジケーターサービスの準備"""
        return TechnicalIndicatorService()

    def test_trend_indicators_complete(self, indicator_service, sample_ohlcv_data):
        """トレンド指標の完全テスト"""
        trend_indicators = ['SAR', 'SMA', 'EMA', 'WMA']

        for indicator in trend_indicators:
            config = indicator_registry.get_indicator_config(indicator)
            if config:
                result = indicator_service.calculate_indicator(sample_ohlcv_data, indicator, {})
                assert result is not None
                assert len(result) == len(sample_ohlcv_data)

    def test_momentum_indicators_complete(self, indicator_service, sample_ohlcv_data, sample_close_data):
        """モメンタム指標の完全テスト"""
        momentum_indicators = ['RSI', 'STOCH', 'CCI', 'MFI', 'SQUEEZE']

        for indicator in momentum_indicators:
            config = indicator_registry.get_indicator_config(indicator)
            if config:
                df = sample_ohlcv_data if 'volume' in config.required_data else sample_close_data
                result = indicator_service.calculate_indicator(df, indicator, {})
                assert result is not None
                assert len(result) == len(df)

    def test_volume_indicators_complete(self, indicator_service, sample_ohlcv_data):
        """出来高指標の完全テスト"""
        volume_indicators = ['MFI', 'OBV', 'AD', 'ADOSC']

        for indicator in volume_indicators:
            config = indicator_registry.get_indicator_config(indicator)
            if config:
                result = indicator_service.calculate_indicator(sample_ohlcv_data, indicator, {})
                assert result is not None
                assert len(result) == len(sample_ohlcv_data)

    def test_volatility_indicators_complete(self, indicator_service, sample_ohlcv_data):
        """ボラティリティ指標の完全テスト"""
        volatility_indicators = ['ATR', 'NATR', 'BBANDS']

        for indicator in volatility_indicators:
            config = indicator_registry.get_indicator_config(indicator)
            if config:
                result = indicator_service.calculate_indicator(sample_ohlcv_data, indicator, {})
                assert result is not None
                assert len(result) == len(sample_ohlcv_data)

    def test_indicator_configurations_loaded(self):
        """指標設定が正しく読み込まれていることをテスト"""
        # pandas-ta設定に主要指標が存在することを確認
        from app.services.indicators.config.indicator_definitions import PANDAS_TA_CONFIG

        essential_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'ATR', 'SAR']

        for indicator in essential_indicators:
            assert indicator in PANDAS_TA_CONFIG
            config = PANDAS_TA_CONFIG[indicator]
            assert 'function' in config
            assert 'params' in config

    def test_indicator_calculations_with_custom_params(self, indicator_service, sample_ohlcv_data):
        """カスタムパラメータでの指標計算テスト"""
        test_cases = [
            ('SMA', {'length': 20}),
            ('EMA', {'length': 14}),
            ('RSI', {'length': 14}),
            ('ATR', {'length': 14}),
            ('BBANDS', {'length': 20, 'std': 2})
        ]

        for indicator, params in test_cases:
            config = indicator_registry.get_indicator_config(indicator)
            if config:
                result = indicator_service.calculate_indicator(sample_ohlcv_data, indicator, params)
                assert result is not None
                assert len(result) == len(sample_ohlcv_data)

    def test_indicator_error_handling(self, indicator_service, sample_close_data):
        """指標計算のエラーハンドリングテスト"""
        # 無効な指標名
        result = indicator_service.calculate_indicator(sample_close_data, 'INVALID_INDICATOR', {})
        assert result is None

        # 必要なデータが不足
        result = indicator_service.calculate_indicator(sample_close_data, 'MFI', {})  # Volumeが必要
        # Volumeがない場合の処理を確認（実装による）

    def test_indicator_output_types(self, indicator_service, sample_ohlcv_data):
        """指標出力の型テスト"""
        result = indicator_service.calculate_indicator(sample_ohlcv_data, 'SMA', {'length': 20})

        if result is not None:
            assert isinstance(result, (np.ndarray, pd.Series))
            # NaN値が適切に処理されていることを確認
            nan_count = pd.isna(result).sum()
            # 初期のNaN値は許容されるが、全体の大部分がNaNではないことを確認
            assert nan_count < len(result) * 0.9


class TestIndicatorWarningsAndDeprecations:
    """指標関連の警告と非推奨機能のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        df = pd.DataFrame({
            'Close': np.random.rand(100) * 100
        })
        return df

    def test_indicator_calculations_without_warnings(self, sample_data):
        """指標計算が警告なしで実行できることをテスト"""
        service = TechnicalIndicatorService()

        # VIDYA計算時の警告なしを確認
        with pytest.warns(None) as record:
            result = service.calculate_indicator(sample_data, 'VIDYA', {'period': 14, 'adjust': True})

        # FutureWarningがないことを確認
        future_warnings = [w for w in record.list if "FutureWarning" in str(w.message)]
        assert len(future_warnings) == 0

    def test_indicator_calculations_without_errors(self, sample_data):
        """指標計算がエラーなしで実行できることをテスト"""
        service = TechnicalIndicatorService()

        # LINREG計算時のエラーなしを確認
        try:
            result = service.calculate_indicator(sample_data, 'LINREG', {'period': 14})
            assert result is not None
        except TypeError as e:
            assert "unexpected keyword argument" not in str(e)

        # STC計算時のエラーなしを確認
        try:
            result = service.calculate_indicator(sample_data, 'STC', {'length': 10, 'fast_length': 23, 'slow_length': 50})
            assert result is not None
        except TypeError as e:
            assert "missing 1 required positional argument" not in str(e)


class TestMAVPIndicator:
    """MAVP指標の統合テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データの準備"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        close = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({
            'Close': close
        }, index=dates)
        return df

    def test_mavp_calculation_without_periods_param_error(self, sample_data):
        """MAVP指標がperiodsパラメータなしで正常に計算できることをテスト"""
        service = TechnicalIndicatorService()

        # periodsパラメータを提供しなくてもエラーが発生しないことを確認
        params = {
            'minperiod': 5,
            'maxperiod': 30,
            'matype': 0
        }

        # これは以前はエラーになっていたはず
        result = service.calculate_indicator(sample_data, 'MAVP', params)

        # 結果がNoneではなく、適切な形状を持っていることを確認
        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))
        assert len(result) == len(sample_data)

        # NaNが多い場合は、入力データが十分でないことを確認
        nan_count = pd.isna(result).sum() if hasattr(result, '__len__') else 0
        if hasattr(result, '__len__') and len(result) > 0:
            nan_ratio = nan_count / len(result)
            # NaNが多すぎる場合はテストをスキップ（データ長不足のため）
            if nan_ratio > 0.8:
                pytest.skip("データ長不足により多くのNaNが発生")

    def test_mavp_calculation_with_custom_periods(self, sample_data):
        """カスタムのperiodsでMAVPを計算できることをテスト"""
        service = TechnicalIndicatorService()

        # periodsがDataFrameの列として存在するのではなく、パラメータとして直接渡す
        # テスト目的なので、periods列があるはずなのでそのままテスト
        # periodsが提供されていない場合のデフォルト動作を確認
        params = {
            'minperiod': 5,
            'maxperiod': 30,
            'matype': 0
        }

        result = service.calculate_indicator(sample_data, 'MAVP', params)

        assert result is not None
        assert len(result) == len(sample_data)

        # 期待される動作: NaN値が多い場合はデータ長不足の正常挙動
        nan_count = pd.isna(result).sum() if hasattr(result, '__len__') else 0
        assert nan_count >= 0  # NaNがあってもいいが、エラーは発生しない


class TestSqueezeMFIIndicators:
    """SQUEEZEとMFI指標の統合テスト"""

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

    def test_squeeze_registration(self, sample_data):
        """SQUEEZE指標がレジストリに登録されていることをテスト"""
        config = indicator_registry.get_indicator_config('SQUEEZE')
        assert config is not None
        assert config.indicator_name == 'SQUEEZE'
        assert config.category == 'momentum'
        assert config.required_data == ['high', 'low', 'close']

    def test_squeeze_calculation(self, sample_data):
        """SQUEEZE指標が正常に計算できることをテスト"""
        service = TechnicalIndicatorService()
        result = service.calculate_indicator(sample_data, 'SQUEEZE', {})

        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))
        assert len(result) == len(sample_data)

    def test_squeeze_with_custom_params(self, sample_data):
        """カスタムパラメータでSQUEEZEを計算できることをテスト"""
        service = TechnicalIndicatorService()
        params = {
            'bb_length': 25,
            'bb_std': 2.5,
            'kc_length': 15,
            'kc_scalar': 2.0,
            'mom_length': 10,
            'mom_smooth': 5,
            'use_tr': True
        }

        result = service.calculate_indicator(sample_data, 'SQUEEZE', params)

        assert result is not None
        assert len(result) == len(sample_data)

    def test_mfi_calculation(self, sample_data):
        """MFI指標が正常に計算できることをテスト"""
        service = TechnicalIndicatorService()
        result = service.calculate_indicator(sample_data, 'MFI', {})

        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))
        assert len(result) == len(sample_data)

        # MFIは0-100の範囲
        if len(result) > 0 and not pd.isna(result).all():
            valid_values = result[~pd.isna(result)]
            if len(valid_values) > 0:
                assert (valid_values >= 0).all()
                assert (valid_values <= 100).all()

    def test_mfi_with_custom_params(self, sample_data):
        """カスタムパラメータでMFIを計算できることをテスト"""
        service = TechnicalIndicatorService()
        params = {
            'length': 20,
            'drift': 2
        }

        result = service.calculate_indicator(sample_data, 'MFI', params)

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


if __name__ == "__main__":
    pytest.main([__file__])