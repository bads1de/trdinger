"""
データ長検証機能のテスト

データ長不足時の動作をテストする。強化版のテストも含む。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.data_validation import (
    validate_data_length_with_fallback,
    validate_ohlcv_data_quality,
    validate_indicator_params
)


@pytest.fixture
def technical_indicator_service():
    """テクニカル指標サービスを初期化"""
    return TechnicalIndicatorService()


@pytest.fixture
def sample_data_100():
    """100期間のサンプルデータを生成"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(95, 115, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    return df


@pytest.fixture
def sample_data_short():
    """10期間の短いサンプルデータを生成"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 10),
        'high': np.random.uniform(105, 115, 10),
        'low': np.random.uniform(95, 105, 10),
        'close': np.random.uniform(95, 115, 10),
        'volume': np.random.uniform(1000, 10000, 10)
    }, index=dates)
    return df


class TestDataLengthValidation:

    def test_ui_data_length_sufficient(self, technical_indicator_service, sample_data_100):
        """UI指標：データ長が十分な場合のテスト"""
        result = technical_indicator_service.calculate_indicator(
            sample_data_100, "UI", {}
        )
        # データ長が十分なので、結果が配列であることを確認
        assert len(result) == len(sample_data_100)

    def test_ui_data_length_insufficient(self, technical_indicator_service, sample_data_short):
        """UI指標：データ長が不足する場合のテスト"""
        result = technical_indicator_service.calculate_indicator(
            sample_data_short, "UI", {}
        )
        # データ長不足なので、NaNの配列が返される
        assert len(result) == len(sample_data_short)
        assert np.isnan(result).all()


    def test_sinwma_data_length_sufficient(self, technical_indicator_service, sample_data_100):
        """SINWMA指標：データ長が十分な場合のテスト"""
        result = technical_indicator_service.calculate_indicator(
            sample_data_100, "SINWMA", {}
        )
        assert len(result) == len(sample_data_100)

    def test_sinwma_data_length_insufficient(self, technical_indicator_service, sample_data_short):
        """SINWMA指標：データ長が不足する場合のテスト"""
        result = technical_indicator_service.calculate_indicator(
            sample_data_short, "SINWMA", {}
        )
        assert len(result) == len(sample_data_short)
        assert np.isnan(result).all()


    def test_validate_data_length_method(self, technical_indicator_service, sample_data_100, sample_data_short):
        """validate_data_lengthメソッドの直接テスト"""

        # UI：データ長十分
        assert technical_indicator_service.validate_data_length(sample_data_100, "UI", {})

        # UI：データ長不足（14必要だが10しかない）
        assert not technical_indicator_service.validate_data_length(sample_data_short, "UI", {})


        # パラメータ指定時の動作確認
        assert not technical_indicator_service.validate_data_length(sample_data_short, "UI", {"length": 20})

    def test_unknown_indicator_passes_validation(self, technical_indicator_service, sample_data_short):
        """未知の指標は検証をパスする"""
        # 設定にない指標は検証不要のはず
        assert technical_indicator_service.validate_data_length(sample_data_short, "UNKNOWN_INDICATOR", {})

    def test_validate_data_length_with_fallback_enhanced(self, sample_data_100, sample_data_short):
        """強化版データ長検証のテスト"""

        # RSI: 十分なデータ長でTrueを返す
        is_valid, min_length = validate_data_length_with_fallback(sample_data_100, "RSI", {"length": 14})
        assert is_valid
        assert min_length == 14

        # RSI: 不足データ長でフォールバックを試行
        # サンプル10はRSIの14より少ないが、フォールバックのmin_required=14//3=4より多い場合
        is_valid, min_length = validate_data_length_with_fallback(sample_data_short, "RSI", {"length": 14})
        assert is_valid  # 不足してもフォールバック可能
        assert min_length == len(sample_data_short)  # 実際のデータ長を返す

    def test_validate_ohlcv_data_quality_prometheus(self, sample_data_100, sample_data_short):
        """prometheus用データ品質検証のテスト"""

        # 正常データ: 問題なし
        issues = validate_ohlcv_data_quality(sample_data_100)
        assert len(issues) == 0

        # 短いデータ: データ長不足を検出
        issues = validate_ohlcv_data_quality(sample_data_short)
        assert len(issues) > 0
        assert any("データ長不足" in issue for issue in issues)

    def test_validate_indicator_params_prometheus(self):
        """prometheus用パラメータ検証のテスト"""

        # 正常パラメータ
        is_valid = validate_indicator_params("RSI", {"length": 14})
        assert is_valid

        # 無効パラメータ: lengthが負数
        is_valid = validate_indicator_params("RSI", {"length": -1})
        assert not is_valid

        # 未知の指標
        is_valid = validate_indicator_params("UNKNOWN", {"length": 14})
        assert not is_valid