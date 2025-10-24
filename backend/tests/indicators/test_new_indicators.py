"""
新規テクニカルインジケーターのテスト

新しく追加されたVHF (Vertical Horizontal Filter)とBIASインジケーターのテスト。
TDD原則に従い、実装前にテストを作成する。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators import TechnicalIndicatorService


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """テスト用のOHLCVデータを生成"""
    periods = 200  # VHFとBIASに必要な十分な長さ
    index = pd.date_range("2022-01-01", periods=periods, freq="H")

    # ランダム性とトレンドを含むデータ
    base = np.linspace(10000, 15000, periods)
    noise = np.random.normal(0, 100, periods)
    close = base + noise

    df = pd.DataFrame(
        {
            "Open": close * np.random.uniform(0.99, 1.01, periods),
            "High": close * np.random.uniform(1.01, 1.03, periods),
            "Low": close * np.random.uniform(0.97, 0.99, periods),
            "Close": close,
            "Volume": np.random.uniform(1000, 5000, periods),
        },
        index=index,
    )

    # ボラティリティを追加
    df["High"] = np.maximum(
        df["High"],
        df[["Open", "Close"]].max(axis=1) * np.random.uniform(1.0, 1.05, periods),
    )
    df["Low"] = np.minimum(
        df["Low"],
        df[["Open", "Close"]].min(axis=1) * np.random.uniform(0.95, 1.0, periods),
    )

    return df


@pytest.fixture
def indicator_service() -> TechnicalIndicatorService:
    """テクニカルインジケーターサービスを提供"""
    return TechnicalIndicatorService()


class TestVHFIndicator:
    """VHF (Vertical Horizontal Filter) インジケーターのテスト"""

    def test_vhf_initialization(self, indicator_service: TechnicalIndicatorService):
        """VHFインジケーターが初期化可能か確認"""
        config = indicator_service.registry.get_indicator_config("VHF")
        assert config is not None, "VHFの設定が見つかりません"
        assert config.adapter_function is not None, "VHFのアダプター関数がありません"

    def test_vhf_basic_calculation(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """VHFの基本計算をテスト"""
        result = indicator_service.calculate_indicator(
            sample_ohlcv, "VHF", {"length": 28}
        )

        # 結果がnumpy配列であるか確認（pandas-taのため）
        assert isinstance(result, np.ndarray), "VHFの結果がnumpy配列でない"
        assert len(result) == len(sample_ohlcv), "VHFの結果の長さが不正"
        assert not np.isnan(result).all(), "VHFの結果がすべてNaN"

        # VHFの値は0-1の範囲内であるべき
        assert result.min() >= 0, "VHFの最小値が0未満"
        assert result.max() <= 1, "VHFの最大値が1を超える"

    def test_vhf_with_custom_length(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """VHFのカスタムパラメータをテスト"""
        result = indicator_service.calculate_indicator(
            sample_ohlcv, "VHF", {"length": 20, "scalar": 100.0}
        )

        assert isinstance(result, np.ndarray), "VHFの結果がnumpy配列でない"
        assert not np.isnan(result).all(), "VHFの結果がすべてNaN"

    def test_vhf_insufficient_data(
        self, indicator_service: TechnicalIndicatorService
    ):
        """VHFがデータ不足を適切に処理するかテスト"""
        # 短すぎるデータ
        df = pd.DataFrame({"Close": [100, 101, 102]})
        result = indicator_service.calculate_indicator(
            df, "VHF", {"length": 28}
        )

        # numpy配列が返されるべき
        assert isinstance(result, np.ndarray), "VHFの結果がnumpy配列でない"
        assert np.isnan(result).all(), "VHFがデータ不足を適切に処理していない"

    def test_vhf_with_nan_values(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """VHFがNaN値を適切に処理するかテスト"""
        df_with_nan = sample_ohlcv.copy()
        # datetimeインデックスを使用してNaNを設定
        nan_indices = df_with_nan.index[10:15]
        df_with_nan.loc[nan_indices, "Close"] = np.nan

        result = indicator_service.calculate_indicator(
            df_with_nan, "VHF", {"length": 28}
        )

        assert isinstance(result, np.ndarray), "VHFの結果がnumpy配列でない"
        # NaNの位置に応じて適切に処理されているべき


class TestBIASIndicator:
    """BIAS インジケーターのテスト"""

    def test_bias_initialization(self, indicator_service: TechnicalIndicatorService):
        """BIASインジケーターが初期化可能か確認"""
        config = indicator_service.registry.get_indicator_config("BIAS")
        assert config is not None, "BIASの設定が見つかりません"
        assert config.adapter_function is not None, "BIASのアダプター関数がありません"

    def test_bias_basic_calculation(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """BIASの基本計算をテスト"""
        result = indicator_service.calculate_indicator(
            sample_ohlcv, "BIAS", {"length": 26}
        )

        # 結果がnumpy配列であるか確認（pandas-taのため）
        assert isinstance(result, np.ndarray), "BIASの結果がnumpy配列でない"
        assert len(result) == len(sample_ohlcv), "BIASの結果の長さが不正"
        assert not np.isnan(result).all(), "BIASの結果がすべてNaN"

        # BIASの値は通常-100から100の範囲内
        assert result.min() >= -100, "BIASの最小値が異常"
        assert result.max() <= 100, "BIASの最大値が異常"

    def test_bias_with_custom_parameters(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """BIASのカスタムパラメータをテスト"""
        result = indicator_service.calculate_indicator(
            sample_ohlcv, "BIAS", {"length": 20, "ma_type": "ema"}
        )

        assert isinstance(result, np.ndarray), "BIASの結果がnumpy配列でない"
        assert not np.isnan(result).all(), "BIASの結果がすべてNaN"

    def test_bias_insufficient_data(
        self, indicator_service: TechnicalIndicatorService
    ):
        """BIASがデータ不足を適切に処理するかテスト"""
        # 短すぎるデータ
        df = pd.DataFrame({"Close": [100, 101, 102]})
        result = indicator_service.calculate_indicator(
            df, "BIAS", {"length": 26}
        )

        assert isinstance(result, np.ndarray), "BIASの結果がnumpy配列でない"
        # 十分なデータがない場合はNaNが含まれているべき

    def test_bias_with_nan_values(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """BIASがNaN値を適切に処理するかテスト"""
        df_with_nan = sample_ohlcv.copy()
        # datetimeインデックスを使用してNaNを設定
        nan_indices = df_with_nan.index[10:15]
        df_with_nan.loc[nan_indices, "Close"] = np.nan

        result = indicator_service.calculate_indicator(
            df_with_nan, "BIAS", {"length": 26}
        )

        assert isinstance(result, np.ndarray), "BIASの結果がnumpy配列でない"
        # NaNの位置に応じて適切に処理されているべき


class TestNewIndicatorsIntegration:
    """新規インジケーターの統合テスト"""

    def test_new_indicators_in_registry(self, indicator_service: TechnicalIndicatorService):
        """新しいインジケーターがレジストリに登録されているか確認"""
        # VHFが登録されているか
        vhf_config = indicator_service.registry.get_indicator_config("VHF")
        assert vhf_config is not None, "VHFがレジストリに登録されていない"

        # BIASが登録されているか
        bias_config = indicator_service.registry.get_indicator_config("BIAS")
        assert bias_config is not None, "BIASがレジストリに登録されていない"

    def test_new_indicators_calculation_with_all_data(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """新しいインジケーターがすべてのOHLCVデータで計算可能かテスト"""
        # VHFのテスト
        vhf_result = indicator_service.calculate_indicator(
            sample_ohlcv, "VHF", {"length": 28}
        )
        assert isinstance(vhf_result, np.ndarray), "VHF計算失敗"
        assert not np.isnan(vhf_result).all(), "VHFがすべてNaN"

        # BIASのテスト
        bias_result = indicator_service.calculate_indicator(
            sample_ohlcv, "BIAS", {"length": 26}
        )
        assert isinstance(bias_result, np.ndarray), "BIAS計算失敗"
        assert not np.isnan(bias_result).all(), "BIASがすべてNaN"

    def test_new_indicators_performance(self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame):
        """新しいインジケーターのパフォーマンスをテスト"""
        import time

        # VHFのパフォーマンステスト
        start_time = time.time()
        for _ in range(10):
            indicator_service.calculate_indicator(
                sample_ohlcv, "VHF", {"length": 28}
            )
        vhf_time = time.time() - start_time

        # BIASのパフォーマンステスト
        start_time = time.time()
        for _ in range(10):
            indicator_service.calculate_indicator(
                sample_ohlcv, "BIAS", {"length": 26}
            )
        bias_time = time.time() - start_time

        # 合理的なパフォーマンスであるか確認（10回実行で1秒以内）
        assert vhf_time < 1.0, f"VHFが遅すぎ: {vhf_time:.3f}秒"
        assert bias_time < 1.0, f"BIASが遅すぎ: {bias_time:.3f}秒"