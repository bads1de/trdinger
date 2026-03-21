"""
indicatorsテスト用の共通フィクスチャとヘルパー
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """テスト用のOHLCVデータを生成（十分な長さ）"""
    periods = 500
    index = pd.date_range("2022-01-01", periods=periods, freq="h")

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
def sample_data() -> pd.Series:
    """テスト用の価格データを生成"""
    return pd.Series(
        np.random.normal(100, 5, 100),
        index=pd.date_range("2023-01-01", periods=100),
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """テスト用のOHLCVデータフレームを生成"""
    rows = 100
    return pd.DataFrame(
        {
            "open": np.random.normal(100, 5, rows),
            "high": np.random.normal(105, 5, rows),
            "low": np.random.normal(95, 5, rows),
            "close": np.random.normal(100, 5, rows),
            "volume": np.random.normal(1000, 100, rows),
        },
        index=pd.date_range("2023-01-01", periods=rows),
    )


@pytest.fixture
def short_data() -> pd.DataFrame:
    """短いテストデータを生成（データ不足テスト用）"""
    return pd.DataFrame(
        {
            "Close": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Volume": [1000, 1100, 900],
        }
    )


@pytest.fixture
def indicator_service():
    """テクニカルインジケーターサービスを提供"""
    from app.services.indicators import TechnicalIndicatorService

    return TechnicalIndicatorService()


class IndicatorTestHelper:
    """インジケーターテスト用のヘルパークラス"""

    @staticmethod
    def assert_series_result(result, expected_length: int, allow_all_nan: bool = False):
        """シリーズ結果の基本検証"""
        assert isinstance(result, pd.Series), "結果がpd.Seriesではありません"
        assert len(result) == expected_length, f"結果の長さが不正: {len(result)} != {expected_length}"
        if not allow_all_nan:
            assert not result.isna().all(), "結果がすべてNaNです"

    @staticmethod
    def assert_tuple_result(result, expected_length: int, expected_count: int):
        """タプル結果の基本検証"""
        assert isinstance(result, tuple), "結果がtupleではありません"
        assert len(result) == expected_count, f"結果の要素数が不正: {len(result)} != {expected_count}"
        for i, series in enumerate(result):
            assert isinstance(series, pd.Series), f"結果[{i}]がpd.Seriesではありません"
            assert len(series) == expected_length, f"結果[{i}]の長さが不正"

    @staticmethod
    def assert_dataframe_result(result, expected_length: int, expected_columns: list = None):
        """データフレーム結果の基本検証"""
        assert isinstance(result, pd.DataFrame), "結果がpd.DataFrameではありません"
        assert len(result) == expected_length, f"結果の長さが不正: {len(result)} != {expected_length}"
        if expected_columns:
            for col in expected_columns:
                assert col in result.columns, f"列 '{col}' が結果に含まれていません"

    @staticmethod
    def assert_value_range(result, min_val: float = None, max_val: float = None):
        """値の範囲を検証"""
        valid_values = result.dropna()
        if len(valid_values) == 0:
            return
        if min_val is not None:
            assert valid_values.min() >= min_val, f"最小値が範囲外: {valid_values.min()} < {min_val}"
        if max_val is not None:
            assert valid_values.max() <= max_val, f"最大値が範囲外: {valid_values.max()} > {max_val}"


@pytest.fixture
def test_helper():
    """テストヘルパーを提供"""
    return IndicatorTestHelper()
