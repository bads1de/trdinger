"""
MultiTimeframeDataProvider テスト

マルチタイムフレームデータプロバイダーのテスト
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from backend.app.services.auto_strategy.services.mtf_data_provider import (
    MultiTimeframeDataProvider,
    TIMEFRAME_TO_MINUTES,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """テスト用のOHLCVデータを生成"""
    # 1時間足のデータを1週間分生成
    start = datetime(2024, 1, 1, 0, 0, 0)
    periods = 168  # 7 days * 24 hours
    dates = [start + timedelta(hours=i) for i in range(periods)]

    data = {
        "Open": [50000 + i * 10 for i in range(periods)],
        "High": [50100 + i * 10 for i in range(periods)],
        "Low": [49900 + i * 10 for i in range(periods)],
        "Close": [50050 + i * 10 for i in range(periods)],
        "Volume": [1000 + i for i in range(periods)],
    }

    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    return df


class TestMultiTimeframeDataProvider:
    """MultiTimeframeDataProvider のテスト"""

    def test_init_with_dataframe(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """DataFrameで初期化できること"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        assert provider.base_timeframe == "1h"
        assert len(provider.base_df) == 168
        assert "1h" in provider.cached_timeframes

    def test_get_base_timeframe_data(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """ベースタイムフレームのデータを取得できること"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        data = provider.get_data("1h")
        assert len(data) == 168

    def test_get_data_with_none_returns_base(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Noneを指定するとベースタイムフレームを返すこと"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        data = provider.get_data(None)
        assert len(data) == 168

    def test_resample_to_4h(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """1h -> 4h へのリサンプリングができること"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        data = provider.get_data("4h")

        # 168時間 / 4時間 = 42バー
        assert len(data) == 42

        # キャッシュされていることを確認
        assert "4h" in provider.cached_timeframes

    def test_resample_to_1d(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """1h -> 1d へのリサンプリングができること"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        data = provider.get_data("1d")

        # 7日分のデータ
        assert len(data) == 7

    def test_ohlcv_values_correct_after_resample(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """リサンプリング後のOHLCV値が正しいこと"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        data_4h = provider.get_data("4h")

        # 最初の4時間バーを確認
        first_bar = data_4h.iloc[0]

        # Open は最初の値
        assert first_bar["Open"] == 50000

        # High は4時間の最大値
        expected_high = max([50100 + i * 10 for i in range(4)])
        assert first_bar["High"] == expected_high

        # Low は4時間の最小値
        expected_low = min([49900 + i * 10 for i in range(4)])
        assert first_bar["Low"] == expected_low

        # Close は最後の値
        assert first_bar["Close"] == 50050 + 3 * 10

        # Volume は合計
        expected_volume = sum([1000 + i for i in range(4)])
        assert first_bar["Volume"] == expected_volume

    def test_cannot_downsample(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """ダウンサンプリングは不可（1h -> 15m）"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        # 15m へのリサンプリングはできない（ベースが1hなので）
        data = provider.get_data("15m")

        # ベースタイムフレームのデータが返される
        assert len(data) == 168

    def test_cache_clear(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """キャッシュをクリアできること"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        # 4h データを取得してキャッシュ
        provider.get_data("4h")
        assert "4h" in provider.cached_timeframes

        # キャッシュをクリア
        provider.clear_cache()

        # ベースタイムフレームだけが残る
        assert provider.cached_timeframes == ["1h"]

    def test_caching_performance(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """同じタイムフレームへの2回目のアクセスはキャッシュから取得"""
        provider = MultiTimeframeDataProvider(
            base_data=sample_ohlcv_data,
            base_timeframe="1h",
        )

        # 1回目のアクセス
        data1 = provider.get_data("4h")

        # 2回目のアクセス（同じオブジェクトが返される）
        data2 = provider.get_data("4h")

        # 同じオブジェクトであることを確認
        assert data1 is data2


class TestTimeframeToMinutes:
    """タイムフレーム変換のテスト"""

    def test_timeframe_mapping(self) -> None:
        """タイムフレームから分への変換が正しいこと"""
        assert TIMEFRAME_TO_MINUTES["1m"] == 1
        assert TIMEFRAME_TO_MINUTES["5m"] == 5
        assert TIMEFRAME_TO_MINUTES["15m"] == 15
        assert TIMEFRAME_TO_MINUTES["30m"] == 30
        assert TIMEFRAME_TO_MINUTES["1h"] == 60
        assert TIMEFRAME_TO_MINUTES["4h"] == 240
        assert TIMEFRAME_TO_MINUTES["1d"] == 1440
        assert TIMEFRAME_TO_MINUTES["1w"] == 10080
