"""
オープンインタレストマージャーのテスト

OIMergerクラスの全機能をテストします:
- 初期化
- DataFrameマージ処理
- データ変換
- エラーハンドリング
- 境界値ケース
"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.services.data_collection.mergers.oi_merger import OIMerger
from database.models import OpenInterestData


@pytest.fixture
def mock_oi_repository():
    """モックオープンインタレストリポジトリ"""
    repo = MagicMock()
    repo.get_open_interest_data = MagicMock()
    return repo


@pytest.fixture
def merger(mock_oi_repository):
    """マージャーインスタンス"""
    return OIMerger(mock_oi_repository)


@pytest.fixture
def sample_df():
    """サンプルDataFrame"""
    data = {
        "close": [29000.0, 29500.0, 30000.0],
        "volume": [100.0, 150.0, 200.0],
    }
    index = pd.DatetimeIndex(
        [
            datetime(2021, 1, 1, 0, 0),
            datetime(2021, 1, 1, 1, 0),
            datetime(2021, 1, 1, 2, 0),
        ]
    )
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_oi_data():
    """サンプルオープンインタレストデータ"""
    oi_data = []
    timestamps = [
        datetime(2021, 1, 1, 0, 0),
        datetime(2021, 1, 1, 1, 0),
        datetime(2021, 1, 1, 2, 0),
    ]
    values = [100000.0, 105000.0, 110000.0]

    for ts, val in zip(timestamps, values):
        oi = MagicMock(spec=OpenInterestData)
        oi.data_timestamp = ts
        oi.open_interest_value = val
        oi_data.append(oi)

    return oi_data


class TestMergerInitialization:
    """マージャー初期化テスト"""

    def test_merger_initialization(self, mock_oi_repository):
        """マージャーが正しく初期化されることを確認"""
        merger = OIMerger(mock_oi_repository)
        assert merger.oi_repo == mock_oi_repository

    def test_merger_with_repository(self, mock_oi_repository):
        """リポジトリが正しく設定されることを確認"""
        merger = OIMerger(mock_oi_repository)
        assert hasattr(merger, "oi_repo")
        assert merger.oi_repo is not None


class TestMergeOIData:
    """OIデータマージテスト"""

    def test_merge_oi_data_success(
        self, merger, mock_oi_repository, sample_df, sample_oi_data
    ):
        """OIデータが正常にマージされることを確認"""
        mock_oi_repository.get_open_interest_data.return_value = sample_oi_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(sample_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns
        assert len(result) == len(sample_df)
        mock_oi_repository.get_open_interest_data.assert_called_once_with(
            symbol="BTC/USDT:USDT", start_time=start_date, end_time=end_date
        )

    def test_merge_oi_data_no_data(self, merger, mock_oi_repository, sample_df):
        """OIデータがない場合、ゼロで埋められることを確認"""
        mock_oi_repository.get_open_interest_data.return_value = []

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(sample_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns
        assert all(result["open_interest"] == 0.0)

    def test_merge_oi_data_with_tolerance(
        self, merger, mock_oi_repository, sample_df, sample_oi_data
    ):
        """toleranceが正しく適用されることを確認"""
        mock_oi_repository.get_open_interest_data.return_value = sample_oi_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(sample_df, "BTC/USDT:USDT", start_date, end_date)

        # 1日以内のデータのみマージされる
        assert "open_interest" in result.columns

    def test_merge_oi_data_different_symbols(
        self, merger, mock_oi_repository, sample_df, sample_oi_data
    ):
        """異なるシンボルでマージが正しく行われることを確認"""
        mock_oi_repository.get_open_interest_data.return_value = sample_oi_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        # BTC/USDT:USDT
        result_btc = merger.merge_oi_data(
            sample_df.copy(), "BTC/USDT:USDT", start_date, end_date
        )

        # ETH/USDT
        result_eth = merger.merge_oi_data(
            sample_df.copy(), "ETH/USDT", start_date, end_date
        )

        assert "open_interest" in result_btc.columns
        assert "open_interest" in result_eth.columns


class TestConvertOIToDataFrame:
    """OI→DataFrame変換テスト"""

    def test_convert_oi_to_dataframe(self, merger, sample_oi_data):
        """OIデータがDataFrameに正しく変換されることを確認"""
        result = merger._convert_oi_to_dataframe(sample_oi_data)

        assert isinstance(result, pd.DataFrame)
        assert "open_interest" in result.columns
        assert len(result) == len(sample_oi_data)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_convert_oi_to_dataframe_values(self, merger, sample_oi_data):
        """変換後のDataFrameの値が正しいことを確認"""
        result = merger._convert_oi_to_dataframe(sample_oi_data)

        expected_values = [100000.0, 105000.0, 110000.0]
        assert list(result["open_interest"]) == expected_values

    def test_convert_oi_to_dataframe_index(self, merger, sample_oi_data):
        """変換後のDataFrameのインデックスが正しいことを確認"""
        result = merger._convert_oi_to_dataframe(sample_oi_data)

        expected_times = [
            datetime(2021, 1, 1, 0, 0),
            datetime(2021, 1, 1, 1, 0),
            datetime(2021, 1, 1, 2, 0),
        ]
        assert list(result.index) == expected_times

    def test_convert_empty_oi_data(self, merger):
        """空のOIデータリストが正しく処理されることを確認"""
        result = merger._convert_oi_to_dataframe([])

        assert isinstance(result, pd.DataFrame)
        assert "open_interest" in result.columns
        assert len(result) == 0


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_merge_oi_data_with_exception(self, merger, mock_oi_repository, sample_df):
        """例外発生時にゼロで埋められることを確認"""
        mock_oi_repository.get_open_interest_data.side_effect = Exception(
            "Database error"
        )

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(sample_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns
        assert all(result["open_interest"] == 0.0)

    def test_merge_oi_data_with_none_repository_response(
        self, merger, mock_oi_repository, sample_df
    ):
        """リポジトリがNoneを返した場合の処理を確認"""
        mock_oi_repository.get_open_interest_data.return_value = None

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(sample_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns
        assert all(result["open_interest"] == 0.0)


class TestEdgeCases:
    """エッジケーステスト"""

    def test_merge_with_empty_dataframe(
        self, merger, mock_oi_repository, sample_oi_data
    ):
        """空のDataFrameとのマージを確認"""
        mock_oi_repository.get_open_interest_data.return_value = sample_oi_data

        empty_df = pd.DataFrame()
        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(empty_df, "BTC/USDT:USDT", start_date, end_date)

        assert isinstance(result, pd.DataFrame)

    def test_merge_with_single_row_dataframe(
        self, merger, mock_oi_repository, sample_oi_data
    ):
        """1行だけのDataFrameとのマージを確認"""
        mock_oi_repository.get_open_interest_data.return_value = sample_oi_data

        single_row_df = pd.DataFrame(
            {"close": [29000.0]}, index=pd.DatetimeIndex([datetime(2021, 1, 1, 0, 0)])
        )
        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(single_row_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns
        assert len(result) == 1

    def test_merge_with_zero_open_interest(self, merger, mock_oi_repository):
        """ゼロ建玉が正しく処理されることを確認"""
        # ゼロ建玉を持つデータ
        oi_data = []
        oi = MagicMock(spec=OpenInterestData)
        oi.data_timestamp = datetime(2021, 1, 1, 0, 0)
        oi.open_interest_value = 0.0
        oi_data.append(oi)

        mock_oi_repository.get_open_interest_data.return_value = oi_data

        sample_df = pd.DataFrame(
            {"close": [29000.0]}, index=pd.DatetimeIndex([datetime(2021, 1, 1, 0, 0)])
        )
        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(sample_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns

    def test_merge_with_large_dataset(self, merger, mock_oi_repository):
        """大量データのマージを確認"""
        # 1000件のデータを作成
        oi_data = []
        for i in range(1000):
            oi = MagicMock(spec=OpenInterestData)
            oi.data_timestamp = datetime(2021, 1, 1, 0, 0) + pd.Timedelta(hours=i)
            oi.open_interest_value = 100000.0 + (i * 100.0)
            oi_data.append(oi)

        mock_oi_repository.get_open_interest_data.return_value = oi_data

        # 1000行のDataFrameを作成
        index = pd.date_range(start="2021-01-01", periods=1000, freq="h")
        large_df = pd.DataFrame({"close": range(1000)}, index=index)

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 12, 31, 0, 0)

        result = merger.merge_oi_data(large_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns
        assert len(result) == 1000

    def test_merge_with_mismatched_timestamps(
        self, merger, mock_oi_repository, sample_df
    ):
        """タイムスタンプが一致しない場合のマージを確認"""
        # 異なる時刻のOIデータ
        oi_data = []
        oi = MagicMock(spec=OpenInterestData)
        oi.data_timestamp = datetime(2021, 1, 1, 0, 30)  # 30分ずれている
        oi.open_interest_value = 100000.0
        oi_data.append(oi)

        mock_oi_repository.get_open_interest_data.return_value = oi_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(sample_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns
        # merge_asof with backward direction should handle this

    def test_merge_with_large_open_interest_values(self, merger, mock_oi_repository):
        """大きな建玉値が正しく処理されることを確認"""
        oi_data = []
        oi = MagicMock(spec=OpenInterestData)
        oi.data_timestamp = datetime(2021, 1, 1, 0, 0)
        oi.open_interest_value = 999999999999.99
        oi_data.append(oi)

        mock_oi_repository.get_open_interest_data.return_value = oi_data

        sample_df = pd.DataFrame(
            {"close": [29000.0]}, index=pd.DatetimeIndex([datetime(2021, 1, 1, 0, 0)])
        )
        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_oi_data(sample_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns

    def test_merge_with_hourly_vs_daily_data(self, merger, mock_oi_repository):
        """時間足と日足の異なる粒度のデータマージを確認"""
        # 日次OIデータ
        oi_data = []
        for i in range(7):  # 1週間分
            oi = MagicMock(spec=OpenInterestData)
            oi.data_timestamp = datetime(2021, 1, 1, 0, 0) + pd.Timedelta(days=i)
            oi.open_interest_value = 100000.0 + (i * 1000.0)
            oi_data.append(oi)

        mock_oi_repository.get_open_interest_data.return_value = oi_data

        # 時間足DataFrame
        index = pd.date_range(
            start="2021-01-01", periods=168, freq="h"
        )  # 1週間分の時間足
        hourly_df = pd.DataFrame({"close": range(168)}, index=index)

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 8, 0, 0)

        result = merger.merge_oi_data(hourly_df, "BTC/USDT:USDT", start_date, end_date)

        assert "open_interest" in result.columns
        assert len(result) == 168
        # toleranceが1日なので、時間足データに日次OIデータがマージされる


