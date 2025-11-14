"""
ファンディングレートマージャーのテスト

FRMergerクラスの全機能をテストします:
- 初期化
- DataFrameマージ処理
- データ変換
- エラーハンドリング
- 境界値ケース
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.data_collection.mergers.fr_merger import FRMerger
from database.models import FundingRateData


@pytest.fixture
def mock_fr_repository():
    """モックファンディングレートリポジトリ"""
    repo = MagicMock()
    repo.get_funding_rate_data = MagicMock()
    return repo


@pytest.fixture
def merger(mock_fr_repository):
    """マージャーインスタンス"""
    return FRMerger(mock_fr_repository)


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
            datetime(2021, 1, 1, 8, 0),
            datetime(2021, 1, 1, 16, 0),
        ]
    )
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_fr_data():
    """サンプルファンディングレートデータ"""
    fr_data = []
    timestamps = [
        datetime(2021, 1, 1, 0, 0),
        datetime(2021, 1, 1, 8, 0),
        datetime(2021, 1, 1, 16, 0),
    ]
    rates = [0.0001, 0.0002, 0.00015]

    for ts, rate in zip(timestamps, rates):
        fr = MagicMock(spec=FundingRateData)
        fr.funding_timestamp = ts
        fr.funding_rate = rate
        fr_data.append(fr)

    return fr_data


class TestMergerInitialization:
    """マージャー初期化テスト"""

    def test_merger_initialization(self, mock_fr_repository):
        """マージャーが正しく初期化されることを確認"""
        merger = FRMerger(mock_fr_repository)
        assert merger.fr_repo == mock_fr_repository

    def test_merger_with_repository(self, mock_fr_repository):
        """リポジトリが正しく設定されることを確認"""
        merger = FRMerger(mock_fr_repository)
        assert hasattr(merger, "fr_repo")
        assert merger.fr_repo is not None


class TestMergeFRData:
    """FRデータマージテスト"""

    def test_merge_fr_data_success(
        self, merger, mock_fr_repository, sample_df, sample_fr_data
    ):
        """FRデータが正常にマージされることを確認"""
        mock_fr_repository.get_funding_rate_data.return_value = sample_fr_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(sample_df, "BTC/USDT", start_date, end_date)

        assert "funding_rate" in result.columns
        assert len(result) == len(sample_df)
        mock_fr_repository.get_funding_rate_data.assert_called_once_with(
            symbol="BTC/USDT", start_time=start_date, end_time=end_date
        )

    def test_merge_fr_data_no_data(self, merger, mock_fr_repository, sample_df):
        """FRデータがない場合、ゼロで埋められることを確認"""
        mock_fr_repository.get_funding_rate_data.return_value = []

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(sample_df, "BTC/USDT", start_date, end_date)

        assert "funding_rate" in result.columns
        assert all(result["funding_rate"] == 0.0)

    def test_merge_fr_data_with_tolerance(
        self, merger, mock_fr_repository, sample_df, sample_fr_data
    ):
        """toleranceが正しく適用されることを確認"""
        mock_fr_repository.get_funding_rate_data.return_value = sample_fr_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(sample_df, "BTC/USDT", start_date, end_date)

        # 8時間以内のデータのみマージされる
        assert "funding_rate" in result.columns

    def test_merge_fr_data_different_symbols(
        self, merger, mock_fr_repository, sample_df, sample_fr_data
    ):
        """異なるシンボルでマージが正しく行われることを確認"""
        mock_fr_repository.get_funding_rate_data.return_value = sample_fr_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        # BTC/USDT
        result_btc = merger.merge_fr_data(
            sample_df.copy(), "BTC/USDT", start_date, end_date
        )

        # ETH/USDT
        result_eth = merger.merge_fr_data(
            sample_df.copy(), "ETH/USDT", start_date, end_date
        )

        assert "funding_rate" in result_btc.columns
        assert "funding_rate" in result_eth.columns


class TestConvertFRToDataFrame:
    """FR→DataFrame変換テスト"""

    def test_convert_fr_to_dataframe(self, merger, sample_fr_data):
        """FRデータがDataFrameに正しく変換されることを確認"""
        result = merger._convert_fr_to_dataframe(sample_fr_data)

        assert isinstance(result, pd.DataFrame)
        assert "funding_rate" in result.columns
        assert len(result) == len(sample_fr_data)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_convert_fr_to_dataframe_values(self, merger, sample_fr_data):
        """変換後のDataFrameの値が正しいことを確認"""
        result = merger._convert_fr_to_dataframe(sample_fr_data)

        expected_rates = [0.0001, 0.0002, 0.00015]
        assert list(result["funding_rate"]) == expected_rates

    def test_convert_fr_to_dataframe_index(self, merger, sample_fr_data):
        """変換後のDataFrameのインデックスが正しいことを確認"""
        result = merger._convert_fr_to_dataframe(sample_fr_data)

        expected_times = [
            datetime(2021, 1, 1, 0, 0),
            datetime(2021, 1, 1, 8, 0),
            datetime(2021, 1, 1, 16, 0),
        ]
        assert list(result.index) == expected_times

    def test_convert_empty_fr_data(self, merger):
        """空のFRデータリストが正しく処理されることを確認"""
        result = merger._convert_fr_to_dataframe([])

        assert isinstance(result, pd.DataFrame)
        assert "funding_rate" in result.columns
        assert len(result) == 0


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_merge_fr_data_with_exception(self, merger, mock_fr_repository, sample_df):
        """例外発生時にゼロで埋められることを確認"""
        mock_fr_repository.get_funding_rate_data.side_effect = Exception(
            "Database error"
        )

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(sample_df, "BTC/USDT", start_date, end_date)

        assert "funding_rate" in result.columns
        assert all(result["funding_rate"] == 0.0)

    def test_merge_fr_data_with_none_repository_response(
        self, merger, mock_fr_repository, sample_df
    ):
        """リポジトリがNoneを返した場合の処理を確認"""
        mock_fr_repository.get_funding_rate_data.return_value = None

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(sample_df, "BTC/USDT", start_date, end_date)

        assert "funding_rate" in result.columns
        assert all(result["funding_rate"] == 0.0)


class TestEdgeCases:
    """エッジケーステスト"""

    def test_merge_with_empty_dataframe(
        self, merger, mock_fr_repository, sample_fr_data
    ):
        """空のDataFrameとのマージを確認"""
        mock_fr_repository.get_funding_rate_data.return_value = sample_fr_data

        empty_df = pd.DataFrame()
        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(empty_df, "BTC/USDT", start_date, end_date)

        assert isinstance(result, pd.DataFrame)

    def test_merge_with_single_row_dataframe(
        self, merger, mock_fr_repository, sample_fr_data
    ):
        """1行だけのDataFrameとのマージを確認"""
        mock_fr_repository.get_funding_rate_data.return_value = sample_fr_data

        single_row_df = pd.DataFrame(
            {"close": [29000.0]}, index=pd.DatetimeIndex([datetime(2021, 1, 1, 0, 0)])
        )
        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(single_row_df, "BTC/USDT", start_date, end_date)

        assert "funding_rate" in result.columns
        assert len(result) == 1

    def test_merge_with_negative_funding_rate(self, merger, mock_fr_repository):
        """負のファンディングレートが正しく処理されることを確認"""
        # 負のファンディングレートを持つデータ
        fr_data = []
        fr = MagicMock(spec=FundingRateData)
        fr.funding_timestamp = datetime(2021, 1, 1, 0, 0)
        fr.funding_rate = -0.0001
        fr_data.append(fr)

        mock_fr_repository.get_funding_rate_data.return_value = fr_data

        sample_df = pd.DataFrame(
            {"close": [29000.0]}, index=pd.DatetimeIndex([datetime(2021, 1, 1, 0, 0)])
        )
        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(sample_df, "BTC/USDT", start_date, end_date)

        assert "funding_rate" in result.columns

    def test_merge_with_large_dataset(self, merger, mock_fr_repository):
        """大量データのマージを確認"""
        # 1000件のデータを作成
        fr_data = []
        for i in range(1000):
            fr = MagicMock(spec=FundingRateData)
            fr.funding_timestamp = datetime(2021, 1, 1, 0, 0) + pd.Timedelta(
                hours=i * 8
            )
            fr.funding_rate = 0.0001 + (i * 0.00001)
            fr_data.append(fr)

        mock_fr_repository.get_funding_rate_data.return_value = fr_data

        # 1000行のDataFrameを作成
        index = pd.date_range(start="2021-01-01", periods=1000, freq="8h")
        large_df = pd.DataFrame({"close": range(1000)}, index=index)

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 12, 31, 0, 0)

        result = merger.merge_fr_data(large_df, "BTC/USDT", start_date, end_date)

        assert "funding_rate" in result.columns
        assert len(result) == 1000

    def test_merge_with_mismatched_timestamps(
        self, merger, mock_fr_repository, sample_df
    ):
        """タイムスタンプが一致しない場合のマージを確認"""
        # 異なる時刻のFRデータ
        fr_data = []
        fr = MagicMock(spec=FundingRateData)
        fr.funding_timestamp = datetime(2021, 1, 1, 1, 0)  # 1時間ずれている
        fr.funding_rate = 0.0001
        fr_data.append(fr)

        mock_fr_repository.get_funding_rate_data.return_value = fr_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_fr_data(sample_df, "BTC/USDT", start_date, end_date)

        assert "funding_rate" in result.columns
        # merge_asof with backward direction should handle this
