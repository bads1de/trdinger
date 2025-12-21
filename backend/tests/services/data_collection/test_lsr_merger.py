"""
Long Short Ratio マージャーのテスト

LSRMergerクラスの全機能をテストします:
- 初期化
- DataFrameマージ処理
- データ変換
- エラーハンドリング
- パラメータ付き検証
"""

from datetime import datetime
from unittest.mock import MagicMock
import pandas as pd
import pytest

from app.services.data_collection.mergers.lsr_merger import LSRMerger
from database.models import LongShortRatioData


@pytest.fixture
def mock_lsr_repository():
    """モックLSRリポジトリ"""
    repo = MagicMock()
    repo.get_long_short_ratio_data = MagicMock()
    return repo


@pytest.fixture
def merger(mock_lsr_repository):
    """マージャーインスタンス"""
    return LSRMerger(mock_lsr_repository)


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
def sample_lsr_data():
    """サンプルLSRデータ"""
    lsr_data = []
    timestamps = [
        datetime(2021, 1, 1, 0, 0),
        datetime(2021, 1, 1, 1, 0),
        datetime(2021, 1, 1, 2, 0),
    ]
    buy_ratios = [0.6, 0.55, 0.5]
    sell_ratios = [0.4, 0.45, 0.5]

    for ts, b_rate, s_rate in zip(timestamps, buy_ratios, sell_ratios):
        lsr = MagicMock(spec=LongShortRatioData)
        lsr.timestamp = ts
        lsr.buy_ratio = b_rate
        lsr.sell_ratio = s_rate
        lsr_data.append(lsr)

    return lsr_data


class TestMergerInitialization:
    """マージャー初期化テスト"""

    def test_merger_initialization(self, mock_lsr_repository):
        """マージャーが正しく初期化されることを確認"""
        merger = LSRMerger(mock_lsr_repository)
        assert merger.lsr_repo == mock_lsr_repository


class TestMergeLSRData:
    """LSRデータマージテスト"""

    def test_merge_lsr_data_success(
        self, merger, mock_lsr_repository, sample_df, sample_lsr_data
    ):
        """LSRデータが正常にマージされることを確認"""
        mock_lsr_repository.get_long_short_ratio_data.return_value = sample_lsr_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_lsr_data(
            sample_df, "BTC/USDT:USDT", "1h", start_date, end_date
        )

        assert "lsr_buy_ratio" in result.columns
        assert "lsr_sell_ratio" in result.columns
        assert len(result) == len(sample_df)
        mock_lsr_repository.get_long_short_ratio_data.assert_called_once_with(
            symbol="BTC/USDT:USDT",
            period="1h",
            start_time=start_date,
            end_time=end_date,
        )

    def test_merge_lsr_data_no_data(self, merger, mock_lsr_repository, sample_df):
        """LSRデータがない場合、デフォルト値（0.5）で埋められることを確認"""
        mock_lsr_repository.get_long_short_ratio_data.return_value = []

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_lsr_data(
            sample_df, "BTC/USDT:USDT", "1h", start_date, end_date
        )

        assert "lsr_buy_ratio" in result.columns
        assert "lsr_sell_ratio" in result.columns
        assert all(result["lsr_buy_ratio"] == 0.5)
        assert all(result["lsr_sell_ratio"] == 0.5)

    def test_merge_lsr_data_with_tolerance(
        self, merger, mock_lsr_repository, sample_df, sample_lsr_data
    ):
        """toleranceが正しく適用されることを確認"""
        mock_lsr_repository.get_long_short_ratio_data.return_value = sample_lsr_data

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        # 4時間以内ならマージされる（デフォルトtolerance）
        result = merger.merge_lsr_data(
            sample_df, "BTC/USDT:USDT", "1h", start_date, end_date
        )

        assert "lsr_buy_ratio" in result.columns
        assert result["lsr_buy_ratio"].notna().all()


class TestConvertLSRToDataFrame:
    """LSR→DataFrame変換テスト"""

    def test_convert_lsr_to_dataframe(self, merger, sample_lsr_data):
        """LSRデータがDataFrameに正しく変換されることを確認"""
        result = merger._convert_lsr_to_dataframe(sample_lsr_data)

        assert isinstance(result, pd.DataFrame)
        assert "lsr_buy_ratio" in result.columns
        assert "lsr_sell_ratio" in result.columns
        assert len(result) == len(sample_lsr_data)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_convert_empty_lsr_data(self, merger):
        """空のLSRデータリストが正しく処理されることを確認"""
        result = merger._convert_lsr_to_dataframe([])

        assert isinstance(result, pd.DataFrame)
        assert "lsr_buy_ratio" in result.columns
        assert len(result) == 0


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_merge_lsr_data_with_exception(
        self, merger, mock_lsr_repository, sample_df
    ):
        """例外発生時にデフォルト値で埋められることを確認"""
        mock_lsr_repository.get_long_short_ratio_data.side_effect = Exception(
            "Database error"
        )

        start_date = datetime(2021, 1, 1, 0, 0)
        end_date = datetime(2021, 1, 2, 0, 0)

        result = merger.merge_lsr_data(
            sample_df, "BTC/USDT:USDT", "1h", start_date, end_date
        )

        assert "lsr_buy_ratio" in result.columns
        assert all(result["lsr_buy_ratio"] == 0.5)
