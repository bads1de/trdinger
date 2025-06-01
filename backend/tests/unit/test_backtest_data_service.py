"""
バックテストデータサービスのテスト
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import Mock
from app.core.services.backtest_data_service import BacktestDataService
from database.models import OHLCVData


class TestBacktestDataService:
    """BacktestDataServiceのテスト"""

    @pytest.fixture
    def mock_ohlcv_repo(self):
        """モックOHLCVリポジトリ"""
        mock_repo = Mock()
        return mock_repo

    @pytest.fixture
    def backtest_data_service(self, mock_ohlcv_repo):
        """BacktestDataServiceインスタンス"""
        service = BacktestDataService()
        service.ohlcv_repo = mock_ohlcv_repo
        return service

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        return [
            OHLCVData(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                open=50000.0,
                high=50500.0,
                low=49500.0,
                close=50200.0,
                volume=100.0,
            ),
            OHLCVData(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
                open=50200.0,
                high=50800.0,
                low=50000.0,
                close=50600.0,
                volume=150.0,
            ),
            OHLCVData(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
                open=50600.0,
                high=51000.0,
                low=50300.0,
                close=50800.0,
                volume=120.0,
            ),
        ]

    def test_get_ohlcv_for_backtest_success(
        self, backtest_data_service, mock_ohlcv_repo, sample_ohlcv_data
    ):
        """正常なOHLCVデータ取得テスト"""
        # モックの設定
        mock_ohlcv_repo.get_ohlcv_data.return_value = sample_ohlcv_data

        # テスト実行
        result_df = backtest_data_service.get_ohlcv_for_backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        # 検証
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert list(result_df.columns) == ["Open", "High", "Low", "Close", "Volume"]

        # データの内容を確認
        assert result_df.iloc[0]["Open"] == 50000.0
        assert result_df.iloc[0]["High"] == 50500.0
        assert result_df.iloc[0]["Low"] == 49500.0
        assert result_df.iloc[0]["Close"] == 50200.0
        assert result_df.iloc[0]["Volume"] == 100.0

        # インデックスがdatetimeであることを確認
        assert isinstance(result_df.index, pd.DatetimeIndex)
        assert result_df.index[0] == pd.Timestamp(
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        )

    def test_get_ohlcv_for_backtest_no_data(
        self, backtest_data_service, mock_ohlcv_repo
    ):
        """データが見つからない場合のテスト"""
        # モックの設定（空のリストを返す）
        mock_ohlcv_repo.get_ohlcv_data.return_value = []

        # テスト実行とエラー確認
        with pytest.raises(ValueError, match="No data found for BTC/USDT 1h"):
            backtest_data_service.get_ohlcv_for_backtest(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )

    def test_get_ohlcv_for_backtest_data_sorting(
        self, backtest_data_service, mock_ohlcv_repo
    ):
        """データのソート機能テスト"""
        # 順序がバラバラのデータを準備
        unsorted_data = [
            OHLCVData(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),  # 3番目
                open=50600.0,
                high=51000.0,
                low=50300.0,
                close=50800.0,
                volume=120.0,
            ),
            OHLCVData(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),  # 1番目
                open=50000.0,
                high=50500.0,
                low=49500.0,
                close=50200.0,
                volume=100.0,
            ),
            OHLCVData(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),  # 2番目
                open=50200.0,
                high=50800.0,
                low=50000.0,
                close=50600.0,
                volume=150.0,
            ),
        ]

        # モックの設定
        mock_ohlcv_repo.get_ohlcv_data.return_value = unsorted_data

        # テスト実行
        result_df = backtest_data_service.get_ohlcv_for_backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        # ソートされていることを確認
        expected_timestamps = [
            pd.Timestamp(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            pd.Timestamp(datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)),
            pd.Timestamp(datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)),
        ]

        assert list(result_df.index) == expected_timestamps
        assert result_df.iloc[0]["Close"] == 50200.0  # 最初のデータ
        assert result_df.iloc[1]["Close"] == 50600.0  # 2番目のデータ
        assert result_df.iloc[2]["Close"] == 50800.0  # 3番目のデータ

    def test_get_ohlcv_for_backtest_repository_call(
        self, backtest_data_service, mock_ohlcv_repo, sample_ohlcv_data
    ):
        """リポジトリが正しいパラメータで呼び出されることを確認"""
        # モックの設定
        mock_ohlcv_repo.get_ohlcv_data.return_value = sample_ohlcv_data

        # テスト実行
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        backtest_data_service.get_ohlcv_for_backtest(
            symbol="BTC/USDT", timeframe="1h", start_date=start_date, end_date=end_date
        )

        # リポジトリが正しいパラメータで呼び出されたことを確認
        mock_ohlcv_repo.get_ohlcv_data.assert_called_once_with(
            symbol="BTC/USDT", timeframe="1h", start_time=start_date, end_time=end_date
        )

    def test_dataframe_structure_validation(self, backtest_data_service):
        """DataFrameの構造検証テスト"""
        # 空のDataFrameのテスト
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            backtest_data_service._validate_dataframe(empty_df)

    def test_convert_ohlcv_data_to_dataframe(
        self, backtest_data_service, sample_ohlcv_data
    ):
        """OHLCVデータのDataFrame変換テスト"""
        result_df = backtest_data_service._convert_to_dataframe(sample_ohlcv_data)

        # 基本的な構造の確認
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert list(result_df.columns) == ["Open", "High", "Low", "Close", "Volume"]

        # データ型の確認
        assert result_df["Open"].dtype == "float64"
        assert result_df["High"].dtype == "float64"
        assert result_df["Low"].dtype == "float64"
        assert result_df["Close"].dtype == "float64"
        assert result_df["Volume"].dtype == "float64"
