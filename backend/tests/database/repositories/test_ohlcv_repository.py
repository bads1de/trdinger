"""
OHLCVRepositoryのテストモジュール

OHLCVデータリポジトリの機能をテストします。
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from database.models import OHLCVData
from database.repositories.ohlcv_repository import OHLCVRepository


@pytest.fixture
def mock_session() -> MagicMock:
    """モックDBセッション"""
    session = MagicMock(spec=Session)
    session.execute = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.scalar = MagicMock()
    session.scalars = MagicMock()

    # bind属性をモック化
    mock_bind = MagicMock()
    mock_engine = MagicMock()
    mock_dialect = MagicMock()
    mock_dialect.name = "sqlite"
    mock_engine.dialect = mock_dialect
    mock_bind.engine = mock_engine
    session.bind = mock_bind

    return session


@pytest.fixture
def repository(mock_session: MagicMock) -> OHLCVRepository:
    """OHLCVRepositoryインスタンス"""
    return OHLCVRepository(mock_session)


@pytest.fixture
def sample_ohlcv_model() -> OHLCVData:
    """サンプルOHLCVDataモデル"""
    return OHLCVData(
        id=1,
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.0,
    )


@pytest.fixture
def sample_ohlcv_records() -> List[Dict[str, Any]]:
    """サンプルOHLCVレコードリスト"""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "timestamp": base_time,
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0,
        },
        {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "timestamp": datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
            "open": 50500.0,
            "high": 51500.0,
            "low": 50000.0,
            "close": 51000.0,
            "volume": 150.0,
        },
    ]


class TestRepositoryInitialization:
    """リポジトリ初期化のテスト"""

    def test_repository_initialization(self, mock_session: MagicMock) -> None:
        """リポジトリが正しく初期化される"""
        repo = OHLCVRepository(mock_session)
        assert repo.db == mock_session
        assert repo.model_class == OHLCVData


class TestInsertOHLCVData:
    """insert_ohlcv_dataメソッドのテスト"""

    @patch("database.repositories.ohlcv_repository.DataValidator")
    def test_insert_ohlcv_data_success(
        self,
        mock_validator: MagicMock,
        repository: OHLCVRepository,
        sample_ohlcv_records: List[Dict[str, Any]],
    ) -> None:
        """OHLCVデータが正常に挿入される"""
        mock_validator.validate_ohlcv_records_simple.return_value = True
        mock_result = MagicMock()
        mock_result.rowcount = 2
        repository.db.execute.return_value = mock_result

        count = repository.insert_ohlcv_data(sample_ohlcv_records)

        assert count == 2
        repository.db.commit.assert_called_once()

    def test_insert_ohlcv_data_empty(self, repository: OHLCVRepository) -> None:
        """空のレコードで0が返される"""
        count = repository.insert_ohlcv_data([])

        assert count == 0

    @patch("database.repositories.ohlcv_repository.DataValidator")
    def test_insert_ohlcv_data_validation_failure(
        self,
        mock_validator: MagicMock,
        repository: OHLCVRepository,
        sample_ohlcv_records: List[Dict[str, Any]],
    ) -> None:
        """検証失敗時にエラーが発生する"""
        mock_validator.validate_ohlcv_records_simple.return_value = False

        with pytest.raises(ValueError):
            repository.insert_ohlcv_data(sample_ohlcv_records)


class TestGetOHLCVData:
    """get_ohlcv_dataメソッドのテスト"""

    def test_get_ohlcv_data_success(
        self,
        repository: OHLCVRepository,
        sample_ohlcv_model: OHLCVData,
    ) -> None:
        """OHLCVデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_ohlcv_model]
        repository.db.scalars.return_value = mock_scalars

        results = repository.get_ohlcv_data(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
        )

        assert len(results) == 1
        assert results[0] == sample_ohlcv_model

    def test_get_ohlcv_data_with_time_range(self, repository: OHLCVRepository) -> None:
        """時間範囲指定でデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        repository.get_ohlcv_data(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_time=start_time,
            end_time=end_time,
        )

        repository.db.scalars.assert_called_once()


class TestGetAllBySymbol:
    """get_all_by_symbolメソッドのテスト"""

    def test_get_all_by_symbol_success(
        self,
        repository: OHLCVRepository,
        sample_ohlcv_model: OHLCVData,
    ) -> None:
        """シンボル指定で全データが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_ohlcv_model]
        repository.db.scalars.return_value = mock_scalars

        results = repository.get_all_by_symbol("BTC/USDT:USDT", "1h")

        assert len(results) == 1


class TestGetLatestOHLCVData:
    """get_latest_ohlcv_dataメソッドのテスト"""

    def test_get_latest_ohlcv_data_success(
        self,
        repository: OHLCVRepository,
        sample_ohlcv_model: OHLCVData,
    ) -> None:
        """最新OHLCVデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_ohlcv_model]
        repository.db.scalars.return_value = mock_scalars

        results = repository.get_latest_ohlcv_data(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            limit=100,
        )

        assert len(results) == 1


class TestTimestampOverrides:
    """タイムスタンプメソッドのオーバーライドテスト"""

    def test_get_latest_timestamp_success(self, repository: OHLCVRepository) -> None:
        """最新タイムスタンプが取得できる"""
        expected_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.return_value = expected_time

        result = repository.get_latest_timestamp(
            "timestamp", {"symbol": "BTC/USDT:USDT"}
        )

        assert result == expected_time

    def test_get_oldest_timestamp_success(self, repository: OHLCVRepository) -> None:
        """最古タイムスタンプが取得できる"""
        expected_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.return_value = expected_time

        result = repository.get_oldest_timestamp(
            "timestamp", {"symbol": "BTC/USDT:USDT"}
        )

        assert result == expected_time


class TestCountMethods:
    """カウントメソッドのテスト"""

    def test_get_data_count_success(self, repository: OHLCVRepository) -> None:
        """データ件数が取得できる"""
        repository.db.scalar.return_value = 100

        count = repository.get_data_count("BTC/USDT:USDT", "1h")

        assert count == 100


class TestDeleteOperations:
    """削除操作のテスト"""

    def test_clear_all_ohlcv_data_success(self, repository: OHLCVRepository) -> None:
        """全OHLCVデータが削除される"""
        mock_result = MagicMock()
        mock_result.rowcount = 10
        repository.db.execute.return_value = mock_result

        count = repository.clear_all_ohlcv_data()

        assert count == 10

    def test_clear_ohlcv_data_by_symbol_success(
        self, repository: OHLCVRepository
    ) -> None:
        """シンボル指定でデータが削除される"""
        mock_result = MagicMock()
        mock_result.rowcount = 5
        repository.db.execute.return_value = mock_result

        count = repository.clear_ohlcv_data_by_symbol("BTC/USDT:USDT")

        assert count == 5

    def test_clear_ohlcv_data_by_symbol_and_timeframe_success(
        self, repository: OHLCVRepository
    ) -> None:
        """シンボルと時間軸指定でデータが削除される"""
        mock_result = MagicMock()
        mock_result.rowcount = 3
        repository.db.execute.return_value = mock_result

        count = repository.clear_ohlcv_data_by_symbol_and_timeframe(
            "BTC/USDT:USDT", "1h"
        )

        assert count == 3


class TestGetOHLCVDataframe:
    """get_ohlcv_dataframeメソッドのテスト"""

    def test_get_ohlcv_dataframe_success(
        self,
        repository: OHLCVRepository,
        sample_ohlcv_model: OHLCVData,
    ) -> None:
        """DataFrameとしてデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_ohlcv_model]
        repository.db.scalars.return_value = mock_scalars

        df = repository.get_ohlcv_dataframe(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_get_ohlcv_dataframe_empty(self, repository: OHLCVRepository) -> None:
        """データがない場合空のDataFrameが返される"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars

        df = repository.get_ohlcv_dataframe(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestSanitizeOHLCVData:
    """sanitize_ohlcv_dataメソッドのテスト"""

    @patch("database.repositories.ohlcv_repository.DataValidator")
    def test_sanitize_ohlcv_data_success(
        self,
        mock_validator: MagicMock,
        repository: OHLCVRepository,
        sample_ohlcv_records: List[Dict[str, Any]],
    ) -> None:
        """OHLCVデータがサニタイズされる"""
        mock_validator.sanitize_ohlcv_data.return_value = sample_ohlcv_records

        result = repository.sanitize_ohlcv_data(sample_ohlcv_records)

        assert result == sample_ohlcv_records
        mock_validator.sanitize_ohlcv_data.assert_called_once()


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @patch("database.repositories.ohlcv_repository.DataValidator")
    def test_insert_handles_db_error(
        self,
        mock_validator: MagicMock,
        repository: OHLCVRepository,
        sample_ohlcv_records: List[Dict[str, Any]],
    ) -> None:
        """データベースエラーが処理される（safe_operationが例外をキャッチ）"""
        mock_validator.validate_ohlcv_records_simple.return_value = True
        repository.db.execute.side_effect = Exception("DB Error")

        # safe_operationデコレータがエラーを処理するため、例外は発生しない
        result = repository.insert_ohlcv_data(sample_ohlcv_records)

        # safe_operationによりエラー時は0が返される
        assert result == 0

    def test_delete_handles_error(self, repository: OHLCVRepository) -> None:
        """削除エラーが処理される"""
        repository.db.execute.side_effect = Exception("Delete Error")

        with pytest.raises(Exception):
            repository.clear_ohlcv_data_by_symbol_and_timeframe("BTC/USDT:USDT", "1h")

        repository.db.rollback.assert_called()
