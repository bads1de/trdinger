"""
BaseRepositoryのテストモジュール

全リポジトリで共通の基底クラスの機能をテストします。
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pandas as pd
import pytest
from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import Session

from database.models import OHLCVData
from database.repositories.base_repository import BaseRepository


@pytest.fixture
def mock_session() -> MagicMock:
    """
    モックDBセッション

    Returns:
        MagicMock: モックされたデータベースセッション
    """
    session = MagicMock(spec=Session)
    session.execute = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.refresh = MagicMock()
    session.add = MagicMock()
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
def repository(mock_session: MagicMock) -> BaseRepository:
    """
    BaseRepositoryインスタンス

    Args:
        mock_session: モックセッション

    Returns:
        BaseRepository: テスト用リポジトリインスタンス
    """
    return BaseRepository(mock_session, OHLCVData)


@pytest.fixture
def sample_ohlcv_model() -> OHLCVData:
    """
    サンプルOHLCVモデルインスタンス

    Returns:
        OHLCVData: テスト用OHLCVデータ
    """
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
        created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """
    サンプルレコードリスト

    Returns:
        List[Dict[str, Any]]: テスト用レコードリスト
    """
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

    def test_repository_initialization(
        self, mock_session: MagicMock
    ) -> None:
        """リポジトリが正しく初期化される"""
        repo = BaseRepository(mock_session, OHLCVData)
        
        assert repo.db == mock_session
        assert repo.model_class == OHLCVData

    def test_repository_with_different_model(
        self, mock_session: MagicMock
    ) -> None:
        """異なるモデルクラスで初期化できる"""
        from database.models import FundingRateData
        
        repo = BaseRepository(mock_session, FundingRateData)
        
        assert repo.model_class == FundingRateData


class TestToDictMethod:
    """to_dictメソッドのテスト"""

    def test_to_dict_basic(
        self, repository: BaseRepository, sample_ohlcv_model: OHLCVData
    ) -> None:
        """モデルインスタンスが辞書に変換される"""
        result = repository.to_dict(sample_ohlcv_model)
        
        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["symbol"] == "BTC/USDT:USDT"
        assert result["timeframe"] == "1h"
        assert result["open"] == 50000.0
        assert result["close"] == 50500.0

    def test_to_dict_datetime_conversion(
        self, repository: BaseRepository, sample_ohlcv_model: OHLCVData
    ) -> None:
        """datetimeがISO形式の文字列に変換される"""
        result = repository.to_dict(sample_ohlcv_model)
        
        assert isinstance(result["timestamp"], str)
        assert "2024-01-01" in result["timestamp"]


class TestBulkInsertMethods:
    """一括挿入メソッドのテスト"""

    def test_bulk_insert_sqlite_success(
        self, repository: BaseRepository, sample_records: List[Dict[str, Any]]
    ) -> None:
        """SQLite用の一括挿入が成功する"""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        repository.db.execute.return_value = mock_result
        
        count = repository._bulk_insert_sqlite_ignore(sample_records)
        
        assert count == 2
        assert repository.db.execute.call_count == 2

    def test_bulk_insert_sqlite_empty_records(
        self, repository: BaseRepository
    ) -> None:
        """空のレコードリストで0が返される"""
        count = repository._bulk_insert_sqlite_ignore([])
        
        assert count == 0
        repository.db.execute.assert_not_called()

    def test_bulk_insert_with_conflict_handling_sqlite(
        self, repository: BaseRepository, sample_records: List[Dict[str, Any]]
    ) -> None:
        """重複処理付き一括挿入がSQLiteで動作する"""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        repository.db.execute.return_value = mock_result
        
        count = repository.bulk_insert_with_conflict_handling(
            sample_records, ["symbol", "timeframe", "timestamp"]
        )
        
        assert count == 2
        repository.db.commit.assert_called_once()

    def test_bulk_insert_with_conflict_handling_empty(
        self, repository: BaseRepository
    ) -> None:
        """空のレコードで0が返される"""
        count = repository.bulk_insert_with_conflict_handling([], ["symbol"])
        
        assert count == 0


class TestTimestampMethods:
    """タイムスタンプ関連メソッドのテスト"""

    def test_get_latest_timestamp_success(
        self, repository: BaseRepository
    ) -> None:
        """最新タイムスタンプが取得できる"""
        expected_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.return_value = expected_time
        
        result = repository.get_latest_timestamp("timestamp", {"symbol": "BTC/USDT"})
        
        assert result == expected_time

    def test_get_latest_timestamp_none(
        self, repository: BaseRepository
    ) -> None:
        """データがない場合Noneが返される"""
        repository.db.scalar.return_value = None
        
        result = repository.get_latest_timestamp("timestamp")
        
        assert result is None

    def test_get_oldest_timestamp_success(
        self, repository: BaseRepository
    ) -> None:
        """最古タイムスタンプが取得できる"""
        expected_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.return_value = expected_time
        
        result = repository.get_oldest_timestamp("timestamp", {"symbol": "BTC/USDT"})
        
        assert result == expected_time

    def test_get_date_range_success(
        self, repository: BaseRepository
    ) -> None:
        """データ期間が取得できる"""
        oldest = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        latest = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.side_effect = [oldest, latest]
        
        oldest_result, latest_result = repository.get_date_range("timestamp")
        
        assert oldest_result == oldest
        assert latest_result == latest


class TestRecordCountMethods:
    """レコード数関連メソッドのテスト"""

    def test_get_record_count_success(
        self, repository: BaseRepository
    ) -> None:
        """レコード数が取得できる"""
        repository.db.scalar.return_value = 100
        
        count = repository.get_record_count({"symbol": "BTC/USDT"})
        
        assert count == 100

    def test_get_record_count_zero(
        self, repository: BaseRepository
    ) -> None:
        """レコードがない場合0が返される"""
        repository.db.scalar.return_value = 0
        
        count = repository.get_record_count()
        
        assert count == 0

    def test_get_record_count_none_returns_zero(
        self, repository: BaseRepository
    ) -> None:
        """Noneの場合0が返される"""
        repository.db.scalar.return_value = None
        
        count = repository.get_record_count()
        
        assert count == 0


class TestDeleteOperations:
    """削除操作のテスト"""

    def test_delete_all_records_success(
        self, repository: BaseRepository
    ) -> None:
        """全レコード削除が成功する"""
        mock_result = MagicMock()
        mock_result.rowcount = 10
        repository.db.execute.return_value = mock_result
        
        count = repository._delete_all_records()
        
        assert count == 10
        repository.db.commit.assert_called_once()

    def test_delete_records_by_filter_success(
        self, repository: BaseRepository
    ) -> None:
        """フィルター付き削除が成功する"""
        mock_result = MagicMock()
        mock_result.rowcount = 5
        repository.db.execute.return_value = mock_result
        
        count = repository._delete_records_by_filter("symbol", "BTC/USDT")
        
        assert count == 5
        repository.db.commit.assert_called_once()

    def test_delete_by_date_range_success(
        self, repository: BaseRepository
    ) -> None:
        """期間指定削除が成功する"""
        mock_result = MagicMock()
        mock_result.rowcount = 3
        repository.db.execute.return_value = mock_result
        
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        
        count = repository.delete_by_date_range(
            "timestamp", start_time, end_time
        )
        
        assert count == 3
        repository.db.commit.assert_called_once()

    def test_delete_old_data_success(
        self, repository: BaseRepository
    ) -> None:
        """古いデータ削除が成功する"""
        mock_result = MagicMock()
        mock_result.rowcount = 7
        repository.db.execute.return_value = mock_result
        
        before_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        count = repository.delete_old_data("timestamp", before_date)
        
        assert count == 7


class TestFilteredDataRetrieval:
    """フィルター付きデータ取得のテスト"""

    def test_get_filtered_data_basic(
        self, repository: BaseRepository, sample_ohlcv_model: OHLCVData
    ) -> None:
        """基本的なフィルター付きデータ取得"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_ohlcv_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_filtered_data(
            filters={"symbol": "BTC/USDT:USDT"}
        )
        
        assert len(results) == 1
        assert results[0] == sample_ohlcv_model

    def test_get_filtered_data_with_time_range(
        self, repository: BaseRepository, sample_ohlcv_model: OHLCVData
    ) -> None:
        """時間範囲フィルター付きデータ取得"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_ohlcv_model]
        repository.db.scalars.return_value = mock_scalars
        
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        
        results = repository.get_filtered_data(
            time_range_column="timestamp",
            start_time=start_time,
            end_time=end_time,
        )
        
        assert len(results) == 1

    def test_get_filtered_data_with_limit(
        self, repository: BaseRepository
    ) -> None:
        """リミット付きデータ取得"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars
        
        repository.get_filtered_data(limit=10)
        
        # limitが呼ばれることを確認（間接的）
        repository.db.scalars.assert_called_once()

    def test_get_latest_records(
        self, repository: BaseRepository, sample_ohlcv_model: OHLCVData
    ) -> None:
        """最新レコード取得が機能する"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_ohlcv_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_latest_records(
            timestamp_column="timestamp",
            limit=10,
        )
        
        assert len(results) == 1


class TestDataFrameConversion:
    """DataFrame変換のテスト"""

    def test_to_dataframe_basic(
        self, repository: BaseRepository, sample_ohlcv_model: OHLCVData
    ) -> None:
        """基本的なDataFrame変換"""
        records = [sample_ohlcv_model]
        column_mapping = {
            "timestamp": "timestamp",
            "open": "open",
            "close": "close",
        }
        
        df = repository.to_dataframe(records, column_mapping)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "open" in df.columns
        assert "close" in df.columns

    def test_to_dataframe_empty_records(
        self, repository: BaseRepository
    ) -> None:
        """空のレコードで空のDataFrameが返される"""
        column_mapping = {"timestamp": "timestamp"}
        
        df = repository.to_dataframe([], column_mapping)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_to_dataframe_with_index(
        self, repository: BaseRepository, sample_ohlcv_model: OHLCVData
    ) -> None:
        """インデックス指定でDataFrame変換"""
        records = [sample_ohlcv_model]
        column_mapping = {
            "timestamp": "timestamp",
            "close": "close",
        }
        
        df = repository.to_dataframe(
            records, column_mapping, index_column="timestamp"
        )
        
        assert df.index.name == "timestamp"


class TestDataStatistics:
    """データ統計のテスト"""

    def test_get_data_statistics_success(
        self, repository: BaseRepository
    ) -> None:
        """データ統計情報が取得できる"""
        mock_result = MagicMock()
        mock_result.total_count = 100
        mock_result.oldest_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_result.newest_timestamp = datetime(2024, 1, 10, tzinfo=timezone.utc)
        repository.db.execute.return_value.first.return_value = mock_result
        
        stats = repository.get_data_statistics("timestamp")
        
        assert stats["total_count"] == 100
        assert stats["date_range_days"] == 9


class TestAvailableSymbols:
    """利用可能シンボル取得のテスト"""

    def test_get_available_symbols_success(
        self, repository: BaseRepository
    ) -> None:
        """利用可能なシンボルが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = ["BTC/USDT", "ETH/USDT"]
        repository.db.scalars.return_value = mock_scalars
        
        symbols = repository.get_available_symbols()
        
        assert len(symbols) == 2
        assert "BTC/USDT" in symbols


class TestValidationMethods:
    """検証メソッドのテスト"""

    def test_validate_records_success(
        self, repository: BaseRepository, sample_records: List[Dict[str, Any]]
    ) -> None:
        """レコード検証が成功する"""
        required_fields = ["symbol", "timestamp", "open"]
        
        is_valid = repository.validate_records(sample_records, required_fields)
        
        assert is_valid is True

    def test_validate_records_missing_field(
        self, repository: BaseRepository
    ) -> None:
        """必須フィールドがない場合Falseが返される"""
        records = [{"symbol": "BTC/USDT"}]
        required_fields = ["symbol", "timestamp"]
        
        is_valid = repository.validate_records(records, required_fields)
        
        assert is_valid is False

    def test_validate_records_empty(
        self, repository: BaseRepository
    ) -> None:
        """空のレコードでTrueが返される"""
        is_valid = repository.validate_records([], ["symbol"])
        
        assert is_valid is True


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_bulk_insert_rollback_on_error(
        self, repository: BaseRepository, sample_records: List[Dict[str, Any]]
    ) -> None:
        """エラー時の処理（safe_operationが例外をキャッチ）"""
        repository.db.execute.side_effect = Exception("DB Error")
        
        # safe_operationデコレータがエラーを処理するため、例外は発生しない
        result = repository.bulk_insert_with_conflict_handling(
            sample_records, ["symbol"]
        )
        
        # safe_operationによりエラー時は0が返される
        assert result == 0

    def test_delete_rollback_on_error(
        self, repository: BaseRepository
    ) -> None:
        """削除エラー時にロールバックされる"""
        repository.db.execute.side_effect = Exception("Delete Error")
        
        with pytest.raises(Exception):
            repository._delete_all_records()
        
        repository.db.rollback.assert_called()