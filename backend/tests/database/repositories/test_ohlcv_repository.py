"""
OHLCVRepositoryのテストモジュール

OHLCVデータリポジトリの機能をテストします。
"""

from datetime import datetime, timezone
from typing import Any
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
def sample_ohlcv_records() -> list[dict[str, Any]]:
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
        sample_ohlcv_records: list[dict[str, Any]],
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
        sample_ohlcv_records: list[dict[str, Any]],
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
        sample_ohlcv_records: list[dict[str, Any]],
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
        sample_ohlcv_records: list[dict[str, Any]],
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


class TestFetchOhlcvDataFrame:
    """fetch_ohlcv_dataframe の等価性テスト（実in-memory SQLiteを使用）"""

    BASE_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def real_session(self) -> Any:
        """実SQLite(in-memory)セッション"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from database.connection import Base

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        yield session
        session.close()
        engine.dispose()

    def _seed_rows(
        self,
        session: Any,
        count: int = 5,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1m",
    ) -> None:
        from datetime import timedelta

        for i in range(count):
            session.add(
                OHLCVData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=self.BASE_TIME + timedelta(minutes=i),
                    open=50000.0 + i,
                    high=50100.0 + i,
                    low=49900.0 + i,
                    close=50050.0 + i,
                    volume=10.0 * (i + 1),
                )
            )
        session.commit()

    def test_equivalent_to_legacy_orm_path(self, real_session: Any) -> None:
        """軽量経路が ORM 経路（get_ohlcv_data + convert）と完全に等価であること"""
        from app.services.backtest.data.data_conversion_service import (
            DataConversionService,
        )

        self._seed_rows(real_session, count=5)
        repo = OHLCVRepository(real_session)

        # 従来経路: ORMエンティティ → 変換サービス
        legacy_entities = repo.get_ohlcv_data(symbol="BTC/USDT:USDT", timeframe="1m")
        legacy_df = DataConversionService().convert_ohlcv_to_dataframe(legacy_entities)

        # 軽量経路: Core SELECT + read_sql
        fast_df = repo.fetch_ohlcv_dataframe(symbol="BTC/USDT:USDT", timeframe="1m")

        pd.testing.assert_frame_equal(fast_df, legacy_df)

    def test_time_range_filter_inclusive(self, real_session: Any) -> None:
        """start/end は従来経路と同じく両端を含む（>= / <=）"""
        from datetime import timedelta

        self._seed_rows(real_session, count=5)
        repo = OHLCVRepository(real_session)

        start = self.BASE_TIME + timedelta(minutes=1)
        end = self.BASE_TIME + timedelta(minutes=3)

        df = repo.fetch_ohlcv_dataframe(
            symbol="BTC/USDT:USDT", timeframe="1m", start_time=start, end_time=end
        )

        assert len(df) == 3
        # インデックスは従来経路と同じ naive UTC（tz情報は付かない）
        assert df.index.min() == start.replace(tzinfo=None)
        assert df.index.max() == end.replace(tzinfo=None)

    def test_empty_result_returns_empty_frame(self, real_session: Any) -> None:
        """該当データなしでは空のDataFrameを返す"""
        repo = OHLCVRepository(real_session)

        df = repo.fetch_ohlcv_dataframe(symbol="BTC/USDT:USDT", timeframe="1m")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_symbol_isolation(self, real_session: Any) -> None:
        """他シンボル・他時間軸のデータは混入しない"""
        self._seed_rows(real_session, count=3, symbol="ETH/USDT:USDT")
        self._seed_rows(real_session, count=2, timeframe="5m")
        self._seed_rows(real_session, count=2)
        repo = OHLCVRepository(real_session)

        df = repo.fetch_ohlcv_dataframe(symbol="BTC/USDT:USDT", timeframe="1m")

        assert len(df) == 2
