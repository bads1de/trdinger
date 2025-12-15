"""
OpenInterestRepositoryのテストモジュール

オープンインタレストリポジトリの機能をテストします。
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from database.models import OpenInterestData
from database.repositories.open_interest_repository import OpenInterestRepository


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
def repository(mock_session: MagicMock) -> OpenInterestRepository:
    """OpenInterestRepositoryインスタンス"""
    return OpenInterestRepository(mock_session)


@pytest.fixture
def sample_open_interest_model() -> OpenInterestData:
    """サンプルOpenInterestDataモデル"""
    return OpenInterestData(
        id=1,
        symbol="BTC/USDT:USDT",
        open_interest_value=1000000.0,
        data_timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_open_interest_records() -> List[Dict[str, Any]]:
    """サンプルオープンインタレストレコードリスト"""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "symbol": "BTC/USDT:USDT",
            "open_interest_value": 1000000.0,
            "data_timestamp": base_time,
            "timestamp": base_time,
        },
        {
            "symbol": "BTC/USDT:USDT",
            "open_interest_value": 1100000.0,
            "data_timestamp": datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
            "timestamp": datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
        },
    ]


class TestRepositoryInitialization:
    """リポジトリ初期化のテスト"""

    def test_repository_initialization(self, mock_session: MagicMock) -> None:
        """リポジトリが正しく初期化される"""
        repo = OpenInterestRepository(mock_session)
        assert repo.db == mock_session
        assert repo.model_class == OpenInterestData


class TestToDictMethod:
    """to_dictメソッドのテスト"""

    def test_to_dict_basic(
        self,
        repository: OpenInterestRepository,
        sample_open_interest_model: OpenInterestData,
    ) -> None:
        """モデルインスタンスが辞書に変換される"""
        result = repository.to_dict(sample_open_interest_model)

        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["symbol"] == "BTC/USDT:USDT"
        assert result["open_interest_value"] == 1000000.0


class TestInsertOpenInterestData:
    """insert_open_interest_dataメソッドのテスト"""

    def test_insert_open_interest_data_success(
        self,
        repository: OpenInterestRepository,
        sample_open_interest_records: List[Dict[str, Any]],
    ) -> None:
        """オープンインタレストデータが正常に挿入される"""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        repository.db.execute.return_value = mock_result

        count = repository.insert_open_interest_data(sample_open_interest_records)

        assert count == 2
        repository.db.commit.assert_called_once()

    def test_insert_open_interest_data_empty(
        self, repository: OpenInterestRepository
    ) -> None:
        """空のレコードで0が返される"""
        count = repository.insert_open_interest_data([])

        assert count == 0


class TestGetOpenInterestData:
    """get_open_interest_dataメソッドのテスト"""

    def test_get_open_interest_data_success(
        self,
        repository: OpenInterestRepository,
        sample_open_interest_model: OpenInterestData,
    ) -> None:
        """オープンインタレストデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_open_interest_model]
        repository.db.scalars.return_value = mock_scalars

        results = repository.get_open_interest_data(symbol="BTC/USDT:USDT")

        assert len(results) == 1
        assert results[0] == sample_open_interest_model

    def test_get_open_interest_data_with_time_range(
        self, repository: OpenInterestRepository
    ) -> None:
        """時間範囲指定でデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        repository.get_open_interest_data(
            symbol="BTC/USDT:USDT",
            start_time=start_time,
            end_time=end_time,
        )

        repository.db.scalars.assert_called_once()

    def test_get_open_interest_data_with_limit(
        self, repository: OpenInterestRepository
    ) -> None:
        """リミット指定でデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars

        repository.get_open_interest_data(symbol="BTC/USDT:USDT", limit=10)

        repository.db.scalars.assert_called_once()


class TestTimestampMethods:
    """タイムスタンプ関連メソッドのテスト"""

    def test_get_latest_open_interest_timestamp_success(
        self, repository: OpenInterestRepository
    ) -> None:
        """最新オープンインタレストタイムスタンプが取得できる"""
        expected_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.return_value = expected_time

        result = repository.get_latest_open_interest_timestamp("BTC/USDT:USDT")

        assert result == expected_time

    def test_get_oldest_open_interest_timestamp_success(
        self, repository: OpenInterestRepository
    ) -> None:
        """最古オープンインタレストタイムスタンプが取得できる"""
        expected_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.return_value = expected_time

        result = repository.get_oldest_open_interest_timestamp("BTC/USDT:USDT")

        assert result == expected_time


class TestDeleteOperations:
    """削除操作のテスト"""

    def test_clear_all_open_interest_data_success(
        self, repository: OpenInterestRepository
    ) -> None:
        """全オープンインタレストデータが削除される"""
        mock_result = MagicMock()
        mock_result.rowcount = 10
        repository.db.execute.return_value = mock_result

        count = repository.clear_all_open_interest_data()

        assert count == 10
        repository.db.commit.assert_called_once()

    def test_clear_open_interest_data_by_symbol_success(
        self, repository: OpenInterestRepository
    ) -> None:
        """シンボル指定でデータが削除される"""
        mock_result = MagicMock()
        mock_result.rowcount = 5
        repository.db.execute.return_value = mock_result

        count = repository.clear_open_interest_data_by_symbol("BTC/USDT:USDT")

        assert count == 5
        repository.db.commit.assert_called_once()


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_insert_handles_db_error(
        self,
        repository: OpenInterestRepository,
        sample_open_interest_records: List[Dict[str, Any]],
    ) -> None:
        """データベースエラーが処理される（safe_operationが例外をキャッチ）"""
        repository.db.execute.side_effect = Exception("DB Error")

        # safe_operationデコレータがエラーを処理するため、例外は発生しない
        result = repository.insert_open_interest_data(sample_open_interest_records)

        # safe_operationによりエラー時は0が返される
        assert result == 0

    def test_delete_handles_error(self, repository: OpenInterestRepository) -> None:
        """削除エラーが処理される"""
        repository.db.execute.side_effect = Exception("Delete Error")

        with pytest.raises(Exception):
            repository.clear_all_open_interest_data()

        repository.db.rollback.assert_called()


