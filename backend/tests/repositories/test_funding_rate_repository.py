"""
FundingRateRepositoryのテストモジュール

ファンディングレートリポジトリの機能をテストします。
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from database.models import FundingRateData
from database.repositories.funding_rate_repository import FundingRateRepository


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
def repository(mock_session: MagicMock) -> FundingRateRepository:
    """
    FundingRateRepositoryインスタンス

    Args:
        mock_session: モックセッション

    Returns:
        FundingRateRepository: テスト用リポジトリインスタンス
    """
    return FundingRateRepository(mock_session)


@pytest.fixture
def sample_funding_rate_model() -> FundingRateData:
    """
    サンプルFundingRateDataモデルインスタンス

    Returns:
        FundingRateData: テスト用ファンディングレートデータ
    """
    return FundingRateData(
        id=1,
        symbol="BTC/USDT:USDT",
        funding_rate=0.0001,
        funding_timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        next_funding_timestamp=datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
        mark_price=50000.0,
        index_price=50010.0,
    )


@pytest.fixture
def sample_funding_rate_records() -> List[Dict[str, Any]]:
    """
    サンプルファンディングレートレコードリスト

    Returns:
        List[Dict[str, Any]]: テスト用レコードリスト
    """
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "symbol": "BTC/USDT:USDT",
            "funding_rate": 0.0001,
            "fundingRate": 0.0001,
            "funding_timestamp": int(base_time.timestamp() * 1000),
            "timestamp": int(base_time.timestamp() * 1000),
            "mark_price": 50000.0,
            "index_price": 50010.0,
        },
        {
            "symbol": "ETH/USDT:USDT",
            "funding_rate": 0.0002,
            "funding_timestamp": int(
                datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
            ),
            "timestamp": int(
                datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
            ),
        },
    ]


class TestRepositoryInitialization:
    """リポジトリ初期化のテスト"""

    def test_repository_initialization(self, mock_session: MagicMock) -> None:
        """リポジトリが正しく初期化される"""
        repo = FundingRateRepository(mock_session)

        assert repo.db == mock_session
        assert repo.model_class == FundingRateData


class TestInsertFundingRateData:
    """insert_funding_rate_dataメソッドのテスト"""

    def test_insert_funding_rate_data_success(
        self,
        repository: FundingRateRepository,
        sample_funding_rate_records: List[Dict[str, Any]],
    ) -> None:
        """ファンディングレートデータが正常に挿入される"""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        repository.db.execute.return_value = mock_result

        count = repository.insert_funding_rate_data(sample_funding_rate_records)

        assert count == 2
        repository.db.commit.assert_called_once()

    def test_insert_funding_rate_data_empty(
        self, repository: FundingRateRepository
    ) -> None:
        """空のレコードで0が返される"""
        count = repository.insert_funding_rate_data([])

        assert count == 0

    def test_insert_funding_rate_data_with_info_field(
        self, repository: FundingRateRepository
    ) -> None:
        """infoフィールドを含むデータが処理される"""
        records = [
            {
                "symbol": "BTC/USDT:USDT",
                "info": {
                    "fundingRate": "0.0001",
                    "fundingRateTimestamp": 1704067200000,
                },
                "timestamp": 1704067200000,
            }
        ]

        mock_result = MagicMock()
        mock_result.rowcount = 1
        repository.db.execute.return_value = mock_result

        count = repository.insert_funding_rate_data(records)

        assert count == 1

    def test_insert_funding_rate_data_skips_invalid(
        self, repository: FundingRateRepository
    ) -> None:
        """無効なレコードがスキップされる"""
        records = [
            {"symbol": "BTC/USDT:USDT"},  # 必須フィールド不足
        ]

        count = repository.insert_funding_rate_data(records)

        assert count == 0


class TestGetFundingRateData:
    """get_funding_rate_dataメソッドのテスト"""

    def test_get_funding_rate_data_success(
        self,
        repository: FundingRateRepository,
        sample_funding_rate_model: FundingRateData,
    ) -> None:
        """ファンディングレートデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_funding_rate_model]
        repository.db.scalars.return_value = mock_scalars

        results = repository.get_funding_rate_data(symbol="BTC/USDT:USDT")

        assert len(results) == 1
        assert results[0] == sample_funding_rate_model

    def test_get_funding_rate_data_with_time_range(
        self,
        repository: FundingRateRepository,
        sample_funding_rate_model: FundingRateData,
    ) -> None:
        """時間範囲指定でデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_funding_rate_model]
        repository.db.scalars.return_value = mock_scalars

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        results = repository.get_funding_rate_data(
            symbol="BTC/USDT:USDT",
            start_time=start_time,
            end_time=end_time,
        )

        assert len(results) == 1

    def test_get_funding_rate_data_with_limit(
        self, repository: FundingRateRepository
    ) -> None:
        """リミット指定でデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars

        repository.get_funding_rate_data(symbol="BTC/USDT:USDT", limit=10)

        repository.db.scalars.assert_called_once()


class TestGetAllBySymbol:
    """get_all_by_symbolメソッドのテスト"""

    def test_get_all_by_symbol_success(
        self,
        repository: FundingRateRepository,
        sample_funding_rate_model: FundingRateData,
    ) -> None:
        """シンボル指定で全データが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_funding_rate_model]
        repository.db.scalars.return_value = mock_scalars

        results = repository.get_all_by_symbol("BTC/USDT:USDT")

        assert len(results) == 1


class TestTimestampMethods:
    """タイムスタンプ関連メソッドのテスト"""

    def test_get_latest_funding_timestamp_success(
        self, repository: FundingRateRepository
    ) -> None:
        """最新ファンディングタイムスタンプが取得できる"""
        expected_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.return_value = expected_time

        result = repository.get_latest_funding_timestamp("BTC/USDT:USDT")

        assert result == expected_time

    def test_get_oldest_funding_timestamp_success(
        self, repository: FundingRateRepository
    ) -> None:
        """最古ファンディングタイムスタンプが取得できる"""
        expected_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        repository.db.scalar.return_value = expected_time

        result = repository.get_oldest_funding_timestamp("BTC/USDT:USDT")

        assert result == expected_time


class TestCountMethods:
    """カウントメソッドのテスト"""

    def test_get_funding_rate_count_success(
        self, repository: FundingRateRepository
    ) -> None:
        """ファンディングレートデータ件数が取得できる"""
        repository.db.scalar.return_value = 100

        count = repository.get_funding_rate_count("BTC/USDT:USDT")

        assert count == 100


class TestDeleteOperations:
    """削除操作のテスト"""

    def test_clear_all_funding_rate_data_success(
        self, repository: FundingRateRepository
    ) -> None:
        """全ファンディングレートデータが削除される"""
        mock_result = MagicMock()
        mock_result.rowcount = 10
        repository.db.execute.return_value = mock_result

        count = repository.clear_all_funding_rate_data()

        assert count == 10
        repository.db.commit.assert_called_once()

    def test_clear_funding_rate_data_by_symbol_success(
        self, repository: FundingRateRepository
    ) -> None:
        """シンボル指定でデータが削除される"""
        mock_result = MagicMock()
        mock_result.rowcount = 5
        repository.db.execute.return_value = mock_result

        count = repository.clear_funding_rate_data_by_symbol("BTC/USDT:USDT")

        assert count == 5
        repository.db.commit.assert_called_once()

    def test_clear_funding_rate_data_by_date_range_success(
        self, repository: FundingRateRepository
    ) -> None:
        """期間指定でデータが削除される"""
        mock_result = MagicMock()
        mock_result.rowcount = 3
        repository.db.execute.return_value = mock_result

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        count = repository.clear_funding_rate_data_by_date_range(
            symbol="BTC/USDT:USDT",
            start_time=start_time,
            end_time=end_time,
        )

        assert count == 3


class TestGetFundingRateDataframe:
    """get_funding_rate_dataframeメソッドのテスト"""

    def test_get_funding_rate_dataframe_success(
        self,
        repository: FundingRateRepository,
        sample_funding_rate_model: FundingRateData,
    ) -> None:
        """DataFrameとしてデータが取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_funding_rate_model]
        repository.db.scalars.return_value = mock_scalars

        df = repository.get_funding_rate_dataframe(symbol="BTC/USDT:USDT")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "funding_rate" in df.columns

    def test_get_funding_rate_dataframe_empty(
        self, repository: FundingRateRepository
    ) -> None:
        """データがない場合空のDataFrameが返される"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars

        df = repository.get_funding_rate_dataframe(symbol="BTC/USDT:USDT")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestDataValidation:
    """データ検証のテスト"""

    def test_insert_handles_zero_funding_rate(
        self, repository: FundingRateRepository
    ) -> None:
        """ファンディングレート0が正しく処理される"""
        records = [
            {
                "symbol": "BTC/USDT:USDT",
                "funding_rate": 0.0,
                "funding_timestamp": 1704067200000,
                "timestamp": 1704067200000,
            }
        ]

        mock_result = MagicMock()
        mock_result.rowcount = 1
        repository.db.execute.return_value = mock_result

        count = repository.insert_funding_rate_data(records)

        assert count == 1

    def test_insert_handles_negative_funding_rate(
        self, repository: FundingRateRepository
    ) -> None:
        """負のファンディングレートが処理される"""
        records = [
            {
                "symbol": "BTC/USDT:USDT",
                "funding_rate": -0.0001,
                "funding_timestamp": 1704067200000,
                "timestamp": 1704067200000,
            }
        ]

        mock_result = MagicMock()
        mock_result.rowcount = 1
        repository.db.execute.return_value = mock_result

        count = repository.insert_funding_rate_data(records)

        assert count == 1


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_insert_handles_conversion_error(
        self, repository: FundingRateRepository
    ) -> None:
        """データ変換エラーが処理される"""
        records = [
            {
                "symbol": "BTC/USDT:USDT",
                "funding_rate": "invalid",  # 無効な値
                "funding_timestamp": 1704067200000,
                "timestamp": 1704067200000,
            }
        ]

        count = repository.insert_funding_rate_data(records)

        # 変換エラーのレコードはスキップされる
        assert count == 0

    def test_insert_rollback_on_db_error(
        self, repository: FundingRateRepository
    ) -> None:
        """データベースエラー時の処理（safe_operationが例外をキャッチ）"""
        repository.db.execute.side_effect = Exception("DB Error")

        records = [
            {
                "symbol": "BTC/USDT:USDT",
                "funding_rate": 0.0001,
                "funding_timestamp": 1704067200000,
                "timestamp": 1704067200000,
            }
        ]

        # safe_operationデコレータがエラーを処理するため、例外は発生しない
        result = repository.insert_funding_rate_data(records)

        # safe_operationによりエラー時は0が返される
        assert result == 0




