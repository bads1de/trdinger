"""
Fear & Greed Index 統合テスト
"""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import copy

from database.connection import Base
from database.models import FearGreedIndexData
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from app.services.data_collection.fear_greed.fear_greed_service import (
    FearGreedIndexService,
)
from app.utils.data_converter import DataValidator


class TestFearGreedIntegration:
    """Fear & Greed Index統合テストクラス"""

    @pytest.fixture(scope="function")
    def db_session(self):
        """テスト用データベースセッション"""
        # インメモリSQLiteデータベースを使用
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)

        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        yield session

        session.close()

    @pytest.fixture
    def fear_greed_repository(self, db_session):
        """FearGreedIndexRepositoryのインスタンス"""
        return FearGreedIndexRepository(db_session)

    @pytest.fixture
    def sample_fear_greed_data(self):
        """サンプルFear & Greed Indexデータ"""
        base_time = datetime.now(timezone.utc).replace(microsecond=0)
        data = []

        classifications = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        values = [10, 30, 50, 70, 90]

        for i in range(5):
            data.append(
                {
                    "value": values[i],
                    "value_classification": classifications[i],
                    "data_timestamp": base_time - timedelta(days=i),
                    "timestamp": base_time,
                }
            )

        return data

    def test_insert_and_get_fear_greed_data(
        self, fear_greed_repository, sample_fear_greed_data
    ):
        """Fear & Greed Indexデータの挿入と取得のテスト"""
        # データを挿入
        inserted_count = fear_greed_repository.insert_fear_greed_data(
            sample_fear_greed_data
        )
        assert inserted_count == 5

        # データを取得
        retrieved_data = fear_greed_repository.get_fear_greed_data()
        assert len(retrieved_data) == 5

        # 最初のレコードを確認
        first_record = retrieved_data[0]
        assert first_record.value == 90  # 最新のデータ（i=0）
        assert first_record.value_classification == "Extreme Greed"

    def test_get_latest_fear_greed_data(
        self, fear_greed_repository, sample_fear_greed_data
    ):
        """最新Fear & Greed Indexデータ取得のテスト"""
        fear_greed_repository.insert_fear_greed_data(sample_fear_greed_data)

        latest_data = fear_greed_repository.get_latest_fear_greed_data(limit=3)
        assert len(latest_data) == 3

        # 降順で取得されることを確認
        assert latest_data[0].data_timestamp > latest_data[1].data_timestamp

    def test_get_latest_data_timestamp(
        self, fear_greed_repository, sample_fear_greed_data
    ):
        """最新データタイムスタンプ取得のテスト"""
        fear_greed_repository.insert_fear_greed_data(sample_fear_greed_data)

        latest_timestamp = fear_greed_repository.get_latest_data_timestamp()

        assert latest_timestamp is not None
        # 最新のタイムスタンプ（i=0のデータ）と一致することを確認
        expected_timestamp = sample_fear_greed_data[0]["data_timestamp"]
        assert latest_timestamp == expected_timestamp

    def test_get_data_count(self, fear_greed_repository, sample_fear_greed_data):
        """データ件数取得のテスト"""
        fear_greed_repository.insert_fear_greed_data(sample_fear_greed_data)

        count = fear_greed_repository.get_data_count()
        assert count == 5

    def test_get_data_range(self, fear_greed_repository, sample_fear_greed_data):
        """データ範囲取得のテスト"""
        fear_greed_repository.insert_fear_greed_data(sample_fear_greed_data)

        data_range = fear_greed_repository.get_data_range()

        assert data_range["total_count"] == 5
        assert data_range["oldest_data"] is not None
        assert data_range["newest_data"] is not None

    def test_delete_old_data(self, fear_greed_repository, sample_fear_greed_data):
        """古いデータ削除のテスト"""
        fear_greed_repository.insert_fear_greed_data(sample_fear_greed_data)

        # 2日前より古いデータを削除
        base_time = datetime.now(timezone.utc)
        cutoff_date = base_time - timedelta(days=2)

        deleted_count = fear_greed_repository.delete_old_data(cutoff_date)
        assert deleted_count == 3  # 3, 4日前のデータが削除される

        # 残りのデータを確認
        remaining_data = fear_greed_repository.get_fear_greed_data()
        assert len(remaining_data) == 2

    def test_duplicate_data_handling(
        self, fear_greed_repository, sample_fear_greed_data
    ):
        """重複データ処理のテスト"""
        # 同じデータを2回挿入
        first_insert = fear_greed_repository.insert_fear_greed_data(
            sample_fear_greed_data
        )
        second_insert = fear_greed_repository.insert_fear_greed_data(
            sample_fear_greed_data
        )

        assert first_insert == 5
        assert second_insert == 0  # 重複データは挿入されない

        # データ件数を確認
        count = fear_greed_repository.get_data_count()
        assert count == 5

    def test_data_validation(self, sample_fear_greed_data):
        """データ検証のテスト"""
        # 有効なデータの検証
        assert DataValidator.validate_fear_greed_data(sample_fear_greed_data) is True

        # 無効なデータの検証
        invalid_data = copy.deepcopy(sample_fear_greed_data)
        invalid_data[0]["value"] = 150  # 範囲外の値

        assert DataValidator.validate_fear_greed_data(invalid_data) is False

    def test_get_fear_greed_dataframe(
        self, fear_greed_repository, sample_fear_greed_data
    ):
        """Fear & Greed IndexデータのDataFrame取得のテスト"""
        fear_greed_repository.insert_fear_greed_data(sample_fear_greed_data)

        df = fear_greed_repository.get_fear_greed_dataframe()

        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["value", "value_classification"]
        assert df.index.name == "data_timestamp"

    def test_data_filtering_by_time_range(
        self, fear_greed_repository, sample_fear_greed_data
    ):
        """時間範囲でのデータフィルタリングのテスト"""
        fear_greed_repository.insert_fear_greed_data(sample_fear_greed_data)

        base_time = datetime.now(timezone.utc).replace(microsecond=0)
        start_time = base_time - timedelta(days=3)
        end_time = base_time - timedelta(days=1)

        filtered_data = fear_greed_repository.get_fear_greed_data(
            start_time=start_time, end_time=end_time
        )

        # 指定した範囲内のデータのみ取得されることを確認
        assert len(filtered_data) == 3  # days 3, 2, 1

        for record in filtered_data:
            # タイムゾーン情報が失われている場合はUTCを設定
            record_timestamp = record.data_timestamp
            if record_timestamp.tzinfo is None:
                record_timestamp = record_timestamp.replace(tzinfo=timezone.utc)
            assert start_time <= record_timestamp <= end_time

    @pytest.mark.asyncio
    async def test_fear_greed_service_data_conversion(self):
        """Fear & Greed Serviceのデータ変換テスト"""
        service = FearGreedIndexService()

        # APIレスポンスのサンプル
        api_data = [
            {
                "value": "25",
                "value_classification": "Fear",
                "timestamp": "1640995200",  # 2022-01-01 00:00:00 UTC
                "time_until_update": "86400",
            },
            {
                "value": "75",
                "value_classification": "Greed",
                "timestamp": "1640908800",  # 2021-12-31 00:00:00 UTC
                "time_until_update": "86400",
            },
        ]

        converted_data = service._convert_api_data_to_db_format(api_data)

        assert len(converted_data) == 2

        # 最初のレコードを確認
        first_record = converted_data[0]
        assert first_record["value"] == 25
        assert first_record["value_classification"] == "Fear"
        assert isinstance(first_record["data_timestamp"], datetime)
        assert isinstance(first_record["timestamp"], datetime)

        # データ検証
        assert DataValidator.validate_fear_greed_data(converted_data) is True

    def test_incremental_update_scenario(self, fear_greed_repository):
        """差分更新シナリオのテスト"""
        # 初期データを挿入
        base_time = datetime.now(timezone.utc).replace(microsecond=0)
        initial_data = [
            {
                "value": 50,
                "value_classification": "Neutral",
                "data_timestamp": base_time - timedelta(days=2),
                "timestamp": base_time - timedelta(hours=1),
            },
            {
                "value": 60,
                "value_classification": "Greed",
                "data_timestamp": base_time - timedelta(days=1),
                "timestamp": base_time - timedelta(hours=1),
            },
        ]

        fear_greed_repository.insert_fear_greed_data(initial_data)

        # 最新タイムスタンプを取得
        latest_timestamp = fear_greed_repository.get_latest_data_timestamp()
        expected_timestamp = base_time - timedelta(days=1)
        assert latest_timestamp == expected_timestamp

        # 新しいデータを追加（差分更新をシミュレート）
        new_data = [
            {
                "value": 70,
                "value_classification": "Greed",
                "data_timestamp": base_time,
                "timestamp": base_time,
            }
        ]

        inserted_count = fear_greed_repository.insert_fear_greed_data(new_data)
        assert inserted_count == 1

        # 更新後の最新タイムスタンプを確認
        updated_latest_timestamp = fear_greed_repository.get_latest_data_timestamp()
        assert updated_latest_timestamp == base_time

        # 総データ数を確認
        total_count = fear_greed_repository.get_data_count()
        assert total_count == 3
