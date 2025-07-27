"""
BaseRepositoryの単体テスト
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from sqlalchemy.orm import Session

from database.repositories.base_repository import BaseRepository
from database.models import OHLCVData


class TestBaseRepository:
    """BaseRepositoryのテストクラス"""

    @pytest.fixture
    def mock_db(self):
        """モックデータベースセッション"""
        return Mock(spec=Session)

    @pytest.fixture
    def base_repository(self, mock_db):
        """BaseRepositoryのインスタンス"""
        return BaseRepository(mock_db, OHLCVData)

    @pytest.fixture
    def sample_records(self):
        """サンプルレコード"""
        now = datetime.now()
        records = []
        for i in range(5):
            record = Mock()
            record.id = i + 1
            record.symbol = "BTC/USDT"
            record.timestamp = now - timedelta(hours=i)
            record.open = 50000.0 + i * 100
            record.high = 51000.0 + i * 100
            record.low = 49000.0 + i * 100
            record.close = 50500.0 + i * 100
            record.volume = 1000.0 + i * 10
            records.append(record)
        return records

    def test_init(self, mock_db):
        """初期化のテスト"""
        repo = BaseRepository(mock_db, OHLCVData)
        assert repo.db == mock_db
        assert repo.model_class == OHLCVData

    @patch("database.repositories.base_repository.DatabaseQueryHelper")
    def test_get_filtered_data(
        self, mock_query_helper, base_repository, sample_records
    ):
        """get_filtered_dataメソッドのテスト"""
        mock_query_helper.get_filtered_records.return_value = sample_records

        filters = {"symbol": "BTC/USDT"}
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()

        result = base_repository.get_filtered_data(
            filters=filters,
            time_range_column="timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="timestamp",
            order_asc=True,
            limit=100,
        )

        assert result == sample_records
        mock_query_helper.get_filtered_records.assert_called_once_with(
            db=base_repository.db,
            model_class=OHLCVData,
            filters=filters,
            time_range_column="timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="timestamp",
            order_asc=True,
            limit=100,
        )

    @patch("database.repositories.base_repository.DatabaseQueryHelper")
    def test_get_latest_records(
        self, mock_query_helper, base_repository, sample_records
    ):
        """get_latest_recordsメソッドのテスト"""
        mock_query_helper.get_filtered_records.return_value = sample_records

        filters = {"symbol": "BTC/USDT"}
        result = base_repository.get_latest_records(
            filters=filters,
            timestamp_column="timestamp",
            limit=50,
        )

        assert result == sample_records
        mock_query_helper.get_filtered_records.assert_called_once_with(
            db=base_repository.db,
            model_class=OHLCVData,
            filters=filters,
            time_range_column="timestamp",
            start_time=None,
            end_time=None,
            order_by_column="timestamp",
            order_asc=False,
            limit=50,
        )

    @patch("database.repositories.base_repository.DatabaseQueryHelper")
    def test_get_data_in_range(
        self, mock_query_helper, base_repository, sample_records
    ):
        """get_data_in_rangeメソッドのテスト"""
        mock_query_helper.get_filtered_records.return_value = sample_records

        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        filters = {"symbol": "BTC/USDT"}

        result = base_repository.get_data_in_range(
            timestamp_column="timestamp",
            start_time=start_time,
            end_time=end_time,
            filters=filters,
            limit=100,
        )

        assert result == sample_records
        mock_query_helper.get_filtered_records.assert_called_once_with(
            db=base_repository.db,
            model_class=OHLCVData,
            filters=filters,
            time_range_column="timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="timestamp",
            order_asc=True,
            limit=100,
        )

    def test_to_dataframe_with_mapping(self, base_repository, sample_records):
        """to_dataframeメソッドのテスト（カラムマッピング有り）"""
        column_mapping = {
            "timestamp": "timestamp",
            "open": "open",
            "close": "close",
            "volume": "volume",
        }

        result = base_repository.to_dataframe(
            records=sample_records,
            column_mapping=column_mapping,
            index_column="timestamp",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert list(result.columns) == ["open", "close", "volume"]
        assert result.index.name == "timestamp"

    def test_to_dataframe_empty_records(self, base_repository):
        """to_dataframeメソッドのテスト（空のレコード）"""
        column_mapping = {
            "timestamp": "timestamp",
            "open": "open",
            "close": "close",
        }

        result = base_repository.to_dataframe(
            records=[],
            column_mapping=column_mapping,
            index_column="timestamp",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["timestamp", "open", "close"]

    def test_delete_by_date_range(self, base_repository, mock_db):
        """delete_by_date_rangeメソッドのテスト"""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.delete.return_value = 10
        mock_db.query.return_value = mock_query

        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        additional_filters = {"symbol": "BTC/USDT"}

        result = base_repository.delete_by_date_range(
            timestamp_column="timestamp",
            start_time=start_time,
            end_time=end_time,
            additional_filters=additional_filters,
        )

        assert result == 10
        mock_db.query.assert_called_once_with(OHLCVData)
        mock_db.commit.assert_called_once()

    def test_delete_old_data(self, base_repository):
        """delete_old_dataメソッドのテスト"""
        before_date = datetime.now() - timedelta(days=7)
        additional_filters = {"symbol": "BTC/USDT"}

        with patch.object(
            base_repository, "delete_by_date_range", return_value=5
        ) as mock_delete:
            result = base_repository.delete_old_data(
                timestamp_column="timestamp",
                before_date=before_date,
                additional_filters=additional_filters,
            )

            assert result == 5
            mock_delete.assert_called_once_with(
                timestamp_column="timestamp",
                end_time=before_date,
                additional_filters=additional_filters,
            )

    def test_get_data_statistics(self, base_repository, mock_db):
        """get_data_statisticsメソッドのテスト"""
        mock_result = Mock()
        mock_result.total_count = 100
        mock_result.oldest_timestamp = datetime.now() - timedelta(days=30)
        mock_result.newest_timestamp = datetime.now()

        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.with_entities.return_value = mock_query
        mock_query.first.return_value = mock_result
        mock_db.query.return_value = mock_query

        filters = {"symbol": "BTC/USDT"}
        result = base_repository.get_data_statistics(
            timestamp_column="timestamp",
            filters=filters,
        )

        assert result["total_count"] == 100
        assert result["oldest_timestamp"] == mock_result.oldest_timestamp
        assert result["newest_timestamp"] == mock_result.newest_timestamp
        assert result["date_range_days"] == 30

    def test_validate_records_success(self, base_repository):
        """validate_recordsメソッドのテスト（成功）"""
        records = [
            {"symbol": "BTC/USDT", "timestamp": datetime.now(), "price": 50000},
            {"symbol": "ETH/USDT", "timestamp": datetime.now(), "price": 3000},
        ]
        required_fields = ["symbol", "timestamp", "price"]

        result = base_repository.validate_records(records, required_fields)
        assert result is True

    def test_validate_records_missing_field(self, base_repository):
        """validate_recordsメソッドのテスト（必須フィールド不足）"""
        records = [
            {"symbol": "BTC/USDT", "timestamp": datetime.now()},  # priceが不足
        ]
        required_fields = ["symbol", "timestamp", "price"]

        result = base_repository.validate_records(records, required_fields)
        assert result is False

    def test_validate_records_with_custom_validation(self, base_repository):
        """validate_recordsメソッドのテスト（カスタム検証関数）"""
        records = [
            {"symbol": "BTC/USDT", "price": 50000},
            {"symbol": "ETH/USDT", "price": 3000},
        ]
        required_fields = ["symbol", "price"]

        def custom_validation(records):
            return all(record["price"] > 0 for record in records)

        result = base_repository.validate_records(
            records, required_fields, custom_validation
        )
        assert result is True

    def test_validate_records_custom_validation_fail(self, base_repository):
        """validate_recordsメソッドのテスト（カスタム検証失敗）"""
        records = [
            {"symbol": "BTC/USDT", "price": -50000},  # 負の価格
        ]
        required_fields = ["symbol", "price"]

        def custom_validation(records):
            return all(record["price"] > 0 for record in records)

        result = base_repository.validate_records(
            records, required_fields, custom_validation
        )
        assert result is False

    def test_validate_records_empty(self, base_repository):
        """validate_recordsメソッドのテスト（空のレコード）"""
        result = base_repository.validate_records([], ["symbol"])
        assert result is True
