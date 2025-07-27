"""
OHLCVRepositoryの統合テスト
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import copy

from database.connection import Base
from database.models import OHLCVData
from database.repositories.ohlcv_repository import OHLCVRepository


class TestOHLCVRepositoryIntegration:
    """OHLCVRepositoryの統合テストクラス"""

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
    def ohlcv_repository(self, db_session):
        """OHLCVRepositoryのインスタンス"""
        return OHLCVRepository(db_session)

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        base_time = datetime.now().replace(microsecond=0)
        data = []

        for i in range(10):
            data.append(
                {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "timestamp": base_time - timedelta(hours=i),
                    "open": 50000.0 + i * 100,
                    "high": 51000.0 + i * 100,
                    "low": 49000.0 + i * 100,
                    "close": 50500.0 + i * 100,
                    "volume": 1000.0 + i * 10,
                }
            )

        return data

    def test_insert_and_get_ohlcv_data(self, ohlcv_repository, sample_ohlcv_data):
        """OHLCVデータの挿入と取得のテスト"""
        # データを挿入
        inserted_count = ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)
        assert inserted_count == 10

        # データを取得
        retrieved_data = ohlcv_repository.get_ohlcv_data("BTC/USDT", "1h")
        assert len(retrieved_data) == 10

        # 最初のレコードを確認
        first_record = retrieved_data[0]
        assert first_record.symbol == "BTC/USDT"
        assert first_record.timeframe == "1h"
        assert first_record.open == 50900.0  # 最新のデータ（i=9）

    def test_get_latest_ohlcv_data(self, ohlcv_repository, sample_ohlcv_data):
        """最新OHLCVデータ取得のテスト"""
        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        latest_data = ohlcv_repository.get_latest_ohlcv_data("BTC/USDT", "1h", limit=5)
        assert len(latest_data) == 5

        # 降順で取得されることを確認
        assert latest_data[0].timestamp > latest_data[1].timestamp

    def test_get_ohlcv_dataframe(self, ohlcv_repository, sample_ohlcv_data):
        """OHLCVデータのDataFrame取得のテスト"""
        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        df = ohlcv_repository.get_ohlcv_dataframe("BTC/USDT", "1h")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"

    def test_get_ohlcv_dataframe_empty(self, ohlcv_repository):
        """空のOHLCVデータのDataFrame取得のテスト"""
        df = ohlcv_repository.get_ohlcv_dataframe("NONEXISTENT/USDT", "1h")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_get_latest_timestamp(self, ohlcv_repository, sample_ohlcv_data):
        """最新タイムスタンプ取得のテスト"""
        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        latest_timestamp = ohlcv_repository.get_latest_timestamp("BTC/USDT", "1h")

        assert latest_timestamp is not None
        # 最新のタイムスタンプ（i=0のデータ）と一致することを確認
        expected_timestamp = sample_ohlcv_data[0]["timestamp"]
        assert latest_timestamp == expected_timestamp

    def test_get_oldest_timestamp(self, ohlcv_repository, sample_ohlcv_data):
        """最古タイムスタンプ取得のテスト"""
        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        oldest_timestamp = ohlcv_repository.get_oldest_timestamp("BTC/USDT", "1h")

        assert oldest_timestamp is not None
        # 最古のタイムスタンプ（i=9のデータ）と一致することを確認
        expected_timestamp = sample_ohlcv_data[9]["timestamp"]
        assert oldest_timestamp == expected_timestamp

    def test_get_data_count(self, ohlcv_repository, sample_ohlcv_data):
        """データ件数取得のテスト"""
        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        count = ohlcv_repository.get_data_count("BTC/USDT", "1h")
        assert count == 10

    def test_get_date_range(self, ohlcv_repository, sample_ohlcv_data):
        """データ期間取得のテスト"""
        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        oldest, newest = ohlcv_repository.get_date_range("BTC/USDT", "1h")

        assert oldest is not None
        assert newest is not None
        assert oldest < newest

    def test_clear_ohlcv_data_by_symbol(self, ohlcv_repository, sample_ohlcv_data):
        """シンボル指定でのOHLCVデータ削除のテスト"""
        # 複数のシンボルのデータを挿入（deep copyを使用）
        eth_data = copy.deepcopy(sample_ohlcv_data)
        for record in eth_data:
            record["symbol"] = "ETH/USDT"

        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)
        ohlcv_repository.insert_ohlcv_data(eth_data)

        # BTCデータのみ削除
        deleted_count = ohlcv_repository.clear_ohlcv_data_by_symbol("BTC/USDT")
        assert deleted_count == 10

        # ETHデータは残っていることを確認
        remaining_data = ohlcv_repository.get_ohlcv_data("ETH/USDT", "1h")
        assert len(remaining_data) == 10

    def test_clear_ohlcv_data_by_timeframe(self, ohlcv_repository, sample_ohlcv_data):
        """時間軸指定でのOHLCVデータ削除のテスト"""
        # 複数の時間軸のデータを挿入（deep copyを使用）
        data_4h = copy.deepcopy(sample_ohlcv_data)
        for record in data_4h:
            record["timeframe"] = "4h"

        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)
        ohlcv_repository.insert_ohlcv_data(data_4h)

        # 1hデータのみ削除
        deleted_count = ohlcv_repository.clear_ohlcv_data_by_timeframe("1h")
        assert deleted_count == 10

        # 4hデータは残っていることを確認
        remaining_data = ohlcv_repository.get_ohlcv_data("BTC/USDT", "4h")
        assert len(remaining_data) == 10

    def test_clear_ohlcv_data_by_date_range(self, ohlcv_repository, sample_ohlcv_data):
        """期間指定でのOHLCVデータ削除のテスト"""
        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        # 最新5時間分のデータを削除
        base_time = datetime.now().replace(microsecond=0)
        start_time = base_time - timedelta(hours=4)

        deleted_count = ohlcv_repository.clear_ohlcv_data_by_date_range(
            "BTC/USDT", "1h", start_time=start_time
        )
        assert deleted_count == 5

        # 残りのデータを確認
        remaining_data = ohlcv_repository.get_ohlcv_data("BTC/USDT", "1h")
        assert len(remaining_data) == 5

    def test_get_available_symbols(self, ohlcv_repository, sample_ohlcv_data):
        """利用可能シンボル取得のテスト"""
        # 複数のシンボルのデータを挿入（deep copyを使用）
        eth_data = copy.deepcopy(sample_ohlcv_data)
        for record in eth_data:
            record["symbol"] = "ETH/USDT"

        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)
        ohlcv_repository.insert_ohlcv_data(eth_data)

        symbols = ohlcv_repository.get_available_symbols()
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols
        assert len(symbols) == 2

    def test_get_available_timeframes(self, ohlcv_repository, sample_ohlcv_data):
        """利用可能時間軸取得のテスト"""
        # 複数の時間軸のデータを挿入（deep copyを使用）
        data_4h = copy.deepcopy(sample_ohlcv_data)
        for record in data_4h:
            record["timeframe"] = "4h"

        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)
        ohlcv_repository.insert_ohlcv_data(data_4h)

        timeframes = ohlcv_repository.get_available_timeframes("BTC/USDT")
        assert "1h" in timeframes
        assert "4h" in timeframes
        assert len(timeframes) == 2

    def test_duplicate_data_handling(self, ohlcv_repository, sample_ohlcv_data):
        """重複データ処理のテスト"""
        # 同じデータを2回挿入
        first_insert = ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)
        second_insert = ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        assert first_insert == 10
        assert second_insert == 0  # 重複データは挿入されない

        # データ件数を確認
        count = ohlcv_repository.get_data_count("BTC/USDT", "1h")
        assert count == 10

    def test_data_filtering_by_time_range(self, ohlcv_repository, sample_ohlcv_data):
        """時間範囲でのデータフィルタリングのテスト"""
        ohlcv_repository.insert_ohlcv_data(sample_ohlcv_data)

        base_time = datetime.now().replace(microsecond=0)
        start_time = base_time - timedelta(hours=5)
        end_time = base_time - timedelta(hours=2)

        filtered_data = ohlcv_repository.get_ohlcv_data(
            "BTC/USDT", "1h", start_time=start_time, end_time=end_time
        )

        # 指定した範囲内のデータのみ取得されることを確認
        assert len(filtered_data) == 4  # hours 5, 4, 3, 2

        for record in filtered_data:
            assert start_time <= record.timestamp <= end_time
