"""
リポジトリ拡張メソッドのテスト

OHLCVRepositoryとFundingRateRepositoryの拡張メソッド（get_all_by_symbol）をテストします。
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from database.models import FundingRateData, OHLCVData
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository


@pytest.fixture
def mock_db_session():
    """データベースセッションのモック"""
    return MagicMock()


@pytest.fixture
def ohlcv_repository(mock_db_session):
    """OHLCVRepositoryのフィクスチャ"""
    return OHLCVRepository(mock_db_session)


@pytest.fixture
def funding_rate_repository(mock_db_session):
    """FundingRateRepositoryのフィクスチャ"""
    return FundingRateRepository(mock_db_session)


@pytest.fixture
def sample_ohlcv_data() -> List[OHLCVData]:
    """サンプルOHLCVデータ"""
    data = []
    for i in range(5):
        ohlcv = OHLCVData(
            id=i + 1,
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            timestamp=datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc),
            open=50000.0 + i * 100,
            high=50100.0 + i * 100,
            low=49900.0 + i * 100,
            close=50050.0 + i * 100,
            volume=1000.0 + i * 10,
        )
        data.append(ohlcv)
    return data


@pytest.fixture
def sample_funding_rate_data() -> List[FundingRateData]:
    """サンプルファンディングレートデータ"""
    data = []
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        # 8時間間隔で日付を進める
        from datetime import timedelta

        timestamp = base_time + timedelta(hours=i * 8)
        fr = FundingRateData(
            id=i + 1,
            symbol="BTC/USDT:USDT",
            funding_rate=0.0001 * (i + 1),
            funding_timestamp=timestamp,
            timestamp=timestamp,
            mark_price=50000.0 + i * 100,
            index_price=50000.0 + i * 100,
        )
        data.append(fr)
    return data


class TestOHLCVRepositoryGetAllBySymbol:
    """OHLCVRepository.get_all_by_symbol()のテストクラス"""

    def test_ohlcv_get_all_by_symbol_success(
        self, ohlcv_repository, mock_db_session, sample_ohlcv_data
    ):
        """
        正常系: 指定シンボルの全OHLCVデータを取得

        テスト内容:
        - get_all_by_symbolメソッドが正しく動作すること
        - 返されるデータが時系列順にソートされていること
        - 正しいシンボルと時間軸でフィルタリングされていること
        """
        # get_filtered_dataメソッドをモック
        with patch.object(
            ohlcv_repository, "get_filtered_data", return_value=sample_ohlcv_data
        ) as mock_get_filtered:
            # テスト実行
            result = ohlcv_repository.get_all_by_symbol(
                symbol="BTC/USDT:USDT", timeframe="1h"
            )

            # 検証
            assert len(result) == 5
            assert result == sample_ohlcv_data

            # get_filtered_dataが正しい引数で呼ばれたか確認
            mock_get_filtered.assert_called_once_with(
                filters={"symbol": "BTC/USDT:USDT", "timeframe": "1h"},
                time_range_column="timestamp",
                start_time=None,
                end_time=None,
                order_by_column="timestamp",
                order_asc=True,
                limit=None,
            )

    def test_ohlcv_get_all_by_symbol_empty(self, ohlcv_repository):
        """
        正常系: データが存在しない場合

        テスト内容:
        - データがない場合、空のリストが返されること
        """
        with patch.object(ohlcv_repository, "get_filtered_data", return_value=[]):
            result = ohlcv_repository.get_all_by_symbol(
                symbol="ETH/USDT:USDT", timeframe="1h"
            )

            assert result == []
            assert isinstance(result, list)

    def test_ohlcv_get_all_by_symbol_with_order(
        self, ohlcv_repository, sample_ohlcv_data
    ):
        """
        正常系: ソート順の確認

        テスト内容:
        - データが時系列順（昇順）にソートされていること
        - timestampが古い順に並んでいること
        """
        with patch.object(
            ohlcv_repository, "get_filtered_data", return_value=sample_ohlcv_data
        ):
            result = ohlcv_repository.get_all_by_symbol(
                symbol="BTC/USDT:USDT", timeframe="1h"
            )

            # タイムスタンプが昇順であることを確認
            for i in range(len(result) - 1):
                assert result[i].timestamp <= result[i + 1].timestamp

    def test_ohlcv_get_all_by_symbol_different_symbols(
        self, ohlcv_repository, sample_ohlcv_data
    ):
        """
        正常系: 異なるシンボルでの取得

        テスト内容:
        - 異なるシンボルで個別にデータを取得できること
        - フィルターが正しく適用されること
        """
        btc_data = sample_ohlcv_data[:3]
        eth_data = [
            OHLCVData(
                id=10,
                symbol="ETH/USDT:USDT",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                open=3000.0,
                high=3100.0,
                low=2900.0,
                close=3050.0,
                volume=500.0,
            )
        ]

        with patch.object(
            ohlcv_repository, "get_filtered_data", side_effect=[btc_data, eth_data]
        ):
            btc_result = ohlcv_repository.get_all_by_symbol(
                symbol="BTC/USDT:USDT", timeframe="1h"
            )
            eth_result = ohlcv_repository.get_all_by_symbol(
                symbol="ETH/USDT:USDT", timeframe="1h"
            )

            assert len(btc_result) == 3
            assert len(eth_result) == 1
            assert btc_result[0].symbol == "BTC/USDT:USDT"
            assert eth_result[0].symbol == "ETH/USDT:USDT"

    def test_ohlcv_get_all_by_symbol_different_timeframes(
        self, ohlcv_repository, sample_ohlcv_data
    ):
        """
        正常系: 異なる時間軸での取得

        テスト内容:
        - 同じシンボルでも異なる時間軸でデータを取得できること
        """
        with patch.object(
            ohlcv_repository, "get_filtered_data", return_value=sample_ohlcv_data
        ) as mock_get_filtered:
            # 1h
            ohlcv_repository.get_all_by_symbol(symbol="BTC/USDT:USDT", timeframe="1h")
            assert mock_get_filtered.call_args[1]["filters"]["timeframe"] == "1h"

            # 4h
            ohlcv_repository.get_all_by_symbol(symbol="BTC/USDT:USDT", timeframe="4h")
            assert mock_get_filtered.call_args[1]["filters"]["timeframe"] == "4h"


class TestFundingRateRepositoryGetAllBySymbol:
    """FundingRateRepository.get_all_by_symbol()のテストクラス"""

    def test_funding_rate_get_all_by_symbol_success(
        self, funding_rate_repository, mock_db_session, sample_funding_rate_data
    ):
        """
        正常系: 指定シンボルの全ファンディングレートデータを取得

        テスト内容:
        - get_all_by_symbolメソッドが正しく動作すること
        - 返されるデータが時系列順にソートされていること
        - 正しいシンボルでフィルタリングされていること
        """
        with patch.object(
            funding_rate_repository,
            "get_filtered_data",
            return_value=sample_funding_rate_data,
        ) as mock_get_filtered:
            result = funding_rate_repository.get_all_by_symbol(symbol="BTC/USDT:USDT")

            # 検証
            assert len(result) == 5
            assert result == sample_funding_rate_data

            # get_filtered_dataが正しい引数で呼ばれたか確認
            mock_get_filtered.assert_called_once_with(
                filters={"symbol": "BTC/USDT:USDT"},
                time_range_column="funding_timestamp",
                start_time=None,
                end_time=None,
                order_by_column="funding_timestamp",
                order_asc=True,
                limit=None,
            )

    def test_funding_rate_get_all_by_symbol_empty(self, funding_rate_repository):
        """
        正常系: データが存在しない場合

        テスト内容:
        - データがない場合、空のリストが返されること
        """
        with patch.object(
            funding_rate_repository, "get_filtered_data", return_value=[]
        ):
            result = funding_rate_repository.get_all_by_symbol(symbol="XRP/USDT:USDT")

            assert result == []
            assert isinstance(result, list)

    def test_funding_rate_get_all_by_symbol_with_order(
        self, funding_rate_repository, sample_funding_rate_data
    ):
        """
        正常系: ソート順の確認

        テスト内容:
        - データが時系列順（昇順）にソートされていること
        - funding_timestampが古い順に並んでいること
        """
        with patch.object(
            funding_rate_repository,
            "get_filtered_data",
            return_value=sample_funding_rate_data,
        ):
            result = funding_rate_repository.get_all_by_symbol(symbol="BTC/USDT:USDT")

            # タイムスタンプが昇順であることを確認
            for i in range(len(result) - 1):
                assert result[i].funding_timestamp <= result[i + 1].funding_timestamp

    def test_funding_rate_get_all_by_symbol_different_symbols(
        self, funding_rate_repository, sample_funding_rate_data
    ):
        """
        正常系: 異なるシンボルでの取得

        テスト内容:
        - 異なるシンボルで個別にデータを取得できること
        - フィルターが正しく適用されること
        """
        btc_data = sample_funding_rate_data[:3]
        eth_data = [
            FundingRateData(
                id=10,
                symbol="ETH/USDT:USDT",
                funding_rate=0.00015,
                funding_timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                mark_price=3000.0,
                index_price=3000.0,
            )
        ]

        with patch.object(
            funding_rate_repository,
            "get_filtered_data",
            side_effect=[btc_data, eth_data],
        ):
            btc_result = funding_rate_repository.get_all_by_symbol(
                symbol="BTC/USDT:USDT"
            )
            eth_result = funding_rate_repository.get_all_by_symbol(
                symbol="ETH/USDT:USDT"
            )

            assert len(btc_result) == 3
            assert len(eth_result) == 1
            assert btc_result[0].symbol == "BTC/USDT:USDT"
            assert eth_result[0].symbol == "ETH/USDT:USDT"

    def test_funding_rate_get_all_by_symbol_rate_values(
        self, funding_rate_repository, sample_funding_rate_data
    ):
        """
        正常系: ファンディングレート値の確認

        テスト内容:
        - ファンディングレート値が正しく取得されること
        - 0.0を含む値も正しく扱われること
        """
        # 0.0を含むデータを作成
        data_with_zero = sample_funding_rate_data.copy()
        data_with_zero[0].funding_rate = 0.0

        with patch.object(
            funding_rate_repository,
            "get_filtered_data",
            return_value=data_with_zero,
        ):
            result = funding_rate_repository.get_all_by_symbol(symbol="BTC/USDT:USDT")

            # 0.0が含まれていることを確認
            assert result[0].funding_rate == 0.0
            # 他の値も正しいことを確認
            for i in range(1, len(result)):
                assert result[i].funding_rate > 0
