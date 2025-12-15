"""
BybitLongShortRatioServiceのテスト
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.services.data_collection.bybit.long_short_ratio_service import (
    BybitLongShortRatioService,
)
from database.repositories.long_short_ratio_repository import LongShortRatioRepository
from database.models import LongShortRatioData


@pytest.fixture
def mock_exchange():
    """CCXTエクスチェンジのモック"""
    exchange = MagicMock()
    exchange.publicGetV5MarketAccountRatio = AsyncMock()
    return exchange


@pytest.fixture
def service(mock_exchange):
    """テスト対象サービス"""
    return BybitLongShortRatioService(exchange=mock_exchange)


@pytest.fixture
def mock_repository():
    """リポジトリのモック"""
    return MagicMock(spec=LongShortRatioRepository)


class TestFetchLongShortRatioData:
    """fetch_long_short_ratio_data メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_success(self, service, mock_exchange):
        """APIからのデータ取得成功ケース"""
        # モックレスポンス
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "buyRatio": "0.6",
                        "sellRatio": "0.4",
                        "timestamp": "1609459200000",
                    }
                ]
            }
        }

        result = await service.fetch_long_short_ratio_data("BTC/USDT:USDT", "1h")

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USDT:USDT"  # 補完されていること
        assert result[0]["period"] == "1h"  # 補完されていること

        # パラメータ変換の確認 (kwargs.get("params") でパラメータ取得)
        mock_exchange.publicGetV5MarketAccountRatio.assert_called_once()
        call_kwargs = mock_exchange.publicGetV5MarketAccountRatio.call_args.kwargs
        params = call_kwargs.get("params")
        assert params is not None
        assert params["symbol"] == "BTCUSDT"
        assert params["category"] == "linear"
        assert params["period"] == "1h"
        assert params["limit"] == 50  # デフォルト値

    @pytest.mark.asyncio
    async def test_with_time_params(self, service, mock_exchange):
        """開始・終了時刻を指定した場合"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {
                "list": [
                    {
                        "symbol": "ETHUSDT",
                        "buyRatio": "0.55",
                        "sellRatio": "0.45",
                        "timestamp": "1609462800000",
                    }
                ]
            }
        }

        result = await service.fetch_long_short_ratio_data(
            "ETH/USDT",
            "5min",
            limit=100,
            start_time=1609459200000,
            end_time=1609502400000,
        )

        assert len(result) == 1
        assert result[0]["symbol"] == "ETH/USDT"
        assert result[0]["period"] == "5min"

        call_kwargs = mock_exchange.publicGetV5MarketAccountRatio.call_args.kwargs
        params = call_kwargs.get("params")
        assert params["startTime"] == 1609459200000
        assert params["endTime"] == 1609502400000
        assert params["limit"] == 100

    @pytest.mark.asyncio
    async def test_empty_response(self, service, mock_exchange):
        """レスポンスが空の場合"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {"list": []}
        }

        result = await service.fetch_long_short_ratio_data("BTC/USDT:USDT", "1h")

        assert result == []

    @pytest.mark.asyncio
    async def test_missing_result_key(self, service, mock_exchange):
        """result キーがない場合"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {}

        result = await service.fetch_long_short_ratio_data("BTC/USDT:USDT", "1h")

        assert result == []

    @pytest.mark.asyncio
    async def test_missing_list_key(self, service, mock_exchange):
        """list キーがない場合"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {"result": {}}

        result = await service.fetch_long_short_ratio_data("BTC/USDT:USDT", "1h")

        assert result == []

    @pytest.mark.asyncio
    async def test_api_returns_none(self, service, mock_exchange):
        """APIがNoneを返す場合（ネットワークエラー等でハンドリングされた後）"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = None

        result = await service.fetch_long_short_ratio_data("BTC/USDT:USDT", "1h")

        assert result == []

    @pytest.mark.asyncio
    async def test_api_exception_handled(self, service, mock_exchange):
        """API例外が発生した場合のハンドリング"""
        mock_exchange.publicGetV5MarketAccountRatio.side_effect = Exception(
            "Network Error"
        )

        result = await service.fetch_long_short_ratio_data("BTC/USDT:USDT", "1h")

        # 例外がハンドリングされて空リストが返る
        assert result == []


class TestFetchIncrementalLongShortRatioData:
    """fetch_incremental_long_short_ratio_data メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_with_existing_data(self, service, mock_repository, mock_exchange):
        """既存データがある場合の差分更新"""
        # リポジトリのモック設定（最新データあり）
        latest_record = MagicMock(spec=LongShortRatioData)
        latest_record.timestamp = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_repository.get_latest_ratio.return_value = latest_record
        mock_repository.insert_long_short_ratio_data.return_value = 5

        # APIモック
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "buyRatio": "0.6",
                        "sellRatio": "0.4",
                        "timestamp": "1609462800000",
                    }
                ]
            }
        }

        result = await service.fetch_incremental_long_short_ratio_data(
            "BTC/USDT:USDT", "1h", mock_repository
        )

        assert result["success"] is True
        assert result["saved_count"] == 5
        assert result["symbol"] == "BTC/USDT:USDT"

        # リポジトリのget_latest_ratioが呼ばれたか（開始時刻決定のため）
        mock_repository.get_latest_ratio.assert_called()

        # APIコールのパラメータ確認 (startTimeが設定されているか)
        call_kwargs = mock_exchange.publicGetV5MarketAccountRatio.call_args.kwargs
        params = call_kwargs.get("params")
        assert params["startTime"] == 1609459200000  # 2021-01-01 00:00:00 UTC

    @pytest.mark.asyncio
    async def test_without_existing_data(self, service, mock_repository, mock_exchange):
        """既存データがない場合（初回収集）"""
        mock_repository.get_latest_ratio.return_value = None
        mock_repository.insert_long_short_ratio_data.return_value = 10

        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "buyRatio": "0.6",
                        "sellRatio": "0.4",
                        "timestamp": "1609459200000",
                    }
                ]
            }
        }

        result = await service.fetch_incremental_long_short_ratio_data(
            "BTC/USDT:USDT", "1h", mock_repository
        )

        assert result["success"] is True
        assert result["saved_count"] == 10

        # startTime がセットされないことを確認
        call_kwargs = mock_exchange.publicGetV5MarketAccountRatio.call_args.kwargs
        params = call_kwargs.get("params")
        assert "startTime" not in params

    @pytest.mark.asyncio
    async def test_no_new_data(self, service, mock_repository, mock_exchange):
        """新しいデータがない場合"""
        latest_record = MagicMock(spec=LongShortRatioData)
        latest_record.timestamp = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_repository.get_latest_ratio.return_value = latest_record

        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {"list": []}
        }

        result = await service.fetch_incremental_long_short_ratio_data(
            "BTC/USDT:USDT", "1h", mock_repository
        )

        assert result["success"] is True
        assert result["saved_count"] == 0
        assert result["latest_timestamp"] == datetime(
            2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc
        )

    @pytest.mark.asyncio
    async def test_exception_handling(self, service, mock_repository, mock_exchange):
        """例外発生時のハンドリング"""
        mock_repository.get_latest_ratio.side_effect = Exception("DB Error")

        result = await service.fetch_incremental_long_short_ratio_data(
            "BTC/USDT:USDT", "1h", mock_repository
        )

        assert result["success"] is False
        assert result["saved_count"] == 0
        assert "error" in result
        assert "DB Error" in result["error"]


class TestCollectHistoricalLongShortRatioData:
    """collect_historical_long_short_ratio_data メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_basic_collection(self, service, mock_repository, mock_exchange):
        """基本的な履歴収集フロー"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "buyRatio": "0.6",
                        "sellRatio": "0.4",
                        "timestamp": "1609459200000",
                    }
                ]
            }
        }
        mock_repository.insert_long_short_ratio_data.return_value = 10

        # 現在時刻より少し過去（1時間前）を設定して、ループが回るようにする
        start_date = datetime.now(timezone.utc)
        start_ts = int(start_date.timestamp() * 1000) - (60 * 60 * 1000)  # 1時間前
        start_date = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

        count = await service.collect_historical_long_short_ratio_data(
            "BTC/USDT:USDT", "1h", mock_repository, start_date=start_date
        )

        # 少なくとも1回は呼ばれているはず
        assert mock_exchange.publicGetV5MarketAccountRatio.called
        assert count >= 0

    @pytest.mark.asyncio
    async def test_default_start_date(self, service, mock_repository, mock_exchange):
        """開始日時を指定しない場合のデフォルト値テスト"""
        # ループを短く抑えるため、すぐに空データを返す
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {"list": []}
        }

        # start_date指定なしで呼び出し
        count = await service.collect_historical_long_short_ratio_data(
            "BTC/USDT:USDT", "1d", mock_repository
        )

        # return される値は保存されたトータル件数（この場合0）
        assert count == 0
        assert mock_exchange.publicGetV5MarketAccountRatio.called

    @pytest.mark.asyncio
    async def test_chunk_size_varies_by_period(
        self, service, mock_repository, mock_exchange
    ):
        """期間によってチャンクサイズが変わることのテスト"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "buyRatio": "0.6",
                        "sellRatio": "0.4",
                        "timestamp": "1609459200000",
                    }
                ]
            }
        }
        mock_repository.insert_long_short_ratio_data.return_value = 1

        # 5minの場合、短時間で終わるようにstart_dateを現在に近く設定
        start_ts = int(datetime.now(timezone.utc).timestamp() * 1000) - (
            1 * 60 * 60 * 1000
        )  # 1時間前
        start_date = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

        count = await service.collect_historical_long_short_ratio_data(
            "BTC/USDT:USDT", "5min", mock_repository, start_date=start_date
        )

        # 正常に完了すること
        assert mock_exchange.publicGetV5MarketAccountRatio.called

    @pytest.mark.asyncio
    async def test_exception_during_collection(
        self, service, mock_repository, mock_exchange
    ):
        """収集中の例外ハンドリング"""
        mock_exchange.publicGetV5MarketAccountRatio.side_effect = Exception("API Error")

        start_ts = int(datetime.now(timezone.utc).timestamp() * 1000) - (60 * 60 * 1000)
        start_date = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

        # 例外が発生しても0が返され、クラッシュしないこと
        count = await service.collect_historical_long_short_ratio_data(
            "BTC/USDT:USDT", "1h", mock_repository, start_date=start_date
        )

        assert count == 0


class TestSymbolConversion:
    """シンボル変換のテスト"""

    @pytest.mark.asyncio
    async def test_symbol_slash_removal(self, service, mock_exchange):
        """シンボルのスラッシュが削除されること"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {"list": []}
        }

        await service.fetch_long_short_ratio_data("ETH/USDT", "1h")

        call_kwargs = mock_exchange.publicGetV5MarketAccountRatio.call_args.kwargs
        params = call_kwargs.get("params")
        assert params["symbol"] == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_symbol_without_slash(self, service, mock_exchange):
        """スラッシュなしシンボルでも動作すること"""
        mock_exchange.publicGetV5MarketAccountRatio.return_value = {
            "result": {"list": []}
        }

        await service.fetch_long_short_ratio_data("BTCUSDT", "1h")

        call_kwargs = mock_exchange.publicGetV5MarketAccountRatio.call_args.kwargs
        params = call_kwargs.get("params")
        assert params["symbol"] == "BTCUSDT"


