"""
市場データサービスのテスト

CCXT ライブラリを使用したBybit取引所からのOHLCVデータ取得機能をテストします。
TDD（テスト駆動開発）に従い、実際のAPI呼び出しを含む統合テストを実装します。

@author Trdinger Development Team
@version 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import List
import ccxt

# テスト対象のモジュール
try:
    from app.core.services.market_data_service import BybitMarketDataService
    from app.config.market_config import MarketDataConfig
except ImportError:
    # テスト実行時にモジュールが存在しない場合のダミー
    BybitMarketDataService = None
    MarketDataConfig = None


class TestBybitMarketDataService:
    """Bybit市場データサービスのテストクラス"""

    @pytest.fixture
    def service(self):
        """テスト用のサービスインスタンスを作成"""
        if BybitMarketDataService is None:
            pytest.skip("BybitMarketDataService が実装されていません")
        return BybitMarketDataService()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access and may fail due to rate limits")
    async def test_fetch_ohlcv_data_success(self, service):
        """
        正常なOHLCVデータ取得のテスト

        実際のBybit APIを呼び出してBTC/USD:BTCのデータを取得し、
        データ形式と内容を検証します。
        """
        # テスト実行
        symbol = 'BTC/USD:BTC'
        timeframe = '1h'
        limit = 10

        result = await service.fetch_ohlcv_data(symbol, timeframe, limit)

        # 基本的な検証
        assert result is not None
        assert isinstance(result, list)
        assert len(result) <= limit
        assert len(result) > 0

        # データ形式の検証
        for candle in result:
            assert isinstance(candle, list)
            assert len(candle) == 6  # [timestamp, open, high, low, close, volume]

            timestamp, open_price, high, low, close, volume = candle

            # タイムスタンプの検証
            assert isinstance(timestamp, (int, float))
            assert timestamp > 0

            # 価格データの検証
            assert isinstance(open_price, (int, float))
            assert isinstance(high, (int, float))
            assert isinstance(low, (int, float))
            assert isinstance(close, (int, float))
            assert isinstance(volume, (int, float))

            # 価格関係の検証
            assert high >= max(open_price, close)
            assert low <= min(open_price, close)
            assert high >= low
            assert open_price > 0
            assert close > 0
            assert volume >= 0

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_data_invalid_symbol(self, service):
        """
        無効なシンボルでのエラーハンドリングテスト
        """
        with pytest.raises((ValueError, ccxt.BadSymbol)):
            await service.fetch_ohlcv_data('INVALID/SYMBOL', '1h', 10)

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_data_invalid_timeframe(self, service):
        """
        無効な時間軸でのエラーハンドリングテスト
        """
        with pytest.raises(ValueError):
            await service.fetch_ohlcv_data('BTC/USD:BTC', 'invalid', 10)

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_data_invalid_limit(self, service):
        """
        無効な制限値でのエラーハンドリングテスト
        """
        # 制限値が小さすぎる場合
        with pytest.raises(ValueError):
            await service.fetch_ohlcv_data('BTC/USD:BTC', '1h', 0)

        # 制限値が大きすぎる場合
        with pytest.raises(ValueError):
            await service.fetch_ohlcv_data('BTC/USD:BTC', '1h', 2000)

    @pytest.mark.asyncio
    async def test_symbol_normalization(self, service):
        """
        シンボル正規化のテスト
        """
        # 様々な形式のシンボルが正規化されることを確認
        test_cases = [
            ('BTCUSD', 'BTC/USD:BTC'),
            ('BTC/USD', 'BTC/USD:BTC'),
            ('btc/usd', 'BTC/USD:BTC'),
            ('BTC-USD', 'BTC/USD:BTC'),
        ]

        for input_symbol, expected in test_cases:
            normalized = service.normalize_symbol(input_symbol)
            assert normalized == expected

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access and may fail due to rate limits")
    async def test_data_freshness(self, service):
        """
        取得データの新しさをテスト

        最新のデータが取得されていることを確認します。
        """
        result = await service.fetch_ohlcv_data('BTC/USD:BTC', '1h', 5)

        # 最新のローソク足のタイムスタンプを確認
        latest_candle = result[-1]
        latest_timestamp = latest_candle[0]

        # 現在時刻との差が妥当な範囲内かチェック（1時間足なので2時間以内）
        current_time = datetime.now(timezone.utc).timestamp() * 1000
        time_diff = current_time - latest_timestamp

        assert time_diff < 2 * 60 * 60 * 1000  # 2時間以内

    def test_service_initialization(self, service):
        """
        サービスの初期化テスト
        """
        assert service is not None
        assert hasattr(service, 'exchange')
        assert service.exchange is not None
        assert isinstance(service.exchange, ccxt.bybit)

    @pytest.mark.asyncio
    async def test_network_error_handling(self, service):
        """
        ネットワークエラーのハンドリングテスト

        注意: このテストは実際のネットワーク状況に依存するため、
        モックを使用することも検討してください。
        """
        # 無効なURLを設定してネットワークエラーを発生させる
        original_urls = service.exchange.urls
        service.exchange.urls = {'api': {'public': 'https://invalid-url.example.com'}}

        try:
            with pytest.raises(ccxt.NetworkError):
                await service.fetch_ohlcv_data('BTC/USD:BTC', '1h', 10)
        finally:
            # 元のURLを復元
            service.exchange.urls = original_urls


class TestMarketDataConfig:
    """設定クラスのテストクラス"""

    def test_symbol_normalization(self):
        """シンボル正規化のテスト"""
        from app.config.market_config import MarketDataConfig

        # 正常なケース
        assert MarketDataConfig.normalize_symbol('BTCUSD') == 'BTC/USD:BTC'
        assert MarketDataConfig.normalize_symbol('BTC/USD') == 'BTC/USD:BTC'
        assert MarketDataConfig.normalize_symbol('btc/usd') == 'BTC/USD:BTC'

        # 無効なシンボル
        with pytest.raises(ValueError):
            MarketDataConfig.normalize_symbol('INVALID')

    def test_timeframe_validation(self):
        """時間軸バリデーションのテスト"""
        from app.config.market_config import MarketDataConfig

        # 有効な時間軸
        assert MarketDataConfig.validate_timeframe('1h') is True
        assert MarketDataConfig.validate_timeframe('1d') is True

        # 無効な時間軸
        assert MarketDataConfig.validate_timeframe('invalid') is False

    def test_limit_validation(self):
        """制限値バリデーションのテスト"""
        from app.config.market_config import MarketDataConfig

        # 有効な制限値
        assert MarketDataConfig.validate_limit(100) is True
        assert MarketDataConfig.validate_limit(1) is True
        assert MarketDataConfig.validate_limit(1000) is True

        # 無効な制限値
        assert MarketDataConfig.validate_limit(0) is False
        assert MarketDataConfig.validate_limit(2000) is False
