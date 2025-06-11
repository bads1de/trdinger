"""
BTCUSDT複数時間足収集のテスト

TDD: 複数時間足（1d, 4h, 1h, 30m, 15m）でのBTC/USDT:USDTデータ収集をテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from data_collector.collector import DataCollector
from app.core.services.market_data_service import BybitMarketDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.connection import SessionLocal, Base, engine


class TestBTCUSDTMultiTimeframeCollection:
    """BTCUSDT複数時間足収集のテストクラス"""

    @pytest.fixture(scope="function")
    def db_session(self):
        """テスト用データベースセッション"""
        Base.metadata.create_all(bind=engine)
        
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
            Base.metadata.drop_all(bind=engine)

    @pytest.fixture
    def ohlcv_repo(self, db_session):
        """OHLCVリポジトリのフィクスチャ"""
        return OHLCVRepository(db_session)

    @pytest.fixture
    def mock_market_service(self):
        """モックマーケットデータサービス"""
        mock_exchange = Mock()
        service = BybitMarketDataService(mock_exchange)
        return service

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        base_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        return [
            [base_timestamp, 50000.0, 51000.0, 49000.0, 50500.0, 1000.0],
            [base_timestamp + 3600000, 50500.0, 51500.0, 49500.0, 51000.0, 1100.0],
            [base_timestamp + 7200000, 51000.0, 52000.0, 50000.0, 51500.0, 1200.0],
        ]

    def test_required_timeframes_are_supported(self):
        """要求された時間足がサポートされていることをテスト"""
        from app.config.market_config import MarketDataConfig
        
        required_timeframes = ["1d", "4h", "1h", "30m", "15m"]
        
        for timeframe in required_timeframes:
            assert timeframe in MarketDataConfig.SUPPORTED_TIMEFRAMES, \
                f"必要な時間足 '{timeframe}' がサポートされていません"

    @pytest.mark.asyncio
    async def test_collect_data_for_all_required_timeframes(self, mock_market_service, ohlcv_repo, sample_ohlcv_data):
        """全ての要求された時間足でデータ収集が可能であることをテスト"""
        required_timeframes = ["1d", "4h", "1h", "30m", "15m"]
        symbol = "BTC/USDT:USDT"
        
        # モックの設定
        with patch.object(mock_market_service, 'fetch_ohlcv_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            # DataCollectorを作成
            collector = DataCollector(mock_market_service)
            
            # 各時間足でデータ収集をテスト
            for timeframe in required_timeframes:
                collected_count = await collector.collect_latest_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    ohlcv_repo=ohlcv_repo
                )
                
                assert collected_count > 0, \
                    f"時間足 '{timeframe}' でデータが収集されませんでした"
                
                # データベースにデータが保存されていることを確認
                data_count = ohlcv_repo.get_data_count(symbol, timeframe)
                assert data_count > 0, \
                    f"時間足 '{timeframe}' のデータがデータベースに保存されていません"

    @pytest.mark.asyncio
    async def test_collect_historical_data_for_multiple_timeframes(self, mock_market_service, ohlcv_repo, sample_ohlcv_data):
        """複数時間足での履歴データ収集をテスト"""
        timeframes = ["1d", "4h", "1h"]  # テスト時間短縮のため一部のみ
        symbol = "BTC/USDT:USDT"
        
        # モックの設定
        with patch.object(mock_market_service, 'fetch_ohlcv_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            collector = DataCollector(mock_market_service)
            
            # 各時間足で履歴データ収集をテスト
            start_time = datetime.now(timezone.utc) - timedelta(days=7)
            end_time = datetime.now(timezone.utc)
            
            for timeframe in timeframes:
                collected_count = await collector.collect_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    ohlcv_repo=ohlcv_repo
                )
                
                assert collected_count > 0, \
                    f"時間足 '{timeframe}' で履歴データが収集されませんでした"

    def test_timeframe_validation_accepts_required_timeframes(self, mock_market_service):
        """時間足バリデーションが要求された時間足を受け入れることをテスト"""
        required_timeframes = ["1d", "4h", "1h", "30m", "15m"]
        
        for timeframe in required_timeframes:
            # _validate_parametersメソッドが存在する場合のテスト
            if hasattr(mock_market_service, '_validate_parameters'):
                try:
                    mock_market_service._validate_parameters("BTC/USDT:USDT", timeframe, 100)
                    # エラーが発生しなければ成功
                except ValueError:
                    pytest.fail(f"時間足 '{timeframe}' のバリデーションが失敗しました")

    @pytest.mark.asyncio
    async def test_data_consistency_across_timeframes(self, mock_market_service, ohlcv_repo, sample_ohlcv_data):
        """時間足間でのデータ一貫性をテスト"""
        symbol = "BTC/USDT:USDT"
        timeframes = ["1h", "4h"]  # 関連する時間足でテスト
        
        with patch.object(mock_market_service, 'fetch_ohlcv_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            collector = DataCollector(mock_market_service)
            
            # 各時間足でデータを収集
            for timeframe in timeframes:
                await collector.collect_latest_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    ohlcv_repo=ohlcv_repo
                )
            
            # データの一貫性チェック
            for timeframe in timeframes:
                data = ohlcv_repo.get_ohlcv_data(symbol, timeframe, limit=1)
                assert len(data) > 0, f"時間足 '{timeframe}' のデータが存在しません"
                
                # データの基本的な妥当性チェック
                record = data[0]
                assert record.symbol == symbol
                assert record.timeframe == timeframe
                assert record.open > 0
                assert record.high >= record.open
                assert record.low <= record.open
                assert record.close > 0
                assert record.volume >= 0

    def test_collector_supports_btcusdt_perpetual_only(self):
        """DataCollectorがBTC/USDT:USDTのみをサポートすることをテスト"""
        # このテストは設定変更後に通るようになる
        from app.config.market_config import MarketDataConfig
        
        # BTC/USDT:USDTがサポートされていることを確認
        assert "BTC/USDT:USDT" in MarketDataConfig.SUPPORTED_SYMBOLS, \
            "BTC/USDT:USDTがサポートされていません"
        
        # 他のシンボルがサポートされていないことを確認
        unsupported_symbols = ["BTC/USDT", "BTCUSD", "ETH/USDT:USDT"]
        for symbol in unsupported_symbols:
            assert symbol not in MarketDataConfig.SUPPORTED_SYMBOLS, \
                f"サポートされるべきでないシンボル '{symbol}' がサポートされています"

    @pytest.mark.asyncio
    async def test_concurrent_collection_for_multiple_timeframes(self, mock_market_service, ohlcv_repo, sample_ohlcv_data):
        """複数時間足での同時データ収集をテスト"""
        symbol = "BTC/USDT:USDT"
        timeframes = ["1d", "4h", "1h", "30m", "15m"]
        
        with patch.object(mock_market_service, 'fetch_ohlcv_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            collector = DataCollector(mock_market_service)
            
            # 同時実行でデータ収集
            tasks = []
            for timeframe in timeframes:
                task = collector.collect_latest_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    ohlcv_repo=ohlcv_repo
                )
                tasks.append(task)
            
            # 全てのタスクを同時実行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 全てのタスクが成功することを確認
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"時間足 '{timeframes[i]}' の収集でエラーが発生: {result}")
                assert result > 0, f"時間足 '{timeframes[i]}' でデータが収集されませんでした"

    def test_timeframe_specific_data_storage(self, ohlcv_repo, db_session):
        """時間足別のデータ保存をテスト"""
        from database.models import OHLCVData
        
        symbol = "BTC/USDT:USDT"
        timeframes = ["1d", "4h", "1h", "30m", "15m"]
        
        # 各時間足でテストデータを作成
        for timeframe in timeframes:
            test_data = OHLCVData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(timezone.utc),
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0
            )
            db_session.add(test_data)
        
        db_session.commit()
        
        # 各時間足のデータが正しく保存されていることを確認
        for timeframe in timeframes:
            count = ohlcv_repo.get_data_count(symbol, timeframe)
            assert count == 1, f"時間足 '{timeframe}' のデータが正しく保存されていません"
