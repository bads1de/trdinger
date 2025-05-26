"""
OHLCVデータ保存機能の結合テスト

実際のBybit APIを呼び出してOHLCVデータを取得し、
データベースに保存する機能の結合テストです。

@author Trdinger Development Team
@version 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, Mock

from app.core.services.market_data_service import BybitMarketDataService
from database.repository import OHLCVRepository
from database.connection import get_db


class TestSaveOHLCVIntegration:
    """OHLCVデータ保存の結合テスト"""

    @pytest.fixture
    def service(self):
        """テスト用のMarketDataServiceインスタンス"""
        return BybitMarketDataService()

    @pytest.mark.asyncio
    async def test_save_ohlcv_data_flow_with_mock(self, service):
        """
        モックを使用したOHLCVデータ保存フローのテスト
        実際のAPI呼び出しなしでデータフローを検証
        """
        # モックOHLCVデータ（実際のBybit APIレスポンス形式）
        mock_ohlcv_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
            [1640998800000, 47200.0, 47600.0, 47000.0, 47400.0, 1200.0],
            [1641002400000, 47400.0, 47800.0, 47300.0, 47600.0, 900.0],
        ]

        # データベース関連のモック
        with patch('app.core.services.market_data_service.get_db') as mock_get_db, \
             patch('app.core.services.market_data_service.OHLCVRepository') as mock_repo_class:
            
            # モック設定
            mock_db = Mock()
            mock_get_db.return_value = iter([mock_db])
            mock_repo = Mock()
            mock_repo.insert_ohlcv_data.return_value = 3
            mock_repo_class.return_value = mock_repo

            # fetch_ohlcv_dataメソッドをモック
            with patch.object(service, 'fetch_ohlcv_data', return_value=mock_ohlcv_data):
                # データ保存実行
                result = await service.save_ohlcv_to_database(
                    mock_ohlcv_data, "BTC/USD:BTC", "1h"
                )

                # 結果検証
                assert result == 3
                mock_repo.insert_ohlcv_data.assert_called_once()

                # 変換されたデータの検証
                call_args = mock_repo.insert_ohlcv_data.call_args[0][0]
                assert len(call_args) == 3

                # 最初のレコードの詳細検証
                first_record = call_args[0]
                assert first_record['symbol'] == "BTC/USD:BTC"
                assert first_record['timeframe'] == "1h"
                assert isinstance(first_record['timestamp'], datetime)
                assert first_record['open'] == 47000.0
                assert first_record['high'] == 47500.0
                assert first_record['low'] == 46800.0
                assert first_record['close'] == 47200.0
                assert first_record['volume'] == 1000.0

    @pytest.mark.asyncio
    async def test_data_conversion_accuracy(self, service):
        """データ変換の精度テスト"""
        # テスト用のOHLCVデータ
        test_ohlcv_data = [
            [1640995200000, 47123.45, 47567.89, 46789.12, 47234.56, 1234.567],
        ]

        with patch('app.core.services.market_data_service.get_db') as mock_get_db, \
             patch('app.core.services.market_data_service.OHLCVRepository') as mock_repo_class:
            
            mock_db = Mock()
            mock_get_db.return_value = iter([mock_db])
            mock_repo = Mock()
            mock_repo.insert_ohlcv_data.return_value = 1
            mock_repo_class.return_value = mock_repo

            # データ保存実行
            await service.save_ohlcv_to_database(
                test_ohlcv_data, "BTC/USD:BTC", "1h"
            )

            # 変換されたデータの精度検証
            call_args = mock_repo.insert_ohlcv_data.call_args[0][0]
            record = call_args[0]

            # 小数点以下の精度が保持されていることを確認
            assert record['open'] == 47123.45
            assert record['high'] == 47567.89
            assert record['low'] == 46789.12
            assert record['close'] == 47234.56
            assert record['volume'] == 1234.567

            # タイムスタンプの変換確認
            expected_timestamp = datetime.fromtimestamp(
                1640995200000 / 1000, tz=timezone.utc
            )
            assert record['timestamp'] == expected_timestamp

    @pytest.mark.asyncio
    async def test_duplicate_data_handling(self, service):
        """重複データ処理のテスト"""
        mock_ohlcv_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
        ]

        with patch('app.core.services.market_data_service.get_db') as mock_get_db, \
             patch('app.core.services.market_data_service.OHLCVRepository') as mock_repo_class:
            
            mock_db = Mock()
            mock_get_db.return_value = iter([mock_db])
            mock_repo = Mock()
            # 重複により0件挿入をシミュレート
            mock_repo.insert_ohlcv_data.return_value = 0
            mock_repo_class.return_value = mock_repo

            # データ保存実行
            result = await service.save_ohlcv_to_database(
                mock_ohlcv_data, "BTC/USD:BTC", "1h"
            )

            # 重複データの場合は0が返されることを確認
            assert result == 0

    @pytest.mark.asyncio
    async def test_empty_data_handling(self, service):
        """空データの処理テスト"""
        empty_ohlcv_data = []

        with patch('app.core.services.market_data_service.get_db') as mock_get_db, \
             patch('app.core.services.market_data_service.OHLCVRepository') as mock_repo_class:
            
            mock_db = Mock()
            mock_get_db.return_value = iter([mock_db])
            mock_repo = Mock()
            mock_repo.insert_ohlcv_data.return_value = 0
            mock_repo_class.return_value = mock_repo

            # 空データでの保存実行
            result = await service.save_ohlcv_to_database(
                empty_ohlcv_data, "BTC/USD:BTC", "1h"
            )

            # 空データの場合は0が返されることを確認
            assert result == 0
            mock_repo.insert_ohlcv_data.assert_called_once_with([])

    @pytest.mark.asyncio
    async def test_database_error_handling(self, service):
        """データベースエラーハンドリングのテスト"""
        mock_ohlcv_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
        ]

        with patch('app.core.services.market_data_service.get_db') as mock_get_db, \
             patch('app.core.services.market_data_service.OHLCVRepository') as mock_repo_class:
            
            mock_db = Mock()
            mock_get_db.return_value = iter([mock_db])
            mock_repo = Mock()
            # データベースエラーをシミュレート
            mock_repo.insert_ohlcv_data.side_effect = Exception("Database connection error")
            mock_repo_class.return_value = mock_repo

            # エラーが適切に伝播されることを確認
            with pytest.raises(Exception) as exc_info:
                await service.save_ohlcv_to_database(
                    mock_ohlcv_data, "BTC/USD:BTC", "1h"
                )
            
            assert "Database connection error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, service):
        """大量データセットの処理テスト"""
        # 1000件のモックデータを生成
        large_ohlcv_data = []
        base_timestamp = 1640995200000
        
        for i in range(1000):
            timestamp = base_timestamp + (i * 3600000)  # 1時間ずつ増加
            price = 47000.0 + (i * 10)  # 価格を少しずつ変化
            large_ohlcv_data.append([
                timestamp,
                price,
                price + 100,
                price - 50,
                price + 50,
                1000.0 + i
            ])

        with patch('app.core.services.market_data_service.get_db') as mock_get_db, \
             patch('app.core.services.market_data_service.OHLCVRepository') as mock_repo_class:
            
            mock_db = Mock()
            mock_get_db.return_value = iter([mock_db])
            mock_repo = Mock()
            mock_repo.insert_ohlcv_data.return_value = 1000
            mock_repo_class.return_value = mock_repo

            # 大量データの保存実行
            result = await service.save_ohlcv_to_database(
                large_ohlcv_data, "BTC/USD:BTC", "1h"
            )

            # 結果検証
            assert result == 1000
            
            # 変換されたデータの件数確認
            call_args = mock_repo.insert_ohlcv_data.call_args[0][0]
            assert len(call_args) == 1000

    @pytest.mark.asyncio
    async def test_multiple_symbols_and_timeframes(self, service):
        """複数銘柄・時間軸の処理テスト"""
        test_cases = [
            ("BTC/USD:BTC", "1h"),
            ("ETH/USD:ETH", "1d"),
            ("BTC/USD:BTC", "4h"),
        ]

        mock_ohlcv_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
        ]

        for symbol, timeframe in test_cases:
            with patch('app.core.services.market_data_service.get_db') as mock_get_db, \
                 patch('app.core.services.market_data_service.OHLCVRepository') as mock_repo_class:
                
                mock_db = Mock()
                mock_get_db.return_value = iter([mock_db])
                mock_repo = Mock()
                mock_repo.insert_ohlcv_data.return_value = 1
                mock_repo_class.return_value = mock_repo

                # データ保存実行
                result = await service.save_ohlcv_to_database(
                    mock_ohlcv_data, symbol, timeframe
                )

                # 結果検証
                assert result == 1
                
                # 正しいシンボルと時間軸が設定されていることを確認
                call_args = mock_repo.insert_ohlcv_data.call_args[0][0]
                record = call_args[0]
                assert record['symbol'] == symbol
                assert record['timeframe'] == timeframe
