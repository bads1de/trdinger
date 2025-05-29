"""
ファンディングレートサービスの単体テスト

TDD実装の一環として、ファンディングレートサービスの動作を検証します。
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from app.core.services.funding_rate_service import BybitFundingRateService
from database.repositories.funding_rate_repository import FundingRateRepository


class TestBybitFundingRateService:
    """BybitFundingRateServiceの単体テスト"""

    @pytest.fixture
    def funding_rate_service(self):
        """ファンディングレートサービスのインスタンスを作成"""
        return BybitFundingRateService()

    @pytest.fixture
    def mock_repository(self):
        """モックリポジトリを作成"""
        return Mock(spec=FundingRateRepository)

    @pytest.fixture
    def sample_funding_rate_data(self):
        """サンプルファンディングレートデータ"""
        return {
            'symbol': 'BTC/USDT:USDT',
            'fundingRate': -0.00015708,
            'timestamp': 1748390156916,
            'datetime': '2025-05-27T23:55:56.916Z',
            'fundingTimestamp': 1748390400000,
            'fundingDatetime': '2025-05-28T00:00:00.000Z',
            'markPrice': 108906.5,
            'indexPrice': 108976.41,
            'info': {'symbol': 'BTCUSDT', 'fundingRate': '-0.00015708'}
        }

    @pytest.fixture
    def sample_funding_history(self):
        """サンプルファンディングレート履歴データ"""
        return [
            {
                'symbol': 'BTC/USDT:USDT',
                'fundingRate': 2.161e-05,
                'timestamp': 1748361600000,
                'datetime': '2025-05-27T16:00:00.000Z',
                'info': {'symbol': 'BTCUSDT', 'fundingRate': '0.00002161'}
            },
            {
                'symbol': 'BTC/USDT:USDT',
                'fundingRate': -0.00015708,
                'timestamp': 1748390400000,
                'datetime': '2025-05-28T00:00:00.000Z',
                'info': {'symbol': 'BTCUSDT', 'fundingRate': '-0.00015708'}
            }
        ]

    def test_normalize_symbol(self, funding_rate_service):
        """シンボル正規化のテスト"""
        # スポット形式から無期限契約形式への変換
        assert funding_rate_service.normalize_symbol('BTC/USDT') == 'BTC/USDT:USDT'
        assert funding_rate_service.normalize_symbol('ETH/USDT') == 'ETH/USDT:USDT'
        assert funding_rate_service.normalize_symbol('BTC/USD') == 'BTC/USD:USD'
        
        # 既に無期限契約形式の場合はそのまま
        assert funding_rate_service.normalize_symbol('BTC/USDT:USDT') == 'BTC/USDT:USDT'
        assert funding_rate_service.normalize_symbol('ETH/USDT:USDT') == 'ETH/USDT:USDT'
        
        # その他の形式はデフォルトでUSDT無期限契約
        assert funding_rate_service.normalize_symbol('BTCUSDT') == 'BTCUSDT:USDT'

    def test_validate_parameters(self, funding_rate_service):
        """パラメータ検証のテスト"""
        # 正常なパラメータ
        funding_rate_service._validate_parameters('BTC/USDT', 100)
        
        # 無効なシンボル
        with pytest.raises(ValueError, match="シンボルが指定されていません"):
            funding_rate_service._validate_parameters('', 100)
        
        with pytest.raises(ValueError, match="シンボルが指定されていません"):
            funding_rate_service._validate_parameters(None, 100)
        
        # 無効なlimit
        with pytest.raises(ValueError, match="limitは1-1000の範囲で指定してください"):
            funding_rate_service._validate_parameters('BTC/USDT', 0)
        
        with pytest.raises(ValueError, match="limitは1-1000の範囲で指定してください"):
            funding_rate_service._validate_parameters('BTC/USDT', 1001)

    @pytest.mark.asyncio
    async def test_fetch_current_funding_rate_success(
        self, funding_rate_service, sample_funding_rate_data
    ):
        """現在のファンディングレート取得成功のテスト"""
        with patch.object(
            funding_rate_service.exchange, 'fetch_funding_rate', 
            return_value=sample_funding_rate_data
        ) as mock_fetch:
            
            result = await funding_rate_service.fetch_current_funding_rate('BTC/USDT')
            
            # 正規化されたシンボルで呼び出されることを確認
            mock_fetch.assert_called_once_with('BTC/USDT:USDT')
            
            # 結果の検証
            assert result == sample_funding_rate_data
            assert result['symbol'] == 'BTC/USDT:USDT'
            assert result['fundingRate'] == -0.00015708

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_history_success(
        self, funding_rate_service, sample_funding_history
    ):
        """ファンディングレート履歴取得成功のテスト"""
        with patch.object(
            funding_rate_service.exchange, 'fetch_funding_rate_history',
            return_value=sample_funding_history
        ) as mock_fetch:
            
            result = await funding_rate_service.fetch_funding_rate_history('BTC/USDT', 100)
            
            # 正規化されたシンボルで呼び出されることを確認
            mock_fetch.assert_called_once_with('BTC/USDT:USDT', None, 100)
            
            # 結果の検証
            assert result == sample_funding_history
            assert len(result) == 2
            assert result[0]['fundingRate'] == 2.161e-05
            assert result[1]['fundingRate'] == -0.00015708

    @pytest.mark.asyncio
    async def test_fetch_and_save_funding_rate_data_success(
        self, funding_rate_service, mock_repository, sample_funding_history
    ):
        """ファンディングレートデータ取得・保存成功のテスト"""
        # モックの設定
        mock_repository.insert_funding_rate_data.return_value = 2
        
        with patch.object(
            funding_rate_service, 'fetch_funding_rate_history',
            return_value=sample_funding_history
        ) as mock_fetch:
            
            result = await funding_rate_service.fetch_and_save_funding_rate_data(
                'BTC/USDT', 100, mock_repository
            )
            
            # fetch_funding_rate_historyが呼び出されることを確認
            mock_fetch.assert_called_once_with('BTC/USDT', 100)
            
            # リポジトリのinsert_funding_rate_dataが呼び出されることを確認
            mock_repository.insert_funding_rate_data.assert_called_once()
            
            # 結果の検証
            assert result['symbol'] == 'BTC/USDT'
            assert result['fetched_count'] == 2
            assert result['saved_count'] == 2
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_save_funding_rate_to_database(
        self, funding_rate_service, mock_repository, sample_funding_history
    ):
        """ファンディングレートデータベース保存のテスト"""
        mock_repository.insert_funding_rate_data.return_value = 2
        
        result = await funding_rate_service._save_funding_rate_to_database(
            sample_funding_history, 'BTC/USDT', mock_repository
        )
        
        # リポジトリが呼び出されることを確認
        mock_repository.insert_funding_rate_data.assert_called_once()
        
        # 呼び出し引数の検証
        call_args = mock_repository.insert_funding_rate_data.call_args[0][0]
        assert len(call_args) == 2
        
        # 最初のレコードの検証
        first_record = call_args[0]
        assert first_record['symbol'] == 'BTC/USDT:USDT'
        assert first_record['funding_rate'] == 2.161e-05
        assert isinstance(first_record['funding_timestamp'], datetime)
        assert isinstance(first_record['timestamp'], datetime)
        
        # 結果の検証
        assert result == 2

    def test_service_initialization(self, funding_rate_service):
        """サービス初期化のテスト"""
        assert funding_rate_service.exchange is not None
        assert funding_rate_service.exchange.id == 'bybit'
        assert funding_rate_service.exchange.options['defaultType'] == 'linear'

    @pytest.mark.asyncio
    async def test_fetch_current_funding_rate_with_ccxt_error(self, funding_rate_service):
        """CCXTエラー時のテスト"""
        import ccxt
        
        with patch.object(
            funding_rate_service.exchange, 'fetch_funding_rate',
            side_effect=ccxt.BadSymbol('Invalid symbol')
        ):
            with pytest.raises(ccxt.BadSymbol):
                await funding_rate_service.fetch_current_funding_rate('INVALID/SYMBOL')

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_history_with_network_error(self, funding_rate_service):
        """ネットワークエラー時のテスト"""
        import ccxt
        
        with patch.object(
            funding_rate_service.exchange, 'fetch_funding_rate_history',
            side_effect=ccxt.NetworkError('Network error')
        ):
            with pytest.raises(ccxt.NetworkError):
                await funding_rate_service.fetch_funding_rate_history('BTC/USDT')

    @pytest.mark.asyncio
    async def test_fetch_and_save_with_database_error(
        self, funding_rate_service, mock_repository, sample_funding_history
    ):
        """データベースエラー時のテスト"""
        mock_repository.insert_funding_rate_data.side_effect = Exception('Database error')
        
        with patch.object(
            funding_rate_service, 'fetch_funding_rate_history',
            return_value=sample_funding_history
        ):
            with pytest.raises(Exception, match='Database error'):
                await funding_rate_service.fetch_and_save_funding_rate_data(
                    'BTC/USDT', 100, mock_repository
                )
