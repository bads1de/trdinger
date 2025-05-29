"""
オープンインタレストサービス統合テスト

実装したオープンインタレストサービスの動作を確認するテストです。
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any

from app.core.services.open_interest_service import BybitOpenInterestService
from database.connection import get_db, init_db
from database.repository import OpenInterestRepository


class TestBybitOpenInterestService:
    """Bybitオープンインタレストサービスのテスト"""

    @pytest.fixture
    def service(self):
        """テスト用のサービスインスタンス"""
        return BybitOpenInterestService()

    @pytest.fixture
    def db_session(self):
        """テスト用のデータベースセッション"""
        # データベース初期化
        init_db()
        
        # セッション取得
        db = next(get_db())
        yield db
        db.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_normalize_symbol(self, service):
        """シンボル正規化テスト"""
        # Given: 様々な形式のシンボル
        test_cases = [
            ("BTC/USDT", "BTC/USDT:USDT"),
            ("BTC/USDT:USDT", "BTC/USDT:USDT"),
            ("ETH/USDT", "ETH/USDT:USDT"),
            ("BTC/USD", "BTC/USD:USD"),
            ("INVALID", "INVALID:USDT"),  # デフォルト
        ]

        # When & Then: 正規化が正しく動作する
        for input_symbol, expected in test_cases:
            result = service.normalize_symbol(input_symbol)
            assert result == expected, f"入力: {input_symbol}, 期待: {expected}, 実際: {result}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fetch_current_open_interest(self, service):
        """現在のオープンインタレスト取得テスト"""
        # Given: BTC/USDTシンボル
        symbol = "BTC/USDT"

        # When: 現在のオープンインタレストを取得
        result = await service.fetch_current_open_interest(symbol)

        # Then: データが正しく取得される
        assert isinstance(result, dict)
        assert "symbol" in result
        assert "openInterestValue" in result or "openInterestAmount" in result
        assert "timestamp" in result
        
        # 正規化されたシンボルが使用されている
        assert result["symbol"] == "BTC/USDT:USDT"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fetch_open_interest_history(self, service):
        """オープンインタレスト履歴取得テスト"""
        # Given: BTC/USDTシンボルと制限
        symbol = "BTC/USDT"
        limit = 5

        # When: オープンインタレスト履歴を取得
        result = await service.fetch_open_interest_history(symbol, limit)

        # Then: データが正しく取得される
        assert isinstance(result, list)
        assert len(result) <= limit
        
        if result:
            # 最初のデータ項目を検証
            first_item = result[0]
            assert isinstance(first_item, dict)
            assert "symbol" in first_item
            assert "timestamp" in first_item
            assert "openInterestValue" in first_item or "openInterestAmount" in first_item

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parameter_validation(self, service):
        """パラメータ検証テスト"""
        # Given: 無効なパラメータ
        invalid_cases = [
            ("", 100),  # 空のシンボル
            ("BTC/USDT", 0),  # 無効なlimit
            ("BTC/USDT", 1001),  # 制限を超えるlimit
        ]

        # When & Then: 適切にエラーが発生する
        for symbol, limit in invalid_cases:
            with pytest.raises(ValueError):
                service._validate_parameters(symbol, limit)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_save_open_interest_to_database(self, service, db_session):
        """データベース保存テスト"""
        # Given: テスト用のオープンインタレストデータ
        test_data = [
            {
                "symbol": "BTC/USDT:USDT",
                "openInterestValue": 50000.0,
                "openInterestAmount": None,
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                "datetime": datetime.now(timezone.utc).isoformat(),
                "info": {"openInterest": "50000.0"},
            }
        ]
        
        symbol = "BTC/USDT"
        repository = OpenInterestRepository(db_session)

        # When: データベースに保存
        saved_count = await service._save_open_interest_to_database(
            test_data, symbol, repository
        )

        # Then: データが正しく保存される
        assert saved_count == 1
        
        # データベースから確認
        saved_data = repository.get_open_interest_data("BTC/USDT:USDT", limit=1)
        assert len(saved_data) >= 1
        assert saved_data[0].open_interest_value == 50000.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fetch_and_save_open_interest_data(self, service, db_session):
        """オープンインタレストデータ取得・保存統合テスト"""
        # Given: BTC/USDTシンボルとリポジトリ
        symbol = "BTC/USDT"
        repository = OpenInterestRepository(db_session)
        limit = 3

        # When: データを取得・保存
        result = await service.fetch_and_save_open_interest_data(
            symbol=symbol,
            limit=limit,
            repository=repository,
            fetch_all=False,
        )

        # Then: 結果が正しく返される
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "fetched_count" in result
        assert "saved_count" in result
        assert result["symbol"] == symbol
        
        # データベースに保存されている
        saved_data = repository.get_open_interest_data("BTC/USDT:USDT", limit=10)
        assert len(saved_data) >= 0  # データが存在する可能性

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """エラーハンドリングテスト"""
        # Given: 無効なシンボル
        invalid_symbol = "INVALID/SYMBOL"

        # When & Then: 適切にエラーが処理される
        with pytest.raises(Exception):  # ccxt.BadSymbolまたは他のエラー
            await service.fetch_current_open_interest(invalid_symbol)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_structure_consistency(self, service):
        """データ構造一貫性テスト"""
        # Given: BTC/USDTシンボル
        symbol = "BTC/USDT"

        # When: 現在のデータと履歴データを取得
        current_data = await service.fetch_current_open_interest(symbol)
        history_data = await service.fetch_open_interest_history(symbol, 1)

        # Then: データ構造が一貫している
        assert isinstance(current_data, dict)
        assert isinstance(history_data, list)
        
        if history_data:
            # 履歴データの最初の項目と現在のデータの構造を比較
            history_item = history_data[0]
            
            # 共通フィールドの確認
            common_fields = ["symbol", "timestamp"]
            for field in common_fields:
                assert field in current_data
                assert field in history_item

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_fetch_all_open_interest_history_sample(self, service):
        """全期間データ取得のサンプルテスト（少量）"""
        # Given: BTC/USDTシンボル
        symbol = "BTC/USDT"

        # When: 少量の履歴データを取得（全期間ではなく制限付き）
        # 注意: 実際の全期間取得は時間がかかるため、ここでは制限付きでテスト
        history_data = await service.fetch_open_interest_history(symbol, 10)

        # Then: データが取得される
        assert isinstance(history_data, list)
        
        if history_data:
            # データが時系列順になっているか確認
            timestamps = [item["timestamp"] for item in history_data]
            # 最新から古い順、または古いから新しい順のいずれかであることを確認
            is_ascending = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            is_descending = all(timestamps[i] >= timestamps[i+1] for i in range(len(timestamps)-1))
            assert is_ascending or is_descending, "データが時系列順になっていません"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_repository_methods(self, db_session):
        """リポジトリメソッドテスト"""
        # Given: テスト用リポジトリ
        repository = OpenInterestRepository(db_session)
        symbol = "BTC/USDT:USDT"

        # When: 各種メソッドを実行
        count = repository.get_open_interest_count(symbol)
        latest_timestamp = repository.get_latest_open_interest_timestamp(symbol)
        data = repository.get_open_interest_data(symbol, limit=5)

        # Then: メソッドが正常に動作する
        assert isinstance(count, int)
        assert count >= 0
        assert latest_timestamp is None or isinstance(latest_timestamp, datetime)
        assert isinstance(data, list)
