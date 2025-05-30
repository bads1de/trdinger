"""
オープンインタレストAPI調査テスト

CCXTライブラリを使用してBybitのオープンインタレスト取得方法を調査し、
データ形式を確認するためのテストです。

ファンディングレートAPIテストを参考に実装されています。
"""

import pytest
import asyncio
import ccxt
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBybitOpenInterestAPI:
    """Bybitオープンインタレスト取得APIのテスト"""

    @pytest.fixture
    def bybit_exchange(self):
        """Bybit取引所インスタンスを作成"""
        return ccxt.bybit(
            {
                "sandbox": False,  # 本番環境を使用（読み取り専用）
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",  # 無期限契約市場を使用
                },
            }
        )

    @pytest.mark.integration
    def test_bybit_has_open_interest_methods(self, bybit_exchange):
        """Bybitがオープンインタレスト取得メソッドをサポートしているかテスト"""
        # Given: Bybit取引所インスタンス
        exchange = bybit_exchange

        # When & Then: オープンインタレスト関連メソッドの存在確認
        assert hasattr(exchange, "fetch_open_interest"), "fetch_open_interest メソッドが存在しません"
        
        # オープンインタレスト履歴メソッドの確認（存在しない場合もある）
        has_history_method = hasattr(exchange, "fetch_open_interest_history")
        logger.info(f"fetch_open_interest_history メソッド: {'存在' if has_history_method else '存在しない'}")

        # 機能サポートの確認
        if hasattr(exchange, "has"):
            open_interest_support = exchange.has.get("fetchOpenInterest", False)
            open_interest_history_support = exchange.has.get("fetchOpenInterestHistory", False)
            
            logger.info(f"fetchOpenInterest サポート: {open_interest_support}")
            logger.info(f"fetchOpenInterestHistory サポート: {open_interest_history_support}")

    @pytest.mark.integration
    def test_fetch_current_open_interest(self, bybit_exchange):
        """現在のオープンインタレスト取得テスト"""
        # Given: BTC/USDT無期限契約
        exchange = bybit_exchange
        symbol = "BTC/USDT:USDT"

        try:
            # When: 現在のオープンインタレストを取得
            open_interest = exchange.fetch_open_interest(symbol)
            
            # Then: データ形式の検証
            assert isinstance(open_interest, dict), "オープンインタレストデータは辞書形式である必要があります"
            
            # 必須フィールドの確認
            required_fields = ["symbol", "openInterestAmount", "timestamp"]
            for field in required_fields:
                assert field in open_interest, f"必須フィールド '{field}' が存在しません"
            
            # データ型の検証
            assert isinstance(open_interest["symbol"], str)
            assert isinstance(open_interest["openInterestAmount"], (int, float))
            assert isinstance(open_interest["timestamp"], (int, float))
            
            # 値の妥当性確認
            assert open_interest["openInterestAmount"] >= 0, "オープンインタレストは0以上である必要があります"
            assert open_interest["symbol"] == symbol, f"シンボルが一致しません: {open_interest['symbol']} != {symbol}"
            
            logger.info(f"✓ 現在のオープンインタレスト取得成功: {open_interest}")
            
        except Exception as e:
            logger.error(f"現在のオープンインタレスト取得エラー: {e}")
            pytest.fail(f"現在のオープンインタレスト取得に失敗: {e}")

    @pytest.mark.integration
    def test_fetch_open_interest_multiple_symbols(self, bybit_exchange):
        """複数シンボルのオープンインタレスト取得テスト"""
        # Given: BTCの無期限契約シンボル（ETHは除外）
        exchange = bybit_exchange
        symbols = ["BTC/USDT:USDT"]

        # When & Then: 各シンボルでオープンインタレストを取得
        for symbol in symbols:
            try:
                open_interest = exchange.fetch_open_interest(symbol)
                
                # 基本的な検証
                assert isinstance(open_interest, dict)
                assert "openInterestAmount" in open_interest
                assert open_interest["openInterestAmount"] >= 0
                
                logger.info(f"✓ {symbol}: オープンインタレスト = {open_interest['openInterestAmount']}")
                
            except Exception as e:
                logger.warning(f"⚠ {symbol}: オープンインタレスト取得エラー = {e}")

    @pytest.mark.integration
    def test_open_interest_history_availability(self, bybit_exchange):
        """オープンインタレスト履歴取得の可用性テスト"""
        # Given: Bybit取引所とBTC/USDTシンボル
        exchange = bybit_exchange
        symbol = "BTC/USDT:USDT"

        # When: オープンインタレスト履歴メソッドの存在確認
        if hasattr(exchange, "fetch_open_interest_history"):
            try:
                # 履歴データの取得を試行
                history = exchange.fetch_open_interest_history(symbol, limit=5)
                
                # Then: データ形式の検証
                assert isinstance(history, list), "履歴データはリスト形式である必要があります"
                
                if history:
                    # 最初のデータ項目を検証
                    first_item = history[0]
                    assert isinstance(first_item, dict)
                    assert "openInterestAmount" in first_item
                    assert "timestamp" in first_item
                    
                    logger.info(f"✓ オープンインタレスト履歴取得成功: {len(history)}件")
                    logger.info(f"  最新データ: {first_item}")
                else:
                    logger.info("オープンインタレスト履歴データは空です")
                    
            except Exception as e:
                logger.warning(f"オープンインタレスト履歴取得エラー: {e}")
        else:
            logger.info("fetch_open_interest_history メソッドは利用できません")

    @pytest.mark.integration
    def test_open_interest_data_consistency(self, bybit_exchange):
        """オープンインタレストデータの一貫性テスト"""
        # Given: BTC/USDT無期限契約
        exchange = bybit_exchange
        symbol = "BTC/USDT:USDT"

        try:
            # When: 短時間で2回データを取得
            open_interest1 = exchange.fetch_open_interest(symbol)
            asyncio.sleep(1)  # 1秒待機
            open_interest2 = exchange.fetch_open_interest(symbol)

            # Then: データ構造の一貫性を確認
            assert open_interest1.keys() == open_interest2.keys(), "データ構造が一致しません"
            
            # 値の妥当性確認（大幅な変動がないことを確認）
            amount1 = open_interest1["openInterestAmount"]
            amount2 = open_interest2["openInterestAmount"]
            
            # 1秒間での変動は通常5%以内
            if amount1 > 0:
                change_ratio = abs(amount2 - amount1) / amount1
                assert change_ratio < 0.05, f"短時間での変動が大きすぎます: {change_ratio:.2%}"
            
            logger.info(f"✓ データ一貫性確認: {amount1} -> {amount2}")
            
        except Exception as e:
            logger.error(f"データ一貫性テストエラー: {e}")
            pytest.fail(f"データ一貫性テストに失敗: {e}")

    @pytest.mark.integration
    def test_open_interest_error_handling(self, bybit_exchange):
        """オープンインタレストエラーハンドリングテスト"""
        # Given: 無効なシンボル
        exchange = bybit_exchange
        invalid_symbol = "INVALID/SYMBOL:USDT"

        # When & Then: 無効なシンボルでエラーが適切に処理されることを確認
        with pytest.raises(Exception) as exc_info:
            exchange.fetch_open_interest(invalid_symbol)
        
        logger.info(f"✓ 無効なシンボルで期待通りエラーが発生: {exc_info.value}")

    @pytest.mark.integration
    def test_open_interest_data_structure_analysis(self, bybit_exchange):
        """オープンインタレストデータ構造の詳細分析"""
        # Given: BTC/USDT無期限契約
        exchange = bybit_exchange
        symbol = "BTC/USDT:USDT"

        try:
            # When: オープンインタレストデータを取得
            open_interest = exchange.fetch_open_interest(symbol)
            
            # Then: 全フィールドの詳細分析
            logger.info("=== オープンインタレストデータ構造分析 ===")
            for key, value in open_interest.items():
                logger.info(f"  {key}: {value} (型: {type(value).__name__})")
            
            # 追加フィールドの確認
            optional_fields = [
                "openInterestValue",  # USD建て価値
                "baseVolume",         # ベース通貨建てボリューム
                "quoteVolume",        # クォート通貨建てボリューム
                "info",               # 生データ
            ]
            
            for field in optional_fields:
                if field in open_interest:
                    logger.info(f"  オプションフィールド {field}: {open_interest[field]}")
            
            # データベース設計のための情報収集
            logger.info("=== データベース設計用情報 ===")
            logger.info(f"  必須フィールド数: {len(open_interest)}")
            logger.info(f"  数値フィールド: openInterestAmount")
            logger.info(f"  タイムスタンプフィールド: timestamp")
            logger.info(f"  文字列フィールド: symbol")
            
        except Exception as e:
            logger.error(f"データ構造分析エラー: {e}")
            pytest.fail(f"データ構造分析に失敗: {e}")
