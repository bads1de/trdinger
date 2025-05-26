"""
市場データAPIエンドポイントのテスト

FastAPIを使用した市場データ取得エンドポイントの統合テストです。
実際のBybit APIを呼び出してOHLCVデータを取得し、APIレスポンスを検証します。

@author Trdinger Development Team
@version 1.0.0
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

# テスト対象のモジュール（まだ存在しないため、インポートエラーが発生する）
try:
    from backend.run import app

    client = TestClient(app)
except ImportError:
    # テスト実行時にモジュールが存在しない場合のダミー
    app = None
    client = None


class TestMarketDataAPI:
    """市場データAPIエンドポイントのテストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        if client is None:
            pytest.skip("FastAPIアプリケーションが設定されていません")

    def test_ohlcv_endpoint_success(self):
        """
        OHLCVエンドポイントの正常動作テスト

        実際のBybit APIを呼び出してBTC/USD:BTCのデータを取得し、
        APIレスポンスの形式と内容を検証します。
        """
        # APIエンドポイントを呼び出し
        response = client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "BTC/USD:BTC", "timeframe": "1h", "limit": 10},
        )

        # ステータスコードの確認
        assert response.status_code == 200

        # レスポンス形式の確認
        data = response.json()
        assert "success" in data
        assert "data" in data
        assert "symbol" in data
        assert "timeframe" in data
        assert "message" in data
        assert "timestamp" in data

        # 成功レスポンスの確認
        assert data["success"] is True
        assert data["symbol"] == "BTC/USD:BTC"
        assert data["timeframe"] == "1h"

        # データの確認
        ohlcv_data = data["data"]
        assert isinstance(ohlcv_data, list)
        assert len(ohlcv_data) <= 10
        assert len(ohlcv_data) > 0

        # 各ローソク足データの確認
        for candle in ohlcv_data:
            assert isinstance(candle, list)
            assert len(candle) == 6  # [timestamp, open, high, low, close, volume]

            timestamp, open_price, high, low, close, volume = candle

            # データ型の確認
            assert isinstance(timestamp, (int, float))
            assert isinstance(open_price, (int, float))
            assert isinstance(high, (int, float))
            assert isinstance(low, (int, float))
            assert isinstance(close, (int, float))
            assert isinstance(volume, (int, float))

            # 価格関係の確認
            assert high >= max(open_price, close)
            assert low <= min(open_price, close)
            assert high >= low
            assert open_price > 0
            assert close > 0
            assert volume >= 0

    def test_ohlcv_endpoint_missing_symbol(self):
        """
        必須パラメータ（symbol）が不足している場合のテスト
        """
        response = client.get(
            "/api/market-data/ohlcv", params={"timeframe": "1h", "limit": 10}
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_ohlcv_endpoint_invalid_symbol(self):
        """
        無効なシンボルでのエラーハンドリングテスト
        """
        response = client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "INVALID/SYMBOL", "timeframe": "1h", "limit": 10},
        )

        assert response.status_code == 400  # Bad Request

        data = response.json()
        # FastAPIのHTTPExceptionはdetailフィールドにエラー情報を格納
        detail = data.get("detail", data)
        assert detail["success"] is False
        assert "message" in detail

    def test_ohlcv_endpoint_invalid_timeframe(self):
        """
        無効な時間軸でのエラーハンドリングテスト
        """
        response = client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "BTC/USD:BTC", "timeframe": "invalid", "limit": 10},
        )

        assert response.status_code == 400  # Bad Request

        data = response.json()
        detail = data.get("detail", data)
        assert detail["success"] is False
        assert "message" in detail

    def test_ohlcv_endpoint_invalid_limit(self):
        """
        無効な制限値でのエラーハンドリングテスト
        """
        # 制限値が小さすぎる場合
        response = client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "BTC/USD:BTC", "timeframe": "1h", "limit": 0},
        )

        assert response.status_code == 422  # FastAPIのバリデーションエラー

        # 制限値が大きすぎる場合
        response = client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "BTC/USD:BTC", "timeframe": "1h", "limit": 2000},
        )

        assert response.status_code == 422  # FastAPIのバリデーションエラー

    def test_ohlcv_endpoint_default_parameters(self):
        """
        デフォルトパラメータでのテスト
        """
        response = client.get(
            "/api/market-data/ohlcv", params={"symbol": "BTC/USD:BTC"}
        )

        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["symbol"] == "BTC/USD:BTC"
        assert data["timeframe"] == "1h"  # デフォルト値

        # デフォルトの制限値（100）以下のデータが返されることを確認
        ohlcv_data = data["data"]
        assert len(ohlcv_data) <= 100

    def test_ohlcv_endpoint_symbol_normalization(self):
        """
        シンボル正規化のテスト

        様々な形式のシンボルが正規化されて処理されることを確認します。
        """
        test_symbols = ["BTCUSD", "BTC/USD", "btc/usd", "BTC-USD"]

        for symbol in test_symbols:
            response = client.get(
                "/api/market-data/ohlcv",
                params={"symbol": symbol, "timeframe": "1h", "limit": 5},
            )

            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["symbol"] == "BTC/USD:BTC"  # 正規化された形式

    def test_ohlcv_endpoint_response_format(self):
        """
        レスポンス形式の詳細テスト
        """
        response = client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "BTC/USD:BTC", "timeframe": "1h", "limit": 5},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()

        # 必須フィールドの確認
        required_fields = [
            "success",
            "data",
            "symbol",
            "timeframe",
            "message",
            "timestamp",
        ]
        for field in required_fields:
            assert field in data, f"必須フィールド '{field}' が不足しています"

        # タイムスタンプ形式の確認
        timestamp_str = data["timestamp"]
        try:
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"無効なタイムスタンプ形式: {timestamp_str}")

    def test_health_check_endpoint(self):
        """
        ヘルスチェックエンドポイントのテスト
        """
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
