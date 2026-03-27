"""
BaseDataCollectionOrchestrationService のテストモジュール

共通基底クラスの全メソッドをテストします:
- _parse_datetime: 日付文字列パース
- _get_db_session: DBセッション管理
- _create_success_response / _create_error_response: レスポンス生成
- _normalize_derivative_symbol: シンボル正規化
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.services.data_collection.orchestration.base_orchestration_service import (
    BaseDataCollectionOrchestrationService,
)


@pytest.fixture
def service() -> BaseDataCollectionOrchestrationService:
    """テスト対象のサービスインスタンス"""
    return BaseDataCollectionOrchestrationService()


class TestParseDatetime:
    """_parse_datetime のテスト"""

    def test_parse_iso_format(self, service: BaseDataCollectionOrchestrationService):
        """ISO形式の日付文字列が正しくパースされる"""
        result = service._parse_datetime("2024-01-15T10:30:00")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_iso_format_with_z(self, service: BaseDataCollectionOrchestrationService):
        """Zサフィックス付きの日付文字列がパースされる"""
        result = service._parse_datetime("2024-06-01T00:00:00Z")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 6

    def test_parse_iso_format_with_timezone(self, service: BaseDataCollectionOrchestrationService):
        """タイムゾーン付きの日付文字列がパースされる"""
        result = service._parse_datetime("2024-01-01T00:00:00+09:00")
        assert isinstance(result, datetime)

    def test_parse_none_returns_none(self, service: BaseDataCollectionOrchestrationService):
        """None を渡すと None が返る"""
        assert service._parse_datetime(None) is None

    def test_parse_empty_string_returns_none(self, service: BaseDataCollectionOrchestrationService):
        """空文字列を渡すと None が返る"""
        assert service._parse_datetime("") is None

    def test_parse_invalid_format_returns_none(self, service: BaseDataCollectionOrchestrationService):
        """不正な形式の文字列を渡すと None が返る"""
        assert service._parse_datetime("not-a-date") is None
        assert service._parse_datetime("2024/01/01") is None
        assert service._parse_datetime("invalid") is None

    def test_parse_date_only(self, service: BaseDataCollectionOrchestrationService):
        """日付のみの文字列がパースされる"""
        result = service._parse_datetime("2024-03-15")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 15


class TestGetDbSession:
    """_get_db_session のテスト"""

    def test_uses_existing_session(self, service: BaseDataCollectionOrchestrationService):
        """既存セッションが渡された場合はそのまま使用される"""
        mock_session = MagicMock()

        with service._get_db_session(mock_session) as session:
            assert session is mock_session

        # 既存セッションは close されない
        mock_session.close.assert_not_called()

    def test_creates_new_session_when_none(self, service: BaseDataCollectionOrchestrationService):
        """None が渡された場合は新規セッションを作成する"""
        mock_session = MagicMock()

        with patch(
            "app.services.data_collection.orchestration.base_orchestration_service.get_db"
        ) as mock_get_db:
            mock_get_db.return_value = iter([mock_session])

            with service._get_db_session(None) as session:
                assert session is mock_session

            # 新規セッションは close される
            mock_session.close.assert_called_once()

    def test_closes_session_on_exception(self, service: BaseDataCollectionOrchestrationService):
        """コンテキスト内で例外が発生してもセッションが close される"""
        mock_session = MagicMock()

        with patch(
            "app.services.data_collection.orchestration.base_orchestration_service.get_db"
        ) as mock_get_db:
            mock_get_db.return_value = iter([mock_session])

            with pytest.raises(RuntimeError):
                with service._get_db_session(None):
                    raise RuntimeError("test error")

            mock_session.close.assert_called_once()


class TestCreateSuccessResponse:
    """_create_success_response のテスト"""

    def test_success_response_basic(self, service: BaseDataCollectionOrchestrationService):
        """基本的な成功レスポンスが生成される"""
        result = service._create_success_response("処理完了")

        assert result["success"] is True
        assert result["message"] == "処理完了"

    def test_success_response_with_data(self, service: BaseDataCollectionOrchestrationService):
        """データ付きの成功レスポンスが生成される"""
        data = {"count": 100, "symbol": "BTC/USDT:USDT"}
        result = service._create_success_response("保存完了", data=data)

        assert result["success"] is True
        assert result["data"]["count"] == 100
        assert result["data"]["symbol"] == "BTC/USDT:USDT"

    def test_success_response_without_data(self, service: BaseDataCollectionOrchestrationService):
        """データなしの成功レスポンスが生成される"""
        result = service._create_success_response("OK")

        assert result["success"] is True
        assert result.get("data") is None or "data" not in result or result["data"] is None


class TestCreateErrorResponse:
    """_create_error_response のテスト"""

    def test_error_response_basic(self, service: BaseDataCollectionOrchestrationService):
        """基本的なエラーレスポンスが生成される"""
        result = service._create_error_response("エラー発生")

        assert result["success"] is False
        assert result["message"] == "エラー発生"

    def test_error_response_with_details(self, service: BaseDataCollectionOrchestrationService):
        """詳細付きのエラーレスポンスが生成される"""
        result = service._create_error_response(
            "失敗",
            details={"reason": "timeout"},
            error_code="NET_001",
            context="データ収集",
        )

        assert result["success"] is False
        assert result["message"] == "失敗"
        assert result["details"]["reason"] == "timeout"
        assert result["error_code"] == "NET_001"

    def test_error_response_with_data(self, service: BaseDataCollectionOrchestrationService):
        """データ付きのエラーレスポンスが生成される"""
        result = service._create_error_response(
            "部分的失敗",
            data={"partial_count": 5},
        )

        assert result["success"] is False
        assert result["data"]["partial_count"] == 5


class TestNormalizeDerivativeSymbol:
    """_normalize_derivative_symbol のテスト"""

    def test_already_normalized(self, service: BaseDataCollectionOrchestrationService):
        """既にコロン記法のシンボルはそのまま返される"""
        assert service._normalize_derivative_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"
        assert service._normalize_derivative_symbol("ETH/USD:USD") == "ETH/USD:USD"

    def test_usdt_pair_adds_suffix(self, service: BaseDataCollectionOrchestrationService):
        """USDT ペアにサフィックスが付与される"""
        assert service._normalize_derivative_symbol("BTC/USDT") == "BTC/USDT:USDT"
        assert service._normalize_derivative_symbol("ETH/USDT") == "ETH/USDT:USDT"

    def test_usd_pair_adds_suffix(self, service: BaseDataCollectionOrchestrationService):
        """USD ペアにサフィックスが付与される"""
        assert service._normalize_derivative_symbol("BTC/USD") == "BTC/USD:USD"
        assert service._normalize_derivative_symbol("ETH/USD") == "ETH/USD:USD"

    def test_plain_symbol_adds_usdt_suffix(self, service: BaseDataCollectionOrchestrationService):
        """プレーンシンボルには :USDT が付与される"""
        assert service._normalize_derivative_symbol("BTC") == "BTC:USDT"
        assert service._normalize_derivative_symbol("SOL") == "SOL:USDT"

    def test_various_symbols(self, service: BaseDataCollectionOrchestrationService):
        """様々なシンボル形式が正しく正規化される"""
        assert service._normalize_derivative_symbol("DOGE/USDT") == "DOGE/USDT:USDT"
        assert service._normalize_derivative_symbol("XRP/USD") == "XRP/USD:USD"
        assert service._normalize_derivative_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"
