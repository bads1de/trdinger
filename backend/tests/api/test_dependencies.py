"""
API依存性注入ファクトリのユニットテスト

依存性注入用のファクトリ関数をテストします。
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.dependencies import (
    _create_service,
    get_auto_strategy_service,
    get_generated_strategy_service_with_db,
    get_long_short_ratio_repository,
    get_long_short_ratio_service,
    get_market_data_orchestration_service,
)


class TestCreateService:
    """_create_service関数のテスト"""

    def test_successful_creation(self):
        """サービスが正常に作成されること"""
        mock_factory = MagicMock(return_value="service_instance")
        result = _create_service(mock_factory, "TestService")
        assert result == "service_instance"
        mock_factory.assert_called_once()

    def test_factory_error_raises_http_exception(self):
        """ファクトリエラー時にHTTPExceptionが発生すること"""
        mock_factory = MagicMock(side_effect=RuntimeError("Connection failed"))

        with pytest.raises(HTTPException) as exc_info:
            _create_service(mock_factory, "TestService")

        assert exc_info.value.status_code == 503
        assert "TestService" in exc_info.value.detail


class TestGetMarketDataOrchestrationService:
    """get_market_data_orchestration_service関数のテスト"""

    @patch("app.api.dependencies.MarketDataOrchestrationService")
    def test_returns_service_instance(self, mock_service_cls):
        """サービスインスタンスが返されること"""
        mock_db = MagicMock()
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        result = get_market_data_orchestration_service(mock_db)

        assert result is mock_service
        mock_service_cls.assert_called_once_with(mock_db)


class TestGetAutoStrategyService:
    """get_auto_strategy_service関数のテスト"""

    @patch("app.api.dependencies.AutoStrategyService")
    def test_returns_service_instance(self, mock_service_cls):
        """サービスインスタンスが返されること"""
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        result = get_auto_strategy_service()

        assert result is mock_service
        mock_service_cls.assert_called_once()

    @patch("app.api.dependencies.AutoStrategyService")
    def test_error_raises_http_exception(self, mock_service_cls):
        """エラー時にHTTPExceptionが発生すること"""
        mock_service_cls.side_effect = RuntimeError("Init failed")

        with pytest.raises(HTTPException) as exc_info:
            get_auto_strategy_service()

        assert exc_info.value.status_code == 503


class TestGetGeneratedStrategyServiceWithDb:
    """get_generated_strategy_service_with_db関数のテスト"""

    @patch("app.api.dependencies.GeneratedStrategyService")
    def test_returns_service_instance(self, mock_service_cls):
        """サービスインスタンスが返されること"""
        mock_db = MagicMock()
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        result = get_generated_strategy_service_with_db(mock_db)

        assert result is mock_service
        mock_service_cls.assert_called_once_with(mock_db)

    @patch("app.api.dependencies.GeneratedStrategyService")
    def test_error_raises_http_exception(self, mock_service_cls):
        """生成戦略サービスの初期化失敗時にHTTPExceptionが発生すること"""
        mock_db = MagicMock()
        mock_service_cls.side_effect = RuntimeError("Init failed")

        with pytest.raises(HTTPException) as exc_info:
            get_generated_strategy_service_with_db(mock_db)

        assert exc_info.value.status_code == 503
        assert "GeneratedStrategyService" in exc_info.value.detail


class TestGetLongShortRatioRepository:
    """get_long_short_ratio_repository関数のテスト"""

    @patch("app.api.dependencies.LongShortRatioRepository")
    def test_returns_repository_instance(self, mock_repo_cls):
        """リポジトリインスタンスが返されること"""
        mock_db = MagicMock()
        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo

        result = get_long_short_ratio_repository(mock_db)

        assert result is mock_repo
        mock_repo_cls.assert_called_once_with(mock_db)


class TestGetLongShortRatioService:
    """get_long_short_ratio_service関数のテスト"""

    @patch("app.api.dependencies.BybitLongShortRatioService")
    def test_returns_service_instance(self, mock_service_cls):
        """サービスインスタンスが返されること"""
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        result = get_long_short_ratio_service()

        assert result is mock_service
        mock_service_cls.assert_called_once()
