"""
DataManagementOrchestrationService のテスト

TDDアプローチでOrchestrationServiceの動作を検証します。
"""

import pytest
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.core.services.data_collection.data_management_orchestration_service import (
    DataManagementOrchestrationService,
)


class TestDataManagementOrchestrationService:
    """DataManagementOrchestrationService のテストクラス"""

    @pytest.fixture
    def mock_db_session(self):
        """モックデータベースセッション"""
        mock_session = Mock(spec=Session)
        return mock_session

    @pytest.fixture
    def orchestration_service(self):
        """OrchestrationService インスタンス"""
        return DataManagementOrchestrationService()

    @pytest.mark.asyncio
    async def test_reset_all_data_success(
        self, orchestration_service, mock_db_session
    ):
        """全データリセットの成功テスト"""
        with patch(
            "app.core.services.data_collection.data_management_orchestration_service.OHLCVRepository"
        ) as mock_ohlcv_repo, patch(
            "app.core.services.data_collection.data_management_orchestration_service.FundingRateRepository"
        ) as mock_fr_repo, patch(
            "app.core.services.data_collection.data_management_orchestration_service.OpenInterestRepository"
        ) as mock_oi_repo:
            # モック設定
            mock_ohlcv_repo.return_value.clear_all_ohlcv_data.return_value = 1000
            mock_fr_repo.return_value.clear_all_funding_rate_data.return_value = 500
            mock_oi_repo.return_value.clear_all_open_interest_data.return_value = 300

            # テスト実行
            result = await orchestration_service.reset_all_data(
                db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["deleted_counts"]["ohlcv"] == 1000
            assert result["data"]["deleted_counts"]["funding_rates"] == 500
            assert result["data"]["deleted_counts"]["open_interest"] == 300
            assert result["data"]["total_deleted"] == 1800
            assert len(result["data"]["errors"]) == 0

    @pytest.mark.asyncio
    async def test_reset_all_data_partial_failure(
        self, orchestration_service, mock_db_session
    ):
        """全データリセットの部分失敗テスト"""
        with patch(
            "app.core.services.data_collection.data_management_orchestration_service.OHLCVRepository"
        ) as mock_ohlcv_repo, patch(
            "app.core.services.data_collection.data_management_orchestration_service.FundingRateRepository"
        ) as mock_fr_repo, patch(
            "app.core.services.data_collection.data_management_orchestration_service.OpenInterestRepository"
        ) as mock_oi_repo:
            # モック設定（一部でエラー）
            mock_ohlcv_repo.return_value.clear_all_ohlcv_data.return_value = 1000
            mock_fr_repo.return_value.clear_all_funding_rate_data.side_effect = Exception(
                "削除エラー"
            )
            mock_oi_repo.return_value.clear_all_open_interest_data.return_value = 300

            # テスト実行
            result = await orchestration_service.reset_all_data(
                db_session=mock_db_session
            )

            # 検証
            assert result["success"] is False
            assert result["data"]["deleted_counts"]["ohlcv"] == 1000
            assert result["data"]["deleted_counts"]["funding_rates"] == 0
            assert result["data"]["deleted_counts"]["open_interest"] == 300
            assert result["data"]["total_deleted"] == 1300
            assert len(result["data"]["errors"]) == 1
            assert "削除エラー" in result["data"]["errors"][0]

    @pytest.mark.asyncio
    async def test_reset_ohlcv_data_success(
        self, orchestration_service, mock_db_session
    ):
        """OHLCVデータリセットの成功テスト"""
        with patch(
            "app.core.services.data_collection.data_management_orchestration_service.OHLCVRepository"
        ) as mock_ohlcv_repo:
            # モック設定
            mock_ohlcv_repo.return_value.clear_all_ohlcv_data.return_value = 1000

            # テスト実行
            result = await orchestration_service.reset_ohlcv_data(
                db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["deleted_count"] == 1000
            assert result["data"]["data_type"] == "ohlcv"

    @pytest.mark.asyncio
    async def test_reset_funding_rate_data_success(
        self, orchestration_service, mock_db_session
    ):
        """ファンディングレートデータリセットの成功テスト"""
        with patch(
            "app.core.services.data_collection.data_management_orchestration_service.FundingRateRepository"
        ) as mock_fr_repo:
            # モック設定
            mock_fr_repo.return_value.clear_all_funding_rate_data.return_value = 500

            # テスト実行
            result = await orchestration_service.reset_funding_rate_data(
                db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["deleted_count"] == 500
            assert result["data"]["data_type"] == "funding_rates"

    @pytest.mark.asyncio
    async def test_reset_open_interest_data_success(
        self, orchestration_service, mock_db_session
    ):
        """オープンインタレストデータリセットの成功テスト"""
        with patch(
            "app.core.services.data_collection.data_management_orchestration_service.OpenInterestRepository"
        ) as mock_oi_repo:
            # モック設定
            mock_oi_repo.return_value.clear_all_open_interest_data.return_value = 300

            # テスト実行
            result = await orchestration_service.reset_open_interest_data(
                db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["deleted_count"] == 300
            assert result["data"]["data_type"] == "open_interest"

    @pytest.mark.asyncio
    async def test_reset_data_by_symbol_success(
        self, orchestration_service, mock_db_session
    ):
        """シンボル別データリセットの成功テスト"""
        symbol = "BTC/USDT:USDT"

        with patch(
            "app.core.services.data_collection.data_management_orchestration_service.OHLCVRepository"
        ) as mock_ohlcv_repo, patch(
            "app.core.services.data_collection.data_management_orchestration_service.FundingRateRepository"
        ) as mock_fr_repo, patch(
            "app.core.services.data_collection.data_management_orchestration_service.OpenInterestRepository"
        ) as mock_oi_repo:
            # モック設定
            mock_ohlcv_repo.return_value.clear_ohlcv_data_by_symbol.return_value = 100
            mock_fr_repo.return_value.clear_funding_rate_data_by_symbol.return_value = 50
            mock_oi_repo.return_value.clear_open_interest_data_by_symbol.return_value = 30

            # テスト実行
            result = await orchestration_service.reset_data_by_symbol(
                symbol=symbol, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["symbol"] == symbol
            assert result["data"]["deleted_counts"]["ohlcv"] == 100
            assert result["data"]["deleted_counts"]["funding_rates"] == 50
            assert result["data"]["deleted_counts"]["open_interest"] == 30
            assert result["data"]["total_deleted"] == 180
            assert len(result["data"]["errors"]) == 0

    @pytest.mark.asyncio
    async def test_exception_handling(self, orchestration_service, mock_db_session):
        """例外処理のテスト"""
        with patch(
            "app.core.services.data_collection.data_management_orchestration_service.OHLCVRepository"
        ) as mock_ohlcv_repo:
            # 例外を発生させる
            mock_ohlcv_repo.side_effect = Exception("データベース接続エラー")

            # テスト実行
            result = await orchestration_service.reset_ohlcv_data(
                db_session=mock_db_session
            )

            # 検証
            assert result["success"] is False
            assert "データベース接続エラー" in result["message"]

    def test_initialization(self, orchestration_service):
        """初期化テスト"""
        assert orchestration_service is not None
        assert hasattr(orchestration_service, "reset_all_data")
        assert hasattr(orchestration_service, "reset_ohlcv_data")
        assert hasattr(orchestration_service, "reset_funding_rate_data")
        assert hasattr(orchestration_service, "reset_open_interest_data")
        assert hasattr(orchestration_service, "reset_data_by_symbol")


if __name__ == "__main__":
    # 単体でテスト実行
    pytest.main([__file__, "-v"])
