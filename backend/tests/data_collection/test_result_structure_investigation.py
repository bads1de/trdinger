"""
結果構造調査テスト

一括差分更新の結果構造を詳しく調査します。
"""

import pytest
import asyncio
import logging
import json
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from app.services.data_collection.historical.historical_data_service import (
    HistoricalDataService,
)
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestResultStructureInvestigation:
    """結果構造調査テスト"""

    @pytest.fixture
    def mock_repositories(self):
        """モックリポジトリを作成"""
        # OHLCVリポジトリ（既存データがあることにする）
        ohlcv_repo = Mock(spec=OHLCVRepository)
        ohlcv_repo.get_data_count.return_value = 100

        # ファンディングレートリポジトリ
        funding_rate_repo = Mock(spec=FundingRateRepository)
        funding_rate_repo.get_latest_funding_timestamp.return_value = None
        funding_rate_repo.insert_funding_rate_data.return_value = 10

        # オープンインタレストリポジトリ
        open_interest_repo = Mock(spec=OpenInterestRepository)
        open_interest_repo.get_latest_open_interest_timestamp.return_value = None
        open_interest_repo.insert_open_interest_data.return_value = 5

        return {
            "ohlcv": ohlcv_repo,
            "funding_rate": funding_rate_repo,
            "open_interest": open_interest_repo,
        }

    @pytest.mark.asyncio
    async def test_historical_service_result_structure(self, mock_repositories):
        """履歴サービスの結果構造を詳しく調査"""
        service = HistoricalDataService()

        logger.info("=== 履歴サービス結果構造調査開始 ===")

        try:
            result = await service.collect_bulk_incremental_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                ohlcv_repository=None,  # OHLCVをスキップしてエラーを回避
                funding_rate_repository=mock_repositories["funding_rate"],
                open_interest_repository=mock_repositories["open_interest"],
            )

            print("\n=== 履歴サービス結果構造 ===")
            print(f"結果のキー: {list(result.keys())}")
            print(f"結果の型: {type(result)}")

            if "data" in result:
                data = result["data"]
                print(f"dataのキー: {list(data.keys())}")
                print(f"dataの型: {type(data)}")

                if "funding_rate" in data:
                    fr_result = data["funding_rate"]
                    print(f"ファンディングレート結果: {fr_result}")
                    print(f"ファンディングレート結果の型: {type(fr_result)}")
                else:
                    print("❌ dataにfunding_rateキーがありません")
            else:
                print("❌ 結果にdataキーがありません")

            # 結果全体をJSONで出力（デバッグ用）
            try:
                result_json = json.dumps(result, indent=2, default=str)
                print(f"\n=== 結果全体（JSON） ===")
                print(result_json)
            except Exception as e:
                print(f"JSON変換エラー: {e}")

        except Exception as e:
            logger.error(f"履歴サービス結果構造調査中にエラー: {e}")
            raise

    @pytest.mark.asyncio
    async def test_orchestration_service_result_structure(self, mock_repositories):
        """統合サービスの結果構造を詳しく調査"""
        service = DataCollectionOrchestrationService()
        mock_db = Mock(spec=Session)

        logger.info("=== 統合サービス結果構造調査開始 ===")

        # リポジトリクラスをモックに置き換え
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
        ) as mock_ohlcv_class:
            with patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.FundingRateRepository"
            ) as mock_funding_class:
                with patch(
                    "app.services.data_collection.orchestration.data_collection_orchestration_service.OpenInterestRepository"
                ) as mock_oi_class:

                    # モックリポジトリを設定
                    mock_ohlcv_class.return_value = mock_repositories["ohlcv"]
                    mock_funding_class.return_value = mock_repositories["funding_rate"]
                    mock_oi_class.return_value = mock_repositories["open_interest"]

                    try:
                        result = await service.execute_bulk_incremental_update(
                            symbol="BTC/USDT:USDT", db=mock_db
                        )

                        print("\n=== 統合サービス結果構造 ===")
                        print(f"結果のキー: {list(result.keys())}")
                        print(f"結果の型: {type(result)}")

                        if "data" in result:
                            data = result["data"]
                            print(f"dataのキー: {list(data.keys())}")
                            print(f"dataの型: {type(data)}")

                            # dataの中身を詳しく調査
                            for key, value in data.items():
                                print(f"data['{key}'] = {value}")
                                print(f"data['{key}']の型 = {type(value)}")

                                if isinstance(value, dict) and "data" in value:
                                    nested_data = value["data"]
                                    print(
                                        f"  ネストされたdata['{key}']['data'] = {nested_data}"
                                    )
                                    if isinstance(nested_data, dict):
                                        print(
                                            f"  ネストされたdataのキー: {list(nested_data.keys())}"
                                        )
                        else:
                            print("❌ 結果にdataキーがありません")

                        # 結果全体をJSONで出力（デバッグ用）
                        try:
                            result_json = json.dumps(result, indent=2, default=str)
                            print(f"\n=== 統合サービス結果全体（JSON） ===")
                            print(result_json)
                        except Exception as e:
                            print(f"JSON変換エラー: {e}")

                    except Exception as e:
                        logger.error(f"統合サービス結果構造調査中にエラー: {e}")
                        raise

    def test_api_response_structure(self):
        """api_response関数の構造を調査"""
        from app.utils.response import api_response

        print("\n=== api_response構造調査 ===")

        # テストデータ
        test_data = {
            "symbol": "BTC/USDT:USDT",
            "data": {
                "funding_rate": {"success": True, "saved_count": 10},
                "open_interest": {"success": True, "saved_count": 5},
            },
            "total_saved_count": 15,
        }

        # api_responseを呼び出し
        response = api_response(
            success=True, message="テストメッセージ", data=test_data
        )

        print(f"api_response結果のキー: {list(response.keys())}")
        print(f"api_response結果の型: {type(response)}")

        if "data" in response:
            data = response["data"]
            print(f"api_response dataのキー: {list(data.keys())}")
            print(f"api_response dataの型: {type(data)}")

            # dataの中身を詳しく調査
            for key, value in data.items():
                print(f"api_response data['{key}'] = {value}")
                print(f"api_response data['{key}']の型 = {type(value)}")

        # 結果全体をJSONで出力
        try:
            response_json = json.dumps(response, indent=2, default=str)
            print(f"\n=== api_response結果全体（JSON） ===")
            print(response_json)
        except Exception as e:
            print(f"JSON変換エラー: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
