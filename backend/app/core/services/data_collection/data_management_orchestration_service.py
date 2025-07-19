"""
データ管理統合管理サービス

APIルーター内に散在していたデータ削除・管理関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)


class DataManagementOrchestrationService:
    """
    データ管理統合管理サービス

    各種データの削除、リセット、管理等の
    統一的な処理を担当します。APIルーターからビジネスロジックを分離し、
    責務を明確化します。
    """

    def __init__(self):
        """初期化"""
        pass

    async def reset_all_data(
        self, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        全データのリセット

        Args:
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        try:
            logger.info("全データリセット開始")

            # リポジトリインスタンス作成
            ohlcv_repo = OHLCVRepository(db_session)
            fr_repo = FundingRateRepository(db_session)
            oi_repo = OpenInterestRepository(db_session)

            # 各データの削除実行
            deleted_counts = {}
            errors = []

            # OHLCVデータ削除
            try:
                deleted_counts["ohlcv"] = ohlcv_repo.clear_all_ohlcv_data()
            except Exception as e:
                errors.append(f"OHLCV削除エラー: {str(e)}")
                deleted_counts["ohlcv"] = 0

            # ファンディングレートデータ削除
            try:
                deleted_counts["funding_rates"] = fr_repo.clear_all_funding_rate_data()
            except Exception as e:
                errors.append(f"ファンディングレート削除エラー: {str(e)}")
                deleted_counts["funding_rates"] = 0

            # オープンインタレストデータ削除
            try:
                deleted_counts["open_interest"] = oi_repo.clear_all_open_interest_data()
            except Exception as e:
                errors.append(f"オープンインタレスト削除エラー: {str(e)}")
                deleted_counts["open_interest"] = 0

            # 結果の集計
            total_deleted = sum(deleted_counts.values())
            success = len(errors) == 0

            response_data = {
                "deleted_counts": deleted_counts,
                "total_deleted": total_deleted,
                "errors": errors,
                "timestamp": datetime.now().isoformat(),
            }

            message = (
                "全データのリセットが完了しました"
                if success
                else "一部のデータリセットでエラーが発生しました"
            )

            logger.info(f"全データリセット完了: {deleted_counts}")
            return APIResponseHelper.api_response(
                success=success,
                message=message,
                data=response_data,
            )

        except Exception as e:
            logger.error(f"全データリセットエラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"全データリセット中にエラーが発生しました: {str(e)}",
                data={
                    "deleted_counts": {},
                    "total_deleted": 0,
                    "errors": [str(e)],
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def reset_ohlcv_data(
        self, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        OHLCVデータのリセット

        Args:
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        try:
            logger.info("OHLCVデータリセット開始")

            ohlcv_repo = OHLCVRepository(db_session)
            deleted_count = ohlcv_repo.clear_all_ohlcv_data()

            response_data = {
                "deleted_count": deleted_count,
                "data_type": "ohlcv",
                "timestamp": datetime.now().isoformat(),
            }

            message = f"OHLCVデータを{deleted_count}件削除しました"

            logger.info(f"OHLCVデータリセット完了: {deleted_count}件")
            return APIResponseHelper.api_response(
                success=True,
                message=message,
                data=response_data,
            )

        except Exception as e:
            logger.error(f"OHLCVデータリセットエラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"OHLCVデータリセット中にエラーが発生しました: {str(e)}",
                data={
                    "deleted_count": 0,
                    "data_type": "ohlcv",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def reset_funding_rate_data(
        self, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        ファンディングレートデータのリセット

        Args:
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        try:
            logger.info("ファンディングレートデータリセット開始")

            fr_repo = FundingRateRepository(db_session)
            deleted_count = fr_repo.clear_all_funding_rate_data()

            response_data = {
                "deleted_count": deleted_count,
                "data_type": "funding_rates",
                "timestamp": datetime.now().isoformat(),
            }

            message = f"ファンディングレートデータを{deleted_count}件削除しました"

            logger.info(f"ファンディングレートデータリセット完了: {deleted_count}件")
            return APIResponseHelper.api_response(
                success=True,
                message=message,
                data=response_data,
            )

        except Exception as e:
            logger.error(
                f"ファンディングレートデータリセットエラー: {e}", exc_info=True
            )
            return APIResponseHelper.api_response(
                success=False,
                message=f"ファンディングレートデータリセット中にエラーが発生しました: {str(e)}",
                data={
                    "deleted_count": 0,
                    "data_type": "funding_rates",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def reset_open_interest_data(
        self, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        オープンインタレストデータのリセット

        Args:
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        try:
            logger.info("オープンインタレストデータリセット開始")

            oi_repo = OpenInterestRepository(db_session)
            deleted_count = oi_repo.clear_all_open_interest_data()

            response_data = {
                "deleted_count": deleted_count,
                "data_type": "open_interest",
                "timestamp": datetime.now().isoformat(),
            }

            message = f"オープンインタレストデータを{deleted_count}件削除しました"

            logger.info(f"オープンインタレストデータリセット完了: {deleted_count}件")
            return APIResponseHelper.api_response(
                success=True,
                message=message,
                data=response_data,
            )

        except Exception as e:
            logger.error(
                f"オープンインタレストデータリセットエラー: {e}", exc_info=True
            )
            return APIResponseHelper.api_response(
                success=False,
                message=f"オープンインタレストデータリセット中にエラーが発生しました: {str(e)}",
                data={
                    "deleted_count": 0,
                    "data_type": "open_interest",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def reset_data_by_symbol(
        self, symbol: str, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        シンボル別データのリセット

        Args:
            symbol: 取引ペア
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        try:
            logger.info(f"シンボル別データリセット開始: {symbol}")

            ohlcv_repo = OHLCVRepository(db_session)
            fr_repo = FundingRateRepository(db_session)
            oi_repo = OpenInterestRepository(db_session)

            deleted_counts = {}
            errors = []

            try:
                deleted_counts["ohlcv"] = ohlcv_repo.clear_ohlcv_data_by_symbol(symbol)
            except Exception as e:
                errors.append(f"OHLCV削除エラー: {str(e)}")
                deleted_counts["ohlcv"] = 0

            try:
                deleted_counts["funding_rates"] = (
                    fr_repo.clear_funding_rate_data_by_symbol(symbol)
                )
            except Exception as e:
                errors.append(f"ファンディングレート削除エラー: {str(e)}")
                deleted_counts["funding_rates"] = 0

            try:
                deleted_counts["open_interest"] = (
                    oi_repo.clear_open_interest_data_by_symbol(symbol)
                )
            except Exception as e:
                errors.append(f"オープンインタレスト削除エラー: {str(e)}")
                deleted_counts["open_interest"] = 0

            total_deleted = sum(deleted_counts.values())
            success = len(errors) == 0

            message = (
                f"シンボル '{symbol}' のデータリセットが完了しました"
                if success
                else f"シンボル '{symbol}' の一部データリセットでエラーが発生しました"
            )

            logger.info(f"シンボル '{symbol}' データリセット完了: {deleted_counts}")
            return APIResponseHelper.api_response(
                success=success,
                message=message,
                data={
                    "symbol": symbol,
                    "deleted_counts": deleted_counts,
                    "total_deleted": total_deleted,
                    "errors": errors,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"シンボル別データリセットエラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"シンボル別データリセット中にエラーが発生しました: {str(e)}",
                data={
                    "symbol": symbol,
                    "deleted_counts": {},
                    "total_deleted": 0,
                    "errors": [str(e)],
                    "timestamp": datetime.now().isoformat(),
                },
            )
