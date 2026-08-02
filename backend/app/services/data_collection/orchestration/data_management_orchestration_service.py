"""
データ管理統合管理サービス

APIルーター内に散在していたデータ削除・管理関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.config.constants import DEFAULT_MARKET_SYMBOL
from app.services.data_collection.orchestration.base_orchestration_service import (
    BaseDataCollectionOrchestrationService,
)
from app.utils.datetime_utils import isoformat_or_none
from app.utils.response import api_response, error_response, now_iso
from database.models import (
    FundingRateData,
    OHLCVData,
    OpenInterestData,
)
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import (
    OpenInterestRepository,
)

logger = logging.getLogger(__name__)


class DataManagementOrchestrationService(BaseDataCollectionOrchestrationService):
    """
    データ管理統合管理サービス

    各種データの削除、リセット、管理等の
    統一的な処理を担当します。APIルーターからビジネスロジックを分離し、
    責務を明確化します。DBセッション管理は基底クラスを継承します。
    """

    def _reset_single_data_type(
        self,
        db_session: Session | None,
        data_type: str,
        display_name: str,
        delete_operation: Callable[[Session], int],
    ) -> dict[str, Any]:
        """
        単一データ種別の削除結果を共通形式で返す

        Args:
            db_session: データベースセッション
            data_type: レスポンスに含めるデータ種別キー
            display_name: ログとメッセージに使う表示名
            delete_operation: 削除処理を行う関数

        Returns:
            削除結果を含む API レスポンス
        """
        try:
            logger.info(f"{display_name}データリセット開始")

            with self._get_db_session(db_session) as session:
                deleted_count = delete_operation(session)

            response_data = {
                "deleted_count": deleted_count,
                "data_type": data_type,
                "timestamp": now_iso(),
            }
            message = f"{display_name}データを{deleted_count}件削除しました"

            logger.info(f"{display_name}データリセット完了: {deleted_count}件")
            return api_response(
                success=True,
                message=message,
                data=response_data,
            )

        except Exception as e:
            logger.error(f"{display_name}データリセットエラー: {e}", exc_info=True)
            return error_response(
                message=f"{display_name}データリセット中にエラーが発生しました: {str(e)}",
                details={
                    "deleted_count": 0,
                    "data_type": data_type,
                    "error": str(e),
                },
            )

    async def reset_all_data(self, db_session: Session | None = None) -> dict[str, Any]:
        """
        全データのリセット

        Args:
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        try:
            logger.info("全データリセット開始")

            # 各データの削除実行
            deleted_counts = {}
            errors = []

            with self._get_db_session(db_session) as session:
                # リポジトリインスタンス作成
                ohlcv_repo = OHLCVRepository(session)
                fr_repo = FundingRateRepository(session)
                oi_repo = OpenInterestRepository(session)

                # OHLCVデータ削除
                try:
                    deleted_counts["ohlcv"] = ohlcv_repo.clear_all_ohlcv_data()
                except Exception as e:
                    errors.append(f"OHLCV削除エラー: {str(e)}")
                    deleted_counts["ohlcv"] = 0

                # ファンディングレートデータ削除
                try:
                    deleted_counts["funding_rates"] = (
                        fr_repo.clear_all_funding_rate_data()
                    )
                except Exception as e:
                    errors.append(f"ファンディングレート削除エラー: {str(e)}")
                    deleted_counts["funding_rates"] = 0

                # オープンインタレストデータ削除
                try:
                    deleted_counts["open_interest"] = (
                        oi_repo.clear_all_open_interest_data()
                    )
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
                "timestamp": now_iso(),
            }

            message = (
                "全データのリセットが完了しました"
                if success
                else "一部のデータリセットでエラーが発生しました"
            )

            logger.info(f"全データリセット完了: {deleted_counts}")
            return api_response(
                success=success,
                message=message,
                data=response_data,
            )

        except Exception as e:
            logger.error(f"全データリセットエラー: {e}", exc_info=True)
            return error_response(
                message=f"全データリセット中にエラーが発生しました: {str(e)}",
                details={
                    "deleted_counts": {},
                    "total_deleted": 0,
                    "errors": [str(e)],
                },
            )

    async def reset_ohlcv_data(
        self, db_session: Session | None = None
    ) -> dict[str, Any]:
        """
        OHLCVデータのリセット

        Args:
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        return self._reset_single_data_type(
            db_session=db_session,
            data_type="ohlcv",
            display_name="OHLCV",
            delete_operation=lambda session: OHLCVRepository(
                session
            ).clear_all_ohlcv_data(),
        )

    async def reset_funding_rate_data(
        self, db_session: Session | None = None
    ) -> dict[str, Any]:
        """
        ファンディングレートデータのリセット

        Args:
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        return self._reset_single_data_type(
            db_session=db_session,
            data_type="funding_rates",
            display_name="ファンディングレート",
            delete_operation=lambda session: FundingRateRepository(
                session
            ).clear_all_funding_rate_data(),
        )

    async def reset_open_interest_data(
        self, db_session: Session | None = None
    ) -> dict[str, Any]:
        """
        オープンインタレストデータのリセット

        Args:
            db_session: データベースセッション

        Returns:
            リセット結果を含む辞書
        """
        return self._reset_single_data_type(
            db_session=db_session,
            data_type="open_interest",
            display_name="オープンインタレスト",
            delete_operation=lambda session: OpenInterestRepository(
                session
            ).clear_all_open_interest_data(),
        )

    async def reset_data_by_symbol(
        self, symbol: str, db_session: Session | None = None
    ) -> dict[str, Any]:
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

            deleted_counts = {}
            errors = []

            with self._get_db_session(db_session) as session:
                ohlcv_repo = OHLCVRepository(session)
                fr_repo = FundingRateRepository(session)
                oi_repo = OpenInterestRepository(session)

                try:
                    deleted_counts["ohlcv"] = ohlcv_repo.clear_ohlcv_data_by_symbol(
                        symbol
                    )
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
            return api_response(
                success=success,
                message=message,
                data={
                    "symbol": symbol,
                    "deleted_counts": deleted_counts,
                    "total_deleted": total_deleted,
                    "errors": errors,
                },
            )
        except Exception as e:
            logger.error(f"シンボル別データリセットエラー: {e}", exc_info=True)
            return error_response(
                message=f"シンボル別データリセット中にエラーが発生しました: {str(e)}",
                details={
                    "symbol": symbol,
                    "deleted_counts": {},
                    "total_deleted": 0,
                    "errors": [str(e)],
                },
            )

    async def get_data_status(
        self, db_session: Session | None = None
    ) -> dict[str, Any]:
        """
        現在のデータ状況を取得（詳細版）

        Args:
            db_session: データベースセッション

        Returns:
            各データタイプの詳細情報（件数、最新・最古データ）
        """
        try:
            with self._get_db_session(db_session) as session:
                # リポジトリインスタンス作成
                ohlcv_repo = OHLCVRepository(session)
                fr_repo = FundingRateRepository(session)
                oi_repo = OpenInterestRepository(session)

                # 各データの件数取得
                ohlcv_count = session.query(OHLCVData).count()
                fr_count = session.query(FundingRateData).count()
                oi_count = session.query(OpenInterestData).count()

                # OHLCV詳細情報（時間足別）
                from app.config.unified_config import unified_config

                timeframes = unified_config.market.supported_timeframes
                symbol = DEFAULT_MARKET_SYMBOL

                ohlcv_details = {}
                for tf in timeframes:
                    count = ohlcv_repo.get_data_count(symbol, tf)
                    latest = ohlcv_repo.get_latest_timestamp(
                        timestamp_column="timestamp",
                        filter_conditions={"symbol": symbol, "timeframe": tf},
                    )
                    oldest = ohlcv_repo.get_oldest_timestamp(
                        timestamp_column="timestamp",
                        filter_conditions={"symbol": symbol, "timeframe": tf},
                    )

                    ohlcv_details[tf] = {
                        "count": count,
                        "latest_timestamp": isoformat_or_none(latest),
                        "oldest_timestamp": isoformat_or_none(oldest),
                    }

                # ファンディングレート詳細情報
                fr_latest = fr_repo.get_latest_funding_timestamp(symbol)
                fr_oldest = fr_repo.get_oldest_funding_timestamp(symbol)

                # オープンインタレスト詳細情報
                oi_latest = oi_repo.get_latest_open_interest_timestamp(symbol)
                oi_oldest = oi_repo.get_oldest_open_interest_timestamp(symbol)

                response_data = {
                    "data_counts": {
                        "ohlcv": ohlcv_count,
                        "funding_rates": fr_count,
                        "open_interest": oi_count,
                    },
                    "total_records": ohlcv_count + fr_count + oi_count,
                    "details": {
                        "ohlcv": {
                            "symbol": symbol,
                            "timeframes": ohlcv_details,
                            "total_count": ohlcv_count,
                        },
                        "funding_rates": {
                            "symbol": symbol,
                            "count": fr_count,
                            "latest_timestamp": isoformat_or_none(fr_latest),
                            "oldest_timestamp": isoformat_or_none(fr_oldest),
                        },
                        "open_interest": {
                            "symbol": symbol,
                            "count": oi_count,
                            "latest_timestamp": isoformat_or_none(oi_latest),
                            "oldest_timestamp": isoformat_or_none(oi_oldest),
                        },
                    },
                }

                return api_response(
                    success=True,
                    data=response_data,
                    message="現在のデータ状況を取得しました",
                )
        except Exception as e:
            logger.error(f"データステータス取得エラー: {e}", exc_info=True)
            return error_response(
                message=f"データステータス取得中にエラーが発生しました: {str(e)}",
                details={"error": str(e)},
            )
