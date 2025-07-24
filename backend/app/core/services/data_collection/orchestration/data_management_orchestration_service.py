"""
データ管理統合管理サービス

APIルーター内に散在していたデータ削除・管理関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from contextlib import contextmanager

from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from app.core.utils.api_utils import APIResponseHelper
from database.connection import SessionLocal
from database.models import (
    OHLCVData,
    FundingRateData,
    OpenInterestData,
    FearGreedIndexData,
)

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

    @contextmanager
    def _get_db_session(self, db_session: Optional[Session] = None):
        """
        データベースセッションを取得するコンテキストマネージャ

        Args:
            db_session: 既存のセッション（Noneの場合は新規作成）

        Yields:
            Session: データベースセッション
        """
        if db_session is not None:
            yield db_session
        else:
            session = SessionLocal()
            try:
                yield session
            finally:
                session.close()

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

            with self._get_db_session(db_session) as session:
                ohlcv_repo = OHLCVRepository(session)
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

            with self._get_db_session(db_session) as session:
                fr_repo = FundingRateRepository(session)
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

            with self._get_db_session(db_session) as session:
                oi_repo = OpenInterestRepository(session)
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

    async def get_data_status(
        self, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
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
                fg_repo = FearGreedIndexRepository(session)

                # 各データの件数取得
                ohlcv_count = session.query(OHLCVData).count()
                fr_count = session.query(FundingRateData).count()
                oi_count = session.query(OpenInterestData).count()
                fg_count = session.query(FearGreedIndexData).count()

                # OHLCV詳細情報（時間足別）
                timeframes = ["15m", "30m", "1h", "4h", "1d"]
                symbol = "BTC/USDT:USDT"

                ohlcv_details = {}
                for tf in timeframes:
                    count = ohlcv_repo.get_data_count(symbol, tf)
                    latest = ohlcv_repo.get_latest_timestamp(symbol, tf)
                    oldest = ohlcv_repo.get_oldest_timestamp(symbol, tf)

                    ohlcv_details[tf] = {
                        "count": count,
                        "latest_timestamp": latest.isoformat() if latest else None,
                        "oldest_timestamp": oldest.isoformat() if oldest else None,
                    }

                # ファンディングレート詳細情報
                fr_latest = fr_repo.get_latest_funding_timestamp(symbol)
                fr_oldest = fr_repo.get_oldest_funding_timestamp(symbol)

                # オープンインタレスト詳細情報
                oi_latest = oi_repo.get_latest_open_interest_timestamp(symbol)
                oi_oldest = oi_repo.get_oldest_open_interest_timestamp(symbol)

                # Fear & Greed Index詳細情報
                fg_latest = fg_repo.get_latest_data_timestamp()
                fg_oldest = fg_repo.get_data_range().get("oldest_data")
                if fg_oldest:
                    from datetime import datetime

                    fg_oldest = datetime.fromisoformat(fg_oldest.replace("Z", "+00:00"))

                response_data = {
                    "data_counts": {
                        "ohlcv": ohlcv_count,
                        "funding_rates": fr_count,
                        "open_interest": oi_count,
                        "fear_greed_index": fg_count,
                    },
                    "total_records": ohlcv_count + fr_count + oi_count + fg_count,
                    "details": {
                        "ohlcv": {
                            "symbol": symbol,
                            "timeframes": ohlcv_details,
                            "total_count": ohlcv_count,
                        },
                        "funding_rates": {
                            "symbol": symbol,
                            "count": fr_count,
                            "latest_timestamp": (
                                fr_latest.isoformat() if fr_latest else None
                            ),
                            "oldest_timestamp": (
                                fr_oldest.isoformat() if fr_oldest else None
                            ),
                        },
                        "open_interest": {
                            "symbol": symbol,
                            "count": oi_count,
                            "latest_timestamp": (
                                oi_latest.isoformat() if oi_latest else None
                            ),
                            "oldest_timestamp": (
                                oi_oldest.isoformat() if oi_oldest else None
                            ),
                        },
                        "fear_greed_index": {
                            "count": fg_count,
                            "latest_timestamp": (
                                fg_latest.isoformat() if fg_latest else None
                            ),
                            "oldest_timestamp": (
                                fg_oldest.isoformat() if fg_oldest else None
                            ),
                        },
                    },
                }

                return APIResponseHelper.api_response(
                    success=True,
                    data=response_data,
                    message="現在のデータ状況を取得しました",
                )
        except Exception as e:
            logger.error(f"データステータス取得エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"データステータス取得中にエラーが発生しました: {str(e)}",
                data={"error": str(e)},
            )
