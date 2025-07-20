"""
データ収集統合管理サービス

APIルーター内に散在していたデータ収集関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks

from ..historical.historical_data_service import HistoricalDataService

from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)


class DataCollectionOrchestrationService:
    """
    データ収集統合管理サービス

    各種データ収集の統合管理、バックグラウンドタスクの管理、
    データ存在チェック、収集結果の統一的な処理を担当します。
    """

    def __init__(self):
        """初期化"""
        self.historical_service = HistoricalDataService()

    async def start_historical_data_collection(
        self,
        symbol: str,
        timeframe: str,
        background_tasks: BackgroundTasks,
        db: Session,
    ) -> Dict[str, Any]:
        """
        履歴データ収集を開始

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            background_tasks: バックグラウンドタスク
            db: データベースセッション

        Returns:
            収集開始結果
        """
        try:
            # シンボル正規化
            normalized_symbol = self._normalize_symbol(symbol)

            # データ存在チェック
            repository = OHLCVRepository(db)
            data_exists = repository.get_data_count(normalized_symbol, timeframe) > 0

            if data_exists:
                logger.info(
                    f"{normalized_symbol} {timeframe} のデータは既にデータベースに存在します。"
                )
                return APIResponseHelper.api_response(
                    success=True,
                    message=f"{normalized_symbol} {timeframe} のデータは既に存在します。新規収集は行いません。",
                    status="exists",
                )

            # バックグラウンドタスクとして実行
            background_tasks.add_task(
                self._collect_historical_background,
                normalized_symbol,
                timeframe,
                db,
            )

            return APIResponseHelper.api_response(
                success=True,
                message=f"{normalized_symbol} {timeframe} の履歴データ収集を開始しました",
                status="started",
            )

        except Exception as e:
            logger.error("履歴データ収集開始エラー", e)
            raise

    async def execute_bulk_incremental_update(
        self, symbol: str, db: Session
    ) -> Dict[str, Any]:
        """
        一括差分更新を実行

        Args:
            symbol: 取引ペア
            db: データベースセッションexc_info=True

        Returns:
            一括差分更新結果
        """
        try:
            ohlcv_repository = OHLCVRepository(db)
            funding_rate_repository = FundingRateRepository(db)
            open_interest_repository = OpenInterestRepository(db)

            # 全時間足を自動的に処理
            result = await self.historical_service.collect_bulk_incremental_data(
                symbol=symbol,
                timeframe="1h",  # デフォルト値（実際は全時間足を処理）
                ohlcv_repository=ohlcv_repository,
                funding_rate_repository=funding_rate_repository,
                open_interest_repository=open_interest_repository,
            )

            return APIResponseHelper.api_response(
                success=True,
                message=f"{symbol} の一括差分更新が完了しました",
                data=result,
            )

        except Exception as e:
            logger.error("一括差分更新エラー", e)
            raise

    async def start_bitcoin_full_data_collection(
        self, background_tasks: BackgroundTasks, db: Session
    ) -> Dict[str, Any]:
        """
        ビットコインの全時間軸データ収集を開始

        Args:
            background_tasks: バックグラウンドタスク
            db: データベースセッション

        Returns:
            収集開始レスポンス
        """
        try:
            # 全時間軸でビットコインデータを収集（要求された5つの時間足のみ）
            timeframes = ["15m", "30m", "1h", "4h", "1d"]

            for timeframe in timeframes:
                background_tasks.add_task(
                    self._collect_historical_background, "BTC/USDT:USDT", timeframe, db
                )

            return APIResponseHelper.api_response(
                success=True,
                message="ビットコインの全時間軸データ収集を開始しました",
                data={"timeframes": timeframes},
                status="started",
            )

        except Exception as e:
            logger.error(
                "ビットコイン全データ収集開始エラー",
                e,
            )
            raise

    async def start_bulk_historical_data_collection(
        self, background_tasks: BackgroundTasks, db: Session
    ) -> Dict[str, Any]:
        """
        全ての取引ペアと全ての時間軸でOHLCVデータを一括収集

        Args:
            background_tasks: バックグラウンドタスク
            db: データベースセッション

        Returns:
            収集開始レスポンス
        """
        try:
            # 取引ペアと時間軸の定義
            symbols = [
                "BTC/USDT:USDT",
            ]
            timeframes = ["15m", "30m", "1h", "4h", "1d"]

            # データ存在チェックと収集タスクの追加
            repository = OHLCVRepository(db)
            collection_tasks = []

            for symbol in symbols:
                for timeframe in timeframes:
                    data_count = repository.get_data_count(symbol, timeframe)
                    if data_count == 0:
                        collection_tasks.append((symbol, timeframe))
                        background_tasks.add_task(
                            self._collect_historical_background,
                            symbol,
                            timeframe,
                            db,
                        )

            return APIResponseHelper.api_response(
                success=True,
                message=f"一括履歴データ収集を開始しました（{len(collection_tasks)}件のタスク）",
                data={
                    "symbols": symbols,
                    "timeframes": timeframes,
                    "collection_tasks": len(collection_tasks),
                },
                status="started",
            )

        except Exception as e:
            logger.error("一括履歴データ収集開始エラー", e)
            raise

    async def start_all_data_bulk_collection(
        self, background_tasks: BackgroundTasks, db: Session
    ) -> Dict[str, Any]:
        """
        全データ（OHLCV・Funding Rate・Open Interest）を一括収集

        Args:
            background_tasks: バックグラウンドタスク
            db: データベースセッション

        Returns:
            収集開始レスポンス
        """
        try:
            # 取引ペアと時間軸の定義
            symbols = [
                "BTC/USDT:USDT",
            ]
            timeframes = ["15m", "30m", "1h", "4h", "1d"]

            # データ存在チェックと収集タスクの追加
            ohlcv_repository = OHLCVRepository(db)
            collection_tasks = []

            for symbol in symbols:
                for timeframe in timeframes:
                    # OHLCVデータの存在チェック
                    ohlcv_count = ohlcv_repository.get_data_count(symbol, timeframe)
                    if ohlcv_count == 0:
                        collection_tasks.append((symbol, timeframe))
                        background_tasks.add_task(
                            self._collect_all_data_background,
                            symbol,
                            timeframe,
                            db,
                        )

            return APIResponseHelper.api_response(
                success=True,
                message=f"全データ一括収集を開始しました（{len(collection_tasks)}件のタスク）",
                data={
                    "symbols": symbols,
                    "timeframes": timeframes,
                    "collection_tasks": len(collection_tasks),
                },
                status="started",
            )

        except Exception as e:
            logger.error("全データ一括収集開始エラー", e)
            raise

    def _normalize_symbol(self, symbol: str) -> str:
        """
        シンボルを正規化

        Args:
            symbol: 元のシンボル

        Returns:
            正規化されたシンボル
        """
        if ":" not in symbol:
            return f"{symbol}:USDT"
        return symbol

    async def _collect_historical_background(
        self, symbol: str, timeframe: str, db: Session
    ):
        """バックグラウンドでの履歴データ収集"""
        try:
            logger.info(f"履歴データ収集開始: {symbol} {timeframe}")

            repository = OHLCVRepository(db)
            result = await self.historical_service.collect_historical_data(
                symbol, timeframe, repository
            )

            if result is not None and result >= 0:
                logger.info(
                    f"履歴データ収集完了: {symbol} {timeframe} - {result}件保存"
                )
            else:
                logger.error(f"履歴データ収集失敗: {symbol} {timeframe}")

        except Exception as e:
            logger.error(
                f"履歴データ収集中にエラーが発生しました: {symbol} {timeframe}", e
            )

    async def _collect_all_data_background(
        self, symbol: str, timeframe: str, db: Session
    ):
        """バックグラウンドでの全データ収集（OHLCV・FR・OI・TI）"""
        try:
            logger.info(f"全データ収集開始: {symbol} {timeframe}")

            # 1. OHLCVデータ収集
            logger.info(f"OHLCV収集開始: {symbol} {timeframe}")
            ohlcv_repository = OHLCVRepository(db)

            ohlcv_result = await self.historical_service.collect_historical_data(
                symbol, timeframe, ohlcv_repository
            )

            if ohlcv_result is not None and ohlcv_result >= 0:
                logger.info(
                    f"OHLCV収集完了: {symbol} {timeframe} - {ohlcv_result}件保存"
                )
            else:
                logger.error(f"OHLCV収集失敗: {symbol} {timeframe}")
                return

            # 2. Funding Rate収集
            try:
                logger.info(f"Funding Rate収集開始: {symbol} {timeframe}")
                from ..bybit.funding_rate_service import BybitFundingRateService

                funding_service = BybitFundingRateService()
                funding_repository = FundingRateRepository(db)

                funding_result = await funding_service.fetch_and_save_funding_rate_data(
                    symbol=symbol, repository=funding_repository, fetch_all=True
                )

                if funding_result["success"]:
                    logger.info(
                        f"Funding Rate収集完了: {symbol} - {funding_result['saved_count']}件保存"
                    )
                else:
                    logger.error(
                        f"Funding Rate収集失敗: {symbol} - {funding_result.get('message')}"
                    )

            except Exception as e:
                logger.error(f"Funding Rate収集エラー: {symbol}", e)

            # 3. Open Interest収集
            try:
                logger.info(f"Open Interest収集開始: {symbol} {timeframe}")
                from ..bybit.open_interest_service import BybitOpenInterestService

                oi_service = BybitOpenInterestService()
                oi_repository = OpenInterestRepository(db)

                oi_result = await oi_service.fetch_and_save_open_interest_data(
                    symbol=symbol,
                    repository=oi_repository,
                    fetch_all=True,
                    interval=timeframe,
                )

                if oi_result["success"]:
                    logger.info(
                        f"Open Interest収集完了: {symbol} {timeframe} - {oi_result['saved_count']}件保存"
                    )
                else:
                    logger.error(
                        f"Open Interest収集失敗: {symbol} {timeframe} - {oi_result.get('message')}"
                    )

            except Exception as e:
                logger.error(f"Open Interest収集エラー: {symbol} {timeframe}", e)

            logger.info(f"全データ収集完了: {symbol} {timeframe}")

        except Exception as e:
            logger.error(
                f"全データ収集中にエラーが発生しました: {symbol} {timeframe}",
                e,
            )
        finally:
            # データベースセッションのクリーンアップ
            if hasattr(db, "close"):
                db.close()
