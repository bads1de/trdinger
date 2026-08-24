"""
一括データ収集オーケストレーターモジュール

一括差分更新・一括収集を担当します。
"""

import logging
from typing import Any

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.config.constants import DEFAULT_MARKET_SYMBOL
from app.config.unified_config import unified_config
from app.utils.data_conversion import normalize_market_symbol
from app.utils.response import api_response
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import (
    OpenInterestRepository,
)

from ..historical.historical_data_service import HistoricalDataService

logger = logging.getLogger(__name__)


class BulkDataOrchestrator:
    """
    一括データ収集オーケストレーター

    一括差分更新・一括収集を管理します。
    """

    def __init__(self) -> None:
        """初期化"""
        self.historical_service = HistoricalDataService()

    async def execute_bulk_incremental_update(
        self, symbol: str, db: Session
    ) -> dict[str, Any]:
        """
        市場全体のデータを最新状態に同期（差分更新オーケストレーション）

        DB 内の既存データ末尾時刻を確認し、現在時刻までの不足分を
        Bybit API 等から取得・補完します。OHLCV の全時間足、
        および FR、OI を一括でインクリメンタル更新します。

        Args:
            symbol: 対象の取引ペア
            db: データベースセッション

        Returns:
            更新が成功した時間軸やデータ種別のサマリーを含むレスポンス
        """
        try:
            ohlcv_repository = OHLCVRepository(db)
            funding_rate_repository = FundingRateRepository(db)
            open_interest_repository = OpenInterestRepository(db)

            # 全時間足を自動的に処理（OHLCV、FR、OI）
            result = await self.historical_service.collect_bulk_incremental_data(
                symbol=symbol,
                timeframe="1h",  # デフォルト値（実際は全時間足を処理）
                ohlcv_repository=ohlcv_repository,
                funding_rate_repository=funding_rate_repository,
                open_interest_repository=open_interest_repository,
            )

            return api_response(
                success=True,
                message=f"{symbol} の一括差分更新が完了しました",
                data=result,  # result全体を返す（data構造を含む）
            )

        except Exception:
            logger.error("一括差分更新エラー", exc_info=True)
            raise

    async def start_bitcoin_full_data_collection(
        self,
        background_tasks: BackgroundTasks,
        db: Session,
        historical_orchestrator: Any = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """
        指定シンボルの全時間軸・全期間データの収集を予約

        システムがサポートする全時間足（1m から 1d まで）に対して
        非同期の収集タスクを発行します。symbol 未指定時は
        デフォルトシンボル（BTC/USDT:USDT）を対象とします。

        Args:
            background_tasks: FastAPI のバックグラウンドタスク管理
            db: データベースセッション
            historical_orchestrator: 履歴データオーケストレーター
            symbol: 収集対象シンボル（例: ETH/USDT:USDT）

        Returns:
            予約された全タスクの情報を保持するレスポンス
        """
        try:
            target_symbol = self._resolve_collect_symbol(symbol)
            # 全時間軸で対象シンボルのデータを収集
            timeframes = unified_config.market.supported_timeframes

            for timeframe in timeframes:
                background_tasks.add_task(
                    historical_orchestrator._collect_historical_background,
                    target_symbol,
                    timeframe,
                    db,
                )

            return api_response(
                success=True,
                message=f"{target_symbol} の全時間軸データ収集を開始しました",
                data={"symbol": target_symbol, "timeframes": timeframes},
                status="started",
            )

        except Exception as e:
            logger.exception("全データ収集開始エラー: %s", e)
            raise

    @staticmethod
    def _resolve_collect_symbol(symbol: str | None) -> str:
        """収集対象シンボルを解決し、ホワイトリスト検証を行う。"""
        candidate = (symbol or DEFAULT_MARKET_SYMBOL).strip()
        normalized = normalize_market_symbol(candidate)
        if normalized not in unified_config.market.supported_symbols:
            supported = ", ".join(unified_config.market.supported_symbols)
            raise ValueError(f"未対応のシンボルです: {candidate} (対応: {supported})")
        return normalized

    async def start_bulk_historical_data_collection(
        self,
        background_tasks: BackgroundTasks,
        db: Session,
        force_update: bool = False,
        start_date: str | None = None,
        historical_orchestrator: Any = None,
    ) -> dict[str, Any]:
        """
        サポートされている全シンボル・全時間軸の履歴データ収集を一括予約

        DB を走査し、データが未取得の箇所について自動的に収集タスクを構築
        および並列実行（バックグラウンド）します。

        Args:
            background_tasks: 非同期タスク管理
            db: データベースセッション
            force_update: 既存データがある場合も削除して再取得するか
            start_date: 全タスクの開始日付（未指定時はデフォルト 2020-03-25）
            historical_orchestrator: 履歴データオーケストレーター

        Returns:
            発行された全タスク数と、対象シンボル・時間軸のリスト
        """
        try:
            # 設定で許可された全シンボル × 全時間軸を対象とする
            symbols = list(unified_config.market.supported_symbols)
            timeframes = unified_config.market.supported_timeframes

            # データ存在チェックと収集タスクの追加
            repository = OHLCVRepository(db)
            collection_tasks = []

            for symbol in symbols:
                for timeframe in timeframes:
                    data_count = repository.get_data_count(symbol, timeframe)

                    # データが存在しない場合、または強制更新が指定されている場合に収集を実行
                    should_collect = data_count == 0 or force_update

                    if should_collect:
                        if force_update and data_count > 0:
                            # 強制更新の場合は既存データを削除
                            deleted_count = (
                                repository.clear_ohlcv_data_by_symbol_and_timeframe(
                                    symbol, timeframe
                                )
                            )
                            logger.info(
                                f"強制更新のため {symbol} {timeframe} の既存データを{deleted_count}件削除しました"
                            )

                        collection_tasks.append((symbol, timeframe))
                        background_tasks.add_task(
                            historical_orchestrator._collect_historical_background,
                            symbol,
                            timeframe,
                            db,
                            start_date,
                        )

            status_message = (
                f"一括履歴データ収集を開始しました（{len(collection_tasks)}件のタスク）"
            )
            if force_update:
                status_message += "（強制更新モード）"

            return api_response(
                success=True,
                message=status_message,
                data={
                    "symbols": symbols,
                    "timeframes": timeframes,
                    "collection_tasks": len(collection_tasks),
                    "force_update": force_update,
                    "start_date": start_date or "2020-03-25",
                },
                status="started",
            )

        except Exception as e:
            logger.error(f"一括履歴データ収集開始エラー: {e}")
            raise

    async def start_all_data_bulk_collection(
        self, background_tasks: BackgroundTasks, db: Session
    ) -> dict[str, Any]:
        """
        全データ（OHLCV・Funding Rate・Open Interest）を一括収集

        Args:
            background_tasks: バックグラウンドタスク
            db: データベースセッション

        Returns:
            収集開始レスポンス
        """
        try:
            # 設定で許可された全シンボル × 全時間軸を対象とする
            symbols = list(unified_config.market.supported_symbols)
            timeframes = unified_config.market.supported_timeframes

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

            return api_response(
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
            logger.error(f"全データ一括収集開始エラー: {e}")
            raise

    async def _collect_all_data_background(
        self, symbol: str, timeframe: str, db: Session
    ) -> None:
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
                from ..bybit.funding_rate_service import (
                    BybitFundingRateService,
                )

                funding_service = BybitFundingRateService()
                funding_repository = FundingRateRepository(db)

                funding_result = await funding_service.fetch_and_save_funding_rate_data(
                    symbol=symbol,
                    repository=funding_repository,
                    fetch_all=True,
                )

                if funding_result["success"]:
                    logger.info(
                        f"Funding Rate収集完了: {symbol} - {funding_result['saved_count']}件保存"
                    )
                else:
                    logger.error(
                        f"Funding Rate収集失敗: {symbol} - {funding_result.get('message')}"
                    )

            except Exception:
                logger.error(f"Funding Rate収集エラー: {symbol}", exc_info=True)

            # 3. Open Interest収集
            try:
                logger.info(f"Open Interest収集開始: {symbol} {timeframe}")
                from ..bybit.open_interest_service import (
                    BybitOpenInterestService,
                )

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

            except Exception:
                logger.error(
                    f"Open Interest収集エラー: {symbol} {timeframe}",
                    exc_info=True,
                )

            logger.info(f"全データ収集完了: {symbol} {timeframe}")

        except Exception:
            logger.exception(
                "全データ収集中にエラーが発生しました: %s %s", symbol, timeframe
            )
        finally:
            # データベースセッションのクリーンアップ
            if hasattr(db, "close"):
                db.close()
