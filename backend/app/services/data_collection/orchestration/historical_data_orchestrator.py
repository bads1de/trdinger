"""
履歴データ収集オーケストレーターモジュール

OHLCV履歴データの収集を担当します。
"""

import logging
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.utils.response import api_response
from database.repositories.ohlcv_repository import OHLCVRepository

from ..historical.historical_data_service import HistoricalDataService

logger = logging.getLogger(__name__)


class HistoricalDataOrchestrator:
    """
    履歴データ収集オーケストレーター

    OHLCV履歴データの収集を管理します。
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
        force_update: bool = False,
        start_date: Optional[str] = None,
        data_validator: Any = None,
        ohlcv_repository_class: Any = None,
    ) -> Dict[str, Any]:
        """
        特定のシンボルと時間軸の過去価格データ（OHLCV）を収集

        データベースにデータが存在しない場合に、バックグラウンドタスクとして
        Bybit等の取引所から全履歴データの取得を開始します。
        `force_update=True` の場合は既存データをクリアして再取得します。

        Args:
            symbol: 取引ペア（例: BTC/USDT:USDT）
            timeframe: 時間軸（1m, 1h, 1d等）
            background_tasks: 非同期実行のためのバックグラウンドタスク管理基盤
            db: データベースセッション
            force_update: Trueの場合、既存データを削除して最初から収集し直す
            start_date: 収集開始日。未指定時はシステムデフォルト（通常 2020-03-25）
            data_validator: データバリデーター

        Returns:
            収集タスクのステータス（started, exists等）を含むレスポンス辞書
        """
        repository_class = ohlcv_repository_class or OHLCVRepository

        # シンボルと時間軸のバリデーション
        if data_validator:
            normalized_symbol = data_validator.validate_symbol_and_timeframe(symbol, timeframe)
        else:
            # デフォルトのバリデーション（テスト用）
            from app.config.unified_config import unified_config
            normalized_symbol = unified_config.market.symbol_mapping.get(symbol, symbol)
            if normalized_symbol not in unified_config.market.supported_symbols:
                raise ValueError(f"サポートされていないシンボル: {symbol}")
            if timeframe not in unified_config.market.supported_timeframes:
                raise ValueError(f"無効な時間軸: {timeframe}")

        # データ存在チェック
        repository = repository_class(db)
        data_exists = repository.get_data_count(normalized_symbol, timeframe) > 0

        if data_exists and not force_update:
            logger.info(
                f"{normalized_symbol} {timeframe} のデータは既にデータベースに存在します。"
            )
            return api_response(
                success=True,
                message=f"{normalized_symbol} {timeframe} のデータは既に存在します。新規収集は行いません。",
                status="exists",
            )

        if data_exists and force_update:
            logger.info(f"{normalized_symbol} {timeframe} のデータを強制更新します。")
            # 既存データを削除
            deleted_count = repository.clear_ohlcv_data_by_symbol_and_timeframe(
                normalized_symbol, timeframe
            )
            logger.info(f"既存データを{deleted_count}件削除しました。")

        # バックグラウンドタスクとして実行
        background_tasks.add_task(
            self._collect_historical_background,
            normalized_symbol,
            timeframe,
            db,
            start_date,
            repository_class,
        )

        status_message = (
            f"{normalized_symbol} {timeframe} の履歴データ収集を開始しました"
        )
        if force_update:
            status_message += "（強制更新モード）"

        return api_response(
            success=True,
            message=status_message,
            status="started",
        )

    async def _collect_historical_background(
        self,
        symbol: str,
        timeframe: str,
        db: Session,
        start_date: Optional[str] = None,
        ohlcv_repository_class: Any = None,
    ):
        """バックグラウンドでの履歴データ収集（ページネーションで全期間取得）"""
        try:
            logger.info(f"履歴データ収集開始: {symbol} {timeframe}")

            repository_class = ohlcv_repository_class or OHLCVRepository
            repository = repository_class(db)

            logger.info("ページネーションで全期間データを取得します")

            result = (
                await self.historical_service.collect_historical_data_with_start_date(
                    symbol,
                    timeframe,
                    repository,
                )
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
