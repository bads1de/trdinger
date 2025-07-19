"""
データ収集API

バックテスト用のOHLCVデータ収集エンドポイント
責務の分離により、ビジネスロジックはサービス層に委譲されています。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict

from app.core.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from app.core.utils.unified_error_handler import UnifiedErrorHandler
from database.connection import get_db, ensure_db_initialized
from database.repositories.ohlcv_repository import OHLCVRepository
from app.config.unified_config import unified_config
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data-collection", tags=["data-collection"])


@router.post("/historical")
async def collect_historical_data(
    background_tasks: BackgroundTasks,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    db: Session = Depends(get_db),
) -> Dict:
    """
    履歴データを包括的に収集
    データベースにデータが存在しない場合のみ新規収集を行います。

    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT）
        timeframe: 時間軸（デフォルト: 1h）
        db: データベースセッション

    Returns:
        収集開始レスポンスまたは既存データ情報
    """

    async def _execute():
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            from fastapi import HTTPException

            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # シンボルと時間軸のバリデーション
        # シンボル正規化
        normalized_symbol = unified_config.market.symbol_mapping.get(symbol, symbol)
        if normalized_symbol not in unified_config.market.supported_symbols:
            raise ValueError(f"サポートされていないシンボル: {symbol}")

        # 時間軸検証
        if timeframe not in unified_config.market.supported_timeframes:
            raise ValueError(f"無効な時間軸: {timeframe}")

        # サービス層に委譲
        orchestration_service = DataCollectionOrchestrationService()
        return await orchestration_service.start_historical_data_collection(
            normalized_symbol, timeframe, background_tasks, db
        )

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/bulk-incremental-update")
async def update_bulk_incremental_data(
    symbol: str = "BTC/USDT:USDT", db: Session = Depends(get_db)
) -> Dict:
    """
    一括差分データを更新（OHLCV、FR、OI、外部市場データ）

    OHLCVは全時間足（15m, 30m, 1h, 4h, 1d）を自動的に処理します。
    外部市場データ（SP500、NASDAQ、DXY、VIX）も同時に更新されます。

    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT:USDT）
        db: データベースセッション

    Returns:
        一括差分更新結果
    """

    async def _execute():
        # サービス層に委譲
        orchestration_service = DataCollectionOrchestrationService()
        return await orchestration_service.execute_bulk_incremental_update(symbol, db)

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/bitcoin-full")
async def collect_bitcoin_full_data(
    background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> Dict:
    """
    ビットコインの全時間軸データを収集（ベータ版機能）

    Args:
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        収集開始レスポンス
    """

    async def _execute():
        # サービス層に委譲
        orchestration_service = DataCollectionOrchestrationService()
        return await orchestration_service.start_bitcoin_full_data_collection(
            background_tasks, db
        )

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/bulk-historical")
async def collect_bulk_historical_data(
    background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> Dict:
    """
    全ての取引ペアと全ての時間軸でOHLCVデータを一括収集

    既存データをチェックし、データが存在しない組み合わせのみ収集を実行します。

    Args:
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        一括収集開始レスポンス
    """

    async def _execute():
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # サービス層に委譲
        orchestration_service = DataCollectionOrchestrationService()
        return await orchestration_service.start_bulk_historical_data_collection(
            background_tasks, db
        )

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.get("/status/{symbol:path}/{timeframe}")
async def get_collection_status(
    symbol: str,
    timeframe: str,
    background_tasks: BackgroundTasks,
    auto_fetch: bool = False,
    db: Session = Depends(get_db),
) -> Dict:
    """
    データ収集状況を確認

    Args:
        symbol: 取引ペア
        timeframe: 時間軸
        auto_fetch: データが存在しない場合に自動フェッチを開始するか
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        データ収集状況
    """

    async def _get_collection_status():
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました", exc_info=True)
            from fastapi import HTTPException

            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # シンボルと時間軸のバリデーション
        # シンボル正規化
        normalized_symbol = unified_config.market.symbol_mapping.get(symbol, symbol)
        if normalized_symbol not in unified_config.market.supported_symbols:
            raise ValueError(f"サポートされていないシンボル: {symbol}")

        # 時間軸検証
        if timeframe not in unified_config.market.supported_timeframes:
            raise ValueError(f"無効な時間軸: {timeframe}")

        repository = OHLCVRepository(db)

        # 正規化されたシンボルでデータ件数を取得
        data_count = repository.get_data_count(normalized_symbol, timeframe)

        # データが存在しない場合の処理
        if data_count == 0:
            if auto_fetch and background_tasks:
                # 自動フェッチを開始
                orchestration_service = DataCollectionOrchestrationService()
                await orchestration_service.start_historical_data_collection(
                    normalized_symbol, timeframe, background_tasks, db
                )
                logger.info(f"自動フェッチを開始: {normalized_symbol} {timeframe}")

                return APIResponseHelper.api_response(
                    success=True,
                    message=f"{normalized_symbol} {timeframe} のデータが存在しないため、自動収集を開始しました。",
                    status="auto_fetch_started",
                    data={
                        "symbol": normalized_symbol,
                        "original_symbol": symbol,
                        "timeframe": timeframe,
                        "data_count": 0,
                    },
                )
            else:
                # フェッチを提案
                return APIResponseHelper.api_response(
                    success=True,
                    message=f"{normalized_symbol} {timeframe} のデータが存在しません。新規収集が必要です。",
                    status="no_data",
                    data={
                        "symbol": normalized_symbol,
                        "original_symbol": symbol,
                        "timeframe": timeframe,
                        "data_count": 0,
                        "suggestion": {
                            "manual_fetch": f"/api/data-collection/historical?symbol={normalized_symbol}&timeframe={timeframe}",
                            "auto_fetch": f"/api/data-collection/status/{symbol}/{timeframe}?auto_fetch=true",
                        },
                    },
                )

        # 最新・最古タイムスタンプを取得
        latest_timestamp = repository.get_latest_timestamp(normalized_symbol, timeframe)
        oldest_timestamp = repository.get_oldest_timestamp(normalized_symbol, timeframe)

        return APIResponseHelper.api_response(
            success=True,
            message="データ収集状況を取得しました。",
            data={
                "symbol": normalized_symbol,
                "original_symbol": symbol,
                "timeframe": timeframe,
                "data_count": data_count,
                "status": "data_exists",
                "latest_timestamp": (
                    latest_timestamp.isoformat() if latest_timestamp else None
                ),
                "oldest_timestamp": (
                    oldest_timestamp.isoformat() if oldest_timestamp else None
                ),
            },
        )

    return await UnifiedErrorHandler.safe_execute_async(_get_collection_status)


@router.post("/all/bulk-collect")
async def collect_all_data_bulk(
    background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> Dict:
    """
    全データ（OHLCV・Funding Rate・Open Interest）を一括収集

    既存データをチェックし、データが存在しない組み合わせのみ収集を実行します。

    Args:
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        全データ一括収集開始レスポンス
    """

    async def _execute():
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # サービス層に委譲
        orchestration_service = DataCollectionOrchestrationService()
        return await orchestration_service.start_all_data_bulk_collection(
            background_tasks, db
        )

    return await UnifiedErrorHandler.safe_execute_async(_execute)
