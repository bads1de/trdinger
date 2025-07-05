"""
データ収集API

バックテスト用のOHLCVデータ収集エンドポイント
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict

from app.core.services.data_collection.historical_data_service import (
    HistoricalDataService,
)
from database.connection import get_db, ensure_db_initialized
from database.repositories.ohlcv_repository import OHLCVRepository
from app.config.market_config import MarketDataConfig
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-collection", tags=["data-collection"])


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
    try:
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # シンボルと時間軸のバリデーション (market_config を使用)
        try:
            normalized_symbol = MarketDataConfig.normalize_symbol(symbol)
            if not MarketDataConfig.validate_timeframe(timeframe):
                raise ValueError(f"無効な時間軸: {timeframe}")
        except ValueError as ve:
            logger.warning(
                f"パラメータ検証エラー: {ve} (symbol: {symbol}, timeframe: {timeframe})"
            )
            logger.warning(
                f"パラメータ検証エラー: {ve} (symbol: {symbol}, timeframe: {timeframe})",
                exc_info=True,
            )
            raise HTTPException(status_code=400, detail=str(ve))

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
            _collect_historical_background,
            normalized_symbol,  # 正規化されたシンボルを使用
            timeframe,
            db,
        )

        return APIResponseHelper.api_response(
            success=True,
            message=f"{normalized_symbol} {timeframe} の履歴データ収集を開始しました",
            status="started",
        )

    except Exception as e:
        logger.error("履歴データ収集開始エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="履歴データ収集開始エラー") from e


@router.post("/update")
async def update_incremental_data(
    symbol: str = "BTC/USDT:USDT", timeframe: str = "1h", db: Session = Depends(get_db)
) -> Dict:
    """
    差分データを更新

    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT）
        timeframe: 時間軸（デフォルト: 1h）
        db: データベースセッション

    Returns:
        更新結果
    """
    try:
        service = HistoricalDataService()
        repository = OHLCVRepository(db)

        result = await service.collect_incremental_data(symbol, timeframe, repository)

        if isinstance(result, dict) and result.get("success"):
            return APIResponseHelper.api_response(
                success=True, message="差分データ更新完了", data=result
            )
        else:
            error_message = (
                result.get("message") if isinstance(result, dict) else str(result)
            )
            return APIResponseHelper.api_response(
                success=False,
                message=f"差分データ更新に失敗しました: {error_message}",
                status="failed",
            )

    except Exception as e:
        logger.error("差分データ更新エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="差分データ更新エラー") from e


@router.post("/bitcoin-full")
async def collect_bitcoin_full_data(
    background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> Dict:
    """
    ビットコインの全時間軸データを収集（ベータ版機能）

    Args:
        db: データベースセッション

    Returns:
        収集開始レスポンス
    """
    try:
        # 全時間軸でビットコインデータを収集（要求された5つの時間足のみ）
        timeframes = ["15m", "30m", "1h", "4h", "1d"]

        for timeframe in timeframes:
            background_tasks.add_task(
                _collect_historical_background, "BTC/USDT:USDT", timeframe, db
            )

        return APIResponseHelper.api_response(
            success=True,
            message="ビットコインの全時間軸データ収集を開始しました",
            data={"timeframes": timeframes},
            status="started",
        )

    except Exception as e:
        logger.error("ビットコイン全データ収集開始エラー", exc_info=True)
        raise HTTPException(
            status_code=500, detail="ビットコイン全データ収集開始エラー"
        ) from e


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
    try:
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        from datetime import datetime, timezone

        # サポートされている取引ペアと時間軸（BTC/USDT:USDTのみ）
        symbols = [
            "BTC/USDT:USDT",
        ]

        # 時間軸設定（要求された全ての時間足を収集）
        timeframes = ["15m", "30m", "1h", "4h", "1d"]

        total_combinations = len(symbols) * len(timeframes)
        started_at = datetime.now(timezone.utc).isoformat()

        repository = OHLCVRepository(db)

        # 既存データをチェックして、実際に収集が必要なタスクを特定
        tasks_to_execute = []
        skipped_tasks = []
        failed_tasks = []

        logger.info(
            f"[修正版] 一括データ収集開始: {len(symbols)}個のシンボル × {len(timeframes)}個の時間軸 = {total_combinations}組み合わせを確認"
        )

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # シンボルの正規化
                    normalized_symbol = MarketDataConfig.normalize_symbol(symbol)

                    # 既存データをチェック
                    data_exists = (
                        repository.get_data_count(normalized_symbol, timeframe) > 0
                    )

                    if data_exists:
                        skipped_tasks.append(
                            {
                                "symbol": normalized_symbol,
                                "original_symbol": symbol,
                                "timeframe": timeframe,
                                "reason": "data_exists",
                            }
                        )
                        logger.debug(
                            f"スキップ: {normalized_symbol} {timeframe} - データが既に存在"
                        )
                    else:
                        # データが存在しない場合のみバックグラウンドタスクとして追加
                        background_tasks.add_task(
                            _collect_historical_background,
                            normalized_symbol,
                            timeframe,
                            db,
                        )
                        tasks_to_execute.append(
                            {
                                "symbol": normalized_symbol,
                                "original_symbol": symbol,
                                "timeframe": timeframe,
                            }
                        )
                        logger.info(f"タスク追加: {normalized_symbol} {timeframe}")

                except Exception as task_error:
                    logger.warning(
                        f"タスク処理エラー {symbol} {timeframe}: {task_error}"
                    )
                    failed_tasks.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "error": str(task_error),
                        }
                    )
                    continue

        actual_tasks = len(tasks_to_execute)
        skipped_count = len(skipped_tasks)
        failed_count = len(failed_tasks)

        logger.info("一括データ収集タスク分析完了:")
        logger.info(f"  - 総組み合わせ数: {total_combinations}")
        logger.info(f"  - 実行タスク数: {actual_tasks}")
        logger.info(f"  - スキップ数: {skipped_count} (既存データ)")
        logger.info(f"  - 失敗数: {failed_count}")

        return APIResponseHelper.api_response(
            success=True,
            message=f"一括データ収集を開始しました（{actual_tasks}タスク実行、{skipped_count}タスクスキップ）",
            status="started",
            data={
                "total_combinations": total_combinations,
                "actual_tasks": actual_tasks,
                "skipped_tasks": skipped_count,
                "failed_tasks": failed_count,
                "started_at": started_at,
                "symbols": symbols,
                "timeframes": timeframes,
                "task_details": {
                    "executing": tasks_to_execute,
                    "skipped": skipped_tasks,
                    "failed": failed_tasks,
                },
            },
        )

    except Exception as e:
        logger.error("一括データ収集開始エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="一括データ収集開始エラー") from e


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
    try:
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました", exc_info=True)
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # シンボルと時間軸のバリデーション
        try:
            normalized_symbol = MarketDataConfig.normalize_symbol(symbol)
            if not MarketDataConfig.validate_timeframe(timeframe):
                raise ValueError(f"無効な時間軸: {timeframe}")
        except ValueError as ve:
            logger.warning(
                f"パラメータ検証エラー: {ve} (symbol: {symbol}, timeframe: {timeframe})"
            )
            logger.warning(
                f"パラメータ検証エラー: {ve} (symbol: {symbol}, timeframe: {timeframe})",
                exc_info=True,
            )
            raise HTTPException(status_code=400, detail=str(ve))

        repository = OHLCVRepository(db)

        # 正規化されたシンボルでデータ件数を取得
        data_count = repository.get_data_count(normalized_symbol, timeframe)

        # データが存在しない場合の処理
        if data_count == 0:
            if auto_fetch and background_tasks:
                # 自動フェッチを開始
                background_tasks.add_task(
                    _collect_historical_background,
                    normalized_symbol,
                    timeframe,
                    db,
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

    except Exception as e:
        logger.error("収集状況確認エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="収集状況確認エラー") from e


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
    try:
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        from datetime import datetime, timezone

        # サポートされている取引ペアと時間軸（BTC/USDT:USDTのみ）
        symbols = [
            "BTC/USDT:USDT",
        ]

        # 時間軸設定（要求された全ての時間足を収集）
        timeframes = ["15m", "30m", "1h", "4h", "1d"]

        total_combinations = len(symbols) * len(timeframes)
        started_at = datetime.now(timezone.utc).isoformat()

        repository = OHLCVRepository(db)

        # 既存データをチェックして、実際に収集が必要なタスクを特定
        tasks_to_execute = []
        skipped_tasks = []
        failed_tasks = []

        logger.info(
            f"全データ一括収集開始: {len(symbols)}個のシンボル × {len(timeframes)}個の時間軸 = {total_combinations}組み合わせを確認"
        )

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # シンボルの正規化
                    normalized_symbol = MarketDataConfig.normalize_symbol(symbol)

                    # 既存データをチェック
                    data_exists = (
                        repository.get_data_count(normalized_symbol, timeframe) > 0
                    )

                    if data_exists:
                        skipped_tasks.append(
                            {
                                "symbol": normalized_symbol,
                                "original_symbol": symbol,
                                "timeframe": timeframe,
                                "reason": "data_exists",
                            }
                        )
                        logger.debug(
                            f"スキップ: {normalized_symbol} {timeframe} - データが既に存在"
                        )
                    else:
                        # データが存在しない場合のみバックグラウンドタスクとして追加
                        background_tasks.add_task(
                            _collect_all_data_background,
                            normalized_symbol,
                            timeframe,
                            db,
                        )
                        tasks_to_execute.append(
                            {
                                "symbol": normalized_symbol,
                                "original_symbol": symbol,
                                "timeframe": timeframe,
                            }
                        )
                        logger.info(
                            f"全データタスク追加: {normalized_symbol} {timeframe}"
                        )

                except Exception as task_error:
                    logger.warning(
                        f"タスク処理エラー {symbol} {timeframe}: {task_error}"
                    )
                    failed_tasks.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "error": str(task_error),
                        }
                    )
                    continue

        actual_tasks = len(tasks_to_execute)
        skipped_count = len(skipped_tasks)
        failed_count = len(failed_tasks)

        logger.info("全データ一括収集タスク分析完了:")
        logger.info(f"  - 総組み合わせ数: {total_combinations}")
        logger.info(f"  - 実行タスク数: {actual_tasks}")
        logger.info(f"  - スキップ数: {skipped_count} (既存データ)")
        logger.info(f"  - 失敗数: {failed_count}")

        return APIResponseHelper.api_response(
            success=True,
            message=f"全データ一括収集を開始しました（{actual_tasks}タスク実行、{skipped_count}タスクスキップ）",
            status="started",
            data={
                "total_combinations": total_combinations,
                "actual_tasks": actual_tasks,
                "skipped_tasks": skipped_count,
                "failed_tasks": failed_count,
                "started_at": started_at,
                "symbols": symbols,
                "timeframes": timeframes,
                "task_details": {
                    "executing": tasks_to_execute,
                    "skipped": skipped_tasks,
                    "failed": failed_tasks,
                },
            },
        )

    except Exception as e:
        logger.error("全データ一括収集開始エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="全データ一括収集開始エラー") from e


async def _collect_all_data_background(symbol: str, timeframe: str, db: Session):
    """バックグラウンドでの全データ収集（OHLCV・FR・OI・TI）"""
    try:
        logger.info(f"全データ収集開始: {symbol} {timeframe}")

        # 1. OHLCVデータ収集
        logger.info(f"OHLCV収集開始: {symbol} {timeframe}")
        historical_service = HistoricalDataService()
        ohlcv_repository = OHLCVRepository(db)

        ohlcv_result = await historical_service.collect_historical_data(
            symbol, timeframe, ohlcv_repository
        )

        # ohlcv_resultが辞書型であり、'success'キーがTrueであることを安全にチェック
        if not isinstance(ohlcv_result, dict) or not ohlcv_result.get("success"):
            error_message = (
                ohlcv_result.get("message")
                if isinstance(ohlcv_result, dict)
                else str(ohlcv_result)
            )
            logger.error(
                f"OHLCV収集失敗: {symbol} {timeframe} - {error_message}",
                exc_info=True,
            )
            return

        saved_count = ohlcv_result.get("saved_count", 0)
        logger.info(f"OHLCV収集完了: {symbol} {timeframe} - {saved_count}件保存")

        # 2. Funding Rate収集
        try:
            logger.info(f"Funding Rate収集開始: {symbol} {timeframe}")
            from app.core.services.data_collection.funding_rate_service import (
                BybitFundingRateService,
            )
            from database.repositories.funding_rate_repository import (
                FundingRateRepository,
            )

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
                    f"Funding Rate収集失敗: {symbol} - {funding_result.get('message')}",
                    exc_info=True,
                )

        except Exception:
            logger.error(f"Funding Rate収集エラー: {symbol}", exc_info=True)

        # 3. Open Interest収集
        try:
            logger.info(f"Open Interest収集開始: {symbol} {timeframe}")
            from app.core.services.data_collection.open_interest_service import (
                BybitOpenInterestService,
            )
            from database.repositories.open_interest_repository import (
                OpenInterestRepository,
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
                    f"Open Interest収集失敗: {symbol} {timeframe} - {oi_result.get('message')}",
                    exc_info=True,
                )

        except Exception:
            logger.error(
                f"Open Interest収集エラー: {symbol} {timeframe}", exc_info=True
            )

        logger.info(f"全データ収集完了: {symbol} {timeframe}")

    except Exception:
        logger.error(f"全データ収集エラー: {symbol} {timeframe}", exc_info=True)
    finally:
        db.close()


async def _collect_historical_background(symbol: str, timeframe: str, db: Session):
    """バックグラウンドでの履歴データ収集"""
    try:
        service = HistoricalDataService()
        repository = OHLCVRepository(db)

        result = await service.collect_historical_data(symbol, timeframe, repository)

        if isinstance(result, dict) and result.get("success"):
            saved_count = result.get("saved_count", 0)
            logger.info(
                f"バックグラウンド収集完了: {symbol} {timeframe} - {saved_count}件保存"
            )
        else:
            error_message = (
                result.get("message") if isinstance(result, dict) else str(result)
            )
            logger.error(
                f"バックグラウンド収集失敗: {symbol} {timeframe} - {error_message}",
                exc_info=True,
            )

    except Exception:
        logger.error(f"バックグラウンド収集エラー: {symbol} {timeframe}", exc_info=True)
    finally:
        db.close()
