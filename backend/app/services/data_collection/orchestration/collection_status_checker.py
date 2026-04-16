"""
収集状況確認モジュール

データ収集状況の確認を担当します。
"""

import logging
from typing import Any, Dict

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.utils.error_handler import safe_operation
from app.utils.response import api_response
from database.repositories.ohlcv_repository import OHLCVRepository

logger = logging.getLogger(__name__)


class CollectionStatusChecker:
    """
    収集状況確認クラス

    データ収集状況を確認します。
    """

    @safe_operation(context="データ収集状況確認", is_api_call=False)
    async def get_collection_status(
        self,
        symbol: str,
        timeframe: str,
        background_tasks: BackgroundTasks,
        auto_fetch: bool,
        db: Session,
        data_validator: Any = None,
        historical_orchestrator: Any = None,
    ) -> Dict[str, Any]:
        """
        データ収集状況を確認

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            auto_fetch: データが存在しない場合に自動フェッチを開始するか
            background_tasks: バックグラウンドタスク
            db: データベースセッション
            data_validator: データバリデーター
            historical_orchestrator: 履歴データオーケストレーター

        Returns:
            データ収集状況
        """
        if data_validator:
            normalized_symbol = data_validator.validate_symbol_and_timeframe(
                symbol, timeframe
            )
        else:
            # デフォルトのバリデーション（テスト用）
            from app.config.unified_config import unified_config

            normalized_symbol = unified_config.market.symbol_mapping.get(symbol, symbol)
            if normalized_symbol not in unified_config.market.supported_symbols:
                raise ValueError(f"サポートされていないシンボル: {symbol}")
            if timeframe not in unified_config.market.supported_timeframes:
                raise ValueError(f"無効な時間軸: {timeframe}")

        repository = OHLCVRepository(db)

        # 正規化されたシンボルでデータ件数を取得
        data_count = repository.get_data_count(normalized_symbol, timeframe)

        # データが存在しない場合の処理
        if data_count == 0:
            if auto_fetch and background_tasks and historical_orchestrator:
                # 自動フェッチを開始
                await historical_orchestrator.start_historical_data_collection(
                    normalized_symbol,
                    timeframe,
                    background_tasks,
                    db,
                    data_validator=data_validator,
                )
                logger.info(f"自動フェッチを開始: {normalized_symbol} {timeframe}")

                return api_response(
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
                return api_response(
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
        latest_timestamp = repository.get_latest_timestamp(
            timestamp_column="timestamp",
            filter_conditions={"symbol": normalized_symbol, "timeframe": timeframe},
        )
        oldest_timestamp = repository.get_oldest_timestamp(
            timestamp_column="timestamp",
            filter_conditions={"symbol": normalized_symbol, "timeframe": timeframe},
        )

        return api_response(
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
