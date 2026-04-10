"""
OI収集オーケストレーターモジュール

Open Interest履歴データの収集を担当します。
"""

import logging
from typing import Any, Dict

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.utils.error_handler import safe_operation
from app.utils.response import api_response
from database.models import OpenInterestData
from database.repositories.open_interest_repository import OpenInterestRepository

logger = logging.getLogger(__name__)


class OICollectionOrchestrator:
    """
    OI収集オーケストレーター

    Open Interest履歴データの収集を管理します。
    """

    @safe_operation(context="OI履歴データ収集開始", is_api_call=True)
    async def start_historical_oi_collection(
        self,
        symbol: str,
        interval: str,
        background_tasks: BackgroundTasks,
        db: Session,
        data_validator: Any = None,
    ) -> Dict[str, Any]:
        """
        OI履歴データ収集を開始（既存データを削除して全期間再取得）

        Args:
            symbol: 取引ペア
            interval: 時間軸
            background_tasks: バックグラウンドタスク
            db: データベースセッション
            data_validator: データバリデーター

        Returns:
            収集開始結果
        """
        # シンボルと時間軸のバリデーション
        if data_validator:
            normalized_symbol = data_validator.validate_symbol_and_timeframe(symbol, interval)
        else:
            # デフォルトのバリデーション（テスト用）
            from app.config.unified_config import unified_config
            normalized_symbol = unified_config.market.symbol_mapping.get(symbol, symbol)
            if normalized_symbol not in unified_config.market.supported_symbols:
                raise ValueError(f"サポートされていないシンボル: {symbol}")
            if interval not in unified_config.market.supported_timeframes:
                raise ValueError(f"無効な時間軸: {interval}")

        # バックグラウンドタスクとして実行
        background_tasks.add_task(
            self._collect_historical_oi_background,
            normalized_symbol,
            interval,
            db,
        )

        return api_response(
            success=True,
            message=f"{normalized_symbol} {interval} のOI履歴データ収集を開始しました（既存データ削除・全期間再取得）",
            status="started",
        )

    async def _collect_historical_oi_background(
        self, symbol: str, interval: str, db: Session
    ):
        """バックグラウンドでのOI履歴データ収集"""
        try:
            logger.info(f"OI履歴データ収集開始: {symbol} {interval}")

            # 既存データを削除
            try:
                count = db.query(OpenInterestData).count()
                if count > 0:
                    db.query(OpenInterestData).delete()
                    db.commit()
                    logger.info(f"既存のOIデータ {count}件を削除しました")
                else:
                    logger.info("既存のOIデータはありません")
            except Exception as e:
                logger.warning(f"OIデータ削除処理中に警告: {e}")
                db.rollback()

            # データ収集
            from ..bybit.open_interest_service import BybitOpenInterestService

            oi_service = BybitOpenInterestService()
            oi_repository = OpenInterestRepository(db)

            logger.info("2020年以降の全OIデータを取得します...")

            result = await oi_service.fetch_and_save_open_interest_data(
                symbol=symbol,
                repository=oi_repository,
                fetch_all=True,
                interval=interval,
            )

            if result["success"]:
                logger.info(
                    f"OI収集成功: {symbol} {interval} - {result['saved_count']}件保存"
                )
            else:
                logger.error(
                    f"OI収集失敗: {symbol} {interval} - {result.get('message')}"
                )

        except Exception as e:
            logger.error(
                f"OI履歴データ収集中にエラーが発生しました: {symbol} {interval}", e
            )
        finally:
            if hasattr(db, "close"):
                db.close()
