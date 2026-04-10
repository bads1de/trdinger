"""
データ収集統合管理サービス

APIルーター内に散在していたデータ収集関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.config.constants import DEFAULT_MARKET_SYMBOL
from app.config.unified_config import unified_config
from app.utils.error_handler import safe_operation
from app.utils.response import api_response
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.ohlcv_repository import OHLCVRepository

from . import historical_data_orchestrator as historical_data_orchestrator_module
from .bulk_data_orchestrator import BulkDataOrchestrator
from .collection_status_checker import CollectionStatusChecker
from .data_validator import DataValidator
from .historical_data_orchestrator import HistoricalDataOrchestrator
from .oi_collection_orchestrator import OICollectionOrchestrator

_BASE_OHLCV_REPOSITORY = OHLCVRepository

logger = logging.getLogger(__name__)


class DataCollectionOrchestrationService:
    """
    データ収集プロセスの統合オーケストレーター

    各データ種別（OHLCV、ファンディングレート、建玉）の収集ロジックを集約し、
    一括差分更新、全履歴の新規取得、ビットコイン特化収集などの
    高レイヤーな「収集タスク」として提供します。
    バックグラウンド実行のスケジューリングやエラーハンドリング、
    および API レスポンス形式の統一も担当します。
    """

    def __init__(self):
        """初期化"""
        self.data_validator = DataValidator()
        self.historical_orchestrator = HistoricalDataOrchestrator()
        self.bulk_data_orchestrator = BulkDataOrchestrator()
        # 旧コードが直接参照していた差分更新用サービスの互換エイリアス
        self.historical_service = self.bulk_data_orchestrator.historical_service
        self.collection_status_checker = CollectionStatusChecker()
        self.oi_collection_orchestrator = OICollectionOrchestrator()

    def validate_symbol_and_timeframe(self, symbol: str, timeframe: str) -> str:
        """
        シンボルと時間軸のバリデーション

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            正規化されたシンボル

        Raises:
            ValueError: バリデーションエラー
        """
        return self.data_validator.validate_symbol_and_timeframe(symbol, timeframe)

    def _resolve_ohlcv_repository_class(self):
        """patch された OHLCVRepository を優先して解決する。"""
        if OHLCVRepository is not _BASE_OHLCV_REPOSITORY:
            return OHLCVRepository

        historical_repository = historical_data_orchestrator_module.OHLCVRepository
        if historical_repository is not _BASE_OHLCV_REPOSITORY:
            return historical_repository

        return _BASE_OHLCV_REPOSITORY

    async def start_historical_data_collection(
        self,
        symbol: str,
        timeframe: str,
        background_tasks: BackgroundTasks,
        db: Session,
        force_update: bool = False,
        start_date: Optional[str] = None,
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

        Returns:
            収集タスクのステータス（started, exists等）を含むレスポンス辞書
        """
        repository_class = self._resolve_ohlcv_repository_class()

        return await self.historical_orchestrator.start_historical_data_collection(
            symbol,
            timeframe,
            background_tasks,
            db,
            force_update,
            start_date,
            self.data_validator,
            ohlcv_repository_class=repository_class,
        )

    async def execute_bulk_incremental_update(
        self, symbol: str, db: Session
    ) -> Dict[str, Any]:
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
        return await self.bulk_data_orchestrator.execute_bulk_incremental_update(symbol, db)

    async def start_bitcoin_full_data_collection(
        self, background_tasks: BackgroundTasks, db: Session
    ) -> Dict[str, Any]:
        """
        ビットコイン（BTC/USDT）の全時間軸、全期間データの収集を予約

        システムがサポートする全時間足（1m から 1d まで）に対して
        非同期の収集タスクを発行します。

        Args:
            background_tasks: FastAPI のバックグラウンドタスク管理
            db: データベースセッション

        Returns:
            予約された全タスクの情報を保持するレスポンス
        """
        return await self.bulk_data_orchestrator.start_bitcoin_full_data_collection(
            background_tasks, db, self.historical_orchestrator
        )

    async def start_bulk_historical_data_collection(
        self,
        background_tasks: BackgroundTasks,
        db: Session,
        force_update: bool = False,
        start_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        サポートされている全シンボル・全時間軸の履歴データ収集を一括予約

        DB を走査し、データが未取得の箇所について自動的に収集タスクを構築
        および並列実行（バックグラウンド）します。

        Args:
            background_tasks: 非同期タスク管理
            db: データベースセッション
            force_update: 既存データがある場合も削除して再取得するか
            start_date: 全タスクの開始日付（未指定時はデフォルト 2020-03-25）

        Returns:
            発行された全タスク数と、対象シンボル・時間軸のリスト
        """
        return await self.bulk_data_orchestrator.start_bulk_historical_data_collection(
            background_tasks, db, force_update, start_date, self.historical_orchestrator
        )

    async def get_collection_status(
        self,
        symbol: str,
        timeframe: str,
        background_tasks: BackgroundTasks,
        auto_fetch: bool,
        db: Session,
    ) -> Dict[str, Any]:
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
        return await self.collection_status_checker.get_collection_status(
            symbol, timeframe, background_tasks, auto_fetch, db, self.data_validator, self.historical_orchestrator
        )

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
        return await self.bulk_data_orchestrator.start_all_data_bulk_collection(background_tasks, db)

    async def start_historical_oi_collection(
        self,
        symbol: str,
        interval: str,
        background_tasks: BackgroundTasks,
        db: Session,
    ) -> Dict[str, Any]:
        """
        OI履歴データ収集を開始（既存データを削除して全期間再取得）

        Args:
            symbol: 取引ペア
            interval: 時間軸
            background_tasks: バックグラウンドタスク
            db: データベースセッション

        Returns:
            収集開始結果
        """
        return await self.oi_collection_orchestrator.start_historical_oi_collection(
            symbol, interval, background_tasks, db, self.data_validator
        )
