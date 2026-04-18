"""
データ収集統合管理サービス

APIルーター内に散在していたデータ収集関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

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
        """
        DataCollectionOrchestrationServiceを初期化

        データ収集プロセスの統合オーケストレーターを初期化します。
        各データ種別（OHLCV、ファンディングレート、建玉）の収集ロジックを
        集約し、一括差分更新、全履歴の新規取得などの高レイヤーな
        「収集タスク」として提供します。

        Note:
            このサービスは以下のサブサービスを初期化します：
            - DataValidator: シンボルと時間軸のバリデーション
            - HistoricalDataOrchestrator: 履歴データ収集
            - BulkDataOrchestrator: 一括データ収集
            - CollectionStatusChecker: データ収集状況確認
            - OICollectionOrchestrator: オープンインタレスト収集
        """
        self.data_validator = DataValidator()
        self.historical_orchestrator = HistoricalDataOrchestrator()
        self.bulk_data_orchestrator = BulkDataOrchestrator()
        # 差分更新用サービス
        self.historical_service = self.bulk_data_orchestrator.historical_service
        self.collection_status_checker = CollectionStatusChecker()
        self.oi_collection_orchestrator = OICollectionOrchestrator()

    def validate_symbol_and_timeframe(self, symbol: str, timeframe: str) -> str:
        """
        シンボルと時間軸のバリデーション

        指定されたシンボルと時間軸が有効かどうかを検証し、
        正規化されたシンボルを返します。

        Args:
            symbol: 取引ペア（例: "BTC/USDT:USDT"）
            timeframe: 時間軸（例: "1h", "1d"）

        Returns:
            str: 正規化されたシンボル

        Raises:
            ValueError: シンボルまたは時間軸が無効な場合

        Note:
            このメソッドはunified_configを直接使用してバリデーションを行います。
            テスト時はunified_configをモックすることでバリデーション動作を変更できます。
        """
        # モジュールレベルのインポートを遅延評価してテスト時のモックを有効にする
        from app.services.data_collection.orchestration.data_validator import (
            unified_config as validator_config,
        )

        normalized_symbol = validator_config.market.symbol_mapping.get(symbol, symbol)
        if normalized_symbol not in validator_config.market.supported_symbols:
            raise ValueError(f"サポートされていないシンボル: {symbol}")

        if timeframe not in validator_config.market.supported_timeframes:
            raise ValueError(f"無効な時間軸: {timeframe}")

        return normalized_symbol

    def _resolve_ohlcv_repository_class(self):
        """
        patchされたOHLCVRepositoryを優先して解決する

        テスト時などにpatchされたOHLCVRepositoryが存在する場合、
        それを優先的に使用します。patchがない場合はデフォルトの
        OHLCVRepositoryを使用します。

        Returns:
            OHLCVRepository: 解決されたリポジトリクラス

        Note:
            このメソッドはテスト時のモック置換をサポートするために
            使用されます。
        """
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
        特定のシンボルと時間軸の過去価格データ（OHLCV）収集を非同期で開始します。

        このメソッドは、以下のプロセスをオーケストレーションします：
        1. 既にデータが存在するか、またはタスクが実行中かチェック。
        2. `force_update=True` の場合、既存データを削除（リセット）してから再取得を予約。
        3. FastAPIの `BackgroundTasks` に収集ジョブを登録し、即座にレスポンスを返却。

        Args:
            symbol (str): 収集対象の取引ペア（例: "BTC/USDT:USDT"）。
            timeframe (str): 収集する時間軸（"1m", "1h" 等）。
            background_tasks (BackgroundTasks): 非同期実行用のタスク管理オブジェクト。
            db (Session): データベースセッション。
            force_update (bool): Trueの場合、既存データを上書きして最初から取得し直します。デフォルトはFalse。
            start_date (Optional[str]): 収集を開始する日付（"2023-01-01"形式）。未指定時は取引所の最大履歴またはデフォルト値が使用されます。

        Returns:
            Dict[str, Any]: タスクの受付状態を示す辞書。
                主なキー：
                - "success" (bool): 予約の成否。
                - "status" (str): "started", "already_running", "exists" 等の状態。
                - "message" (str): ユーザー向けの進捗メッセージ。

        Note:
            実際のデータ取得はバックグラウンドで実行されるため、このメソッドの終了は「データ収集の完了」を意味しません。
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
        市場全体の最新データを一括で同期（差分更新）します。

        このメソッドは、システムが管理する以下の全データを最新の状態に保つために実行されます：
        1. OHLCVデータ（全サポート時間軸）。
        2. デリバティブ指標（ファンディングレート、建玉残高）。
        3. その他の市場統計。

        内部プロセス：
        - 各データ種別の「最新のタイムスタンプ」をDBから取得。
        - 取引所APIを呼び出し、DB末尾から現在時刻までの差分を取得・保存。

        Args:
            symbol (str): 同期対象の取引ペア。
            db (Session): データベースセッション。

        Returns:
            Dict[str, Any]: 同期結果のサマリー。
                更新されたレコード数、スキップされた時間軸、エラーが発生した項目のリスト等を含みます。

        Raises:
            Exception: 取引所APIへの通信エラーや、DB書き込み時の致命的なエラー。
        """
        return await self.bulk_data_orchestrator.execute_bulk_incremental_update(
            symbol, db
        )

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
            symbol,
            timeframe,
            background_tasks,
            auto_fetch,
            db,
            self.data_validator,
            self.historical_orchestrator,
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
        return await self.bulk_data_orchestrator.start_all_data_bulk_collection(
            background_tasks, db
        )

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
