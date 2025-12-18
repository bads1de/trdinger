"""
データ収集統合管理サービス

APIルーター内に散在していたデータ収集関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.config.unified_config import unified_config
from app.utils.error_handler import safe_operation
from app.utils.response import api_response
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

from ..historical.historical_data_service import HistoricalDataService

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
        self.historical_service = HistoricalDataService()

    @safe_operation(context="シンボル・時間軸バリデーション", is_api_call=False)
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
        # シンボル正規化
        normalized_symbol = unified_config.market.symbol_mapping.get(symbol, symbol)
        if normalized_symbol not in unified_config.market.supported_symbols:
            raise ValueError(f"サポートされていないシンボル: {symbol}")

        # 時間軸検証
        if timeframe not in unified_config.market.supported_timeframes:
            raise ValueError(f"無効な時間軸: {timeframe}")

        return normalized_symbol

    @safe_operation(context="履歴データ収集開始", is_api_call=True)
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
        # シンボルと時間軸のバリデーション
        normalized_symbol = self.validate_symbol_and_timeframe(symbol, timeframe)

        # データ存在チェック
        repository = OHLCVRepository(db)
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
        try:
            # 全時間軸でビットコインデータを収集
            timeframes = unified_config.market.supported_timeframes

            for timeframe in timeframes:
                background_tasks.add_task(
                    self._collect_historical_background, "BTC/USDT:USDT", timeframe, db
                )

            return api_response(
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
        try:
            # 取引ペアと時間軸の定義
            symbols = [
                "BTC/USDT:USDT",
            ]
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
                            self._collect_historical_background,
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
            logger.error("一括履歴データ収集開始エラー", e)
            raise

    @safe_operation(context="データ収集状況確認", is_api_call=True)
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
        from app.config.unified_config import unified_config

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
                await self.start_historical_data_collection(
                    normalized_symbol, timeframe, background_tasks, db
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
            logger.error("全データ一括収集開始エラー", e)
            raise

    async def _collect_historical_background(
        self, symbol: str, timeframe: str, db: Session, start_date: Optional[str] = None
    ):
        """バックグラウンドでの履歴データ収集（ページネーションで全期間取得）"""
        try:
            logger.info(f"履歴データ収集開始: {symbol} {timeframe}")

            repository = OHLCVRepository(db)

            logger.info("ページネーションで全期間データを取得します")

            result = (
                await self.historical_service.collect_historical_data_with_start_date(
                    symbol,
                    timeframe,
                    repository,
                    None,  # since_timestamp は使用せずページネーションで全データ取得
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

    @safe_operation(context="OI履歴データ収集開始", is_api_call=True)
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
        # シンボルと時間軸のバリデーション
        normalized_symbol = self.validate_symbol_and_timeframe(symbol, interval)

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
            from database.models import OpenInterestData

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
