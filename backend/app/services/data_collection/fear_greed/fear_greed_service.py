"""
Fear & Greed Index データ収集サービス

Alternative.me APIを使用してFear & Greed Indexデータを取得し、
データベースに保存する機能を提供します。
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp

from app.utils.data_converter import DataValidator
from app.utils.unified_error_handler import (
    UnifiedDataError,
    UnifiedErrorHandler,
    unified_safe_operation,
)
from database.repositories.fear_greed_repository import FearGreedIndexRepository

logger = logging.getLogger(__name__)


class FearGreedIndexService:
    """Fear & Greed Index データ収集サービス"""

    def __init__(self):
        """サービスを初期化"""
        self.api_url = "https://api.alternative.me/fng/"
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def __aenter__(self):
        """非同期コンテキストマネージャーの開始"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーの終了"""
        if self.session:
            await self.session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """セッションを取得（必要に応じて作成）"""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    async def fetch_fear_greed_data(self, limit: int = 30) -> List[dict]:
        """
        Alternative.me APIからFear & Greed Indexデータを取得

        Args:
            limit: 取得するデータ数（デフォルト: 30、最大: 不明）

        Returns:
            Fear & Greed Index データのリスト

        Raises:
            aiohttp.ClientError: HTTP通信エラーの場合
            ValueError: APIレスポンスが無効な場合
        """
        return await UnifiedErrorHandler.safe_execute(
            func=self._fetch_fear_greed_data_impl,
            default_return=[],
            error_message="Fear & Greed Index データ取得",
            log_level="error",
            is_api_call=False,
        )(limit)

    async def _fetch_fear_greed_data_impl(self, limit: int) -> List[dict]:
        """Fear & Greed Index データ取得の実装"""
        session = await self._get_session()

        # APIパラメータ
        params = {"limit": limit, "format": "json"}

        logger.info(f"Fear & Greed Index データを取得中: limit={limit}")

        async with session.get(self.api_url, params=params) as response:
            if response.status != 200:
                raise UnifiedDataError(
                    f"API request failed with status {response.status}: {await response.text()}"
                )

            data = await response.json()

            # レスポンス構造の検証
            if not isinstance(data, dict):
                raise UnifiedDataError("APIレスポンスが辞書形式ではありません")

            if "data" not in data:
                raise UnifiedDataError("APIレスポンスに'data'フィールドがありません")

            if not isinstance(data["data"], list):
                raise UnifiedDataError(
                    "APIレスポンスの'data'フィールドがリスト形式ではありません"
                )

            # メタデータのエラーチェック
            if "metadata" in data and data["metadata"].get("error"):
                raise UnifiedDataError(f"API error: {data['metadata']['error']}")

            raw_data = data["data"]
            logger.info(f"Fear & Greed Index データを {len(raw_data)} 件取得しました")

            # データ変換
            converted_data = self._convert_api_data_to_db_format(raw_data)

            return converted_data

    def _convert_api_data_to_db_format(self, api_data: List[dict]) -> List[dict]:
        """
        APIレスポンスをデータベース形式に変換

        Args:
            api_data: APIから取得した生データ

        Returns:
            データベース挿入用のデータリスト
        """
        db_records = []
        current_time = datetime.now(timezone.utc)

        for item in api_data:
            try:
                # 必須フィールドの存在確認
                if not all(
                    key in item
                    for key in ["value", "value_classification", "timestamp"]
                ):
                    logger.warning(
                        f"必須フィールドが不足しているデータをスキップ: {item}"
                    )
                    continue

                # タイムスタンプの変換（Unix timestamp -> datetime）
                timestamp_unix = int(item["timestamp"])
                data_timestamp = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)

                db_record = {
                    "value": int(item["value"]),
                    "value_classification": str(item["value_classification"]),
                    "data_timestamp": data_timestamp,
                    "timestamp": current_time,
                }

                db_records.append(db_record)

            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"データ変換エラー、スキップします: {item}, エラー: {e}")
                continue

        logger.info(f"Fear & Greed Index データを {len(db_records)} 件変換しました")
        return db_records

    async def fetch_and_save_fear_greed_data(
        self,
        limit: int = 30,
        repository: Optional[FearGreedIndexRepository] = None,
    ) -> dict:
        """
        Fear & Greed Indexデータを取得してデータベースに保存

        Args:
            limit: 取得するデータ数
            repository: FearGreedIndexRepository（テスト用）

        Returns:
            処理結果を含む辞書
        """
        return UnifiedErrorHandler.safe_execute(
            func=lambda: self._fetch_and_save_fear_greed_data_impl(limit, repository),
            default_return={
                "success": False,
                "fetched_count": 0,
                "inserted_count": 0,
                "error": "処理中にエラーが発生しました",
                "message": "Fear & Greed Indexデータの取得・保存に失敗しました",
            },
            error_message="Fear & Greed Index データ取得・保存",
            log_level="error",
            is_api_call=False,
        )

    async def _fetch_and_save_fear_greed_data_impl(
        self, limit: int, repository: FearGreedIndexRepository
    ) -> dict:
        """Fear & Greed Index データ取得・保存の実装"""
        # データ取得
        fear_greed_data = await self.fetch_fear_greed_data(limit)

        if not fear_greed_data:
            logger.warning("取得したFear & Greed Indexデータが空です")
            return {
                "success": True,
                "fetched_count": 0,
                "inserted_count": 0,
                "message": "取得データが空でした",
            }

        # データ検証
        if not DataValidator.validate_fear_greed_data(fear_greed_data):
            raise UnifiedDataError("取得したFear & Greed Indexデータが無効です")

        # データベースに保存
        if repository:
            inserted_count = repository.insert_fear_greed_data(fear_greed_data)
        else:
            # 実際の運用時はここでリポジトリを作成
            logger.warning(
                "リポジトリが提供されていません。データは保存されませんでした。"
            )
            inserted_count = 0

        result = {
            "success": True,
            "fetched_count": len(fear_greed_data),
            "inserted_count": inserted_count,
            "message": f"Fear & Greed Indexデータを {inserted_count} 件保存しました",
        }

        logger.info(result["message"])
        return result

    @unified_safe_operation(
        default_return={
            "success": False,
            "error": "処理中にエラーが発生しました",
        },
        context="最新データ情報取得",
        is_api_call=False,
    )
    async def get_latest_data_info(self, repository: FearGreedIndexRepository) -> dict:
        """
        最新のデータ情報を取得

        Args:
            repository: FearGreedIndexRepository

        Returns:
            最新データ情報
        """
        data_range = repository.get_data_range()
        latest_timestamp = repository.get_latest_data_timestamp()

        return {
            "success": True,
            "data_range": data_range,
            "latest_timestamp": (
                latest_timestamp.isoformat() if latest_timestamp else None
            ),
        }

    async def close(self):
        """リソースのクリーンアップ"""
        if self.session:
            await self.session.close()
            self.session = None
