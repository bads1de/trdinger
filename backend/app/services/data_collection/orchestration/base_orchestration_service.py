import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.utils.response import api_response
from database.connection import get_db

logger = logging.getLogger(__name__)


class BaseDataCollectionOrchestrationService:
    """
    データ収集オーケストレーションサービスの基底クラス
    共通のユーティリティメソッドを提供します。
    """

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        文字列の日付をdatetimeオブジェクトに変換する

        Args:
            date_str: 日付文字列（ISO形式、例: "2023-01-01T00:00:00"）

        Returns:
            datetimeオブジェクト、またはNone
        """
        if not date_str:
            return None
        try:
            # ISO形式の日付文字列をパース（例: "2023-01-01T00:00:00"）
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError as e:
            logger.error(f"日付文字列のパースに失敗しました: {date_str}, エラー: {e}")
            return None

    def _get_db_session(self, db_session: Optional[Session] = None) -> Session:
        """
        データベースセッションを取得または作成する

        Args:
            db_session: 既存のセッション（オプション）

        Returns:
            データベースセッション
        """
        if db_session:
            return db_session
        
        # セッションがない場合は新しく作成（ジェネレータから取得）
        return next(get_db())

    def _create_success_response(
        self, message: str, data: Optional[dict] = None
    ) -> dict:
        """
        成功レスポンスを作成する

        Args:
            message: メッセージ
            data: データ辞書

        Returns:
            APIレスポンス辞書
        """
        return api_response(success=True, message=message, data=data)
