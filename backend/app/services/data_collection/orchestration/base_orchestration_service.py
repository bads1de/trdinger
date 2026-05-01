"""
データ収集オーケストレーション基底サービス

データ収集サービスの共通ユーティリティを提供します。
日時パース、データベースセッション管理、レスポンス作成、
シンボル正規化などの機能を含みます。
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.utils.data_conversion import normalize_market_symbol
from app.utils.response import api_response, error_response
from database.connection import get_db

logger = logging.getLogger(__name__)


class BaseDataCollectionOrchestrationService:
    """
    データ収集オーケストレーションサービスの基底クラス

    データ収集サービスの共通のユーティリティメソッドを提供します。
    日時パース、データベースセッション管理、レスポンス作成、
    シンボル正規化などの機能を含みます。
    """

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        文字列の日付をdatetimeオブジェクトに変換する。

        ISO 8601形式の日付文字列をdatetimeオブジェクトに変換します。
        'Z'サフィックスを'+00:00'に置換してUTCタイムゾーンとして扱います。

        Args:
            date_str: 日付文字列（ISO 8601 形式、例: "2023-01-01T00:00:00"）

        Returns:
            Optional[datetime]: 変換されたdatetimeオブジェクト、失敗時または空文字時はNone
        """
        if not date_str:
            return None

        normalized = date_str.strip()
        if not normalized:
            return None

        try:
            return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError:
            logger.error(f"日付文字列のパースに失敗しました: {date_str}")
            return None

    @contextmanager
    def _get_db_session(self, db_session: Optional[Session] = None):
        """
        データベースセッションを取得するコンテキストマネージャ

        既存のセッションが渡された場合はそのまま使用し、
        Noneの場合は新規セッションを作成して終了時にクローズします。

        Args:
            db_session: 既存のセッション（Noneの場合は新規作成）

        Yields:
            Session: データベースセッション
        """
        if db_session is not None:
            # 既存セッションはそのまま使用（呼び出し元が管理）
            yield db_session
        else:
            # 新規セッションを作成し、終了時にクローズ
            session = next(get_db())
            try:
                yield session
            finally:
                session.close()

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

    def _create_error_response(
        self,
        message: str,
        *,
        details: Optional[dict] = None,
        error_code: Optional[str] = None,
        context: Optional[str] = None,
        data: Optional[dict] = None,
    ) -> dict:
        """
        標準化されたエラーレスポンスを作成する。

        app.utils.error_response.error_response のラッパー関数です。

        Args:
            message: エラーメッセージ
            details: エラーの詳細情報（オプション）
            error_code: エラーコード（オプション）
            context: エラーのコンテキスト情報（オプション）
            data: 追加データ（オプション）

        Returns:
            dict: 標準化されたエラーレスポンス辞書
        """
        return error_response(
            message=message,
            error_code=error_code,
            details=details,
            context=context,
            data=data,
        )
