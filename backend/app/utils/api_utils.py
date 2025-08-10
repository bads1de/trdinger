"""
API共通ユーティリティ
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class APIResponseHelper:
    """API レスポンス形式の共通ヘルパークラス"""

    @staticmethod
    def error_response(
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        エラーレスポンスを生成

        Args:
            message: エラーメッセージ
            error_code: エラーコード
            details: エラー詳細

        Returns:
            エラーレスポンス辞書
        """
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        if error_code:
            response["error_code"] = error_code

        if details:
            response["details"] = details

        return response

    @staticmethod
    def api_response(
        success: bool,
        message: str,
        status: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        標準化されたAPIレスポンスを生成するヘルパー関数。

        Args:
            success: 成功フラグ
            message: メッセージ
            status: ステータス文字列（オプション）
            data: レスポンスデータ（辞書型、オプション）

        Returns:
            標準化されたAPIレスポンス辞書
        """
        response = {"success": success, "message": message}
        if status:
            response["status"] = status
        if data is not None:
            response["data"] = data
        response["timestamp"] = datetime.now().isoformat()
        return response

    @staticmethod
    def api_list_response(
        success: bool,
        message: str,
        items: List[Any],
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        リストデータ用の標準化されたAPIレスポンスを生成するヘルパー関数。

        Args:
            success: 成功フラグ
            message: メッセージ
            items: リストデータ
            status: ステータス文字列（オプション）
            metadata: メタデータ（オプション）

        Returns:
            標準化されたAPIレスポンス辞書
        """
        response_data = {"items": items}
        if metadata:
            response_data.update(metadata)

        return APIResponseHelper.api_response(
            success=success,
            message=message,
            status=status,
            data=response_data,
        )

    @staticmethod
    def success_response(
        data: Optional[Dict[str, Any]] = None,
        message: str = "成功",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        成功レスポンスを生成（後方互換性のため）

        Args:
            data: レスポンスデータ
            message: メッセージ
            metadata: メタデータ

        Returns:
            成功レスポンス辞書
        """
        response_data = data
        if metadata:
            if isinstance(data, dict):
                response_data = {**data, "metadata": metadata}
            else:
                response_data = {"data": data, "metadata": metadata}

        return APIResponseHelper.api_response(
            success=True,
            message=message,
            data=response_data,
        )
