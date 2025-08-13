"""
API共通ユーティリティ
"""
 
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from .response import make_error_response, make_api_response, make_list_response, make_success_response
 
logger = logging.getLogger(__name__)
 
 
class APIResponseHelper:
    """API レスポンス形式の共通ヘルパークラス（薄いラッパ）"""
 
    @staticmethod
    def error_response(
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        エラーレスポンスを生成（共通ユーティリティへ委譲）
        """
        return make_error_response(message=message, error_code=error_code, details=details)

    @staticmethod
    def api_response(
        success: bool,
        message: str,
        status: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        標準化されたAPIレスポンスを生成（共通ユーティリティへ委譲）
        """
        return make_api_response(success=success, message=message, status=status, data=data)

    @staticmethod
    def api_list_response(
        success: bool,
        message: str,
        items: List[Any],
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        リストデータ用の標準化されたAPIレスポンスを生成（共通ユーティリティへ委譲）
        """
        return make_list_response(success=success, message=message, items=items, status=status, metadata=metadata)

    @staticmethod
    def success_response(
        data: Optional[Dict[str, Any]] = None,
        message: str = "成功",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        成功レスポンスを生成（後方互換性のため）
        """
        return make_success_response(data=data, message=message, metadata=metadata)
