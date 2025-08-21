from datetime import datetime
from typing import Any, Dict, Optional


def now_iso() -> str:
    return datetime.now().isoformat()


def error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    エラーレスポンスを生成

    message: エラーメッセージ
    error_code: エラーコード
    details: エラー詳細
    context: エラーコンテキスト

    Returns:
        生成されたエラーレスポンス
    """
    response: Dict[str, Any] = {
        "success": False,
        "message": message,
        "timestamp": now_iso(),
    }
    if error_code:
        response["error_code"] = error_code
    if details:
        response["details"] = details
    if context:
        response["context"] = context
    return response


def api_response(
    success: bool,
    message: str = "",
    status: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    """
    汎用APIレスポンス生成ユーティリティ。

    既存コードベースでは呼び出し側が様々なキーワード引数（message, data, error, status_code 等）を
    使用しているため、互換性を保つために柔軟に受け付けるようにしています。

    Rules:
    - message はオプション（空文字が許容される）
    - error が指定された場合は "error" フィールドを含める
    - status_code が指定された場合は "status_code" フィールドを含める
    """
    response: Dict[str, Any] = {"success": success}

    if message:
        response["message"] = message

    if error:
        response["error"] = error

    if status:
        response["status"] = status

    if data is not None:
        response["data"] = data

    if status_code is not None:
        response["status_code"] = status_code

    response["timestamp"] = now_iso()
    return response
