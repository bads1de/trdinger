from datetime import datetime
from typing import Any, Dict, Mapping, Optional


def now_iso() -> str:
    return datetime.now().isoformat()


def ensure_response_dict(result: Any) -> Dict[str, Any]:
    """辞書互換のレスポンス値を dict に正規化する。"""
    if isinstance(result, dict):
        return result

    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    dict_method = getattr(result, "dict", None)
    if callable(dict_method):
        dumped = dict_method()
        if isinstance(dumped, dict):
            return dumped

    return {}


def extract_response_data(
    result: Mapping[str, Any],
    key: str = "data",
) -> Dict[str, Any]:
    """レスポンスのネストされた dict ペイロードを取り出す。"""
    payload = result.get(key)
    return payload if isinstance(payload, dict) else {}


def _build_response(
    success: bool,
    fields: Dict[str, Any],
) -> Dict[str, Any]:
    """
    レスポンス辞書を構築する内部ヘルパー関数

    Args:
        success: 成功フラグ
        fields: 追加フィールドの辞書（値がNoneや空文字のものは除外される）

    Returns:
        構築されたレスポンス辞書
    """
    response: Dict[str, Any] = {"success": success}

    for key, value in fields.items():
        if value is not None and value != "":
            response[key] = value

    response["timestamp"] = now_iso()
    return response


def error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    エラーレスポンスを生成

    message: エラーメッセージ
    error_code: エラーコード
    details: エラー詳細
    context: エラーコンテキスト
    data: 返却データ（必須フィールドとして追加）

    Returns:
        生成されたエラーレスポンス
    """
    fields = {
        "message": message,
        "error_code": error_code,
        "details": details,
        "context": context,
        "data": data,
    }
    return _build_response(success=False, fields=fields)


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
    fields = {
        "message": message,
        "error": error,
        "status": status,
        "data": data,
        "status_code": status_code,
    }
    return _build_response(success=success, fields=fields)



