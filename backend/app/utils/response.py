from datetime import datetime
from typing import Any, Dict, Mapping, Optional


def now_iso() -> str:
    """
    現在時刻をISO8601形式の文字列で返します。

    Returns:
        str: ISO8601形式の現在時刻
    """
    return datetime.now().isoformat()


def ensure_response_dict(result: Any) -> Dict[str, Any]:
    """
    辞書互換のレスポンス値を dict に正規化する

    Pydanticモデルや辞書など、様々な形式のレスポンス値を
    標準的な辞書形式に変換します。model_dump()メソッドや
    dict()メソッドを試行し、失敗した場合は空辞書を返します。

    Args:
        result: 正規化対象のレスポンス値（辞書、Pydanticモデル等）

    Returns:
        Dict[str, Any]: 正規化された辞書。変換失敗時は空辞書
    """
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
    """
    レスポンスのネストされた dict ペイロードを取り出す

    レスポンス辞書から指定されたキーの値を取り出します。
    値が辞書でない場合は空辞書を返します。

    Args:
        result: ペイロードを含むレスポンス辞書
        key: 取り出すキー名（デフォルト: "data"）

    Returns:
        Dict[str, Any]: 指定されたキーの辞書値。
                       キーが存在しないか値が辞書でない場合は空辞書
    """
    payload = result.get(key)
    return payload if isinstance(payload, dict) else {}


def _build_response(
    success: bool,
    fields: Dict[str, Any],
) -> Dict[str, Any]:
    """レスポンス辞書を構築する内部ヘルパー関数。

    成功/失敗フラグと追加フィールドから、標準化されたAPIレスポンス辞書を生成します。
    Noneや空文字のフィールドは自動的に除外され、タイムスタンプが付加されます。

    Args:
        success: 成功フラグ。処理が成功した場合はTrue。
        fields: 追加フィールドの辞書。
            "data", "message", "error" などのキーを含む。
            値がNoneまたは空文字("")の場合は除外される。

    Returns:
        Dict[str, Any]: 標準化されたレスポンス辞書。
            常に "success"（bool）と "timestamp"（ISO形式文字列）を含み、
            それに加えて有効なfieldsのキーが含まれる。
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

    標準化されたエラーレスポンス形式で失敗時の情報を返します。
    successフィールドはFalseに設定されます。

    Args:
        message: エラーメッセージ（必須）
        error_code: エラーコード（オプション）
        details: エラー詳細情報を含む辞書（オプション）
        context: エラーが発生したコンテキスト情報（オプション）
        data: 返却データ（オプション、必須フィールドとして追加）

    Returns:
        Dict[str, Any]: 生成されたエラーレスポンス辞書。
                       以下のキーを含みます：
                       - success: False
                       - message: エラーメッセージ
                       - error_code: エラーコード（指定時）
                       - details: エラー詳細（指定時）
                       - context: コンテキスト（指定時）
                       - data: 返却データ（指定時）
                       - timestamp: ISO8601形式の現在時刻
    """
    fields = {
        "message": message,
        "error_code": error_code,
        "details": details,
        "context": context,
        "data": data,
    }
    return _build_response(success=False, fields=fields)


def result_response(
    success: bool,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
    status: Optional[str] = None,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    """
    成功/失敗どちらのレスポンスもまとめて生成する統一関数

    successがTrueの場合はapi_response、Falseの場合はerror_responseを内部的に呼び出します。

    Args:
        success: 成功フラグ（True/False）
        message: レスポンスメッセージ
        data: 返却データ（成功時）
        error_code: エラーコード（失敗時）
        details: エラー詳細情報（失敗時）
        context: エラーコンテキスト（失敗時）
        status: ステータス文字列
        status_code: HTTPステータスコード

    Returns:
        構築された統一レスポンス辞書
    """
    if success:
        return api_response(
            success=True,
            message=message,
            status=status,
            data=data,
            status_code=status_code,
        )

    return error_response(
        message=message,
        error_code=error_code,
        details=details,
        context=context,
        data=data,
    )


def api_response(
    success: bool,
    message: str = "",
    status: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    """
    汎用APIレスポンス生成ユーティリティ

    既存コードベースでは呼び出し側が様々なキーワード引数
    （message, data, error, status_code 等）を使用しているため、
    互換性を保つために柔軟に受け付けるようにしています。

    Args:
        success: 成功フラグ（必須）
        message: レスポンスメッセージ（オプション、空文字許容）
        status: ステータス文字列（オプション）
        data: 返却データ（オプション）
        error: エラーメッセージ（オプション、指定時は"error"フィールドを含める）
        status_code: HTTPステータスコード（オプション、指定時は"status_code"フィールドを含める）

    Returns:
        Dict[str, Any]: 生成されたAPIレスポンス辞書。
                       以下のキーを含みます：
                       - success: 成功フラグ
                       - message: メッセージ（指定時）
                       - status: ステータス（指定時）
                       - data: 返却データ（指定時）
                       - error: エラーメッセージ（指定時）
                       - status_code: HTTPステータスコード（指定時）
                       - timestamp: ISO8601形式の現在時刻

    Note:
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
