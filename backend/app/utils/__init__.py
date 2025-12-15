"""
共通ユーティリティパッケージ

アプリケーション全体で使用される共通機能を提供します。

主な機能:
- data_processing: データの前処理、変換、検証
- response: APIレスポンスの標準化
- error_handler: エラーハンドリング
- duplicate_filter_handler: 重複データのフィルタリング
- data_conversion: データ形式の変換

各モジュールは特定の責務を持ち、再利用可能な形で実装されています。
"""

# よく使われる機能を明示的にエクスポート
from .data_conversion import (
    DataConversionError,
    FundingRateDataConverter,
    OHLCVDataConverter,
    OpenInterestDataConverter,
    parse_timestamp_safe,
)
from .duplicate_filter_handler import DuplicateFilter
from .error_handler import (
    DataError,
    ErrorHandler,
    ModelError,
    TimeoutError,
    ValidationError,
    operation_context,
    safe_execute,
    safe_operation,
    timeout_decorator,
)
from .response import api_response, error_response

__all__ = [
    # レスポンスユーティリティ
    "api_response",
    "error_response",
    # エラーハンドリング
    "ErrorHandler",
    "safe_execute",
    "safe_operation",
    "timeout_decorator",
    "operation_context",
    # カスタム例外
    "TimeoutError",
    "ValidationError",
    "DataError",
    "ModelError",
    # データ変換
    "OHLCVDataConverter",
    "FundingRateDataConverter",
    "OpenInterestDataConverter",
    "DataConversionError",
    "parse_timestamp_safe",
    # ログフィルター
    "DuplicateFilter",
]



