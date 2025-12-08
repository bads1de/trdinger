"""
データバリデーションユーティリティ（非推奨）

このモジュールは後方互換性のために維持されています。
新しいコードでは以下を使用してください：
- レコード形式: app.utils.data_processing.validators.RecordValidator
- DataFrame形式: app.utils.data_processing.validators.validate_ohlcv_data 等
"""

import warnings

# 新しいモジュールからインポート
from app.utils.data_processing.validators.record_validator import (
    DataValidator,
    RecordValidator,
)

# 非推奨警告を発行するラッパークラス
_original_DataValidator = DataValidator


class DataValidator(_original_DataValidator):
    """
    後方互換性のためのラッパークラス

    .. deprecated::
        このクラスは非推奨です。
        代わりに `app.utils.data_processing.validators.RecordValidator` を使用してください。
    """

    @classmethod
    def validate_ohlcv_records_simple(cls, ohlcv_records):
        warnings.warn(
            "DataValidator は非推奨です。"
            "app.utils.data_processing.validators.RecordValidator を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )
        return _original_DataValidator.validate_ohlcv_records_simple(ohlcv_records)

    @classmethod
    def sanitize_ohlcv_data(cls, ohlcv_records):
        warnings.warn(
            "DataValidator は非推奨です。"
            "app.utils.data_processing.validators.RecordValidator を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )
        return _original_DataValidator.sanitize_ohlcv_data(ohlcv_records)


__all__ = ["DataValidator", "RecordValidator"]
