"""
正規化ユーティリティ関数

auto_strategy全体で使用されるEnum正規化の共通機能を提供します。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NormalizationUtils:
    """正規化ユーティリティクラス

    Enum名の正規化などの共通機能を提供します。
    """

    @staticmethod
    def normalize_enum_name(value: Any, default: str = "") -> str:
        """Enumまたは文字列を比較用の文字列に正規化

        Enum値や文字列、その他の値を文字列表現に変換します。
        Enum値の場合は .value 属性を抽出します。

        Args:
            value: 正規化する値（Enum、文字列、またはNone）。
            default: 値がNoneの場合に返すデフォルト文字列。

        Returns:
            str: 正規化された文字列。
        """
        if value is None:
            return default
        if hasattr(value, "value"):
            value = value.value
        return str(value)


# 外部で使用可能な便利関数
normalize_enum_name = NormalizationUtils.normalize_enum_name
