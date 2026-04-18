"""
条件バリデーター
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

from app.utils.error_handler import safe_operation

from ...config.constants import DATA_SOURCES, OPERATORS

logger = logging.getLogger(__name__)


class ConditionValidator:
    """条件の妥当性検証を担当します。"""

    def __init__(self, indicator_validator) -> None:
        """初期化"""
        self.indicator_validator = indicator_validator
        self.valid_operators = OPERATORS
        self.valid_data_sources = DATA_SOURCES

    @safe_operation(
        context="条件バリデーション",
        is_api_call=False,
        default_return=(False, "バリデーションエラー"),
    )
    def validate_condition(self, condition) -> Tuple[bool, str]:
        """条件の妥当性を検証"""
        if not all(
            hasattr(condition, attr)
            for attr in ["operator", "left_operand", "right_operand"]
        ):
            return False, "条件オブジェクトに必要な属性がありません"

        if condition.operator not in self.valid_operators:
            return False, f"無効な演算子: {condition.operator}"

        left_valid, left_error = self._is_valid_operand_detailed(condition.left_operand)
        if not left_valid:
            return False, f"無効な左オペランド: {left_error}"

        right_valid, right_error = self._is_valid_operand_detailed(
            condition.right_operand
        )
        if not right_valid:
            return False, f"無効な右オペランド: {right_error}"

        # シンプルな価格比較を排除
        if self._is_trivial_condition(condition):
            return False, "シンプルな価格比較は禁止されています"

        return True, ""

    @safe_operation(
        context="オペランド検証", is_api_call=False, default_return=(False, "エラー")
    )
    def _is_valid_operand_detailed(self, operand: Any) -> Tuple[bool, str]:
        """オペランドの妥当性を詳細に検証"""
        if operand is None:
            return False, "オペランドがNoneです"

        if isinstance(operand, (int, float, np.number)):
            return True, ""

        if isinstance(operand, str):
            val = operand.strip()
            if not val:
                return False, "空のオペランド"
            # 数値、指標名、データソースのいずれかならOK
            try:
                float(val)
                return True, ""
            except ValueError:
                # データソースは大文字小文字を区別せずチェック
                if val.lower() in [ds.lower() for ds in self.valid_data_sources]:
                    return True, ""
                if self._is_indicator_name(val):
                    return True, ""
            return False, f"無効な文字列オペランド: '{val}'"

        if isinstance(operand, dict):
            op_type = operand.get("type")
            if op_type == "indicator":
                name = str(operand.get("name", "")).strip()
                if self._is_indicator_name(name):
                    return True, ""
                return False, f"無効な指標名: '{name}'"
            elif op_type == "price":
                name = str(operand.get("name", "")).strip()
                if name in self.valid_data_sources:
                    return True, ""
                return False, f"無効なデータソース: '{name}'"
            elif op_type == "value":
                v = operand.get("value")
                try:
                    if v is not None:
                        float(v)
                    return True, ""
                except (ValueError, TypeError):
                    return False, f"無効な数値: '{v}'"
            return False, f"無効な辞書タイプ: '{op_type}'"

        return False, f"未対応の型: {type(operand)}"

    def _is_indicator_name(self, name: str) -> bool:
        """文字列が有効な指標名（またはその派生）かを判定"""
        if not name:
            return False
        clean_name = name.strip()

        valid_indicator_types = self.indicator_validator.valid_indicator_types

        # 一般的なインジケータ出力接頭辞を明示的に許可
        allowed_prefixes = [
            "DMP",
            "DMN",
            "BBU",
            "BBL",
            "MACDS",
            "MACDH",
            "KAMA",
            "T3",
            "RSI",
            "ADX",
            "ATR",
        ]
        upper_name = clean_name.upper()
        for prefix in allowed_prefixes:
            if upper_name.startswith(prefix):
                return True

        # 1. 完全一致
        if clean_name in valid_indicator_types:
            return True

        # 2. アンダースコアで分割して接頭辞を試行
        parts = clean_name.split("_")
        for i in range(len(parts), 0, -1):
            prefix = "_".join(parts[:i])
            if prefix in valid_indicator_types:
                return True

        # 3. 大文字小文字を無視して試行
        upper_types = [t.upper() for t in valid_indicator_types]
        if clean_name.upper() in upper_types:
            return True

        upper_parts = clean_name.upper().split("_")
        for i in range(len(upper_parts), 0, -1):
            prefix = "_".join(upper_parts[:i])
            if prefix in upper_types:
                return True

        return False

    @safe_operation(context="条件クリーニング", is_api_call=False, default_return=False)
    def clean_condition(self, condition) -> bool:
        """条件をクリーニングして修正可能な問題を自動修正"""
        from ..conditions import ConditionGroup

        if isinstance(condition, ConditionGroup):
            return True

        if hasattr(condition, "left_operand") and isinstance(
            condition.left_operand, str
        ):
            condition.left_operand = condition.left_operand.strip()

        if hasattr(condition, "right_operand") and isinstance(
            condition.right_operand, str
        ):
            condition.right_operand = condition.right_operand.strip()

        if hasattr(condition, "left_operand") and isinstance(
            condition.left_operand, dict
        ):
            condition.left_operand = self._extract_operand_from_dict(
                condition.left_operand
            )

        if hasattr(condition, "right_operand") and isinstance(
            condition.right_operand, dict
        ):
            condition.right_operand = self._extract_operand_from_dict(
                condition.right_operand
            )

        if hasattr(condition, "operator"):
            if condition.operator == "above":
                condition.operator = ">"
            elif condition.operator == "below":
                condition.operator = "<"

        return True

    @safe_operation(context="辞書オペランド抽出", is_api_call=False, default_return="")
    def _extract_operand_from_dict(self, operand_dict: dict) -> str:
        """辞書形式のオペランドから文字列を抽出"""
        if operand_dict.get("type") == "indicator":
            return operand_dict.get("name", "")
        elif operand_dict.get("type") == "price":
            return operand_dict.get("name", "")
        elif operand_dict.get("type") == "value":
            value = operand_dict.get("value")
            return str(value) if value is not None else ""
        else:
            return str(operand_dict.get("name", ""))

    @safe_operation(
        context="条件のトリビアル判定", is_api_call=False, default_return=False
    )
    def _is_trivial_condition(self, condition) -> bool:
        """条件がシンプルすぎる（トリビアル）かどうかを判定"""
        if not hasattr(condition, "left_operand") or not hasattr(
            condition, "right_operand"
        ):
            return False

        left = condition.left_operand
        right = condition.right_operand
        operator = condition.operator

        price_fields = {"close", "open", "high", "low"}

        # 同じ価格データの比較は常にトリビアル
        if left == right and left in price_fields:
            return True

        # 非常にシンプルな数値比較
        if left in price_fields and isinstance(right, (int, float)):
            if operator in [">", "<"] and (
                right == 1.0 or right == 0.0 or abs(right) > 10
            ):
                return True

        return False
