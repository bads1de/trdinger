"""
遺伝子バリデーター
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from app.utils.error_handler import safe_operation

from ..utils.indicator_utils import get_all_indicators
from ..config.constants import (
    CONSTRAINTS,
    ERROR_CODES,
    POSITION_SIZING_LIMITS,
    TPSL_LIMITS,
    OPERATORS,
    DATA_SOURCES,
)

logger = logging.getLogger(__name__)


class GeneValidator:
    """
    遺伝子バリデーター

    戦略遺伝子の妥当性検証を担当します。
    """

    def __init__(self) -> None:
        """初期化"""
        # 定数ではなくユーティリティの動的取得を使用
        self.valid_indicator_types = get_all_indicators()
        self.valid_operators = OPERATORS
        self.valid_data_sources = DATA_SOURCES

    @safe_operation(
        context="指標遺伝子バリデーション", is_api_call=False, default_return=False
    )
    def validate_indicator_gene(self, indicator_gene) -> bool:
        """指標遺伝子の妥当性を検証"""
        if not indicator_gene.type or not isinstance(indicator_gene.type, str):
            logger.warning(f"指標タイプが無効: {indicator_gene.type}")

            return False
        if not isinstance(indicator_gene.parameters, dict):
            logger.warning(f"指標パラメータが無効: {indicator_gene.parameters}")
            return False

        # ログ: 指標タイプが有効リストに含まれているかを確認
        logger.debug(
            f"指標タイプ {indicator_gene.type} が valid_indicator_types に含まれているか: {indicator_gene.type in self.valid_indicator_types}"
        )
        if indicator_gene.type not in self.valid_indicator_types:
            logger.warning(
                f"無効な指標タイプ: {indicator_gene.type}, 有効なタイプ: {self.valid_indicator_types[:10]}..."
            )  # 先頭10個のみ表示
            return False

        if "period" in indicator_gene.parameters:
            period = indicator_gene.parameters["period"]
            if not isinstance(period, (int, float)) or period <= 0:
                logger.warning(f"無効な期間パラメータ: {period}")
                return False

        logger.debug(f"指標タイプ {indicator_gene.type} は有効です")
        return True

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
        context="オペランド検証",
        is_api_call=False,
        default_return=(False, "オペランド検証エラー"),
    )
    def _is_valid_operand_detailed(self, operand) -> Tuple[bool, str]:
        """オペランドの妥当性を詳細に検証"""
        try:
            if operand is None:
                return False, "オペランドがNoneです"

            if isinstance(operand, (int, float)):
                return True, ""

            if isinstance(operand, str):
                if not operand or not operand.strip():
                    return False, "オペランドが空文字列です"

                operand = operand.strip()

                try:
                    float(operand)
                    return True, ""
                except ValueError:
                    pass

                if (
                    self._is_indicator_name(operand)
                    or operand in self.valid_data_sources
                ):
                    return True, ""

                return False, f"無効な文字列オペランド: '{operand}'"

            if isinstance(operand, dict):
                return self._validate_dict_operand_detailed(operand)

            return False, f"サポートされていないオペランド型: {type(operand)}"
        except Exception as e:
            return False, f"オペランド検証エラー: {e}"

    @safe_operation(
        context="辞書オペランド検証",
        is_api_call=False,
        default_return=(False, "辞書検証エラー"),
    )
    def _validate_dict_operand_detailed(self, operand: dict) -> Tuple[bool, str]:
        """辞書形式のオペランドを詳細に検証"""
        try:
            if operand.get("type") == "indicator":
                indicator_name = operand.get("name")
                if not indicator_name or not isinstance(indicator_name, str):
                    return False, "指標タイプの辞書にnameが設定されていません"
                if self._is_indicator_name(indicator_name.strip()):
                    return True, ""
                else:
                    return False, f"無効な指標名: '{indicator_name}'"

            elif operand.get("type") == "price":
                price_name = operand.get("name")
                if not price_name or not isinstance(price_name, str):
                    return False, "価格タイプの辞書にnameが設定されていません"
                if price_name.strip() in self.valid_data_sources:
                    return True, ""
                else:
                    return False, f"無効な価格データソース: '{price_name}'"

            elif operand.get("type") == "value":
                value = operand.get("value")
                if value is None:
                    return False, "数値タイプの辞書にvalueが設定されていません"
                if isinstance(value, (int, float)):
                    return True, ""
                elif isinstance(value, str):
                    try:
                        float(value.strip())
                        return True, ""
                    except ValueError:
                        return False, f"数値に変換できない文字列: '{value}'"
                else:
                    return False, f"無効な数値型: {type(value)}"

            else:
                return False, f"無効な辞書タイプ: '{operand.get('type')}'"
        except Exception as e:
            return False, f"辞書オペランド検証エラー: {e}"

    @safe_operation(context="指標名判定", is_api_call=False, default_return=False)
    def _is_indicator_name(self, name: str) -> bool:
        """指標名かどうかを判定"""
        if not name or not name.strip():
            return False

        name = name.strip()

        if name in self.valid_indicator_types:
            return True

        if "_" in name:
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                potential_indicator = parts[0].strip()
                potential_param = parts[1].strip()

                try:
                    float(potential_param)
                    if potential_indicator in self.valid_indicator_types:
                        return True
                except ValueError:
                    pass

            indicator_type = name.split("_")[0].strip()
            if indicator_type in self.valid_indicator_types:
                return True

        if name.endswith(("_0", "_1", "_2", "_3", "_4")):
            indicator_type = name.rsplit("_", 1)[0].strip()
            if indicator_type in self.valid_indicator_types:
                return True

        return False

    @safe_operation(context="条件クリーニング", is_api_call=False, default_return=False)
    def clean_condition(self, condition) -> bool:
        """条件をクリーニングして修正可能な問題を自動修正"""
        if isinstance(condition.left_operand, str):
            condition.left_operand = condition.left_operand.strip()

        if isinstance(condition.right_operand, str):
            condition.right_operand = condition.right_operand.strip()

        if isinstance(condition.left_operand, dict):
            condition.left_operand = self._extract_operand_from_dict(
                condition.left_operand
            )

        if isinstance(condition.right_operand, dict):
            condition.right_operand = self._extract_operand_from_dict(
                condition.right_operand
            )

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
        context="戦略遺伝子バリデーション",
        is_api_call=False,
        default_return=(False, ["バリデーションエラー"]),
    )
    def validate_strategy_gene(self, strategy_gene) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証"""
        from .condition import ConditionGroup

        errors: List[str] = []

        # 指標数の制約チェック
        max_indicators = getattr(strategy_gene, "MAX_INDICATORS", 5)
        if len(strategy_gene.indicators) > max_indicators:
            errors.append(
                f"指標数が上限({max_indicators})を超えています: {len(strategy_gene.indicators)}"
            )

        # 指標の妥当性チェック
        for i, indicator in enumerate(strategy_gene.indicators):
            if not self.validate_indicator_gene(indicator):
                errors.append(f"指標{i}が無効です: {indicator.type}")

        # 条件の妥当性チェック
        def _validate_mixed_conditions(cond_list, label_prefix: str):
            for i, condition in enumerate(cond_list):
                if isinstance(condition, ConditionGroup):
                    for j, c in enumerate(condition.conditions):
                        self.clean_condition(c)
                        is_valid, error_detail = self.validate_condition(c)
                        if not is_valid:
                            errors.append(
                                f"{label_prefix}OR子条件{j}が無効です: {error_detail}"
                            )
                else:
                    self.clean_condition(condition)
                    is_valid, error_detail = self.validate_condition(condition)
                    if not is_valid:
                        errors.append(f"{label_prefix}{i}が無効です: {error_detail}")

        _validate_mixed_conditions(strategy_gene.entry_conditions, "エントリー条件")
        _validate_mixed_conditions(
            strategy_gene.long_entry_conditions, "ロングエントリー条件"
        )
        _validate_mixed_conditions(
            strategy_gene.short_entry_conditions, "ショートエントリー条件"
        )

        for i, condition in enumerate(strategy_gene.exit_conditions):
            self.clean_condition(condition)
            is_valid, error_detail = self.validate_condition(condition)
            if not is_valid:
                errors.append(f"イグジット条件{i}が無効です: {error_detail}")

        # 最低限の条件チェック
        has_entry_conditions = (
            bool(strategy_gene.entry_conditions)
            or bool(strategy_gene.long_entry_conditions)
            or bool(strategy_gene.short_entry_conditions)
        )
        if not has_entry_conditions:
            errors.append("エントリー条件が設定されていません")

        # イグジット条件またはTP/SL遺伝子の存在チェック
        if not strategy_gene.exit_conditions:
            if not (strategy_gene.tpsl_gene and strategy_gene.tpsl_gene.enabled):
                errors.append("イグジット条件が設定されていません")

        # 有効な指標の存在チェック
        enabled_indicators = [ind for ind in strategy_gene.indicators if ind.enabled]
        if not enabled_indicators:
            errors.append("有効な指標が設定されていません")

        return len(errors) == 0, errors

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

        # 基本的な価格データ
        price_fields = {"close", "open", "high", "low"}

        # closeとopenの直接比較はトリビアル
        if (
            left in price_fields
            and right in price_fields
            and operator in [">", "<", ">=", "<="]
        ):
            return True

        # 同じ価格データの比較（例: close > close）は常にトリビアル
        if left == right and left in price_fields:
            return True

        # 非常にシンプルな数値比較（例: close > 1.0 など、意味のない閾値）
        if left in price_fields and isinstance(right, (int, float)):
            # 価格比として意味のない値（1.0 や 0.0 など）
            if operator in [">", "<"] and (
                right == 1.0 or right == 0.0 or abs(right) > 10
            ):
                return True

        return False
