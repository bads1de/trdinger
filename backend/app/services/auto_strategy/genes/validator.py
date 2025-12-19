"""
遺伝子バリデーター
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Any

import numpy as np

from app.utils.error_handler import safe_operation

from ..config.constants import (
    DATA_SOURCES,
    OPERATORS,
)
from ..utils.indicator_utils import get_all_indicators

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

        # タイムフレームのバリデーション（設定されている場合のみ）
        timeframe = getattr(indicator_gene, "timeframe", None)
        if timeframe is not None:
            from ..config.constants import SUPPORTED_TIMEFRAMES

            if timeframe not in SUPPORTED_TIMEFRAMES:
                logger.warning(
                    f"無効なタイムフレーム: {timeframe}, "
                    f"サポートされるタイムフレーム: {SUPPORTED_TIMEFRAMES}"
                )
                return False

        # パラメータ依存関係制約のバリデーション
        from app.services.indicators.config import indicator_registry

        config = indicator_registry.get_indicator_config(indicator_gene.type)
        if config and config.parameter_constraints:
            is_valid, errors = config.validate_constraints(indicator_gene.parameters)
            if not is_valid:
                logger.warning(f"パラメータ制約違反 ({indicator_gene.type}): {errors}")
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
        context="オペランド検証", is_api_call=False, default_return=(False, "エラー")
    )
    def _is_valid_operand_detailed(self, operand: Any) -> Tuple[bool, str]:
        """
        オペランドの妥当性を詳細に検証
        """
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
        """
        文字列が有効な指標名（またはその派生）かを判定
        """
        if not name:
            return False
        clean_name = name.strip()
        
        # 1. 完全一致
        if clean_name in self.valid_indicator_types:
            return True

        # 2. アンダースコアで分割して接頭辞を試行
        # LIQUIDATION_CASCADE_xxxx -> LIQUIDATION_CASCADE を探す
        parts = clean_name.split("_")
        for i in range(len(parts), 0, -1):
            prefix = "_".join(parts[:i])
            if prefix in self.valid_indicator_types:
                return True

        # 3. 大文字小文字を無視して試行
        upper_types = [t.upper() for t in self.valid_indicator_types]
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
        from .conditions import ConditionGroup
        
        # ConditionGroup の場合はスキップ（内部の条件は _validate_all_conditions で再帰的に処理される）
        if isinstance(condition, ConditionGroup):
            return True

        if hasattr(condition, "left_operand") and isinstance(condition.left_operand, str):
            condition.left_operand = condition.left_operand.strip()

        if hasattr(condition, "right_operand") and isinstance(condition.right_operand, str):
            condition.right_operand = condition.right_operand.strip()

        if hasattr(condition, "left_operand") and isinstance(condition.left_operand, dict):
            condition.left_operand = self._extract_operand_from_dict(
                condition.left_operand
            )

        if hasattr(condition, "right_operand") and isinstance(condition.right_operand, dict):
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
        context="戦略遺伝子バリデーション",
        is_api_call=False,
        default_return=(False, ["バリデーションエラー"]),
    )
    def validate_strategy_gene(self, strategy_gene) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証"""
        errors: List[str] = []

        # 1. 指標数の制約チェック
        from ..config import GAConfig

        max_indicators = GAConfig().max_indicators
        if len(strategy_gene.indicators) > max_indicators:
            errors.append(
                f"指標数が上限({max_indicators})を超えています: {len(strategy_gene.indicators)}"
            )

        # 2. 個別指標の妥当性
        for i, indicator in enumerate(strategy_gene.indicators):
            if not self.validate_indicator_gene(indicator):
                errors.append(f"指標{i}が無効です: {indicator.type}")

        # 3. 条件の妥当性（ロング・ショート）
        self._validate_all_conditions(
            strategy_gene.long_entry_conditions, "ロング", errors
        )
        self._validate_all_conditions(
            strategy_gene.short_entry_conditions, "ショート", errors
        )

        if not (
            strategy_gene.long_entry_conditions or strategy_gene.short_entry_conditions
        ):
            errors.append("エントリー条件が設定されていません")

        # 4. TP/SL設定の検証（ループで一括処理）
        sub_genes = [
            ("共通", strategy_gene.tpsl_gene),
            ("ロング", strategy_gene.long_tpsl_gene),
            ("ショート", strategy_gene.short_tpsl_gene),
        ]
        has_any_tpsl = False
        for label, gene in sub_genes:
            if gene and gene.enabled:
                has_any_tpsl = True
                valid, sub_errors = gene.validate()
                if not valid:
                    errors.extend([f"{label}TP/SL: {e}" for e in sub_errors])

        if not has_any_tpsl:
            errors.append("有効なTP/SL設定（イグジット条件）がありません")

        # 5. 有効な指標の存在
        if not any(ind.enabled for ind in strategy_gene.indicators):
            errors.append("有効な指標が設定されていません")

        return len(errors) == 0, errors

    def _validate_all_conditions(self, cond_list: List, label: str, errors: List[str]):
        """条件リストを再帰的に検証"""
        from .conditions import ConditionGroup

        def _recursive(cond, path: str):
            if isinstance(cond, ConditionGroup):
                for j, sub in enumerate(cond.conditions):
                    _recursive(sub, f"{path} -> グループ{j}")
            else:
                self.clean_condition(cond)
                valid, detail = self.validate_condition(cond)
                if not valid:
                    errors.append(f"{path}が無効: {detail}")

        for i, condition in enumerate(cond_list):
            _recursive(condition, f"{label}エントリー条件{i}")

    @safe_operation(
        context="条件のトリビアル判定", is_api_call=False, default_return=False
    )
    def _is_trivial_condition(self, condition) -> bool:
        """条件がシンプルすぎる（トリビアル）かどうかを判定

        緩和ルール:
        - 異なる価格データ同士の比較（close > open, close > high[1] など）は
          プライスアクションとして有効なため許容する
        - 同一価格データの比較（close > close）は数学的に無意味なので排除
        - 意味のない閾値比較（close > 1.0 など）は排除
        """
        if not hasattr(condition, "left_operand") or not hasattr(
            condition, "right_operand"
        ):
            return False

        left = condition.left_operand
        right = condition.right_operand
        operator = condition.operator

        # 基本的な価格データ
        price_fields = {"close", "open", "high", "low"}

        # 同じ価格データの比較（例: close > close）は常にトリビアル
        # 注意: 異なる価格データの比較（close > open など）はプライスアクションとして有効なので許容
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
