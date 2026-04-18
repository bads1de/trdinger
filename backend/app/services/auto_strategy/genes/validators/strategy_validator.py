"""
戦略遺伝子バリデーター
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from app.utils.error_handler import safe_operation

logger = logging.getLogger(__name__)


class StrategyValidator:
    """戦略遺伝子の妥当性検証を担当します。"""

    def __init__(self, indicator_validator, condition_validator) -> None:
        """初期化"""
        self.indicator_validator = indicator_validator
        self.condition_validator = condition_validator

    @safe_operation(
        context="戦略遺伝子バリデーション",
        is_api_call=False,
        default_return=(False, ["バリデーションエラー"]),
    )
    def validate_strategy_gene(
        self, strategy_gene, config: Optional[Any] = None
    ) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証"""
        errors: List[str] = []

        # 1. 指標数の制約チェック
        from ...config import GAConfig

        effective_config = config if config is not None else GAConfig()
        max_indicators = effective_config.max_indicators
        if len(strategy_gene.indicators) > max_indicators:
            errors.append(
                f"指標数が上限({max_indicators})を超えています: {len(strategy_gene.indicators)}"
            )

        # 2. 個別指標の妥当性
        for i, indicator in enumerate(strategy_gene.indicators):
            if not self.indicator_validator.validate_indicator_gene(indicator):
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

        # 4. TP/SL設定の検証
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
        from ..conditions import ConditionGroup

        def _recursive(cond, path: str):
            if isinstance(cond, ConditionGroup):
                for j, sub in enumerate(cond.conditions):
                    _recursive(sub, f"{path} -> グループ{j}")
            else:
                self.condition_validator.clean_condition(cond)
                valid, detail = self.condition_validator.validate_condition(cond)
                if not valid:
                    errors.append(f"{path}が無効: {detail}")

        for i, condition in enumerate(cond_list):
            _recursive(condition, f"{label}エントリー条件{i}")
