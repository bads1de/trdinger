"""
遺伝子バリデーター

バリデーターモジュールを統合して、後方互換性を維持します。
"""

from __future__ import annotations

from collections.abc import Collection
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..config.ga.ga_config import GAConfig

from .validators.condition_validator import ConditionValidator
from .validators.indicator_validator import IndicatorValidator
from .validators.strategy_validator import StrategyValidator


class GeneValidator:
    """
    遺伝子バリデーター

    戦略遺伝子の妥当性検証を担当します。
    内部的に IndicatorValidator, ConditionValidator, StrategyValidator を使用します。
    """

    def __init__(self) -> None:
        """初期化"""
        self._indicator_validator = IndicatorValidator()
        self._condition_validator = ConditionValidator(
            self._indicator_validator
        )
        self._strategy_validator = StrategyValidator(
            self._indicator_validator, self._condition_validator
        )

    def validate_indicator_gene(self, indicator_gene: object) -> bool:
        """指標遺伝子の妥当性を検証"""
        return self._indicator_validator.validate_indicator_gene(
            indicator_gene
        )

    def validate_indicator_gene_for_generation(
        self,
        indicator_gene: object,
        indicator_universe_mode: Union[str, Enum] = "curated",
        allowed_indicators: Optional[Collection[str]] = None,
    ) -> bool:
        """GA 生成・変異で使う指標遺伝子をユニバース込みで検証する。"""
        return (
            self._indicator_validator.validate_indicator_gene_for_generation(
                indicator_gene, indicator_universe_mode, allowed_indicators
            )
        )

    def validate_condition(self, condition: object) -> Tuple[bool, str]:
        """条件の妥当性を検証"""
        return self._condition_validator.validate_condition(condition)

    def clean_condition(self, condition: object) -> bool:
        """条件をクリーニングして修正可能な問題を自動修正"""
        return self._condition_validator.clean_condition(condition)

    def validate_strategy_gene(
        self, strategy_gene: object, config: Optional["GAConfig"] = None
    ) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証"""
        return self._strategy_validator.validate_strategy_gene(
            strategy_gene, config
        )
