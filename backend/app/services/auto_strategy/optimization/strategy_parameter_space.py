"""
戦略パラメータ空間

StrategyGene から Optuna の探索空間を動的に構築します。
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Union

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
    indicator_registry,
)
from app.services.ml.optimization.optuna_optimizer import ParameterSpace

from ..genes.conditions import Condition, ConditionGroup
from ..genes.indicator import IndicatorGene
from ..genes.strategy import StrategyGene
from ..genes.tpsl import TPSLGene

logger = logging.getLogger(__name__)


class StrategyParameterSpace:
    """
    StrategyGene から Optuna パラメータ空間を構築

    GAで発見された戦略構造（インジケーターの組み合わせ、条件構造）に対して、
    Optunaで最適化可能なパラメータ空間を動的に生成します。
    """

    # インジケーターパラメータのデフォルト範囲
    DEFAULT_INDICATOR_RANGES: Dict[str, Dict[str, Any]] = {
        "length": {"low": 5, "high": 100, "type": "integer"},
        "period": {"low": 5, "high": 100, "type": "integer"},
        "fast": {"low": 5, "high": 50, "type": "integer"},
        "slow": {"low": 10, "high": 100, "type": "integer"},
        "signal": {"low": 5, "high": 30, "type": "integer"},
        "k_length": {"low": 5, "high": 30, "type": "integer"},
        "d_length": {"low": 2, "high": 10, "type": "integer"},
        "smooth_k": {"low": 1, "high": 5, "type": "integer"},
        "std": {"low": 1.0, "high": 4.0, "type": "real"},
        "multiplier": {"low": 1.0, "high": 5.0, "type": "real"},
    }

    # TPSLパラメータの範囲
    TPSL_RANGES: Dict[str, Dict[str, Any]] = {
        "stop_loss_pct": {"low": 0.005, "high": 0.10, "type": "real"},
        "take_profit_pct": {"low": 0.01, "high": 0.20, "type": "real"},
        "risk_reward_ratio": {"low": 1.0, "high": 5.0, "type": "real"},
        "atr_multiplier_sl": {"low": 0.5, "high": 4.0, "type": "real"},
        "atr_multiplier_tp": {"low": 1.0, "high": 6.0, "type": "real"},
        "atr_period": {"low": 7, "high": 28, "type": "integer"},
        "trailing_step_pct": {"low": 0.005, "high": 0.03, "type": "real"},
    }

    # 条件の閾値パラメータの範囲（スケールタイプ別）
    THRESHOLD_RANGES: Dict[str, Dict[str, Any]] = {
        "oscillator_0_100": {"low": 10.0, "high": 90.0, "type": "real"},
        "oscillator_plus_minus_100": {"low": -80.0, "high": 80.0, "type": "real"},
        "momentum_zero_centered": {"low": -1.0, "high": 1.0, "type": "real"},
        "default": {"low": -100.0, "high": 100.0, "type": "real"},
    }

    def __init__(self):
        """初期化"""
        pass

    def build_parameter_space(
        self,
        gene: StrategyGene,
        include_indicators: bool = True,
        include_tpsl: bool = True,
        include_thresholds: bool = False,
    ) -> Dict[str, ParameterSpace]:
        """
        StrategyGene からパラメータ空間を構築

        Args:
            gene: 対象の戦略遺伝子
            include_indicators: インジケーターパラメータを含めるか
            include_tpsl: TPSLパラメータを含めるか
            include_thresholds: 条件の閾値パラメータを含めるか

        Returns:
            Optuna用パラメータ空間の辞書
        """
        parameter_space: Dict[str, ParameterSpace] = {}

        if include_indicators:
            indicator_params = self._build_indicator_params(gene.indicators)
            parameter_space.update(indicator_params)

        if include_tpsl:
            # 各TPSL遺伝子をループで処理
            tpsl_fields = [
                ("tpsl_gene", "tpsl"),
                ("long_tpsl_gene", "long_tpsl"),
                ("short_tpsl_gene", "short_tpsl")
            ]
            for field_name, prefix in tpsl_fields:
                gene_obj = getattr(gene, field_name, None)
                if gene_obj:
                    parameter_space.update(self._build_tpsl_params(gene_obj, prefix=prefix))

        if include_thresholds:
            # ロング条件の閾値
            long_threshold_params = self._build_threshold_params(
                gene.long_entry_conditions, prefix="long"
            )
            parameter_space.update(long_threshold_params)

            # ショート条件の閾値
            short_threshold_params = self._build_threshold_params(
                gene.short_entry_conditions, prefix="short"
            )
            parameter_space.update(short_threshold_params)

        logger.debug(f"構築されたパラメータ空間: {len(parameter_space)} 個のパラメータ")
        return parameter_space

    def _build_indicator_params(
        self, indicators: List[IndicatorGene]
    ) -> Dict[str, ParameterSpace]:
        """インジケーターパラメータ空間を構築"""
        params: Dict[str, ParameterSpace] = {}

        for idx, indicator in enumerate(indicators):
            if not indicator.enabled:
                continue

            prefix = f"ind_{idx}"
            indicator_type = indicator.type

            # レジストリから設定を取得
            config = indicator_registry.get_indicator_config(indicator_type)
            param_ranges = self._get_indicator_param_ranges(config, indicator_type)

            for param_name, param_value in indicator.parameters.items():
                # 数値パラメータのみ対象
                if not isinstance(param_value, (int, float)):
                    continue

                range_info = param_ranges.get(
                    param_name, self.DEFAULT_INDICATOR_RANGES.get(param_name)
                )
                if not range_info:
                    continue

                param_key = f"{prefix}_{param_name}"
                params[param_key] = ParameterSpace(
                    type=range_info.get("type", "integer"),
                    low=range_info.get("low", 2),
                    high=range_info.get("high", 100),
                )

        return params

    def _get_indicator_param_ranges(
        self, config: Optional[IndicatorConfig], indicator_type: str
    ) -> Dict[str, Dict[str, Any]]:
        """インジケーター設定からパラメータ範囲を取得"""
        ranges: Dict[str, Dict[str, Any]] = {}

        if config and config.parameters:
            for param_name, param_config in config.parameters.items():
                if hasattr(param_config, "min_value") and hasattr(
                    param_config, "max_value"
                ):
                    if (
                        param_config.min_value is not None
                        and param_config.max_value is not None
                    ):
                        # 整数か浮動小数点かを判定
                        default_val = getattr(param_config, "default_value", 14)
                        param_type = (
                            "integer" if isinstance(default_val, int) else "real"
                        )

                        ranges[param_name] = {
                            "low": param_config.min_value,
                            "high": param_config.max_value,
                            "type": param_type,
                        }

        return ranges

    def _build_tpsl_params(
        self, tpsl_gene: TPSLGene, prefix: str
    ) -> Dict[str, ParameterSpace]:
        """TPSLパラメータ空間を構築"""
        params: Dict[str, ParameterSpace] = {}

        for param_name, range_info in self.TPSL_RANGES.items():
            current_value = getattr(tpsl_gene, param_name, None)
            if current_value is None:
                continue

            param_key = f"{prefix}_{param_name}"
            params[param_key] = ParameterSpace(
                type=range_info["type"],
                low=range_info["low"],
                high=range_info["high"],
            )

        return params

    def _build_threshold_params(
        self,
        conditions: List[Union[Condition, ConditionGroup]],
        prefix: str,
    ) -> Dict[str, ParameterSpace]:
        """条件の閾値パラメータ空間を構築"""
        params: Dict[str, ParameterSpace] = {}
        threshold_idx = 0

        for condition in conditions:
            if isinstance(condition, ConditionGroup):
                # 再帰的に処理
                nested_params = self._build_threshold_params(
                    condition.conditions, f"{prefix}_grp{threshold_idx}"
                )
                params.update(nested_params)
                threshold_idx += 1
            elif isinstance(condition, Condition):
                # right_operand が数値の場合のみ対象
                if isinstance(condition.right_operand, (int, float)):
                    param_key = f"{prefix}_thresh_{threshold_idx}"
                    range_info = self.THRESHOLD_RANGES["default"]
                    params[param_key] = ParameterSpace(
                        type=range_info["type"],
                        low=range_info["low"],
                        high=range_info["high"],
                    )
                    threshold_idx += 1

        return params

    def apply_params_to_gene(
        self, gene: StrategyGene, params: Dict[str, Any]
    ) -> StrategyGene:
        """
        最適化されたパラメータを遺伝子に適用

        Args:
            gene: 元の戦略遺伝子
            params: 最適化されたパラメータ辞書

        Returns:
            パラメータが適用された新しい StrategyGene
        """
        # 深いコピーを作成して元の遺伝子を保護
        new_gene = copy.deepcopy(gene)

        # インジケーターパラメータを適用
        self._apply_indicator_params(new_gene.indicators, params)

        # TPSLパラメータを適用
        tpsl_fields = [
            ("tpsl_gene", "tpsl"),
            ("long_tpsl_gene", "long_tpsl"),
            ("short_tpsl_gene", "short_tpsl")
        ]
        for field_name, prefix in tpsl_fields:
            gene_obj = getattr(new_gene, field_name, None)
            if gene_obj:
                self._apply_tpsl_params(gene_obj, params, prefix=prefix)

        # 閾値パラメータを適用
        self._apply_threshold_params(
            new_gene.long_entry_conditions, params, prefix="long"
        )
        self._apply_threshold_params(
            new_gene.short_entry_conditions, params, prefix="short"
        )

        logger.debug(f"パラメータ適用完了: {len(params)} 個")
        return new_gene

    def _apply_indicator_params(
        self, indicators: List[IndicatorGene], params: Dict[str, Any]
    ) -> None:
        """インジケーターパラメータを適用"""
        for idx, indicator in enumerate(indicators):
            if not indicator.enabled:
                continue

            prefix = f"ind_{idx}"

            for param_name in list(indicator.parameters.keys()):
                param_key = f"{prefix}_{param_name}"
                if param_key in params:
                    # 元の型を維持
                    original_value = indicator.parameters[param_name]
                    new_value = params[param_key]

                    if isinstance(original_value, int):
                        indicator.parameters[param_name] = int(new_value)
                    else:
                        indicator.parameters[param_name] = float(new_value)

    def _apply_tpsl_params(
        self, tpsl_gene: TPSLGene, params: Dict[str, Any], prefix: str
    ) -> None:
        """TPSLパラメータを適用"""
        for param_name in self.TPSL_RANGES.keys():
            param_key = f"{prefix}_{param_name}"
            if param_key in params:
                setattr(tpsl_gene, param_name, params[param_key])

    def _apply_threshold_params(
        self,
        conditions: List[Union[Condition, ConditionGroup]],
        params: Dict[str, Any],
        prefix: str,
    ) -> None:
        """条件の閾値パラメータを適用"""
        threshold_idx = 0

        for condition in conditions:
            if isinstance(condition, ConditionGroup):
                self._apply_threshold_params(
                    condition.conditions, params, f"{prefix}_grp{threshold_idx}"
                )
                threshold_idx += 1
            elif isinstance(condition, Condition):
                if isinstance(condition.right_operand, (int, float)):
                    param_key = f"{prefix}_thresh_{threshold_idx}"
                    if param_key in params:
                        # 元の型を維持
                        if isinstance(condition.right_operand, int):
                            condition.right_operand = int(params[param_key])
                        else:
                            condition.right_operand = float(params[param_key])
                    threshold_idx += 1





