import logging
import random
from typing import List, Tuple, Union, Dict
from app.utils.error_handler import safe_operation
from ..constants import (
    IndicatorType,
)
from ..utils.indicator_characteristics import INDICATOR_CHARACTERISTICS
from app.services.indicators.config import indicator_registry
from ..utils.yaml_utils import YamlIndicatorUtils

from ..models.strategy_models import Condition, IndicatorGene, ConditionGroup
from .strategies import (
    ComplexConditionsStrategy,
)


logger = logging.getLogger(__name__)


class ConditionGenerator:
    """
    責務を集約したロング・ショート条件生成器

    計画書に基づいて以下の戦略を実装：
    1. 異なる指標の組み合わせ戦略
    2. 時間軸分離戦略
    3. 複合条件戦略
    4. 指標特性活用戦略
    """

    @safe_operation(context="ConditionGenerator初期化", is_api_call=False)
    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化（統合後）

        Args:
            enable_smart_generation: 新しいスマート生成を有効にするか
        """
        self.enable_smart_generation = enable_smart_generation
        self.logger = logger

        # YAML設定を読み込み
        self.yaml_config = YamlIndicatorUtils.load_yaml_config_for_indicators()

        self.context = {
            "timeframe": None,
            "symbol": None,
            "regime_gating": False,
            "threshold_profile": "normal",
        }

        # geneに含まれる指標一覧をオプションで保持
        self.indicators: List[IndicatorGene] | None = None

    @safe_operation(context="コンテキスト設定", is_api_call=False)
    def set_context(
        self,
        *,
        timeframe: str | None = None,
        symbol: str | None = None,
        regime_gating: bool | None = None,
        threshold_profile: str | None = None,
    ):
        """生成コンテキストを設定（RSI閾値やレジーム切替に利用）"""
        if timeframe is not None:
            self.context["timeframe"] = timeframe
        if symbol is not None:
            self.context["symbol"] = symbol
        if regime_gating is not None:
            self.context["regime_gating"] = bool(regime_gating)
        if threshold_profile is not None:
            if threshold_profile not in ("aggressive", "normal", "conservative"):
                threshold_profile = "normal"
            self.context["threshold_profile"] = threshold_profile

    @safe_operation(context="バランス条件生成", is_api_call=False)
    def generate_balanced_conditions(self, indicators: List[IndicatorGene]) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        バランスの取れたロング・ショート条件を生成

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        if not self.enable_smart_generation:
            raise ValueError("スマート生成が無効です")

        if not indicators:
            raise ValueError("指標リストが空です")

        # 統合された戦略を使用（ComplexConditionsStrategy）
        strategy = ComplexConditionsStrategy(self)
        longs, shorts, exits = strategy.generate_conditions(indicators)

        # 条件数を3個以内に制限
        if len(longs) > 3:
            longs = random.sample(longs, 3) if len(longs) >= 3 else longs
        if len(shorts) > 3:
            shorts = random.sample(shorts, 3) if len(shorts) >= 3 else shorts

        # 条件が生成できなかった場合は例外を投げる
        if not longs:
            raise RuntimeError("ロング条件を生成できませんでした")
        if not shorts:
            raise RuntimeError("ショート条件を生成できませんでした")

        # 型を明示的に変換して返す
        long_conditions: List[Union[Condition, ConditionGroup]] = list(longs)
        short_conditions: List[Union[Condition, ConditionGroup]] = list(shorts)
        exit_conditions: List[Condition] = list(exits)

        return long_conditions, short_conditions, exit_conditions

    @safe_operation(context="戦略タイプ選択", is_api_call=False)
    def _generic_long_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """統合された汎用ロング条件生成"""
        self.logger.debug(f"Generating long conditions for {ind.type}")
        config = YamlIndicatorUtils.get_indicator_config_from_yaml(
            self.yaml_config, ind.type
        )
        self.logger.debug(f"Config for {ind.type}: {config}")
        if config:
            threshold = YamlIndicatorUtils.get_threshold_from_yaml(
                self.yaml_config, config, "long", self.context
            )
            if threshold is not None:
                self.logger.debug(f"Using threshold {threshold} for {ind.type}")
                # DEBUG: 条件生成詳細ログ
                final_condition = Condition(
                    left_operand=ind.type, operator=">", right_operand=threshold
                )
                self.logger.debug(f"Generated long condition: {ind.type} > {threshold}")
                return [final_condition]
        self.logger.warning(f"No threshold found for {ind.type}, using 0 as fallback")
        final_condition = Condition(
            left_operand=ind.type, operator=">", right_operand=0
        )
        self.logger.debug(f"Generated fallback long condition: {ind.type} > 0")
        return [final_condition]

    def _generic_short_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """統合された汎用ショート条件生成"""
        self.logger.debug(f"Generating short conditions for {ind.type}")
        config = YamlIndicatorUtils.get_indicator_config_from_yaml(
            self.yaml_config, ind.type
        )
        self.logger.debug(f"Config for {ind.type}: {config}")
        if config:
            threshold = YamlIndicatorUtils.get_threshold_from_yaml(
                self.yaml_config, config, "short", self.context
            )
            if threshold is not None:
                self.logger.debug(f"Using threshold {threshold} for {ind.type}")
                return [
                    Condition(
                        left_operand=ind.type, operator="<", right_operand=threshold
                    )
                ]
        self.logger.warning(f"No threshold found for {ind.type}, using 0 as fallback")
        return [Condition(left_operand=ind.type, operator="<", right_operand=0)]

    def _create_type_based_conditions(
        self, indicator: IndicatorGene, side: str
    ) -> List[Condition]:
        """統合された型別条件生成 - YAML設定優先"""

        # YAML設定チェック
        config = YamlIndicatorUtils.get_indicator_config_from_yaml(
            self.yaml_config, indicator.type
        )
        if config:
            threshold = YamlIndicatorUtils.get_threshold_from_yaml(
                self.yaml_config, config, side, self.context
            )
            if threshold is not None:
                final_condition = Condition(
                    left_operand=indicator.type,
                    operator=">" if side == "long" else "<",
                    right_operand=threshold,
                )
                self.logger.debug(
                    f"YAML-based {side} condition for {indicator.type}: {indicator.type} {'>' if side == 'long' else '<'} {threshold}"
                )
                return [final_condition]

        # デフォルト
        final_condition = Condition(
            left_operand=indicator.type,
            operator=">" if side == "long" else "<",
            right_operand=0,
        )
        self.logger.debug(
            f"Default {side} condition for {indicator.type}: {indicator.type} {'>' if side == 'long' else '<'} 0"
        )
        return [final_condition]

    def _create_trend_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """統合されたトレンド系ロング条件生成"""
        return self._create_type_based_conditions(indicator, "long")

    def _create_trend_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """統合されたトレンド系ショート条件生成"""
        return self._create_type_based_conditions(indicator, "short")

    def _create_momentum_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """統合されたモメンタム系ロング条件生成"""
        return self._create_type_based_conditions(indicator, "long")

    def _create_momentum_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """統合されたモメンタム系ショート条件生成"""
        return self._create_type_based_conditions(indicator, "short")

    @safe_operation(context="指標タイプ取得", is_api_call=False)
    def _get_indicator_type(self, indicator: IndicatorGene) -> IndicatorType:
        """指標のタイプを取得"""
        config = YamlIndicatorUtils.get_indicator_config_from_yaml(
            self.yaml_config, indicator.type
        )
        if config and "type" in config:
            type_str = config["type"]
            if type_str == "momentum":
                return IndicatorType.MOMENTUM
            elif type_str == "trend":
                return IndicatorType.TREND
            elif type_str == "volatility":
                return IndicatorType.VOLATILITY

        cfg = indicator_registry.get_indicator_config(indicator.type)
        if cfg and hasattr(cfg, "category") and getattr(cfg, "category", None):
            cat = getattr(cfg, "category")
            if cat == "momentum":
                return IndicatorType.MOMENTUM
            elif cat == "trend":
                return IndicatorType.TREND

        if indicator.type in INDICATOR_CHARACTERISTICS:
            return INDICATOR_CHARACTERISTICS[indicator.type]["type"]

        raise ValueError(f"不明な指標タイプ: {indicator.type}")

    @safe_operation(context="動的指標分類", is_api_call=False)
    def _dynamic_classify(
        self, indicators: List[IndicatorGene]
    ) -> Dict[IndicatorType, List[IndicatorGene]]:
        """
        動的指標分類
        """
        categorized = {
            IndicatorType.MOMENTUM: [],
            IndicatorType.TREND: [],
            IndicatorType.VOLATILITY: [],
        }

        for ind in indicators:
            if not ind.enabled:
                continue

            name = ind.type
            cfg = indicator_registry.get_indicator_config(name)

            if cfg and hasattr(cfg, "category") and getattr(cfg, "category", None):
                cat = getattr(cfg, "category")
                if cat == "momentum":
                    categorized[IndicatorType.MOMENTUM].append(ind)
                elif cat == "trend":
                    categorized[IndicatorType.TREND].append(ind)
                elif cat == "volatility":
                    categorized[IndicatorType.VOLATILITY].append(ind)
                else:
                    categorized[IndicatorType.TREND].append(ind)
            elif name in INDICATOR_CHARACTERISTICS:
                char = INDICATOR_CHARACTERISTICS[name]
                categorized[char["type"]].append(ind)
            else:
                raise ValueError(f"分類できない指標タイプ: {name}")

        return categorized
