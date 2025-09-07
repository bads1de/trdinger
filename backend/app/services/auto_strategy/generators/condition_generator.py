import logging
import random
import copy
from typing import List, Tuple, Union, Dict, Any, Optional
from app.services.auto_strategy.config.constants import (
    INDICATOR_CHARACTERISTICS,
    IndicatorType,
    StrategyType,
)
from app.services.indicators.config import indicator_registry
from ..utils.common_utils import YamlIndicatorUtils

from ..models.strategy_models import Condition, IndicatorGene, ConditionGroup


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

        # 生成時の相場・実行コンテキスト（timeframeやsymbol）+ML統合
        self.context = {
            "timeframe": None,
            "symbol": None,
            "regime_gating": False,
            "threshold_profile": "normal",
        }

        # geneに含まれる指標一覧をオプションで保持
        self.indicators: List[IndicatorGene] | None = None

    def set_context(
        self,
        *,
        timeframe: str | None = None,
        symbol: str | None = None,
        regime_gating: bool | None = None,
        threshold_profile: str | None = None,
    ):
        """生成コンテキストを設定（RSI閾値やレジーム切替に利用）"""
        try:
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
        except Exception:
            pass

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
        try:
            if not self.enable_smart_generation:
                return self._generate_fallback_conditions()

            if not indicators:
                return self._generate_fallback_conditions()

            # 戦略タイプを選択
            strategy_type = self._select_strategy_type(indicators)
            self.logger.info(f"生成戦略タイプ: {strategy_type.name}")

            # 選択された戦略に基づいて条件を生成
            if strategy_type == StrategyType.DIFFERENT_INDICATORS:
                longs, shorts, exits = self._generate_different_indicators_strategy(
                    indicators
                )
            # TIME_SEPARATION 戦略は削除したので、削除
            elif strategy_type == StrategyType.TIME_SEPARATION:
                longs, shorts, exits = self._generate_fallback_conditions()
            elif strategy_type == StrategyType.COMPLEX_CONDITIONS:
                longs, shorts, exits = self._generate_complex_conditions_strategy(
                    indicators
                )
            elif strategy_type == StrategyType.INDICATOR_CHARACTERISTICS:
                longs, shorts, exits = (
                    self._generate_indicator_characteristics_strategy(indicators)
                )
            else:
                longs, shorts, exits = self._generate_fallback_conditions()

            # 簡素化: 条件数を2個以内に制限
            if len(longs) > 2:
                longs = random.sample(longs, 2) if len(longs) >= 2 else longs
            if len(shorts) > 2:
                shorts = random.sample(shorts, 2) if len(shorts) >= 2 else shorts

            # 万一どちらかが空の場合は、成立しやすいデフォルトを適用
            if not longs:
                longs = [
                    Condition(left_operand="close", operator=">", right_operand="open")
                ]
            if not shorts:
                shorts = [
                    Condition(left_operand="close", operator="<", right_operand="open")
                ]

            # 型を明示的に変換して返す
            long_conditions: List[Union[Condition, ConditionGroup]] = list(longs)
            short_conditions: List[Union[Condition, ConditionGroup]] = list(shorts)
            exit_conditions: List[Condition] = list(exits)

            return long_conditions, short_conditions, exit_conditions

        except Exception as e:
            self.logger.error(f"スマート条件生成エラー: {e}")
            return self._generate_fallback_conditions()

    def _select_strategy_type(self, indicators: List[IndicatorGene]) -> StrategyType:
        """
        簡素化された戦略タイプ選択

        Args:
            indicators: 指標リスト

        Returns:
            選択された戦略タイプ
        """
        try:
            # ML指標がある場合はMLファースト戦略
            ml_indicators = [
                ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
            ]

            # テクニカル指標の数分析
            technical_indices = [
                ind
                for ind in indicators
                if ind.enabled and not ind.type.startswith("ML_")
            ]

            # ML指標とテクニカル指標の混合の場合は組み合わせ戦略
            if ml_indicators and technical_indices:
                return StrategyType.DIFFERENT_INDICATORS

            # ML指標のみの場合はML専用戦略
            if ml_indicators and not technical_indices:
                return StrategyType.INDICATOR_CHARACTERISTICS

            # テクニカル指標のみの場合
            indicator_types = set()
            for ind in technical_indices:
                if ind.type in INDICATOR_CHARACTERISTICS:
                    indicator_types.add(INDICATOR_CHARACTERISTICS[ind.type]["type"])

            # 異なるタイプの指標がある場合は組み合わせ戦略
            if len(indicator_types) >= 2:
                return StrategyType.DIFFERENT_INDICATORS

            # デフォルトは複合条件戦略
            return StrategyType.COMPLEX_CONDITIONS

        except Exception as e:
            self.logger.error(f"戦略タイプ選択エラー: {e}")
            return StrategyType.DIFFERENT_INDICATORS

    def _generate_fallback_conditions(
        self,
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        フォールバック条件を生成

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        long_conditions: List[Union[Condition, ConditionGroup]] = [
            Condition(left_operand="close", operator=">", right_operand="open")
        ]
        short_conditions: List[Union[Condition, ConditionGroup]] = [
            Condition(left_operand="close", operator="<", right_operand="open")
        ]
        exit_conditions: List[Condition] = []
        return long_conditions, short_conditions, exit_conditions

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
        final_condition = Condition(left_operand=ind.type, operator=">", right_operand=0)
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

    def _generate_different_indicators_strategy(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        異なる指標の組み合わせ戦略

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            # 指標をタイプ別に分類（未特性化はレジストリから動的分類）
            indicators_by_type = self._dynamic_classify(indicators)
            self.logger.debug(
                f"指標分類: {[f'{k.name}:{len(v)}' for k, v in indicators_by_type.items() if v]}"
            )

            # トレンド系 + モメンタム系の組み合わせを優先
            long_conditions: List[Union[Condition, ConditionGroup]] = []
            short_conditions: List[Union[Condition, ConditionGroup]] = []

            # ML指標とテクニカル指標の混合戦略
            ml_indicators = [
                ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
            ]
            self.logger.debug(f"ML指標数: {len(ml_indicators)}")

            # 簡素化: 利用可能な指標タイプから1つずつ条件を生成
            self.logger.debug("ロング条件生成開始")
            if indicators_by_type[IndicatorType.TREND]:
                long_conditions.extend(
                    self._create_trend_long_conditions(
                        random.choice(indicators_by_type[IndicatorType.TREND])
                    )
                )
                self.logger.debug(
                    f"トレンド指標からロング条件追加: {len(long_conditions)}"
                )

            if indicators_by_type[IndicatorType.MOMENTUM]:
                long_conditions.extend(
                    self._create_momentum_long_conditions(
                        random.choice(indicators_by_type[IndicatorType.MOMENTUM])
                    )
                )
                self.logger.debug(
                    f"モメンタム指標からロング条件追加: {len(long_conditions)}"
                )
            # ML指標がある場合は追加
            if ml_indicators:
                long_conditions.extend(self._create_ml_long_conditions(ml_indicators))
                self.logger.debug(f"ML指標からロング条件追加: {len(long_conditions)}")

            # ショート条件の簡素化
            self.logger.debug("ショート条件生成開始")
            if indicators_by_type[IndicatorType.TREND]:
                short_conditions.extend(
                    self._create_trend_short_conditions(
                        random.choice(indicators_by_type[IndicatorType.TREND])
                    )
                )
                self.logger.debug(
                    f"トレンド指標からショート条件追加: {len(short_conditions)}"
                )

            if indicators_by_type[IndicatorType.MOMENTUM]:
                short_conditions.extend(
                    self._create_momentum_short_conditions(
                        random.choice(indicators_by_type[IndicatorType.MOMENTUM])
                    )
                )
                self.logger.debug(
                    f"モメンタム指標からショート条件追加: {len(short_conditions)}"
                )

            # ML指標がある場合の対向条件
            if ml_indicators and len(ml_indicators) >= 2:
                if any(ind.type == "ML_DOWN_PROB" for ind in ml_indicators):
                    short_conditions.append(
                        Condition(
                            left_operand="ML_DOWN_PROB", operator=">", right_operand=0.6
                        )
                    )


            # 最低条件数の保証（簡素化）
            if not long_conditions:
                long_conditions = [
                    Condition(left_operand="close", operator=">", right_operand="open")
                ]
            if not short_conditions:
                short_conditions = [
                    Condition(left_operand="close", operator="<", right_operand="open")
                ]

            # 型を明示的に変換して返す
            long_result: List[Union[Condition, ConditionGroup]] = list(long_conditions)
            short_result: List[Union[Condition, ConditionGroup]] = list(
                short_conditions
            )
            exit_result: List[Condition] = []

            return long_result, short_result, exit_result

        except Exception as e:
            self.logger.error(f"異なる指標組み合わせ戦略エラー: {e}")
            return self._generate_fallback_conditions()

    def _create_type_based_conditions(
        self, indicator: IndicatorGene, side: str
    ) -> List[Condition]:
        """統合された型別条件生成 - YAML設定またはデフォルト"""
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
                self.logger.debug(f"YAML-based {side} condition for {indicator.type}: {indicator.type} {'>' if side == 'long' else '<'} {threshold}")
                return [final_condition]

        # RSI特別処理
        if indicator.type == "RSI":
            tf = self.context.get("timeframe", "")
            if tf in ("15m", "15min", "15"):
                threshold = 30 if side == "long" else 70
            elif tf in ("1h", "60"):
                threshold = 35 if side == "long" else 65
            elif tf in ("4h", "240"):
                threshold = 40 if side == "long" else 60
            else:
                threshold = 35 if side == "long" else 65
            final_condition = Condition(
                left_operand=indicator.type,
                operator="<" if side == "long" else ">",
                right_operand=threshold,
            )
            self.logger.debug(f"RSI {side} condition: {indicator.type} {'<' if side == 'long' else '>'} {threshold}")
            return [final_condition]


        # デフォルト
        final_condition = Condition(
            left_operand=indicator.type,
            operator=">" if side == "long" else "<",
            right_operand=0,
        )
        self.logger.debug(f"Default {side} condition for {indicator.type}: {indicator.type} {'>' if side == 'long' else '<'} 0")
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


    def _generate_indicator_characteristics_strategy(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        ML指標中心の簡素化戦略

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            long_conditions = []
            short_conditions = []

            # ML指標のみの処理を簡素化
            ml_indicators = [
                ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
            ]
            if ml_indicators:
                long_conditions.extend(self._create_ml_long_conditions(ml_indicators))

                # 複数のML指標があれば基本的な対向条件も追加
                if len(ml_indicators) >= 2:
                    short_conditions.extend(
                        [
                            Condition(
                                left_operand="ML_DOWN_PROB",
                                operator=">",
                                right_operand=0.6,
                            )
                        ]
                    )

            # 条件が空の場合はフォールバック
            if not long_conditions:
                return self._generate_fallback_conditions()

            return long_conditions, short_conditions, []

        except Exception as e:
            self.logger.error(f"指標特性活用戦略エラー: {e}")
            return self._generate_fallback_conditions()

    def _generate_complex_conditions_strategy(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        複合条件戦略（複数の条件を組み合わせて確率を高める）

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            # 利用可能な指標から複数の条件を組み合わせ
            long_conditions = []
            short_conditions = []

            for indicator in indicators[:3]:  # 最大3つの指標を使用してバランスを改善
                if not indicator.enabled:
                    continue

                if indicator.type in INDICATOR_CHARACTERISTICS:
                    # 各指標の特性に基づいて条件を追加
                    char = INDICATOR_CHARACTERISTICS[indicator.type]
                    indicator_type = char["type"]

                    # インジケータタイプに応じて適切な条件生成メソッドを呼び出し
                    if indicator_type == IndicatorType.MOMENTUM:
                        long_conds = self._create_momentum_long_conditions(indicator)
                        short_conds = self._create_momentum_short_conditions(indicator)
                    elif indicator_type == IndicatorType.TREND:
                        long_conds = self._create_trend_long_conditions(indicator)
                        short_conds = self._create_trend_short_conditions(indicator)
                    else:
                        # Unknown indicator type - use generic conditions
                        long_conds = self._generic_long_conditions(indicator)
                        short_conds = self._generic_short_conditions(indicator)

                    if long_conds:
                        long_conditions.extend(long_conds)

                    # ショート条件も同様に生成（全ての指標タイプに対応）
                    if short_conds:
                        short_conditions.extend(short_conds)

            # 条件が空の場合は汎用条件で補完（AND増加で厳しくならないよう空の場合のみ）
            if not long_conditions:
                for indicator in indicators[:2]:
                    if not indicator.enabled:
                        continue
                    long_conditions.extend(self._generic_long_conditions(indicator))
                    short_conditions.extend(self._generic_short_conditions(indicator))

                # それでも空ならフォールバック
                if not long_conditions:
                    return self._generate_fallback_conditions()

            return long_conditions, short_conditions, []

        except Exception as e:
            self.logger.error(f"複合条件戦略エラー: {e}")
            return self._generate_fallback_conditions()

    def _create_ml_long_conditions(
        self, indicators: List[IndicatorGene]
    ) -> List[Condition]:
        """
        ML予測を活用した簡素化したロング条件生成

        Args:
            indicators: 指標リスト

        Returns:
            ロング条件のリスト
        """
        try:
            conditions = []

            if any(ind.type == "ML_UP_PROB" for ind in indicators if ind.enabled):
                # 上昇予測確率が基準以上
                conditions.append(
                    Condition(
                        left_operand="ML_UP_PROB", operator=">", right_operand=0.6
                    )
                )

            if any(ind.type == "ML_DOWN_PROB" for ind in indicators if ind.enabled):
                # 下落予測確率が基準以下
                conditions.append(
                    Condition(
                        left_operand="ML_DOWN_PROB", operator="<", right_operand=0.4
                    )
                )

            return conditions
        except Exception as e:
            self.logger.error(f"MLロング条件生成エラー: {e}")
            return []

    def _get_indicator_type(self, indicator: IndicatorGene) -> IndicatorType:
        """指標のタイプを取得"""
        try:
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

            return IndicatorType.TREND
        except Exception:
            return IndicatorType.TREND

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

            try:
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
                    categorized[IndicatorType.TREND].append(ind)
            except Exception:
                categorized[IndicatorType.TREND].append(ind)

        return categorized

    def apply_threshold_context(
        self,
        indicators: List[IndicatorGene],
        context_override: Dict[str, Any] | None = None,
    ) -> Dict[str, List[Union[Condition, ConditionGroup]]]:
        """
        統合されたthreshold/profile処理メソッド
        """
        original_context: Dict[str, Any] | None = None
        if context_override:
            original_context = copy.deepcopy(self.context)
            self.context.update(context_override)

        try:
            categorized = self._dynamic_classify(indicators)

            long_conditions = []
            short_conditions = []
            metadata = {
                "applied_profile": self.context["threshold_profile"],
                "timeframe": self.context["timeframe"],
                "generated_indicators": len(indicators),
                "category_counts": {k.name: len(v) for k, v in categorized.items()},
            }

            for indicator in indicators:
                if not indicator.enabled:
                    continue

                indicator_type = self._get_indicator_type(indicator)
                if indicator_type == IndicatorType.MOMENTUM:
                    long_conditions.extend(
                        self._create_momentum_long_conditions(indicator)
                    )
                    short_conditions.extend(
                        self._create_momentum_short_conditions(indicator)
                    )
                elif indicator_type == IndicatorType.TREND:
                    long_conditions.extend(
                        self._create_trend_long_conditions(indicator)
                    )
                    short_conditions.extend(
                        self._create_trend_short_conditions(indicator)
                    )
                else:
                    long_conditions.extend(self._generic_long_conditions(indicator))
                    short_conditions.extend(self._generic_short_conditions(indicator))

            if not long_conditions:
                long_conditions = [
                    Condition(left_operand="close", operator=">", right_operand="open")
                ]
            if not short_conditions:
                short_conditions = [
                    Condition(left_operand="close", operator="<", right_operand="open")
                ]

            if long_conditions is None:
                long_conditions = []
            if short_conditions is None:
                short_conditions = []

            result = {
                "long_conditions": long_conditions,
                "short_conditions": short_conditions,
                "threshold_metadata": metadata,
            }

            return result

        finally:
            if original_context is not None:
                self.context = original_context

    def _generate_yaml_based_conditions(
        self, indicator: IndicatorGene, side: str
    ) -> List[Condition]:
        """YAML設定に基づいて条件を生成"""
        try:
            config = YamlIndicatorUtils.get_indicator_config_from_yaml(
                self.yaml_config, indicator.type
            )
            if config:
                threshold = YamlIndicatorUtils.get_threshold_from_yaml(
                    self.yaml_config, config, side, self.context
                )
                if threshold is not None:
                    operator = ">" if side == "long" else "<"
                    return [
                        Condition(
                            left_operand=indicator.type,
                            operator=operator,
                            right_operand=threshold,
                        )
                    ]
            return [
                Condition(left_operand=indicator.type, operator=">", right_operand=0)
            ]
        except Exception as e:
            self.logger.error(f"YAML条件生成エラー: {e}")
            return []

    def test_yaml_conditions(
        self, test_indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """YAMLベースの条件生成テスト"""
        try:
            return YamlIndicatorUtils.test_yaml_conditions_with_generator(
                yaml_config=self.yaml_config,
                test_indicators=test_indicators,
            )
        except Exception as e:
            self.logger.error(f"YAMLテストエラー: {e}")
            return {"error": str(e)}
