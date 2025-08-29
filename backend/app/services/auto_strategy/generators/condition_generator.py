"""
条件生成統合ジェネレーター

"""

import logging
import copy
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import yaml
from app.services.indicators.config import indicator_registry
from app.services.indicators.config.indicator_config import IndicatorScaleType
from app.services.auto_strategy.core.indicator_policies import ThresholdPolicy
from app.services.auto_strategy.config.constants import (
    INDICATOR_CHARACTERISTICS,
    IndicatorType,
)
from app.services.auto_strategy.models.strategy_models import (
    Condition,
    IndicatorGene,
    ConditionGroup,
)
from app.services.auto_strategy.utils.common_utils import (
    YamlLoadUtils,
    YamlTestUtils,
)


logger = logging.getLogger(__name__)


class ConditionGenerator:
    """
    責務: 条件生成ロジックの一元化

    統合アプローチ:
    1. SmartConditionGeneratorから条件生成メソッドを抽出
    2. 統合メソッドで重複排除
    3. 新機能apply_threshold_context()でprofile/threshold統合
    4. YAML設定ファイルからの条件生成統合
    """

    def __init__(self):
        """初期化"""
        self.logger = logger
        # YAML設定を読み込み
        self.yaml_config = self._load_yaml_config()
        # 条件生成コンテキスト（threshold profileと統合）
        self.context: Dict[str, Any] = {
            "timeframe": None,
            "symbol": None,
            "regime_gating": False,
            "threshold_profile": "normal",
        }
        # 生成時のgene比較に使用
        self.indicators: List[IndicatorGene] | None = None

    def _load_yaml_config(self) -> Dict[str, Any]:
        """技術指標のYAML設定を読み込み"""
        config_path = (
            Path(__file__).parent.parent / "config" / "technical_indicators_config.yaml"
        )
        # YamlLoadUtilsを使用して読み込み
        config = YamlLoadUtils.load_yaml_config(config_path)

        # 設定検証
        is_valid, errors = YamlLoadUtils.validate_yaml_config(config)
        if not is_valid:
            logger.error(f"YAML設定検証エラー: {errors}")
            # 検証エラーがあっても基本構造は維持

        return config

    def test_yaml_conditions(self, test_indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """YAMLベースの条件生成をテスト

        Args:
            test_indicators: テスト対象の指標（指定なしの場合は全て）

        Returns:
            テスト結果の辞書
        """
        return YamlTestUtils.test_yaml_based_conditions(
            yaml_config=self.yaml_config,
            condition_generator_class=ConditionGenerator,
            test_indicators=test_indicators
        )

    def set_context(
        self,
        *,
        timeframe: str | None = None,
        symbol: str | None = None,
        regime_gating: bool | None = None,
        threshold_profile: str | None = None,
    ) -> None:
        """
        生成コンテキストを設定（RSI閾値やレジーム切替に利用）

        Args:
            timeframe: 時間軸設定
            symbol: シンボル設定
            regime_gating: レジームゲート有効化フラグ
            threshold_profile: 'aggressive' | 'normal' | 'conservative'
        """
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

    def apply_threshold_context(
        self,
        indicators: List[IndicatorGene],
        context_override: Dict[str, Any] | None = None,
    ) -> Dict[str, List[Union[Condition, ConditionGroup]]]:
        """
        新機能: profile/threshold統合処理メソッド

        計画書Phase 1.3 要求事項:
        - 各指標タイプの条件をコンテキストに基づいて適切に生成
        - threshold_profileを基にThresholdPolicy適用
        - 枠組み化された条件生成フロー

        Args:
            indicators: 適用対象指標リスト
            context_override: コンテキスト上書き（オプション）

        Returns:
            統合された条件辞書
            {
                "long_conditions": [...],
                "short_conditions": [...],
                "threshold_metadata": {...}
            }
        """
        original_context: Dict[str, Any] | None = None
        if context_override:
            original_context = copy.deepcopy(self.context)
            self.context.update(context_override)

        try:
            # 指標分類
            categorized = self._dynamic_classify(indicators)

            # 各タイプの条件生成
            long_conditions = []
            short_conditions = []
            metadata = {
                "applied_profile": self.context["threshold_profile"],
                "timeframe": self.context["timeframe"],
                "generated_indicators": len(indicators),
                "category_counts": {k.name: len(v) for k, v in categorized.items()},
            }

            # 統合メソッド呼び出しで条件生成
            for indicator in indicators:
                if not indicator.enabled:
                    continue

                # 指標タイプに応じたメソッド選択・呼び出し
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
                elif indicator_type == IndicatorType.STATISTICS:
                    long_conditions.extend(
                        self._create_statistics_long_conditions(indicator)
                    )
                    short_conditions.extend(
                        self._create_statistics_short_conditions(indicator)
                    )
                elif indicator_type == IndicatorType.PATTERN_RECOGNITION:
                    long_conditions.extend(
                        self._create_pattern_long_conditions(indicator)
                    )
                    short_conditions.extend(
                        self._create_pattern_short_conditions(indicator)
                    )
                else:
                    # 未知タイプは汎用条件で対応
                    long_conditions.extend(self._generic_long_conditions(indicator))
                    short_conditions.extend(self._generic_short_conditions(indicator))

            # 成立性保証（最小1つの条件）
            if not long_conditions:
                long_conditions.append(
                    Condition(left_operand="close", operator=">", right_operand="open")
                )
            if not short_conditions:
                short_conditions.append(
                    Condition(left_operand="close", operator="<", right_operand="open")
                )

            # Noneチェック
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
                assert original_context is not None
                self.context = original_context

    def _get_indicator_config_from_yaml(
        self, indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """YAMLから指標設定を取得"""
        indicators_config = self.yaml_config.get("indicators", {})
        return indicators_config.get(indicator_name)

    def _get_threshold_from_yaml(self, config: Dict[str, Any], side: str) -> Any:
        """YAMLから適切な閾値を取得"""
        if not config:
            return None

        profile = self.context.get("threshold_profile", "normal")
        thresholds = config.get("thresholds", {})

        if thresholds is None:
            thresholds = {}

        # 特定のprofileがある場合
        if profile in thresholds:
            threshold_config = thresholds[profile]
            if side == "long" and "long_lt" in threshold_config:
                return threshold_config["long_lt"]
            elif side == "short" and "short_gt" in threshold_config:
                return threshold_config["short_gt"]
            elif side == "long" and "long_gt" in threshold_config:
                return threshold_config["long_gt"]

        # 全profile共通の閾値
        if "all" in thresholds:
            all_config = thresholds["all"]
            if side == "long" and "pos_threshold" in all_config:
                return all_config["pos_threshold"]
            elif side == "short" and "neg_threshold" in all_config:
                return all_config["neg_threshold"]
            elif side == "long" and "long_gt" in all_config:
                return all_config["long_gt"]
            elif side == "short" and "short_lt" in all_config:
                return all_config["short_lt"]

        return None

    def _get_indicator_type(self, indicator: IndicatorGene) -> IndicatorType:
        """指標のタイプを安全に取得"""
        try:
            # YAML設定からタイプを取得
            config = self._get_indicator_config_from_yaml(indicator.type)
            if config and "type" in config:
                type_str = config["type"]
                if type_str == "momentum":
                    return IndicatorType.MOMENTUM
                elif type_str == "trend":
                    return IndicatorType.TREND
                elif type_str == "statistics":
                    return IndicatorType.STATISTICS
                elif type_str == "pattern_recognition":
                    return IndicatorType.PATTERN_RECOGNITION
                elif type_str == "volatility":
                    return IndicatorType.VOLATILITY

            # レジストリ情報から分類
            cfg = indicator_registry.get_indicator_config(indicator.type)
            if cfg and hasattr(cfg, "category") and getattr(cfg, "category", None):
                cat = getattr(cfg, "category")
                if cat == "momentum":
                    return IndicatorType.MOMENTUM
                elif cat == "trend":
                    return IndicatorType.TREND
                elif cat == "statistics":
                    return IndicatorType.STATISTICS
                elif cat == "pattern_recognition":
                    return IndicatorType.PATTERN_RECOGNITION

            # INDICATOR_CHARACTERISTICS fallback
            if indicator.type in INDICATOR_CHARACTERISTICS:
                return INDICATOR_CHARACTERISTICS[indicator.type]["type"]

            return IndicatorType.TREND  # デフォルト
        except Exception:
            return IndicatorType.TREND

    # ===== 統合メソッド実装 =====

    def _generate_yaml_based_conditions(
        self, ind: IndicatorGene, side: str
    ) -> List[Condition]:
        """YAML設定に基づく条件生成"""
        name = ind.type
        config = self._get_indicator_config_from_yaml(name)

        if not config:
            return self._generate_fallback_conditions(name, side)

        conditions_config = config.get("conditions", {})
        threshold_value = self._get_threshold_from_yaml(config, side)

        # 条件テンプレート
        template = conditions_config.get(side)
        if not template or template == "null":
            return []

        # テンプレートのパースと適用
        left_operand = name  # デフォルト
        operator = ">"
        right_operand = 0

        if template == "{left_operand} < {threshold}":
            operator = "<"
            right_operand = threshold_value
        elif template == "{left_operand} > {threshold}":
            operator = ">"
            right_operand = threshold_value
        elif template == "{left_operand}_0 < {threshold}":
            left_operand = f"{name}_0"
            operator = "<"
            right_operand = threshold_value
        elif template == "{left_operand}_0 > {threshold}":
            left_operand = f"{name}_0"
            operator = ">"
            right_operand = threshold_value
        elif template == "{left_operand}_2 < {threshold}":
            left_operand = f"{name}_2"
            operator = "<"
            right_operand = threshold_value
        elif template == "close > {left_operand}":
            left_operand = "close"
            operator = ">"
            right_operand = name
        elif template == "close < {left_operand}":
            left_operand = "close"
            operator = "<"
            right_operand = name
        elif template == "close > {left_operand}_lower":
            left_operand = "close"
            operator = ">"
            right_operand = f"{name}_lower"
        elif template == "close < {left_operand}_upper":
            left_operand = "close"
            operator = "<"
            right_operand = f"{name}_upper"
        elif template == "{left_operand} > {long_threshold}":
            operator = ">"
            # longThresholdの取得ロジックが必要
            config_thresholds = config.get("thresholds", {}).get(
                self.context.get("threshold_profile", "normal"), {}
            )
            abs_limit = config_thresholds.get("abs_limit", 100)
            right_operand = -abs_limit / 20.0
        elif template == "{left_operand} < {short_threshold}":
            operator = "<"
            config_thresholds = config.get("thresholds", {}).get(
                self.context.get("threshold_profile", "normal"), {}
            )
            abs_limit = config_thresholds.get("abs_limit", 100)
            right_operand = abs_limit / 20.0
        elif template == "{left_operand} != 0":
            operator = "!="
            right_operand = 0
        elif template == "{left_operand} > 0":
            operator = ">"
            right_operand = 0
        elif template == "{left_operand} < 0":
            operator = "<"
            right_operand = 0
        elif template == "{left_operand} > {pos_threshold}":
            operator = ">"
            right_operand = threshold_value if threshold_value is not None else 0.3
        elif template == "{left_operand} < {neg_threshold}":
            operator = "<"
            right_operand = threshold_value if threshold_value is not None else -0.7

        # None のチェック
        right_operand = 0 if right_operand is None else right_operand

        return [
            Condition(
                left_operand=left_operand,
                operator=operator,
                right_operand=right_operand,
            )
        ]

    def _generate_fallback_conditions(self, name: str, side: str) -> List[Condition]:
        """YAMLがない場合のフォールバック条件生成"""
        # 既存のロジックを維持
        cfg = indicator_registry.get_indicator_config(name)
        scale = getattr(cfg, "scale_type", None) if cfg else None

        profile = self.context.get("threshold_profile", "normal")

        if side == "long":
            thr = -0.0 if profile == "aggressive" else 0
        else:
            thr = 0.0 if profile == "aggressive" else 0
            if side == "short":
                thr = float(thr)

        return [
            Condition(
                left_operand=name,
                operator=">" if side == "long" else "<",
                right_operand=thr,
            )
        ]

    def _generic_long_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """YAMLベースの基盤条件ロジック: ロング条件生成"""
        return self._generate_yaml_based_conditions(ind, "long")

    def _generic_short_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """YAMLベースの基盤条件ロジック: ショート条件生成"""
        return self._generate_yaml_based_conditions(ind, "short")

    def _create_trend_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """YAMLベーストレンド系ロング条件ロジック"""
        return self._generate_yaml_based_conditions(indicator, "long")

    def _create_trend_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """YAMLベーストレンド系ショート条件ロジック"""
        return self._generate_yaml_based_conditions(indicator, "short")

    def _create_momentum_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """YAMLベースモメンタム系ロング条件ロジック"""
        # RSIの場合は特殊ロジックを適用（timeframe依存）
        if indicator.type == "RSI":
            tf = self.context.get("timeframe", "")
            if tf in ("15m", "15min", "15"):
                threshold = 30
            elif tf in ("1h", "60"):
                threshold = 35
            elif tf in ("4h", "240"):
                threshold = 40
            else:
                threshold = 35

            return [
                Condition(
                    left_operand=indicator.type, operator="<", right_operand=threshold
                )
            ]

        # 他のインジケーターはYAML設定を使用
        return self._generate_yaml_based_conditions(indicator, "long")

    def _create_momentum_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """YAMLベースモメンタム系ショート条件ロジック"""
        # RSIの場合は特殊ロジックを適用（timeframe依存）
        if indicator.type == "RSI":
            tf = self.context.get("timeframe", "")
            if tf in ("15m", "15min", "15"):
                threshold = 70
            elif tf in ("1h", "60"):
                threshold = 65
            elif tf in ("4h", "240"):
                threshold = 60
            else:
                threshold = 65

            return [
                Condition(
                    left_operand=indicator.type, operator=">", right_operand=threshold
                )
            ]

        # 他のインジケーターはYAML設定を使用
        return self._generate_yaml_based_conditions(indicator, "short")

    def _create_statistics_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """YAMLベース統計系ロング条件ロジック"""
        return self._generate_yaml_based_conditions(indicator, "long")

    def _create_statistics_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """YAMLベース統計系ショート条件ロジック"""
        return self._generate_yaml_based_conditions(indicator, "short")

    def _create_pattern_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """YAMLベースパターン認識系ロング条件ロジック"""
        return self._generate_yaml_based_conditions(indicator, "long")

    def _create_pattern_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """パターン認識系ショート条件ロジック"""
        if indicator.type in [
            "CDL_HANGING_MAN",
            "CDL_DARK_CLOUD_COVER",
            "CDL_THREE_BLACK_CROWS",
        ]:
            return [
                Condition(
                    left_operand=f"{indicator.type}", operator="<", right_operand=0
                )
            ]
        elif indicator.type == "CDL_DOJI":
            return [
                Condition(
                    left_operand=f"{indicator.type}", operator="!=", right_operand=0
                )
            ]
        return []

    def _dynamic_classify(
        self, indicators: List[IndicatorGene]
    ) -> Dict[IndicatorType, List[IndicatorGene]]:
        """
        レジストリ情報に基づく動的指標分類

        統合処理: SmartConditionGenerator._dynamic_classify()を最適化・統合
        """
        categorized = {
            IndicatorType.MOMENTUM: [],
            IndicatorType.TREND: [],
            IndicatorType.VOLATILITY: [],
            IndicatorType.STATISTICS: [],
            IndicatorType.PATTERN_RECOGNITION: [],
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
                    elif cat == "statistics":
                        categorized[IndicatorType.STATISTICS].append(ind)
                    elif cat == "pattern_recognition":
                        categorized[IndicatorType.PATTERN_RECOGNITION].append(ind)
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
