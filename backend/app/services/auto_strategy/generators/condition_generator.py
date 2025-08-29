"""
条件生成統合ジェネレーター - Phase 1.3 リファクタリング

SmartConditionGeneratorから条件生成ロジックを分離・統合
設計目標: -900行削減、12重複統合、デプロイ効率化
"""

import logging
import copy
import random
from typing import List, Dict, Any, Union
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


logger = logging.getLogger(__name__)


class ConditionGenerator:
    """
    責務: 条件生成ロジックの一元化

    統合アプローチ:
    1. SmartConditionGeneratorから条件生成メソッドを抽出
    2. 統合メソッドで重複排除
    3. 新機能apply_threshold_context()でprofile/threshold統合
    """

    def __init__(self):
        """初期化"""
        self.logger = logger
        # 条件生成コンテキスト（threshold profileと統合）
        self.context: Dict[str, Any] = {
            "timeframe": None,
            "symbol": None,
            "regime_gating": False,
            "threshold_profile": "normal",
        }
        # 生成時のgene比較に使用
        self.indicators: List[IndicatorGene] | None = None

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

    def _get_indicator_type(self, indicator: IndicatorGene) -> IndicatorType:
        """指標のタイプを安全に取得"""
        try:
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

    def _generic_long_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """基盤条件ロジック: ロング条件生成"""
        name = ind.type
        cfg = indicator_registry.get_indicator_config(name)
        scale = getattr(cfg, "scale_type", None) if cfg else None

        # 価格系指標（統一処理）
        if name in (
            "SMA",
            "EMA",
            "WMA",
            "KAMA",
            "T3",
            "TRIMA",
            "MIDPOINT",
        ):
            trend_names_in_gene = [
                ind.type
                for ind in (self.indicators or [])
                if getattr(ind, "enabled", True)
            ]
            right_name = name if name in trend_names_in_gene else "open"
            return [
                Condition(left_operand="close", operator=">", right_operand=right_name)
            ]

        # オシレーター系指標
        if scale == IndicatorScaleType.OSCILLATOR_0_100:
            profile = self.context.get("threshold_profile", "normal")
            if name in {"RSI", "STOCH", "STOCHRSI", "KDJ", "QQE", "MFI"}:
                policy = ThresholdPolicy.get(profile)
                thr = (
                    policy.rsi_long_lt
                    if name != "MFI"
                    else (policy.mfi_long_lt or policy.rsi_long_lt)
                )
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]
            elif name == "ADX":
                thr = ThresholdPolicy.get(profile).adx_trend_min
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]
            else:
                thr = (
                    48
                    if profile == "aggressive"
                    else (52 if profile == "conservative" else 50)
                )
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]

        # ±100オシレーター系
        if scale == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
            profile = self.context.get("threshold_profile", "normal")
            if name == "CCI":
                lim = ThresholdPolicy.get(profile).cci_abs_limit or 100
                return [
                    Condition(
                        left_operand=name,
                        operator=">",
                        right_operand=float(-lim / 20.0),
                    )
                ]
            elif name == "WILLR":
                p = ThresholdPolicy.get(profile)
                thr = -p.willr_long_lt if p.willr_long_lt is not None else -50
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]
            else:
                thr = (
                    -2
                    if profile == "aggressive"
                    else (2 if profile == "conservative" else 0)
                )
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]

        # ゼロ中心系
        if scale == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
            thr = -0.0 if self.context.get("threshold_profile") == "aggressive" else 0
            return [
                Condition(left_operand=name, operator=">", right_operand=float(thr))
            ]

        # 比率・絶対価格系（トレンド系以外スキップ）
        if scale in {IndicatorScaleType.PRICE_RATIO, IndicatorScaleType.PRICE_ABSOLUTE}:
            return []

        # フォールバック
        thr = -0.0 if self.context.get("threshold_profile") == "aggressive" else 0
        return [Condition(left_operand=name, operator=">", right_operand=float(thr))]

    def _generic_short_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """基盤条件ロジック: ショート条件生成"""
        name = ind.type
        cfg = indicator_registry.get_indicator_config(name)
        scale = getattr(cfg, "scale_type", None) if cfg else None

        # 価格系指標
        if name in (
            "SMA",
            "EMA",
            "WMA",
            "KAMA",
            "T3",
            "TRIMA",
            "MIDPOINT",
        ):
            trend_names_in_gene = [
                ind.type
                for ind in (self.indicators or [])
                if getattr(ind, "enabled", True)
            ]
            right_name = name if name in trend_names_in_gene else "open"
            return [
                Condition(left_operand="close", operator="<", right_operand=right_name)
            ]

        # 0-100オシレーター系
        if scale == IndicatorScaleType.OSCILLATOR_0_100:
            profile = self.context.get("threshold_profile", "normal")
            if name in {"RSI", "STOCH", "STOCHRSI", "KDJ", "QQE", "MFI"}:
                p = ThresholdPolicy.get(profile)
                thr = (
                    p.rsi_short_gt
                    if name != "MFI"
                    else (p.mfi_short_gt or p.rsi_short_gt)
                )
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]
            elif name == "ADX":
                thr = float(100 - ThresholdPolicy.get(profile).adx_trend_min)
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]
            else:
                thr = (
                    52
                    if profile == "aggressive"
                    else (48 if profile == "conservative" else 50)
                )
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]

        # ±100オシレーター系
        if scale == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
            profile = self.context.get("threshold_profile", "normal")
            if name == "CCI":
                thr = (
                    5
                    if profile == "aggressive"
                    else (-5 if profile == "conservative" else 0)
                )
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]
            elif name == "WILLR":
                thr = (
                    -40
                    if profile == "aggressive"
                    else (-60 if profile == "conservative" else -50)
                )
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]
            else:
                thr = (
                    2
                    if profile == "aggressive"
                    else (-2 if profile == "conservative" else 0)
                )
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]

        # ゼロ中心系
        if scale == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
            return [Condition(left_operand=name, operator="<", right_operand=0.0)]

        # 比率・絶対価格系（スキップ）
        if scale in {IndicatorScaleType.PRICE_RATIO, IndicatorScaleType.PRICE_ABSOLUTE}:
            return []

        # フォールバック
        return [Condition(left_operand=name, operator="<", right_operand=0.0)]

    def _create_trend_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """トレンド系ロング条件ロジック"""
        if indicator.type in ["SMA", "EMA", "WMA", "TRIMA"]:
            return [
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand=indicator.type,  # 素名使用
                )
            ]
        return []

    def _create_trend_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """トレンド系ショート条件ロジック"""
        if indicator.type in ["SMA", "EMA", "WMA", "TRIMA"]:
            return [
                Condition(
                    left_operand="close",
                    operator="<",
                    right_operand=indicator.type,  # 素名使用
                )
            ]
        return []

    def _create_momentum_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """モメンタム系ロング条件ロジック"""
        if indicator.type == "RSI":
            tf = self.context.get("timeframe", "")
            if tf in ("15m", "15min", "15"):
                base = 55
            elif tf in ("1h", "60"):
                base = 60
            elif tf in ("4h", "240"):
                base = 65
            else:
                base = 60
            return [
                Condition(
                    left_operand=indicator.type,
                    operator="<",
                    right_operand=max(25, base - 25),  # 売られすぎ判定
                )
            ]
        elif indicator.type == "STOCH":
            return [Condition(left_operand="STOCH_0", operator="<", right_operand=20)]
        elif indicator.type == "CCI":
            return [
                Condition(
                    left_operand=indicator.type,
                    operator="<",
                    right_operand=random.uniform(-150, -80),
                )
            ]
        elif indicator.type in {"MACD", "MACDEXT"}:
            return [
                Condition(
                    left_operand=(
                        "MACD_0" if indicator.type == "MACD" else "MACDEXT_0"
                    ),
                    operator=">",
                    right_operand=0,
                )
            ]
        return []

    def _create_momentum_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """モメンタム系ショート条件ロジック"""
        if indicator.type == "RSI":
            tf = self.context.get("timeframe", "")
            if tf in ("15m", "15min", "15"):
                base = 55
            elif tf in ("1h", "60"):
                base = 60
            elif tf in ("4h", "240"):
                base = 65
            else:
                base = 60
            return [
                Condition(
                    left_operand=indicator.type,
                    operator=">",
                    right_operand=min(75, base + 25),  # 買われすぎ判定
                )
            ]
        elif indicator.type == "STOCH":
            return [Condition(left_operand="STOCH_0", operator=">", right_operand=80)]
        elif indicator.type == "CCI":
            return [
                Condition(
                    left_operand=indicator.type,
                    operator=">",
                    right_operand=random.uniform(80, 150),
                )
            ]
        elif indicator.type in {"MACD", "MACDEXT"}:
            return [
                Condition(
                    left_operand=(
                        "MACD_0" if indicator.type == "MACD" else "MACDEXT_0"
                    ),
                    operator="<",
                    right_operand=0,
                )
            ]
        return []

    def _create_statistics_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """統計系ロング条件ロジック"""
        if indicator.type == "CORREL":
            return [
                Condition(
                    left_operand=indicator.type,
                    operator=">",
                    right_operand=random.uniform(0.3, 0.7),
                )
            ]
        elif indicator.type == "LINEARREG_ANGLE":
            return [
                Condition(
                    left_operand=indicator.type,
                    operator=">",
                    right_operand=random.uniform(10, 45),
                )
            ]
        elif indicator.type == "LINEARREG_SLOPE":
            return [
                Condition(left_operand=indicator.type, operator=">", right_operand=0)
            ]
        elif indicator.type in ["LINEARREG", "TSF"]:
            return [
                Condition(
                    left_operand="close", operator=">", right_operand=indicator.type
                )
            ]
        return []

    def _create_statistics_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """統計系ショート条件ロジック"""
        if indicator.type == "CORREL":
            return [
                Condition(
                    left_operand=indicator.type,
                    operator="<",
                    right_operand=random.uniform(-0.7, -0.3),
                )
            ]
        elif indicator.type == "LINEARREG_ANGLE":
            return [
                Condition(
                    left_operand=indicator.type,
                    operator="<",
                    right_operand=random.uniform(-45, -10),
                )
            ]
        elif indicator.type == "LINEARREG_SLOPE":
            return [
                Condition(left_operand=indicator.type, operator="<", right_operand=0)
            ]
        elif indicator.type in ["LINEARREG", "TSF"]:
            return [
                Condition(
                    left_operand="close", operator="<", right_operand=indicator.type
                )
            ]
        return []

    def _create_pattern_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """パターン認識系ロング条件ロジック"""
        if indicator.type in ["CDL_HAMMER", "CDL_PIERCING", "CDL_THREE_WHITE_SOLDIERS"]:
            return [
                Condition(
                    left_operand=f"{indicator.type}", operator=">", right_operand=0
                )
            ]
        elif indicator.type == "CDL_DOJI":
            return [
                Condition(
                    left_operand=f"{indicator.type}", operator="!=", right_operand=0
                )
            ]
        return []

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
