import logging
import random
from enum import Enum
from typing import List, Tuple, Union
from app.services.indicators.config import indicator_registry
from app.services.indicators.config.indicator_config import IndicatorScaleType
from app.services.auto_strategy.core.indicator_policies import ThresholdPolicy

from ..models.strategy_models import Condition, IndicatorGene, ConditionGroup


logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """戦略タイプ"""

    DIFFERENT_INDICATORS = "different_indicators"  # 異なる指標の組み合わせ
    TIME_SEPARATION = "time_separation"  # 時間軸分離
    COMPLEX_CONDITIONS = "complex_conditions"  # 複合条件
    INDICATOR_CHARACTERISTICS = "indicator_characteristics"  # 指標特性活用


class IndicatorType(Enum):
    """指標分類"""

    MOMENTUM = "momentum"  # モメンタム系
    TREND = "trend"  # トレンド系
    VOLATILITY = "volatility"  # ボラティリティ系

    STATISTICS = "statistics"  # 統計系
    PATTERN_RECOGNITION = "pattern_recognition"  # パターン認識系


# 指標特性データベース（計画書の設計に基づく）
INDICATOR_CHARACTERISTICS = {
    "RSI": {
        "type": IndicatorType.MOMENTUM,
        "range": (0, 100),
        "long_zones": [(0, 30), (40, 60)],
        "short_zones": [(40, 60), (70, 100)],
        "neutral_zone": (40, 60),
        "oversold_threshold": 30,
        "overbought_threshold": 70,
    },
    "STOCH": {
        "type": IndicatorType.MOMENTUM,
        "range": (0, 100),
        "long_zones": [(0, 20), (40, 60)],
        "short_zones": [(40, 60), (80, 100)],
        "neutral_zone": (40, 60),
        "oversold_threshold": 20,
        "overbought_threshold": 80,
    },
    "CCI": {
        "type": IndicatorType.MOMENTUM,
        "range": (-200, 200),
        "long_zones": [(-200, -100), (-50, 50)],
        "short_zones": [(-50, 50), (100, 200)],
        "neutral_zone": (-50, 50),
        "oversold_threshold": -100,
        "overbought_threshold": 100,
    },
    "MACD": {
        "type": IndicatorType.MOMENTUM,
        "range": None,  # 価格依存
        "zero_cross": True,
        "signal_line": True,
    },
    "MACDEXT": {
        "type": IndicatorType.MOMENTUM,
        "range": None,  # 価格依存
        "zero_cross": True,
        "signal_line": True,
    },
    "SMA": {
        "type": IndicatorType.TREND,
        "price_comparison": True,
        "trend_following": True,
    },
    "EMA": {
        "type": IndicatorType.TREND,
        "price_comparison": True,
        "trend_following": True,
    },
    "MAMA": {"type": IndicatorType.TREND, "price_comparison": True, "adaptive": True},
    "ADX": {
        "type": IndicatorType.TREND,
        "range": (0, 100),
        "trend_strength": True,
        "no_direction": True,  # 方向性を示さない
        "strong_trend_threshold": 25,
    },
    "BB": {
        "type": IndicatorType.VOLATILITY,
        "components": ["upper", "middle", "lower"],
        "mean_reversion": True,
        "breakout_strategy": True,
    },
    "ATR": {
        "type": IndicatorType.VOLATILITY,
        "range": (0, None),
        "volatility_measure": True,
    },
    # 統計系インジケータ
    "BETA": {
        "type": IndicatorType.STATISTICS,
        "range": (-2, 2),  # 一般的なベータ値範囲
        "correlation_measure": True,
        "zero_cross": True,
    },
    "CORREL": {
        "type": IndicatorType.STATISTICS,
        "range": (-1, 1),  # 相関係数
        "correlation_measure": True,
        "zero_cross": True,
    },
    "LINEARREG": {
        "type": IndicatorType.STATISTICS,
        "range": None,  # 価格依存
        "price_comparison": True,
        "trend_following": True,
    },
    "LINEARREG_ANGLE": {
        "type": IndicatorType.STATISTICS,
        "range": (-90, 90),  # 角度
        "zero_cross": True,
        "trend_strength": True,
    },
    "LINEARREG_INTERCEPT": {
        "type": IndicatorType.STATISTICS,
        "range": None,  # 価格依存
        "price_comparison": True,
    },
    "LINEARREG_SLOPE": {
        "type": IndicatorType.STATISTICS,
        "range": None,  # 価格変化率依存
        "zero_cross": True,
        "trend_strength": True,
    },
    "STDDEV": {
        "type": IndicatorType.STATISTICS,
        "range": (0, None),  # 常に正値
        "volatility_measure": True,
    },
    "TSF": {
        "type": IndicatorType.STATISTICS,
        "range": None,  # 価格依存
        "price_comparison": True,
        "trend_following": True,
    },
    "VAR": {
        "type": IndicatorType.STATISTICS,
        "range": (0, None),  # 常に正値
        "volatility_measure": True,
    },
    # パターン認識系インジケータ
    "CDL_DOJI": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),  # パターン強度
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
    },
    # ML予測確率指標
    "ML_UP_PROB": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0.6, 1.0)],
        "short_zones": [(0, 0.4)],
        "neutral_zone": (0.4, 0.6),
        "high_confidence_threshold": 0.7,
    },
    "ML_DOWN_PROB": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0, 0.4)],
        "short_zones": [(0.6, 1.0)],
        "neutral_zone": (0.4, 0.6),
        "high_confidence_threshold": 0.7,
    },
    "ML_RANGE_PROB": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0, 0.3)],
        "short_zones": [(0, 0.3)],
        "neutral_zone": (0.7, 1.0),
        "high_confidence_threshold": 0.8,
    },
    "CDL_HAMMER": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bullish_pattern": True,
    },
    "CDL_HANGING_MAN": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bearish_pattern": True,
    },
    "CDL_SHOOTING_STAR": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bearish_pattern": True,
    },
    "CDL_ENGULFING": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
    },
    "CDL_HARAMI": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
    },
    "CDL_PIERCING": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bullish_pattern": True,
    },
    "CDL_THREE_BLACK_CROWS": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "continuation_pattern": True,
        "bearish_pattern": True,
    },
    "CDL_THREE_WHITE_SOLDIERS": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "continuation_pattern": True,
        "bullish_pattern": True,
    },
    "CDL_DARK_CLOUD_COVER": {
        "type": IndicatorType.PATTERN_RECOGNITION,
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bearish_pattern": True,
    },
}

# 組み合わせルール（計画書の設計に基づく）
COMBINATION_RULES = {
    "trend_momentum": {
        "description": "トレンド系 + モメンタム系",
        "long_indicators": [IndicatorType.TREND, IndicatorType.MOMENTUM],
        "short_indicators": [IndicatorType.TREND, IndicatorType.MOMENTUM],
        "weight": 0.4,
    },
    "volatility_trend": {
        "description": "ボラティリティ系 + トレンド系",
        "long_indicators": [IndicatorType.VOLATILITY, IndicatorType.TREND],
        "short_indicators": [IndicatorType.VOLATILITY, IndicatorType.TREND],
        "weight": 0.3,
    },
    "momentum_volatility": {
        "description": "モメンタム系 + ボラティリティ系",
        "long_indicators": [IndicatorType.MOMENTUM, IndicatorType.VOLATILITY],
        "short_indicators": [IndicatorType.MOMENTUM, IndicatorType.VOLATILITY],
        "weight": 0.2,
    },
    "single_indicator_multi_timeframe": {
        "description": "単一指標の複数時間軸",
        "long_indicators": [IndicatorType.MOMENTUM],
        "short_indicators": [IndicatorType.MOMENTUM],
        "weight": 0.1,
    },
}


class SmartConditionGenerator:
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
        初期化

        Args:
            enable_smart_generation: 新しいスマート生成を有効にするか
        """
        self.enable_smart_generation = enable_smart_generation
        self.logger = logger
        # 生成時の相場・実行コンテキスト（timeframeやsymbol）
        # regime_gating: レジーム条件をAND前置するか（デフォルトは無効化して約定性を優先）
        # threshold_profile: 'aggressive' | 'normal' | 'conservative' で閾値チューニング
        self.context = {
            "timeframe": None,
            "symbol": None,
            "regime_gating": False,
            "threshold_profile": "normal",
        }
        # geneに含まれる指標一覧をオプションで保持（素名比較時の安全な参照に利用）
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

            # 選択された戦略に基づいて条件を生成
            if strategy_type == StrategyType.DIFFERENT_INDICATORS:
                longs, shorts, exits = self._generate_different_indicators_strategy(
                    indicators
                )
            elif strategy_type == StrategyType.TIME_SEPARATION:
                longs, shorts, exits = self._generate_time_separation_strategy(
                    indicators
                )
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

            # 成立性向上のため、各サイドの条件数を最大2に制限（ANDの過剰による成立低下を防止）
            import random as _rnd

            def _prefer_condition(conds, prefer: str):
                """成立しやすい条件を優先して1つ選ぶ。なければランダム。"""
                if not conds:
                    return []

                def is_price_vs_trend(c):
                    return (
                        isinstance(c.right_operand, str)
                        and c.left_operand in ("close", "open")
                        and any(
                            name in str(c.right_operand)
                            for name in (
                                "SMA",
                                "EMA",
                                "WMA",
                                "TRIMA",
                                "KAMA",
                                "T3",
                                "BBANDS",
                            )
                        )
                        and (
                            (prefer == "long" and c.operator == ">")
                            or (prefer == "short" and c.operator == "<")
                        )
                    )

                def is_macd_zero_cross(c):
                    return (
                        isinstance(c.left_operand, str)
                        and (
                            c.left_operand.startswith("MACD")
                            or c.left_operand.startswith("MACDEXT")
                        )
                        and isinstance(c.right_operand, (int, float))
                        and (
                            (
                                prefer == "long"
                                and c.operator == ">"
                                and c.right_operand == 0
                            )
                            or (
                                prefer == "short"
                                and c.operator == "<"
                                and c.right_operand == 0
                            )
                        )
                    )

                def is_rsi_threshold(c):
                    return (
                        isinstance(c.left_operand, str)
                        and c.left_operand.startswith("RSI")
                        and isinstance(c.right_operand, (int, float))
                        and (
                            (
                                prefer == "long"
                                and c.operator == "<"
                                and c.right_operand <= 45
                            )
                            or (
                                prefer == "short"
                                and c.operator == ">"
                                and c.right_operand >= 55
                            )
                        )
                    )

                def is_price_vs_open(c):
                    return (
                        isinstance(c.right_operand, str)
                        and c.right_operand == "open"
                        and c.left_operand == "close"
                        and (
                            (prefer == "long" and c.operator == ">")
                            or (prefer == "short" and c.operator == "<")
                        )
                    )

                # 優先順にフィルター
                for chooser in (
                    is_price_vs_trend,
                    is_macd_zero_cross,
                    is_rsi_threshold,
                    is_price_vs_open,
                ):
                    filtered = [c for c in conds if chooser(c)]
                    if filtered:
                        return [_rnd.choice(filtered)]

                # 見つからなければランダム
                return [_rnd.choice(conds)]

            # ORグループ候補保存（削減前の候補を保持）
            long_candidates = list(longs)
            short_candidates = list(shorts)

            # 各サイド1条件に制限しつつ、優先ロジックで選定
            longs = _prefer_condition(longs, "long")
            shorts = _prefer_condition(shorts, "short")

            # 追加保証: 価格vsトレンド条件が両サイドに全く無い場合はロング側に1つ注入
            try:

                def _has_price_vs_trend(conds):
                    from app.services.auto_strategy.models.strategy_models import (
                        ConditionGroup as _CG,
                    )

                    return any(
                        (
                            isinstance(c, Condition)
                            and c.left_operand in ("close", "open")
                            and isinstance(c.right_operand, str)
                            and c.right_operand
                            in ("SMA", "EMA", "WMA", "TRIMA", "KAMA")
                        )
                        for c in conds
                        if not isinstance(c, _CG)
                    )

                if not _has_price_vs_trend(longs) and not _has_price_vs_trend(shorts):
                    # 指標からトレンド候補を選択
                    trend_names = [
                        ind.type
                        for ind in indicators
                        if ind.enabled
                        and INDICATOR_CHARACTERISTICS.get(ind.type, {}).get("type")
                        == IndicatorType.TREND
                    ]
                    pref = [
                        n for n in trend_names if n in ("SMA", "EMA", "WMA", "TRIMA")
                    ]
                    chosen = (
                        pref[0] if pref else (trend_names[0] if trend_names else None)
                    )
                    if chosen:
                        longs = longs + [
                            Condition(
                                left_operand="close", operator=">", right_operand=chosen
                            )
                        ]
                        # 最大2条件までに抑制
                        if len(longs) > 2:
                            longs = longs[:2]
            except Exception:
                pass

            # OR条件グループを追加して A AND (B OR C) の形を試みる
            try:
                from app.services.auto_strategy.models.strategy_models import (
                    ConditionGroup as _CG,
                )

                # ロング側
                lg_or = self.generate_or_group("long", long_candidates)
                if lg_or and all(not isinstance(x, _CG) for x in longs):
                    longs = longs + lg_or
                # ショート側
                sh_or = self.generate_or_group("short", short_candidates)
                if sh_or and all(not isinstance(x, _CG) for x in shorts):
                    shorts = shorts + sh_or

                # レジーム切替（オプトイン: context.regime_gating=True のときのみ）
                try:
                    context = getattr(self, "context", None)
                    if (
                        context
                        and isinstance(context, dict)
                        and context.get("regime_gating")
                    ):
                        present = {ind.type for ind in indicators if ind.enabled}
                        # トレンドゲート: ADX>25（あれば） + CHOP<50（あれば）
                        trend_gate: list[Condition] = []
                        if "ADX" in present:
                            trend_gate.append(
                                Condition(
                                    left_operand="ADX", operator=">", right_operand=25
                                )
                            )
                        if "CHOP" in present:
                            trend_gate.append(
                                Condition(
                                    left_operand="CHOP", operator="<", right_operand=50
                                )
                            )
                        # レンジゲート: CHOP>55（あれば）
                        range_gate: list[Condition] = []
                        if "CHOP" in present:
                            range_gate.append(
                                Condition(
                                    left_operand="CHOP", operator=">", right_operand=55
                                )
                            )
                        # ゲートは対象インジケータがある場合のみ適用（ANDで前置）
                        if longs and trend_gate:
                            longs = trend_gate + longs
                        if shorts and range_gate:
                            shorts = range_gate + shorts
                except Exception:
                    pass
            except Exception:
                pass

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

    def generate_or_group(self, prefer, candidates):
        """(B OR C) のORグループを1つ作って返す。見つからなければ空。"""
        try:
            # price vs trend 系を優先して2候補取り、無ければRSI/MACD系、最後に価格vs open
            import random as _rnd

            def is_price_vs_trend(c):
                return (
                    isinstance(c.right_operand, str)
                    and c.left_operand in ("close", "open")
                    and any(
                        name in str(c.right_operand)
                        for name in (
                            "SMA",
                            "EMA",
                            "WMA",
                            "TRIMA",
                            "KAMA",
                            "T3",
                            "BBANDS",
                        )
                    )
                )

            def is_rsi_or_macd(c):
                return isinstance(c.left_operand, str) and (
                    c.left_operand.startswith("RSI")
                    or c.left_operand.startswith("MACD")
                    or c.left_operand.startswith("MACDEXT")
                )

            pool = [c for c in candidates if is_price_vs_trend(c)]
            if len(pool) < 2:
                pool += [c for c in candidates if is_rsi_or_macd(c) and c not in pool]
            if len(pool) < 2:
                pool += [
                    c
                    for c in candidates
                    if (isinstance(c.right_operand, str) and c.right_operand == "open")
                    and c not in pool
                ]

            pool = [c for c in pool if c in candidates]
            if len(pool) >= 2:
                selected = _rnd.sample(pool, 2)
            elif len(pool) == 1:
                selected = pool
            else:
                selected = []

            if selected:
                from app.services.auto_strategy.models.strategy_models import (
                    ConditionGroup,
                )

                return [ConditionGroup(conditions=selected)]
            return []
        except Exception:
            return []

    def _select_strategy_type(self, indicators: List[IndicatorGene]) -> StrategyType:
        """
        利用可能な指標に基づいて戦略タイプを選択

        Args:
            indicators: 指標リスト

        Returns:
            選択された戦略タイプ
        """
        try:
            # 指標の種類を分析
            indicator_types = set()
            ml_indicators = []
            technical_indicators = []

            for indicator in indicators:
                if indicator.enabled and indicator.type in INDICATOR_CHARACTERISTICS:
                    char = INDICATOR_CHARACTERISTICS[indicator.type]
                    indicator_types.add(char["type"])

                    # ML指標とテクニカル指標を分類
                    if char.get("ml_prediction", False):
                        ml_indicators.append(indicator)
                    else:
                        technical_indicators.append(indicator)

            # ML指標のみの場合は指標特性活用戦略（ML専用ロジック）
            if ml_indicators and not technical_indicators:
                return StrategyType.INDICATOR_CHARACTERISTICS

            # ML指標とテクニカル指標の混合の場合は異なる指標の組み合わせ戦略
            if ml_indicators and technical_indicators:
                return StrategyType.DIFFERENT_INDICATORS

            # 複数の指標タイプがある場合は異なる指標の組み合わせ戦略
            if len(indicator_types) >= 2:
                return StrategyType.DIFFERENT_INDICATORS

            # 同じ指標が複数ある場合は時間軸分離戦略
            indicator_counts = {}
            for indicator in indicators:
                if indicator.enabled:
                    indicator_counts[indicator.type] = (
                        indicator_counts.get(indicator.type, 0) + 1
                    )

            if any(count >= 2 for count in indicator_counts.values()):
                return StrategyType.TIME_SEPARATION

            # ボリンジャーバンドがある場合は指標特性活用戦略
            if any(
                indicator.type == "BB" and indicator.enabled for indicator in indicators
            ):
                return StrategyType.INDICATOR_CHARACTERISTICS

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

    def _dynamic_classify(self, indicators: List[IndicatorGene]) -> dict:
        """レジストリ情報に基づいて指標を動的に分類（未特性化を拾う）"""
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
            if cfg and hasattr(cfg, "category") and getattr(cfg, "category", None):
                try:
                    # config.category は str のはず
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
                        # 未知カテゴリは一旦トレンドに寄せる（価格系多めのため）
                        categorized[IndicatorType.TREND].append(ind)
                except Exception:
                    categorized[IndicatorType.TREND].append(ind)
            else:
                # INDICATOR_CHARACTERISTICS にあれば既存分類
                if ind.type in INDICATOR_CHARACTERISTICS:
                    char = INDICATOR_CHARACTERISTICS[ind.type]
                    categorized[char["type"]].append(ind)
                else:
                    # それでも分類不可ならトレンドへ
                    categorized[IndicatorType.TREND].append(ind)
        return categorized

    def _generic_long_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """レジストリのスケール/特性ベースで汎用ロング条件を生成（閾値拡充）"""
        name = ind.type
        cfg = indicator_registry.get_indicator_config(name)
        scale = getattr(cfg, "scale_type", None) if cfg else None

        # 価格系は素直に価格比較（VALID_INDICATOR_TYPESに含まれる指標のみ）
        if name in (
            "SMA",
            "EMA",
            "WMA",
            "KAMA",
            "T3",
            "TRIMA",
            "MIDPOINT",
        ):
            # 右オペランドは gene に含まれるトレンド名に限定。無ければ 'open' に退避
            trend_names_in_gene = [
                ind.type
                for ind in (self.indicators or [])
                if getattr(ind, "enabled", True)
            ]
            right_name = name if name in trend_names_in_gene else "open"
            return [
                Condition(left_operand="close", operator=">", right_operand=right_name)
            ]

        # 0-100オシレーター系（RSI/MFI/STOCH/KDJ/QQE/ADX等）
        if scale == IndicatorScaleType.OSCILLATOR_0_100:
            context = getattr(self, "context", None)
            profile = (
                context.get("threshold_profile", "normal")
                if context and isinstance(context, dict)
                else "normal"
            )
            if name in {"RSI", "STOCH", "STOCHRSI", "KDJ", "QQE", "MFI"}:
                # プロファイルごとの閾値は ThresholdPolicy に一元化
                policy = ThresholdPolicy.get(profile)
                thr = (
                    policy.rsi_long_lt
                    if name != "MFI"
                    else (policy.mfi_long_lt or policy.rsi_long_lt)
                )
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]
            if name == "ADX":
                # ADXは方向ではなくトレンド強度 → フィルタとして利用
                thr = ThresholdPolicy.get(profile).adx_trend_min
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]
            # デフォルト
            thr = (
                48
                if profile == "aggressive"
                else (52 if profile == "conservative" else 50)
            )
            return [
                Condition(left_operand=name, operator=">", right_operand=float(thr))
            ]

        # ±100系（CCI/WILLR/AROONOSC等）
        if scale == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
            context = getattr(self, "context", None)
            profile = (
                context.get("threshold_profile", "normal")
                if context and isinstance(context, dict)
                else "normal"
            )
            if name == "CCI":
                # 方向性のある強めのしきい値（ポリシー化）
                lim = ThresholdPolicy.get(profile).cci_abs_limit or 100
                return [
                    Condition(
                        left_operand=name,
                        operator=">",
                        right_operand=float(-lim / 20.0),
                    )
                ]
            if name == "WILLR":
                # -100..0（-50の上は相対的に強め）→ ポリシー化
                p = ThresholdPolicy.get(profile)
                thr = -p.willr_long_lt if p.willr_long_lt is not None else -50
                return [
                    Condition(left_operand=name, operator=">", right_operand=float(thr))
                ]
            # AO, AROONOSC 等はゼロ上
            thr = (
                -2
                if profile == "aggressive"
                else (2 if profile == "conservative" else 0)
            )
            return [
                Condition(left_operand=name, operator=">", right_operand=float(thr))
            ]

        # ゼロセンター系（MACD/PPO/APO/TRIX/TSIなど）
        if scale == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
            context = getattr(self, "context", None)
            thr = (
                -0.0
                if context
                and isinstance(context, dict)
                and context.get("threshold_profile") == "aggressive"
                else 0
            )
            return [
                Condition(left_operand=name, operator=">", right_operand=float(thr))
            ]

        # 比率/絶対価格
        # 注意: PRICE_RATIO/PRICE_ABSOLUTE の中でもトレンド系以外（ATR/TRANGE/STDDEV等）は価格との直接比較は不適切
        # ここでは汎用生成を行わずスキップし、他のロジック（価格vsトレンド補完等）に委ねる
        if scale in {IndicatorScaleType.PRICE_RATIO, IndicatorScaleType.PRICE_ABSOLUTE}:
            return []

        # フォールバック
        context = getattr(self, "context", None)
        thr = (
            -0.0
            if context
            and isinstance(context, dict)
            and context.get("threshold_profile") == "aggressive"
            else 0
        )
        return [Condition(left_operand=name, operator=">", right_operand=float(thr))]

    def _generic_short_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """レジストリのスケール/特性ベースで汎用ショート条件を生成（閾値拡充）"""
        name = ind.type
        cfg = indicator_registry.get_indicator_config(name)
        scale = getattr(cfg, "scale_type", None) if cfg else None

        # 価格系は素直に価格比較
        if name in (
            "SMA",
            "EMA",
            "WMA",
            "KAMA",
            "T3",
            "TRIMA",
            "MIDPOINT",
        ):
            # 右オペランドは gene に含まれるトレンド名に限定。無ければ 'open' に退避
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
            context = getattr(self, "context", None)
            profile = (
                context.get("threshold_profile", "normal")
                if context and isinstance(context, dict)
                else "normal"
            )
            if name in {"RSI", "STOCH", "STOCHRSI", "KDJ", "QQE", "MFI"}:
                p = ThresholdPolicy.get(profile)
                thr = (
                    p.rsi_short_gt
                    if name != "MFI"
                    else (p.mfi_short_gt or p.rsi_short_gt)
                )
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]
            if name == "ADX":
                # トレンド強度フィルタ（方向性は持たない）
                thr = float(100 - ThresholdPolicy.get(profile).adx_trend_min)
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]
            thr = (
                52
                if profile == "aggressive"
                else (48 if profile == "conservative" else 50)
            )
            return [
                Condition(left_operand=name, operator="<", right_operand=float(thr))
            ]

        # ±100系
        if scale == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
            profile = (
                (self.context or {}).get("threshold_profile", "normal")
                if hasattr(self, "context")
                else "normal"
            )
            if name == "CCI":
                thr = (
                    5
                    if profile == "aggressive"
                    else (-5 if profile == "conservative" else 0)
                )
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]
            if name == "WILLR":
                thr = (
                    -40
                    if profile == "aggressive"
                    else (-60 if profile == "conservative" else -50)
                )
                return [
                    Condition(left_operand=name, operator="<", right_operand=float(thr))
                ]
            thr = (
                2
                if profile == "aggressive"
                else (-2 if profile == "conservative" else 0)
            )
            return [
                Condition(left_operand=name, operator="<", right_operand=float(thr))
            ]

        # ゼロセンター系
        if scale == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
            thr = 0.0
            return [
                Condition(left_operand=name, operator="<", right_operand=float(thr))
            ]

        # 比率/絶対価格
        # 注意: PRICE_RATIO/PRICE_ABSOLUTE の中でもトレンド系以外（ATR/TRANGE/STDDEV等）は価格との直接比較は不適切
        # ここでは汎用生成を行わずスキップ
        if scale in {IndicatorScaleType.PRICE_RATIO, IndicatorScaleType.PRICE_ABSOLUTE}:
            return []

        # フォールバック
        thr = 0.0  # デフォルト閾値を設定
        return [Condition(left_operand=name, operator="<", right_operand=float(thr))]

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

            # トレンド系 + モメンタム系の組み合わせを優先
            long_conditions: List[Union[Condition, ConditionGroup]] = []
            short_conditions: List[Union[Condition, ConditionGroup]] = []

            # ML指標とテクニカル指標の混合戦略
            ml_indicators = [
                ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
            ]
            technical_indicators = [
                ind
                for ind in indicators
                if ind.enabled and not ind.type.startswith("ML_")
            ]

            # ロング条件：ML指標 + テクニカル指標の組み合わせ
            if ml_indicators and technical_indicators:
                # ML予測とテクニカル指標の合意による高信頼度条件
                ml_long_conditions = self._create_ml_long_conditions(ml_indicators)
                long_conditions.extend(ml_long_conditions)

                # テクニカル指標でML予測を補強
                if indicators_by_type[IndicatorType.TREND]:
                    trend_indicator = random.choice(
                        indicators_by_type[IndicatorType.TREND]
                    )
                    long_conditions.extend(
                        self._create_trend_long_conditions(trend_indicator)
                    )

                if indicators_by_type[IndicatorType.MOMENTUM]:
                    momentum_indicator = random.choice(
                        indicators_by_type[IndicatorType.MOMENTUM]
                    )
                    long_conditions.extend(
                        self._create_momentum_long_conditions(momentum_indicator)
                    )
            else:
                # 従来のテクニカル指標のみの処理
                if indicators_by_type[IndicatorType.TREND]:
                    trend_indicator = random.choice(
                        indicators_by_type[IndicatorType.TREND]
                    )
                    long_conditions.extend(
                        self._create_trend_long_conditions(trend_indicator)
                    )

                if indicators_by_type[IndicatorType.MOMENTUM]:
                    momentum_indicator = random.choice(
                        indicators_by_type[IndicatorType.MOMENTUM]
                    )
                    long_conditions.extend(
                        self._create_momentum_long_conditions(momentum_indicator)
                    )

            # ショート条件：基本的な条件のみ使用
            if ml_indicators:
                # ML指標がある場合は基本的なロング条件のみ生成
                ml_long_conditions = self._create_ml_long_conditions(ml_indicators)
                long_conditions.extend(ml_long_conditions)

            # 基本的なショート条件の生成（常に有効な形に限定）
            # 価格 vs トレンド系のショート条件を優先して1つ作成
            trend_pool_all = list(indicators_by_type[IndicatorType.TREND])
            pref_names = {"SMA", "EMA", "WMA", "TRIMA"}
            trend_pool = [
                ind for ind in trend_pool_all if ind.type in pref_names
            ] or trend_pool_all
            if trend_pool:
                trend_indicator = random.choice(trend_pool)
                trend_name = trend_indicator.type
                short_conditions.append(
                    Condition(
                        left_operand="close", operator="<", right_operand=trend_name
                    )
                )

            # モメンタム系からも1つだけ追加（閾値はレンジに基づく）
            if indicators_by_type[IndicatorType.MOMENTUM]:
                momentum_indicator = random.choice(
                    indicators_by_type[IndicatorType.MOMENTUM]
                )
                momentum_name = momentum_indicator.type
                if momentum_name == "RSI":
                    threshold = random.uniform(60, 75)
                    short_conditions.append(
                        Condition(
                            left_operand=momentum_name,
                            operator=">",
                            right_operand=threshold,
                        )
                    )
                elif momentum_name in {"STOCH", "STOCHRSI"}:
                    threshold = random.uniform(70, 90)
                    short_conditions.append(
                        Condition(
                            left_operand=momentum_name,
                            operator=">",
                            right_operand=threshold,
                        )
                    )
                elif momentum_name in {"MACD", "MACDEXT", "PPO", "APO", "TRIX", "TSI"}:
                    # MACD/MACDEXTショートはゼロ下 or シグナル下で評価器が扱えるようにメイン名で
                    short_conditions.append(
                        Condition(
                            left_operand=(
                                "MACD_0"
                                if momentum_name == "MACD"
                                else (
                                    "MACDEXT_0"
                                    if momentum_name == "MACDEXT"
                                    else momentum_name
                                )
                            ),
                            operator="<",
                            right_operand=0,
                        )
                    )

            # 最低1つのショート条件を保証
            if not short_conditions:
                short_conditions.append(
                    Condition(left_operand="close", operator="<", right_operand="open")
                )

            # 新しいカテゴリのインジケータを活用

            # 統計系指標の追加（ロング条件のみ）
            if indicators_by_type[IndicatorType.STATISTICS]:
                stats_indicator = random.choice(
                    indicators_by_type[IndicatorType.STATISTICS]
                )
                long_conditions.extend(
                    self._create_statistics_long_conditions(stats_indicator)
                )

            # パターン認識系指標の追加（ロング条件のみ）
            if indicators_by_type[IndicatorType.PATTERN_RECOGNITION]:
                pattern_indicator = random.choice(
                    indicators_by_type[IndicatorType.PATTERN_RECOGNITION]
                )
                long_conditions.extend(
                    self._create_pattern_long_conditions(pattern_indicator)
                )

            # 成立性を下支えするため、各サイドに価格 vs トレンド系の条件を最低1つ保証
            def _ensure_price_vs_trend(
                side_conds: List[Union[Condition, ConditionGroup]], prefer: str
            ):
                has_price_vs_trend = any(
                    isinstance(c, Condition)  # 明示的にCondition型をチェック
                    and isinstance(c.right_operand, str)
                    and c.left_operand in ("close", "open")
                    and c.right_operand in ("SMA", "EMA", "WMA", "TRIMA", "KAMA")
                    for c in side_conds
                )
                if not has_price_vs_trend:
                    # トレンド指標があればそれを優先、無ければ価格系列(open)に退避
                    chosen_name = None
                    if indicators_by_type[IndicatorType.TREND]:
                        # gene に含まれるトレンド系のみを使う
                        pref_names = {"SMA", "EMA", "WMA", "TRIMA"}
                        trend_names_in_gene = [
                            ind.type
                            for ind in indicators_by_type[IndicatorType.TREND]
                            if ind.enabled and ind.type in pref_names
                        ] or [
                            ind.type
                            for ind in indicators_by_type[IndicatorType.TREND]
                            if ind.enabled
                        ]
                        if trend_names_in_gene:
                            chosen_name = random.choice(trend_names_in_gene)
                    if not chosen_name:
                        chosen_name = "open"

                    side_conds.append(
                        Condition(
                            left_operand="close",
                            operator=">" if prefer == "long" else "<",
                            right_operand=chosen_name,
                        )
                    )

            _ensure_price_vs_trend(long_conditions, "long")
            _ensure_price_vs_trend(short_conditions, "short")

            # 条件が空の場合はフォールバック
            if not long_conditions:
                if indicators_by_type[IndicatorType.TREND]:
                    trend_indicator = random.choice(
                        indicators_by_type[IndicatorType.TREND]
                    )
                    long_conditions = [
                        Condition(
                            left_operand="close",
                            operator=">",
                            right_operand=trend_indicator.type,
                        )
                    ]
                else:
                    long_conditions = [
                        Condition(
                            left_operand="close", operator=">", right_operand="open"
                        )
                    ]
            if not short_conditions:
                if indicators_by_type[IndicatorType.TREND]:
                    trend_indicator = random.choice(
                        indicators_by_type[IndicatorType.TREND]
                    )
                    short_conditions = [
                        Condition(
                            left_operand="close",
                            operator="<",
                            right_operand=trend_indicator.type,
                        )
                    ]
                else:
                    short_conditions = [
                        Condition(
                            left_operand="close", operator="<", right_operand="open"
                        )
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

    def _create_trend_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """トレンド系指標のロング条件を生成"""
        # テスト互換性: 素名優先
        indicator_name = indicator.type

        if indicator.type in ["SMA", "EMA", "WMA", "TRIMA"]:
            return [
                Condition(
                    left_operand="close", operator=">", right_operand=indicator_name
                )
            ]
        else:
            return []

    def _create_trend_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """トレンド系指標のショート条件を生成"""
        # テスト互換性: 素名優先
        indicator_name = indicator.type

        if indicator.type in ["SMA", "EMA", "WMA", "TRIMA"]:
            return [
                Condition(
                    left_operand="close", operator="<", right_operand=indicator_name
                )
            ]
        else:
            return []

    def _create_momentum_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """モメンタム系指標のロング条件を生成"""
        # テスト互換性: レジストリ由来の素名を優先（RSI_14 -> RSI）
        indicator_name = indicator.type

        if indicator.type == "RSI":
            # RSI: 時間軸依存閾値（例: 15m/1h/4hで 55/60/65 を中心にゾーン化）
            context = getattr(self, "context", None)
            tf = (
                context.get("timeframe")
                if context and isinstance(context, dict)
                else None
            )
            if tf in ("15m", "15min", "15"):
                base = 55
            elif tf in ("1h", "60"):
                base = 60
            elif tf in ("4h", "240"):
                base = 65
            else:
                base = 60
            # ロングは売られすぎ or 中立下限割れからの回復狙い
            return [
                Condition(
                    left_operand=indicator_name,
                    operator="<",
                    right_operand=max(25, base - 25),
                )
            ]
        elif indicator.type == "STOCH":
            # STOCH: %K/%D クロス + ゾーン（20/80）
            # 評価器側の簡便性のため、%K(=STOCH_0), %D(=STOCH_1) 名で扱う
            return [
                # ゾーン条件（売られすぎ）
                Condition(left_operand="STOCH_0", operator="<", right_operand=20),
            ]
        elif indicator.type == "CCI":
            # CCI: 売られすぎ領域でロング
            threshold = random.uniform(-150, -80)
            return [
                Condition(
                    left_operand=indicator_name, operator="<", right_operand=threshold
                )
            ]
        elif indicator.type in {"MACD", "MACDEXT"}:
            # MACD/MACDEXT: ゼロライン or シグナルクロスのハイブリッド
            # 評価器では *_0=メイン, *_1=シグナル
            return [
                Condition(
                    left_operand=(
                        "MACD_0" if indicator.type == "MACD" else "MACDEXT_0"
                    ),
                    operator=">",
                    right_operand=0,
                )
            ]
        else:
            return []

    def _create_momentum_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """モメンタム系指標のショート条件を生成"""
        # テスト互換性: レジストリ由来の素名を優先（RSI_14 -> RSI）
        indicator_name = indicator.type

        if indicator.type == "RSI":
            # RSI: 時間軸依存閾値（例: 15m/1h/4hで 55/60/65 を中心にゾーン化）
            context = getattr(self, "context", None)
            tf = (
                context.get("timeframe")
                if context and isinstance(context, dict)
                else None
            )
            if tf in ("15m", "15min", "15"):
                base = 55
            elif tf in ("1h", "60"):
                base = 60
            elif tf in ("4h", "240"):
                base = 65
            else:
                base = 60
            # ショートは買われすぎ or 中立上限超えからの下降狙い
            return [
                Condition(
                    left_operand=indicator_name,
                    operator=">",
                    right_operand=min(75, base + 25),
                )
            ]
        elif indicator.type == "STOCH":
            # STOCH: %K/%D クロス + ゾーン（20/80）
            # 評価器側の簡便性のため、%K(=STOCH_0), %D(=STOCH_1) 名で扱う
            return [
                # ゾーン条件（買われすぎ）
                Condition(left_operand="STOCH_0", operator=">", right_operand=80),
            ]
        elif indicator.type == "CCI":
            # CCI: 買われすぎ領域でショート
            threshold = random.uniform(80, 150)
            return [
                Condition(
                    left_operand=indicator_name, operator=">", right_operand=threshold
                )
            ]
        elif indicator.type in {"MACD", "MACDEXT"}:
            # MACD/MACDEXT: ゼロライン or シグナルクロスのハイブリッド
            # 評価器では *_0=メイン, *_1=シグナル
            return [
                Condition(
                    left_operand=(
                        "MACD_0" if indicator.type == "MACD" else "MACDEXT_0"
                    ),
                    operator="<",
                    right_operand=0,
                )
            ]
        else:
            return []

    def _create_statistics_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """統計系指標のロング条件を生成"""
        # テスト互換性: 指標は素名で参照
        indicator_name = indicator.type

        if indicator.type == "CORREL":
            # 正の相関でロング
            threshold = random.uniform(0.3, 0.7)
            return [
                Condition(
                    left_operand=indicator_name, operator=">", right_operand=threshold
                )
            ]
        elif indicator.type == "LINEARREG_ANGLE":
            # 上昇角度でロング
            threshold = random.uniform(10, 45)
            return [
                Condition(
                    left_operand=indicator_name, operator=">", right_operand=threshold
                )
            ]
        elif indicator.type == "LINEARREG_SLOPE":
            # 正の傾きでロング
            return [
                Condition(left_operand=indicator_name, operator=">", right_operand=0)
            ]
        elif indicator.type in ["LINEARREG", "TSF"]:
            # 価格が回帰線より上でロング
            return [
                Condition(
                    left_operand="close", operator=">", right_operand=indicator_name
                )
            ]
        else:
            return []

    def _create_statistics_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """統計系指標のショート条件を生成"""
        # テスト互換性: 指標は素名で参照
        indicator_name = indicator.type

        if indicator.type == "CORREL":
            # 負の相関でショート
            threshold = random.uniform(-0.7, -0.3)
            return [
                Condition(
                    left_operand=indicator_name, operator="<", right_operand=threshold
                )
            ]
        elif indicator.type == "LINEARREG_ANGLE":
            # 下降角度でショート
            threshold = random.uniform(-45, -10)
            return [
                Condition(
                    left_operand=indicator_name, operator="<", right_operand=threshold
                )
            ]
        elif indicator.type == "LINEARREG_SLOPE":
            # 負の傾きでショート
            return [
                Condition(left_operand=indicator_name, operator="<", right_operand=0)
            ]
        elif indicator.type in ["LINEARREG", "TSF"]:
            # 価格が回帰線より下でショート
            return [
                Condition(
                    left_operand="close", operator="<", right_operand=indicator_name
                )
            ]
        else:
            return []

    def _create_pattern_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """パターン認識系指標のロング条件を生成"""
        indicator_name = f"{indicator.type}"

        if indicator.type in ["CDL_HAMMER", "CDL_PIERCING", "CDL_THREE_WHITE_SOLDIERS"]:
            # 強気パターンでロング
            return [
                Condition(left_operand=indicator_name, operator=">", right_operand=0)
            ]
        elif indicator.type == "CDL_DOJI":
            # ドージパターンは反転の可能性（文脈依存）
            return [
                Condition(left_operand=indicator_name, operator="!=", right_operand=0)
            ]
        else:
            return []

    def _create_pattern_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """パターン認識系指標のショート条件を生成"""
        indicator_name = f"{indicator.type}"

        if indicator.type in ["CDL_HANGING_MAN", "CDL_DARK_CLOUD_COVER", "CDL_THREE_BLACK_CROWS"]:
            # 弱気パターンでショート
            return [
                Condition(left_operand=indicator_name, operator="<", right_operand=0)
            ]
        elif indicator.type == "CDL_DOJI":
            # ドージパターンは反転の可能性（文脈依存）
            return [
                Condition(left_operand=indicator_name, operator="!=", right_operand=0)
            ]
        else:
            return []

    def _generate_time_separation_strategy(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        時間軸分離戦略（同じ指標の異なる期間を使用）

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            # 同じタイプの指標を見つける
            indicator_groups = {}
            for indicator in indicators:
                if indicator.enabled:
                    if indicator.type not in indicator_groups:
                        indicator_groups[indicator.type] = []
                    indicator_groups[indicator.type].append(indicator)

            long_conditions = []
            short_conditions = []

            # 複数の期間を持つ指標を選択
            for indicator_type, indicator_list in indicator_groups.items():
                if (
                    len(indicator_list) >= 2
                    and indicator_type in INDICATOR_CHARACTERISTICS
                ):
                    # 短期と長期の指標を選択
                    sorted_indicators = sorted(
                        indicator_list,
                        key=lambda x: (
                            getattr(x, "parameters", {}).get("period", 14)
                            if isinstance(getattr(x, "parameters", {}), dict)
                            else 14
                        ),
                    )
                    short_term = sorted_indicators[0]
                    long_term = sorted_indicators[-1]

                    # 短期・長期組み合わせ条件を生成
                    if indicator_type == "RSI":
                        # 短期RSI売られすぎ + 長期RSI上昇トレンド
                        short_name = f"{short_term.type}_{getattr(short_term, 'parameters', {}).get('period', 7) if isinstance(getattr(short_term, 'parameters', {}), dict) else 7}"
                        long_name = f"{long_term.type}_{getattr(long_term, 'parameters', {}).get('period', 21) if isinstance(getattr(long_term, 'parameters', {}), dict) else 21}"

                        long_conditions.extend(
                            [
                                Condition(
                                    left_operand=short_name,
                                    operator="<",
                                    right_operand=30,
                                ),
                                Condition(
                                    left_operand=long_name,
                                    operator=">",
                                    right_operand=50,
                                ),
                            ]
                        )

                        short_conditions.extend(
                            [
                                Condition(
                                    left_operand=short_name,
                                    operator=">",
                                    right_operand=70,
                                ),
                                Condition(
                                    left_operand=long_name,
                                    operator="<",
                                    right_operand=50,
                                ),
                            ]
                        )
                    break

            # 条件が空の場合はフォールバック
            if not long_conditions:
                return self._generate_fallback_conditions()

            return long_conditions, short_conditions, []

        except Exception as e:
            self.logger.error(f"時間軸分離戦略エラー: {e}")
            return self._generate_fallback_conditions()

    def _generate_indicator_characteristics_strategy(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        指標特性活用戦略（ボリンジャーバンドの正しい実装など）

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            long_conditions = []
            short_conditions = []

            # ML指標専用の処理
            ml_indicators = [
                ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
            ]
            if ml_indicators:
                # ML指標のみの場合の特別な戦略（ロング条件のみ）
                ml_long_conditions = self._create_ml_long_conditions(indicators)
                long_conditions.extend(ml_long_conditions)

                # ML指標の組み合わせ戦略
                if len(ml_indicators) >= 2:
                    # 複数のML指標の合意による高信頼度条件
                    up_prob_indicators = [
                        ind for ind in ml_indicators if ind.type == "ML_UP_PROB"
                    ]
                    down_prob_indicators = [
                        ind for ind in ml_indicators if ind.type == "ML_DOWN_PROB"
                    ]

                    if up_prob_indicators and down_prob_indicators:
                        # 上昇確率が高く、下落確率が低い場合
                        long_conditions.append(
                            Condition(
                                left_operand="ML_UP_PROB",
                                operator=">",
                                right_operand=0.7,
                            )
                        )
                        long_conditions.append(
                            Condition(
                                left_operand="ML_DOWN_PROB",
                                operator="<",
                                right_operand=0.3,
                            )
                        )

                        # 下落確率が高く、上昇確率が低い場合
                        short_conditions.append(
                            Condition(
                                left_operand="ML_DOWN_PROB",
                                operator=">",
                                right_operand=0.7,
                            )
                        )
                        short_conditions.append(
                            Condition(
                                left_operand="ML_UP_PROB",
                                operator="<",
                                right_operand=0.3,
                            )
                        )

            # ボリンジャーバンドの特性を活用
            bb_indicators = [
                ind for ind in indicators if ind.type == "BB" and ind.enabled
            ]
            if bb_indicators and not ml_indicators:  # ML指標がない場合のみ
                bb_indicator = bb_indicators[0]
                period = (
                    getattr(bb_indicator, "parameters", {}).get("period", 20)
                    if isinstance(getattr(bb_indicator, "parameters", {}), dict)
                    else 20
                )

                # ボリンジャーバンドの3つの値を活用（計画書の設計通り）
                bb_upper = f"BB_Upper_{period}"
                bb_middle = f"BB_Middle_{period}"
                bb_lower = f"BB_Lower_{period}"

                # 逆張り戦略：バンド突破後の回帰を狙う
                long_conditions.extend(
                    [
                        Condition(
                            left_operand="close", operator="<", right_operand=bb_lower
                        ),  # 下限突破
                        Condition(
                            left_operand="close", operator=">", right_operand=bb_middle
                        ),  # 中央線回復
                    ]
                )

                short_conditions.extend(
                    [
                        Condition(
                            left_operand="close", operator="<", right_operand=bb_upper
                        ),  # 上限突破
                        Condition(
                            left_operand="close", operator=">", right_operand=bb_middle
                        ),  # 中央線割れ
                    ]
                )

            # ADXの正しい活用（方向性指標との組み合わせ）
            adx_indicators = [
                ind for ind in indicators if ind.type == "ADX" and ind.enabled
            ]
            if adx_indicators and not bb_indicators:
                adx_indicator = adx_indicators[0]
                period = (
                    getattr(adx_indicator, "parameters", {}).get("period", 14)
                    if isinstance(getattr(adx_indicator, "parameters", {}), dict)
                    else 14
                )
                adx_name = f"ADX_{period}"

                # ADX + 価格方向の組み合わせ（計画書の設計通り）
                long_conditions.extend(
                    [
                        Condition(
                            left_operand=adx_name, operator=">", right_operand=25
                        ),  # 強いトレンド
                        Condition(
                            left_operand="close", operator=">", right_operand="open"
                        ),  # 上昇方向
                    ]
                )

                short_conditions.extend(
                    [
                        Condition(
                            left_operand=adx_name, operator=">", right_operand=25
                        ),  # 強いトレンド
                        Condition(
                            left_operand="close", operator="<", right_operand="open"
                        ),  # 下降方向
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
                    elif indicator_type == IndicatorType.STATISTICS:
                        long_conds = self._create_statistics_long_conditions(indicator)
                        short_conds = self._create_statistics_short_conditions(indicator)
                    elif indicator_type == IndicatorType.PATTERN_RECOGNITION:
                        long_conds = self._create_pattern_long_conditions(indicator)
                        short_conds = self._create_pattern_short_conditions(indicator)
                    else:
                        # 未知の指標タイプの場合は汎用条件を使用
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
        ML予測を活用したロング条件を生成

        Args:
            indicators: 指標リスト

        Returns:
            ロング条件のリスト
        """
        try:
            conditions = []

            # ML予測確率指標を探す
            ml_indicators = [
                ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
            ]

            if any(ind.type == "ML_UP_PROB" for ind in ml_indicators):
                # 上昇予測確率が高い場合
                conditions.append(
                    Condition(
                        left_operand="ML_UP_PROB", operator=">", right_operand=0.6
                    )
                )
                self.logger.debug("Added ML_UP_PROB > 0.6 to long conditions")

            if any(ind.type == "ML_DOWN_PROB" for ind in ml_indicators):
                # 下落予測確率が低い場合
                conditions.append(
                    Condition(
                        left_operand="ML_DOWN_PROB", operator="<", right_operand=0.4
                    )
                )
                self.logger.debug("Added ML_DOWN_PROB < 0.4 to long conditions")

            if any(ind.type == "ML_RANGE_PROB" for ind in ml_indicators):
                # レンジ予測確率が低い場合（トレンド発生の可能性）
                conditions.append(
                    Condition(
                        left_operand="ML_RANGE_PROB", operator="<", right_operand=0.3
                    )
                )
                self.logger.debug("Added ML_RANGE_PROB < 0.3 to long conditions")

            # 高信頼度のML予測条件
            if any(ind.type == "ML_UP_PROB" for ind in ml_indicators):
                # 非常に高い上昇確率
                conditions.append(
                    Condition(
                        left_operand="ML_UP_PROB", operator=">", right_operand=0.8
                    )
                )

            # ML予測の組み合わせ条件
            if any(ind.type == "ML_UP_PROB" for ind in ml_indicators) and any(
                ind.type == "ML_DOWN_PROB" for ind in ml_indicators
            ):
                # 上昇確率が下落確率より高い
                conditions.append(
                    Condition(
                        left_operand="ML_UP_PROB",
                        operator=">",
                        right_operand="ML_DOWN_PROB",
                    )
                )

            return conditions

        except Exception as e:
            self.logger.error(f"MLロング条件生成エラー: {e}")
            return []
