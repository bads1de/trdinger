import logging
import random
from enum import Enum
from typing import List, Tuple

from ..models.gene_strategy import Condition, IndicatorGene

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
    MATH_TRANSFORM = "math_transform"  # 数学変換系
    MATH_OPERATORS = "math_operators"  # 数学演算子系
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
    # 数学変換系インジケータ（三角関数）
    "ACOS": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": (0, 3.14159),  # アークコサイン範囲
        "math_function": True,
    },
    "ASIN": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": (-1.5708, 1.5708),  # アークサイン範囲
        "math_function": True,
        "zero_cross": True,
    },
    "ATAN": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": (-1.5708, 1.5708),  # アークタンジェント範囲
        "math_function": True,
        "zero_cross": True,
    },
    "COS": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": (-1, 1),  # コサイン範囲
        "math_function": True,
        "zero_cross": True,
    },
    "SIN": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": (-1, 1),  # サイン範囲
        "math_function": True,
        "zero_cross": True,
    },
    "TAN": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": None,  # タンジェントは無限大になる可能性
        "math_function": True,
        "zero_cross": True,
    },
    # 数学変換系インジケータ（その他の数学関数）
    "CEIL": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": None,  # 入力依存
        "math_function": True,
        "price_comparison": True,
    },
    "FLOOR": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": None,  # 入力依存
        "math_function": True,
        "price_comparison": True,
    },
    "SQRT": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": (0, None),  # 常に正値
        "math_function": True,
    },
    "LN": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": None,  # 自然対数
        "math_function": True,
        "zero_cross": True,
    },
    "LOG10": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": None,  # 常用対数
        "math_function": True,
        "zero_cross": True,
    },
    "EXP": {
        "type": IndicatorType.MATH_TRANSFORM,
        "range": (0, None),  # 指数関数は常に正値
        "math_function": True,
    },
    # 数学演算子系インジケータ
    "ADD": {
        "type": IndicatorType.MATH_OPERATORS,
        "range": None,  # 入力依存
        "math_operator": True,
        "requires_two_inputs": True,
    },
    "SUB": {
        "type": IndicatorType.MATH_OPERATORS,
        "range": None,  # 入力依存
        "math_operator": True,
        "requires_two_inputs": True,
        "zero_cross": True,
    },
    "MULT": {
        "type": IndicatorType.MATH_OPERATORS,
        "range": None,  # 入力依存
        "math_operator": True,
        "requires_two_inputs": True,
    },
    "DIV": {
        "type": IndicatorType.MATH_OPERATORS,
        "range": None,  # 入力依存
        "math_operator": True,
        "requires_two_inputs": True,
    },
    "MAX": {
        "type": IndicatorType.MATH_OPERATORS,
        "range": None,  # 入力依存
        "math_operator": True,
        "requires_two_inputs": True,
    },
    "MIN": {
        "type": IndicatorType.MATH_OPERATORS,
        "range": None,  # 入力依存
        "math_operator": True,
        "requires_two_inputs": True,
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

    def generate_balanced_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
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
                                "MA",
                                "HT_TRENDLINE",
                                "LINEARREG",
                                "TSF",
                                "BB_Middle",
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
                        and c.left_operand.startswith("MACD")
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

            # OR条件グループを追加して A AND (B OR C) の形を試みる
            try:
                from app.services.auto_strategy.models.condition_group import (
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

            return longs, shorts, exits

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
                            "MA",
                            "HT_TRENDLINE",
                            "LINEARREG",
                            "TSF",
                            "BB_Middle",
                        )
                    )
                )

            def is_rsi_or_macd(c):
                return isinstance(c.left_operand, str) and (
                    c.left_operand.startswith("RSI")
                    or c.left_operand.startswith("MACD")
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
                from app.services.auto_strategy.models.condition_group import (
                    ConditionGroup,
                )

                return [ConditionGroup(conditions=selected)]
            return []
        except Exception:
            return []

        except Exception as e:
            self.logger.error(f"スマート条件生成エラー: {e}")
            return self._generate_fallback_conditions()

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
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        フォールバック条件を生成

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        long_conditions = [
            Condition(left_operand="close", operator=">", right_operand="open")
        ]
        short_conditions = [
            Condition(left_operand="close", operator="<", right_operand="open")
        ]
        exit_conditions = []

        return long_conditions, short_conditions, exit_conditions

    def _generate_different_indicators_strategy(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        異なる指標の組み合わせ戦略

        Args:
            indicators: 指標リスト

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        try:
            # 指標をタイプ別に分類
            indicators_by_type = {
                IndicatorType.MOMENTUM: [],
                IndicatorType.TREND: [],
                IndicatorType.VOLATILITY: [],
                IndicatorType.STATISTICS: [],
                IndicatorType.MATH_TRANSFORM: [],
                IndicatorType.MATH_OPERATORS: [],
                IndicatorType.PATTERN_RECOGNITION: [],
            }

            for indicator in indicators:
                if indicator.enabled and indicator.type in INDICATOR_CHARACTERISTICS:
                    char = INDICATOR_CHARACTERISTICS[indicator.type]
                    indicators_by_type[char["type"]].append(indicator)

            # トレンド系 + モメンタム系の組み合わせを優先
            long_conditions = []
            short_conditions = []

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

            # 基本的なショート条件の生成
            if indicators_by_type[IndicatorType.TREND]:
                trend_indicator = random.choice(indicators_by_type[IndicatorType.TREND])
                # トレンド指標の基本的なショート条件
                # テスト互換性: 素名で参照
                trend_name = trend_indicator.type
                if trend_indicator.type in ["SMA", "EMA", "MAMA"]:
                    short_conditions.append(
                        Condition(
                            left_operand="close", operator="<", right_operand=trend_name
                        )
                    )

            if indicators_by_type[IndicatorType.MOMENTUM]:
                momentum_indicator = random.choice(
                    indicators_by_type[IndicatorType.MOMENTUM]
                )
                # モメンタム指標の基本的なショート条件
                # テスト互換性: 素名で参照
                momentum_name = momentum_indicator.type
                if momentum_indicator.type == "RSI":
                    # RSI: 買われすぎ領域でショート
                    threshold = random.uniform(55, 75)
                    short_conditions.append(
                        Condition(
                            left_operand=momentum_name,
                            operator=">",
                            right_operand=threshold,
                        )
                    )
                elif momentum_indicator.type == "MACD":
                    # MACD: ゼロライン下抜けでショート
                    short_conditions.append(
                        Condition(
                            left_operand=momentum_name, operator="<", right_operand=0
                        )
                    )

                # 拡張ショート条件が生成されなかった場合でも、最低1つのショート条件を保証
                if not short_conditions:
                    short_conditions.append(
                        Condition(
                            left_operand="close", operator="<", right_operand="open"
                        )
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
            def _ensure_price_vs_trend(side_conds: List[Condition], prefer: str):
                has_price_vs_trend = any(
                    isinstance(c.right_operand, str)
                    and c.left_operand in ("close", "open")
                    and c.right_operand in ("SMA", "EMA", "MAMA")
                    for c in side_conds
                )
                if not has_price_vs_trend and indicators_by_type[IndicatorType.TREND]:
                    trend_indicator = random.choice(
                        indicators_by_type[IndicatorType.TREND]
                    )
                    trend_name = trend_indicator.type
                    if prefer == "long":
                        side_conds.append(
                            Condition(
                                left_operand="close",
                                operator=">",
                                right_operand=trend_name,
                            )
                        )
                    else:
                        side_conds.append(
                            Condition(
                                left_operand="close",
                                operator="<",
                                right_operand=trend_name,
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

            return long_conditions, short_conditions, []

        except Exception as e:
            self.logger.error(f"異なる指標組み合わせ戦略エラー: {e}")
            return self._generate_fallback_conditions()

    def _create_trend_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """トレンド系指標のロング条件を生成"""
        # テスト互換性: 素名優先
        indicator_name = indicator.type

        if indicator.type in ["SMA", "EMA", "MAMA"]:
            return [
                Condition(
                    left_operand="close", operator=">", right_operand=indicator_name
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
            # RSI: 売られすぎ領域でロング
            threshold = random.uniform(25, 45)
            return [
                Condition(
                    left_operand=indicator_name, operator="<", right_operand=threshold
                )
            ]
        elif indicator.type == "STOCH":
            # STOCH: 売られすぎ領域でロング
            threshold = random.uniform(15, 25)
            return [
                Condition(
                    left_operand=indicator_name, operator="<", right_operand=threshold
                )
            ]
        elif indicator.type == "CCI":
            # CCI: 売られすぎ領域でロング
            threshold = random.uniform(-150, -80)
            return [
                Condition(
                    left_operand=indicator_name, operator="<", right_operand=threshold
                )
            ]
        elif indicator.type == "MACD":
            # MACD: ゼロライン上抜けでロング
            return [
                Condition(left_operand=indicator_name, operator=">", right_operand=0)
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

    def _generate_time_separation_strategy(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
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
                        indicator_list, key=lambda x: x.parameters.get("period", 14)
                    )
                    short_term = sorted_indicators[0]
                    long_term = sorted_indicators[-1]

                    # 短期・長期組み合わせ条件を生成
                    if indicator_type == "RSI":
                        # 短期RSI売られすぎ + 長期RSI上昇トレンド
                        short_name = f"{short_term.type}_{short_term.parameters.get('period', 7)}"
                        long_name = (
                            f"{long_term.type}_{long_term.parameters.get('period', 21)}"
                        )

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
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
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
                period = bb_indicator.parameters.get("period", 20)

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
                period = adx_indicator.parameters.get("period", 14)
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
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
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

            for indicator in indicators[:2]:  # 最大2つの指標を使用
                if not indicator.enabled:
                    continue

                if indicator.type in INDICATOR_CHARACTERISTICS:
                    # 各指標の特性に基づいて条件を追加
                    char = INDICATOR_CHARACTERISTICS[indicator.type]
                    indicator_type = char["type"]

                    long_conds = []

                    # インジケータタイプに応じて適切な条件生成メソッドを呼び出し（ロング条件のみ）
                    if indicator_type == IndicatorType.MOMENTUM:
                        long_conds = self._create_momentum_long_conditions(indicator)
                    elif indicator_type == IndicatorType.TREND:
                        long_conds = self._create_trend_long_conditions(indicator)

                    elif indicator_type == IndicatorType.STATISTICS:
                        long_conds = self._create_statistics_long_conditions(indicator)
                    elif indicator_type == IndicatorType.PATTERN_RECOGNITION:
                        long_conds = self._create_pattern_long_conditions(indicator)

                    if long_conds:
                        long_conditions.extend(long_conds)

                    # 基本的なショート条件を生成
                    if (
                        indicator_type == IndicatorType.MOMENTUM
                        and indicator.type == "RSI"
                    ):
                        # テスト互換性: 素名で参照
                        indicator_name = indicator.type
                        threshold = random.uniform(65, 85)
                        short_conditions.append(
                            Condition(
                                left_operand=indicator_name,
                                operator=">",
                                right_operand=threshold,
                            )
                        )

            # 条件が空の場合はフォールバック
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
