import logging
import random
from typing import List, Tuple, Union
from app.services.auto_strategy.config.constants import (
    INDICATOR_CHARACTERISTICS,
    IndicatorType,
    StrategyType,
)

from ..models.strategy_models import Condition, IndicatorGene, ConditionGroup
from .condition_generator import ConditionGenerator  # Phase 1.3: 条件生成統合


logger = logging.getLogger(__name__)


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

        # Phase 1.3: ConditionGenerator統合
        # 条件生成ロジックを一元化
        self.condition_generator = ConditionGenerator()
        self.condition_generator.indicators = None  # 後で設定

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

            # Phase 1.3: ConditionGeneratorのコンテキストも同期
            # 統合された条件生成に反映
            self.condition_generator.set_context(
                timeframe=timeframe,
                symbol=symbol,
                regime_gating=regime_gating,
                threshold_profile=threshold_profile,
            )
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

    # 削除されたメソッド: _dynamic_classify は ConditionGenerator._dynamic_classify を利用するように変更済み

    def _generic_long_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """Phase 1.3: 統合された汎用ロング条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._generic_long_conditions(ind)

    def _generic_short_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """Phase 1.3: 統合された汎用ショート条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._generic_short_conditions(ind)

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
            # Phase 1.3: ConditionGeneratorを活用（統合された動的分類）
            # 指標をタイプ別に分類（未特性化はレジストリから動的分類）
            indicators_by_type = self.condition_generator._dynamic_classify(indicators)

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
        """Phase 1.3: 統合されたトレンド系ロング条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._create_trend_long_conditions(indicator)

    def _create_trend_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Phase 1.3: 統合されたトレンド系ショート条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._create_trend_short_conditions(indicator)

    def _create_momentum_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Phase 1.3: 統合されたモメンタム系ロング条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._create_momentum_long_conditions(indicator)

    def _create_momentum_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Phase 1.3: 統合されたモメンタム系ショート条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._create_momentum_short_conditions(indicator)

    def _create_statistics_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Phase 1.3: 統合された統計系ロング条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._create_statistics_long_conditions(indicator)

    def _create_statistics_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Phase 1.3: 統合された統計系ショート条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._create_statistics_short_conditions(indicator)

    def _create_pattern_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Phase 1.3: 統合されたパターン認識系ロング条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._create_pattern_long_conditions(indicator)

    def _create_pattern_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Phase 1.3: 統合されたパターン認識系ショート条件生成 - ConditionGenerator委譲"""
        return self.condition_generator._create_pattern_short_conditions(indicator)

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
                        short_conds = self._create_statistics_short_conditions(
                            indicator
                        )
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
