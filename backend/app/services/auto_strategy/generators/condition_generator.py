import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from app.services.indicators.config import IndicatorScaleType, indicator_registry
from app.utils.error_handler import safe_operation

from ..config.constants import (
    IndicatorType,
)
from ..genes import Condition, ConditionGroup, IndicatorGene
from ..utils.indicator_references import build_indicator_reference_name
from .complex_conditions_strategy import ComplexConditionsStrategy
from .mtf_strategy import MTFStrategy

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

    # 定数
    OR_GROUP_PROBABILITY = 0.3
    OSCILLATOR_HIGH_THRESHOLD = 70.0
    OSCILLATOR_LOW_THRESHOLD = 30.0
    OSCILLATOR_MIDPOINT = 50.0
    ZERO_THRESHOLD = 0.0
    PRICE_RATIO_LONG_THRESHOLD = 1.01
    PRICE_RATIO_SHORT_THRESHOLD = 0.99
    EXIT_MOMENTUM_HIGH_THRESHOLD = 10.0
    EXIT_MOMENTUM_LOW_THRESHOLD = -10.0

    @safe_operation(context="ConditionGenerator初期化", is_api_call=False)
    def __init__(self, ga_config: Optional[Any] = None):
        """
        初期化（統合後）

        Args:
            ga_config: GAConfigオブジェクト (オプション)
        """
        self.logger = logger
        self.ga_config_obj = ga_config  # GAConfigオブジェクトを保持

        self.context: Dict[str, Any] = {
            "timeframe": None,
            "symbol": None,
            "threshold_profile": "normal",
        }

    @safe_operation(context="コンテキスト設定", is_api_call=False)
    def set_context(
        self,
        *,
        timeframe: str | None = None,
        symbol: str | None = None,
        threshold_profile: str | None = None,
        regime_thresholds: dict | None = None,
    ):
        """生成コンテキストを設定（RSI閾値などに利用）"""
        if timeframe is not None:
            self.context["timeframe"] = timeframe
        if symbol is not None:
            self.context["symbol"] = symbol
        if threshold_profile is not None:
            if threshold_profile not in ("aggressive", "normal", "conservative"):
                threshold_profile = "normal"
            self.context["threshold_profile"] = threshold_profile
        if regime_thresholds is not None:
            self.context["regime_thresholds"] = regime_thresholds

    @safe_operation(context="条件正規化", is_api_call=False)
    def normalize_conditions(
        self,
        conds: List[Union[Condition, ConditionGroup]],
        side: str,
        indicators: List[IndicatorGene],
        purpose: str = "entry",
    ) -> List[Union[Condition, ConditionGroup]]:
        """
        生成された生の条件リストを、戦略として実行可能な形式に正規化し、堅牢性を高めるフォールバックを注入します。

        主な処理ステップ：
        1. **トレンドフォールバックの生成**: 指標リストから移動平均（SMA/EMA）等のトレンド系指標を自動選定し、
           「価格がトレンドの上/下にあること」という基本的な条件（フォールバック）を作成します。
        2. **冗長性の排除**: 既に同一の条件が存在する場合、重複を避けます。
        3. **論理構造の最適化**:
           - 条件が0件の場合：フォールバックのみを返します。
           - 条件が複数ある場合：それらを `OR` グループとしてまとめ、成立性を高めます。
        4. **トップレベル統合**: 正規化された `OR` グループとフォールバック条件を組み合わせたリストを返却します。

        Args:
            conds (List[Union[Condition, ConditionGroup]]): 生成された元の条件リスト。
            side (str): "long" または "short"。
            indicators (List[IndicatorGene]): 戦略で使用可能な全指標のリスト（フォールバック選定用）。

        Returns:
            List[Union[Condition, ConditionGroup]]: 正規化・補強された論理条件のリスト。
        """
        fallback = self._build_fallback_condition(
            side,
            indicators,
            purpose=purpose,
        )

        if not conds:
            return [fallback]

        # 平坦化（既に OR グループがある場合は中身だけ取り出す）
        flat = []
        for c in conds:
            if isinstance(c, ConditionGroup) and c.operator == "OR":
                flat.extend(c.conditions)
            else:
                flat.append(c)

        # フォールバックの重複チェック（Condition型のみ対象）
        exists = any(
            isinstance(x, Condition)
            and x.left_operand == fallback.left_operand
            and x.operator == fallback.operator
            and x.right_operand == fallback.right_operand
            for x in flat
        )

        if len(flat) == 1 and isinstance(flat[0], Condition):
            return cast(
                List[Union[Condition, ConditionGroup]],
                flat if exists else flat + [fallback],
            )

        top_level: List[Union[Condition, ConditionGroup]] = [
            ConditionGroup(operator="OR", conditions=flat)
        ]
        # 存在していてもトップレベルに1本は追加して可視化と成立性の底上げを図る
        top_level.append(fallback)
        return top_level

    def _select_trend_indicator(
        self,
        indicators: List[IndicatorGene],
    ) -> Optional[IndicatorGene]:
        """フォールバックに使うトレンド系指標を選ぶ。"""
        trend_pref = ("SMA", "EMA")
        trend_categories = {"trend", "overlap", "custom"}
        trend_pool: List[IndicatorGene] = []
        for ind in indicators or []:
            if not getattr(ind, "enabled", True):
                continue
            cfg = indicator_registry.get_indicator_config(ind.type)
            if cfg and getattr(cfg, "category", None) in trend_categories:
                trend_pool.append(ind)

        pref = [ind for ind in trend_pool if ind.type in trend_pref]
        if pref:
            return random.choice(pref)
        if trend_pool:
            return random.choice(trend_pool)
        return None

    @staticmethod
    def _resolve_fallback_operator(side: str, purpose: str) -> str:
        """entry/exit の用途に応じてフォールバック演算子を決める。"""
        normalized_purpose = "exit" if purpose == "exit" else "entry"
        normalized_side = "short" if side == "short" else "long"

        if normalized_purpose == "exit":
            return "<" if normalized_side == "long" else ">"
        return ">" if normalized_side == "long" else "<"

    def _build_fallback_condition(
        self,
        side: str,
        indicators: List[IndicatorGene],
        *,
        purpose: str = "entry",
    ) -> Condition:
        """用途に応じたフォールバック条件を構築する。"""
        selected_trend_indicator = self._select_trend_indicator(indicators)
        trend_name = (
            self._get_indicator_name(selected_trend_indicator)
            if selected_trend_indicator
            else "open"
        )
        return Condition(
            left_operand="close",
            operator=self._resolve_fallback_operator(side, purpose),
            right_operand=trend_name,
        )

    @safe_operation(context="バランス条件生成", is_api_call=False)
    def generate_balanced_conditions(self, indicators: List[IndicatorGene]) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        与えられたテクニカル指標群から、統計的にバランスの取れたロング・ショート条件を「スマートに」生成します。

        このメソッドは単純なランダム生成ではなく、以下の高度な戦略を組み合わせて「意味のある」条件を構築します：
        1. **複合条件戦略 (`ComplexConditionsStrategy`)**: RSIの過熱感とMACDのクロス等、異なる性質の指標を組み合わせた論理式を生成。
        2. **マルチタイムフレーム戦略 (`MTFStrategy`)**: 上位足（1h足での日足トレンド判定等）を条件に含め、相場環境に沿ったエントリーを可能にします。
        3. **スケールマッチング**: 価格 vs 移動平均（価格スケール）、RSI vs 70（0-100スケール）等、比較対象のスケールが一致するようにオペランドを選定。
        4. **自動正規化**: 生成された条件を `normalize_conditions` に通し、論理的な成立性とフォールバックを保証。

        Args:
            indicators (List[IndicatorGene]): 戦略の構成要素となる指標遺伝子のリスト。

        Returns:
            Tuple[List, List, List]: (ロング条件リスト, ショート条件リスト, 共通/メタ条件リスト) のタプル。
        """
        if not indicators:
            return self.generate_fallback_conditions(indicators)

        # 統合された戦略パターン（Complex & MTF）
        longs, shorts = [], []

        # 1. 複雑な組み合わせ戦略
        try:
            strategy = ComplexConditionsStrategy(self)
            complex_longs, complex_shorts, _ = strategy.generate_conditions(indicators)
            longs.extend(complex_longs)
            shorts.extend(complex_shorts)
        except Exception as e:
            logger.warning(f"ComplexStrategy生成失敗: {e}")

        # 2. MTF戦略
        try:
            mtf_strategy = MTFStrategy(self)
            mtf_longs, mtf_shorts, _ = mtf_strategy.generate_conditions(indicators)
            longs.extend(mtf_longs)
            shorts.extend(mtf_shorts)
        except Exception as e:
            logger.debug(f"MTFStrategy生成スキップ: {e}")

        # 3. 制限とフォールバック
        return self._finalize_conditions(longs, shorts, indicators, purpose="entry")

    def _finalize_conditions(
        self,
        longs: List[Union[Condition, ConditionGroup]],
        shorts: List[Union[Condition, ConditionGroup]],
        indicators: List[IndicatorGene],
        *,
        purpose: str = "entry",
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        条件リストを最終化する（共通メソッド）

        Args:
            longs: ロング条件リスト
            shorts: ショート条件リスト
            indicators: 指標リスト
            purpose: "entry" または "exit"

        Returns:
            (正規化されたロング条件, 正規化されたショート条件, 空リスト)
        """
        max_conds = getattr(self.ga_config_obj, "max_conditions", 3)

        def _finalize(lst, side):
            if not lst:
                return self.normalize_conditions([], side, indicators, purpose=purpose)
            res = random.sample(lst, max_conds) if len(lst) > max_conds else lst
            return self.normalize_conditions(res, side, indicators, purpose=purpose)

        return _finalize(longs, "long"), _finalize(shorts, "short"), []

    @safe_operation(context="イグジット条件生成", is_api_call=False)
    def generate_exit_conditions(self, indicators: List[IndicatorGene]) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        保有ポジションの解消を目的にした exit 条件を生成する。

        entry 条件の反転コピーではなく、
        1. トレンド破綻
        2. 逆行クロス
        3. 利確寄りのバンド到達 / オシレーター過熱
        を候補として構築する。
        """
        if not indicators:
            return self.generate_fallback_exit_conditions(indicators)

        longs: List[Union[Condition, ConditionGroup]] = []
        shorts: List[Union[Condition, ConditionGroup]] = []

        trend_longs, trend_shorts = self._create_trend_reversal_exit_conditions(
            indicators
        )
        longs.extend(trend_longs)
        shorts.extend(trend_shorts)

        cross_longs, cross_shorts = self._create_cross_exit_conditions(indicators)
        longs.extend(cross_longs)
        shorts.extend(cross_shorts)

        tp_longs, tp_shorts = self._create_take_profit_exit_conditions(indicators)
        longs.extend(tp_longs)
        shorts.extend(tp_shorts)

        return self._finalize_conditions(longs, shorts, indicators, purpose="exit")

    def generate_fallback_exit_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List, List, List]:
        """exit 条件が作れない場合のフォールバック。"""
        return (
            self.normalize_conditions([], "long", indicators, purpose="exit"),
            self.normalize_conditions([], "short", indicators, purpose="exit"),
            [],
        )

    def generate_fallback_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List, List, List]:
        """従来の生成ロジック（フォールバック）"""
        longs, shorts = [], []
        for ind in indicators:
            if not ind.enabled:
                continue
            name = self._get_indicator_name(ind)
            longs.extend(self._create_side_conditions(ind, "long", name))
            shorts.extend(self._create_side_conditions(ind, "short", name))
        return longs, shorts, []

    def _create_trend_reversal_exit_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]], List[Union[Condition, ConditionGroup]]
    ]:
        """トレンド破綻やモメンタム反転を exit 条件として生成する。"""
        longs: List[Union[Condition, ConditionGroup]] = []
        shorts: List[Union[Condition, ConditionGroup]] = []

        classified = self._classify_indicators(indicators)
        trend_candidates = classified[IndicatorType.TREND]
        momentum_candidates = classified[IndicatorType.MOMENTUM]

        if trend_candidates:
            trend = random.choice(trend_candidates)
            trend_name = self._get_indicator_name(trend)
            longs.append(
                Condition(left_operand="close", operator="<", right_operand=trend_name)
            )
            shorts.append(
                Condition(left_operand="close", operator=">", right_operand=trend_name)
            )

        if momentum_candidates:
            momentum = random.choice(momentum_candidates)
            cfg = indicator_registry.get_indicator_config(momentum.type)
            scale_type = cfg.scale_type if cfg else None
            momentum_name = self._get_indicator_name(momentum)

            if scale_type == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
                longs.append(
                    Condition(
                        left_operand=momentum_name, operator="<", right_operand=0.0
                    )
                )
                shorts.append(
                    Condition(
                        left_operand=momentum_name, operator=">", right_operand=0.0
                    )
                )
            elif scale_type == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
                longs.append(
                    Condition(
                        left_operand=momentum_name, operator="<", right_operand=self.EXIT_MOMENTUM_LOW_THRESHOLD
                    )
                )
                shorts.append(
                    Condition(
                        left_operand=momentum_name, operator=">", right_operand=self.EXIT_MOMENTUM_HIGH_THRESHOLD
                    )
                )

        return longs, shorts

    def _create_cross_exit_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]], List[Union[Condition, ConditionGroup]]
    ]:
        """移動平均の逆行クロスを exit 条件として生成する。"""
        longs: List[Union[Condition, ConditionGroup]] = []
        shorts: List[Union[Condition, ConditionGroup]] = []

        price_inds = [
            indicator
            for indicator in indicators
            if indicator.enabled and self._is_price_scale(indicator)
        ]
        if len(price_inds) < 2:
            return longs, shorts

        i1, i2 = random.sample(price_inds, 2)
        p1, p2 = i1.parameters.get("period", 0), i2.parameters.get("period", 0)
        if abs(p1 - p2) < 1:
            return longs, shorts

        fast_ma, slow_ma = (i1, i2) if p1 < p2 else (i2, i1)
        fast_name = self._get_indicator_name(fast_ma)
        slow_name = self._get_indicator_name(slow_ma)

        longs.append(
            Condition(left_operand=fast_name, operator="<", right_operand=slow_name)
        )
        shorts.append(
            Condition(left_operand=fast_name, operator=">", right_operand=slow_name)
        )
        return longs, shorts

    def _create_take_profit_exit_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]], List[Union[Condition, ConditionGroup]]
    ]:
        """利確寄りの到達条件を exit 候補として生成する。"""
        longs: List[Union[Condition, ConditionGroup]] = []
        shorts: List[Union[Condition, ConditionGroup]] = []

        band_candidates = [
            indicator
            for indicator in indicators
            if indicator.enabled and self._is_band_indicator(indicator)
        ]
        if band_candidates:
            band = random.choice(band_candidates)
            upper_name, lower_name = self._get_band_names(band)
            longs.append(
                Condition(left_operand="close", operator=">", right_operand=upper_name)
            )
            shorts.append(
                Condition(left_operand="close", operator="<", right_operand=lower_name)
            )

        for indicator in indicators:
            if not indicator.enabled:
                continue

            cfg = indicator_registry.get_indicator_config(indicator.type)
            scale_type = cfg.scale_type if cfg else None
            indicator_name = self._get_indicator_name(indicator)

            if scale_type == IndicatorScaleType.OSCILLATOR_0_100:
                longs.append(
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=self.OSCILLATOR_HIGH_THRESHOLD
                    )
                )
                shorts.append(
                    Condition(
                        left_operand=indicator_name, operator="<", right_operand=self.OSCILLATOR_LOW_THRESHOLD
                    )
                )

        return longs, shorts

    def _get_indicator_name(self, indicator: IndicatorGene) -> str:
        """IndicatorCalculatorと一致する一意な指標名を取得"""
        return build_indicator_reference_name(indicator)

    def _get_band_names(self, indicator: IndicatorGene) -> Tuple[str, str]:
        """バンド指標のUpper/Lower名を取得"""
        cfg = indicator_registry.get_indicator_config(indicator.type)
        up_idx, low_idx = 2, 0  # デフォルト [lower, mid, upper]

        if cfg and cfg.return_cols:
            for i, col in enumerate(cfg.return_cols):
                c = col.lower()
                if any(k in c for k in ["upper", "top", "high"]):
                    up_idx = i
                if any(k in c for k in ["lower", "bottom", "low"]):
                    low_idx = i

        return (
            build_indicator_reference_name(indicator, up_idx),
            build_indicator_reference_name(indicator, low_idx),
        )

    def _is_price_scale(self, indicator: IndicatorGene) -> bool:
        """価格スケールの指標かどうか"""
        cfg = indicator_registry.get_indicator_config(indicator.type)
        if cfg:
            return cfg.scale_type == IndicatorScaleType.PRICE_RATIO
        return indicator.type in ["SMA", "EMA", "WMA", "HMA", "KAMA", "TRIMA", "VWAP"]

    def _is_band_indicator(self, indicator: IndicatorGene) -> bool:
        """バンド系指標（Upper/Lowerを持つ）かどうか"""
        cfg = indicator_registry.get_indicator_config(indicator.type)
        if cfg and cfg.return_cols and len(cfg.return_cols) >= 2:
            cols = [c.lower() for c in cfg.return_cols]
            has_up = any(k in c for c in cols for k in ["upper", "top", "high"])
            has_low = any(k in c for c in cols for k in ["lower", "bottom", "low"])
            if has_up and has_low:
                return True
        return any(
            k in indicator.type.upper() for k in ["BB", "BAND", "KELTNER", "DONCHIAN"]
        )

    def _create_side_condition(
        self, indicator: IndicatorGene, side: str, name: Optional[str] = None
    ) -> Condition:
        """単一のサイド別条件を生成（ヘルパー）"""
        conds = self._create_side_conditions(indicator, side)
        res = conds[0]
        if name:
            res.left_operand = name
        return res

    def _build_and_groups(
        self,
        long_conditions: List[Union[Condition, ConditionGroup]],
        short_conditions: List[Union[Condition, ConditionGroup]],
    ) -> Tuple[ConditionGroup, ConditionGroup]:
        """ロング/ショートの条件リストを同じ構造の AND グループにする。"""
        return (
            ConditionGroup(operator="AND", conditions=long_conditions),
            ConditionGroup(operator="AND", conditions=short_conditions),
        )

    @safe_operation(context="サイド別条件生成", is_api_call=False)
    def _create_side_conditions(
        self, indicator: IndicatorGene, side: str, name: Optional[str] = None
    ) -> List[Condition]:
        """統合されたサイド別条件生成ロジック"""
        target_name = name or indicator.type

        # 直接レジストリから設定オブジェクトを取得
        config_obj = indicator_registry.get_indicator_config(indicator.type)
        threshold: Union[float, str] = 0.0

        if config_obj:
            # 設定から閾値を取得を試みる
            profile = self.context.get("threshold_profile", "normal")
            thresholds = config_obj.thresholds or {}

            val = None
            if profile in thresholds and thresholds[profile]:
                profile_config = thresholds[profile]
                if side == "long":
                    if "long_gt" in profile_config:
                        val = profile_config["long_gt"]
                    elif "long_lt" in profile_config:
                        val = profile_config["long_lt"]
                elif side == "short":
                    if "short_lt" in profile_config:
                        val = profile_config["short_lt"]
                    elif "short_gt" in profile_config:
                        val = profile_config["short_gt"]

            if val is not None:
                threshold = val
            else:
                # 設定から取得できない場合のスマートなデフォルト値
                scale_type = config_obj.scale_type
                if scale_type in (
                    IndicatorScaleType.PRICE_RATIO,
                    IndicatorScaleType.PRICE_ABSOLUTE,
                ):
                    # 価格スケールなら0ではなくCloseと比較
                    threshold = "close"
                elif scale_type == IndicatorScaleType.OSCILLATOR_0_100:
                    # 0-100オシレーターなら50を基準
                    threshold = self.OSCILLATOR_MIDPOINT
                elif scale_type == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
                    # ±100なら0でOK
                    threshold = self.ZERO_THRESHOLD
                elif scale_type == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
                    # 0中心なら0でOK
                    threshold = self.ZERO_THRESHOLD
                elif scale_type in (
                    IndicatorScaleType.FUNDING_RATE,
                    IndicatorScaleType.OPEN_INTEREST,
                ):
                    # OI/FR 派生はゼロ基準の閾値を使う
                    threshold = self.ZERO_THRESHOLD
                elif scale_type == IndicatorScaleType.VOLUME:
                    # 出来高
                    threshold = "SMA"

        return [
            Condition(
                left_operand=target_name,
                operator=">" if side == "long" else "<",
                right_operand=threshold,
            )
        ]

    def _structure_conditions(
        self, conditions: List[Union[Condition, ConditionGroup]]
    ) -> List[Union[Condition, ConditionGroup]]:
        """条件リストを確率的に階層化"""
        if len(conditions) < 2:
            return conditions
        res: List[Union[Condition, ConditionGroup]] = []
        i = 0
        while i < len(conditions):
            if i + 1 < len(conditions) and random.random() < self.OR_GROUP_PROBABILITY:
                res.append(
                    ConditionGroup(
                        operator="OR", conditions=[conditions[i], conditions[i + 1]]
                    )
                )
                i += 2
            else:
                res.append(conditions[i])
                i += 1
        return res

    @safe_operation(context="指標タイプ取得", is_api_call=False)
    def _get_indicator_type(
        self, indicator: Union[IndicatorGene, str]
    ) -> IndicatorType:
        """
        指標のタイプを取得（統合版）
        優先順位: indicator_registry
        """
        raw_name = indicator.type if isinstance(indicator, IndicatorGene) else indicator
        # レジストリがエイリアス解決を自動で行う
        indicator_name = raw_name.upper()

        # indicator_registryをチェック
        cfg = indicator_registry.get_indicator_config(indicator_name)

        if cfg:
            # 設定内のカテゴリー情報を使用
            if getattr(cfg, "category", None):
                cat = str(getattr(cfg, "category", "")).lower()

                # 部分一致による推定
                if any(
                    k in cat
                    for k in ["momentum", "oscillator", "technical", "custom", "cycle"]
                ):
                    return IndicatorType.MOMENTUM
                elif any(
                    k in cat for k in ["trend", "overlap", "moving average", "candles"]
                ):
                    return IndicatorType.TREND
                elif any(k in cat for k in ["volatility", "statistics"]):
                    return IndicatorType.VOLATILITY

        # デフォルト
        return IndicatorType.TREND

    @safe_operation(context="動的指標分類", is_api_call=False)
    def _dynamic_classify(
        self, indicators: List[IndicatorGene]
    ) -> Dict[IndicatorType, List[IndicatorGene]]:
        """動的指標分類（統合されたタイプ取得メソッドを使用）"""
        categorized: Dict[IndicatorType, List[IndicatorGene]] = {
            IndicatorType.MOMENTUM: [],
            IndicatorType.TREND: [],
            IndicatorType.VOLATILITY: [],
        }

        for ind in indicators:
            if not ind.enabled:
                continue

            try:
                ind_type = self._get_indicator_type(ind)
                categorized[ind_type].append(ind)
            except Exception as e:
                logger.error(f"指標 {ind.type} の分類中にエラー: {e}")
                categorized[IndicatorType.TREND].append(ind)

        return categorized

    def _classify_indicators(
        self, indicators: List[IndicatorGene]
    ) -> Dict[IndicatorType, List[IndicatorGene]]:
        """_dynamic_classify のエイリアス（ComplexConditionsStrategy互換用）"""
        return self._dynamic_classify(indicators)
