import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from app.services.indicators.config import indicator_registry, IndicatorScaleType
from app.utils.error_handler import safe_operation

from ..config.constants import (
    IndicatorType,
)
from ..genes import Condition, ConditionGroup, IndicatorGene
from ..utils.indicator_utils import IndicatorCharacteristics
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

    @safe_operation(context="ConditionGenerator初期化", is_api_call=False)
    def __init__(
        self, enable_smart_generation: bool = True, ga_config: Optional[Any] = None
    ):
        """
        初期化（統合後）

        Args:
            enable_smart_generation: 新しいスマート生成を有効にするか
            ga_config: GAConfigオブジェクト (オプション)
        """
        self.enable_smart_generation = enable_smart_generation
        self.logger = logger
        self.ga_config_obj = ga_config  # GAConfigオブジェクトを保持

        # 設定を読み込み
        self.indicator_config = IndicatorCharacteristics.load_indicator_config()

        self.context = {
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
    ) -> List[Union[Condition, ConditionGroup]]:
        """
        条件の正規化/組立ヘルパー：
        - フォールバック（価格 vs トレンド or open）の注入
        - 1件なら素条件のまま、2件以上なら OR グルーピング
        """
        # トレンド系指標の優先順位
        trend_pref = ("SMA", "EMA")

        # フォールバック候補の抽出
        trend_pool = []
        for ind in indicators or []:
            if not getattr(ind, "enabled", True):
                continue
            cfg = indicator_registry.get_indicator_config(ind.type)
            if cfg and getattr(cfg, "category", None) == "trend":
                trend_pool.append(ind.type)

        # 優先候補の決定
        pref = [n for n in trend_pool if n in trend_pref]
        trend_name = (
            random.choice(pref)
            if pref
            else (
                random.choice(trend_pool) if trend_pool else random.choice(trend_pref)
            )
        )

        fallback = Condition(
            left_operand="close",
            operator=">" if side == "long" else "<",
            right_operand=trend_name or "open",
        )

        if not conds:
            return [fallback]

        # 平坦化（既に OR グループがある場合は中身だけ取り出す）
        flat: List[Condition] = []
        for c in conds:
            if isinstance(c, ConditionGroup):
                flat.extend(c.conditions)
            else:
                flat.append(c)

        # フォールバックの重複チェック
        exists = any(
            x.left_operand == fallback.left_operand
            and x.operator == fallback.operator
            and x.right_operand == fallback.right_operand
            for x in flat
        )

        if len(flat) == 1:
            return cast(
                List[Union[Condition, ConditionGroup]],
                flat if exists else flat + [fallback],
            )

        top_level: List[Union[Condition, ConditionGroup]] = [
            ConditionGroup(conditions=flat)
        ]
        # 存在していてもトップレベルに1本は追加して可視化と成立性の底上げを図る
        top_level.append(fallback)
        return top_level

    @safe_operation(context="バランス条件生成", is_api_call=False)
    def generate_balanced_conditions(self, indicators: List[IndicatorGene]) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """
        バランスの取れたロング・ショート条件を生成
        """
        if not self.enable_smart_generation or not indicators:
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
        max_conds = getattr(self.ga_config_obj, "max_conditions", 3)

        def _finalize(lst, side):
            if not lst:
                return self.normalize_conditions([], side, indicators)
            res = random.sample(lst, max_conds) if len(lst) > max_conds else lst
            return self.normalize_conditions(res, side, indicators)

        return _finalize(longs, "long"), _finalize(shorts, "short"), []

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

    def _get_indicator_name(self, indicator: IndicatorGene) -> str:
        """IndicatorCalculatorと一致する一意な指標名を取得"""
        if indicator.id:
            return f"{indicator.type}_{indicator.id[:8]}"
        return indicator.type

    def _get_band_names(self, indicator: IndicatorGene) -> Tuple[str, str]:
        """バンド指標のUpper/Lower名を取得"""
        base = self._get_indicator_name(indicator)
        cfg = indicator_registry.get_indicator_config(indicator.type)
        up_idx, low_idx = 2, 0  # デフォルト [lower, mid, upper]

        if cfg and cfg.return_cols:
            for i, col in enumerate(cfg.return_cols):
                c = col.lower()
                if any(k in c for k in ["upper", "top", "high"]):
                    up_idx = i
                if any(k in c for k in ["lower", "bottom", "low"]):
                    low_idx = i

        return f"{base}_{up_idx}", f"{base}_{low_idx}"

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

    @safe_operation(context="サイド別条件生成", is_api_call=False)
    def _create_side_conditions(
        self, indicator: IndicatorGene, side: str, name: Optional[str] = None
    ) -> List[Condition]:
        """統合されたサイド別条件生成ロジック"""
        target_name = name or indicator.type
        
        # オブジェクトではなく、ユーティリティ経由で辞書形式の設定を取得（後方互換性とget()メソッド確保のため）
        config_dict = IndicatorCharacteristics.get_indicator_config(
            self.indicator_config, indicator.type
        )
        # 内部処理用のクラスオブジェクトも取得
        config_obj = indicator_registry.get_indicator_config(indicator.type)

        threshold: Union[float, str] = 0.0
        
        # 設定から閾値を取得を試みる
        if config_dict:
            val = IndicatorCharacteristics.get_threshold_from_config(
                self.indicator_config, config_dict, side, self.context
            )
            if val is not None:
                threshold = val
            elif config_obj:
                # 設定から取得できない場合のスマートなデフォルト値
                scale_type = config_obj.scale_type
                if scale_type in (IndicatorScaleType.PRICE_RATIO, IndicatorScaleType.PRICE_ABSOLUTE):
                    # 価格スケールなら0ではなくCloseと比較
                    threshold = "close"
                elif scale_type == IndicatorScaleType.OSCILLATOR_0_100:
                    # 0-100オシレーターなら50を基準
                    threshold = 50.0
                elif scale_type == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
                    # ±100なら0でOK
                    threshold = 0.0
                elif scale_type == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
                    # 0中心なら0でOK
                    threshold = 0.0
                elif scale_type == IndicatorScaleType.PRICE_RATIO:
                    # 比率なら1.0
                    threshold = 1.0
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
        res = []
        i = 0
        while i < len(conditions):
            if i + 1 < len(conditions) and random.random() < 0.3:
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
        優先順位: YAML設定 > indicator_registry > Characteristics
        """
        raw_name = indicator.type if isinstance(indicator, IndicatorGene) else indicator
        # レジストリがエイリアス解決を自動で行う
        indicator_name = raw_name.upper()

        # 1. 設定をチェック
        config = IndicatorCharacteristics.get_indicator_config(
            self.indicator_config, indicator_name
        )
        if config and "type" in config:
            type_map = {
                "momentum": IndicatorType.MOMENTUM,
                "trend": IndicatorType.TREND,
                "volatility": IndicatorType.VOLATILITY,
                "volume": IndicatorType.MOMENTUM,
                "statistic": IndicatorType.VOLATILITY,  # 統計系はボラティリティとして扱う
                "candles": IndicatorType.TREND,        # ローソク足パターンはトレンド判断に含める
                "cycle": IndicatorType.MOMENTUM,       # サイクル系はモメンタムとして扱う
            }
            if config["type"] in type_map:
                return type_map[config["type"]]

        # 2. indicator_registryをチェック
        cfg = indicator_registry.get_indicator_config(indicator_name)
        if cfg and hasattr(cfg, "category") and getattr(cfg, "category", None):
            cat = str(getattr(cfg, "category", "")).lower()
            if any(k in cat for k in ["momentum", "oscillator", "technical", "custom", "cycle"]):
                return IndicatorType.MOMENTUM
            elif any(k in cat for k in ["trend", "overlap", "moving average", "candles"]):
                return IndicatorType.TREND
            elif any(k in cat for k in ["volatility", "statistics"]):
                return IndicatorType.VOLATILITY

        # 3. Characteristicsをチェック
        characteristics = IndicatorCharacteristics.get_characteristics()
        if indicator_name in characteristics:
            return characteristics[indicator_name]["type"]

        # デフォルト
        return IndicatorType.TREND

    @safe_operation(context="動的指標分類", is_api_call=False)
    def _dynamic_classify(
        self, indicators: List[IndicatorGene]
    ) -> Dict[IndicatorType, List[IndicatorGene]]:
        """動的指標分類（統合されたタイプ取得メソッドを使用）"""
        categorized = {
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
