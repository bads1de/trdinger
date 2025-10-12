import logging
import random
from typing import List, Tuple, Union, Dict, Optional
from app.utils.error_handler import safe_operation
from ..constants import (
    IndicatorType,
)
from ..utils.indicator_characteristics import INDICATOR_CHARACTERISTICS
from app.services.indicators.config import indicator_registry
from app.services.backtest.backtest_service import BacktestService
from ..utils.yaml_utils import YamlIndicatorUtils
from ..core.condition_evolver import ConditionEvolver, YamlIndicatorUtils as CoreYamlIndicatorUtils

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
            "regime_thresholds": None,
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
        regime_thresholds: Optional[Dict] = None,
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
        if regime_thresholds is not None:
            self.context["regime_thresholds"] = regime_thresholds

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
        self.logger.debug(f"{ind.type}のロング条件を生成中")
        config = YamlIndicatorUtils.get_indicator_config_from_yaml(
            self.yaml_config, ind.type
        )
        self.logger.debug(f"{ind.type}の設定: {config}")
        if config:
            threshold = YamlIndicatorUtils.get_threshold_from_yaml(
                self.yaml_config, config, "long", self.context
            )
            if threshold is not None:
                self.logger.debug(f"{ind.type}に閾値{threshold}を使用")
                # DEBUG: 条件生成詳細ログ
                final_condition = Condition(
                    left_operand=ind.type, operator=">", right_operand=threshold
                )
                self.logger.debug(f"ロング条件生成: {ind.type} > {threshold}")
                return [final_condition]
        self.logger.warning(f"{ind.type}の閾値が見つからないため、フォールバックとして0を使用")
        final_condition = Condition(
            left_operand=ind.type, operator=">", right_operand=0
        )
        self.logger.debug(f"フォールバックロング条件生成: {ind.type} > 0")
        return [final_condition]

    def _generic_short_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """統合された汎用ショート条件生成"""
        self.logger.debug(f"{ind.type}のショート条件を生成中")
        config = YamlIndicatorUtils.get_indicator_config_from_yaml(
            self.yaml_config, ind.type
        )
        self.logger.debug(f"{ind.type}の設定: {config}")
        if config:
            threshold = YamlIndicatorUtils.get_threshold_from_yaml(
                self.yaml_config, config, "short", self.context
            )
            if threshold is not None:
                self.logger.debug(f"{ind.type}に閾値{threshold}を使用")
                return [
                    Condition(
                        left_operand=ind.type, operator="<", right_operand=threshold
                    )
                ]
        self.logger.warning(f"{ind.type}の閾値が見つからないため、フォールバックとして0を使用")
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
                    f"YAMLベースの{side}条件 ({indicator.type}): {indicator.type} {'>' if side == 'long' else '<'} {threshold}"
                )
                return [final_condition]

        # デフォルト
        final_condition = Condition(
            left_operand=indicator.type,
            operator=">" if side == "long" else "<",
            right_operand=0,
        )
        self.logger.debug(
            f"デフォルト{side}条件 ({indicator.type}): {indicator.type} {'>' if side == 'long' else '<'} 0"
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


class GAConditionGenerator(ConditionGenerator):
    """
    階層的GA統合を備えた拡張条件生成器

    ConditionGeneratorを継承し、ConditionEvolverとの連携により
    32指標全てに対応した階層的GA最適化を提供します。

    Attributes:
        use_hierarchical_ga: 階層的GAを有効にするかどうか
        condition_evolver: ConditionEvolverインスタンス
        backtest_service: バックテストサービス（依存関係注入用）
    """

    def __init__(
        self,
        enable_smart_generation: bool = True,
        use_hierarchical_ga: bool = True,
        backtest_service: Optional['BacktestService'] = None  # 型アノテーションのみ
    ):
        """
        拡張条件生成器の初期化

        Args:
            enable_smart_generation: スマート生成を有効にするか
            use_hierarchical_ga: 階層的GAを有効にするか（デフォルト: True）
            backtest_service: バックテストサービス（ConditionEvolver用）
        """
        # 親クラスの初期化
        super().__init__(enable_smart_generation)

        # 階層的GA設定
        self.use_hierarchical_ga = use_hierarchical_ga
        self.backtest_service = backtest_service
        self.condition_evolver: Optional[ConditionEvolver] = None

        # GA設定
        self.ga_config = {
            "population_size": 20,
            "generations": 10,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
        }

        # 初期化状態
        self._ga_initialized = False

        self.logger.info(
            f"GAConditionGenerator 初期化完了: "
            f"階層的GA={'有効' if use_hierarchical_ga else '無効'}"
        )

    def initialize_ga_components(self) -> bool:
        """
        GAコンポーネントの初期化

        Returns:
            初期化成功の場合True
        """
        if self._ga_initialized:
            return True

        try:
            if self.backtest_service is None:
                self.logger.warning("BacktestServiceが設定されていないため、GA機能は制限されます")
                return False

            # ConditionEvolver用のYamlIndicatorUtilsを作成（メタデータ使用）
            yaml_indicator_utils = CoreYamlIndicatorUtils()

            # ConditionEvolverインスタンスを作成
            self.condition_evolver = ConditionEvolver(
                backtest_service=self.backtest_service,
                yaml_indicator_utils=yaml_indicator_utils
            )

            self._ga_initialized = True
            self.logger.info("GAコンポーネント初期化完了")
            return True

        except Exception as e:
            self.logger.error(f"GAコンポーネント初期化エラー: {e}")
            return False

    @safe_operation(context="階層的GA条件生成", is_api_call=False)
    def generate_hierarchical_ga_conditions(
        self,
        indicators: List[IndicatorGene],
        backtest_config: Optional[Dict[str, any]] = None
    ) -> Tuple[List[Union[Condition, ConditionGroup]], List[Union[Condition, ConditionGroup]], List[Condition]]:
        """
        階層的GAによる最適化条件生成

        Args:
            indicators: 指標リスト
            backtest_config: バックテスト設定

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        if not self.use_hierarchical_ga:
            self.logger.info("階層的GAが無効のため、標準生成にフォールバック")
            return self.generate_balanced_conditions(indicators)

        if not self.initialize_ga_components():
            self.logger.warning("GAコンポーネント初期化失敗のため、標準生成にフォールバック")
            return self.generate_balanced_conditions(indicators)

        if self.condition_evolver is None:
            raise RuntimeError("ConditionEvolverが初期化されていません")

        try:
            self.logger.info(f"階層的GA条件生成開始: {len(indicators)}個の指標")

            # バックテスト設定の準備
            if backtest_config is None:
                backtest_config = {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1h",
                    "initial_balance": 10000,
                    "fee_rate": 0.001,
                }

            # 32指標全てに対応した並列処理
            optimized_conditions = []

            # 指標タイプ別に処理を分ける（並列化対応）
            indicator_types = self._dynamic_classify(indicators)

            for indicator_type, type_indicators in indicator_types.items():
                if not type_indicators:
                    continue

                self.logger.info(f"{indicator_type.name}タイプの指標を処理: {len(type_indicators)}個")

                # 各指標に対してGA最適化を実行
                for indicator in type_indicators:
                    try:
                        # ConditionEvolverで最適化
                        evolution_result = self.condition_evolver.run_evolution(
                            backtest_config=backtest_config,
                            population_size=self.ga_config["population_size"],
                            generations=self.ga_config["generations"]
                        )

                        if evolution_result and "best_condition" in evolution_result:
                            best_condition = evolution_result["best_condition"]
                            optimized_conditions.append(best_condition)
                            self.logger.info(f"指標 {indicator.type} の最適化完了: {best_condition}")
                        else:
                            self.logger.warning(f"指標 {indicator.type} の最適化に失敗")

                    except Exception as e:
                        self.logger.error(f"指標 {indicator.type} のGA最適化エラー: {e}")
                        # フォールバック: 標準条件生成
                        try:
                            fallback_conditions = self._create_type_based_conditions(indicator, "long")
                            optimized_conditions.extend(fallback_conditions)
                            self.logger.info(f"指標 {indicator.type} のフォールバック条件生成完了")
                        except Exception as fallback_error:
                            self.logger.error(f"指標 {indicator.type} のフォールバック処理も失敗: {fallback_error}")

            # 最適化された条件からロング・ショート条件を分離
            long_conditions = []
            short_conditions = []
            exit_conditions = []

            for condition in optimized_conditions:
                if hasattr(condition, 'direction'):
                    if condition.direction == "long":
                        long_conditions.append(condition)
                    elif condition.direction == "short":
                        short_conditions.append(condition)

            # 条件が生成できなかった場合は完全フォールバック
            if not long_conditions:
                self.logger.warning("ロング条件が生成されなかったため、標準生成にフォールバック")
                fallback_longs, fallback_shorts, fallback_exits = self.generate_balanced_conditions(indicators)
                long_conditions = fallback_longs
                short_conditions = fallback_shorts
                exit_conditions = fallback_exits

            # 条件数を制限
            if len(long_conditions) > 3:
                long_conditions = random.sample(long_conditions, 3)
            if len(short_conditions) > 3:
                short_conditions = random.sample(short_conditions, 3)

            self.logger.info(
                f"階層的GA条件生成完了: "
                f"ロング={len(long_conditions)}件, "
                f"ショート={len(short_conditions)}件"
            )

            return long_conditions, short_conditions, exit_conditions

        except Exception as e:
            self.logger.error(f"階層的GA条件生成エラー: {e}")
            # 完全フォールバック
            try:
                return self.generate_balanced_conditions(indicators)
            except Exception as fallback_error:
                self.logger.error(f"フォールバック処理も失敗: {fallback_error}")
                raise RuntimeError(f"条件生成に完全に失敗しました: {e}")

    def set_ga_config(
        self,
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        crossover_rate: Optional[float] = None,
        mutation_rate: Optional[float] = None,
    ):
        """
        GA設定を更新

        Args:
            population_size: 個体群サイズ
            generations: 世代数
            crossover_rate: 交叉率
            mutation_rate: 突然変異率
        """
        if population_size is not None:
            self.ga_config["population_size"] = population_size
        if generations is not None:
            self.ga_config["generations"] = generations
        if crossover_rate is not None:
            self.ga_config["crossover_rate"] = crossover_rate
        if mutation_rate is not None:
            self.ga_config["mutation_rate"] = mutation_rate

        self.logger.info(f"GA設定更新: {self.ga_config}")

    @safe_operation(context="GA最適化実行", is_api_call=False)
    def optimize_single_condition(
        self,
        indicator: IndicatorGene,
        direction: str,
        backtest_config: Dict[str, any]
    ) -> Optional[Condition]:
        """
        単一指標の条件をGAで最適化

        Args:
            indicator: 最適化対象の指標
            direction: 方向（long/short）
            backtest_config: バックテスト設定

        Returns:
            最適化された条件、失敗時はNone
        """
        if not self.initialize_ga_components() or self.condition_evolver is None:
            return None

        try:
            # 単一指標用の進化実行
            evolution_result = self.condition_evolver.run_evolution(
                backtest_config=backtest_config,
                population_size=self.ga_config["population_size"] // 2,  # 単一指標なので個体数を減らす
                generations=self.ga_config["generations"] // 2,
            )

            if evolution_result and "best_condition" in evolution_result:
                best_condition = evolution_result["best_condition"]

                # 指定された方向に一致するか確認
                if best_condition.direction == direction:
                    return best_condition

            return None

        except Exception as e:
            self.logger.error(f"単一条件最適化エラー ({indicator.type}, {direction}): {e}")
            return None
