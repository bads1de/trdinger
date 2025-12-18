import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from app.services.backtest.backtest_service import BacktestService
from app.services.indicators.config import indicator_registry, IndicatorScaleType
from app.utils.error_handler import safe_operation

from ..config.constants import (
    IndicatorType,
    OPERATORS,
)
from ..core.condition_evolver import (
    ConditionEvolver,
)
from ..core.condition_evolver import YamlIndicatorUtils as CoreYamlIndicatorUtils
from ..core.operand_grouping import operand_grouping_system
from ..genes import Condition, ConditionGroup, IndicatorGene
from ..utils.indicator_utils import get_all_indicators
from ..utils.yaml_utils import YamlIndicatorUtils
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
    5. 純粋ランダム条件生成（RandomConditionGeneratorより統合）
    6. オペランド生成（OperandGeneratorより統合）
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

        # YAML設定を読み込み
        self.yaml_config = YamlIndicatorUtils.load_yaml_config_for_indicators()

        self.context = {
            "timeframe": None,
            "symbol": None,
            "threshold_profile": "normal",
        }

        # geneに含まれる指標一覧をオプションで保持
        self.indicators: List[IndicatorGene] | None = None

        # ランダム生成用設定（OperandGeneratorより統合）
        self.available_operators = OPERATORS
        self.price_data_weight = getattr(ga_config, "price_data_weight", 5) if ga_config else 5
        self.volume_data_weight = getattr(ga_config, "volume_data_weight", 2) if ga_config else 2
        self.oi_fr_data_weight = getattr(ga_config, "oi_fr_data_weight", 1) if ga_config else 1
        self._valid_indicator_names = self._initialize_valid_indicators()

    def _initialize_valid_indicators(self) -> set:
        """有効な指標名を初期化"""
        try:
            return set(get_all_indicators())
        except Exception:
            return set()

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

    def generate_random_conditions(
        self, indicators: List[any], condition_type: str
    ) -> List[Condition]:
        """ランダムな条件リストを生成"""
        # 条件数はプロファイルや生成器の方針により 1〜max_conditions に広げる
        # ここでは min_conditions〜max_conditions の範囲で選択（下限>上限にならないようにガード）
        low = 3  # デフォルト値
        high = 5  # デフォルト値

        if self.ga_config_obj:
            if hasattr(self.ga_config_obj, "min_conditions"):
                low = int(self.ga_config_obj.min_conditions)
            if hasattr(self.ga_config_obj, "max_conditions"):
                high = int(self.ga_config_obj.max_conditions)

        if high < low:
            low, high = high, low
        num_conditions = random.randint(low, max(low, high))
        conditions = []

        for _ in range(num_conditions):
            condition = self._generate_single_condition(indicators, condition_type)
            if condition:
                conditions.append(condition)

        # 最低1つの条件は保証
        if not conditions:
            conditions.append(self._generate_fallback_condition(condition_type))

        return conditions

    def _generate_single_condition(
        self, indicators: List[any], condition_type: str
    ) -> Condition:
        """単一の条件を生成"""
        # 左オペランドの選択
        left_operand = self.choose_operand(indicators)

        # 演算子の選択
        operator = random.choice(self.available_operators)

        # 右オペランドの選択
        right_operand = self.choose_right_operand(
            left_operand, indicators, condition_type
        )

        return Condition(
            left_operand=left_operand, operator=operator, right_operand=right_operand
        )

    def choose_operand(self, indicators: List[any]) -> str:
        """オペランドを選択（指標名またはデータソース）"""
        choices = []

        # テクニカル指標名を追加
        for indicator_gene in indicators:
            indicator_type = indicator_gene.type
            if (
                self._valid_indicator_names
                and indicator_type in self._valid_indicator_names
            ):
                choices.append(indicator_type)

        # 基本データソース
        basic_sources = ["close", "open", "high", "low"]
        choices.extend(basic_sources * self.price_data_weight)
        choices.extend(["volume"] * self.volume_data_weight)
        choices.extend(["OpenInterest", "FundingRate"] * self.oi_fr_data_weight)

        return random.choice(choices) if choices else "close"

    def choose_right_operand(
        self, left_operand: str, indicators: List[any], condition_type: str
    ):
        """右オペランドを選択（指標名、データソース、または数値）"""
        if self.ga_config_obj and random.random() < getattr(self.ga_config_obj, "numeric_threshold_probability", 0.5):
            return self.generate_threshold_value(left_operand, condition_type)

        compatible_operand = self.choose_compatible_operand(left_operand, indicators)

        if compatible_operand != left_operand:
            compatibility = operand_grouping_system.get_compatibility_score(
                left_operand, compatible_operand
            )
            min_score = getattr(self.ga_config_obj, "min_compatibility_score", 0.1) if self.ga_config_obj else 0.1
            if compatibility < min_score:
                return self.generate_threshold_value(left_operand, condition_type)

        return compatible_operand

    def choose_compatible_operand(
        self, left_operand: str, indicators: List[any]
    ) -> str:
        """左オペランドと互換性の高い右オペランドを選択"""
        available_operands = []
        for indicator_gene in indicators:
            available_operands.append(indicator_gene.type)

        available_operands.extend(["close", "open", "high", "low", "volume", "OpenInterest", "FundingRate"])

        strict_score = getattr(self.ga_config_obj, "strict_compatibility_score", 0.8) if self.ga_config_obj else 0.8
        strict_compatible = operand_grouping_system.get_compatible_operands(
            left_operand,
            available_operands,
            min_compatibility=strict_score,
        )

        if strict_compatible:
            return random.choice(strict_compatible)

        min_score = getattr(self.ga_config_obj, "min_compatibility_score", 0.1) if self.ga_config_obj else 0.1
        high_compatible = operand_grouping_system.get_compatible_operands(
            left_operand,
            available_operands,
            min_compatibility=min_score,
        )

        if high_compatible:
            return random.choice(high_compatible)

        fallback_operands = [op for op in available_operands if op != left_operand]
        return random.choice(fallback_operands) if fallback_operands else "close"

    def generate_threshold_value(self, operand: str, condition_type: str) -> float:
        """オペランドの型に応じて、データ駆動で閾値を生成"""
        if "FundingRate" in operand:
            return self._get_safe_threshold("funding_rate", [0.0001, 0.001], allow_choice=True)
        if "OpenInterest" in operand:
            return self._get_safe_threshold("open_interest", [1000000, 50000000], allow_choice=True)
        if operand == "volume":
            return self._get_safe_threshold("volume", [1000, 100000])

        indicator_config = indicator_registry.get_indicator_config(operand)
        if indicator_config and indicator_config.scale_type:
            scale_type = indicator_config.scale_type
            if scale_type == IndicatorScaleType.OSCILLATOR_0_100:
                return self._get_safe_threshold("oscillator_0_100", [20, 80])
            if scale_type == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
                return self._get_safe_threshold("oscillator_plus_minus_100", [-100, 100])
            if scale_type == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
                return self._get_safe_threshold("momentum_zero_centered", [-0.5, 0.5])
            if scale_type in (IndicatorScaleType.PRICE_RATIO, IndicatorScaleType.PRICE_ABSOLUTE):
                return self._get_safe_threshold("price_ratio", [0.95, 1.05])
            if scale_type == IndicatorScaleType.VOLUME:
                return self._get_safe_threshold("volume", [1000, 100000])

        return self._get_safe_threshold("price_ratio", [0.95, 1.05])

    def _get_safe_threshold(
        self, key: str, default_range: List[float], allow_choice: bool = False
    ) -> float:
        """設定から値を取得し、安全に閾値を生成する"""
        config_ranges = getattr(self.ga_config_obj, "threshold_ranges", {}) if self.ga_config_obj else {}
        range_ = config_ranges.get(key, default_range)

        if isinstance(range_, list):
            if allow_choice and len(range_) > 2:
                try:
                    return float(random.choice(range_))
                except (ValueError, TypeError):
                    pass
            if len(range_) >= 2 and isinstance(range_[0], (int, float)) and isinstance(range_[1], (int, float)):
                return random.uniform(range_[0], range_[1])
        return random.uniform(default_range[0], default_range[1])

    def _generate_fallback_condition(self, condition_type: str) -> Condition:
        """フォールバック用の基本条件を生成（JSON形式の指標名）"""
        if condition_type == "entry":
            return Condition(left_operand="close", operator=">", right_operand="SMA")
        else:
            return Condition(left_operand="close", operator="<", right_operand="SMA")

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

        # MTF戦略を追加（プロフェッショナルなトレンドフォローロジック）
        try:
            mtf_strategy = MTFStrategy(self)
            mtf_longs, mtf_shorts, _ = mtf_strategy.generate_conditions(indicators)
            longs.extend(mtf_longs)
            shorts.extend(mtf_shorts)
            self.logger.debug(
                f"MTF条件追加: Long={len(mtf_longs)}, Short={len(mtf_shorts)}"
            )
        except Exception as e:
            self.logger.warning(f"MTF戦略生成失敗（スキップします）: {e}")

        # 条件数制限を取得（configからまたはデフォルト）
        max_conditions = 3
        if self.ga_config_obj and hasattr(self.ga_config_obj, "max_conditions"):
            max_conditions = self.ga_config_obj.max_conditions

        # 条件数を制限
        if len(longs) > max_conditions:
            longs = random.sample(longs, max_conditions)
        if len(shorts) > max_conditions:
            shorts = random.sample(shorts, max_conditions)

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

    @safe_operation(context="サイド別条件生成", is_api_call=False)
    def _create_side_conditions(self, indicator: IndicatorGene, side: str) -> List[Condition]:
        """統合されたサイド別条件生成ロジック"""
        self.logger.debug(f"{indicator.type}の{side}条件を生成中")
        
        # 1. YAML設定から閾値を取得
        config = YamlIndicatorUtils.get_indicator_config_from_yaml(
            self.yaml_config, indicator.type
        )
        if config:
            threshold = YamlIndicatorUtils.get_threshold_from_yaml(
                self.yaml_config, config, side, self.context
            )
            if threshold is not None:
                return [Condition(
                    left_operand=indicator.type,
                    operator=">" if side == "long" else "<",
                    right_operand=threshold
                )]

        # 2. フォールバック
        self.logger.warning(f"{indicator.type}の閾値が見つからないため、デフォルト値を使用")
        return [Condition(
            left_operand=indicator.type,
            operator=">" if side == "long" else "<",
            right_operand=0
        )]

    def _generic_long_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """旧互換用: ロング条件生成"""
        return self._create_side_conditions(ind, "long")

    def _generic_short_conditions(self, ind: IndicatorGene) -> List[Condition]:
        """旧互換用: ショート条件生成"""
        return self._create_side_conditions(ind, "short")

    def _create_type_based_conditions(self, indicator: IndicatorGene, side: str) -> List[Condition]:
        """旧互換用: 型別条件生成"""
        return self._create_side_conditions(indicator, side)

    def _create_momentum_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """旧互換用: モメンタム系ショート条件生成"""
        return self._create_side_conditions(indicator, "short")

    @safe_operation(context="指標タイプ取得", is_api_call=False)
    def _get_indicator_type(self, indicator: Union[IndicatorGene, str]) -> IndicatorType:
        """
        指標のタイプを取得（統合版）
        優先順位: YAML設定 > indicator_registry > Characteristics
        """
        indicator_name = indicator.type if isinstance(indicator, IndicatorGene) else indicator
        
        # 1. YAML設定をチェック
        config = YamlIndicatorUtils.get_indicator_config_from_yaml(
            self.yaml_config, indicator_name
        )
        if config and "type" in config:
            type_map = {
                "momentum": IndicatorType.MOMENTUM,
                "trend": IndicatorType.TREND,
                "volatility": IndicatorType.VOLATILITY
            }
            if config["type"] in type_map:
                return type_map[config["type"]]

        # 2. indicator_registryをチェック
        cfg = indicator_registry.get_indicator_config(indicator_name)
        if cfg and hasattr(cfg, "category") and getattr(cfg, "category", None):
            cat = getattr(cfg, "category")
            if cat == "momentum":
                return IndicatorType.MOMENTUM
            elif cat in ("trend", "volatility"):
                return IndicatorType.TREND if cat == "trend" else IndicatorType.VOLATILITY

        # 3. Characteristicsをチェック
        characteristics = YamlIndicatorUtils.get_characteristics()
        if indicator_name in characteristics:
            return characteristics[indicator_name]["type"]

        # デフォルト
        logger.warning(f"指標 {indicator_name} のタイプを特定できませんでした。トレンド系として扱います。")
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
        backtest_service: Optional["BacktestService"] = None,  # 型アノテーションのみ
        ga_config: Optional[Any] = None,
    ):
        """
        拡張条件生成器の初期化

        Args:
            enable_smart_generation: スマート生成を有効にするか
            use_hierarchical_ga: 階層的GAを有効にするか（デフォルト: True）
            backtest_service: バックテストサービス（ConditionEvolver用）
            ga_config: GAConfigオブジェクト
        """
        # 親クラスの初期化
        super().__init__(enable_smart_generation, ga_config)

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

        if self.ga_config_obj and hasattr(self.ga_config_obj, "hierarchical_ga_config"):
            # Configのサブ設定で上書き
            self.ga_config.update(self.ga_config_obj.hierarchical_ga_config)

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
                self.logger.warning(
                    "BacktestServiceが設定されていないため、GA機能は制限されます"
                )
                return False

            # ConditionEvolver用のYamlIndicatorUtilsを作成（メタデータ使用）
            yaml_indicator_utils = CoreYamlIndicatorUtils()

            # ConditionEvolverインスタンスを作成
            self.condition_evolver = ConditionEvolver(
                backtest_service=self.backtest_service,
                yaml_indicator_utils=yaml_indicator_utils,
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
        backtest_config: Optional[Dict[str, any]] = None,
    ) -> Tuple[
        List[Union[Condition, ConditionGroup]],
        List[Union[Condition, ConditionGroup]],
        List[Condition],
    ]:
        """階層的GAによる最適化条件生成"""
        if not self.use_hierarchical_ga or not self.initialize_ga_components():
            self.logger.info("階層的GAが無効または初期化失敗のため、標準生成を使用")
            return self.generate_balanced_conditions(indicators)

        try:
            self.logger.info(f"階層的GA条件生成開始: {len(indicators)}個の指標")
            backtest_config = backtest_config or {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "initial_balance": 10000,
                "fee_rate": 0.001,
            }

            optimized_conditions = []
            indicator_types = self._dynamic_classify(indicators)

            for indicator_type, type_indicators in indicator_types.items():
                if not type_indicators:
                    continue

                for indicator in type_indicators:
                    try:
                        evolution_result = self.condition_evolver.run_evolution(
                            backtest_config=backtest_config,
                            population_size=self.ga_config["population_size"],
                            generations=self.ga_config["generations"],
                        )

                        if evolution_result and "best_condition" in evolution_result:
                            optimized_conditions.append(evolution_result["best_condition"])
                        else:
                            # 個別フォールバック
                            optimized_conditions.extend(self._create_side_conditions(indicator, "long"))
                    except Exception as e:
                        self.logger.error(f"指標 {indicator.type} のGA最適化エラー: {e}")
                        optimized_conditions.extend(self._create_side_conditions(indicator, "long"))

            if not optimized_conditions:
                return self.generate_balanced_conditions(indicators)

            return self._process_generated_conditions(optimized_conditions, indicators)

        except Exception as e:
            self.logger.error(f"階層的GA条件生成エラー: {e}")
            return self.generate_balanced_conditions(indicators)

    def _process_generated_conditions(
        self, conditions: List[Condition], indicators: List[IndicatorGene]
    ) -> Tuple[List[Union[Condition, ConditionGroup]], List[Union[Condition, ConditionGroup]], List[Condition]]:
        """生成された条件の分離、制限、および最終フォールバック処理"""
        long_conditions = [c for c in conditions if getattr(c, "direction", "long") == "long"]
        short_conditions = [c for c in conditions if getattr(c, "direction", "long") == "short"]

        # ロング条件がない場合はバランス生成から補完
        if not long_conditions:
            self.logger.warning("ロング条件不足のため、標準生成で補完します")
            fallback_longs, fallback_shorts, fallback_exits = self.generate_balanced_conditions(indicators)
            return fallback_longs, fallback_shorts, fallback_exits

        # 条件数を制限
        max_conditions = getattr(self.ga_config_obj, "max_conditions", 3)
        if len(long_conditions) > max_conditions:
            long_conditions = random.sample(long_conditions, max_conditions)
        if len(short_conditions) > max_conditions:
            short_conditions = random.sample(short_conditions, max_conditions)

        return long_conditions, short_conditions, []


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
        self, indicator: IndicatorGene, direction: str, backtest_config: Dict[str, any]
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
                population_size=self.ga_config["population_size"]
                // 2,  # 単一指標なので個体数を減らす
                generations=self.ga_config["generations"] // 2,
            )

            if evolution_result and "best_condition" in evolution_result:
                best_condition = evolution_result["best_condition"]

                # 指定された方向に一致するか確認
                cond_direction = getattr(best_condition, "direction", None)
                if cond_direction == direction:
                    return best_condition

            return None

        except Exception as e:
            self.logger.error(
                f"単一条件最適化エラー ({indicator.type}, {direction}): {e}"
            )
            return None





