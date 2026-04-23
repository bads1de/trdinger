"""
ランダム遺伝子生成器

OI/FRデータを含む多様な戦略遺伝子をランダムに生成します。
スケール不一致問題を解決するため、オペランドグループ化システムを使用します。
"""

from __future__ import annotations

import logging
import random
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
)

if TYPE_CHECKING:
    from ..config.ga.ga_config import GAConfig

from app.utils.error_handler import safe_operation

from ..genes import (
    ExitGene,
    IndicatorGene,
    PositionSizingGene,
    PositionSizingMethod,
    StrategyGene,
    TPSLGene,
    create_random_entry_gene,
    create_random_exit_gene,
    create_random_position_sizing_gene,
    create_random_tpsl_gene,
    generate_random_indicators,
)
from ..genes.tool import ToolGene
from ..tools import tool_registry
from .condition_generator import ConditionGenerator

logger = logging.getLogger(__name__)


class Cloneable(Protocol):
    """clone() メソッドを持つオブジェクトのプロトコル"""

    def clone(self) -> Any: ...


T = TypeVar("T", bound=Cloneable)


class RandomGeneGenerator:
    """
    ランダム戦略遺伝子生成器

    OI/FRデータソースを含む多様な戦略遺伝子を生成します。
    """

    # キャッシュ設定
    CACHE_SIZE = 10
    TOOL_ENABLE_PROBABILITY = 0.5
    DEFAULT_POSITION_SIZE = 0.1

    # フィルター優先度に応じた有効化確率
    TOOL_ENABLE_PROBABILITIES = {
        "essential": 0.8,  # 必須フィルター（週末など）
        "optional": 0.3,  # オプションフィルター（トレンド、出来高など）
        "disabled": 0.1,  # デフォルト無効フィルター（その他）
    }

    # フィルターコスト定義（予算システム）
    TOOL_COSTS = {
        "essential": 0,  # 必須フィルターはコスト0
        "optional": 1,  # オプションフィルターはコスト1
        "disabled": 2,  # デフォルト無効フィルターはコスト2
    }

    def __init__(
        self,
        config: "GAConfig",
        smart_context: dict | None = None,
    ):
        """
        初期化

        Args:
            config: GA設定オブジェクト
            smart_context: スマート条件生成のコンテキスト（timeframe/symbol/threshold_profile/regime_gating）
        """
        self.config = config
        self.smart_context = smart_context or {}

        # 常にスマート生成を使用
        self.smart_condition_generator = ConditionGenerator(ga_config=config)
        # コンテキストがあれば適用
        try:
            smart_context = smart_context or {}
            self.smart_condition_generator.set_context(
                timeframe=smart_context.get("timeframe"),
                symbol=smart_context.get("symbol"),
                threshold_profile=smart_context.get("threshold_profile"),
                regime_thresholds=smart_context.get("regime_thresholds"),
            )
        except Exception as e:
            logger.debug("スマートコンテキストの初期化に失敗しました: %s", e)
            pass

        # 設定値を取得（型安全）
        self.max_indicators = config.max_indicators
        self.min_indicators = config.min_indicators
        self.max_conditions = config.max_conditions
        self.min_conditions = config.min_conditions
        self.threshold_ranges = config.threshold_ranges

        # 最適化: キャッシュ
        self._indicator_cache: List[IndicatorGene] = []
        self._tpsl_cache: List[TPSLGene] = []
        self._exit_gene_cache: List[ExitGene] = []
        self._position_sizing_cache: List[PositionSizingGene] = []
        self._tool_genes_template: Optional[List[ToolGene]] = None

    @staticmethod
    def _create_enabled_tpsl_gene() -> TPSLGene:
        """
        ランダム生成したTP/SL遺伝子を有効化して返す

        ランダムに生成されたTP/SL遺伝子のenabledフラグをTrueに設定して返します。

        Returns:
            TPSLGene: 有効化されたTP/SL遺伝子
        """
        gene = create_random_tpsl_gene()
        gene.enabled = True
        return gene

    def _initialize_caches(self) -> None:
        """
        生成用キャッシュを初期化する

        インジケーター、TP/SL、ポジションサイジング、ツール遺伝子の
        キャッシュを事前に生成して、ランダム生成のパフォーマンスを向上させます。

        Note:
            各キャッシュにCACHE_SIZE個のサンプルを生成します。
        """
        for _ in range(self.CACHE_SIZE):
            indicators = generate_random_indicators(self.config)
            if indicators:
                self._indicator_cache.extend(indicators)

        for _ in range(self.CACHE_SIZE):
            self._tpsl_cache.append(create_random_tpsl_gene())

        for _ in range(self.CACHE_SIZE):
            self._exit_gene_cache.append(create_random_exit_gene(self.config))

        for _ in range(self.CACHE_SIZE):
            self._position_sizing_cache.append(
                create_random_position_sizing_gene()
            )

        self._tool_genes_template = self._generate_tool_genes_template()

    @staticmethod
    def _clone_or_create_gene(
        cache: Sequence[T],
        creator: Callable[[], T],
        *,
        postprocess: Optional[Callable[[T], None]] = None,
    ) -> T:
        """
        キャッシュから clone し、無ければ creator で生成する

        キャッシュから遺伝子をクローンして返します。キャッシュが空の場合は、
        creator関数で新しい遺伝子を生成します。

        Args:
            cache: 遺伝子キャッシュリスト
            creator: 新規生成関数
            postprocess: 生成後の後処理関数（オプション）

        Returns:
            クローンまたは生成された遺伝子
        """
        gene = random.choice(cache).clone() if cache else creator()
        if postprocess is not None:
            postprocess(gene)
        return gene

    def _generate_tool_genes_template(self) -> List[ToolGene]:
        """
        ツール遺伝子テンプレートを生成する

        ツールレジストリから全ツールを取得し、ツール遺伝子テンプレートを生成します。
        各ツールの有効フラグは優先度に応じた確率で設定されます。

        Returns:
            List[ToolGene]: ツール遺伝子テンプレートリスト
        """
        tool_genes: List[ToolGene] = []
        for tool in tool_registry.get_all():
            # 優先度に応じた確率を取得
            priority = tool.definition.priority
            probability = self.TOOL_ENABLE_PROBABILITIES.get(
                priority, self.TOOL_ENABLE_PROBABILITY
            )
            tool_genes.append(
                ToolGene(
                    tool_name=tool.name,
                    enabled=random.random() < probability,
                    params=tool.get_default_params(),
                )
            )
        # フィルター数制限を強制
        return self._enforce_filter_limit(tool_genes)

    def _enforce_filter_limit(self, tool_genes: List[Any]) -> List[Any]:
        """
        フィルター数制限を強制する

        max_enabled_filtersを超える場合、コストの高いフィルターからランダムに無効化します。

        Args:
            tool_genes: ツール遺伝子リスト

        Returns:
            制限を適用したツール遺伝子リスト
        """
        max_filters = getattr(self.config, "max_enabled_filters", 3)

        # 有効なフィルターをコスト順にソート（コストが高い＝優先度低い）
        enabled_filters = [
            t for t in tool_genes if hasattr(t, "enabled") and t.enabled
        ]
        disabled_filters = [
            t for t in tool_genes if hasattr(t, "enabled") and not t.enabled
        ]

        # コストを計算
        def get_cost(tool_gene: ToolGene) -> int:
            tool = tool_registry.get(tool_gene.tool_name)
            if tool:
                priority = tool.definition.priority
                return self.TOOL_COSTS.get(priority, 1)
            return 1

        enabled_filters.sort(key=get_cost, reverse=True)

        # 制限を超える場合、コストの高いフィルターから無効化
        disabled_count = 0
        while len(enabled_filters) > max_filters:
            tool_to_disable = enabled_filters.pop(0)
            tool_to_disable.enabled = False
            disabled_filters.append(tool_to_disable)
            disabled_count += 1

        return tool_genes

    def _get_cached_indicators(self) -> List[IndicatorGene]:
        """
        キャッシュからインジケーターを取得する

        キャッシュからランダムにインジケーターを取得します。
        キャッシュが空の場合は初期化し、キャッシュサイズが不足する場合は
        新規生成を行います。

        Returns:
            List[IndicatorGene]: インジケーター遺伝子リスト

        Note:
            インジケーター数はmin_indicatorsからmax_indicatorsの間でランダムに決定されます。
        """
        if not self._indicator_cache:
            self._initialize_caches()

        n_indicators = random.randint(self.min_indicators, self.max_indicators)
        if len(self._indicator_cache) >= n_indicators:
            return [
                ind.clone()
                for ind in random.sample(self._indicator_cache, n_indicators)
            ]
        return generate_random_indicators(self.config)

    def _get_cached_tpsl(self) -> TPSLGene:
        """
        キャッシュからTP/SL遺伝子を取得する

        キャッシュからTP/SL遺伝子を取得します。キャッシュが空の場合は初期化します。
        取得した遺伝子のenabledフラグはTrueに設定されます。

        Returns:
            TPSLGene: TP/SL遺伝子
        """
        if not self._tpsl_cache:
            self._initialize_caches()

        return self._clone_or_create_gene(
            self._tpsl_cache,
            lambda: create_random_tpsl_gene(),
            postprocess=lambda gene: setattr(gene, "enabled", True),
        )

    def _get_cached_position_sizing(self) -> PositionSizingGene:
        """
        キャッシュからポジションサイジング遺伝子を取得する

        キャッシュからポジションサイジング遺伝子を取得します。
        キャッシュが空の場合は初期化します。

        Returns:
            PositionSizingGene: ポジションサイジング遺伝子
        """
        if not self._position_sizing_cache:
            self._initialize_caches()

        return self._clone_or_create_gene(
            self._position_sizing_cache,
            lambda: create_random_position_sizing_gene(),
        )

    def _get_cached_exit_gene(self) -> ExitGene:
        """
        キャッシュから exit_gene を取得する

        Returns:
            ExitGene: イグジット遺伝子
        """
        if not self._exit_gene_cache:
            self._initialize_caches()

        return self._clone_or_create_gene(
            self._exit_gene_cache,
            lambda: create_random_exit_gene(self.config),
        )

    def _get_cached_tool_genes(self) -> List[ToolGene]:
        """
        キャッシュからツール遺伝子を取得する

        キャッシュからツール遺伝子テンプレートを取得し、クローンして返します。
        各ツールの有効フラグは再抽選されます。

        Returns:
            List[ToolGene]: ツール遺伝子リスト
        """
        if self._tool_genes_template is None:
            self._initialize_caches()

        return [
            self._clone_tool_gene_template(tool)
            for tool in self._tool_genes_template or []
        ]

    def _clone_tool_gene_template(self, tool: ToolGene) -> ToolGene:
        """
        ツール遺伝子テンプレートを clone し、enabled を維持する

        ツール遺伝子テンプレートをクローンし、有効フラグを維持します。
        （フィルター制限を適用したテンプレートの状態を保持）

        Args:
            tool: ツール遺伝子テンプレート

        Returns:
            ToolGene: クローンされたツール遺伝子
        """
        cloned = tool.clone()
        # enabledフラグをランダムに再設定しない（テンプレートの状態を維持）
        # cloned.enabled = random.random() < self.TOOL_ENABLE_PROBABILITY
        return cloned

    @safe_operation(
        context="ランダム戦略遺伝子生成",
        is_api_call=False,
        default_return=StrategyGene(
            indicators=[
                IndicatorGene(
                    type="SMA", parameters={"period": 20}, enabled=True
                )
            ],
            long_entry_conditions=[],
            short_entry_conditions=[],
            long_exit_conditions=[],
            short_exit_conditions=[],
            risk_management={},
            tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005),
            long_tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005),
            short_tpsl_gene=TPSLGene(
                take_profit_pct=0.01, stop_loss_pct=0.005
            ),
            exit_gene=ExitGene(),
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY, fixed_quantity=1000
            ),
            metadata={"generated_by": "Fallback"},
        ),
    )
    def generate_random_gene(self) -> StrategyGene:
        """
        ランダムなパラメータと論理構造を持つ、新しい取引戦略の「遺伝子」を生成します。

        このメソッドは、初期集団の生成や突然変異の際に呼び出され、以下のステップで一つの完結した戦略設計図を構築します：
        1. **テクニカル指標の生成**: `IndicatorUniverse` から RSI, MACD, SMA 等をランダムに選択。
        2. **TP/SL設定**: 全体用およびロング・ショート個別の利確・損切り幅を生成。
        3. **取引条件（ロジック）の生成**: `ConditionGenerator` を使用して、指標間の比較や閾値判定等の論理式を構築。
           - 「スマート生成」により、価格スケールと指標スケールの不一致を避けた意味のある条件を生成します。
        4. **ポジションサイジング設定**: 固定量、資産比率、ボラティリティ比例等の資金管理手法を決定。
        5. **エントリー管理設定**: エントリー順序や同時注文制限等の実行ルールを設定。
        6. **補助ツール設定**: 週末フィルタやボラティリティフィルタ等の外部制約をランダムに有効化。

        Returns:
            StrategyGene: 生成された戦略の全パラメータを保持する遺伝子オブジェクト。
        """
        # キャッシュを活用して指標とサブ遺伝子を生成
        indicators = self._get_cached_indicators()
        tpsl_gene = self._get_cached_tpsl()
        long_tpsl_gene = self._get_cached_tpsl()
        short_tpsl_gene = self._get_cached_tpsl()
        exit_gene = self._get_cached_exit_gene()

        # ロング・ショート条件を生成（SmartConditionGeneratorを使用）
        # geneに含まれる指標一覧を渡して、素名比較時のフォールバックを安定化
        try:
            setattr(self.smart_condition_generator, "indicators", indicators)
        except Exception as e:
            logger.debug("指標キャッシュの設定に失敗しました: %s", e)
            pass
        long_entry_conditions, short_entry_conditions, _ = (
            self.smart_condition_generator.generate_balanced_conditions(
                indicators
            )
        )

        # 条件の成立性を底上げ：OR 正規化と価格vsトレンド(or open)フォールバックをコアに委譲
        long_entry_conditions = (
            self.smart_condition_generator.normalize_conditions(
                long_entry_conditions, "long", indicators
            )
        )
        short_entry_conditions = (
            self.smart_condition_generator.normalize_conditions(
                short_entry_conditions, "short", indicators
            )
        )
        long_exit_conditions, short_exit_conditions, _ = (
            self.smart_condition_generator.generate_exit_conditions(indicators)
        )

        # リスク管理設定（従来方式）
        risk_management = {
            "position_size": self.DEFAULT_POSITION_SIZE,  # デフォルト値（実際にはposition_sizing_geneが使用される）
        }

        # ポジションサイジング遺伝子を生成（GA最適化対象）
        position_sizing_gene = self._get_cached_position_sizing()

        # エントリー遺伝子を生成（GA最適化対象）
        entry_gene = create_random_entry_gene(self.config)
        long_entry_gene = create_random_entry_gene(self.config)
        short_entry_gene = create_random_entry_gene(self.config)

        # ツール遺伝子を生成（週末フィルターなど）
        tool_genes = self._get_cached_tool_genes()

        return StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_entry_conditions,
            short_entry_conditions=short_entry_conditions,
            tpsl_gene=tpsl_gene,
            long_tpsl_gene=long_tpsl_gene,
            short_tpsl_gene=short_tpsl_gene,
            position_sizing_gene=position_sizing_gene,
            entry_gene=entry_gene,
            long_entry_gene=long_entry_gene,
            short_entry_gene=short_entry_gene,
            exit_gene=exit_gene,
            long_exit_conditions=long_exit_conditions,
            short_exit_conditions=short_exit_conditions,
            tool_genes=tool_genes,
            risk_management=risk_management,
            metadata={"generated_by": "RandomGeneGenerator"},
        )

    def _clear_caches(self) -> None:
        """
        生成キャッシュをクリアする

        すべてのキャッシュ（インジケーター、TP/SL、ポジションサイジング、ツール）を
        クリアして、メモリを解放します。
        """
        self._indicator_cache.clear()
        self._tpsl_cache.clear()
        self._exit_gene_cache.clear()
        self._position_sizing_cache.clear()
        self._tool_genes_template = None

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        キャッシュ統計を返す

        各キャッシュのサイズを返します。

        Returns:
            Dict[str, Any]: キャッシュ統計辞書
                - indicator_cache_size: インジケーターキャッシュサイズ
                - tpsl_cache_size: TP/SLキャッシュサイズ
                - exit_gene_cache_size: ExitGeneキャッシュサイズ
                - position_sizing_cache_size: ポジションサイジングキャッシュサイズ
                - tool_genes_template_size: ツール遺伝子テンプレートサイズ
        """
        return {
            "indicator_cache_size": len(self._indicator_cache),
            "tpsl_cache_size": len(self._tpsl_cache),
            "exit_gene_cache_size": len(self._exit_gene_cache),
            "position_sizing_cache_size": len(self._position_sizing_cache),
            "tool_genes_template_size": len(self._tool_genes_template or []),
        }
