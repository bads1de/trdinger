"""
ランダム遺伝子生成器

OI/FRデータを含む多様な戦略遺伝子をランダムに生成します。
スケール不一致問題を解決するため、オペランドグループ化システムを使用します。
"""

import logging
import random
from typing import Any, List

from app.utils.error_handler import safe_operation

from ..genes import (
    IndicatorGene,
    PositionSizingGene,
    PositionSizingMethod,
    StrategyGene,
    TPSLGene,
    create_random_entry_gene,
    create_random_position_sizing_gene,
    create_random_tpsl_gene,
    generate_random_indicators,
)
from ..genes.tool import ToolGene
from ..serializers.serialization import GeneSerializer
from ..tools import tool_registry

from .condition_generator import ConditionGenerator

logger = logging.getLogger(__name__)


class RandomGeneGenerator:
    """
    ランダム戦略遺伝子生成器

    OI/FRデータソースを含む多様な戦略遺伝子を生成します。
    """

    # トレンド系指標の優先順位（normalize_conditionsに移譲されたため削除可能だが、一旦メソッド削除を優先）

    def __init__(
        self,
        config: Any,
        enable_smart_generation: bool = True,
        smart_context: dict | None = None,
    ):
        """
        初期化

        Args:
            config: GA設定オブジェクト
            enable_smart_generation: ConditionGeneratorを使用するか
            smart_context: スマート条件生成のコンテキスト（timeframe/symbol/threshold_profile/regime_gating）
        """
        self.config = config
        self.enable_smart_generation = enable_smart_generation
        self.smart_context = smart_context or {}
        self.serializer = GeneSerializer(
            enable_smart_generation
        )  # GeneSerializerのインスタンスを作成
        self.smart_condition_generator = ConditionGenerator(
            enable_smart_generation, ga_config=config
        )
        # コンテキストがあれば適用
        try:
            smart_context = smart_context or {}
            self.smart_condition_generator.set_context(
                timeframe=smart_context.get("timeframe"),
                symbol=smart_context.get("symbol"),
                threshold_profile=smart_context.get("threshold_profile"),
                regime_thresholds=smart_context.get("regime_thresholds"),
            )
        except Exception:
            pass

        # 設定値を取得（型安全）
        self.max_indicators = config.max_indicators
        self.min_indicators = config.min_indicators
        self.max_conditions = config.max_conditions
        self.min_conditions = config.min_conditions
        self.threshold_ranges = config.threshold_ranges

    @safe_operation(
        context="ランダム戦略遺伝子生成",
        is_api_call=False,
        default_return=StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
            tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005),
            long_tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005),
            short_tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005),
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY, fixed_quantity=1000
            ),
            metadata={"generated_by": "Fallback"},
        ),
    )
    def generate_random_gene(self) -> StrategyGene:
        """
        ランダムな戦略遺伝子を生成

        指標、TP/SL、エントリー、ポジションサイジング、ツール設定をランダムに組み合わせた
        完全な戦略遺伝子を構築します。スマート生成が有効な場合は、
        ConditionGeneratorを使用して意味のある取引条件を生成します。

        Returns:
            生成されたStrategyGeneオブジェクト
        """
        # 指標を生成
        indicators = generate_random_indicators(self.config)

        # TP/SL遺伝子を生成
        tpsl_gene = create_random_tpsl_gene(self.config)

        # Auto-StrategyではTP/SLを常に有効化
        if tpsl_gene:
            tpsl_gene.enabled = True

        # Long/Short TP/SL遺伝子も個別に生成
        long_tpsl_gene = create_random_tpsl_gene(self.config)
        if long_tpsl_gene:
            long_tpsl_gene.enabled = True

        xtpsl_gene = create_random_tpsl_gene(self.config)
        if xtpsl_gene:
            xtpsl_gene.enabled = True
        short_tpsl_gene = xtpsl_gene

        # ロング・ショート条件を生成（SmartConditionGeneratorを使用）
        # geneに含まれる指標一覧を渡して、素名比較時のフォールバックを安定化
        try:
            self.smart_condition_generator.indicators = indicators
        except Exception:
            pass
        long_entry_conditions, short_entry_conditions, _ = (
            self.smart_condition_generator.generate_balanced_conditions(indicators)
        )

        # 条件の成立性を底上げ：OR 正規化と価格vsトレンド(or open)フォールバックをコアに委譲
        long_entry_conditions = self.smart_condition_generator.normalize_conditions(
            long_entry_conditions, "long", indicators
        )
        short_entry_conditions = self.smart_condition_generator.normalize_conditions(
            short_entry_conditions, "short", indicators
        )

        # リスク管理設定（従来方式、後方互換性のため保持）
        risk_management = {
            "position_size": 0.1,  # デフォルト値（実際にはposition_sizing_geneが使用される）
        }

        # ポジションサイジング遺伝子を生成（GA最適化対象）
        position_sizing_gene = create_random_position_sizing_gene(self.config)

        # エントリー遺伝子を生成（GA最適化対象）
        entry_gene = create_random_entry_gene(self.config)
        long_entry_gene = create_random_entry_gene(self.config)
        short_entry_gene = create_random_entry_gene(self.config)

        # ツール遺伝子を生成（週末フィルターなど）
        tool_genes = self._generate_tool_genes()

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
            tool_genes=tool_genes,
            risk_management=risk_management,
            metadata={"generated_by": "RandomGeneGenerator"},
        )

    def _generate_tool_genes(self) -> List[ToolGene]:
        """
        ツール遺伝子のリストをランダムに生成

        登録されたすべてのツール（週末フィルター等）に対して、
        ランダムに有効/無効を決定し、デフォルトパラメータを持つToolGeneを生成します。

        Returns:
            生成されたToolGeneのリスト
        """
        tool_genes = []

        # すべての登録済みツールを取得
        for tool in tool_registry.get_all():
            # 50%の確率で有効化
            enabled = random.random() < 0.5

            # デフォルトパラメータを取得
            params = tool.get_default_params()

            tool_genes.append(
                ToolGene(
                    tool_name=tool.name,
                    enabled=enabled,
                    params=params,
                )
            )

        return tool_genes
