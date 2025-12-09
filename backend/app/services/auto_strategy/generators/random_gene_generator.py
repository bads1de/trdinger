"""
ランダム遺伝子生成器

OI/FRデータを含む多様な戦略遺伝子をランダムに生成します。
スケール不一致問題を解決するため、オペランドグループ化システムを使用します。
"""

import logging
import random
from typing import Any, List, Union, cast

from app.services.indicators.config import indicator_registry
from app.utils.error_handler import safe_operation

from ..models.strategy_models import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    PositionSizingGene,
    PositionSizingMethod,
    StrategyGene,
    TPSLGene,
)
from ..serializers.gene_serialization import GeneSerializer
from .condition_generator import ConditionGenerator
from .random.condition_generator import ConditionGenerator as RandomConditionGenerator
from .random.indicator_generator import IndicatorGenerator
from .random.operand_generator import OperandGenerator
from .random.position_sizing_generator import PositionSizingGenerator
from .random.tpsl_generator import TPSLGenerator

logger = logging.getLogger(__name__)


class RandomGeneGenerator:
    """
    ランダム戦略遺伝子生成器

    OI/FRデータソースを含む多様な戦略遺伝子を生成します。
    """

    # トレンド系指標の優先順位
    TREND_PREF = (
        "SMA",
        "EMA",
    )

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
        self.smart_condition_generator = ConditionGenerator(enable_smart_generation)
        # コンテキストがあれば適用
        try:
            smart_context = smart_context or {}
            self.smart_condition_generator.set_context(
                timeframe=smart_context.get("timeframe"),
                symbol=smart_context.get("symbol"),
                threshold_profile=smart_context.get("threshold_profile"),
            )
        except Exception:
            pass

        # 設定値を取得（型安全）
        self.max_indicators = config.max_indicators
        self.min_indicators = config.min_indicators
        self.max_conditions = config.max_conditions
        self.min_conditions = config.min_conditions
        self.threshold_ranges = config.threshold_ranges

        # 分割されたジェネレーターのインスタンスを作成
        self.indicator_generator = IndicatorGenerator(config)
        self.condition_generator = RandomConditionGenerator(config)
        self.tpsl_generator = TPSLGenerator(config)
        self.position_sizing_generator = PositionSizingGenerator(config)
        self.operand_generator = OperandGenerator(config)

    def _ensure_or_with_fallback(
        self, conds: List[Union[Condition, ConditionGroup]], side: str, indicators
    ) -> List[Union[Condition, ConditionGroup]]:
        """
        条件の正規化/組立ヘルパー：
        - フォールバック（価格 vs トレンド or open）の注入
        - 1件なら素条件のまま、2件以上なら OR グルーピング
        """
        # フォールバック (PriceTrendPolicyのロジックを直接実装)
        trend_pool = []
        for ind in indicators or []:
            if not getattr(ind, "enabled", True):
                continue
            cfg = indicator_registry.get_indicator_config(ind.type)
            if cfg and getattr(cfg, "category", None) == "trend":
                trend_pool.append(ind.type)
        # 優先候補
        pref = [n for n in trend_pool if n in self.TREND_PREF]
        if pref:
            trend_name = random.choice(pref)
        elif trend_pool:
            trend_name = random.choice(trend_pool)
        else:
            trend_name = random.choice(self.TREND_PREF)
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

    @safe_operation(
        context="ランダム戦略遺伝子生成",
        is_api_call=False,
        default_return=StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            entry_conditions=[],
            exit_conditions=[],
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

        Returns:
            生成された戦略遺伝子
        """
        # 指標を生成
        indicators = self.indicator_generator.generate_random_indicators()

        # 条件を生成（後方互換性のため保持）
        entry_conditions = self.condition_generator.generate_random_conditions(
            indicators, "entry"
        )

        # TP/SL遺伝子を先に生成してイグジット条件生成を調整
        tpsl_gene = self.tpsl_generator.generate_tpsl_gene()

        # Auto-StrategyではTP/SLを常に有効化し、エグジット条件は冗長のため生成しない
        if tpsl_gene:
            tpsl_gene.enabled = True

        # Long/Short TP/SL遺伝子も個別に生成
        long_tpsl_gene = self.tpsl_generator.generate_tpsl_gene()
        if long_tpsl_gene:
            long_tpsl_gene.enabled = True

        short_tpsl_gene = self.tpsl_generator.generate_tpsl_gene()
        if short_tpsl_gene:
            short_tpsl_gene.enabled = True

        # TP/SL遺伝子が有効な場合はイグジット条件を最小化
        if tpsl_gene and tpsl_gene.enabled:
            exit_conditions = []
        else:
            exit_conditions = self.condition_generator.generate_random_conditions(
                indicators, "exit"
            )

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
        long_entry_conditions = self._ensure_or_with_fallback(
            long_entry_conditions, "long", indicators
        )
        short_entry_conditions = self._ensure_or_with_fallback(
            short_entry_conditions, "short", indicators
        )

        # リスク管理設定（従来方式、後方互換性のため保持）
        risk_management = {
            "position_size": 0.1,  # デフォルト値（実際にはposition_sizing_geneが使用される）
        }

        # ポジションサイジング遺伝子を生成（GA最適化対象）
        position_sizing_gene = (
            self.position_sizing_generator.generate_position_sizing_gene()
        )

        gene = StrategyGene(
            indicators=indicators,
            entry_conditions=entry_conditions,  # 後方互換性
            exit_conditions=exit_conditions,
            long_entry_conditions=long_entry_conditions,  # 新機能
            short_entry_conditions=short_entry_conditions,  # 新機能
            risk_management=risk_management,
            tpsl_gene=tpsl_gene,  # 新しいTP/SL遺伝子
            long_tpsl_gene=long_tpsl_gene,  # Long TP/SL
            short_tpsl_gene=short_tpsl_gene,  # Short TP/SL
            position_sizing_gene=position_sizing_gene,  # 新しいポジションサイジング遺伝子
            metadata={"generated_by": "RandomGeneGenerator"},
        )

        return gene
