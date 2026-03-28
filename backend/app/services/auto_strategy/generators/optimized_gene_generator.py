"""
最適化されたランダム遺伝子生成器

パフォーマンス最適化版の遺伝子生成を提供します。
"""

import logging
import random
from typing import Any, Dict, List, Optional

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
from ..tools import tool_registry

from .condition_generator import ConditionGenerator

logger = logging.getLogger(__name__)


class OptimizedGeneGenerator:
    """
    最適化されたランダム戦略遺伝子生成器

    主な最適化ポイント:
    1. インジケーター生成のキャッシュ
    2. 条件生成の効率化
    3. 遺伝子組み立ての最適化
    4. 事前計算されたパラメータの再利用
    """

    def __init__(
        self,
        config: Any,
        smart_context: dict | None = None,
    ):
        """
        初期化

        Args:
            config: GA設定オブジェクト
            smart_context: スマート条件生成のコンテキスト
        """
        self.config = config
        self.smart_context: Dict[str, Any] = smart_context or {}

        # スマート条件生成器
        self.smart_condition_generator = ConditionGenerator(ga_config=config)
        try:
            self.smart_condition_generator.set_context(
                timeframe=self.smart_context.get("timeframe"),
                symbol=self.smart_context.get("symbol"),
                threshold_profile=self.smart_context.get("threshold_profile"),
                regime_thresholds=self.smart_context.get("regime_thresholds"),
            )
        except Exception:
            pass

        # 設定値を事前計算
        self.max_indicators = config.max_indicators
        self.min_indicators = config.min_indicators
        self.max_conditions = config.max_conditions
        self.min_conditions = config.min_conditions

        # キャッシュ
        self._indicator_cache: List[IndicatorGene] = []
        self._tpsl_cache: List[TPSLGene] = []
        self._position_sizing_cache: List[PositionSizingGene] = []

        # 事前計算されたツール遺伝子
        self._tool_genes_template: Optional[List[ToolGene]] = None

    def _initialize_caches(self):
        """キャッシュを初期化"""
        # インジケーターキャッシュを生成
        for _ in range(10):
            indicators = generate_random_indicators(self.config)
            if indicators:
                self._indicator_cache.extend(indicators)

        # TPSLキャッシュを生成
        for _ in range(10):
            tpsl = create_random_tpsl_gene(self.config)
            self._tpsl_cache.append(tpsl)

        # ポジションサイジングキャッシュを生成
        for _ in range(10):
            ps = create_random_position_sizing_gene(self.config)
            self._position_sizing_cache.append(ps)

        # ツール遺伝子テンプレートを生成
        self._tool_genes_template = self._generate_tool_genes_template()

    def _generate_tool_genes_template(self) -> List[ToolGene]:
        """ツール遺伝子テンプレートを生成"""
        tool_genes = []
        for tool in tool_registry.get_all():
            enabled = random.random() < 0.5
            params = tool.get_default_params()
            tool_genes.append(
                ToolGene(
                    tool_name=tool.name,
                    enabled=enabled,
                    params=params,
                )
            )
        return tool_genes

    def _get_cached_indicators(self) -> List[IndicatorGene]:
        """キャッシュからインジケーターを取得"""
        if not self._indicator_cache:
            self._initialize_caches()

        # ランダムに選択
        n_indicators = random.randint(self.min_indicators, self.max_indicators)
        if len(self._indicator_cache) >= n_indicators:
            return random.sample(self._indicator_cache, n_indicators)
        else:
            return generate_random_indicators(self.config)

    def _get_cached_tpsl(self) -> TPSLGene:
        """キャッシュからTPSL遺伝子を取得"""
        if not self._tpsl_cache:
            self._initialize_caches()

        if self._tpsl_cache:
            tpsl = random.choice(self._tpsl_cache).clone()
            tpsl.enabled = True
            return tpsl
        else:
            tpsl = create_random_tpsl_gene(self.config)
            tpsl.enabled = True
            return tpsl

    def _get_cached_position_sizing(self) -> PositionSizingGene:
        """キャッシュからポジションサイジング遺伝子を取得"""
        if not self._position_sizing_cache:
            self._initialize_caches()

        if self._position_sizing_cache:
            return random.choice(self._position_sizing_cache).clone()
        else:
            return create_random_position_sizing_gene(self.config)

    def _get_cached_tool_genes(self) -> List[ToolGene]:
        """キャッシュからツール遺伝子を取得"""
        if self._tool_genes_template is None:
            self._initialize_caches()

        # テンプレートをクローンしてランダムに有効/無効を設定
        tool_genes: List[ToolGene] = []
        for tool in self._tool_genes_template or []:
            enabled = random.random() < 0.5
            tool_genes.append(
                ToolGene(
                    tool_name=tool.tool_name,
                    enabled=enabled,
                    params=tool.params.copy(),
                )
            )
        return tool_genes

    @safe_operation(
        context="最適化ランダム戦略遺伝子生成",
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
        ランダムな戦略遺伝子を生成（最適化版）

        最適化:
        - キャッシュからの再利用
        - 事前計算されたパラメータ
        - 効率的な遺伝子組み立て
        """
        # キャッシュからインジケーターを取得
        indicators = self._get_cached_indicators()

        # キャッシュからTPSL遺伝子を取得
        tpsl_gene = self._get_cached_tpsl()
        long_tpsl_gene = self._get_cached_tpsl()
        short_tpsl_gene = self._get_cached_tpsl()

        # 条件を生成
        try:
            self.smart_condition_generator.indicators = indicators
        except Exception:
            pass

        long_entry_conditions, short_entry_conditions, _ = (
            self.smart_condition_generator.generate_balanced_conditions(indicators)
        )

        # 条件を正規化
        long_entry_conditions = self.smart_condition_generator.normalize_conditions(
            long_entry_conditions, "long", indicators
        )
        short_entry_conditions = self.smart_condition_generator.normalize_conditions(
            short_entry_conditions, "short", indicators
        )

        # リスク管理設定
        risk_management = {"position_size": 0.1}

        # キャッシュからポジションサイジング遺伝子を取得
        position_sizing_gene = self._get_cached_position_sizing()

        # エントリー遺伝子を生成
        entry_gene = create_random_entry_gene(self.config)
        long_entry_gene = create_random_entry_gene(self.config)
        short_entry_gene = create_random_entry_gene(self.config)

        # キャッシュからツール遺伝子を取得
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
            tool_genes=tool_genes,
            risk_management=risk_management,
            metadata={"generated_by": "OptimizedGeneGenerator"},
        )

    def clear_caches(self):
        """キャッシュをクリア"""
        self._indicator_cache.clear()
        self._tpsl_cache.clear()
        self._position_sizing_cache.clear()
        self._tool_genes_template = None
