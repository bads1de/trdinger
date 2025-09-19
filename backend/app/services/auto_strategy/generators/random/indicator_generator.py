"""
指標生成器

ランダム戦略の指標部分を生成する専門ジェネレーター
@safe_operationデコレータを使用した堅牢なエラーハンドリングを導入
"""

import logging
import random
from typing import List, Any

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry
from app.utils.error_handler import safe_operation
from ...constants import CURATED_TECHNICAL_INDICATORS
from ...models.strategy_models import IndicatorGene
from ...utils.indicator_utils import get_all_indicators
from ..indicator_composition_service import IndicatorCompositionService

logger = logging.getLogger(__name__)


class IndicatorGenerator:
    """
    ランダム戦略の指標生成を担当するクラス
    """

    # トレンド系指標の優先順位
    TREND_PREF = (
        "SMA",
        "EMA",
        "WMA",
        "DEMA",
        "TEMA",
        "T3",
        "KAMA",
        "SAR",
    )

    def __init__(self, config: Any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config
        self.indicator_service = TechnicalIndicatorService()
        self._setup_coverage_tracking(config)
        self._valid_indicator_names = self._initialize_valid_indicators()
        self.composition_service = IndicatorCompositionService(config)
        self.available_indicators = self._setup_indicators_by_mode(config)

    def _initialize_valid_indicators(self) -> set:
        """有効な指標名を初期化"""
        try:
            return set(get_all_indicators())
        except Exception:
            return set()

    def _setup_coverage_tracking(self, config: Any):
        """カバレッジトラッキングを設定"""
        try:
            self._coverage_cycle = list(getattr(config, "allowed_indicators", []) or [])
            if self._coverage_cycle:
                random.shuffle(self._coverage_cycle)
            self._coverage_idx = 0
            self._coverage_pick = None
        except Exception:
            self._coverage_cycle = []
            self._coverage_idx = 0
            self._coverage_pick = None

    @safe_operation(default_return=IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True), context="デフォルト指標遺伝子生成")
    def _get_default_indicator_gene(self) -> IndicatorGene:
        """デフォルトの指標遺伝子を返す"""
        return IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)

    @safe_operation(default_return={}, context="指標パラメータ生成")
    def _safe_generate_parameters(self, indicator_type: str) -> dict:
        """安全に指標パラメータを生成"""
        return indicator_registry.generate_parameters_for_indicator(indicator_type)

    @safe_operation(default_return=None, context="JSON設定生成")
    def _safe_generate_json_config(self, indicator_gene: IndicatorGene):
        """安全にJSON設定を生成"""
        json_config = indicator_gene.get_json_config()
        indicator_gene.json_config = json_config


    @safe_operation(default_return=[], context="指標モード別設定")
    def _setup_indicators_by_mode(self, config: Any) -> List[str]:
        """
        指標モードに応じて利用可能な指標を設定

        Args:
            config: GA設定

        Returns:
            利用可能な指標のリスト
        """
        # テクニカル指標を取得
        technical_indicators = list(
            self.indicator_service.get_supported_indicators().keys()
        )

        # テクニカル指標のみを使用
        available_indicators = technical_indicators
        logger.info(
            f"指標モード: テクニカルオンリー ({len(available_indicators)}個の指標)"
        )

        # allowed_indicators によりさらに絞り込み
        allowed = set(getattr(config, "allowed_indicators", []) or [])
        if allowed:
            available_indicators = [
                ind for ind in available_indicators if ind in allowed
            ]

        # カバレッジモード: allowed 指定時は1つは巡回候補を確実に含める
        if self._coverage_cycle:
            self._coverage_pick = self._coverage_cycle[
                self._coverage_idx % len(self._coverage_cycle)
            ]
            self._coverage_idx += 1

        return available_indicators

    @safe_operation(default_return=[], context="ランダム指標生成")
    def generate_random_indicators(self) -> List[IndicatorGene]:
        """ランダムな指標リストを生成"""
        # coverage_pick は上位で準備済み
        coverage_pick = getattr(self, "_coverage_pick", None)

        num_indicators = random.randint(
            self.config.min_indicators, self.config.max_indicators
        )
        indicators = []

        # 1つ目は coverage_pick を優先
        if coverage_pick and coverage_pick in self.available_indicators:
            first_type = coverage_pick
        else:
            first_type = (
                random.choice(self.available_indicators)
                if self.available_indicators
                else "SMA"
            )

        # 1本目の指標生成
        indicator_gene = self._create_indicator_gene(first_type)
        indicators.append(indicator_gene)

        # 残りの指標生成
        for i in range(1, num_indicators):
            if not self.available_indicators:
                indicator_type = "SMA"
            else:
                indicator_type = random.choice(self.available_indicators)

            indicator_gene = self._create_indicator_gene(indicator_type)
            indicators.append(indicator_gene)

            # allowed_indicators が明示指定されている場合は早期リターン
            if getattr(self.config, "allowed_indicators", None):
                return indicators

        # 指標構成サービスで成立性を底上げ
        indicators = self._enhance_indicators_with_composition_service(indicators)

        return indicators

    @safe_operation(default_return=IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True), context="指標遺伝子作成")
    def _create_indicator_gene(self, indicator_type: str) -> IndicatorGene:
        """指標遺伝子を作成"""
        parameters = self._safe_generate_parameters(indicator_type)
        indicator_gene = IndicatorGene(
            type=indicator_type, parameters=parameters, enabled=True
        )
        self._safe_generate_json_config(indicator_gene)
        return indicator_gene

    @safe_operation(default_return=[], context="指標構成サービス適用")
    def _enhance_indicators_with_composition_service(self, indicators: List[IndicatorGene]) -> List[IndicatorGene]:
        """指標構成サービスを適用して成立性を底上げ"""
        # トレンド指標の強制追加
        indicators = self.composition_service.enhance_with_trend_indicators(
            indicators, self.available_indicators
        )

        # MAクロス戦略を可能にするための追加
        indicators = self.composition_service.enhance_with_ma_cross_strategy(
            indicators, self.available_indicators
        )

        return indicators
