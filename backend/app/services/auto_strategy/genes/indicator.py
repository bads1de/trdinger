"""
指標遺伝子モデル
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry
from app.utils.error_handler import safe_operation

logger = logging.getLogger(__name__)


@dataclass
class IndicatorGene:
    """
    指標遺伝子

    単一のテクニカル指標の設定を表現します。

    Attributes:
        type: 指標タイプ（例: "SMA", "RSI"）
        parameters: 指標パラメータ（例: {"period": 20}）
        enabled: 指標が有効かどうか
        timeframe: この指標が計算されるタイムフレーム。
            None の場合は戦略のデフォルトタイムフレームを使用。
            例: "1h", "4h", "1d" など
        id: 指標の一意識別子（オプション）。複数の同じ種類の指標を区別するために使用。
        json_config: JSON形式の設定キャッシュ
    """

    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    timeframe: Optional[str] = None
    id: Optional[str] = None
    json_config: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """指標遺伝子の妥当性を検証"""
        from .validator import GeneValidator

        validator = GeneValidator()
        return validator.validate_indicator_gene(self)

    def get_json_config(self) -> Dict[str, Any]:
        """JSON形式の設定を取得"""
        try:
            from app.services.indicators.config import indicator_registry

            config = indicator_registry.get_indicator_config(self.type)
            if config:
                resolved_params = {}
                for param_name, param_config in config.parameters.items():
                    resolved_params[param_name] = self.parameters.get(
                        param_name, param_config.default_value
                    )
                return {"indicator": self.type, "parameters": resolved_params}
            return {"indicator": self.type, "parameters": self.parameters}
        except ImportError:
            return {"indicator": self.type, "parameters": self.parameters}


@safe_operation(
    default_return=IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
    context="指標遺伝子作成",
)
def create_random_indicator_gene(
    indicator_type: str, config: Any = None, timeframe: str | None = None
) -> IndicatorGene:
    """
    ランダムなパラメータを持つ指標遺伝子を作成

    Args:
        indicator_type: 指標タイプ（例: "SMA", "RSI"）
        config: GA設定オブジェクト（オプション）
        timeframe: この指標が計算されるタイムフレーム

    Returns:
        指標遺伝子オブジェクト
    """
    # パラメータ生成
    preset = getattr(config, "parameter_range_preset", None) if config else None
    parameters = indicator_registry.generate_parameters_for_indicator(
        indicator_type, preset=preset
    )

    indicator_gene = IndicatorGene(
        type=indicator_type,
        parameters=parameters,
        enabled=True,
        timeframe=timeframe,
        id=str(uuid.uuid4()),
    )

    # JSON設定をキャッシュ
    indicator_gene.json_config = indicator_gene.get_json_config()

    return indicator_gene


@safe_operation(default_return=[], context="ランダム指標リスト生成")
def generate_random_indicators(config: Any) -> List[IndicatorGene]:
    """
    設定に基づいてランダムな指標リストを生成

    Args:
        config: GA設定オブジェクト

    Returns:
        指標遺伝子のリスト
    """
    # 利用可能な指標のリストを取得
    indicator_service = TechnicalIndicatorService()
    available_indicators = list(indicator_service.get_supported_indicators().keys())

    if not available_indicators:
        return [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]

    # 指標の個数を決定
    num_indicators = random.randint(config.min_indicators, config.max_indicators)
    indicators = []

    # MTF設定の準備
    from ..config.constants import SUPPORTED_TIMEFRAMES

    available_timeframes = config.available_timeframes or SUPPORTED_TIMEFRAMES

    def get_random_timeframe():
        if not config.enable_multi_timeframe:
            return None
        if random.random() > config.mtf_indicator_probability:
            return None
        return random.choice(available_timeframes)

    # 各指標を生成
    for _ in range(num_indicators):
        indicator_type = random.choice(available_indicators)
        timeframe = get_random_timeframe()
        indicator_gene = create_random_indicator_gene(indicator_type, config, timeframe)
        indicators.append(indicator_gene)

    # 指標構成の底上げ（MAクロス戦略など）
    indicators = _enhance_with_ma_cross(indicators, available_indicators, config)

    return indicators


def _enhance_with_ma_cross(indicators: List[IndicatorGene], available: List[str], config: Any) -> List[IndicatorGene]:
    """MAクロス戦略を可能にするためにMA指標を確率的に追加"""
    from ..config.constants import MOVING_AVERAGE_INDICATORS, PREFERRED_MA_INDICATORS
    
    ma_count = sum(1 for ind in indicators if ind.type in MOVING_AVERAGE_INDICATORS)
    if ma_count < 2 and random.random() < 0.25:
        ma_pool = [n for n in available if n in MOVING_AVERAGE_INDICATORS]
        if ma_pool:
            preferred = [n for n in ma_pool if n in PREFERRED_MA_INDICATORS]
            chosen = random.choice(preferred or ma_pool)
            
            # 既存の期間と被らないように調整
            existing_periods = {ind.parameters.get("period") for ind in indicators if "period" in ind.parameters}
            period = random.choice([10, 14, 20, 30, 50])
            while period in existing_periods and len(existing_periods) < 5:
                period = random.choice([10, 14, 20, 30, 50])
            
            indicators.append(IndicatorGene(type=chosen, parameters={"period": period}))
            if len(indicators) > getattr(config, "max_indicators", 5):
                for i, ind in enumerate(indicators):
                    if ind.type not in MOVING_AVERAGE_INDICATORS:
                        indicators.pop(i)
                        break
    return indicators






