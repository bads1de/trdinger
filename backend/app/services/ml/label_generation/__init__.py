"""
ラベル生成モジュール

価格変化率から3クラス分類（上昇・下落・レンジ）のラベルを生成するためのユーティリティを提供します。
scikit-learnのKBinsDiscretizerとPipelineを活用し、シンプルで効率的な実装を実現します。

メタラベリング（Fakeout Detection）向け:
- SignalGenerator: ブレイクアウトなどのイベントを検出
- LabelGenerationService: イベントベースのラベリングをサポート
"""

from .enums import ThresholdMethod
from .event_driven import BarrierProfile, EventDrivenLabelGenerator
from .label_generation_service import LabelGenerationService
from .presets import (
    apply_preset_by_name,
    get_common_presets,
)
from .signal_generator import SignalGenerator
from .transformer import PriceChangeTransformer
from .utils import (
    create_label_pipeline,
    optimize_label_generation_with_gridsearch,
)

# 向後互換性のため、__all__を定義
__all__ = [
    "EventDrivenLabelGenerator",
    "PriceChangeTransformer",
    "ThresholdMethod",
    "create_label_pipeline",
    "optimize_label_generation_with_gridsearch",
    "BarrierProfile",
    "get_common_presets",
    "apply_preset_by_name",
    "SignalGenerator",
    "LabelGenerationService",
]
