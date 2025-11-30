"""
ラベル生成モジュール

価格変化率から3クラス分類（上昇・下落・レンジ）のラベルを生成するためのユーティリティを提供します。
scikit-learnのKBinsDiscretizerとPipelineを活用し、シンプルで効率的な実装を実現します。
"""

from .enums import ThresholdMethod
from .event_driven import BarrierProfile, EventDrivenLabelGenerator
from .label_generator import LabelGenerator
from .presets import (
    apply_preset_by_name,
    forward_classification_preset,
    get_common_presets,
)
from .transformer import PriceChangeTransformer
from .utils import (
    create_label_pipeline,
    optimize_label_generation_with_gridsearch,
)

# 向後互換性のため、__all__を定義
__all__ = [
    "LabelGenerator",
    "EventDrivenLabelGenerator",
    "PriceChangeTransformer",
    "ThresholdMethod",
    "create_label_pipeline",
    "optimize_label_generation_with_gridsearch",
    "BarrierProfile",
    "forward_classification_preset",
    "get_common_presets",
    "apply_preset_by_name",
]
