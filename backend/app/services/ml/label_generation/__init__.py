"""
ラベル生成モジュール（二値分類 / メタラベリング専用）

エントリーシグナルの有効性を判定するための二値ラベル（0/1）を生成します。
Triple Barrier MethodやTrend Scanningを用いて、ダマシ（False Signal）を検出します。

主要コンポーネント:
- SignalGenerator: ブレイクアウトなどのイベントを検出
- LabelGenerationService: イベントベースのメタラベリングをサポート
"""

from .enums import ThresholdMethod
from .event_driven import BarrierProfile, EventDrivenLabelGenerator
from .label_generation_service import LabelGenerationService
from .presets import (
    apply_preset_by_name,
    get_common_presets,
)
from .signal_generator import SignalGenerator

# 向後互換性のため、__all__を定義
__all__ = [
    "EventDrivenLabelGenerator",
    "ThresholdMethod",
    "BarrierProfile",
    "get_common_presets",
    "apply_preset_by_name",
    "SignalGenerator",
    "LabelGenerationService",
]
