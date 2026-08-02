"""
ラベル生成モジュール（二値分類 / メタラベリング専用）

エントリーシグナルの有効性を判定するための二値ラベル（0/1）を生成します。
Triple Barrier MethodやTrend Scanningを用いて、ダマシ（False Signal）を検出します。

主要コンポーネント:
- SignalGenerator: ブレイクアウトなどのイベントを検出
- LabelGenerationService: イベントベースのメタラベリングをサポート
"""

from .event_driven import BarrierProfile, EventDrivenLabelGenerator
from .label_cache import LabelCache, ThresholdMethod
from .presets import (
    apply_preset_by_name,
    get_common_presets,
)
from .signal_generator import SignalGenerator

# LabelGenerationService は循環依存回避のため遅延インポート
_ATTRIBUTE_EXPORTS = {
    "LabelGenerationService": ".label_generation_service",
}

# パブリックAPIを明示的に定義
__all__ = [
    "EventDrivenLabelGenerator",
    "ThresholdMethod",
    "BarrierProfile",
    "get_common_presets",
    "apply_preset_by_name",
    "SignalGenerator",
    "LabelCache",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
