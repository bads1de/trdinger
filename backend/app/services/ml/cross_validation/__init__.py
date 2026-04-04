"""
交差検証モジュール

時系列データに適した交差検証手法を提供します。
データリーケージを防ぐためのパージ（除外期間）設定もサポートします。

主なコンポーネント:
- purged_kfold.py: パージ付きK分割交差検証（時系列対応）
- factory.py: 時間軸に応じたCV分割器のファクトリー
"""

from .factory import (
    create_temporal_cv_splitter,
    get_t1_series,
    infer_timeframe,
)
from .purged_kfold import PurgedKFold

__all__ = [
    "PurgedKFold",
    "create_temporal_cv_splitter",
    "get_t1_series",
    "infer_timeframe",
]
