"""
列挙型定義モジュール

ラベル生成に関連する列挙型を定義します。
"""

from enum import Enum


class ThresholdMethod(Enum):
    """閾値計算方法"""

    TRIPLE_BARRIER = "triple_barrier"  # Triple Barrier Method (利確/損切り/時間切れ)
    TREND_SCANNING = "trend_scanning"  # Trend Scanning Method (t値によるトレンド判定)



