"""
MLターゲット生成モジュール

機械学習モデルの目的変数（ターゲット）を生成するサービスを提供します。
ボラティリティターゲットなど、取引戦略に適したターゲット計算を行います。

主なコンポーネント:
- volatility_target_service.py: ボラティリティ正規化ターゲットの生成
"""

from .volatility_target_service import VolatilityTargetService

__all__ = ["VolatilityTargetService"]
