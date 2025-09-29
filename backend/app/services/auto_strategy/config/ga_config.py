"""
GA設定モジュール

GAConfigのPydanticモデルを提供します。
"""

from pydantic import BaseModel, Field


class GAConfig(BaseModel):
    """
    遺伝的アルゴリズム設定

    レジーム適応統合のための設定を管理します。
    """

    fallback_symbol: str = Field(default="BTC/USDT", description="フォールバックシンボル")
    fallback_timeframe: str = Field(default="1d", description="フォールバックタイムフレーム")
    fallback_start_date: str = Field(default="2024-01-01", description="フォールバック開始日")
    fallback_end_date: str = Field(default="2024-04-09", description="フォールバック終了日")
    fallback_initial_capital: float = Field(default=100000.0, description="フォールバック初期資本")
    fallback_commission_rate: float = Field(default=0.001, description="フォールバック手数料率")

    regime_adaptation_enabled: bool = Field(
        default=False,
        description="レジーム適応を有効化"
    )