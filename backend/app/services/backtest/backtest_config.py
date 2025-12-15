"""
バックテスト設定スキーマ

バックテスト実行に必要な設定データの構造を定義します。
"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.config.unified_config import unified_config


class GeneratedGAParameters(BaseModel):
    """GA生成戦略のパラメータ"""

    strategy_gene: Dict[str, Any]  # 将来的にはStrategyGeneモデルそのものに置き換える
    ml_filter_enabled: bool = False
    ml_model_path: Optional[str] = None
    ml_predictor: Optional[Any] = None  # MLモデルインスタンス
    ml_filter_threshold: float = 0.1
    minute_data: Optional[Any] = None  # DataFrameなどはPydanticで検証しにくいためAny


class StrategyConfig(BaseModel):
    """戦略設定"""

    strategy_type: Literal["GENERATED_GA", "MANUAL"]
    parameters: Union[GeneratedGAParameters, Dict[str, Any]]

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, v, info):
        """strategy_typeに基づいてパラメータを適切なモデルに変換"""
        if isinstance(v, dict) and info.data.get("strategy_type") == "GENERATED_GA":
            return GeneratedGAParameters(**v)
        return v


class BacktestConfig(BaseModel):
    """
    バックテスト実行設定

    これまで辞書で受け渡されていた設定を型安全に定義します。
    """

    strategy_name: str = Field(..., min_length=1)
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(..., gt=0)
    commission_rate: float = Field(
        default=unified_config.backtest.default_commission_rate, ge=0, le=1
    )
    slippage: float = Field(0.0, ge=0)
    strategy_config: StrategyConfig

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        """日付文字列をdatetimeオブジェクトに変換"""
        if isinstance(v, str):
            # 'Z'付きのISOフォーマットなどの対応
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                # pandasのto_datetimeのような柔軟なパースが必要な場合
                import pandas as pd

                return pd.to_datetime(v).to_pydatetime()
        return v



