"""
バックテスト設定スキーマ

バックテスト実行に必要な設定データの構造を定義します。
"""

from datetime import datetime
from math import isclose
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config.constants import SUPPORTED_TIMEFRAMES
from app.services.backtest.shared import (
    current_datetime_like,
    normalize_datetimes_for_comparison,
    parse_datetime_value,
)

VALID_TIMEFRAMES = SUPPORTED_TIMEFRAMES


class BacktestRunConfigValidationError(Exception):
    """バックテスト設定検証エラー"""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class GeneratedGAParameters(BaseModel):
    """GA生成戦略のパラメータ"""

    strategy_gene: Dict[str, Any]  # 将来的にはStrategyGeneモデルそのものに置き換える
    volatility_gate_enabled: bool = False
    volatility_model_path: Optional[str] = None
    ml_predictor: Optional[Any] = None  # MLモデルインスタンス
    evaluation_start: Optional[Any] = None
    minute_data: Optional[Any] = None  # DataFrameなどはPydanticで検証しにくいためAny
    enable_early_termination: bool = False
    early_termination_max_drawdown: Optional[float] = None
    early_termination_min_trades: Optional[int] = None
    early_termination_min_trade_check_progress: float = 0.5
    early_termination_trade_pace_tolerance: float = 0.5
    early_termination_min_expectancy: Optional[float] = None
    early_termination_expectancy_min_trades: int = 5
    early_termination_expectancy_progress: float = 0.6


class StrategyConfig(BaseModel):
    """戦略設定"""

    strategy_type: Literal["GENERATED_GA", "MANUAL"]
    parameters: Union[GeneratedGAParameters, Dict[str, Any]]

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, v, info):
        """strategy_typeに基づいてパラメータを適切なモデルに変換"""
        strategy_type = info.data.get("strategy_type")

        if isinstance(v, dict) and strategy_type == "GENERATED_GA":
            return GeneratedGAParameters(**v)
        return v


class BacktestRunConfig(BaseModel):
    """
    バックテスト実行設定（APIリクエストスキーマ）

    これまで辞書で受け渡されていた設定を型安全に定義します。
    """

    strategy_name: str = Field(..., min_length=1)
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(..., gt=0)
    commission_rate: float = Field(default=0.001, ge=0, le=1)
    spread: float = Field(0.0, ge=0)
    slippage: float = Field(0.0, ge=0)
    leverage: float = Field(1.0, ge=1.0)
    strategy_config: StrategyConfig

    @model_validator(mode="before")
    @classmethod
    def normalize_execution_cost_aliases(cls, data: Any) -> Any:
        """spread/slippage の互換キーを同期する。"""
        if not isinstance(data, dict):
            return data

        working = dict(data)
        spread = working.get("spread")
        slippage = working.get("slippage")

        if spread is not None and slippage is not None:
            try:
                spread_value = float(spread)
                slippage_value = float(slippage)
            except (TypeError, ValueError):
                return working
            if not isclose(spread_value, slippage_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("spread と slippage は同じ値である必要があります")
            return working

        if spread is None and slippage is not None:
            working["spread"] = slippage
        elif slippage is None and spread is not None:
            working["slippage"] = spread

        return working

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """timeframeの値を検証"""
        if v not in VALID_TIMEFRAMES:
            raise ValueError(
                f"timeframeは {VALID_TIMEFRAMES} のいずれかである必要があります"
            )
        return v

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        """日付文字列をdatetimeオブジェクトに変換"""
        if isinstance(v, str):
            return parse_datetime_value(v)
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "BacktestRunConfig":
        """日付の整合性を検証"""
        errors: List[str] = []

        start_date, end_date = normalize_datetimes_for_comparison(
            self.start_date, self.end_date
        )

        if start_date >= end_date:
            errors.append("start_dateはend_dateより前である必要があります")

        now = current_datetime_like(end_date)
        if end_date > now:
            errors.append("end_dateは現在時刻より前である必要があります")

        if errors:
            raise BacktestRunConfigValidationError(
                f"バックテスト設定が無効です: {', '.join(errors)}", errors
            )

        return self
