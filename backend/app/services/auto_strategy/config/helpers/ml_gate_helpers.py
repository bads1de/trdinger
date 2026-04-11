"""
ML gate 設定ヘルパー関数

volatility gate 設定の正規化・解決ヘルパーを提供します。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class MLGateSettings:
    """ML gate の有効状態とモデルパスを表す正規化済み設定。"""

    enabled: bool
    model_path: Optional[str]


def _read_value(source: Any, key: str) -> Any:
    """dict / オブジェクトのどちらからでも値を取得する。"""
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _resolve_model_path(*candidates: Any) -> Optional[str]:
    """最初に見つかった有効なモデルパスを返す。"""
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        return str(candidate)
    return None


def resolve_ml_gate_settings(source: Any) -> MLGateSettings:
    """volatility gate 設定を共通形に解決する。"""
    gate_enabled = bool(_read_value(source, "volatility_gate_enabled"))
    model_path = _resolve_model_path(
        _read_value(source, "volatility_model_path"),
    )

    # hybrid_configからも読み取り（優先）
    hybrid_config = _read_value(source, "hybrid_config")
    if hybrid_config is not None:
        gate_enabled = gate_enabled or bool(
            _read_value(hybrid_config, "volatility_gate_enabled")
        )
        model_path = model_path or _resolve_model_path(
            _read_value(hybrid_config, "volatility_model_path"),
        )

    return MLGateSettings(enabled=gate_enabled, model_path=model_path)


def normalize_ml_gate_fields(source: Any) -> dict[str, Optional[str] | bool]:
    """volatility gate 設定を正規化する。"""
    settings = resolve_ml_gate_settings(source)
    return {
        "volatility_gate_enabled": settings.enabled,
        "volatility_model_path": settings.model_path,
    }
