"""
設定ヘルパー関数

ML filter/volatility gate 設定と robustness regime window の正規化・検証ヘルパーを統合。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from app.utils.datetime_utils import (
    normalize_datetimes_for_comparison,
    parse_datetime_optional,
)


# === ML filter / volatility gate 設定 ===


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
    """volatility gate / legacy ML filter 設定を共通形に解決する。"""
    # フラットフィールドから読み取り
    gate_enabled = bool(
        _read_value(source, "volatility_gate_enabled")
        or _read_value(source, "ml_filter_enabled")
    )
    model_path = _resolve_model_path(
        _read_value(source, "volatility_model_path"),
        _read_value(source, "ml_model_path"),
    )

    # hybrid_configからも読み取り（優先）
    hybrid_config = _read_value(source, "hybrid_config")
    if hybrid_config is not None:
        gate_enabled = gate_enabled or bool(
            _read_value(hybrid_config, "volatility_gate_enabled")
            or _read_value(hybrid_config, "ml_filter_enabled")
        )
        model_path = model_path or _resolve_model_path(
            _read_value(hybrid_config, "volatility_model_path"),
            _read_value(hybrid_config, "ml_model_path"),
        )

    return MLGateSettings(enabled=gate_enabled, model_path=model_path)


def normalize_ml_gate_fields(source: Any) -> dict[str, Optional[str] | bool]:
    """互換フィールドを同期済みの辞書へ正規化する。"""
    settings = resolve_ml_gate_settings(source)
    return {
        "volatility_gate_enabled": settings.enabled,
        "ml_filter_enabled": settings.enabled,
        "volatility_model_path": settings.model_path,
        "ml_model_path": settings.model_path,
    }


# === Robustness regime window 設定 ===


@dataclass(frozen=True)
class RobustnessRegimeWindow:
    """正規化済み robustness regime window。"""

    name: str
    start_date: str
    end_date: str

    @property
    def signature(self) -> tuple[str, str, str]:
        """キャッシュキー向けの安定したシグネチャを返す。"""
        return self.name, self.start_date, self.end_date


def _coerce_window_text(window: Mapping[str, Any], key: str) -> str:
    """window の値を文字列化して前後空白を除去する。"""
    return str(window.get(key, "") or "").strip()


def normalize_robustness_regime_window(
    window: Any,
) -> Optional[RobustnessRegimeWindow]:
    """regime window をシナリオ生成向けに正規化する。"""
    if not isinstance(window, Mapping):
        return None

    name = _coerce_window_text(window, "name")
    start_date = _coerce_window_text(window, "start_date")
    end_date = _coerce_window_text(window, "end_date")
    if not name or not start_date or not end_date:
        return None

    return RobustnessRegimeWindow(
        name=name,
        start_date=start_date,
        end_date=end_date,
    )


def normalize_robustness_regime_windows(
    windows: Any,
) -> list[RobustnessRegimeWindow]:
    """regime window のリストを正規化する。"""
    if isinstance(windows, (str, bytes)) or isinstance(windows, Mapping):
        return []
    if not isinstance(windows, Iterable):
        return []

    normalized_windows: list[RobustnessRegimeWindow] = []
    for window in windows:
        normalized = normalize_robustness_regime_window(window)
        if normalized is not None:
            normalized_windows.append(normalized)
    return normalized_windows


def validate_robustness_regime_window(window: Any) -> list[str]:
    """regime window の妥当性を検証する。"""
    if not isinstance(window, Mapping):
        return ["robustness の regime window は辞書である必要があります"]

    name = window.get("name")
    if not name or not isinstance(name, str):
        return ["robustness の regime window は name が必要です"]

    start_date = window.get("start_date")
    end_date = window.get("end_date")
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        return ["robustness の regime window は start_date/end_date が必要です"]

    normalized = normalize_robustness_regime_window(window)
    if normalized is None:
        return ["robustness の regime window の日付形式が不正です"]

    parsed_start = parse_datetime_optional(normalized.start_date)
    parsed_end = parse_datetime_optional(normalized.end_date)
    if parsed_start is None or parsed_end is None:
        return ["robustness の regime window の日付形式が不正です"]

    normalized_start, normalized_end = normalize_datetimes_for_comparison(
        parsed_start,
        parsed_end,
    )
    if normalized_start >= normalized_end:
        return [
            "robustness の regime window は start_date < end_date である必要があります"
        ]

    return []
