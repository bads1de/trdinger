"""
Robustness regime window 設定ヘルパー関数

regime window の正規化・検証ヘルパーを提供します。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.utils.datetime_utils import (
    normalize_datetimes_for_comparison,
    parse_datetime_optional,
)


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
