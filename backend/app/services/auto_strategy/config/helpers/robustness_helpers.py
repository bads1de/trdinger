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
    """
    正規化済み robustness regime window

    robustness検証用の期間設定を正規化して保持するデータクラスです。

    Attributes:
        name: 期間名
        start_date: 開始日
        end_date: 終了日
    """

    name: str
    start_date: str
    end_date: str

    @property
    def signature(self) -> tuple[str, str, str]:
        """
        キャッシュキー向けの安定したシグネチャを返す

        Returns:
            tuple[str, str, str]: (name, start_date, end_date)のタプル
        """
        return self.name, self.start_date, self.end_date


def _coerce_window_text(window: Mapping[str, Any], key: str) -> str:
    """
    window の値を文字列化して前後空白を除去する

    Args:
        window: window設定辞書
        key: 取得するキー名

    Returns:
        str: 文字列化された値（前後空白除去済み）
    """
    return str(window.get(key, "") or "").strip()


def normalize_robustness_regime_window(
    window: Any,
) -> Optional[RobustnessRegimeWindow]:
    """
    regime window をシナリオ生成向けに正規化する

    window設定を正規化してRobustnessRegimeWindowオブジェクトに変換します。

    Args:
        window: window設定（辞書形式）

    Returns:
        Optional[RobustnessRegimeWindow]: 正規化されたwindowオブジェクト
                                       不正な形式の場合はNone
    """
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
    """
    regime window のリストを正規化する

    windowリストを正規化してRobustnessRegimeWindowオブジェクトのリストに変換します。

    Args:
        windows: windowリスト

    Returns:
        list[RobustnessRegimeWindow]: 正規化されたwindowオブジェクトのリスト

    Note:
        - 文字列、バイト列、辞書の場合は空リストを返します
        - Iterableでない場合は空リストを返します
    """
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
    """
    regime window の妥当性を検証する

    window設定の妥当性を検証してエラーメッセージリストを返します。

    Args:
        window: window設定（辞書形式）

    Returns:
        list[str]: エラーメッセージのリスト（妥当な場合は空リスト）

    検証項目:
        - 辞書形式であること
        - nameが存在すること
        - start_dateとend_dateが存在すること
        - 日付形式が正しいこと
        - start_date < end_dateであること
    """
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
