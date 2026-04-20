"""
インジケーター参照名の共通ヘルパー
"""

from __future__ import annotations

from typing import Any, Optional


def build_indicator_reference_name(
    indicator: object,
    output_index: Optional[int] = None,
) -> str:
    """
    実行時に使う一意なインジケーター参照名を生成する。

    規約:
    - `TYPE`
    - `TYPE_TIMEFRAME`
    - `TYPE_TIMEFRAME_ID[:8]`
    - `TYPE_TIMEFRAME_ID[:8]_OUTPUT_INDEX`
    """
    parts = [str(getattr(indicator, "type", ""))]

    timeframe = getattr(indicator, "timeframe", None)
    if timeframe:
        parts.append(str(timeframe))

    indicator_id = getattr(indicator, "id", None)
    if indicator_id:
        parts.append(str(indicator_id)[:8])

    if output_index is not None:
        parts.append(str(output_index))

    return "_".join(part for part in parts if part)
