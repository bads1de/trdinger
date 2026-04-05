"""
multi-fidelity 評価の補助ユーティリティ。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from math import ceil
from typing import Any, Dict

import pandas as pd

from app.services.auto_strategy.config.ga import GAConfig


def is_coarse_fidelity(config: Any) -> bool:
    """設定が coarse fidelity かを返す。"""
    return str(getattr(config, "_evaluation_fidelity", "full")) == "coarse"


def is_multi_fidelity_enabled(config: Any) -> bool:
    """multi-fidelity を有効とみなす条件を返す。"""
    return bool(getattr(config, "enable_multi_fidelity_evaluation", False))


def build_coarse_ga_config(config: GAConfig) -> GAConfig:
    """粗評価用に GA 設定を縮退させたコピーを返す。"""
    coarse = deepcopy(config)
    coarse.enable_walk_forward = False
    coarse.enable_purged_kfold = False
    if float(getattr(coarse, "oos_split_ratio", 0.0) or 0.0) <= 0.0:
        coarse.oos_split_ratio = float(
            getattr(coarse, "multi_fidelity_oos_ratio", 0.2) or 0.2
        )
    setattr(coarse, "_evaluation_fidelity", "coarse")
    return coarse


def adjust_backtest_config_for_fidelity(
    backtest_config: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """coarse fidelity の場合だけ評価期間を末尾側へ圧縮する。"""
    adjusted = backtest_config.copy()
    if not is_coarse_fidelity(config):
        return adjusted

    start_date = adjusted.get("start_date")
    end_date = adjusted.get("end_date")
    if not start_date or not end_date:
        return adjusted

    try:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
    except Exception:
        return adjusted

    if start_ts >= end_ts:
        return adjusted

    window_ratio = float(getattr(config, "multi_fidelity_window_ratio", 0.3) or 0.3)
    total_duration = end_ts - start_ts
    coarse_duration = total_duration * window_ratio
    if coarse_duration <= timedelta(0):
        return adjusted

    coarse_start = end_ts - coarse_duration
    adjusted["start_date"] = _format_timestamp_like_input(coarse_start, start_date)
    adjusted["end_date"] = _format_timestamp_like_input(end_ts, end_date)
    return adjusted


def get_multi_fidelity_candidate_limit(population_size: int, config: Any) -> int:
    """フル評価へ昇格する候補数を返す。"""
    if population_size <= 0:
        return 0

    ratio = float(getattr(config, "multi_fidelity_candidate_ratio", 0.25) or 0.25)
    min_candidates = int(getattr(config, "multi_fidelity_min_candidates", 3) or 3)
    return min(population_size, max(min_candidates, int(ceil(population_size * ratio))))


def _format_timestamp_like_input(value: pd.Timestamp, source: Any) -> Any:
    """入力形式に寄せて日時を書き戻す。"""
    if isinstance(source, str):
        if "T" in source:
            return value.isoformat(sep="T")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value.to_pydatetime()
