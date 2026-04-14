"""
multi-fidelity 評価の補助ユーティリティ。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from math import ceil
from typing import Any, Dict, Mapping

import pandas as pd

from app.services.auto_strategy.config.ga import GAConfig
from app.utils.datetime_utils import parse_datetime_range_optional

from .evaluation_report import _DATETIME_FORMAT


def is_coarse_fidelity(config: Any) -> bool:
    """設定が coarse fidelity かを返す。"""
    return str(getattr(config, "_evaluation_fidelity", "full")) == "coarse"


def _coerce_bool(value: Any) -> bool:
    """Mock などの曖昧な値を安全に bool へ寄せる。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _coerce_float(value: Any, default: float) -> float:
    """float 変換できない値は既定値へフォールバックする。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _get_evaluation_config(source: object) -> object:
    """評価設定のネスト先を取得する。"""
    if isinstance(source, Mapping):
        return source.get("evaluation_config")
    return getattr(source, "evaluation_config", None)


def is_multi_fidelity_enabled(config: Any) -> bool:
    """multi-fidelity を有効とみなす条件を返す。"""
    evaluation_config = _get_evaluation_config(config)
    if evaluation_config is None:
        return False
    return _coerce_bool(
        getattr(evaluation_config, "enable_multi_fidelity_evaluation", False)
    )


def build_coarse_ga_config(config: GAConfig) -> GAConfig:
    """
    GAの初期世代で使用する「粗評価（Coarse Fidelity）」用の設定オブジェクトを構築します。

    目的：
    - 進化の初期段階では個体の大まかな傾向を把握すれば十分であるため、
      計算負荷の高い WFA（Walk-Forward Analysis）やパージング付き交差検証を無効化し、
      全体の計算時間を大幅に短縮します。

    Args:
        config (GAConfig): 元の精密なGA設定。

    Returns:
        GAConfig: 計算負荷を下げた縮退版の設定。`_evaluation_fidelity` 属性に "coarse" が設定されます。
    """
    coarse = deepcopy(config)
    coarse.evaluation_config.enable_walk_forward = False
    coarse.enable_purged_kfold = False
    oos_split_ratio = _coerce_float(
        getattr(coarse.evaluation_config, "oos_split_ratio", 0.0),
        0.0,
    )
    if oos_split_ratio <= 0.0:
        coarse.evaluation_config.oos_split_ratio = _coerce_float(
            getattr(coarse.evaluation_config, "multi_fidelity_oos_ratio", 0.2),
            0.2,
        )
    setattr(coarse, "_evaluation_fidelity", "coarse")
    return coarse


def adjust_backtest_config_for_fidelity(
    backtest_config: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """
    評価精度設定に基づいて、バックテストの実行範囲（期間）を調整します。

    Coarse Fidelity（粗評価）が有効な場合：
    - 全期間のバックテストではなく、直近の `multi_fidelity_window_ratio`（デフォルト30%）
      の期間のみを抽出して評価を行います。これにより、シミュレーション時間を劇的に短縮できます。

    Args:
        backtest_config (Dict[str, Any]): 元のバックテスト設定（銘柄、全期間等）。
        config (Any): 現在のGA設定。粗評価フラグを確認するために使用。

    Returns:
        Dict[str, Any]: 必要に応じて期間が短縮されたバックテスト設定。
    """
    adjusted = backtest_config.copy()
    if not is_coarse_fidelity(config):
        return adjusted

    evaluation_config = _get_evaluation_config(config)
    if evaluation_config is None:
        return adjusted

    start_date = adjusted.get("start_date")
    end_date = adjusted.get("end_date")
    if not start_date or not end_date:
        return adjusted

    parsed_range = parse_datetime_range_optional(start_date, end_date)
    if parsed_range is None:
        return adjusted

    start_dt, end_dt = parsed_range
    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)

    window_ratio = _coerce_float(
        getattr(evaluation_config, "multi_fidelity_window_ratio", 0.3),
        0.3,
    )
    total_duration = end_ts - start_ts
    coarse_duration = total_duration * window_ratio
    if coarse_duration <= timedelta(0):
        return adjusted

    coarse_start = end_ts - coarse_duration
    adjusted["start_date"] = _format_timestamp_like_input(coarse_start, start_date)  # type: ignore[arg-type]
    adjusted["end_date"] = _format_timestamp_like_input(end_ts, end_date)  # type: ignore[arg-type]
    return adjusted


def get_multi_fidelity_candidate_limit(population_size: int, config: Any) -> int:
    """フル評価へ昇格する候補数を返す。"""
    if population_size <= 0:
        return 0

    evaluation_config = _get_evaluation_config(config)
    if evaluation_config is None:
        return 0

    ratio = _coerce_float(
        getattr(evaluation_config, "multi_fidelity_candidate_ratio", 0.25), 0.25
    )
    try:
        min_candidates = int(
            getattr(evaluation_config, "multi_fidelity_min_candidates", 3) or 3
        )
    except (TypeError, ValueError):
        min_candidates = 3
    return min(population_size, max(min_candidates, int(ceil(population_size * ratio))))


def _format_timestamp_like_input(value: pd.Timestamp, source: object) -> object:
    """入力形式に寄せて日時を書き戻す。"""
    if isinstance(source, str):
        if "T" in source:
            return value.isoformat(sep="T")
        return value.strftime(_DATETIME_FORMAT)
    return value.to_pydatetime()
