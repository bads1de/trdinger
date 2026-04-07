"""バックテスト実行設定を組み立てる共通ヘルパー。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

_MISSING = object()


def _copy_config_source(source: Any) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    return {}


def _get_value(source: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(source, Mapping):
        if key in source and source[key] is not None:
            return source[key]
    else:
        value = getattr(source, key, _MISSING)
        if value is not _MISSING and value is not None:
            return value

    if default is not _MISSING:
        return default

    if isinstance(source, Mapping):
        raise KeyError(key)
    raise AttributeError(f"{type(source).__name__} に {key} がありません")


def _get_optional_value(source: Any, key: str) -> Any:
    try:
        return _get_value(source, key)
    except (AttributeError, KeyError):
        return _MISSING


def _normalize_strategy_config(strategy_config: Any) -> Dict[str, Any]:
    if hasattr(strategy_config, "model_dump"):
        return strategy_config.model_dump()
    if isinstance(strategy_config, Mapping):
        return dict(strategy_config)
    raise TypeError(
        "strategy_config は mapping または model_dump 対応オブジェクトである必要があります"
    )


def ensure_backtest_defaults(
    backtest_config: Mapping[str, Any], defaults: Mapping[str, Any]
) -> Dict[str, Any]:
    """不足しているバックテスト設定をデフォルト値で補完する。"""
    working = dict(backtest_config)
    for key, value in defaults.items():
        if value is None:
            continue
        if key not in working or working[key] is None:
            working[key] = value
    return working


def build_execution_config(
    source: Any,
    *,
    strategy_name: Optional[str] = None,
    strategy_config: Any = None,
    defaults: Optional[Mapping[str, Any]] = None,
    default_slippage: float = 0.0,
    default_leverage: float = 1.0,
) -> Dict[str, Any]:
    """
    dict / Pydantic モデルからバックテスト実行設定を組み立てる。

    dict の場合は元のキーを保持したまま、必要な値を補完する。
    """
    working = _copy_config_source(source)
    if defaults:
        working = ensure_backtest_defaults(working, defaults)

    resolved_strategy_name = (
        strategy_name
        if strategy_name is not None
        else _get_value(source, "strategy_name")
    )
    working["strategy_name"] = resolved_strategy_name

    resolved_strategy_config = (
        strategy_config
        if strategy_config is not None
        else _get_value(source, "strategy_config")
    )
    working["strategy_config"] = _normalize_strategy_config(resolved_strategy_config)

    for key in (
        "symbol",
        "timeframe",
        "start_date",
        "end_date",
        "initial_capital",
        "commission_rate",
    ):
        value = working.get(key, _MISSING)
        if value is _MISSING or value is None:
            value = _get_optional_value(source, key)
        if (value is _MISSING or value is None) and defaults:
            value = defaults.get(key, _MISSING)
        if value is not _MISSING and value is not None:
            working[key] = value

    slippage_default = (
        defaults.get("slippage", default_slippage) if defaults else default_slippage
    )
    leverage_default = (
        defaults.get("leverage", default_leverage) if defaults else default_leverage
    )

    if "slippage" not in working or working["slippage"] is None:
        slippage_value = _get_optional_value(source, "slippage")
        if slippage_value is _MISSING or slippage_value is None:
            slippage_value = slippage_default
        if slippage_value is not _MISSING and slippage_value is not None:
            working["slippage"] = slippage_value

    if "leverage" not in working or working["leverage"] is None:
        leverage_value = _get_optional_value(source, "leverage")
        if leverage_value is _MISSING or leverage_value is None:
            leverage_value = leverage_default
        if leverage_value is not _MISSING and leverage_value is not None:
            working["leverage"] = leverage_value

    return working
