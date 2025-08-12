"""
指標パラメータ正規化ユーティリティ

- period -> length 変換
- 必須 length のデフォルト補完
"""
from __future__ import annotations

import inspect
from typing import Any, Dict

from .config.indicator_config import IndicatorConfig


def normalize_params(indicator_type: str, params: Dict[str, Any], config: IndicatorConfig) -> Dict[str, Any]:
    converted_params: Dict[str, Any] = {}
    # period -> length 変換（例外指標はここで外すことも可能）
    period_based = {"MA", "MAVP", "MAX", "MIN", "SUM", "BETA", "CORREL", "LINEARREG", "LINEARREG_SLOPE", "STDDEV", "VAR"}
    for key, value in params.items():
        if key == "period" and indicator_type not in period_based:
            converted_params["length"] = value
        else:
            converted_params[key] = value

    # length 必須のアダプタにデフォルト補完
    try:
        sig = inspect.signature(config.adapter_function)
        if "length" in sig.parameters and "length" not in converted_params:
            default_len = params.get("period")
            if default_len is None and config.parameters:
                if "period" in config.parameters:
                    default_len = config.parameters["period"].default_value
                elif "length" in config.parameters:
                    default_len = config.parameters["length"].default_value
            converted_params["length"] = default_len if default_len is not None else 14
    except Exception:
        pass

    return converted_params

