from __future__ import annotations

from typing import Any, Dict

MANIFEST_PRICE_TRANSFORM: Dict[str, Dict[str, Any]] = {
    "KAMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "price_transform",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.kama",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 30,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "KAMA期間",
                }
            },
            "pandas_function": "kama",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 30},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": {
                "aggressive": {"efficiency_ratio": 2.5},
                "conservative": {"efficiency_ratio": 10.0},
                "normal": {"efficiency_ratio": 5.0},
            },
            "type": "price_transform",
        },
    }
}



