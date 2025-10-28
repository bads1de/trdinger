from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Optional

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
    IndicatorConfigRegistry,
    indicator_registry,
)


logger = logging.getLogger(__name__)


def _resolve_callable(path: Optional[str]) -> Optional[Callable[..., Any]]:
    if not path:
        return None

    parts = path.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        attr = module
        for attribute_name in parts[index:]:
            try:
                attr = getattr(attr, attribute_name)
            except AttributeError:
                logger.warning(
                    "Failed to resolve attribute '%s' on module '%s' for path '%s'",
                    attribute_name,
                    module_name,
                    path,
                )
                return None
        return attr

    raise ImportError(f"Callable path could not be resolved: {path}")


def _min_length_stoch(params: Dict[str, Any]) -> int:
    return (
        params.get("k_length", 14)
        + params.get("d_length", 3)
        + params.get("smooth_k", 3)
    )


def _min_length_generic_length(params: Dict[str, Any], default: int) -> int:
    return max(2, params.get("length", default))


def _min_length_sma(params: Dict[str, Any]) -> int:
    return _min_length_generic_length(params, 20)


def _min_length_dpo(params: Dict[str, Any]) -> int:
    return _min_length_generic_length(params, 20)


def _min_length_ema(params: Dict[str, Any]) -> int:
    return _min_length_generic_length(params, 20)


def _min_length_wma(params: Dict[str, Any]) -> int:
    return _min_length_generic_length(params, 20)


def _min_length_rsi(params: Dict[str, Any]) -> int:
    return max(2, params.get("length", 14))


def _min_length_macd(params: Dict[str, Any]) -> int:
    return params.get("slow", 26) + params.get("signal", 9) + 5


def _min_length_bb(params: Dict[str, Any]) -> int:
    return params.get("length", 20)


def _min_length_supertrend(params: Dict[str, Any]) -> int:
    return params.get("length", 10) + 10


def _min_length_tema(params: Dict[str, Any]) -> int:
    return max(3, params.get("length", 14) // 2)


_MIN_LENGTH_FUNCTIONS = {
    "BB": _min_length_bb,
    "DPO": _min_length_dpo,
    "EMA": _min_length_ema,
    "MACD": _min_length_macd,
    "RSI": _min_length_rsi,
    "SMA": _min_length_sma,
    "STOCH": _min_length_stoch,
    "SUPERTREND": _min_length_supertrend,
    "TEMA": _min_length_tema,
    "WMA": _min_length_wma,
}


MANIFEST: Dict[str, Dict[str, Any]] = {
    "ACCBANDS": {
        "config": {
            "result_type": "complex",
            "scale_type": "price_absolute",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.accbands",
            "required_data": ["high", "low", "close"],
            "output_names": ["ACCBANDS_Upper", "ACCBANDS_Middle", "ACCBANDS_Lower"],
            "default_output": "ACCBANDS_Middle",
            "aliases": None,
            "param_map": {"length": "length"},
            "parameters": {},
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}_lower",
                "short": "close < {left_operand}_upper",
            },
            "scale_type": "price_absolute",
            "thresholds": {
                "aggressive": {"acceleration": 0.005},
                "conservative": {"acceleration": 0.02},
                "normal": {"acceleration": 0.01},
            },
            "type": "volatility",
        },
    },
    "AD": {
        "config": {
            "result_type": "single",
            "scale_type": "volume",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.ad",
            "required_data": ["high", "low", "close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {},
            "parameters": {},
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "volume",
        },
    },
    "ADOSC": {
        "config": {
            "result_type": "single",
            "scale_type": "volume",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.adosc",
            "required_data": ["high", "low", "close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"fast": "fast", "slow": "slow"},
            "parameters": {
                "fast": {
                    "default_value": 3,
                    "min_value": 1,
                    "max_value": 20,
                    "description": None,
                },
                "slow": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 40,
                    "description": None,
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"fast": 3, "slow": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_gt": -10000, "short_lt": 10000},
                "conservative": {"long_gt": -2000, "short_lt": 2000},
                "normal": {"long_gt": -5000, "short_lt": 5000},
            },
            "type": "volume",
        },
    },
    "ADX": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.adx",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"period": "length"},
            "parameters": {
                "period": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "ADX計算期間",
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"period": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"trend_min": 18},
                "conservative": {"trend_min": 30},
                "normal": {"trend_min": 25},
            },
            "type": "momentum",
        },
    },
    "ALMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.alma",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "length": "length",
                "sigma": "sigma",
                "distribution_offset": "distribution_offset",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "ALMA計算期間",
                },
                "sigma": {
                    "default_value": 6.0,
                    "min_value": 0.1,
                    "max_value": 10.0,
                    "description": "ガウス分布のシグマ",
                },
                "distribution_offset": {
                    "default_value": 0.85,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "description": "ガウス分布のオフセット",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "出力シフト量",
                },
            },
            "pandas_function": "alma",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {
                "length": 10,
                "sigma": 6.0,
                "distribution_offset": 0.85,
                "offset": 0,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "BIAS": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_plus_minus_100",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.bias",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length", "ma_type": "ma_type", "offset": "offset"},
            "parameters": {
                "length": {
                    "default_value": 26,
                    "min_value": 5,
                    "max_value": 100,
                    "description": "BIAS計算期間",
                },
                "ma_type": {
                    "default_value": "sma",
                    "min_value": None,
                    "max_value": None,
                    "description": "移動平均の種類 (sma, ema, wma, hma, zlma)",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "出力シフト量",
                }
            },
            "pandas_function": "bias",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 26, "ma_type": "sma", "offset": 0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand} * 1.05",
                "short": "close < {left_operand} * 0.95",
            },
            "scale_type": "oscillator_plus_minus_100",
            "thresholds": {"all": {"long_lt": -5, "short_gt": 5}},
            "type": "trend",
        },
    },
    "ATR": {
        "config": {
            "result_type": "single",
            "scale_type": "price_absolute",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.atr",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "ATR計算期間",
                }
            },
            "pandas_function": "atr",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}_current + {multiplier}",
                "short": "close < {left_operand}_current - {multiplier}",
            },
            "scale_type": "price_absolute",
            "thresholds": {
                "aggressive": {"multiplier": 0.5},
                "conservative": {"multiplier": 1.5},
                "normal": {"multiplier": 1.0},
            },
            "type": "volatility",
        },
    },
    "BB": {
        "config": {
            "result_type": "complex",
            "scale_type": "price_ratio",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.bbands",
            "required_data": ["close"],
            "output_names": ["BB_Upper", "BB_Middle", "BB_Lower"],
            "default_output": "BB_Middle",
            "aliases": None,
            "param_map": {"close": "data", "length": "length", "std": "std"},
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 5,
                    "max_value": 100,
                    "description": "ボリンジャーバンド期間",
                },
                "std": {
                    "default_value": 2.0,
                    "min_value": 0.5,
                    "max_value": 5.0,
                    "description": "標準偏差倍数",
                },
            },
            "pandas_function": "bbands",
            "data_column": "Close",
            "data_columns": None,
            "returns": "multiple",
            "return_cols": ["BBL", "BBM", "BBU"],
            "multi_column": False,
            "default_values": {"length": 20, "std": 2.0},
            "min_length_func": "BB",
        },
        "yaml": {
            "components": ["upper", "middle", "lower"],
            "conditions": {
                "long": "close < {left_operand}_lower",
                "short": "close > {left_operand}_upper",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "volatility",
        },
    },
    "CCI": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_plus_minus_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.cci",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"period": "length"},
            "parameters": {
                "period": {
                    "default_value": 14,
                    "min_value": 5,
                    "max_value": 50,
                    "description": "CCI計算期間",
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"period": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {long_threshold}",
                "short": "{left_operand} < {short_threshold}",
            },
            "scale_type": "oscillator_plus_minus_100",
            "thresholds": {
                "aggressive": {"abs_limit": 100, "long_lt": -100},
                "conservative": {"abs_limit": 100, "long_lt": -100},
                "normal": {"abs_limit": 100, "long_lt": -100},
            },
            "type": "momentum",
        },
    },
    "CMF": {
        "config": {
            "result_type": "single",
            "scale_type": "price_absolute",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.cmf",
            "required_data": ["high", "low", "close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"length": "length"},
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 100,
                    "description": None,
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 20},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}_mid",
                "short": "close < {left_operand}_mid",
            },
            "scale_type": "price_absolute",
            "thresholds": {
                "aggressive": {"multiplier": 1.5},
                "conservative": {"multiplier": 2.5},
                "normal": {"multiplier": 2.0},
            },
            "type": "volume",
        },
    },
    "CMO": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_plus_minus_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.cmo",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "CMO計算期間",
                }
            },
            "pandas_function": "cmo",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_plus_minus_100",
            "thresholds": {"all": {"long_gt": 50, "short_lt": -50}},
            "type": "momentum",
        },
    },
    "DEMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.dema",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "二重指数移動平均期間",
                }
            },
            "pandas_function": "dema",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "DONCHIAN": {
        "config": {
            "result_type": "complex",
            "scale_type": "price_absolute",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.donchian",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
            },
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "Donchian Channels期間",
                }
            },
            "pandas_function": "donchian",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "multiple",
            "return_cols": ["DC_LB", "DC_MB", "DC_UB"],
            "multi_column": True,
            "default_values": {"length": 20},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}_mid",
                "short": "close < {left_operand}_mid",
            },
            "scale_type": "price_absolute",
            "thresholds": {
                "aggressive": {"period": 10},
                "combo": {
                    "condition": "(close - lower) / (upper - lower) >= primary or (close - lower) / (upper - lower) <= secondary",
                    "primary": 0.9,
                    "secondary": 0.1,
                },
                "conservative": {"period": 30},
                "normal": {"period": 20},
                "range": {"break_out_threshold": 0.8, "consolidation_threshold": 0.3},
            },
            "type": "volatility",
        },
    },
    "DPO": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.dpo",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "length": "length",
                "centered": "centered",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "DPO計算期間",
                },
                "centered": {
                    "default_value": True,
                    "min_value": None,
                    "max_value": None,
                    "description": "中心化するかどうか",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -50,
                    "max_value": 50,
                    "description": "出力オフセット",
                },
            },
            "pandas_function": "dpo",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 20, "centered": True, "offset": 0},
            "min_length_func": "DPO",
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"normal": {"threshold": 0}},
            "type": "trend",
        },
    },
    "EFI": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.efi",
            "required_data": ["close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"length": "length"},
            "parameters": {
                "length": {
                    "default_value": 13,
                    "min_value": 2,
                    "max_value": 200,
                    "description": None,
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 13},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"all": {"long_gt": 0, "short_lt": 0}},
            "type": "volume",
        },
    },
    "EOM": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.eom",
            "required_data": ["high", "low", "close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "length": "length",
                "divisor": "divisor",
                "drift": "drift",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "EOM計算期間",
                },
                "divisor": {
                    "default_value": 100000000.0,
                    "min_value": None,
                    "max_value": None,
                    "description": "正規化に使用する除数",
                },
                "drift": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "変化を測るドリフト",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "出力オフセット",
                },
            },
            "pandas_function": "eom",
            "data_column": None,
            "data_columns": ["High", "Low", "Close", "Volume"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {
                "length": 14,
                "divisor": 100000000.0,
                "drift": 1,
                "offset": 0,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"normal": {"threshold": 0}},
            "type": "volume",
        },
    },
    "FISHER": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.fisher",
            "required_data": ["high", "low"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "length": "length",
                "signal": "signal",
            },
            "parameters": {
                "length": {
                    "default_value": 9,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "Fisher期間",
                },
                "signal": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 20,
                    "description": "シグナル期間",
                },
            },
            "pandas_function": "fisher",
            "data_column": None,
            "data_columns": ["High", "Low"],
            "returns": "multiple",
            "return_cols": ["FISHERT", "FISHERTs"],
            "multi_column": True,
            "default_values": {"length": 9, "signal": 1},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > 0",
                "short": "{left_operand}_0 < 0",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "momentum",
        },
    },
    "FRAMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.original.OriginalIndicators.frama",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "close", "length": "length", "slow": "slow"},
            "parameters": {
                "length": {
                    "default_value": 16,
                    "min_value": 4,
                    "max_value": 200,
                    "description": "FRAMA計算ウィンドウ",
                },
                "slow": {
                    "default_value": 200,
                    "min_value": 1,
                    "max_value": 500,
                    "description": "最大スムージング期間",
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 16, "slow": 200},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "SUPER_SMOOTHER": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.original.OriginalIndicators.super_smoother",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "close", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "Super Smoother期間",
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "ELDER_RAY": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "original",
            "adapter_function": "app.services.indicators.technical_indicators.original.OriginalIndicators.elder_ray",
            "required_data": ["high", "low", "close"],
            "output_names": ["ELDER_RAY_Bull", "ELDER_RAY_Bear"],
            "default_output": "ELDER_RAY_Bull",
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
                "ema_length": "ema_length"
            },
            "parameters": {
                "length": {
                    "default_value": 13,
                    "min_value": 5,
                    "max_value": 50,
                    "description": "Elder Ray計算期間"
                },
                "ema_length": {
                    "default_value": 16,
                    "min_value": 5,
                    "max_value": 50,
                    "description": "EMA計算期間"
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "multiple",
            "return_cols": ["BULL", "BEAR"],
            "multi_column": True,
            "default_values": {
                "length": 13,
                "ema_length": 16
            },
            "min_length_func": None
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > {threshold}",
                "short": "{left_operand}_1 < {threshold}"
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {
                    "long_gt": 0.5,
                    "short_lt": -0.5
                },
                "conservative": {
                    "long_gt": 1.5,
                    "short_lt": -1.5
                },
                "normal": {
                    "long_gt": 1.0,
                    "short_lt": -1.0
                }
            },
            "type": "momentum"
        }
    },
    "ADAPTIVE_ENTROPY": {
        "config": {
            "result_type": "complex",
            "scale_type": "oscillator_plus_minus_100",
            "category": "original",
            "adapter_function": "app.services.indicators.technical_indicators.original.OriginalIndicators.adaptive_entropy",
            "required_data": ["close"],
            "output_names": ["ADAPTIVE_ENTROPY_OSC", "ADAPTIVE_ENTROPY_SIGNAL", "ADAPTIVE_ENTROPY_RATIO"],
            "default_output": "ADAPTIVE_ENTROPY_OSC",
            "aliases": None,
            "param_map": {
                "close": "close",
                "short_length": "short_length",
                "long_length": "long_length",
                "signal_length": "signal_length"
            },
            "parameters": {
                "short_length": {
                    "default_value": 14,
                    "min_value": 5,
                    "max_value": 50,
                    "description": "短期エントロピー計算期間"
                },
                "long_length": {
                    "default_value": 28,
                    "min_value": 10,
                    "max_value": 100,
                    "description": "長期エントロピー計算期間"
                },
                "signal_length": {
                    "default_value": 5,
                    "min_value": 2,
                    "max_value": 20,
                    "description": "信号線平滑化期間"
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "multiple",
            "return_cols": ["OSC", "SIGNAL", "RATIO"],
            "multi_column": False,
            "default_values": {
                "short_length": 14,
                "long_length": 28,
                "signal_length": 5
            },
            "min_length_func": None
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_1 > {left_operand}_0",
                "short": "{left_operand}_1 < {left_operand}_0"
            },
            "scale_type": "oscillator_zero_centered",
            "thresholds": {
                "aggressive": {
                    "long_gt": 0.2,
                    "short_lt": -0.2
                },
                "conservative": {
                    "long_gt": 0.8,
                    "short_lt": -0.8
                },
                "normal": {
                    "long_gt": 0.5,
                    "short_lt": -0.5
                }
            },
            "type": "original"
        }
    },
    "QUANTUM_FLOW": {
        "config": {
            "result_type": "complex",
            "scale_type": "oscillator_plus_minus_100",
            "category": "original",
            "adapter_function": "app.services.indicators.technical_indicators.original.OriginalIndicators.quantum_flow",
            "required_data": ["close", "high", "low", "volume"],
            "output_names": ["QUANTUM_FLOW", "QUANTUM_FLOW_SIGNAL"],
            "default_output": "QUANTUM_FLOW",
            "aliases": None,
            "param_map": {
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "length": "length",
                "flow_length": "flow_length"
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 5,
                    "max_value": 50,
                    "description": "ウェーブレット計算期間"
                },
                "flow_length": {
                    "default_value": 9,
                    "min_value": 3,
                    "max_value": 20,
                    "description": "フロースコア計算期間"
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": ["Close", "High", "Low", "Volume"],
            "returns": "multiple",
            "return_cols": ["FLOW", "SIGNAL"],
            "multi_column": False,
            "default_values": {
                "length": 14,
                "flow_length": 9
            },
            "min_length_func": None
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_1 > {left_operand}_0",
                "short": "{left_operand}_1 < {left_operand}_0"
            },
            "scale_type": "oscillator_zero_centered",
            "thresholds": {
                "aggressive": {
                    "long_gt": 0.1,
                    "short_lt": -0.1
                },
                "conservative": {
                    "long_gt": 0.3,
                    "short_lt": -0.3
                },
                "normal": {
                    "long_gt": 0.2,
                    "short_lt": -0.2
                }
            },
            "type": "original"
        }
    },
    "EMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.ema",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "EMA計算期間",
                }
            },
            "pandas_function": "ema",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 20},
            "min_length_func": "EMA",
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": "close",
            "type": "trend",
        },
    },
    "HMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.hma",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "HMA計算期間",
                }
            },
            "pandas_function": "hma",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 20},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
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
    },
    "KELTNER": {
        "config": {
            "result_type": "complex",
            "scale_type": "price_ratio",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.keltner",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
                "multiplier": "multiplier",
            },
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "Keltner Channels期間",
                },
                "multiplier": {
                    "default_value": 2.0,
                    "min_value": 0.5,
                    "max_value": 5.0,
                    "description": "ATR倍数",
                },
            },
            "pandas_function": "kc",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "multiple",
            "return_cols": ["KC_LB", "KC_MID", "KC_UB"],
            "multi_column": True,
            "default_values": {"length": 20, "multiplier": 2.0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}_upper",
                "short": "close < {left_operand}_lower",
            },
            "scale_type": "price_absolute",
            "thresholds": {
                "aggressive": {"multiplier": 1.0},
                "combo": {
                    "condition": "distance >= primary or distance <= secondary",
                    "primary": 2.5,
                    "secondary": -2.5,
                },
                "conservative": {"multiplier": 2.0},
                "normal": {"multiplier": 1.5},
                "range": {"break_out_lower": -1.8, "break_out_upper": 1.8},
            },
            "type": "volatility",
        },
    },
    "KST": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.kst",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "roc1": "roc1",
                "roc2": "roc2",
                "roc3": "roc3",
                "roc4": "roc4",
                "sma1": "sma1",
                "sma2": "sma2",
                "sma3": "sma3",
                "sma4": "sma4",
                "signal": "signal",
            },
            "parameters": {
                "roc1": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 60,
                    "description": "ROC1期間",
                },
                "roc2": {
                    "default_value": 15,
                    "min_value": 2,
                    "max_value": 60,
                    "description": "ROC2期間",
                },
                "roc3": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 60,
                    "description": "ROC3期間",
                },
                "roc4": {
                    "default_value": 30,
                    "min_value": 2,
                    "max_value": 90,
                    "description": "ROC4期間",
                },
                "sma1": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 60,
                    "description": "SMA1期間",
                },
                "sma2": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 60,
                    "description": "SMA2期間",
                },
                "sma3": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 60,
                    "description": "SMA3期間",
                },
                "sma4": {
                    "default_value": 15,
                    "min_value": 2,
                    "max_value": 90,
                    "description": "SMA4期間",
                },
                "signal": {
                    "default_value": 9,
                    "min_value": 1,
                    "max_value": 50,
                    "description": "シグナル期間",
                },
            },
            "pandas_function": "kst",
            "data_column": "Close",
            "data_columns": None,
            "returns": "multiple",
            "return_cols": ["KST", "KSTs"],
            "multi_column": False,
            "default_values": {
                "roc1": 10,
                "roc2": 15,
                "roc3": 20,
                "roc4": 30,
                "sma1": 10,
                "sma2": 10,
                "sma3": 10,
                "sma4": 15,
                "signal": 9,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > 0",
                "short": "{left_operand}_0 < 0",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "momentum",
        },
    },
    "MACD": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.macd",
            "required_data": ["close"],
            "output_names": ["MACD_0", "MACD_1", "MACD_2"],
            "default_output": "MACD_0",
            "aliases": None,
            "param_map": {
                "close": "data",
                "fast": "fast",
                "slow": "slow",
                "signal": "signal",
            },
            "parameters": {
                "fast": {
                    "default_value": 12,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "高速移動平均期間",
                },
                "slow": {
                    "default_value": 26,
                    "min_value": 5,
                    "max_value": 200,
                    "description": "低速移動平均期間",
                },
                "signal": {
                    "default_value": 9,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "シグナル線期間",
                },
            },
            "pandas_function": "macd",
            "data_column": "Close",
            "data_columns": None,
            "returns": "multiple",
            "return_cols": ["MACD", "Signal", "Histogram"],
            "multi_column": False,
            "default_values": {"fast": 12, "slow": 26, "signal": 9},
            "min_length_func": "MACD",
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > 0",
                "short": "{left_operand}_0 < 0",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "momentum",
        },
    },
    "PPO": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.ppo",
            "required_data": ["close"],
            "output_names": ["PPO_0", "PPO_1", "PPO_2"],
            "default_output": "PPO_0",
            "aliases": None,
            "param_map": {
                "close": "data",
                "fast": "fast",
                "slow": "slow",
                "signal": "signal",
            },
            "parameters": {
                "fast": {
                    "default_value": 12,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "高速移動平均期間",
                },
                "slow": {
                    "default_value": 26,
                    "min_value": 5,
                    "max_value": 200,
                    "description": "低速移動平均期間",
                },
                "signal": {
                    "default_value": 9,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "シグナル線期間",
                },
            },
            "pandas_function": "ppo",
            "data_column": "Close",
            "data_columns": None,
            "returns": "multiple",
            "return_cols": ["PPO", "PPOs", "PPOh"],
            "multi_column": False,
            "default_values": {"fast": 12, "slow": 26, "signal": 9},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > 0",
                "short": "{left_operand}_0 < 0",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "momentum",
        },
    },
    "MFI": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.mfi",
            "required_data": ["high", "low", "close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"length": "length"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "MFI計算期間",
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_lt": 20, "short_gt": 85},
                "conservative": {"long_lt": 40, "short_gt": 60},
                "normal": {"long_lt": 30, "short_gt": 70},
            },
            "type": "volume",
        },
    },
    "MOM": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.mom",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": ["MOMENTUM"],
            "param_map": {"close": "data", "period": "length", "length": "length"},
            "parameters": {
                "period": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "モメンタム計算期間",
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"period": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_lt": -10, "short_gt": 10},
                "conservative": {"long_lt": -2, "short_gt": 2},
                "normal": {"long_lt": -5, "short_gt": 5},
            },
            "type": "momentum",
        },
    },
    "OBV": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.obv",
            "required_data": ["close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {},
            "parameters": {},
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {"long": "{left_operand} > 0", "short": "{left_operand} < 0"},
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "volume",
        },
    },
    "QQE": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.qqe",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length", "smooth": "smooth"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "QQE計算期間",
                },
                "smooth": {
                    "default_value": 5,
                    "min_value": 1,
                    "max_value": 50,
                    "description": "QQE平滑化期間",
                },
            },
            "pandas_function": "qqe",
            "data_column": "Close",
            "data_columns": None,
            "returns": "multiple",
            "return_cols": ["QQE", "QQE_SIGNAL"],
            "multi_column": False,
            "default_values": {"length": 14, "smooth": 5},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_lt": 15, "short_gt": 85},
                "conservative": {"long_lt": 25, "short_gt": 75},
                "normal": {"long_lt": 20, "short_gt": 80},
            },
            "type": "momentum",
        },
    },
    "ROC": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.roc",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 10,
                    "min_value": 1,
                    "max_value": 100,
                    "description": "ROC計算期間",
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_lt": -3.0, "short_gt": 3.0},
                "combo": {
                    "condition": "abs_value >= primary or abs_value <= -secondary",
                    "primary": -5.0,
                    "secondary": 5.0,
                },
                "conservative": {"long_lt": -0.5, "short_gt": 0.5},
                "normal": {"long_lt": -1.5, "short_gt": 1.5},
                "range": {
                    "long_gt": -2.0,
                    "long_lt": 0.0,
                    "short_gt": 0.0,
                    "short_lt": 2.0,
                },
            },
            "type": "momentum",
        },
    },
    "CTI": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.cti",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 12,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "CTI計算期間",
                }
            },
            "pandas_function": "cti",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 12},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {"long": "{left_operand} > 0", "short": "{left_operand} < 0"},
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "momentum",
        },
    },
    "RVI": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.rvi",
            "required_data": ["close", "high", "low"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "close",
                "high": "high",
                "low": "low",
                "length": "length",
                "scalar": "scalar",
                "refined": "refined",
                "thirds": "thirds",
                "mamode": "mamode",
                "drift": "drift",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "RVI計算期間",
                },
                "scalar": {
                    "default_value": 100.0,
                    "min_value": 1.0,
                    "max_value": 200.0,
                    "description": "指数スケール係数",
                },
                "refined": {
                    "default_value": False,
                    "min_value": None,
                    "max_value": None,
                    "description": "Refined モードを使用",
                },
                "thirds": {
                    "default_value": False,
                    "min_value": None,
                    "max_value": None,
                    "description": "Thirds モードを使用",
                },
                "mamode": {
                    "default_value": "ema",
                    "min_value": None,
                    "max_value": None,
                    "description": "平滑化移動平均モード",
                },
                "drift": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "差分に使用するドリフト",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "出力オフセット",
                },
            },
            "pandas_function": "rvi",
            "data_column": None,
            "data_columns": ["Close", "High", "Low"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {
                "length": 14,
                "scalar": 100.0,
                "refined": False,
                "thirds": False,
                "mamode": "ema",
                "drift": 1,
                "offset": 0,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 55, "short_lt": 45},
                "normal": {"long_gt": 60, "short_lt": 40},
                "conservative": {"long_gt": 65, "short_lt": 35},
            },
            "type": "volatility",
        },
    },
    "RSI": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.rsi",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "RSI計算期間",
                }
            },
            "pandas_function": "rsi",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": "RSI",
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 70, "short_lt": 30},
                "conservative": {"long_gt": 80, "short_lt": 20},
                "normal": {"long_gt": 75, "short_lt": 25},
            },
            "type": "momentum",
        },
    },
    "SAR": {
        "config": {
            "result_type": "single",
            "scale_type": "price_absolute",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.sar",
            "required_data": ["high", "low"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"high": "high", "low": "low", "af": "af", "max_af": "max_af"},
            "parameters": {
                "af": {
                    "default_value": 0.02,
                    "min_value": 0.01,
                    "max_value": 0.1,
                    "description": "加速因子",
                },
                "max_af": {
                    "default_value": 0.2,
                    "min_value": 0.1,
                    "max_value": 1.0,
                    "description": "最大加速因子",
                },
            },
            "pandas_function": "psar",
            "data_column": None,
            "data_columns": ["High", "Low"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {"af": 0.02, "max_af": 0.2},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "SMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.sma",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "SMA計算期間",
                }
            },
            "pandas_function": "sma",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 20},
            "min_length_func": "SMA",
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "SQUEEZE": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.squeeze",
            "required_data": ["high", "low", "close"],
            "output_names": ["SQZ"],
            "default_output": "SQZ",
            "aliases": ["SQUEEZE"],
            "param_map": {
                "bb_length": "bb_length",
                "bb_std": "bb_std",
                "kc_length": "kc_length",
                "kc_scalar": "kc_scalar",
                "mom_length": "mom_length",
                "mom_smooth": "mom_smooth",
                "use_tr": "use_tr",
            },
            "parameters": {
                "bb_length": {
                    "default_value": 20,
                    "min_value": 5,
                    "max_value": 100,
                    "description": "Bollinger Bands length",
                },
                "bb_std": {
                    "default_value": 2.0,
                    "min_value": 0.5,
                    "max_value": 5.0,
                    "description": "Bollinger Bands standard deviation",
                },
                "kc_length": {
                    "default_value": 20,
                    "min_value": 5,
                    "max_value": 100,
                    "description": "Keltner Channels length",
                },
                "kc_scalar": {
                    "default_value": 1.5,
                    "min_value": 0.1,
                    "max_value": 5.0,
                    "description": "Keltner Channels scalar",
                },
                "mom_length": {
                    "default_value": 12,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "Momentum length",
                },
                "mom_smooth": {
                    "default_value": 6,
                    "min_value": 1,
                    "max_value": 20,
                    "description": "Momentum smoothing",
                },
                "use_tr": {
                    "default_value": True,
                    "min_value": None,
                    "max_value": None,
                    "description": "Use True Range for calculations",
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {
                "bb_length": 20,
                "bb_std": 2.0,
                "kc_length": 20,
                "kc_scalar": 1.5,
                "mom_length": 12,
                "mom_smooth": 6,
                "use_tr": True,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_lt": 15, "short_gt": 85},
                "normal": {"long_lt": 20, "short_gt": 80},
                "conservative": {"long_lt": 25, "short_gt": 75},
            },
            "type": "momentum",
        },
    },
    "STOCH": {
        "config": {
            "result_type": "complex",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.stoch",
            "required_data": ["high", "low", "close"],
            "output_names": ["STOCH_0", "STOCH_1"],
            "default_output": "STOCH_0",
            "aliases": ["STOCH"],
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "k_length": "k",
                "smooth_k": "smooth_k",
                "d_length": "d",
            },
            "parameters": {
                "k_length": {
                    "default_value": 14,
                    "min_value": 1,
                    "max_value": 30,
                    "description": "K期間",
                },
                "smooth_k": {
                    "default_value": 3,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "K平滑化期間",
                },
                "d_length": {
                    "default_value": 3,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "D期間",
                },
            },
            "pandas_function": "stoch",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "multiple",
            "return_cols": ["STOCHk", "STOCHd"],
            "multi_column": True,
            "default_values": {"k_length": 14, "smooth_k": 3, "d_length": 3},
            "min_length_func": "STOCH",
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 < {threshold}",
                "short": "{left_operand}_0 > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_lt": 15, "short_gt": 85},
                "conservative": {"long_lt": 25, "short_gt": 75},
                "normal": {"long_lt": 20, "short_gt": 80},
            },
            "type": "momentum",
        },
    },
    "STOCHRSI": {
        "config": {
            "result_type": "complex",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.stochrsi",
            "required_data": ["close"],
            "output_names": ["STOCHRSI_K", "STOCHRSI_D"],
            "default_output": "STOCHRSI_K",
            "aliases": ["STOCHRSI"],
            "param_map": {
                "close": "data",
                "rsi_length": "rsi_length",
                "stoch_length": "stoch_length",
                "k": "k",
                "d": "d",
            },
            "parameters": {
                "rsi_length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "RSI計算期間",
                },
                "stoch_length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "Stochastic計算期間",
                },
                "k": {
                    "default_value": 3,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "K平滑化期間",
                },
                "d": {
                    "default_value": 3,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "D平滑化期間",
                },
            },
            "pandas_function": "stochrsi",
            "data_column": "Close",
            "data_columns": None,
            "returns": "multiple",
            "return_cols": ["STOCHRSIk", "STOCHRSId"],
            "multi_column": False,
            "default_values": {"rsi_length": 14, "stoch_length": 14, "k": 3, "d": 3},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > {threshold}",
                "short": "{left_operand}_0 < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 70, "short_lt": 30},
                "conservative": {"long_gt": 80, "short_lt": 20},
                "normal": {"long_gt": 75, "short_lt": 25},
            },
            "type": "momentum",
        },
    },
    "SUPERTREND": {
        "config": {
            "result_type": "complex",
            "scale_type": "price_absolute",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.supertrend",
            "required_data": ["high", "low", "close"],
            "output_names": [
                "SUPERTREND_Lower",
                "SUPERTREND_Upper",
                "SUPERTREND_Direction",
            ],
            "default_output": "SUPERTREND_Direction",
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
                "multiplier": "multiplier",
            },
            "parameters": {
                "length": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "ATR期間",
                },
                "multiplier": {
                    "default_value": 3.0,
                    "min_value": 1.0,
                    "max_value": 10.0,
                    "description": "ATR倍数",
                },
            },
            "pandas_function": "supertrend",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "complex",
            "return_cols": ["ST", "D"],
            "multi_column": True,
            "default_values": {"length": 10, "multiplier": 3.0},
            "min_length_func": "SUPERTREND",
        },
        "yaml": {
            "components": ["lower", "upper", "direction"],
            "conditions": {
                "long": "{left_operand}_2 > 0",
                "short": "{left_operand}_2 < 0",
            },
            "scale_type": "price_absolute",
            "thresholds": {
                "aggressive": {"multiplier": 1.5},
                "conservative": {"multiplier": 4.5},
                "normal": {"multiplier": 3.0},
            },
            "type": "volatility",
        },
    },
    "UO": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.uo",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "fast": "fast",
                "medium": "medium",
                "slow": "slow",
            },
            "parameters": {
                "fast": {
                    "default_value": 7,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "短期期間",
                },
                "medium": {
                    "default_value": 14,
                    "min_value": 3,
                    "max_value": 100,
                    "description": "中期期間",
                },
                "slow": {
                    "default_value": 28,
                    "min_value": 5,
                    "max_value": 200,
                    "description": "長期期間",
                },
            },
            "pandas_function": "uo",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {"fast": 7, "medium": 14, "slow": 28},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 55, "short_lt": 45},
                "conservative": {"long_gt": 65, "short_lt": 35},
                "normal": {"long_gt": 60, "short_lt": 40},
            },
            "type": "momentum",
        },
    },
    "T3": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.t3",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "length": "length",
                "a": "a",
                "vfactor": "vfactor",
            },
            "parameters": {
                "length": {
                    "default_value": 5,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "T3移動平均期間",
                },
                "a": {
                    "default_value": 0.7,
                    "min_value": 0.1,
                    "max_value": 1.0,
                    "description": "T3スムージングファクター",
                },
                "vfactor": {
                    "default_value": 0.7,
                    "min_value": 0.1,
                    "max_value": 1.0,
                    "description": "pandas-ta互換用パラメータ",
                },
            },
            "pandas_function": "t3",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 5, "a": 0.7, "vfactor": 0.7},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "APO": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.apo",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "fast": "fast",
                "slow": "slow",
                "ma_mode": "ma_mode",
            },
            "parameters": {
                "fast": {
                    "default_value": 12,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "短期EMA期間",
                },
                "slow": {
                    "default_value": 26,
                    "min_value": 5,
                    "max_value": 200,
                    "description": "長期EMA期間",
                },
                "ma_mode": {
                    "default_value": "ema",
                    "min_value": None,
                    "max_value": None,
                    "description": "移動平均モード",
                },
            },
            "pandas_function": "apo",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"fast": 12, "slow": 26, "ma_mode": "ema"},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {"long": "{left_operand} > 0", "short": "{left_operand} < 0"},
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "momentum",
        },
    },
    "LINREG": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.linreg",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "length": "length",
                "scalar": "scalar",
                "intercept": "intercept",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "回帰期間",
                },
                "scalar": {
                    "default_value": 1.0,
                    "min_value": 0.1,
                    "max_value": 10.0,
                    "description": "スケーリング係数",
                },
                "intercept": {
                    "default_value": False,
                    "min_value": None,
                    "max_value": None,
                    "description": "切片を返すかどうか",
                },
            },
            "pandas_function": None,
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14, "scalar": 1.0, "intercept": False},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "LINREGSLOPE": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.linregslope",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length", "scalar": "scalar"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "回帰期間",
                },
                "scalar": {
                    "default_value": 1.0,
                    "min_value": 0.1,
                    "max_value": 10.0,
                    "description": "スケーリング係数",
                },
            },
            "pandas_function": None,
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14, "scalar": 1.0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {"long": "{left_operand} > 0", "short": "{left_operand} < 0"},
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "trend",
        },
    },
    "NATR": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.natr",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "NATR計算期間",
                }
            },
            "pandas_function": "natr",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_lt": 1.5, "short_gt": 5.0},
                "normal": {"long_lt": 2.0, "short_gt": 4.0},
                "conservative": {"long_lt": 2.5, "short_gt": 3.5},
            },
            "type": "volatility",
        },
    },
    "KVO": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.kvo",
            "required_data": ["high", "low", "close", "volume"],
            "output_names": ["KVO", "KVO_SIGNAL"],
            "default_output": "KVO",
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "fast": "fast",
                "slow": "slow",
                "signal": "signal",
                "scalar": "scalar",
                "mamode": "mamode",
                "drift": "drift",
            },
            "parameters": {
                "fast": {
                    "default_value": 34,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "短期EMA期間",
                },
                "slow": {
                    "default_value": 55,
                    "min_value": 5,
                    "max_value": 200,
                    "description": "長期EMA期間",
                },
                "signal": {
                    "default_value": 13,
                    "min_value": 1,
                    "max_value": 100,
                    "description": "シグナルEMA期間",
                },
                "scalar": {
                    "default_value": 100.0,
                    "min_value": 1.0,
                    "max_value": 200.0,
                    "description": "スケール係数",
                },
                "mamode": {
                    "default_value": "ema",
                    "min_value": None,
                    "max_value": None,
                    "description": "平滑化モード",
                },
                "drift": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "差分に用いるドリフト",
                },
            },
            "pandas_function": "kvo",
            "data_column": None,
            "data_columns": ["High", "Low", "Close", "Volume"],
            "returns": "multiple",
            "return_cols": ["KVO", "KVOs"],
            "multi_column": True,
            "default_values": {
                "fast": 34,
                "slow": 55,
                "signal": 13,
                "scalar": 100.0,
                "mamode": "ema",
                "drift": 1,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > 0",
                "short": "{left_operand}_0 < 0",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "volume",
        },
    },
    "PVO": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.pvo",
            "required_data": ["volume"],
            "output_names": ["PVO", "PVO_SIGNAL", "PVO_HIST"],
            "default_output": "PVO",
            "aliases": None,
            "param_map": {
                "volume": "volume",
                "fast": "fast",
                "slow": "slow",
                "signal": "signal",
                "scalar": "scalar",
            },
            "parameters": {
                "fast": {
                    "default_value": 12,
                    "min_value": 1,
                    "max_value": 200,
                    "description": "高速EMA期間",
                },
                "slow": {
                    "default_value": 26,
                    "min_value": 2,
                    "max_value": 300,
                    "description": "低速EMA期間",
                },
                "signal": {
                    "default_value": 9,
                    "min_value": 1,
                    "max_value": 200,
                    "description": "シグナル期間",
                },
                "scalar": {
                    "default_value": 100.0,
                    "min_value": 1.0,
                    "max_value": 200.0,
                    "description": "スケール係数",
                },
            },
            "pandas_function": "pvo",
            "data_columns": ["Volume"],
            "returns": "multiple",
            "return_cols": ["PVO", "PVOs", "PVOh"],
            "multi_column": True,
            "default_values": {"fast": 12, "slow": 26, "signal": 9, "scalar": 100.0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > 0",
                "short": "{left_operand}_0 < 0",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "volume",
        },
    },
    "PVT": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.pvt",
            "required_data": ["close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "close", "volume": "volume"},
            "parameters": {},
            "pandas_function": "pvt",
            "data_column": "Close",
            "data_columns": ["Volume"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"normal": {"threshold": 0}},
            "type": "volume",
        },
    },
    "NVI": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.nvi",
            "required_data": ["close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "close", "volume": "volume"},
            "parameters": {},
            "pandas_function": "nvi",
            "data_column": "Close",
            "data_columns": ["Volume"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"normal": {"threshold": 0}},
            "type": "volume",
        },
    },
    "TRIMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.trima",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length", "talib": "talib"},
            "parameters": {
                "length": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "TRIMA期間",
                }
            },
            "pandas_function": "trima",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "TEMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.tema",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "三重指数移動平均期間",
                }
            },
            "pandas_function": "tema",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": "TEMA",
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "UI": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.ui",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "UI計算期間",
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {"all": {"long_lt": 30, "short_gt": 70}},
            "type": "volatility",
        },
    },
    "VWMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.vwma",
            "required_data": ["close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"volume": "volume", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "VWMA期間",
                }
            },
            "pandas_function": "vwma",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 20},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "VHF": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.vhf",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"length": "length", "scalar": "scalar", "drift": "drift", "offset": "offset"},
            "parameters": {
                "length": {
                    "default_value": 28,
                    "min_value": 10,
                    "max_value": 100,
                    "description": "VHF期間",
                },
                "scalar": {
                    "default_value": 100.0,
                    "min_value": 1.0,
                    "max_value": 1000.0,
                    "description": "VHFスカラー",
                },
                "drift": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "ドリフト期間",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": 0,
                    "max_value": 10,
                    "description": "オフセット",
                }
            },
            "pandas_function": "vhf",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 28, "scalar": 100.0, "drift": 1, "offset": 0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {"all": {"long_lt": 0.3, "short_gt": 0.7}},
            "type": "volatility",
        },
    },
    "ZLMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.zlma",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "length": "length",
                "mamode": "mamode",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "ZLMA期間",
                },
                "mamode": {
                    "default_value": "ema",
                    "min_value": None,
                    "max_value": None,
                    "description": "元となる移動平均モード",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "シフト量",
                },
            },
            "pandas_function": "zlma",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 10, "mamode": "ema", "offset": 0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "VORTEX": {
        "config": {
            "result_type": "complex",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.vortex",
            "required_data": ["high", "low", "close"],
            "output_names": ["VORTEX_Positive", "VORTEX_Negative"],
            "default_output": "VORTEX_Positive",
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
                "drift": "drift",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "Vortex期間",
                },
                "drift": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "差分に使用するドリフト",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "出力オフセット",
                },
            },
            "pandas_function": "vortex",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "multiple",
            "return_cols": ["VTXP", "VTXM"],
            "multi_column": True,
            "default_values": {"length": 14, "drift": 1, "offset": 0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > {left_operand}_1",
                "short": "{left_operand}_0 < {left_operand}_1",
            },
            "scale_type": "price_ratio",
            "thresholds": None,
            "type": "trend",
        },
    },
    "VWAP": {
        "config": {
            "result_type": "single",
            "scale_type": "volume",
            "category": "volume",
            "adapter_function": "app.services.indicators.technical_indicators.volume.VolumeIndicators.vwap",
            "required_data": ["high", "low", "close", "volume"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {},
            "parameters": {},
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "volume",
        },
    },
    "WILLR": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_plus_minus_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.willr",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"period": "length"},
            "parameters": {
                "period": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "Williams %R計算期間",
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"period": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_plus_minus_100",
            "thresholds": {
                "aggressive": {"long_lt": -95, "short_gt": -5},
                "conservative": {"long_lt": -75, "short_gt": -25},
                "normal": {"long_lt": -80, "short_gt": -20},
            },
            "type": "momentum",
        },
    },
    "STC": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.stc",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "fast": "fast",
                "slow": "slow",
                "cycle": "cycle",
                "d1": "d1",
                "d2": "d2",
            },
            "parameters": {
                "fast": {
                    "default_value": 23,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "STC fast period",
                },
                "slow": {
                    "default_value": 50,
                    "min_value": 5,
                    "max_value": 100,
                    "description": "STC slow period",
                },
                "cycle": {
                    "default_value": 10,
                    "min_value": 3,
                    "max_value": 50,
                    "description": "STC cycle period",
                },
                "d1": {
                    "default_value": 3,
                    "min_value": 2,
                    "max_value": 20,
                    "description": "STC first smoothing",
                },
                "d2": {
                    "default_value": 3,
                    "min_value": 2,
                    "max_value": 20,
                    "description": "STC second smoothing",
                },
            },
            "pandas_function": "stc",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"fast": 23, "slow": 50, "cycle": 10, "d1": 3, "d2": 3},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_lt": 25, "short_gt": 75},
                "conservative": {"long_lt": 35, "short_gt": 65},
                "normal": {"long_lt": 30, "short_gt": 70},
            },
            "type": "momentum",
        },
    },
    "WMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.wma",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 20,
                    "min_value": 2,
                    "max_value": 200,
                    "description": None,
                }
            },
            "pandas_function": "wma",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 20},
            "min_length_func": "WMA",
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "TRIX": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.trix",
            "required_data": ["close"],
            "output_names": ["TRIX_Value", "TRIX_Signal", "TRIX_Histogram"],
            "default_output": "TRIX_Value",
            "aliases": None,
            "param_map": {
                "close": "data",
                "length": "length",
                "signal": "signal",
                "scalar": "scalar",
                "drift": "drift",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 15,
                    "min_value": 2,
                    "max_value": 200,
                    "description": "TRIX計算期間",
                },
                "signal": {
                    "default_value": 9,
                    "min_value": 1,
                    "max_value": 50,
                    "description": "シグナル期間",
                },
                "scalar": {
                    "default_value": 100.0,
                    "min_value": 1.0,
                    "max_value": 200.0,
                    "description": "スケール調整係数",
                },
                "drift": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "ドリフト期間",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "出力オフセット",
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "multiple",
            "return_cols": None,
            "multi_column": False,
            "default_values": {
                "length": 15,
                "signal": 9,
                "scalar": 100.0,
                "drift": 1,
                "offset": 0,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > {left_operand}_1",
                "short": "{left_operand}_0 < {left_operand}_1",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {"zero_cross": True},
            "type": "momentum",
        },
    },
    "TSI": {
        "config": {
            "result_type": "complex",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.tsi",
            "required_data": ["close"],
            "output_names": ["TSI_0", "TSI_1"],
            "default_output": "TSI_0",
            "aliases": None,
            "param_map": {
                "close": "data",
                "fast": "fast",
                "slow": "slow",
                "signal": "signal",
                "scalar": "scalar",
                "mamode": "mamode",
                "drift": "drift",
            },
            "parameters": {
                "fast": {
                    "default_value": 13,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "TSI短期期間",
                },
                "slow": {
                    "default_value": 25,
                    "min_value": 5,
                    "max_value": 50,
                    "description": "TSI長期期間",
                },
                "signal": {
                    "default_value": 13,
                    "min_value": 1,
                    "max_value": 50,
                    "description": "TSIシグナル期間",
                },
                "scalar": {
                    "default_value": 100.0,
                    "min_value": 1.0,
                    "max_value": 200.0,
                    "description": "TSIスカラー",
                },
                "mamode": {
                    "default_value": "ema",
                    "min_value": None,
                    "max_value": None,
                    "description": "移動平均モード",
                },
                "drift": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "ドリフト期間",
                },
            },
            "pandas_function": "tsi",
            "data_column": "Close",
            "data_columns": None,
            "returns": "multiple",
            "return_cols": ["TSI", "TSIs"],
            "multi_column": False,
            "default_values": {
                "fast": 13,
                "slow": 25,
                "signal": 13,
                "scalar": 100.0,
                "mamode": "ema",
                "drift": 1,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > {threshold}",
                "short": "{left_operand}_0 < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 25, "short_lt": -25},
                "conservative": {"long_gt": 40, "short_lt": -40},
                "normal": {"long_gt": 30, "short_lt": -30},
            },
            "type": "momentum",
        },
    },
    "PGO": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_plus_minus_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.pgo",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "PGO計算期間",
                }
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_plus_minus_100",
            "thresholds": {
                "aggressive": {"long_gt": 10, "short_lt": -10},
                "conservative": {"long_gt": 40, "short_lt": -40},
                "normal": {"long_gt": 25, "short_lt": -25},
            },
            "type": "momentum",
        },
    },
    "MASSI": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.massi",
            "required_data": ["high", "low"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"high": "high", "low": "low", "fast": "fast", "slow": "slow"},
            "parameters": {
                "fast": {
                    "default_value": 9,
                    "min_value": 2,
                    "max_value": 20,
                    "description": "短期EMA期間",
                },
                "slow": {
                    "default_value": 25,
                    "min_value": 10,
                    "max_value": 50,
                    "description": "長期EMA期間",
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"fast": 9, "slow": 25},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 85, "short_lt": 65},
                "conservative": {"long_gt": 95, "short_lt": 85},
                "normal": {"long_gt": 90, "short_lt": 75},
            },
            "type": "momentum",
        },
    },
    "PSL": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.psl",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "close",
                "length": "length",
                "scalar": "scalar",
                "drift": "drift",
                "open_": "open_",
            },
            "parameters": {
                "length": {
                    "default_value": 12,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "心理ライン期間",
                },
                "scalar": {
                    "default_value": 100.0,
                    "min_value": 1.0,
                    "max_value": 200.0,
                    "description": "スケーリング係数",
                },
                "drift": {
                    "default_value": 1,
                    "min_value": 1,
                    "max_value": 10,
                    "description": "ドリフト期間",
                },
                "open_": {
                    "default_value": None,
                    "min_value": None,
                    "max_value": None,
                    "description": "始値データ",
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {
                "length": 12,
                "scalar": 100.0,
                "drift": 1,
                "open_": None,
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 65, "short_lt": 35},
                "conservative": {"long_gt": 80, "short_lt": 20},
                "normal": {"long_gt": 75, "short_lt": 25},
            },
            "type": "momentum",
        },
    },
    "AO": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.ao",
            "required_data": ["high", "low"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"high": "high", "low": "low", "fast": "fast", "slow": "slow"},
            "parameters": {
                "fast": {
                    "default_value": 5,
                    "min_value": 2,
                    "max_value": 20,
                    "description": "AO短期期間",
                },
                "slow": {
                    "default_value": 34,
                    "min_value": 10,
                    "max_value": 50,
                    "description": "AO長期期間",
                },
            },
            "pandas_function": "ao",
            "data_column": None,
            "data_columns": ["High", "Low"],
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"fast": 5, "slow": 34},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_gt": 0.1, "short_lt": -0.1},
                "conservative": {"long_gt": 0.3, "short_lt": -0.3},
                "normal": {"long_gt": 0.2, "short_lt": -0.2},
            },
            "type": "momentum",
        },
    },
    "AROON": {
        "config": {
            "result_type": "complex",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.aroon",
            "required_data": ["high", "low"],
            "output_names": ["AROON_UP", "AROON_DOWN"],
            "default_output": "AROON_UP",
            "aliases": None,
            "param_map": {"high": "high", "low": "low", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "AROON期間",
                }
            },
            "pandas_function": "aroon",
            "data_column": None,
            "data_columns": ["High", "Low"],
            "returns": "multiple",
            "return_cols": ["AROON_UP", "AROON_DOWN"],
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > {threshold}",
                "short": "{left_operand}_1 < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 70, "short_lt": 30},
                "conservative": {"long_gt": 90, "short_lt": 10},
                "normal": {"long_gt": 80, "short_lt": 20},
            },
            "type": "momentum",
        },
    },
    "CHOP": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.chop",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "CHOP期間",
                }
            },
            "pandas_function": "chop",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_lt": 40, "short_gt": 60},
                "conservative": {"long_lt": 20, "short_gt": 80},
                "normal": {"long_lt": 30, "short_gt": 70},
            },
            "type": "volatility",
        },
    },
    "BOP": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.bop",
            "required_data": ["open", "high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "open": "open_",
                "high": "high",
                "low": "low",
                "close": "close",
            },
            "parameters": {},
            "pandas_function": "bop",
            "data_column": None,
            "data_columns": ["Open", "High", "Low", "Close"],
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_gt": 0.1, "short_lt": -0.1},
                "conservative": {"long_gt": 0.3, "short_lt": -0.3},
                "normal": {"long_gt": 0.2, "short_lt": -0.2},
            },
            "type": "momentum",
        },
    },
    "CG": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.cg",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "CG期間",
                }
            },
            "pandas_function": "cg",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 60, "short_lt": 40},
                "conservative": {"long_gt": 80, "short_lt": 20},
                "normal": {"long_gt": 70, "short_lt": 30},
            },
            "type": "momentum",
        },
    },
    "ICHIMOKU": {
        "config": {
            "result_type": "complex",
            "scale_type": "price_absolute",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.ichimoku",
            "required_data": ["high", "low", "close"],
            "output_names": ["ICHIMOKU_Tenkan", "ICHIMOKU_Kijun", "ICHIMOKU_Senkou_A", "ICHIMOKU_Senkou_B", "ICHIMOKU_Chikou"],
            "default_output": "ICHIMOKU_Tenkan",
            "aliases": ["ICHIMOKU"],
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "tenkan_period": "tenkan_period",
                "kijun_period": "kijun_period",
                "senkou_span_b_period": "senkou_span_b_period"
            },
            "parameters": {
                "tenkan_period": {
                    "default_value": 9,
                    "min_value": 3,
                    "max_value": 50,
                    "description": "Tenkan-sen (conversion line) period",
                },
                "kijun_period": {
                    "default_value": 26,
                    "min_value": 10,
                    "max_value": 100,
                    "description": "Kijun-sen (base line) period",
                },
                "senkou_span_b_period": {
                    "default_value": 52,
                    "min_value": 20,
                    "max_value": 200,
                    "description": "Senkou Span B (leading span B) period",
                }
            },
            "pandas_function": "ichimoku",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "multiple",
            "return_cols": ["TENKAN", "KIJUN", "SENKOU", "SANSEN", "CHIKOU"],
            "multi_column": True,
            "default_values": {"tenkan_period": 9, "kijun_period": 26, "senkou_span_b_period": 52},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_tenkan_sen > {left_operand}_kijun_sen",
                "short": "{left_operand}_tenkan_sen < {left_operand}_kijun_sen",
            },
            "scale_type": "price_absolute",
            "thresholds": {
                "aggressive": {"tenkan_gt_kijun": True},
                "conservative": {"tenkan_gt_kijun": False},
                "normal": {"tenkan_gt_kijun": True},
            },
            "type": "momentum",
        },
    },
    "COPPOCK": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.coppock",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": ["COPC"],
            "param_map": {
                "close": "close",
                "length": "length",
                "fast": "fast",
                "slow": "slow",
            },
            "parameters": {
                "length": {
                    "default_value": 11,
                    "min_value": 5,
                    "max_value": 30,
                    "description": "Coppock主期間",
                },
                "fast": {
                    "default_value": 14,
                    "min_value": 5,
                    "max_value": 30,
                    "description": "短期期間",
                },
                "slow": {
                    "default_value": 10,
                    "min_value": 5,
                    "max_value": 30,
                    "description": "長期期間",
                },
            },
            "pandas_function": "coppock",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 11, "fast": 14, "slow": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 50, "short_lt": 30},
                "conservative": {"long_gt": 80, "short_lt": 20},
                "normal": {"long_gt": 60, "short_lt": 40},
            },
            "type": "momentum",
        },
    },
    "AMAT": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.amat",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {
                "close": "data",
                "fast": "fast",
                "slow": "slow",
                "signal": "signal",
            },
            "parameters": {
                "fast": {
                    "default_value": 3,
                    "min_value": 2,
                    "max_value": 20,
                    "description": "AMAT短期期間",
                },
                "slow": {
                    "default_value": 30,
                    "min_value": 10,
                    "max_value": 50,
                    "description": "AMAT長期期間",
                },
                "signal": {
                    "default_value": 10,
                    "min_value": 5,
                    "max_value": 30,
                    "description": "シグナル期間",
                },
            },
            "pandas_function": "amat",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"fast": 3, "slow": 30, "signal": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "RMA": {
        "config": {
            "result_type": "single",
            "scale_type": "price_ratio",
            "category": "trend",
            "adapter_function": "app.services.indicators.technical_indicators.trend.TrendIndicators.rma",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": None,
            "param_map": {"close": "data", "length": "length"},
            "parameters": {
                "length": {
                    "default_value": 10,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "RMA期間",
                }
            },
            "pandas_function": "rma",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 10},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "close > {left_operand}",
                "short": "close < {left_operand}",
            },
            "scale_type": "price_absolute",
            "thresholds": None,
            "type": "trend",
        },
    },
    "GRI": {
        "config": {
            "result_type": "single",
            "scale_type": "momentum_zero_centered",
            "category": "volatility",
            "adapter_function": "app.services.indicators.technical_indicators.volatility.VolatilityIndicators.gri",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": ["gopalakrishnan_range"],
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "GRI計算期間",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "出力オフセット",
                },
            },
            "pandas_function": "kvo",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14, "offset": 0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "left_operand} < {threshold}",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_gt": 10, "short_lt": -10},
                "conservative": {"long_gt": 30, "short_lt": -30},
                "normal": {"long_gt": 20, "short_lt": -20},
            },
            "type": "volatility",
        },
    },
    "PRIME_OSC": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "original",
            "adapter_function": "app.services.indicators.technical_indicators.original.OriginalIndicators.prime_oscillator",
            "required_data": ["close"],
            "output_names": ["PRIME_OSC", "PRIME_SIGNAL"],
            "default_output": "PRIME_OSC",
            "aliases": ["Prime_Oscillator"],
            "param_map": {
                "close": "close",
                "length": "length",
                "signal_length": "signal_length",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 50,
                    "description": "Prime Number Oscillator計算期間",
                },
                "signal_length": {
                    "default_value": 3,
                    "min_value": 2,
                    "max_value": 20,
                    "description": "Signal line計算期間",
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "multiple",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 14, "signal_length": 3},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > {left_operand}_1",
                "short": "{left_operand}_0 < {left_operand}_1",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_gt": 5, "short_lt": -5},
                "conservative": {"long_gt": 15, "short_lt": -15},
                "normal": {"long_gt": 10, "short_lt": -10},
            },
            "type": "original",
        },
    },
    "WPR": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.williams_r",
            "required_data": ["high", "low", "close"],
            "output_names": None,
            "default_output": None,
            "aliases": ["Williams_%R", "Williams_R"],
            "param_map": {
                "high": "high",
                "low": "low",
                "close": "close",
                "length": "length",
            },
            "parameters": {
                "length": {
                    "default_value": 14,
                    "min_value": 2,
                    "max_value": 100,
                    "description": "Williams %R計算期間",
                }
            },
            "pandas_function": "wpr",
            "data_column": None,
            "data_columns": ["High", "Low", "Close"],
            "returns": "single",
            "return_cols": None,
            "multi_column": True,
            "default_values": {"length": 14},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} < {threshold}",
                "short": "{left_operand} > {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_lt": 20, "short_gt": 80},
                "conservative": {"long_lt": 10, "short_gt": 90},
                "normal": {"long_lt": 15, "short_gt": 85},
            },
            "type": "momentum",
        },
    },
    "PSY": {
        "config": {
            "result_type": "single",
            "scale_type": "oscillator_0_100",
            "category": "momentum",
            "adapter_function": "app.services.indicators.technical_indicators.momentum.MomentumIndicators.psychological_line",
            "required_data": ["close"],
            "output_names": None,
            "default_output": None,
            "aliases": ["Psychological_Line", "PSY"],
            "param_map": {
                "close": "close",
                "length": "length",
                "offset": "offset",
            },
            "parameters": {
                "length": {
                    "default_value": 12,
                    "min_value": 5,
                    "max_value": 30,
                    "description": "Psychological Line計算期間",
                },
                "offset": {
                    "default_value": 0,
                    "min_value": -10,
                    "max_value": 10,
                    "description": "出力オフセット",
                }
            },
            "pandas_function": "psl",
            "data_column": "Close",
            "data_columns": None,
            "returns": "single",
            "return_cols": None,
            "multi_column": False,
            "default_values": {"length": 12, "offset": 0},
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand} > {threshold}",
                "short": "{left_operand} < {threshold}",
            },
            "scale_type": "oscillator_0_100",
            "thresholds": {
                "aggressive": {"long_gt": 75, "short_lt": 25},
                "conservative": {"long_gt": 80, "short_lt": 20},
                "normal": {"long_gt": 70, "short_lt": 30},
            },
            "type": "momentum",
        },
    },
    "FIBO_CYCLE": {
        "config": {
            "result_type": "complex",
            "scale_type": "momentum_zero_centered",
            "category": "original",
            "adapter_function": "app.services.indicators.technical_indicators.original.OriginalIndicators.fibonacci_cycle",
            "required_data": ["close"],
            "output_names": ["FIBO_CYCLE", "FIBO_SIGNAL"],
            "default_output": "FIBO_CYCLE",
            "aliases": ["Fibonacci_Cycle"],
            "param_map": {
                "close": "close",
                "cycle_periods": "cycle_periods",
                "fib_ratios": "fib_ratios",
            },
            "parameters": {
                "cycle_periods": {
                    "default_value": [8, 13, 21, 34, 55],
                    "min_value": [5, 8, 13],
                    "max_value": [21, 34, 55, 89, 144],
                    "description": "Fibonacci Cycle計算に使用する期間リスト",
                },
                "fib_ratios": {
                    "default_value": [0.618, 1.0, 1.618, 2.618],
                    "min_value": [0.5, 0.8, 1.2, 2.0],
                    "max_value": [0.8, 1.2, 2.0, 3.0],
                    "description": "Fibonacci比率リスト",
                },
            },
            "pandas_function": None,
            "data_column": None,
            "data_columns": None,
            "returns": "multiple",
            "return_cols": None,
            "multi_column": False,
            "default_values": {
                "cycle_periods": [8, 13, 21, 34, 55],
                "fib_ratios": [0.618, 1.0, 1.618, 2.618]
            },
            "min_length_func": None,
        },
        "yaml": {
            "conditions": {
                "long": "{left_operand}_0 > {left_operand}_1",
                "short": "{left_operand}_0 < {left_operand}_1",
            },
            "scale_type": "momentum_zero_centered",
            "thresholds": {
                "aggressive": {"long_gt": 0.5, "short_lt": -0.5},
                "conservative": {"long_gt": 2.0, "short_lt": -2.0},
                "normal": {"long_gt": 1.0, "short_lt": -1.0},
            },
            "type": "original",
        },
    },
}


SCALE_TYPES: Dict[str, Dict[str, Any]] = {
    "oscillator_0_100": {"range": [0, 100]},
    "oscillator_plus_minus_100": {"range": [-100, 100]},
    "momentum_zero_centered": {"range": None},
    "price_absolute": {"range": None},
}


DEFAULT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "oscillator_0_100": {
        "aggressive": {"long_lt": 48},
        "conservative": {"long_lt": 52},
        "normal": {"long_lt": 50},
    },
    "oscillator_plus_minus_100": {
        "aggressive": {"long_gt": -2},
        "conservative": {"long_gt": 2},
        "normal": {"long_gt": 0},
    },
    "momentum_zero_centered": {
        "aggressive": {"long_gt": -0.1},
        "conservative": {"long_gt": 0.1},
        "normal": {"long_gt": 0.0},
    },
}


def _create_parameter_configs(
    parameters: Dict[str, Dict[str, Any]],
) -> Dict[str, ParameterConfig]:
    result: Dict[str, ParameterConfig] = {}
    for name, meta in parameters.items():
        result[name] = ParameterConfig(
            name=name,
            default_value=meta.get("default_value"),
            min_value=meta.get("min_value"),
            max_value=meta.get("max_value"),
            description=meta.get("description"),
        )
    return result


def register_indicator_manifest(
    registry: Optional[IndicatorConfigRegistry] = None,
) -> None:
    target_registry = registry or indicator_registry
    target_registry.reset()

    for name, definition in MANIFEST.items():
        config_meta = definition["config"]
        result_type = (
            IndicatorResultType(config_meta["result_type"])
            if config_meta.get("result_type")
            else IndicatorResultType.SINGLE
        )
        scale_type = (
            IndicatorScaleType(config_meta["scale_type"])
            if config_meta.get("scale_type")
            else IndicatorScaleType.PRICE_RATIO
        )

        parameters = _create_parameter_configs(config_meta.get("parameters", {}))
        indicator_config = IndicatorConfig(
            indicator_name=name,
            adapter_function=_resolve_callable(config_meta.get("adapter_function")),
            required_data=config_meta.get("required_data", []),
            result_type=result_type,
            scale_type=scale_type,
            category=config_meta.get("category"),
            output_names=config_meta.get("output_names"),
            default_output=config_meta.get("default_output"),
            aliases=config_meta.get("aliases"),
            param_map=config_meta.get("param_map", {}),
            parameters=parameters,
            pandas_function=config_meta.get("pandas_function"),
            data_column=config_meta.get("data_column"),
            data_columns=config_meta.get("data_columns"),
            returns=config_meta.get("returns", "single"),
            return_cols=config_meta.get("return_cols"),
            multi_column=config_meta.get("multi_column", False),
            default_values=config_meta.get("default_values", {}),
            min_length_func=(
                _MIN_LENGTH_FUNCTIONS.get(name)
                if isinstance(config_meta.get("min_length_func"), str)
                else None
            ),
        )

        target_registry.register(indicator_config)


def manifest_to_yaml_dict() -> Dict[str, Any]:
    indicators = {name: data["yaml"] for name, data in MANIFEST.items()}
    return {
        "indicators": indicators,
        "scale_types": SCALE_TYPES,
        "default_thresholds": DEFAULT_THRESHOLDS,
    }
