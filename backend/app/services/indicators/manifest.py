from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Optional

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
    IndicatorConfigRegistry,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
    indicator_registry,
)

from .manifests.momentum import MANIFEST_MOMENTUM
from .manifests.original import MANIFEST_ORIGINAL
from .manifests.price_transform import MANIFEST_PRICE_TRANSFORM
from .manifests.trend import MANIFEST_TREND
from .manifests.volatility import MANIFEST_VOLATILITY
from .manifests.volume import MANIFEST_VOLUME

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
    **MANIFEST_MOMENTUM,
    **MANIFEST_ORIGINAL,
    **MANIFEST_PRICE_TRANSFORM,
    **MANIFEST_TREND,
    **MANIFEST_VOLATILITY,
    **MANIFEST_VOLUME,
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
