"""
ML共通ユーティリティモジュール

このモジュールは、MLサービス全体で使用される共通の定数、設定管理、
評価指標計算、およびユーティリティ関数を提供します。
"""

from __future__ import annotations

from importlib import import_module

from .base_resource_manager import BaseResourceManager, CleanupLevel
from .exceptions import (
    MLBaseError,
    MLDataError,
    MLFeatureError,
    MLModelError,
    MLPredictionError,
    MLTrainingError,
    MLValidationError,
)
from .registry import AlgorithmRegistry, ModelMetadata, algorithm_registry
from .utils import (
    calculate_historical_volatility,
    calculate_price_change,
    calculate_realized_volatility,
    calculate_volatility_atr,
    calculate_volatility_std,
    generate_cache_key,
    get_feature_importance_unified,
    optimize_dtypes,
    predict_class_from_proba,
    prepare_data_for_prediction,
    validate_training_inputs,
)

_CONFIG_EXPORTS = {
    "MLConfigManager",
    "ml_config_manager",
    "get_default_ensemble_config",
    "get_default_single_model_config",
}


def __getattr__(name: str) -> type:
    if name in _CONFIG_EXPORTS:
        module = import_module(".config", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_CONFIG_EXPORTS})


__all__ = [
    "CleanupLevel",
    "BaseResourceManager",
    "MLBaseError",
    "MLDataError",
    "MLValidationError",
    "MLModelError",
    "MLTrainingError",
    "MLPredictionError",
    "MLFeatureError",
    "AlgorithmRegistry",
    "algorithm_registry",
    "ModelMetadata",
    "optimize_dtypes",
    "generate_cache_key",
    "validate_training_inputs",
    "prepare_data_for_prediction",
    "predict_class_from_proba",
    "get_feature_importance_unified",
    "calculate_price_change",
    "calculate_volatility_std",
    "calculate_volatility_atr",
    "calculate_historical_volatility",
    "calculate_realized_volatility",
    "MLConfigManager",
    "ml_config_manager",
    "get_default_ensemble_config",
    "get_default_single_model_config",
]
