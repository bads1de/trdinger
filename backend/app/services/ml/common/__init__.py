"""
ML共通ユーティリティモジュール

このモジュールは、MLサービス全体で使用される共通の定数、設定管理、
評価指標計算、およびユーティリティ関数を提供します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    calculate_true_range,
    calculate_volatility_atr,
    calculate_volatility_std,
    generate_cache_key,
    get_feature_importance_unified,
    optimize_dtypes,
    predict_class_from_proba,
    prepare_data_for_prediction,
    validate_training_inputs,
)

if TYPE_CHECKING:
    from .config import (
        MLConfigManager,
        get_default_ensemble_config,
        get_default_single_model_config,
        ml_config_manager,
    )

_CONFIG_EXPORTS = {
    "MLConfigManager": ".config",
    "ml_config_manager": ".config",
    "get_default_ensemble_config": ".config",
    "get_default_single_model_config": ".config",
}

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
    "calculate_true_range",
    "MLConfigManager",
    "ml_config_manager",
    "get_default_ensemble_config",
    "get_default_single_model_config",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _CONFIG_EXPORTS)
