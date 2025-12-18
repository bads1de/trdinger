"""
MLパイプライン モジュール

このモジュールは、前処理パイプラインを特徴量選択とスケーリング機能で拡張した
MLアルゴリズムに最適化された機械学習中心のパイプラインを提供します。

パイプラインはscikit-learnの慣例に従い、MLワークフローとシームレスに統合されます。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from app.utils.data_processing.pipelines.preprocessing_pipeline import (
    create_preprocessing_pipeline,
)

logger = logging.getLogger(__name__)


def _dataframe_to_array(X: Any) -> np.ndarray:
    """DataFrameをnumpy配列に変換"""
    if not isinstance(X, pd.DataFrame):
        return X
    if X.empty:
        return np.array([]).reshape(0, 0)
    return X.fillna(0.0).values


def create_ml_pipeline(
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    selection_method: str = "f_regression",
    scaling: bool = True,
    scaling_method: str = "standard",
    preprocessing_params: Optional[Dict[str, Any]] = None,
    is_classification: bool = False,
    **kwargs: Any,
) -> Pipeline:
    """MLパイプラインを作成（分類・回帰共通）"""
    from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression

    logger.info(f"{'分類' if is_classification else '回帰'}パイプラインを作成中...")

    steps = [("preprocessing", create_preprocessing_pipeline(**(preprocessing_params or {})))]

    # 特徴量選択
    if feature_selection and n_features is not None and n_features > 0:
        selectors = {
            "f_regression": f_regression, "f_classif": f_classif,
            "mutual_info": mutual_info_classif if is_classification else mutual_info_regression
        }
        if selection_method not in selectors:
            raise ValueError(f"サポートされていない選択方法: {selection_method}")
        steps.append(("feature_selection", SelectKBest(score_func=selectors[selection_method], k=n_features)))

    # スケーリング
    if scaling:
        scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}
        if scaling_method not in scalers:
            raise ValueError(f"サポートされていないスケーリング方法: {scaling_method}")
        steps.append(("scaler", scalers[scaling_method]))

    return Pipeline(steps)


def create_classification_pipeline(**kwargs) -> Pipeline:
    """分類用パイプライン"""
    kwargs.setdefault("selection_method", "f_classif")
    return create_ml_pipeline(is_classification=True, **kwargs)


def create_regression_pipeline(**kwargs) -> Pipeline:
    """回帰用パイプライン"""
    return create_ml_pipeline(is_classification=False, **kwargs)


def get_ml_pipeline_info(pipeline: Pipeline) -> Dict[str, Any]:
    """
    MLパイプラインの情報を取得。

    Args:
        pipeline: 適合済みのMLパイプライン

    Returns:
        パイプライン情報を含む辞書
    """
    info = {
        "pipeline_type": "ml",
        "n_steps": len(pipeline.steps),
        "step_names": [step[0] for step in pipeline.steps],
        "has_preprocessing": "preprocessing" in [step[0] for step in pipeline.steps],
        "has_feature_selection": "feature_selection"
        in [step[0] for step in pipeline.steps],
        "has_scaling": "scaler" in [step[0] for step in pipeline.steps],
    }

    if hasattr(pipeline, "feature_names_in_"):
        info["n_features_in"] = len(pipeline.feature_names_in_)

    try:
        if hasattr(pipeline, "get_feature_names_out"):
            feature_names_out = pipeline.get_feature_names_out()
            info["n_features_out"] = len(feature_names_out)
    except Exception:
        info["n_features_out"] = None

    return info


def optimize_ml_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "regression",
    max_features: Optional[int] = None,
) -> Pipeline:
    """
    データ特性に基づいて最適化されたMLパイプラインを作成。

    Args:
        X: 特徴量DataFrame
        y: ターゲット系列
        task_type: MLタスクの種類 ('regression', 'classification')
        max_features: 考慮する最大特徴量数

    Returns:
        最適化されたMLパイプライン
    """
    logger.info(f"{task_type}タスク用のMLパイプラインを最適化中...")

    n_features = X.shape[1]
    n_samples = X.shape[0]

    # Determine optimal number of features
    if max_features is None:
        max_features = min(n_features, max(5, n_samples // 10))

    # Choose appropriate scaling method based on data
    scaling_method = "robust" if task_type == "regression" else "standard"

    # Create optimized pipeline
    if task_type == "regression":
        pipeline = create_regression_pipeline(
            feature_selection=True,
            n_features=max_features,
            scaling_method=scaling_method,
        )
    elif task_type == "classification":
        pipeline = create_classification_pipeline(
            feature_selection=True,
            n_features=max_features,
            scaling_method=scaling_method,
        )
    else:
        pipeline = create_ml_pipeline(
            feature_selection=True,
            n_features=max_features,
            scaling_method=scaling_method,
        )

    logger.info(f"Optimized pipeline created with max {max_features} features")
    return pipeline



