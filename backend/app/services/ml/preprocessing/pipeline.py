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
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from app.utils.data_processing.pipelines.preprocessing_pipeline import (
    create_preprocessing_pipeline,
)

logger = logging.getLogger(__name__)


def _dataframe_to_array(X):
    """
    MLアルゴリズムのためにDataFrameをnumpy配列に変換。

    Args:
        X: 入力データ（DataFrameまたはarray-like）

    Returns:
        numpy配列に変換されたデータ
    """
    if isinstance(X, pd.DataFrame):
        # Check for NaN values before conversion
        if X.isnull().values.any():
            # Fill NaN values with 0 before conversion
            X = X.fillna(0.0)

        # Handle empty DataFrames
        if X.empty:
            # Return empty 2D array with proper shape
            return np.array([]).reshape(0, 0)

        return X.values
    return X


def create_ml_pipeline(
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    selection_method: str = "f_regression",
    scaling: bool = True,
    scaling_method: str = "standard",
    preprocessing_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Pipeline:
    """
    機械学習中心のパイプラインを作成。

    このパイプラインには以下が含まれます：
    - ベース前処理（外れ値除去、補間、エンコーディング、dtype最適化）
    - 特徴量選択（オプション）
    - 特徴量スケーリング（オプション）

    Args:
        feature_selection: 特徴量選択を実行するかどうか
        n_features: 選択する特徴量の数（Noneの場合、すべての特徴量を保持）
        selection_method: 特徴量選択方法 ('f_regression', 'mutual_info')
        scaling: 特徴量スケーリングを適用するかどうか
        scaling_method: スケーリング方法 ('standard', 'robust', 'minmax')
        preprocessing_params: 前処理パイプラインのパラメータ
        **kwargs: 追加パラメータ

    Returns:
        設定されたMLパイプライン
    """
    logger.info("MLパイプラインを作成中...")

    # ベース前処理パイプライン
    preprocessing_params = preprocessing_params or {}
    preprocessing_pipeline = create_preprocessing_pipeline(**preprocessing_params)

    # ML特有のステップ
    ml_steps = [("preprocessing", preprocessing_pipeline)]

    # 特徴量選択（オプション）
    if feature_selection and n_features is not None and n_features > 0:
        # 特徴量選択前に配列に変換
        ml_steps.append(
            ("to_array", FunctionTransformer(func=_dataframe_to_array, validate=False))
        )
        if selection_method == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=n_features)
        elif selection_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        else:
            raise ValueError(f"サポートされていない選択方法: {selection_method}")

        ml_steps.append(("feature_selection", selector))
        logger.info(
            f"{selection_method}を使用して{n_features}個の特徴量で特徴量選択を追加"
        )

    # 特徴量スケーリング（オプション）
    if scaling:
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"サポートされていないスケーリング方法: {scaling_method}")

        ml_steps.append(("scaler", scaler))
        logger.info(f"{scaling_method}スケーリングを追加")

    pipeline = Pipeline(ml_steps)

    logger.info("MLパイプラインが正常に作成されました")
    return pipeline


def create_classification_pipeline(
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    selection_method: str = "f_classif",
    scaling: bool = True,
    scaling_method: str = "standard",
    preprocessing_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Pipeline:
    """
    分類中心のMLパイプラインを作成。

    Args:
        feature_selection: 特徴量選択を実行するかどうか
        n_features: 選択する特徴量の数
        selection_method: 分類用の特徴量選択方法
        scaling: 特徴量スケーリングを適用するかどうか
        scaling_method: スケーリング方法
        preprocessing_params: 前処理パイプラインのパラメータ
        **kwargs: 追加パラメータ

    Returns:
        設定された分類パイプライン
    """
    logger.info("分類パイプラインを作成中...")

    from sklearn.feature_selection import f_classif, mutual_info_classif

    # Base preprocessing pipeline
    preprocessing_params = preprocessing_params or {}
    preprocessing_pipeline = create_preprocessing_pipeline(**preprocessing_params)

    # Classification-specific steps
    steps = [("preprocessing", preprocessing_pipeline)]

    # Feature selection for classification
    if feature_selection and n_features is not None and n_features > 0:
        # Convert to array before feature selection
        steps.append(
            ("to_array", FunctionTransformer(func=_dataframe_to_array, validate=False))
        )
        if selection_method == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif selection_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            raise ValueError(
                f"Unsupported classification selection method: {selection_method}"
            )

        steps.append(("feature_selection", selector))
        logger.info(
            f"Added classification feature selection with {n_features} features"
        )

    # Feature scaling
    if scaling:
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")

        steps.append(("scaler", scaler))
        logger.info(f"Added {scaling_method} scaling for classification")

    pipeline = Pipeline(steps)

    logger.info("Classification pipeline created successfully")
    return pipeline


def create_regression_pipeline(
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    selection_method: str = "f_regression",
    scaling: bool = True,
    scaling_method: str = "robust",
    preprocessing_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Pipeline:
    """
    回帰中心のMLパイプラインを作成。

    Args:
        feature_selection: 特徴量選択を実行するかどうか
        n_features: 選択する特徴量の数
        selection_method: 回帰用の特徴量選択方法
        scaling: 特徴量スケーリングを適用するかどうか
        scaling_method: スケーリング方法（回帰ではrobustがよく好まれる）
        preprocessing_params: 前処理パイプラインのパラメータ
        **kwargs: 追加パラメータ

    Returns:
        設定された回帰パイプライン
    """
    logger.info("回帰パイプラインを作成中...")

    # 回帰特有のパラメータを使用
    return create_ml_pipeline(
        feature_selection=feature_selection,
        n_features=n_features,
        selection_method=selection_method,
        scaling=scaling,
        scaling_method=scaling_method,
        preprocessing_params=preprocessing_params,
        **kwargs,
    )


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
