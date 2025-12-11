"""
包括的パイプライン モジュール

このモジュールは、利用可能なすべてのトランスフォーマーと処理ステップを組み合わせた
最も完全なデータ処理パイプラインを提供し、エンドツーエンドのデータ準備を実現します。

包括的パイプラインには以下が含まれます：
- 完全な前処理（外れ値除去、補間、エンコーディング、dtype最適化）
- 特徴量選択
- 特徴量スケーリング
- オプションの高度な変換

これは生産MLワークフローで最も機能豊富なパイプラインです。
"""

import logging
from typing import Any, Dict, Optional


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .preprocessing_pipeline import create_preprocessing_pipeline

logger = logging.getLogger(__name__)


def create_comprehensive_pipeline(
    # 前処理パラメータ
    outlier_removal: bool = True,
    outlier_method: str = "isolation_forest",
    outlier_contamination: float = 0.1,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    categorical_fill_value: str = "Unknown",
    categorical_encoding: str = "label",
    optimize_dtypes: bool = True,
    # MLパイプライン パラメータ
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    selection_method: str = "f_regression",
    scaling: bool = False,
    scaling_method: str = "standard",
    # 高度な機能
    polynomial_features: bool = False,
    polynomial_degree: int = 2,
    interaction_only: bool = False,
    # カスタム パラメータ
    preprocessing_params: Optional[Dict[str, Any]] = None,
    ml_params: Optional[Dict[str, Any]] = None,
    target_column: Optional[str] = None,
    **kwargs: Any,
) -> Pipeline:
    """
    包括的なデータ処理パイプラインを作成。

    このパイプラインは、生産MLワークフローに適した完全な
    エンドツーエンドのデータ準備のための利用可能なすべての処理ステップを組み合わせます。

    Args:
        outlier_removal: 外れ値除去を実行するかどうか
        outlier_method: 外れ値検出の方法
        outlier_contamination: 外れ値の予想割合
        numeric_strategy: 数値列の補間戦略
        categorical_strategy: カテゴリ列の補間戦略
        categorical_fill_value: カテゴリ欠損値の埋め値
        categorical_encoding: カテゴリ変数のエンコーディング方法
        optimize_dtypes: データタイプを最適化するかどうか

        feature_selection: 特徴量選択を実行するかどうか
        n_features: 選択する特徴量の数
        selection_method: 特徴量選択方法
        scaling: 特徴量スケーリングを適用するかどうか
        scaling_method: スケーリング方法

        polynomial_features: 多項式特徴量を追加するかどうか
        polynomial_degree: 多項式特徴量の次数
        interaction_only: 相互作用特徴量のみを含めるかどうか

        preprocessing_params: カスタム前処理パラメータ
        ml_params: カスタムMLパイプライン パラメータ
        target_column: ターゲット列の名前（特徴量選択用）

        **kwargs: 追加パラメータ

    Returns:
        設定された包括的パイプライン
    """
    logger.info("包括的パイプラインを作成中...")

    # Build preprocessing parameters
    if preprocessing_params is None:
        preprocessing_params = {}

    # Override with explicit parameters
    preprocessing_params.update(
        {
            "outlier_method": outlier_method if outlier_removal else None,
            "outlier_contamination": outlier_contamination,
            "numeric_strategy": numeric_strategy,
            "categorical_strategy": categorical_strategy,
            "categorical_fill_value": categorical_fill_value,
            "categorical_encoding": categorical_encoding,
            "optimize_dtypes": optimize_dtypes,
        }
    )

    # Create base preprocessing pipeline
    create_preprocessing_pipeline(**preprocessing_params)

    # Build ML pipeline parameters
    if ml_params is None:
        ml_params = {}

    ml_params.update(
        {
            "feature_selection": feature_selection,
            "n_features": n_features,
            "selection_method": selection_method,
            "scaling": scaling,
            "scaling_method": scaling_method,
            "preprocessing_params": preprocessing_params,
        }
    )

    # Create ML pipeline
    from app.services.ml.preprocessing.pipeline import create_ml_pipeline

    ml_pipeline = create_ml_pipeline(**ml_params)

    # Combine pipelines
    steps = [("ml_pipeline", ml_pipeline)]

    # リクエストされた場合多項式特徴量を追加
    if polynomial_features:
        poly_features = PolynomialFeatures(
            degree=polynomial_degree,
            interaction_only=interaction_only,
            include_bias=False,
        )
        steps.insert(0, ("polynomial_features", poly_features))
        logger.info(f"多項式特徴量を追加 (degree={polynomial_degree})")

    pipeline = Pipeline(steps)

    # Configure pipeline to output pandas DataFrames
    # This replaces the manual array_to_dataframe conversion and ensures
    # feature names are preserved throughout the pipeline
    try:
        pipeline.set_output(transform="pandas")
    except Exception as e:
        logger.warning(f"Could not set pandas output for pipeline: {e}")

    logger.info("包括的パイプラインが正常に作成されました")
    return pipeline


def create_production_pipeline(
    target_column: str,
    feature_selection: bool = True,
    n_features: Optional[int] = None,
    scaling_method: str = "robust",
    include_polynomial: bool = False,
) -> Pipeline:
    """
    生産対応の包括的パイプラインを作成。

    Args:
        target_column: ターゲット列の名前
        feature_selection: 特徴量選択を実行するかどうか
        n_features: 選択する特徴量の数
        scaling_method: 生産用のスケーリング方法
        include_polynomial: 多項式特徴量を含めるかどうか

    Returns:
        生産対応のパイプライン
    """
    logger.info("生産パイプラインを作成中...")

    return create_comprehensive_pipeline(
        # Robust preprocessing
        outlier_removal=True,
        outlier_method="isolation_forest",
        numeric_strategy="median",
        categorical_encoding="label",
        # ML features
        feature_selection=feature_selection,
        n_features=n_features,
        selection_method="f_regression",
        scaling=True,
        scaling_method=scaling_method,
        # Advanced features
        polynomial_features=include_polynomial,
        polynomial_degree=2,
        # Target specification
        target_column=target_column,
    )


def create_eda_pipeline(
    include_detailed_preprocessing: bool = True,
    include_feature_engineering: bool = False,
) -> Pipeline:
    """
    探索的データ分析用に最適化されたパイプラインを作成。

    Args:
        include_detailed_preprocessing: 詳細な前処理を含めるかどうか
        include_feature_engineering: 特徴量エンジニアリングを含めるかどうか

    Returns:
        EDA最適化パイプライン
    """
    logger.info("EDAパイプラインを作成中...")

    return create_comprehensive_pipeline(
        # Comprehensive preprocessing for EDA
        outlier_removal=include_detailed_preprocessing,
        optimize_dtypes=include_detailed_preprocessing,
        # Minimal ML features for EDA
        feature_selection=False,
        scaling=False,
        # Feature engineering if requested
        polynomial_features=include_feature_engineering,
        polynomial_degree=2,
        interaction_only=True,
    )


def get_comprehensive_pipeline_info(pipeline: Pipeline) -> Dict[str, Any]:
    """
    包括的パイプラインの詳細情報を取得。

    Args:
        pipeline: 適合済みの包括的パイプライン

    Returns:
        詳細なパイプライン情報を含む辞書
    """
    info = {
        "pipeline_type": "comprehensive",
        "n_steps": len(pipeline.steps),
        "step_names": [step[0] for step in pipeline.steps],
    }

    # Check for specific components
    step_names = [step[0] for step in pipeline.steps]
    info.update(
        {
            "has_preprocessing": any("preprocess" in name for name in step_names),
            "has_feature_selection": any("selection" in name for name in step_names),
            "has_scaling": any("scaler" in name for name in step_names),
            "has_polynomial_features": any("polynomial" in name for name in step_names),
        }
    )

    # Get ML pipeline info if available
    for step_name, step_obj in pipeline.steps:
        if step_name == "ml_pipeline":
            try:
                from app.services.ml.preprocessing.pipeline import get_ml_pipeline_info

                ml_info = get_ml_pipeline_info(step_obj)
                info["ml_pipeline_info"] = ml_info
            except Exception:
                pass
            break

    if hasattr(pipeline, "feature_names_in_"):
        info["n_features_in"] = len(pipeline.feature_names_in_)

    try:
        if hasattr(pipeline, "get_feature_names_out"):
            feature_names_out = pipeline.get_feature_names_out()
            info["n_features_out"] = len(feature_names_out)
    except Exception:
        info["n_features_out"] = None

    return info


def validate_comprehensive_pipeline(
    pipeline: Pipeline, X: pd.DataFrame, y: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    サンプルデータで包括的パイプラインを検証。

    Args:
        pipeline: 検証するパイプライン
        X: 特徴量DataFrame
        y: ターゲット系列（オプション）

    Returns:
        検証結果辞書
    """
    logger.info("包括的パイプラインを検証中...")

    validation_results = {
        "pipeline_creation": True,
        "fit_success": False,
        "transform_success": False,
        "output_shape": None,
        "processing_time": None,
        "errors": [],
    }

    try:
        import time

        start_time = time.time()

        # Test fit
        if y is not None:
            pipeline.fit(X, y)
        else:
            pipeline.fit(X)

        validation_results["fit_success"] = True

        # Test transform
        result = pipeline.transform(X)
        validation_results["transform_success"] = True
        validation_results["output_shape"] = (
            result.shape if hasattr(result, "shape") else None
        )

        validation_results["processing_time"] = time.time() - start_time

    except Exception as e:
        validation_results["errors"].append(str(e))
        logger.error(f"Pipeline validation error: {e}")

    return validation_results


def optimize_comprehensive_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "regression",
    time_budget: float = 60.0,
) -> Pipeline:
    """
    データ特性と制約に基づいて最適化された包括的パイプラインを作成。

    Args:
        X: 特徴量DataFrame
        y: ターゲット系列
        task_type: MLタスクの種類 ('regression', 'classification')
        time_budget: 最適化のための時間予算（秒）

    Returns:
        最適化された包括的パイプライン
    """
    logger.info(f"{task_type}用の包括的パイプラインを最適化中...")

    n_features = X.shape[1]
    n_samples = X.shape[0]

    # Determine optimal settings based on data size
    if n_samples < 1000:
        # Small dataset - use all features, minimal preprocessing
        feature_selection = False
        n_features_opt = None
        outlier_removal = False
    elif n_samples < 10000:
        # Medium dataset - moderate feature selection
        feature_selection = True
        n_features_opt = min(n_features, max(10, n_samples // 100))
        outlier_removal = True
    else:
        # Large dataset - aggressive feature selection
        feature_selection = True
        n_features_opt = min(n_features, max(20, n_samples // 500))
        outlier_removal = True

    # Choose scaling method based on task
    scaling_method = "robust" if task_type == "regression" else "standard"

    # Create optimized pipeline
    pipeline = create_comprehensive_pipeline(
        outlier_removal=outlier_removal,
        feature_selection=feature_selection,
        n_features=n_features_opt,
        selection_method="f_regression" if task_type == "regression" else "f_classif",
        scaling=True,
        scaling_method=scaling_method,
        polynomial_features=False,  # Disable by default for optimization
        target_column=y.name if hasattr(y, "name") else None,
    )

    logger.info(
        f"Optimized comprehensive pipeline created with {n_features_opt or 'all'} features"
    )
    return pipeline
