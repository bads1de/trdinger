"""
ML Pipeline Module

This module provides a machine learning-focused pipeline that extends the preprocessing
pipeline with feature selection and scaling capabilities optimized for ML algorithms.

The pipeline follows scikit-learn conventions and integrates seamlessly with ML workflows.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from .preprocessing_pipeline import create_preprocessing_pipeline

logger = logging.getLogger(__name__)


def create_ml_pipeline(
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    selection_method: str = "f_regression",
    scaling: bool = True,
    scaling_method: str = "standard",
    preprocessing_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Pipeline:
    """
    Create a machine learning-focused pipeline.

    This pipeline includes:
    - Base preprocessing (outlier removal, imputation, encoding, dtype optimization)
    - Feature selection (optional)
    - Feature scaling (optional)

    Args:
        feature_selection: Whether to perform feature selection
        n_features: Number of features to select (if None, keeps all features)
        selection_method: Feature selection method ('f_regression', 'mutual_info')
        scaling: Whether to apply feature scaling
        scaling_method: Scaling method ('standard', 'robust', 'minmax')
        preprocessing_params: Parameters for the preprocessing pipeline
        **kwargs: Additional parameters

    Returns:
        Configured ML Pipeline
    """
    logger.info("Creating ML pipeline...")

    # Base preprocessing pipeline
    preprocessing_params = preprocessing_params or {}
    preprocessing_pipeline = create_preprocessing_pipeline(**preprocessing_params)

    # Convert DataFrame to array for ML steps
    def dataframe_to_array(X):
        """Convert DataFrame to numpy array for ML algorithms."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return X

    from sklearn.preprocessing import FunctionTransformer

    # ML-specific steps
    ml_steps = [("preprocessing", preprocessing_pipeline)]

    # Feature selection (optional)
    if feature_selection and n_features is not None and n_features > 0:
        # Convert to array before feature selection
        ml_steps.append(("to_array", FunctionTransformer(func=dataframe_to_array, validate=False)))
        if selection_method == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=n_features)
        elif selection_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        else:
            raise ValueError(f"Unsupported selection method: {selection_method}")

        ml_steps.append(("feature_selection", selector))
        logger.info(f"Added feature selection with {n_features} features using {selection_method}")

    # Feature scaling (optional)
    if scaling:
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")

        ml_steps.append(("scaler", scaler))
        logger.info(f"Added {scaling_method} scaling")

    pipeline = Pipeline(ml_steps)

    logger.info("ML pipeline created successfully")
    return pipeline


def create_classification_pipeline(
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    selection_method: str = "f_classif",
    scaling: bool = True,
    scaling_method: str = "standard",
    preprocessing_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Pipeline:
    """
    Create a classification-focused ML pipeline.

    Args:
        feature_selection: Whether to perform feature selection
        n_features: Number of features to select
        selection_method: Feature selection method for classification
        scaling: Whether to apply feature scaling
        scaling_method: Scaling method
        preprocessing_params: Parameters for the preprocessing pipeline
        **kwargs: Additional parameters

    Returns:
        Configured classification Pipeline
    """
    logger.info("Creating classification pipeline...")

    from sklearn.feature_selection import f_classif, mutual_info_classif

    # Base preprocessing pipeline
    preprocessing_params = preprocessing_params or {}
    preprocessing_pipeline = create_preprocessing_pipeline(**preprocessing_params)

    # Convert DataFrame to array for ML steps
    def dataframe_to_array(X):
        """Convert DataFrame to numpy array for ML algorithms."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return X

    from sklearn.preprocessing import FunctionTransformer

    # Classification-specific steps
    steps = [("preprocessing", preprocessing_pipeline)]

    # Feature selection for classification
    if feature_selection and n_features is not None and n_features > 0:
        # Convert to array before feature selection
        steps.append(("to_array", FunctionTransformer(func=dataframe_to_array, validate=False)))
        if selection_method == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif selection_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            raise ValueError(f"Unsupported classification selection method: {selection_method}")

        steps.append(("feature_selection", selector))
        logger.info(f"Added classification feature selection with {n_features} features")

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
    **kwargs: Any
) -> Pipeline:
    """
    Create a regression-focused ML pipeline.

    Args:
        feature_selection: Whether to perform feature selection
        n_features: Number of features to select
        selection_method: Feature selection method for regression
        scaling: Whether to apply feature scaling
        scaling_method: Scaling method (robust is often preferred for regression)
        preprocessing_params: Parameters for the preprocessing pipeline
        **kwargs: Additional parameters

    Returns:
        Configured regression Pipeline
    """
    logger.info("Creating regression pipeline...")

    # Use regression-specific parameters
    return create_ml_pipeline(
        feature_selection=feature_selection,
        n_features=n_features,
        selection_method=selection_method,
        scaling=scaling,
        scaling_method=scaling_method,
        preprocessing_params=preprocessing_params,
        **kwargs
    )


def get_ml_pipeline_info(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Get information about an ML pipeline.

    Args:
        pipeline: Fitted ML Pipeline

    Returns:
        Dictionary with pipeline information
    """
    info = {
        "pipeline_type": "ml",
        "n_steps": len(pipeline.steps),
        "step_names": [step[0] for step in pipeline.steps],
        "has_preprocessing": "preprocessing" in [step[0] for step in pipeline.steps],
        "has_feature_selection": "feature_selection" in [step[0] for step in pipeline.steps],
        "has_scaling": "scaler" in [step[0] for step in pipeline.steps]
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
    max_features: Optional[int] = None
) -> Pipeline:
    """
    Create an optimized ML pipeline based on data characteristics.

    Args:
        X: Feature DataFrame
        y: Target series
        task_type: Type of ML task ('regression', 'classification')
        max_features: Maximum number of features to consider

    Returns:
        Optimized ML Pipeline
    """
    logger.info(f"Optimizing ML pipeline for {task_type} task...")

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
            scaling_method=scaling_method
        )
    elif task_type == "classification":
        pipeline = create_classification_pipeline(
            feature_selection=True,
            n_features=max_features,
            scaling_method=scaling_method
        )
    else:
        pipeline = create_ml_pipeline(
            feature_selection=True,
            n_features=max_features,
            scaling_method=scaling_method
        )

    logger.info(f"Optimized pipeline created with max {max_features} features")
    return pipeline