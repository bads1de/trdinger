"""
Comprehensive Pipeline Module

This module provides the most complete data processing pipeline that combines
all available transformers and processing steps for end-to-end data preparation.

The comprehensive pipeline includes:
- Full preprocessing (outlier removal, imputation, encoding, dtype optimization)
- Feature selection
- Feature scaling
- Optional advanced transformations

This is the most feature-rich pipeline for production ML workflows.
"""

import logging
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

from .preprocessing_pipeline import create_preprocessing_pipeline
from .ml_pipeline import create_ml_pipeline

logger = logging.getLogger(__name__)


def create_comprehensive_pipeline(
    # Preprocessing parameters
    outlier_removal: bool = True,
    outlier_method: str = "isolation_forest",
    outlier_contamination: float = 0.1,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    categorical_fill_value: str = "Unknown",
    categorical_encoding: str = "label",
    optimize_dtypes: bool = True,

    # ML pipeline parameters
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    selection_method: str = "f_regression",
    scaling: bool = False,
    scaling_method: str = "standard",

    # Advanced features
    polynomial_features: bool = False,
    polynomial_degree: int = 2,
    interaction_only: bool = False,

    # Custom parameters
    preprocessing_params: Optional[Dict[str, Any]] = None,
    ml_params: Optional[Dict[str, Any]] = None,
    target_column: Optional[str] = None,

    **kwargs: Any
) -> Pipeline:
    """
    Create a comprehensive data processing pipeline.

    This pipeline combines all available processing steps for complete
    end-to-end data preparation suitable for production ML workflows.

    Args:
        outlier_removal: Whether to perform outlier removal
        outlier_method: Method for outlier detection
        outlier_contamination: Expected proportion of outliers
        numeric_strategy: Imputation strategy for numerical columns
        categorical_strategy: Imputation strategy for categorical columns
        categorical_fill_value: Fill value for categorical missing values
        categorical_encoding: Encoding method for categorical variables
        optimize_dtypes: Whether to optimize data types

        feature_selection: Whether to perform feature selection
        n_features: Number of features to select
        selection_method: Feature selection method
        scaling: Whether to apply feature scaling
        scaling_method: Scaling method

        polynomial_features: Whether to add polynomial features
        polynomial_degree: Degree for polynomial features
        interaction_only: Whether to include only interaction features

        preprocessing_params: Custom preprocessing parameters
        ml_params: Custom ML pipeline parameters
        target_column: Name of target column (for feature selection)

        **kwargs: Additional parameters

    Returns:
        Configured comprehensive Pipeline
    """
    logger.info("Creating comprehensive pipeline...")

    # Build preprocessing parameters
    if preprocessing_params is None:
        preprocessing_params = {}

    # Override with explicit parameters
    preprocessing_params.update({
        'outlier_method': outlier_method if outlier_removal else None,
        'outlier_contamination': outlier_contamination,
        'numeric_strategy': numeric_strategy,
        'categorical_strategy': categorical_strategy,
        'categorical_fill_value': categorical_fill_value,
        'categorical_encoding': categorical_encoding,
        'optimize_dtypes': optimize_dtypes,
    })

    # Create base preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline(**preprocessing_params)

    # Build ML pipeline parameters
    if ml_params is None:
        ml_params = {}

    ml_params.update({
        'feature_selection': feature_selection,
        'n_features': n_features,
        'selection_method': selection_method,
        'scaling': scaling,
        'scaling_method': scaling_method,
        'preprocessing_params': preprocessing_params,
    })

    # Create ML pipeline
    ml_pipeline = create_ml_pipeline(**ml_params)

    # Combine pipelines
    steps = [("ml_pipeline", ml_pipeline)]

    # Convert back to DataFrame if needed
    def array_to_dataframe(X):
        """Convert numpy array back to DataFrame if needed."""
        if isinstance(X, np.ndarray):
            # Try to get feature names from the pipeline
            try:
                feature_names = ml_pipeline.get_feature_names_out()
                return pd.DataFrame(X, columns=feature_names)
            except Exception:
                # Fallback to generic column names
                columns = [f'feature_{i}' for i in range(X.shape[1])]
                return pd.DataFrame(X, columns=columns)
        return X

    from sklearn.preprocessing import FunctionTransformer
    # Only add DataFrame converter if scaling is enabled (which returns numpy array)
    if scaling:
        steps.append(("to_dataframe", FunctionTransformer(func=array_to_dataframe, validate=False)))

    # Add polynomial features if requested
    if polynomial_features:
        poly_features = PolynomialFeatures(
            degree=polynomial_degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        steps.insert(0, ("polynomial_features", poly_features))
        logger.info(f"Added polynomial features (degree={polynomial_degree})")

    pipeline = Pipeline(steps)

    logger.info("Comprehensive pipeline created successfully")
    return pipeline


def create_production_pipeline(
    target_column: str,
    feature_selection: bool = True,
    n_features: Optional[int] = None,
    scaling_method: str = "robust",
    include_polynomial: bool = False
) -> Pipeline:
    """
    Create a production-ready comprehensive pipeline.

    Args:
        target_column: Name of the target column
        feature_selection: Whether to perform feature selection
        n_features: Number of features to select
        scaling_method: Scaling method for production
        include_polynomial: Whether to include polynomial features

    Returns:
        Production-ready Pipeline
    """
    logger.info("Creating production pipeline...")

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
        target_column=target_column
    )


def create_eda_pipeline(
    include_detailed_preprocessing: bool = True,
    include_feature_engineering: bool = False
) -> Pipeline:
    """
    Create a pipeline optimized for exploratory data analysis.

    Args:
        include_detailed_preprocessing: Whether to include detailed preprocessing
        include_feature_engineering: Whether to include feature engineering

    Returns:
        EDA-optimized Pipeline
    """
    logger.info("Creating EDA pipeline...")

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
        interaction_only=True
    )


def get_comprehensive_pipeline_info(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Get detailed information about a comprehensive pipeline.

    Args:
        pipeline: Fitted comprehensive Pipeline

    Returns:
        Dictionary with detailed pipeline information
    """
    info = {
        "pipeline_type": "comprehensive",
        "n_steps": len(pipeline.steps),
        "step_names": [step[0] for step in pipeline.steps],
    }

    # Check for specific components
    step_names = [step[0] for step in pipeline.steps]
    info.update({
        "has_preprocessing": any('preprocess' in name for name in step_names),
        "has_feature_selection": any('selection' in name for name in step_names),
        "has_scaling": any('scaler' in name for name in step_names),
        "has_polynomial_features": any('polynomial' in name for name in step_names),
    })

    # Get ML pipeline info if available
    for step_name, step_obj in pipeline.steps:
        if step_name == "ml_pipeline":
            try:
                from .ml_pipeline import get_ml_pipeline_info
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
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Validate a comprehensive pipeline with sample data.

    Args:
        pipeline: Pipeline to validate
        X: Feature DataFrame
        y: Target series (optional)

    Returns:
        Validation results dictionary
    """
    logger.info("Validating comprehensive pipeline...")

    validation_results = {
        "pipeline_creation": True,
        "fit_success": False,
        "transform_success": False,
        "output_shape": None,
        "processing_time": None,
        "errors": []
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
        validation_results["output_shape"] = result.shape if hasattr(result, 'shape') else None

        validation_results["processing_time"] = time.time() - start_time

    except Exception as e:
        validation_results["errors"].append(str(e))
        logger.error(f"Pipeline validation error: {e}")

    return validation_results


def optimize_comprehensive_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "regression",
    time_budget: float = 60.0
) -> Pipeline:
    """
    Create an optimized comprehensive pipeline based on data characteristics and constraints.

    Args:
        X: Feature DataFrame
        y: Target series
        task_type: Type of ML task ('regression', 'classification')
        time_budget: Time budget in seconds for optimization

    Returns:
        Optimized comprehensive Pipeline
    """
    logger.info(f"Optimizing comprehensive pipeline for {task_type}...")

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
        target_column=y.name if hasattr(y, 'name') else None
    )

    logger.info(f"Optimized comprehensive pipeline created with {n_features_opt or 'all'} features")
    return pipeline