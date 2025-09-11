"""
Data Processing Pipelines Package

This package provides modular, reusable pipelines for data processing workflows.
All pipelines follow scikit-learn conventions and can be used seamlessly in ML workflows.

Available Pipelines:
- PreprocessingPipeline: Basic data preprocessing (outlier removal, imputation, encoding)
- MLPipeline: ML-focused pipeline with feature selection and scaling
- ComprehensivePipeline: Complete end-to-end data processing pipeline

Usage:
    from backend.app.utils.data_processing.pipelines import (
        create_preprocessing_pipeline,
        create_ml_pipeline,
        create_comprehensive_pipeline
    )
"""

from .preprocessing_pipeline import (
    create_preprocessing_pipeline,
    create_basic_preprocessing_pipeline,
    get_pipeline_info as get_preprocessing_pipeline_info
)

from .ml_pipeline import (
    create_ml_pipeline,
    create_classification_pipeline,
    create_regression_pipeline,
    get_ml_pipeline_info,
    optimize_ml_pipeline
)

from .comprehensive_pipeline import (
    create_comprehensive_pipeline,
    create_production_pipeline,
    create_eda_pipeline,
    get_comprehensive_pipeline_info,
    validate_comprehensive_pipeline,
    optimize_comprehensive_pipeline
)

__all__ = [
    # Preprocessing Pipeline
    'create_preprocessing_pipeline',
    'create_basic_preprocessing_pipeline',
    'get_preprocessing_pipeline_info',

    # ML Pipeline
    'create_ml_pipeline',
    'create_classification_pipeline',
    'create_regression_pipeline',
    'get_ml_pipeline_info',
    'optimize_ml_pipeline',

    # Comprehensive Pipeline
    'create_comprehensive_pipeline',
    'create_production_pipeline',
    'create_eda_pipeline',
    'get_comprehensive_pipeline_info',
    'validate_comprehensive_pipeline',
    'optimize_comprehensive_pipeline',
]