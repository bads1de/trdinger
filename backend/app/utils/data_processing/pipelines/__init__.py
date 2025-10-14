"""
データ処理パイプライン パッケージ

このパッケージは、データ処理ワークフロー用のモジュラーで再利用可能なパイプラインを提供します。
すべてのパイプラインはscikit-learnの慣例に従い、MLワークフローでシームレスに使用できます。

利用可能なパイプライン：
- PreprocessingPipeline: 基本的なデータ前処理（外れ値除去、補間、エンコーディング）
- MLPipeline: 特徴量選択とスケーリングを備えたML中心のパイプライン
- ComprehensivePipeline: 完全なエンドツーエンドのデータ処理パイプライン

使用法：
    from backend.app.utils.data_processing.pipelines import (
        create_preprocessing_pipeline,
        create_ml_pipeline,
        create_comprehensive_pipeline
    )
"""

from .preprocessing_pipeline import (
    create_preprocessing_pipeline,
    create_basic_preprocessing_pipeline,
    get_pipeline_info as get_preprocessing_pipeline_info,
)

from .ml_pipeline import (
    create_ml_pipeline,
    create_classification_pipeline,
    create_regression_pipeline,
    get_ml_pipeline_info,
    optimize_ml_pipeline,
)

from .comprehensive_pipeline import (
    create_comprehensive_pipeline,
    create_production_pipeline,
    create_eda_pipeline,
    get_comprehensive_pipeline_info,
    validate_comprehensive_pipeline,
    optimize_comprehensive_pipeline,
)

__all__ = [
    # Preprocessing Pipeline
    "create_preprocessing_pipeline",
    "create_basic_preprocessing_pipeline",
    "get_preprocessing_pipeline_info",
    # ML Pipeline
    "create_ml_pipeline",
    "create_classification_pipeline",
    "create_regression_pipeline",
    "get_ml_pipeline_info",
    "optimize_ml_pipeline",
    # Comprehensive Pipeline
    "create_comprehensive_pipeline",
    "create_production_pipeline",
    "create_eda_pipeline",
    "get_comprehensive_pipeline_info",
    "validate_comprehensive_pipeline",
    "optimize_comprehensive_pipeline",
]
