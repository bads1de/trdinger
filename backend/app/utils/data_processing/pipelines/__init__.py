"""
データ処理パイプライン パッケージ

このパッケージは、データ処理ワークフロー用のモジュラーで再利用可能なパイプラインを提供します。
すべてのパイプラインはscikit-learnの慣例に従い、MLワークフローでシームレスに使用できます。

利用可能なパイプライン：
- PreprocessingPipeline: 基本的なデータ前処理（外れ値除去、補間、エンコーディング）

包括的パイプライン（ComprehensivePipeline）は app.services.ml.preprocessing に移動しました。

使用法：
    from backend.app.utils.data_processing.pipelines import (
        create_preprocessing_pipeline,
    )
"""

from .preprocessing_pipeline import (
    create_basic_preprocessing_pipeline,
    create_preprocessing_pipeline,
)
from .preprocessing_pipeline import get_pipeline_info as get_preprocessing_pipeline_info

__all__ = [
    # Preprocessing Pipeline
    "create_preprocessing_pipeline",
    "create_basic_preprocessing_pipeline",
    "get_preprocessing_pipeline_info",
]



