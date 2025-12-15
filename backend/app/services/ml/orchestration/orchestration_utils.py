"""
オーケストレーション共通ユーティリティ

オーケストレーションサービス間で共通して使用される
モデル情報取得ロジックを提供します。
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from app.services.ml.common.evaluation_utils import get_default_metrics
from app.services.ml.model_manager import model_manager

logger = logging.getLogger(__name__)


def load_model_metadata_safely(model_path: str) -> Optional[Dict[str, Any]]:
    """
    モデルファイルからメタデータを安全に読み込む

    サイドカーJSONが存在する場合は軽量なJSONを読み込み、
    存在しない場合はjoblibからメタデータを抽出します。

    Args:
        model_path: モデルファイルパス

    Returns:
        読み込まれたモデルデータ(metadataキーを含む)、失敗時はNone
    """
    try:
        # 新しいload_metadata_onlyを使用（サイドカーJSON優先）
        metadata_data = model_manager.load_metadata_only(model_path)
        if not metadata_data or "metadata" not in metadata_data:
            return None
        return metadata_data
    except Exception as e:
        logger.warning(f"モデルメタデータ読み込みエラー {model_path}: {e}")
        return None


def get_latest_model_with_info(
    model_name_pattern: str = "*",
) -> Optional[Dict[str, Any]]:
    """
    最新モデルの情報とメトリクスを取得する共通ヘルパー

    このヘルパーは以下の情報を含む辞書を返します:
    - path: モデルファイルパス
    - metadata: モデルメタデータ
    - metrics: 抽出されたパフォーマンスメトリクス
    - file_info: ファイル情報(サイズ、更新時刻)

    Args:
        model_name_pattern: モデル名のパターン(デフォルト: "*")

    Returns:
        モデル情報辞書、モデルが見つからない場合や読み込みに失敗した場合はNone

    Example:
        >>> info = get_latest_model_with_info()
        >>> if info:
        ...     print(f"Model: {info['path']}")
        ...     print(f"Accuracy: {info['metrics']['accuracy']}")
    """
    try:
        latest_model = model_manager.get_latest_model(model_name_pattern)
        if not latest_model or not os.path.exists(latest_model):
            return None

        model_data = load_model_metadata_safely(latest_model)
        if not model_data:
            return None

        metadata = model_data["metadata"]

        # パフォーマンスメトリクスを抽出
        metrics = model_manager.extract_model_performance_metrics(
            latest_model, metadata=metadata
        )

        # ファイル情報を取得
        file_info = {
            "size_mb": os.path.getsize(latest_model) / (1024 * 1024),
            "modified_at": datetime.fromtimestamp(os.path.getmtime(latest_model)),
        }

        return {
            "path": latest_model,
            "metadata": metadata,
            "metrics": metrics,
            "file_info": file_info,
        }

    except Exception as e:
        logger.warning(f"最新モデル情報取得エラー: {e}")
        return None


def get_model_info_with_defaults(
    model_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    モデル情報にデフォルト値を適用

    Args:
        model_info: get_latest_model_with_infoから返されたモデル情報、またはNone

    Returns:
        デフォルト値が適用されたモデル情報辞書
    """
    if not model_info:
        # モデルが存在しない場合のデフォルト
        default_metrics = get_default_metrics()
        return {
            **default_metrics,
            "model_type": "No Model",
            "feature_count": 0,
            "training_samples": 0,
            "last_updated": "未学習",
            "file_size_mb": 0.0,
        }

    # モデル情報が存在する場合
    metadata = model_info["metadata"]
    # metricsキーが存在しない場合でも安全にアクセス
    metrics = model_info.get("metrics", get_default_metrics())
    file_info = model_info["file_info"]

    return {
        **metrics,
        "model_type": metadata.get("model_type", "Unknown"),
        "feature_count": metadata.get("feature_count", 0),
        "training_samples": metadata.get("training_samples", 0),
        "test_samples": metadata.get("test_samples", 0),
        "last_updated": file_info["modified_at"].isoformat(),
        "file_size_mb": file_info["size_mb"],
        "num_classes": metadata.get("num_classes", 2),
        "best_iteration": metadata.get("best_iteration", 0),
        "train_test_split": metadata.get("train_test_split", 0.8),
        "random_state": metadata.get("random_state", 42),
        "feature_importance": metadata.get("feature_importance", {}),
        "classification_report": metadata.get("classification_report", {}),
    }


