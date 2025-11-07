"""
MLモデル評価ユーティリティ

モデルの予測結果を評価するための共通関数を提供します。
評価ロジックを一元化し、コードの重複を削減します。
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..evaluation.metrics import MetricsCalculator, MetricsConfig


def evaluate_model_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    予測結果を評価するための共通関数

    Args:
        y_true: 実際のターゲット値
        y_pred: 予測値
        y_pred_proba: 予測確率（オプション）

    Returns:
        評価指標の辞書
    """
    # 統一された評価指標計算器を使用
    config = MetricsConfig(
        include_balanced_accuracy=True,
        include_pr_auc=True,
        include_roc_auc=True,
        include_confusion_matrix=True,
        include_classification_report=True,
        average_method="weighted",
        zero_division="0",
    )

    metrics_calculator = MetricsCalculator(config)

    # numpy配列に変換（pandas Seriesの場合）
    y_true_array = y_true.values if hasattr(y_true, "values") else y_true

    # 包括的な評価指標を計算
    result = metrics_calculator.calculate_comprehensive_metrics(
        y_true_array, y_pred, y_pred_proba
    )

    return result
