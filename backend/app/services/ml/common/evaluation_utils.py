"""
MLモデル評価ユーティリティ

モデルの予測結果を評価するための共通関数を提供します。
評価ロジックを一元化し、コードの重複を削減します。
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# 統一されたMetricsCalculatorのグローバルインスタンスをインポート
from ..evaluation.metrics import metrics_collector


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
    # numpy配列に変換（pandas Seriesの場合）
    y_true_array = y_true.values if hasattr(y_true, "values") else y_true

    # グローバルなmetrics_collectorを使用して包括的な評価指標を計算
    # metrics_collectorは既に初期化済みで、設定を持っている
    result = metrics_collector.calculate_comprehensive_metrics(
        y_true_array, y_pred, y_pred_proba
    )

    return result


def get_default_metrics() -> Dict[str, float]:
    """デフォルトの評価メトリクス辞書を返す（全て0.0初期化）"""
    keys = [
        "accuracy", "precision", "recall", "f1_score", "auc_score", "auc_roc", "auc_pr",
        "balanced_accuracy", "matthews_corrcoef", "cohen_kappa", "specificity", "sensitivity",
        "npv", "ppv", "log_loss", "brier_score", "loss", "val_accuracy", "val_loss", "training_time"
    ]
    return {k: 0.0 for k in keys}



