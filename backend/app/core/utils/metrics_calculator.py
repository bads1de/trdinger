"""
評価指標計算ユーティリティ

ML関連サービスで共通して使用される評価指標計算機能を提供します。
BaseMLTrainerから切り出された共通ロジックです。
"""

import logging
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

logger = logging.getLogger(__name__)


def calculate_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    詳細な評価指標を計算

    Args:
        y_true: 実際のラベル
        y_pred: 予測ラベル
        y_pred_proba: 予測確率（オプション）

    Returns:
        評価指標の辞書
    """
    metrics = {}

    try:
        # 基本的な評価指標
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1_score"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # 新しい評価指標
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(y_true, y_pred)
        )
        metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

        # 混同行列から特異度を計算
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # 二値分類の場合
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics["npv"] = (
                float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            )  # Negative Predictive Value
            metrics["ppv"] = (
                float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            )  # Positive Predictive Value

        # 確率ベースの指標（予測確率が利用可能な場合）
        if y_pred_proba is not None:
            try:
                # AUC-ROC
                if len(np.unique(y_true)) > 2:
                    # 多クラス分類
                    metrics["auc_roc"] = float(
                        roc_auc_score(
                            y_true,
                            y_pred_proba,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )
                else:
                    # 二値分類
                    metrics["auc_roc"] = float(
                        roc_auc_score(
                            y_true,
                            (
                                y_pred_proba[:, 1]
                                if y_pred_proba.ndim > 1
                                else y_pred_proba
                            ),
                        )
                    )

                # PR-AUC (Precision-Recall AUC)
                if len(np.unique(y_true)) == 2:
                    metrics["auc_pr"] = float(
                        average_precision_score(
                            y_true,
                            (
                                y_pred_proba[:, 1]
                                if y_pred_proba.ndim > 1
                                else y_pred_proba
                            ),
                        )
                    )

                # Log Loss
                metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))

                # Brier Score (二値分類のみ)
                if len(np.unique(y_true)) == 2:
                    y_prob_positive = (
                        y_pred_proba[:, 1]
                        if y_pred_proba.ndim > 1
                        else y_pred_proba
                    )
                    metrics["brier_score"] = float(
                        brier_score_loss(y_true, y_prob_positive)
                    )

            except Exception as e:
                logger.warning(f"確率ベース指標計算エラー: {e}")

    except Exception as e:
        logger.error(f"評価指標計算エラー: {e}")

    return metrics


def get_default_metrics() -> Dict[str, float]:
    """
    デフォルトの評価指標を返す

    Returns:
        デフォルト値が設定された評価指標の辞書
    """
    return {
        # 基本指標
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        # AUC指標
        "auc_roc": 0.0,
        "auc_pr": 0.0,
        # 高度な指標
        "balanced_accuracy": 0.0,
        "matthews_corrcoef": 0.0,
        "cohen_kappa": 0.0,
        # 専門指標
        "specificity": 0.0,
        "sensitivity": 0.0,
        "npv": 0.0,
        "ppv": 0.0,
        # 確率指標
        "log_loss": 0.0,
        "brier_score": 0.0,
    }
