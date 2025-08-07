"""
評価指標計算ユーティリティ

ML関連サービスで共通して使用される評価指標計算機能を提供します。
BaseMLTrainerから切り出された共通ロジックです。
"""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def calculate_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    詳細な評価指標を計算

    .. deprecated:: 2.1
        この関数は非推奨です。代わりに `EnhancedMetricsCalculator` を使用してください。

    Args:
        y_true: 実際のラベル
        y_pred: 予測ラベル
        y_pred_proba: 予測確率（オプション）

    Returns:
        評価指標の辞書
    """
    import warnings

    warnings.warn(
        "calculate_detailed_metrics は非推奨です。"
        "代わりに app.services.ml.evaluation.enhanced_metrics.EnhancedMetricsCalculator を使用してください。",
        DeprecationWarning,
        stacklevel=2,
    )
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
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
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
        else:  # 多クラス分類の場合
            # 各クラスの特異度と感度を計算し、平均を取る
            n_classes = cm.shape[0]
            specificities = []
            sensitivities = []
            npvs = []
            ppvs = []

            for i in range(n_classes):
                # クラスiに対する二値分類として計算
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp

                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

                specificities.append(specificity)
                sensitivities.append(sensitivity)
                npvs.append(npv)
                ppvs.append(ppv)

            # 重み付き平均を計算（クラス頻度で重み付け）
            class_weights = np.bincount(y_true) / len(y_true)
            metrics["specificity"] = float(
                np.average(specificities, weights=class_weights)
            )
            metrics["sensitivity"] = float(
                np.average(sensitivities, weights=class_weights)
            )
            metrics["npv"] = float(np.average(npvs, weights=class_weights))
            metrics["ppv"] = float(np.average(ppvs, weights=class_weights))

        # 確率ベースの指標（予測確率が利用可能な場合）
        if y_pred_proba is not None:
            try:
                n_unique_classes = len(np.unique(y_true))

                # AUC-ROC
                if n_unique_classes > 2:
                    # 多クラス分類
                    try:
                        metrics["auc_roc"] = float(
                            roc_auc_score(
                                y_true,
                                y_pred_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                        )
                    except ValueError as e:
                        logger.warning(f"多クラスAUC-ROC計算エラー: {e}")
                        metrics["auc_roc"] = 0.0
                else:
                    # 二値分類
                    try:
                        y_prob_positive = (
                            y_pred_proba[:, 1]
                            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1
                            else y_pred_proba.ravel()
                        )
                        metrics["auc_roc"] = float(
                            roc_auc_score(y_true, y_prob_positive)
                        )
                    except ValueError as e:
                        logger.warning(f"二値分類AUC-ROC計算エラー: {e}")
                        metrics["auc_roc"] = 0.0

                # PR-AUC (Precision-Recall AUC)
                if n_unique_classes == 2:
                    # 二値分類
                    try:
                        y_prob_positive = (
                            y_pred_proba[:, 1]
                            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1
                            else y_pred_proba.ravel()
                        )
                        metrics["auc_pr"] = float(
                            average_precision_score(y_true, y_prob_positive)
                        )
                    except ValueError as e:
                        logger.warning(f"二値分類PR-AUC計算エラー: {e}")
                        metrics["auc_pr"] = 0.0
                else:
                    # 多クラス分類：各クラスのPR-AUCを計算して平均
                    try:
                        pr_aucs = []
                        for i in range(y_pred_proba.shape[1]):
                            y_true_binary = (y_true == i).astype(int)
                            if (
                                np.sum(y_true_binary) > 0
                            ):  # クラスが存在する場合のみ計算
                                pr_auc = average_precision_score(
                                    y_true_binary, y_pred_proba[:, i]
                                )
                                pr_aucs.append(pr_auc)

                        if pr_aucs:
                            metrics["auc_pr"] = float(np.mean(pr_aucs))
                        else:
                            metrics["auc_pr"] = 0.0
                    except Exception as e:
                        logger.warning(f"多クラスPR-AUC計算エラー: {e}")
                        metrics["auc_pr"] = 0.0

                # Log Loss
                try:
                    metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                except ValueError as e:
                    logger.warning(f"Log Loss計算エラー: {e}")
                    metrics["log_loss"] = 0.0

                # Brier Score (二値分類のみ)
                if n_unique_classes == 2:
                    try:
                        y_prob_positive = (
                            y_pred_proba[:, 1]
                            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1
                            else y_pred_proba.ravel()
                        )
                        metrics["brier_score"] = float(
                            brier_score_loss(y_true, y_prob_positive)
                        )
                    except ValueError as e:
                        logger.warning(f"Brier Score計算エラー: {e}")
                        metrics["brier_score"] = 0.0
                else:
                    # 多クラス分類では各クラスのBrier Scoreを計算して平均
                    try:
                        brier_scores = []
                        for i in range(y_pred_proba.shape[1]):
                            y_true_binary = (y_true == i).astype(int)
                            brier_score = brier_score_loss(
                                y_true_binary, y_pred_proba[:, i]
                            )
                            brier_scores.append(brier_score)
                        metrics["brier_score"] = float(np.mean(brier_scores))
                    except Exception as e:
                        logger.warning(f"多クラスBrier Score計算エラー: {e}")
                        metrics["brier_score"] = 0.0

            except Exception as e:
                logger.warning(f"確率ベース指標計算エラー: {e}")
                # エラー時はデフォルト値を設定
                metrics.update(
                    {
                        "auc_roc": 0.0,
                        "auc_pr": 0.0,
                        "log_loss": 0.0,
                        "brier_score": 0.0,
                    }
                )

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
