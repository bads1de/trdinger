"""
MLモデル評価ユーティリティ

モデルの予測結果を評価するための共通関数を提供します。
一般指標とメタラベリング（Fakeout Detection）用の指標をカバーします。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# 統一されたMetricsCalculatorのグローバルインスタンスをインポート
from ..evaluation.metrics import metrics_collector

logger = logging.getLogger(__name__)


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
    y_true_array = y_true.values if hasattr(y_true, "values") else y_true
    return metrics_collector.calculate_comprehensive_metrics(
        y_true_array, y_pred, y_pred_proba
    )


def get_default_metrics() -> Dict[str, float]:
    """デフォルトの評価メトリクス辞書を返す（全て0.0初期化）"""
    keys = [
        "accuracy", "precision", "recall", "f1_score", "auc_score", "auc_roc", "auc_pr",
        "balanced_accuracy", "matthews_corrcoef", "cohen_kappa", "specificity", "sensitivity",
        "npv", "ppv", "log_loss", "brier_score", "loss", "val_accuracy", "val_loss", "training_time"
    ]
    return {k: 0.0 for k in keys}


# --- メタラベリング評価 ---


def evaluate_meta_labeling(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """メタラベリング用の評価指標を計算"""
    y_t = y_true.values if hasattr(y_true, "values") else y_true
    res = metrics_collector.calculate_comprehensive_metrics(y_t, y_pred, y_pred_proba) or {}

    # 必須キーのデフォルト値を保証
    for key in ["precision", "recall", "f1_score", "accuracy", "specificity", 
                "true_positives", "true_negatives", "false_positives", "false_negatives"]:
        if key not in res:
            res[key] = 0.0 if "positives" not in key and "negatives" not in key else 0

    p = res.get("precision", 0.0)
    res.update({
        "win_rate": p,
        "signal_adoption_rate": np.sum(y_pred) / len(y_pred) if len(y_pred) > 0 else 0.0,
        "expected_value": (p * 1.0) + ((1 - p) * -1.0),
        "total_samples": len(y_t),
        "positive_samples": int(np.sum(y_t)),
        "negative_samples": int(len(y_t) - np.sum(y_t))
    })
    
    for k, v in [("meta_f1", "f1_score"), ("meta_precision", "precision"), ("meta_recall", "recall")]:
        if v in res:
            res[k] = res[v]

    return res


def print_meta_labeling_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> None:
    """メタラベリング評価レポートを出力"""
    metrics = evaluate_meta_labeling(y_true, y_pred, y_pred_proba)

    print("\n" + "=" * 60)
    print("📊 Meta-Labeling Evaluation Report (Fakeout Detection)")
    print("=" * 60)

    print("\n🎯 最重要指標（Primary Metrics）:")
    print(f"  Precision (Win Rate):  {metrics['precision']:.4f}  ★最重要★")
    print(f"  F1-Score:              {metrics['f1_score']:.4f}")

    print("\n📈 補助指標（Secondary Metrics）:")
    print(f"  Recall (Sensitivity):  {metrics['recall']:.4f}")
    print(f"  Specificity:           {metrics['specificity']:.4f}")
    print(f"  Accuracy:              {metrics['accuracy']:.4f}")

    print("\n💰 実用指標（Practical Metrics）:")
    print(f"  Signal Adoption Rate:  {metrics['signal_adoption_rate']:.2%}")
    print(f"  Expected Value:        {metrics['expected_value']:.4f}")

    print("\n🔢 Confusion Matrix:")
    print(f"  True Positives (TP):   {metrics['true_positives']}")
    print(f"  True Negatives (TN):   {metrics['true_negatives']}")
    print(f"  False Positives (FP):  {metrics['false_positives']}")
    print(f"  False Negatives (FN):  {metrics['false_negatives']}")

    print("\n📊 データ分布:")
    print(f"  Total Samples:         {metrics['total_samples']}")
    print(f"  Positive Samples:      {metrics['positive_samples']}")
    print(f"  Negative Samples:      {metrics['negative_samples']}")

    if "roc_auc" in metrics:
        print("\n🎲 確率ベース指標:")
        print(f"  ROC-AUC:               {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:                {metrics['pr_auc']:.4f}")

    print("\n" + "=" * 60)
    print("\n💡 解釈ガイド:")
    if metrics["precision"] >= 0.60:
        print("  ✅ Precision >= 60%: 優秀なモデルです")
    elif metrics["precision"] >= 0.55:
        print("  ⚠️  Precision 55-60%: 実用的ですが改善の余地があります")
    else:
        print("  ❌ Precision < 55%: モデルの改善が必要です")

    if metrics["signal_adoption_rate"] < 0.1:
        print("  ⚠️  シグナル採択率が低い（<10%）: 機会損失の可能性")
    elif metrics["signal_adoption_rate"] > 0.5:
        print("  ⚠️  シグナル採択率が高い（>50%）: フィルタリングが甘い可能性")
    print("=" * 60 + "\n")


def find_optimal_threshold(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    metric: str = "precision",
    min_recall: float = 0.3,
) -> Dict[str, Any]:
    """最適な確率閾値を見つける"""
    y_true_array = y_true.values if hasattr(y_true, "values") else y_true

    if len(y_pred_proba.shape) > 1:
        proba_positive = y_pred_proba[:, 1]
    else:
        proba_positive = y_pred_proba

    precisions, recalls, thresholds = precision_recall_curve(
        y_true_array, proba_positive
    )

    valid_indices = recalls[:-1] >= min_recall
    if not np.any(valid_indices):
        logger.warning(f"Recall >= {min_recall} を満たす閾値が見つかりません")
        return {"optimal_threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    valid_precisions = precisions[:-1][valid_indices]
    valid_recalls = recalls[:-1][valid_indices]
    valid_thresholds = thresholds[valid_indices]

    if metric == "precision":
        best_idx = np.argmax(valid_precisions)
    elif metric == "f1":
        f1_scores = 2 * (valid_precisions * valid_recalls) / (valid_precisions + valid_recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    optimal_threshold = valid_thresholds[best_idx]
    metrics = evaluate_meta_labeling(y_true, (proba_positive >= optimal_threshold).astype(int), y_pred_proba, threshold=optimal_threshold)

    return {
        "optimal_threshold": float(optimal_threshold),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "signal_adoption_rate": metrics["signal_adoption_rate"],
        "expected_value": metrics["expected_value"],
    }
