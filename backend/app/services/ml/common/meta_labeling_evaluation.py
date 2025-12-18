"""
メタラベリング（Fakeout Detection）用評価ユーティリティ

Meta-Labelingでは Precision（適合率）が最重要指標となります。
「エントリーした時にどれだけ勝てるか」を評価します。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)


def evaluate_meta_labeling(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """メタラベリング用の評価指標を計算"""
    from ..evaluation.metrics import metrics_collector
    
    # 統一評価器で基本メトリクスを計算
    y_t = y_true.values if hasattr(y_true, "values") else y_true
    res = metrics_collector.calculate_comprehensive_metrics(y_t, y_pred, y_pred_proba)

    # メタラベリング固有の指標を追加
    p = res.get("precision", 0.0)
    res.update({
        "win_rate": p,
        "signal_adoption_rate": np.sum(y_pred) / len(y_pred) if len(y_pred) > 0 else 0.0,
        "expected_value": (p * 1.0) + ((1 - p) * -1.0),
        "total_samples": len(y_t),
        "positive_samples": int(np.sum(y_t)),
        "negative_samples": int(len(y_t) - np.sum(y_t))
    })
    
    # 互換性のためのキー追加
    for k, v in [("meta_f1", "f1_score"), ("meta_precision", "precision"), ("meta_recall", "recall")]:
        if v in res:
            res[k] = res[v]

    return res


def print_meta_labeling_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> None:
    """
    メタラベリング評価レポートを出力

    Args:
        y_true: 実際のターゲット値
        y_pred: 予測値
        y_pred_proba: 予測確率（オプション）
    """
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

    # 解釈ガイド
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
    """
    最適な確率閾値を見つける

    Meta-Labelingでは Precision を最大化しつつ、
    Recall が一定以上（機会損失を避ける）になる閾値を探します。

    Args:
        y_true: 実際のターゲット値
        y_pred_proba: 予測確率
        metric: 最適化する指標（"precision", "f1"）
        min_recall: 最小Recall制約（デフォルト: 0.3）

    Returns:
        最適閾値と各種指標の辞書
    """
    y_true_array = y_true.values if hasattr(y_true, "values") else y_true

    if len(y_pred_proba.shape) > 1:
        proba_positive = y_pred_proba[:, 1]
    else:
        proba_positive = y_pred_proba

    # Precision-Recall曲線を計算
    precisions, recalls, thresholds = precision_recall_curve(
        y_true_array, proba_positive
    )

    # Recall制約を満たす閾値のみを考慮
    valid_indices = recalls[:-1] >= min_recall

    if not np.any(valid_indices):
        logger.warning(f"Recall >= {min_recall} を満たす閾値が見つかりません")
        return {
            "optimal_threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    valid_precisions = precisions[:-1][valid_indices]
    valid_recalls = recalls[:-1][valid_indices]
    valid_thresholds = thresholds[valid_indices]

    if metric == "precision":
        # Precisionを最大化
        best_idx = np.argmax(valid_precisions)
    elif metric == "f1":
        # F1-Scoreを最大化
        f1_scores = (
            2
            * (valid_precisions * valid_recalls)
            / (valid_precisions + valid_recalls + 1e-10)
        )
        best_idx = np.argmax(f1_scores)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    optimal_threshold = valid_thresholds[best_idx]

    # 最適閾値での評価
    y_pred_optimal = (proba_positive >= optimal_threshold).astype(int)
    metrics = evaluate_meta_labeling(
        y_true, y_pred_optimal, y_pred_proba, threshold=optimal_threshold
    )

    result = {
        "optimal_threshold": float(optimal_threshold),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "signal_adoption_rate": metrics["signal_adoption_rate"],
        "expected_value": metrics["expected_value"],
    }

    logger.info(
        f"最適閾値: {optimal_threshold:.3f} "
        f"(Precision={metrics['precision']:.3f}, "
        f"Recall={metrics['recall']:.3f})"
    )

    return result



