"""
メタラベリング（Fakeout Detection）用評価ユーティリティ

Meta-Labelingでは Precision（適合率）が最重要指標となります。
「エントリーした時にどれだけ勝てるか」を評価します。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_meta_labeling(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    メタラベリング用の評価指標を計算

    Precision（適合率）を最重要指標とし、
    「MLモデルがOKと判定したシグナルの勝率」を測定します。

    Args:
        y_true: 実際のターゲット値（0=失敗, 1=成功）
        y_pred: 予測値（0 or 1）
        y_pred_proba: 予測確率（オプション）
        threshold: 確率閾値（デフォルト: 0.5）

    Returns:
        評価指標の辞書
    """
    # numpy配列に変換
    y_true_array = y_true.values if hasattr(y_true, "values") else y_true

    # 確率から予測クラスを生成（閾値調整可能）
    if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
        # 2クラス分類の場合、クラス1の確率を使用
        y_pred_from_proba = (y_pred_proba[:, 1] >= threshold).astype(int)
    else:
        y_pred_from_proba = y_pred

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_from_proba).ravel()

    # Precision（適合率）- 最重要指標
    # MLがOKと言った時に実際に成功した割合
    precision = precision_score(y_true_array, y_pred_from_proba, zero_division=0.0)

    # Recall（再現率）
    # 実際の成功シグナルをどれだけ拾えたか
    recall = recall_score(y_true_array, y_pred_from_proba, zero_division=0.0)

    # F1-Score（精度と再現率のバランス）
    f1 = f1_score(y_true_array, y_pred_from_proba, zero_division=0.0)

    # Accuracy（全体の正答率）- メタラベリングではあまり重視しない
    accuracy = accuracy_score(y_true_array, y_pred_from_proba)

    # Specificity（特異度）
    # 失敗シグナルを正しく見抜けた割合
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Positive Predictive Value (PPV) = Precision
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Negative Predictive Value (NPV)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Win Rate（勝率）= Precision と同じだが明示的に
    win_rate = precision

    # シグナル採択率（何%のシグナルを採用したか）
    signal_adoption_rate = np.sum(y_pred_from_proba) / len(y_pred_from_proba)

    # Expected Value（期待値）の簡易計算
    # 勝ち時の利益を1、負け時の損失を-1と仮定
    expected_value = (precision * 1.0) + ((1 - precision) * -1.0)

    result = {
        # === メタラベリング最重要指標 ===
        "precision": precision,  # 最重要: MLがOKと言った時の勝率
        "win_rate": win_rate,  # Precision と同じだが明示的
        "f1_score": f1,  # 精度と再現率のバランス
        # === 補助指標 ===
        "recall": recall,  # 成功シグナルの検出率
        "accuracy": accuracy,  # 全体の正答率
        "specificity": specificity,  # 失敗シグナルの検出率
        # === 実用指標 ===
        "signal_adoption_rate": signal_adoption_rate,  # シグナル採択率
        "expected_value": expected_value,  # 期待値（簡易版）
        # === Confusion Matrix ===
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        # === その他 ===
        "ppv": ppv,
        "npv": npv,
        "total_samples": len(y_true_array),
        "positive_samples": int(np.sum(y_true_array)),
        "negative_samples": int(len(y_true_array) - np.sum(y_true_array)),
    }

    # 確率が利用可能な場合、ROC-AUCとPR-AUCを計算
    if y_pred_proba is not None:
        try:
            if len(y_pred_proba.shape) > 1:
                proba_positive = y_pred_proba[:, 1]
            else:
                proba_positive = y_pred_proba

            # ROC-AUC
            roc_auc = roc_auc_score(y_true_array, proba_positive)
            result["roc_auc"] = roc_auc

            # PR-AUC（Precision-Recall AUC）- メタラベリングで重要
            pr_auc = average_precision_score(y_true_array, proba_positive)
            result["pr_auc"] = pr_auc

        except Exception as e:
            logger.warning(f"ROC/PR-AUC計算エラー: {e}")
            result["roc_auc"] = 0.0
            result["pr_auc"] = 0.0

    return result


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
