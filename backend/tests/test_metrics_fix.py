"""
性能指標修正のテストスクリプト

修正された性能指標計算ロジックをテストし、
すべての指標が正しく計算されることを確認します。
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.metrics_calculator import calculate_detailed_metrics
from app.services.ml.evaluation.enhanced_metrics import (
    EnhancedMetricsCalculator,
    MetricsConfig,
)

logger = logging.getLogger(__name__)


def create_test_data():
    """テスト用の3クラス分類データを作成"""
    np.random.seed(42)

    # 3クラス分類のテストデータ
    n_samples = 1000
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])

    # 予測値（少しノイズを加えて現実的に）
    y_pred = y_true.copy()
    # 10%の予測を間違える
    noise_indices = np.random.choice(
        n_samples, size=int(n_samples * 0.1), replace=False
    )
    for idx in noise_indices:
        # 異なるクラスにランダムに変更
        other_classes = [c for c in [0, 1, 2] if c != y_true[idx]]
        y_pred[idx] = np.random.choice(other_classes)

    # 予測確率（ソフトマックス風に）
    y_pred_proba = np.random.dirichlet([2, 1, 1], size=n_samples)
    # 正解クラスの確率を高くする
    for i in range(n_samples):
        y_pred_proba[i, y_true[i]] += 0.3
        y_pred_proba[i] = y_pred_proba[i] / np.sum(y_pred_proba[i])  # 正規化

    return y_true, y_pred, y_pred_proba


def test_metrics_calculator():
    """metrics_calculator.pyのテスト"""
    print("=" * 80)
    print("metrics_calculator.py のテスト")
    print("=" * 80)

    y_true, y_pred, y_pred_proba = create_test_data()

    print(f"テストデータ: {len(y_true)}サンプル, 3クラス分類")
    print(f"クラス分布: {np.bincount(y_true)}")
    print(f"予測確率形状: {y_pred_proba.shape}")

    # 性能指標を計算
    metrics = calculate_detailed_metrics(y_true, y_pred, y_pred_proba)

    print("\n計算された性能指標:")
    print("-" * 60)

    # 基本指標
    print("【基本指標】")
    print(f"  精度 (accuracy): {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  適合率 (precision): {metrics.get('precision', 'N/A'):.4f}")
    print(f"  再現率 (recall): {metrics.get('recall', 'N/A'):.4f}")
    print(f"  F1スコア (f1_score): {metrics.get('f1_score', 'N/A'):.4f}")

    # 高度な指標
    print("\n【高度な指標】")
    print(
        f"  バランス精度 (balanced_accuracy): {metrics.get('balanced_accuracy', 'N/A'):.4f}"
    )
    print(
        f"  マシューズ相関係数 (matthews_corrcoef): {metrics.get('matthews_corrcoef', 'N/A'):.4f}"
    )
    print(f"  コーエンのカッパ (cohen_kappa): {metrics.get('cohen_kappa', 'N/A'):.4f}")

    # AUC指標
    print("\n【AUC指標】")
    print(f"  AUC-ROC (auc_roc): {metrics.get('auc_roc', 'N/A'):.4f}")
    print(f"  AUC-PR (auc_pr): {metrics.get('auc_pr', 'N/A'):.4f}")

    # 専門指標
    print("\n【専門指標】")
    print(f"  特異度 (specificity): {metrics.get('specificity', 'N/A'):.4f}")
    print(f"  感度 (sensitivity): {metrics.get('sensitivity', 'N/A'):.4f}")
    print(f"  陰性的中率 (npv): {metrics.get('npv', 'N/A'):.4f}")
    print(f"  陽性的中率 (ppv): {metrics.get('ppv', 'N/A'):.4f}")

    # 確率指標
    print("\n【確率指標】")
    print(f"  対数損失 (log_loss): {metrics.get('log_loss', 'N/A'):.4f}")
    print(f"  ブライアスコア (brier_score): {metrics.get('brier_score', 'N/A'):.4f}")

    # 0.0の指標をチェック
    zero_metrics = [k for k, v in metrics.items() if v == 0.0]
    if zero_metrics:
        print(f"\n⚠️  0.0の指標: {zero_metrics}")
    else:
        print("\n✅ すべての指標が0.0以外の値を持っています")

    return metrics


def test_enhanced_metrics_calculator():
    """enhanced_metrics.pyのテスト"""
    print("\n" + "=" * 80)
    print("enhanced_metrics.py のテスト")
    print("=" * 80)

    y_true, y_pred, y_pred_proba = create_test_data()

    # 設定作成
    config = MetricsConfig(
        include_balanced_accuracy=True,
        include_pr_auc=True,
        include_roc_auc=True,
        include_confusion_matrix=True,
        include_classification_report=True,
        average_method="weighted",
        zero_division=0,
    )

    # 評価指標計算
    calculator = EnhancedMetricsCalculator(config)
    metrics = calculator.calculate_comprehensive_metrics(
        y_true, y_pred, y_pred_proba, class_names=["Down", "Hold", "Up"]
    )

    print("\n計算された性能指標:")
    print("-" * 60)

    # 重要な指標のみ表示
    important_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "balanced_accuracy",
        "matthews_corrcoef",
        "cohen_kappa",
        "roc_auc",
        "pr_auc",
        "specificity",
        "sensitivity",
        "npv",
        "ppv",
        "log_loss",
        "brier_score",
    ]

    for metric in important_metrics:
        value = metrics.get(metric, "N/A")
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # 0.0の指標をチェック
    zero_metrics = [
        k
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and v == 0.0 and k in important_metrics
    ]
    if zero_metrics:
        print(f"\n⚠️  0.0の指標: {zero_metrics}")
    else:
        print("\n✅ すべての重要指標が0.0以外の値を持っています")

    return metrics


def compare_results():
    """両方の結果を比較"""
    print("\n" + "=" * 80)
    print("結果比較")
    print("=" * 80)

    y_true, y_pred, y_pred_proba = create_test_data()

    # metrics_calculator
    metrics1 = calculate_detailed_metrics(y_true, y_pred, y_pred_proba)

    # enhanced_metrics
    config = MetricsConfig(
        include_balanced_accuracy=True,
        include_pr_auc=True,
        include_roc_auc=True,
        include_confusion_matrix=True,
        include_classification_report=True,
        average_method="weighted",
        zero_division=0,
    )
    calculator = EnhancedMetricsCalculator(config)
    metrics2 = calculator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)

    # 共通指標を比較
    common_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "balanced_accuracy",
        "matthews_corrcoef",
        "cohen_kappa",
        "auc_roc",
        "auc_pr",
        "specificity",
        "sensitivity",
        "npv",
        "ppv",
        "log_loss",
        "brier_score",
    ]

    print("指標比較 (metrics_calculator vs enhanced_metrics):")
    print("-" * 60)

    for metric in common_metrics:
        val1 = metrics1.get(metric, "N/A")
        val2 = metrics2.get(metric, "N/A")

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = abs(val1 - val2)
            status = "✅" if diff < 0.001 else "⚠️"
            print(f"  {metric}: {val1:.4f} vs {val2:.4f} {status}")
        else:
            print(f"  {metric}: {val1} vs {val2}")


if __name__ == "__main__":
    print("性能指標修正テスト開始")
    print("=" * 80)

    try:
        # 各テストを実行
        test_metrics_calculator()
        test_enhanced_metrics_calculator()
        compare_results()

        print("\n" + "=" * 80)
        print("✅ テスト完了")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
