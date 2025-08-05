#!/usr/bin/env python3
"""
シンプルな評価ロジック統一テスト

リファクタリング後の評価ロジックが正常に動作するかを簡単にテストします。
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from app.services.ml.evaluation.enhanced_metrics import (
    EnhancedMetricsCalculator,
    MetricsConfig,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_enhanced_metrics_comprehensive():
    """EnhancedMetricsCalculatorの包括的テスト"""
    logger.info("=== EnhancedMetricsCalculator包括的テスト ===")

    try:
        # テストデータ作成（より現実的なデータ）
        np.random.seed(42)
        n_samples = 100

        # 不均衡データを作成
        y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])

        # 予測データ（ある程度の精度を持つように）
        y_pred = y_true.copy()
        # 20%の予測を間違える
        wrong_indices = np.random.choice(
            n_samples, size=int(n_samples * 0.2), replace=False
        )
        for idx in wrong_indices:
            y_pred[idx] = np.random.choice([0, 1, 2])

        # 予測確率（softmax風に）
        logits = np.random.randn(n_samples, 3)
        y_pred_proba = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

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

        # 結果確認（多クラス分類では roc_auc_ovr を使用）
        required_metrics = [
            "accuracy",
            "balanced_accuracy",
            "f1_score",
            "precision",
            "recall",
            "roc_auc_ovr",
            "pr_auc_macro",
            "confusion_matrix",
            "classification_report",
        ]

        missing_metrics = []
        for metric in required_metrics:
            if metric not in metrics:
                missing_metrics.append(metric)

        if missing_metrics:
            logger.error(f"不足している評価指標: {missing_metrics}")
            return False

        # 指標の妥当性チェック
        assert (
            0 <= metrics["accuracy"] <= 1
        ), f"accuracy範囲エラー: {metrics['accuracy']}"
        assert (
            0 <= metrics["balanced_accuracy"] <= 1
        ), f"balanced_accuracy範囲エラー: {metrics['balanced_accuracy']}"
        assert (
            0 <= metrics["f1_score"] <= 1
        ), f"f1_score範囲エラー: {metrics['f1_score']}"

        logger.info("✅ 包括的評価指標計算成功")
        logger.info(f"   accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"   f1_score: {metrics['f1_score']:.4f}")
        logger.info(f"   precision: {metrics['precision']:.4f}")
        logger.info(f"   recall: {metrics['recall']:.4f}")
        logger.info(f"   roc_auc_ovr: {metrics['roc_auc_ovr']:.4f}")
        logger.info(f"   pr_auc_macro: {metrics['pr_auc_macro']:.4f}")
        logger.info(f"   総評価指標数: {len(metrics)}")

        return True

    except Exception as e:
        logger.error(f"包括的評価テストエラー: {e}")
        return False


def test_binary_classification():
    """二値分類での評価テスト"""
    logger.info("=== 二値分類評価テスト ===")

    try:
        # 二値分類テストデータ
        np.random.seed(42)
        n_samples = 80

        y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        y_pred = y_true.copy()

        # 15%の予測を間違える
        wrong_indices = np.random.choice(
            n_samples, size=int(n_samples * 0.15), replace=False
        )
        for idx in wrong_indices:
            y_pred[idx] = 1 - y_pred[idx]  # 0->1, 1->0

        # 二値分類の予測確率
        y_pred_proba = np.random.rand(n_samples, 2)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

        # 評価指標計算
        config = MetricsConfig()
        calculator = EnhancedMetricsCalculator(config)
        metrics = calculator.calculate_comprehensive_metrics(
            y_true, y_pred, y_pred_proba, class_names=["Negative", "Positive"]
        )

        # 二値分類特有の指標確認
        binary_metrics = ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc"]
        for metric in binary_metrics:
            assert metric in metrics, f"二値分類指標 {metric} が見つかりません"
            assert 0 <= metrics[metric] <= 1, f"{metric} 範囲エラー: {metrics[metric]}"

        logger.info("✅ 二値分類評価成功")
        logger.info(f"   accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"   roc_auc: {metrics['roc_auc']:.4f}")
        logger.info(f"   pr_auc: {metrics['pr_auc']:.4f}")

        return True

    except Exception as e:
        logger.error(f"二値分類評価テストエラー: {e}")
        return False


def test_multiclass_classification():
    """多クラス分類での評価テスト"""
    logger.info("=== 多クラス分類評価テスト ===")

    try:
        # 多クラス分類テストデータ（5クラス）
        np.random.seed(42)
        n_samples = 100
        n_classes = 5

        y_true = np.random.choice(range(n_classes), size=n_samples)
        y_pred = y_true.copy()

        # 25%の予測を間違える
        wrong_indices = np.random.choice(
            n_samples, size=int(n_samples * 0.25), replace=False
        )
        for idx in wrong_indices:
            y_pred[idx] = np.random.choice(range(n_classes))

        # 多クラス予測確率
        y_pred_proba = np.random.dirichlet([1] * n_classes, size=n_samples)

        # 評価指標計算
        config = MetricsConfig()
        calculator = EnhancedMetricsCalculator(config)
        metrics = calculator.calculate_comprehensive_metrics(
            y_true,
            y_pred,
            y_pred_proba,
            class_names=[f"Class_{i}" for i in range(n_classes)],
        )

        # 多クラス分類特有の指標確認
        multiclass_metrics = [
            "accuracy",
            "balanced_accuracy",
            "roc_auc_ovr",
            "pr_auc_macro",
        ]
        for metric in multiclass_metrics:
            assert metric in metrics, f"多クラス分類指標 {metric} が見つかりません"
            assert 0 <= metrics[metric] <= 1, f"{metric} 範囲エラー: {metrics[metric]}"

        logger.info("✅ 多クラス分類評価成功")
        logger.info(f"   accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"   roc_auc_ovr: {metrics['roc_auc_ovr']:.4f}")
        logger.info(f"   pr_auc_macro: {metrics['pr_auc_macro']:.4f}")

        return True

    except Exception as e:
        logger.error(f"多クラス分類評価テストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    logger.info("=" * 60)
    logger.info("🚀 シンプル評価ロジック統一テスト開始")
    logger.info("=" * 60)

    tests = [
        ("包括的評価", test_enhanced_metrics_comprehensive),
        ("二値分類", test_binary_classification),
        ("多クラス分類", test_multiclass_classification),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"{test_name}テスト実行エラー: {e}")
            results.append((test_name, False))

    # 結果サマリー
    logger.info("=" * 60)
    logger.info("📊 テスト結果サマリー")
    logger.info("=" * 60)

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        logger.info(f"{test_name}: {status}")

    logger.info(
        f"成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)"
    )

    if success_count == total_count:
        logger.info("🎉 全テスト成功！評価ロジック統一は正常に動作しています。")
        logger.info("✅ リファクタリング2.1は成功しました。")
        return True
    else:
        logger.warning(
            f"⚠️ 一部テストが失敗しました。({total_count-success_count}個の失敗)"
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
