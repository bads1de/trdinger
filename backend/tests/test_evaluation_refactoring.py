#!/usr/bin/env python3
"""
評価ロジック統一のリファクタリングテスト

2.1のリファクタリング後の動作を検証します。
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
from app.services.ml.ensemble.base_ensemble import BaseEnsemble
from app.services.ml.models.lightgbm_wrapper import LightGBMModel

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_data(n_samples=100, n_features=5, n_classes=3):
    """テストデータを作成"""
    np.random.seed(42)

    # 特徴量データ
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # ターゲットデータ（不均衡データ）
    y = np.random.choice(range(n_classes), size=n_samples, p=[0.5, 0.3, 0.2])
    y = pd.Series(y, name="target")

    return X, y


def test_enhanced_metrics_calculator():
    """EnhancedMetricsCalculatorの基本動作テスト"""
    logger.info("=== EnhancedMetricsCalculator基本動作テスト ===")

    # テストデータ作成
    X, y = create_test_data(n_samples=100, n_features=5, n_classes=3)

    # 予測データを模擬
    y_pred = np.random.choice([0, 1, 2], size=len(y))
    y_pred_proba = np.random.dirichlet([1, 1, 1], size=len(y))

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
        y.values, y_pred, y_pred_proba, class_names=["Down", "Hold", "Up"]
    )

    # 結果確認
    assert "accuracy" in metrics, "accuracy指標が見つかりません"
    assert "balanced_accuracy" in metrics, "balanced_accuracy指標が見つかりません"
    assert "f1_score" in metrics, "f1_score指標が見つかりません"

    logger.info(f"✅ 基本指標計算成功: accuracy={metrics.get('accuracy', 0):.4f}")
    logger.info(f"   balanced_accuracy={metrics.get('balanced_accuracy', 0):.4f}")
    logger.info(f"   f1_score={metrics.get('f1_score', 0):.4f}")

    return metrics


def test_base_ensemble_evaluation():
    """BaseEnsembleの評価ロジックテスト"""
    logger.info("=== BaseEnsemble評価ロジックテスト ===")

    try:
        # テストデータ作成
        X, y = create_test_data(n_samples=50, n_features=3, n_classes=3)

        # 予測データを模擬
        y_pred = np.random.choice([0, 1, 2], size=len(y))
        y_pred_proba = np.random.dirichlet([1, 1, 1], size=len(y))

        # BaseEnsembleのインスタンス作成（テスト用）
        ensemble = BaseEnsemble()

        # _evaluate_predictionsメソッドをテスト
        metrics = ensemble._evaluate_predictions(y, y_pred, y_pred_proba)

        # 結果確認
        assert "accuracy" in metrics, "accuracy指標が見つかりません"
        assert "balanced_accuracy" in metrics, "balanced_accuracy指標が見つかりません"

        logger.info(
            f"✅ BaseEnsemble評価成功: accuracy={metrics.get('accuracy', 0):.4f}"
        )
        logger.info(f"   評価指標数: {len(metrics)}")

        return metrics

    except Exception as e:
        logger.error(f"BaseEnsemble評価テストエラー: {e}")
        return None


def test_lightgbm_model_evaluation():
    """LightGBMModelの評価ロジックテスト（インポートのみ）"""
    logger.info("=== LightGBMModel評価ロジックテスト ===")

    try:
        # LightGBMModelのインポートテスト
        model = LightGBMModel()
        logger.info("✅ LightGBMModel初期化成功")

        # 実際の学習は重いので、インポートと初期化のみテスト
        assert hasattr(model, "train"), "trainメソッドが見つかりません"
        assert hasattr(model, "predict"), "predictメソッドが見つかりません"

        logger.info("✅ LightGBMModel基本メソッド確認完了")
        return True

    except Exception as e:
        logger.error(f"LightGBMModel評価テストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    logger.info("=" * 60)
    logger.info("🚀 評価ロジック統一リファクタリング検証テスト開始")
    logger.info("=" * 60)

    results = {}

    try:
        # 1. EnhancedMetricsCalculator基本動作テスト
        results["enhanced_metrics"] = test_enhanced_metrics_calculator()

        # 2. BaseEnsemble評価ロジックテスト
        results["base_ensemble"] = test_base_ensemble_evaluation()

        # 3. LightGBMModel評価ロジックテスト
        results["lightgbm_model"] = test_lightgbm_model_evaluation()

        # 結果サマリー
        logger.info("=" * 60)
        logger.info("📊 テスト結果サマリー")
        logger.info("=" * 60)

        success_count = 0
        total_count = 3

        if results["enhanced_metrics"] is not None:
            logger.info("✅ EnhancedMetricsCalculator: 成功")
            success_count += 1
        else:
            logger.error("❌ EnhancedMetricsCalculator: 失敗")

        if results["base_ensemble"] is not None:
            logger.info("✅ BaseEnsemble評価ロジック: 成功")
            success_count += 1
        else:
            logger.error("❌ BaseEnsemble評価ロジック: 失敗")

        if results["lightgbm_model"]:
            logger.info("✅ LightGBMModel評価ロジック: 成功")
            success_count += 1
        else:
            logger.error("❌ LightGBMModel評価ロジック: 失敗")

        logger.info(
            f"成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)"
        )

        if success_count == total_count:
            logger.info("🎉 全テスト成功！リファクタリングは正常に動作しています。")
            return True
        else:
            logger.warning(
                f"⚠️ 一部テストが失敗しました。({total_count-success_count}個の失敗)"
            )
            return False

    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
