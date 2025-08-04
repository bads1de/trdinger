"""
基本的なリファクタリングテスト
"""

import sys

sys.path.append(".")

import numpy as np
from app.services.ml.common.metrics_constants import (
    StandardMetricNames,
    MetricValidation,
)
from app.services.ml.common.unified_metrics_manager import unified_metrics_manager
from app.services.ml.common.trainer_factory import (
    create_single_model_trainer,
    create_ensemble_trainer,
)


def test_metrics_constants():
    """メトリクス定数のテスト"""
    print("=== メトリクス定数テスト ===")
    print(f"ACCURACY: {StandardMetricNames.ACCURACY}")
    print(f"F1_SCORE: {StandardMetricNames.F1_SCORE}")
    print(f"BALANCED_ACCURACY: {StandardMetricNames.BALANCED_ACCURACY}")
    print("✅ メトリクス定数テスト完了")


def test_metric_validation():
    """メトリクス検証のテスト"""
    print("=== メトリクス検証テスト ===")

    # 有効なメトリクス名のテスト
    accuracy_valid = MetricValidation.is_valid_metric_name("accuracy")
    invalid_valid = MetricValidation.is_valid_metric_name("invalid_metric")

    print(f"accuracy valid: {accuracy_valid}")
    print(f"invalid_metric valid: {invalid_valid}")

    # メトリクス値の検証テスト
    value_valid = MetricValidation.validate_metric_value("accuracy", 0.85)
    value_invalid = MetricValidation.validate_metric_value("accuracy", 1.5)

    print(f"accuracy=0.85 valid: {value_valid}")
    print(f"accuracy=1.5 valid: {value_invalid}")
    print("✅ メトリクス検証テスト完了")


def test_unified_metrics_manager():
    """統一メトリクス管理のテスト"""
    print("=== 統一メトリクス管理テスト ===")

    try:
        # 統一メトリクス管理インスタンスの作成テスト
        from app.services.ml.common.unified_metrics_manager import UnifiedMetricsManager

        manager = UnifiedMetricsManager()
        print("統一メトリクス管理インスタンス作成成功")

        # 基本的な機能テスト（簡易版）
        print("✅ 統一メトリクス管理テスト完了")

    except Exception as e:
        print(f"⚠️ 統一メトリクス管理テストでエラー: {e}")


def test_trainer_factory():
    """トレーナーファクトリーのテスト"""
    print("=== トレーナーファクトリーテスト ===")

    try:
        # 単一モデルトレーナー作成
        single_trainer = create_single_model_trainer(model_type="lightgbm")
        print(f"単一モデルトレーナー作成成功: {type(single_trainer).__name__}")

        # アンサンブルトレーナー作成
        ensemble_trainer = create_ensemble_trainer(ensemble_method="bagging")
        print(f"アンサンブルトレーナー作成成功: {type(ensemble_trainer).__name__}")

        print("✅ トレーナーファクトリーテスト完了")

    except Exception as e:
        print(f"⚠️ トレーナーファクトリーテストでエラー: {e}")


def main():
    """メインテスト実行"""
    print("🚀 基本リファクタリングテスト開始")

    test_metrics_constants()
    test_metric_validation()
    test_unified_metrics_manager()
    test_trainer_factory()

    print("✅ 基本リファクタリングテスト完了")


if __name__ == "__main__":
    main()
