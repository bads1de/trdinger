"""
モデル再学習と性能指標検証スクリプト

修正された性能指標計算ロジックを使用して
実際のモデルを再学習し、すべての指標が正しく表示されることを確認します。
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.ml.ml_training_service import MLTrainingService
from app.services.ml.model_manager import model_manager

logger = logging.getLogger(__name__)


def create_sample_training_data():
    """サンプル学習データを作成"""
    import numpy as np

    np.random.seed(42)

    # 1000サンプルのOHLCVデータを作成
    n_samples = 1000

    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="h"),
        "Open": 50000 + np.random.randn(n_samples) * 1000,
        "High": 50000 + np.random.randn(n_samples) * 1000 + 500,
        "Low": 50000 + np.random.randn(n_samples) * 1000 - 500,
        "Close": 50000 + np.random.randn(n_samples) * 1000,
        "Volume": np.random.uniform(100, 1000, n_samples),
    }

    df = pd.DataFrame(data)

    # 価格の整合性を保つ
    for i in range(len(df)):
        prices = [
            df.loc[i, "Open"],
            df.loc[i, "High"],
            df.loc[i, "Low"],
            df.loc[i, "Close"],
        ]
        df.loc[i, "High"] = max(prices)
        df.loc[i, "Low"] = min(prices)

    return df


def retrain_model_and_verify():
    """モデルを再学習して性能指標を検証"""
    print("=" * 80)
    print("モデル再学習と性能指標検証")
    print("=" * 80)

    try:
        # MLTrainingServiceを初期化
        ml_service = MLTrainingService(trainer_type="ensemble")

        # サンプルデータを作成
        print("1. サンプル学習データを作成中...")
        training_data = create_sample_training_data()
        print(f"   データサイズ: {len(training_data)}行")

        # モデルを学習
        print("\n2. モデル学習を開始...")
        result = ml_service.train_model(
            training_data=training_data,
            save_model=True,
            model_name="metrics_test_model",
        )

        print("✅ モデル学習完了")

        # 学習結果の性能指標を確認
        print("\n3. 学習結果の性能指標:")
        print("-" * 60)

        # 重要な指標を表示
        important_metrics = [
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

        for metric in important_metrics:
            value = result.get(metric, "N/A")
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

        # 0.0の指標をチェック
        zero_metrics = [
            k
            for k, v in result.items()
            if isinstance(v, (int, float)) and v == 0.0 and k in important_metrics
        ]
        if zero_metrics:
            print(f"\n⚠️  0.0の指標: {zero_metrics}")
        else:
            print("\n✅ すべての重要指標が0.0以外の値を持っています")

        # 保存されたモデルのメタデータを確認
        print("\n4. 保存されたモデルのメタデータ確認:")
        print("-" * 60)

        models = model_manager.list_models("*")
        if models:
            latest_model = models[0]  # 最新のモデル
            print(f"最新モデル: {latest_model['name']}")

            model_data = model_manager.load_model(latest_model["path"])
            if model_data and "metadata" in model_data:
                metadata = model_data["metadata"]

                print("\nメタデータの性能指標:")
                for metric in important_metrics:
                    value = metadata.get(metric, "N/A")
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")

                # メタデータでの0.0指標をチェック
                zero_metadata_metrics = [
                    k
                    for k, v in metadata.items()
                    if isinstance(v, (int, float))
                    and v == 0.0
                    and k in important_metrics
                ]
                if zero_metadata_metrics:
                    print(f"\n⚠️  メタデータで0.0の指標: {zero_metadata_metrics}")
                else:
                    print(
                        "\n✅ メタデータのすべての重要指標が0.0以外の値を持っています"
                    )
            else:
                print("メタデータが見つかりません")
        else:
            print("保存されたモデルが見つかりません")

        return result

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return None


def verify_api_response():
    """API応答での性能指標を確認"""
    print("\n" + "=" * 80)
    print("API応答での性能指標確認")
    print("=" * 80)

    try:
        from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator

        # MLOrchestratorを初期化
        orchestrator = MLOrchestrator()

        # ML状態を取得
        status = orchestrator.get_model_status()

        print("ML状態の性能指標:")
        print("-" * 60)

        performance_metrics = status.get("performance_metrics", {})

        important_metrics = [
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

        for metric in important_metrics:
            value = performance_metrics.get(metric, "N/A")
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

        # 0.0の指標をチェック
        zero_api_metrics = [
            k
            for k, v in performance_metrics.items()
            if isinstance(v, (int, float)) and v == 0.0 and k in important_metrics
        ]
        if zero_api_metrics:
            print(f"\n⚠️  API応答で0.0の指標: {zero_api_metrics}")
            return False
        else:
            print("\n✅ API応答のすべての重要指標が0.0以外の値を持っています")
            return True

    except Exception as e:
        print(f"❌ API確認中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("モデル再学習と性能指標検証開始")
    print("=" * 80)

    # モデル再学習と検証
    training_result = retrain_model_and_verify()

    if training_result:
        # API応答確認
        api_success = verify_api_response()

        print("\n" + "=" * 80)
        if api_success:
            print("✅ すべての検証が成功しました")
            print("フロントエンドで性能指標が正しく表示されるはずです")
        else:
            print("⚠️  一部の検証で問題が見つかりました")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ モデル学習に失敗しました")
        print("=" * 80)
