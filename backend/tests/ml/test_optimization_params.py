"""
最適化パラメータが正しく適用されているかをテストするスクリプト
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.ml.lightgbm_trainer import LightGBMTrainer
from app.core.services.ml.ml_training_service import (
    MLTrainingService,
    OptimizationSettings,
)


def create_test_data():
    """テスト用のダミーデータを作成"""
    np.random.seed(42)

    # 1000行のOHLCVデータを作成
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="1H")

    data = {
        "timestamp": dates,
        "Open": np.random.uniform(40000, 50000, 1000),
        "High": np.random.uniform(40000, 50000, 1000),
        "Low": np.random.uniform(40000, 50000, 1000),
        "Close": np.random.uniform(40000, 50000, 1000),
        "Volume": np.random.uniform(100, 1000, 1000),
    }

    # HighとLowを調整
    for i in range(1000):
        data["High"][i] = max(data["Open"][i], data["Close"][i]) + np.random.uniform(
            0, 100
        )
        data["Low"][i] = min(data["Open"][i], data["Close"][i]) - np.random.uniform(
            0, 100
        )

    return pd.DataFrame(data)


def test_parameter_application():
    """パラメータが正しく適用されるかテスト"""
    print("🧪 パラメータ適用テストを開始...")

    # テストデータを作成
    training_data = create_test_data()

    # LightGBMTrainerを直接テスト
    trainer = LightGBMTrainer()

    # テスト1: デフォルトパラメータでの学習
    print("\n📊 テスト1: デフォルトパラメータ")
    result1 = trainer.train_model(
        training_data=training_data, save_model=False, test_size=0.2, random_state=42
    )
    print(f"デフォルト結果: 精度={result1.get('accuracy', 'N/A'):.4f}")

    # テスト2: 異なるlearning_rateでの学習
    print("\n📊 テスト2: learning_rate=0.01")
    result2 = trainer.train_model(
        training_data=training_data,
        save_model=False,
        test_size=0.2,
        random_state=42,
        learning_rate=0.01,  # デフォルトより小さい値
    )
    print(f"learning_rate=0.01結果: 精度={result2.get('accuracy', 'N/A'):.4f}")

    # テスト3: 異なるnum_leavesでの学習
    print("\n📊 テスト3: num_leaves=10")
    result3 = trainer.train_model(
        training_data=training_data,
        save_model=False,
        test_size=0.2,
        random_state=42,
        num_leaves=10,  # デフォルトより小さい値
    )
    print(f"num_leaves=10結果: 精度={result3.get('accuracy', 'N/A'):.4f}")

    # テスト4: 複数パラメータ同時変更
    print("\n📊 テスト4: 複数パラメータ変更")
    result4 = trainer.train_model(
        training_data=training_data,
        save_model=False,
        test_size=0.2,
        random_state=42,
        learning_rate=0.2,
        num_leaves=50,
        feature_fraction=0.7,
    )
    print(f"複数パラメータ変更結果: 精度={result4.get('accuracy', 'N/A'):.4f}")

    # 結果比較
    print("\n📈 結果比較:")
    results = [
        ("デフォルト", result1.get("accuracy", 0)),
        ("learning_rate=0.01", result2.get("accuracy", 0)),
        ("num_leaves=10", result3.get("accuracy", 0)),
        ("複数パラメータ", result4.get("accuracy", 0)),
    ]

    for name, accuracy in results:
        print(f"  {name}: {accuracy:.4f}")

    # 精度が異なることを確認
    accuracies = [r[1] for r in results]
    unique_accuracies = len(set(accuracies))

    if unique_accuracies > 1:
        print("✅ パラメータが正しく適用されています（精度が変化している）")
        return True
    else:
        print("❌ パラメータが適用されていません（精度が同じ）")
        return False


def test_optimization_service():
    """最適化サービスのテスト"""
    print("\n🚀 最適化サービステストを開始...")

    # テストデータを作成
    training_data = create_test_data()

    # 最適化設定
    optimization_settings = OptimizationSettings(
        enabled=True,
        method="random",
        n_calls=3,  # テスト用に少なく設定
        parameter_space={
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.2},
            "num_leaves": {"type": "integer", "low": 10, "high": 50},
        },
    )

    # MLTrainingServiceでテスト
    service = MLTrainingService()

    try:
        result = service.train_model(
            training_data=training_data,
            save_model=False,
            optimization_settings=optimization_settings,
            random_state=42,
        )

        print(
            f"最適化結果: {result.get('optimization_result', {}).get('best_score', 'N/A')}"
        )
        print("✅ 最適化サービスが正常に動作しています")
        return True

    except Exception as e:
        print(f"❌ 最適化サービスでエラー: {e}")
        return False


if __name__ == "__main__":
    print("🔬 ML最適化パラメータテストを開始")
    print("=" * 60)

    # パラメータ適用テスト
    param_test_passed = test_parameter_application()

    # 最適化サービステスト
    optimization_test_passed = test_optimization_service()

    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー:")
    print(f"  パラメータ適用テスト: {'✅ PASS' if param_test_passed else '❌ FAIL'}")
    print(
        f"  最適化サービステスト: {'✅ PASS' if optimization_test_passed else '❌ FAIL'}"
    )

    if param_test_passed and optimization_test_passed:
        print("\n🎉 全てのテストが成功しました！")
    else:
        print("\n⚠️  一部のテストが失敗しました。")
