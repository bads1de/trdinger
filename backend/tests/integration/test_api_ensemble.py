"""
アンサンブル学習API統合テスト

APIエンドポイント経由でアンサンブル学習が正しく動作することを確認します。
"""

import sys
import os
import json
import asyncio
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.api.ml_training import (
    MLTrainingConfig,
    EnsembleConfig,
    BaggingParamsConfig,
    StackingParamsConfig,
)
from app.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingOrchestrationService,
)


def create_test_config(ensemble_method="bagging"):
    """テスト用のMLTrainingConfigを作成"""

    if ensemble_method == "bagging":
        ensemble_config = EnsembleConfig(
            enabled=True,
            method="bagging",
            bagging_params=BaggingParamsConfig(n_estimators=3, bootstrap_fraction=0.8),
        )
    elif ensemble_method == "stacking":
        ensemble_config = EnsembleConfig(
            enabled=True,
            method="stacking",
            stacking_params=StackingParamsConfig(
                base_models=["lightgbm", "random_forest"],
                meta_model="logistic_regression",
                cv_folds=3,
                use_probas=True,
            ),
        )
    else:
        ensemble_config = EnsembleConfig(enabled=True, method="bagging")

    # 過去30日間のデータを使用
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    config = MLTrainingConfig(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        validation_split=0.2,
        prediction_horizon=24,
        threshold_up=0.02,
        threshold_down=-0.02,
        save_model=False,  # テスト用なので保存しない
        train_test_split=0.8,
        cross_validation_folds=5,
        random_state=42,
        early_stopping_rounds=50,
        max_depth=10,
        n_estimators=50,
        learning_rate=0.1,
        ensemble_config=ensemble_config,
    )

    return config


async def test_bagging_api():
    """バギングアンサンブルAPIテスト"""
    print("\n=== バギングアンサンブルAPIテスト ===")

    try:
        config = create_test_config("bagging")
        orchestration_service = MLTrainingOrchestrationService()

        print("バギングアンサンブル学習を開始...")
        print(f"設定: {config.ensemble_config.method}")
        print(f"n_estimators: {config.ensemble_config.bagging_params.n_estimators}")

        # 学習実行（バックグラウンドタスクなしでテスト）
        result = await orchestration_service._train_ml_model_background(config, None)

        print(f"バギング学習完了!")
        print(f"成功: {result.get('success', False)}")
        print(f"精度: {result.get('accuracy', 'N/A')}")
        print(f"アンサンブル手法: {result.get('ensemble_method', 'N/A')}")

        return result.get("success", False)

    except Exception as e:
        print(f"バギングAPIテストでエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_stacking_api():
    """スタッキングアンサンブルAPIテスト"""
    print("\n=== スタッキングアンサンブルAPIテスト ===")

    try:
        config = create_test_config("stacking")
        orchestration_service = MLTrainingOrchestrationService()

        print("スタッキングアンサンブル学習を開始...")
        print(f"設定: {config.ensemble_config.method}")
        print(f"ベースモデル: {config.ensemble_config.stacking_params.base_models}")
        print(f"メタモデル: {config.ensemble_config.stacking_params.meta_model}")

        # 学習実行（バックグラウンドタスクなしでテスト）
        result = await orchestration_service._train_ml_model_background(config, None)

        print(f"スタッキング学習完了!")
        print(f"成功: {result.get('success', False)}")
        print(f"精度: {result.get('accuracy', 'N/A')}")
        print(f"アンサンブル手法: {result.get('ensemble_method', 'N/A')}")

        return result.get("success", False)

    except Exception as e:
        print(f"スタッキングAPIテストでエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_config_validation():
    """設定検証テスト"""
    print("\n=== 設定検証テスト ===")

    try:
        # 正常な設定
        config1 = create_test_config("bagging")
        print(f"✅ バギング設定検証: {config1.ensemble_config.enabled}")

        config2 = create_test_config("stacking")
        print(f"✅ スタッキング設定検証: {config2.ensemble_config.enabled}")

        # デフォルト設定
        config3 = MLTrainingConfig(
            symbol="BTC/USDT:USDT", start_date="2024-01-01", end_date="2024-01-31"
        )
        print(f"✅ デフォルト設定検証: {config3.ensemble_config.enabled}")
        print(f"   デフォルト手法: {config3.ensemble_config.method}")

        return True

    except Exception as e:
        print(f"設定検証テストでエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """メイン関数"""
    print("🚀 アンサンブル学習API統合テストを開始")
    print("=" * 60)

    results = []

    # 各テストを実行
    results.append(("設定検証", await test_config_validation()))
    results.append(("バギングAPI", await test_bagging_api()))
    results.append(("スタッキングAPI", await test_stacking_api()))

    # 結果をまとめ
    print("\n" + "=" * 60)
    print("=== API統合テスト結果まとめ ===")
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 全てのAPI統合テストが成功しました！")
        print("アンサンブル学習がAPIレベルで正しく動作しています。")
    else:
        print("\n⚠️ 一部のAPI統合テストが失敗しました。")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
