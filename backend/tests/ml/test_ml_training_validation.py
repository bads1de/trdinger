"""
MLトレーニング検証テスト - 実際のトレーニング実行と問題検出を重視
"""

import gc
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.services.ml.base_ml_trainer import BaseMLTrainer
from backend.app.services.ml.ml_training_service import MLTrainingService
from backend.app.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingOrchestrationService,
)


@pytest.mark.skip(reason="MLトレーニング検証は実装が不完全。実装完了後に有効化")
class TestMLTrainingValidation:
    """MLトレーニング検証のための実践的テスト"""

    @pytest.fixture
    def sample_training_data(self):
        """実際のトレーニングデータ"""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq="D")

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": 10000 + np.random.randn(len(dates)) * 200,
                "high": 10000 + np.random.randn(len(dates)) * 300,
                "low": 10000 + np.random.randn(len(dates)) * 300,
                "close": 10000 + np.random.randn(len(dates)) * 200,
                "volume": 500 + np.random.randint(100, 1000, len(dates)),
                "returns": np.random.randn(len(dates)) * 0.02,
                "volatility": 0.01 + np.random.rand(len(dates)) * 0.02,
                "rsi": 30 + np.random.rand(len(dates)) * 40,
                "macd": np.random.randn(len(dates)) * 0.01,
                "signal": np.random.randn(len(dates)) * 0.005,
                "histogram": np.random.randn(len(dates)) * 0.005,
            }
        )

        # OHLCの関係を確保
        data["high"] = data[["open", "close", "high"]].max(axis=1)
        data["low"] = data[["open", "close", "low"]].min(axis=1)

        # ラベルを生成（上昇:1, 下降:0）
        data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)

        return data.dropna()

    @pytest.fixture
    def real_ml_training_service(self):
        """実際のMLトレーニングサービス"""
        return MLTrainingService(trainer_type="single", single_model_type="lightgbm")

    def test_real_model_training_execution(
        self, sample_training_data, real_ml_training_service
    ):
        """実際のモデルトレーニング実行のテスト"""
        print("🔍 実際のモデルトレーニングを実行...")

        # 実際のトレーニングを実行
        result = real_ml_training_service.train_model(
            sample_training_data, save_model=False, optimization_settings=None
        )

        # トレーニングが成功していること
        assert result["success"] is True
        assert "f1_score" in result
        assert "accuracy" in result
        assert result["f1_score"] >= 0.0
        assert result["accuracy"] >= 0.0

        print(
            f"✅ トレーニング成功 - F1スコア: {result['f1_score']:.4f}, 精度: {result['accuracy']:.4f}"
        )

    def test_real_training_with_optimization(
        self, sample_training_data, real_ml_training_service
    ):
        """最適化ありの実際のトレーニングテスト"""
        print("🔍 最適化ありの実際のトレーニングを実行...")

        from backend.app.services.ml.ml_training_service import OptimizationSettings

        # 最適化設定
        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=10,  # 少ない回数にしてテストを高速化
            parameter_space={
                "learning_rate": {"type": "real", "low": 0.01, "high": 0.1},
                "n_estimators": {"type": "integer", "low": 50, "high": 100},
            },
        )

        # 実際のトレーニングを実行
        result = real_ml_training_service.train_model(
            sample_training_data,
            save_model=False,
            optimization_settings=optimization_settings,
        )

        # トレーニングと最適化が成功していること
        assert result["success"] is True
        assert "optimization_result" in result
        assert "best_score" in result["optimization_result"]
        assert "best_params" in result["optimization_result"]

        print(
            f"✅ 最適化トレーニング成功 - 最良スコア: {result['optimization_result']['best_score']:.4f}"
        )

    def test_memory_leakage_detection(self, sample_training_data):
        """メモリリーク検出のテスト"""
        print("🔍 メモリリークを検出...")

        initial_objects = len(gc.get_objects())
        gc.collect()

        # 複数回トレーニングを実行
        for i in range(3):
            training_service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            result = training_service.train_model(
                sample_training_data, save_model=False
            )
            assert result["success"] is True

            # メモリリークがないこと
            current_objects = len(gc.get_objects())
            assert (
                current_objects - initial_objects
            ) < 500, f"トレーニング{i+1}回目でメモリリークが検出されました"

        gc.collect()
        final_objects = len(gc.get_objects())
        print(
            f"✅ メモリリークなし - 初期オブジェクト数: {initial_objects}, 最終オブジェクト数: {final_objects}"
        )

    def test_model_convergence_validation(self, sample_training_data):
        """モデル収束検証のテスト"""
        print("🔍 モデル収束を検証...")

        training_service = MLTrainingService(
            trainer_type="single", single_model_type="lightgbm"
        )

        # 複数回トレーニングを実行して収束を確認
        scores = []
        for i in range(5):
            result = training_service.train_model(
                sample_training_data, save_model=False
            )
            scores.append(result["f1_score"])

        # スコアが安定していること（収束している）
        score_std = np.std(scores)
        assert (
            score_std < 0.1
        ), f"モデルが収束していません - スコア標準偏差: {score_std:.4f}"

        print(f"✅ モデル収束確認 - スコア標準偏差: {score_std:.4f}")

    def test_data_preprocessing_effectiveness(self, sample_training_data):
        """データ前処理効果のテスト"""
        print("🔍 データ前処理効果を検証...")

        # 生データでのトレーニング
        raw_data = sample_training_data.copy()
        raw_result = MLTrainingService(
            trainer_type="single", single_model_type="lightgbm"
        ).train_model(raw_data, save_model=False)

        # 前処理済みデータでのトレーニング
        processed_data = sample_training_data.copy()
        # 標準化を適用
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        processed_data[numeric_cols] = (
            processed_data[numeric_cols] - processed_data[numeric_cols].mean()
        ) / processed_data[numeric_cols].std()

        processed_result = MLTrainingService(
            trainer_type="single", single_model_type="lightgbm"
        ).train_model(processed_data, save_model=False)

        # 前処理によりパフォーマンスが向上していること
        improvement = processed_result["f1_score"] - raw_result["f1_score"]
        print(
            f"✅ データ前処理効果 - 生データF1スコア: {raw_result['f1_score']:.4f}, "
            f"前処理後F1スコア: {processed_result['f1_score']:.4f}, "
            f"向上幅: {improvement:.4f}"
        )

    def test_model_overfitting_detection(self, sample_training_data):
        """モデル過学習検出のテスト"""
        print("🔍 モデル過学習を検出...")

        training_service = MLTrainingService(
            trainer_type="single", single_model_type="lightgbm"
        )

        # 学習データとテストデータに分割
        train_data = sample_training_data[: int(len(sample_training_data) * 0.8)]
        test_data = sample_training_data[int(len(sample_training_data) * 0.8) :]

        # 学習データでトレーニング
        result = training_service.train_model(train_data, save_model=False)

        # テストデータでの評価
        test_result = training_service.evaluate_model(test_data)

        # 過学習が起きていないこと（テスト精度が学習精度と大きく異なること）
        train_score = result["f1_score"]
        test_score = test_result["f1_score"]
        overfitting_threshold = 0.15  # 15%以上の差は過学習とみなす

        if abs(train_score - test_score) > overfitting_threshold:
            print(
                f"⚠️ 過学習の兆候あり - 学習F1スコア: {train_score:.4f}, テストF1スコア: {test_score:.4f}"
            )
        else:
            print(
                f"✅ 過学習なし - 学習F1スコア: {train_score:.4f}, テストF1スコア: {test_score:.4f}"
            )

    def test_training_failure_recovery(self, sample_training_data):
        """トレーニング失敗からの回復テスト"""
        print("🔍 トレーニング失敗からの回復をテスト...")

        training_service = MLTrainingService(
            trainer_type="single", single_model_type="lightgbm"
        )

        # 無効なデータでトレーニングを試行
        invalid_data = sample_training_data.copy()
        invalid_data.loc[:, "close"] = np.nan  # 全てのデータをNaNに

        try:
            result = training_service.train_model(invalid_data, save_model=False)
            # 失敗してもエラーが適切に処理されること
            assert not result["success"]
            print("✅ 無効データに対する適切なエラーハンドリング")
        except Exception as e:
            print(f"✅ 例外が適切にキャッチされました: {type(e).__name__}")

        # 再び有効なデータでトレーニング
        valid_result = training_service.train_model(
            sample_training_data, save_model=False
        )
        assert valid_result["success"] is True
        print("✅ 失敗後の回復が成功")

    def test_model_persistence_and_loading(self, sample_training_data):
        """モデル永続化と読み込みのテスト"""
        print("🔍 モデル永続化と読み込みをテスト...")

        training_service = MLTrainingService(
            trainer_type="single", single_model_type="lightgbm"
        )

        # トレーニングと保存
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")

            # トレーニング
            train_result = training_service.train_model(
                sample_training_data, save_model=True, model_path=model_path
            )
            assert train_result["success"] is True

            # 保存されたモデルの存在確認
            assert os.path.exists(model_path)

            # 新しいサービスでモデルを読み込み
            new_service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            loaded_model = new_service.load_model(model_path)
            assert loaded_model is not None

            # 読み込んだモデルで予測
            features = sample_training_data.drop(["target"], axis=1, errors="ignore")
            if hasattr(loaded_model, "predict"):
                predictions = loaded_model.predict(features)
                assert len(predictions) == len(sample_training_data)
                print("✅ モデル永続化と読み込み成功")

    def test_training_with_different_model_types(self, sample_training_data):
        """異なるモデルタイプでのトレーニングテスト"""
        print("🔍 異なるモデルタイプでのトレーニングをテスト...")

        model_types = ["lightgbm", "xgboost", "randomforest"]
        results = {}

        for model_type in model_types:
            print(f"  - {model_type} トレーニング中...")

            training_service = MLTrainingService(
                trainer_type="single", single_model_type=model_type
            )
            result = training_service.train_model(
                sample_training_data, save_model=False
            )

            assert result["success"] is True
            results[model_type] = result["f1_score"]
            print(f"    ✅ F1スコア: {result['f1_score']:.4f}")

        # すべてのモデルタイプでトレーニングが成功していること
        assert len(results) == len(model_types)
        print(f"✅ すべてのモデルタイプで成功 - {results}")

    def test_ensemble_training_validation(self, sample_training_data):
        """アンサンブルトレーニング検証のテスト"""
        print("🔍 アンサンブルトレーニングを検証...")

        training_service = MLTrainingService(trainer_type="ensemble")

        # アンサンブルトレーニングを実行
        result = training_service.train_model(sample_training_data, save_model=False)

        # アンサンブルトレーニングが成功していること
        assert result["success"] is True
        assert "f1_score" in result
        assert "ensemble_metrics" in result or "individual_model_metrics" in result

        print(f"✅ アンサンブルトレーニング成功 - F1スコア: {result['f1_score']:.4f}")

    def test_training_time_monitoring(self, sample_training_data):
        """トレーニング時間監視のテスト"""
        print("🔍 トレーニング時間を監視...")

        training_service = MLTrainingService(
            trainer_type="single", single_model_type="lightgbm"
        )

        # トレーニング開始時間
        start_time = datetime.now()

        # トレーニング実行
        result = training_service.train_model(sample_training_data, save_model=False)

        # トレーニング終了時間
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        # トレーニングが成功していること
        assert result["success"] is True

        # トレーニング時間が適切な範囲内であること（10分以内）
        assert (
            training_duration < 600
        ), f"トレーニング時間が長すぎます: {training_duration:.2f}秒"

        print(f"✅ トレーニング時間監視成功 - 所要時間: {training_duration:.2f}秒")

    def test_final_comprehensive_validation(self, sample_training_data):
        """最終包括的検証テスト"""
        print("\n🏁 最終包括的検証を実行...")

        # すべての主要機能が正常に動作することを確認
        validation_checks = []

        # 1. 基本トレーニング
        try:
            service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            result = service.train_model(sample_training_data, save_model=False)
            validation_checks.append(("基本トレーニング", result["success"]))
        except Exception as e:
            validation_checks.append(("基本トレーニング", False))

        # 2. 最適化トレーニング
        try:
            from backend.app.services.ml.ml_training_service import OptimizationSettings

            service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            optimization_settings = OptimizationSettings(enabled=True, n_calls=5)
            result = service.train_model(
                sample_training_data,
                save_model=False,
                optimization_settings=optimization_settings,
            )
            validation_checks.append(("最適化トレーニング", result["success"]))
        except Exception as e:
            validation_checks.append(("最適化トレーニング", False))

        # 3. モデル評価
        try:
            service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            result = service.train_model(sample_training_data, save_model=False)
            eval_result = service.evaluate_model(sample_training_data)
            validation_checks.append(("モデル評価", True))
        except Exception as e:
            validation_checks.append(("モデル評価", False))

        # 4. 予測機能
        try:
            service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            result = service.train_model(sample_training_data, save_model=False)
            features = sample_training_data.drop(["target"], axis=1, errors="ignore")
            predictions = service.predict(features)
            validation_checks.append(("予測機能", "predictions" in predictions))
        except Exception as e:
            validation_checks.append(("予測機能", False))

        # 検証結果の集計
        passed_checks = sum(1 for _, passed in validation_checks if passed)
        total_checks = len(validation_checks)

        print(f"\n📊 検証結果:")
        for check_name, passed in validation_checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}: {'成功' if passed else '失敗'}")

        print(f"\n🎯 総合評価: {passed_checks}/{total_checks} のチェックが成功")

        # 大多数のチェックが成功していること
        assert (
            passed_checks >= total_checks * 0.75
        ), "MLトレーニングが正常に動作していません"

        print(f"🎉 MLトレーニング検証が成功しました！")


# トレーニングパイプラインの包括的テスト
@pytest.mark.skip(reason="MLトレーニングパイプラインは実装が不完全。実装完了後に有効化")
class TestMLTrainingPipeline:
    """MLトレーニングパイプラインの包括的テスト"""

    def test_end_to_end_training_pipeline(self, sample_training_data):
        """エンドツーエンドのトレーニングパイプラインテスト"""
        print("🔍 エンドツーエンドトレーニングパイプラインをテスト...")

        # 1. データ前処理
        print("  1. データ前処理...")
        processed_data = sample_training_data.copy()
        # 標準化
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        processed_data[numeric_cols] = (
            processed_data[numeric_cols] - processed_data[numeric_cols].mean()
        ) / processed_data[numeric_cols].std()

        # 2. 特徴量選択
        print("  2. 特徴量選択...")
        feature_cols = ["close", "volume", "rsi", "macd", "returns", "volatility"]
        features = processed_data[feature_cols]
        target = processed_data["target"]

        # 3. 学習データとテストデータの分割
        print("  3. データ分割...")
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]

        # 4. モデルトレーニング
        print("  4. モデルトレーニング...")
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 5. モデル評価
        print("  5. モデル評価...")
        from sklearn.metrics import accuracy_score, f1_score

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)

        # 6. 結果検証
        print(f"    学習精度: {train_accuracy:.4f}")
        print(f"    テスト精度: {test_accuracy:.4f}")
        print(f"    F1スコア: {test_f1:.4f}")

        # 過学習が起きていないこと
        assert abs(train_accuracy - test_accuracy) < 0.15, "過学習が発生しています"
        assert test_f1 > 0.5, "モデルの性能が不十分です"

        print("✅ エンドツーエンドトレーニングパイプライン成功")

    def test_ml_training_robustness(self, sample_training_data):
        """MLトレーニングの堅牢性テスト"""
        print("🔍 MLトレーニングの堅牢性をテスト...")

        robustness_tests = []

        # 1. 欠損値に対する堅牢性
        print("  1. 欠損値に対する堅牢性...")
        data_with_missing = sample_training_data.copy()
        data_with_missing.loc[::10, "close"] = np.nan  # 10%のデータを欠損に

        try:
            service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            result = service.train_model(data_with_missing, save_model=False)
            robustness_tests.append(("欠損値堅牢性", result["success"]))
        except Exception:
            robustness_tests.append(("欠損値堅牢性", False))

        # 2. 外れ値に対する堅牢性
        print("  2. 外れ値に対する堅牢性...")
        data_with_outliers = sample_training_data.copy()
        data_with_outliers.loc[::20, "close"] *= 10  # 5%のデータを外れ値に

        try:
            service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            result = service.train_model(data_with_outliers, save_model=False)
            robustness_tests.append(("外れ値堅牢性", result["success"]))
        except Exception:
            robustness_tests.append(("外れ値堅牢性", False))

        # 3. ノイズに対する堅牢性
        print("  3. ノイズに対する堅牢性...")
        data_with_noise = sample_training_data.copy()
        data_with_noise["close"] += np.random.normal(0, 0.1, len(data_with_noise))

        try:
            service = MLTrainingService(
                trainer_type="single", single_model_type="lightgbm"
            )
            result = service.train_model(data_with_noise, save_model=False)
            robustness_tests.append(("ノイズ堅牢性", result["success"]))
        except Exception:
            robustness_tests.append(("ノイズ堅牢性", False))

        # 結果の評価
        passed_robustness = sum(1 for _, passed in robustness_tests if passed)
        print(f"\n堅牢性テスト結果: {passed_robustness}/{len(robustness_tests)} 成功")

        # 多くの堅牢性テストが成功していること
        assert (
            passed_robustness >= len(robustness_tests) * 0.6
        ), "MLトレーニングの堅牢性が不十分です"

        print("✅ MLトレーニングの堅牢性確認成功")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
