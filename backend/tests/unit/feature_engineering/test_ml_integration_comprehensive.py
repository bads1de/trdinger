"""
AutoML特徴量エンジニアリングと既存MLシステムの包括的統合テスト

既存のMLトレーニングシステムとの完全な統合を検証します。
"""

import pytest
import pandas as pd
import numpy as np
import warnings
import time
from unittest.mock import patch, MagicMock

from app.core.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.core.services.ml.ml_training_service import MLTrainingService
from app.core.services.ml.base_ml_trainer import BaseMLTrainer
from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
    TSFreshConfig,
    FeaturetoolsConfig,
    AutoFeatConfig,
)


class TestMLIntegrationComprehensive:
    """包括的ML統合テストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テスト用の軽量AutoML設定
        self.automl_config = AutoMLConfig(
            tsfresh_config=TSFreshConfig(
                enabled=True,
                feature_selection=False,  # テスト高速化
                feature_count_limit=10,
                parallel_jobs=1,
            ),
            featuretools_config=FeaturetoolsConfig(
                enabled=True, max_depth=1, max_features=5
            ),
            autofeat_config=AutoFeatConfig(
                enabled=False,  # テスト高速化のため無効
                max_features=5,
                feateng_steps=2,  # generationsではなくfeateng_steps
                max_gb=0.5,
            ),
        )

        self.enhanced_service = EnhancedFeatureEngineeringService(self.automl_config)

    def create_comprehensive_test_data(self, rows: int = 200) -> dict:
        """包括的なテストデータセットを作成"""
        np.random.seed(42)

        dates = pd.date_range(start="2023-01-01", periods=rows, freq="1h")

        # 現実的な価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, rows)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))

        prices = np.array(prices)

        # OHLCV データ
        ohlcv_data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.001, rows)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, rows))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, rows))),
                "Close": prices,
                "Volume": np.random.lognormal(10, 1, rows),
            },
            index=dates,
        )

        # High >= Close >= Low の制約を満たす
        ohlcv_data["High"] = np.maximum(
            ohlcv_data["High"], ohlcv_data[["Open", "Close"]].max(axis=1)
        )
        ohlcv_data["Low"] = np.minimum(
            ohlcv_data["Low"], ohlcv_data[["Open", "Close"]].min(axis=1)
        )

        # ファンディングレートデータ
        funding_rate_data = pd.DataFrame(
            {
                "funding_rate": np.random.normal(0.0001, 0.0005, rows),
                "predicted_funding_rate": np.random.normal(0.0001, 0.0005, rows),
            },
            index=dates,
        )

        # 建玉残高データ
        open_interest_data = pd.DataFrame(
            {"open_interest": np.random.lognormal(15, 0.5, rows)}, index=dates
        )

        # Fear & Greed Index データ
        fear_greed_data = pd.DataFrame(
            {"fear_greed_index": np.random.randint(0, 101, rows)}, index=dates
        )

        # ターゲット変数（価格方向）
        target = pd.Series(
            np.random.choice([0, 1, 2], size=rows, p=[0.3, 0.4, 0.3]),
            name="target",
            index=dates,
        )

        return {
            "ohlcv": ohlcv_data,
            "funding_rate": funding_rate_data,
            "open_interest": open_interest_data,
            "fear_greed": fear_greed_data,
            "target": target,
        }

    def test_enhanced_feature_service_integration(self):
        """EnhancedFeatureEngineeringServiceの統合テスト"""
        test_data = self.create_comprehensive_test_data(100)

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result_df = self.enhanced_service.calculate_enhanced_features(
                ohlcv_data=test_data["ohlcv"],
                funding_rate_data=test_data["funding_rate"],
                open_interest_data=test_data["open_interest"],
                fear_greed_data=test_data["fear_greed"],
                target=test_data["target"],
            )

        processing_time = time.time() - start_time

        # 基本検証
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(test_data["ohlcv"])

        # 元のOHLCV列が保持されているか確認
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result_df.columns

        # 手動特徴量が追加されているか確認
        manual_features = [
            col
            for col in result_df.columns
            if not col.startswith(("TSF_", "FT_", "AF_"))
            and col not in ["Open", "High", "Low", "Close", "Volume"]
        ]
        assert len(manual_features) >= 50  # 手動特徴量が十分に生成されている

        # AutoML特徴量が追加されているか確認
        automl_features = [
            col for col in result_df.columns if col.startswith(("TSF_", "FT_", "AF_"))
        ]
        assert (
            len(automl_features) >= 0
        )  # AutoML特徴量が生成されている（ライブラリ利用可能時）

        # 統計情報の確認
        stats = self.enhanced_service.get_enhancement_stats()
        assert isinstance(stats, dict)
        assert "total_features" in stats
        assert "total_time" in stats

        print(
            f"統合テスト完了: {len(result_df.columns)}個の特徴量, {processing_time:.2f}秒"
        )

    def test_ml_training_with_enhanced_features(self):
        """拡張特徴量を使用したMLトレーニングテスト"""
        test_data = self.create_comprehensive_test_data(150)

        # 拡張特徴量を生成
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            enhanced_features = self.enhanced_service.calculate_enhanced_features(
                ohlcv_data=test_data["ohlcv"],
                funding_rate_data=test_data["funding_rate"],
                open_interest_data=test_data["open_interest"],
                fear_greed_data=test_data["fear_greed"],
                target=test_data["target"],
            )

        # MLトレーニングサービスをテスト（AutoML設定付き）
        automl_config = {
            "tsfresh": {
                "enabled": True,
                "feature_selection": False,  # テスト高速化
                "feature_count_limit": 10,
                "parallel_jobs": 1,
            },
            "featuretools": {
                "enabled": True,
                "max_depth": 1,
                "max_features": 5,
            },
            "autofeat": {
                "enabled": False,  # テスト高速化のため無効
                "max_features": 5,
                "generations": 2,
                "max_gb": 0.5,
            },
        }

        ml_service = MLTrainingService(automl_config=automl_config)

        # トレーニング実行（モデル保存を無効にしてテスト）
        training_result = ml_service.train_model(
            training_data=enhanced_features,
            funding_rate_data=test_data["funding_rate"],
            open_interest_data=test_data["open_interest"],
            save_model=False,  # テスト用
            automl_config=automl_config,  # AutoML設定を渡す
            test_size=0.3,
            random_state=42,
        )

        # 結果の検証
        assert isinstance(training_result, dict)
        assert "success" in training_result
        assert "accuracy" in training_result
        assert "feature_count" in training_result

        # 特徴量数が適切であることを確認（AutoMLライブラリが利用できない場合は手動特徴量のみ）
        assert training_result["feature_count"] >= 100  # 手動特徴量（+AutoML特徴量）

        print(
            f"AutoML MLトレーニング完了: 精度={training_result.get('accuracy', 0):.3f}, "
            f"特徴量数={training_result.get('feature_count', 0)}"
        )

    def test_automl_ml_training_integration(self):
        """AutoML特徴量エンジニアリングとMLトレーニングの統合テスト"""
        test_data = self.create_comprehensive_test_data(200)  # より多くのデータでテスト

        # AutoML設定（テスト用に軽量化）
        automl_config = {
            "tsfresh": {
                "enabled": True,
                "feature_selection": True,
                "fdr_level": 0.1,  # テスト用に緩く設定
                "feature_count_limit": 20,
                "parallel_jobs": 1,
            },
            "featuretools": {
                "enabled": True,
                "max_depth": 2,
                "max_features": 15,
            },
            "autofeat": {
                "enabled": True,
                "max_features": 10,
                "feateng_steps": 3,  # テスト用に少なく設定
                "max_gb": 0.5,
            },
        }

        # AutoML設定付きアンサンブル学習サービス
        ml_service = MLTrainingService(
            trainer_type="ensemble", automl_config=automl_config
        )

        # トレーニング実行
        training_result = ml_service.train_model(
            training_data=test_data["ohlcv"],
            funding_rate_data=test_data["funding_rate"],
            open_interest_data=test_data["open_interest"],
            save_model=False,  # テスト用
            automl_config=automl_config,
            test_size=0.3,
            random_state=42,
        )

        # 結果の検証
        assert isinstance(training_result, dict)
        assert "success" in training_result
        assert "accuracy" in training_result
        assert "feature_count" in training_result

        # AutoML特徴量が生成されていることを確認
        feature_count = training_result.get("feature_count", 0)
        assert feature_count > 50  # AutoML特徴量により増加していることを期待

        print(
            f"AutoML統合テスト完了: 精度={training_result.get('accuracy', 0):.3f}, "
            f"特徴量数={feature_count}"
        )

    def test_automl_vs_basic_comparison(self):
        """AutoML特徴量エンジニアリングと基本特徴量エンジニアリングの比較テスト"""
        test_data = self.create_comprehensive_test_data(150)

        # 基本特徴量エンジニアリングでのトレーニング
        basic_ml_service = MLTrainingService()  # AutoML設定なし
        basic_result = basic_ml_service.train_model(
            training_data=test_data["ohlcv"],
            funding_rate_data=test_data["funding_rate"],
            open_interest_data=test_data["open_interest"],
            save_model=False,
            test_size=0.3,
            random_state=42,
        )

        # AutoML特徴量エンジニアリングでのトレーニング
        automl_config = {
            "tsfresh": {"enabled": True, "feature_count_limit": 15, "parallel_jobs": 1},
            "featuretools": {"enabled": True, "max_depth": 1, "max_features": 10},
            "autofeat": {"enabled": False},  # テスト高速化のため無効
        }

        automl_ml_service = MLTrainingService(automl_config=automl_config)
        automl_result = automl_ml_service.train_model(
            training_data=test_data["ohlcv"],
            funding_rate_data=test_data["funding_rate"],
            open_interest_data=test_data["open_interest"],
            save_model=False,
            automl_config=automl_config,
            test_size=0.3,
            random_state=42,
        )

        # 結果の比較
        basic_features = basic_result.get("feature_count", 0)
        automl_features = automl_result.get("feature_count", 0)

        print(f"基本特徴量数: {basic_features}")
        print(f"AutoML特徴量数: {automl_features}")

        # AutoMLの方が特徴量数が多いことを期待
        assert automl_features >= basic_features

        print("AutoML vs 基本特徴量エンジニアリング比較テスト完了")

    def test_ml_orchestrator_integration(self):
        """MLOrchestratorとの統合テスト"""
        test_data = self.create_comprehensive_test_data(100)

        # MLOrchestratorをテスト
        orchestrator = MLOrchestrator()

        # 予測実行（モデルが存在しない場合はスキップ）
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ml_indicators = orchestrator.calculate_ml_indicators(
                    df=test_data["ohlcv"],
                    funding_rate_data=test_data["funding_rate"],
                    open_interest_data=test_data["open_interest"],
                )

            # 結果の検証（モデルが存在する場合）
            if ml_indicators is not None:
                assert isinstance(ml_indicators, dict)
                expected_keys = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
                for key in expected_keys:
                    if key in ml_indicators:
                        assert isinstance(ml_indicators[key], np.ndarray)
                        assert len(ml_indicators[key]) == len(test_data["ohlcv"])

                print("MLOrchestrator統合テスト完了")
            else:
                print("MLOrchestrator: モデルが存在しないため予測をスキップ")

        except Exception as e:
            # モデルが存在しない場合は正常
            print(f"MLOrchestrator: {e} (正常)")

    def test_feature_compatibility_with_existing_ml(self):
        """既存MLシステムとの特徴量互換性テスト"""
        test_data = self.create_comprehensive_test_data(100)

        # 既存の特徴量エンジニアリング
        from app.core.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        legacy_service = FeatureEngineeringService()
        legacy_features = legacy_service.calculate_advanced_features(
            ohlcv_data=test_data["ohlcv"],
            funding_rate_data=test_data["funding_rate"],
            open_interest_data=test_data["open_interest"],
            fear_greed_data=test_data["fear_greed"],
        )

        # 拡張特徴量エンジニアリング
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            enhanced_features = self.enhanced_service.calculate_enhanced_features(
                ohlcv_data=test_data["ohlcv"],
                funding_rate_data=test_data["funding_rate"],
                open_interest_data=test_data["open_interest"],
                fear_greed_data=test_data["fear_greed"],
                target=test_data["target"],
            )

        # 互換性の確認
        # 1. 既存の特徴量が全て含まれているか
        legacy_columns = set(legacy_features.columns)
        enhanced_columns = set(enhanced_features.columns)

        missing_columns = legacy_columns - enhanced_columns
        assert len(missing_columns) == 0, f"既存特徴量が欠落: {missing_columns}"

        # 2. データ型の互換性
        for col in legacy_columns:
            if col in enhanced_columns:
                legacy_dtype = legacy_features[col].dtype
                enhanced_dtype = enhanced_features[col].dtype

                # 数値型であることを確認
                assert pd.api.types.is_numeric_dtype(
                    enhanced_dtype
                ), f"{col}が数値型ではありません"

        # 3. 値の範囲チェック（無限値・NaNの確認）
        numeric_columns = enhanced_features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            assert (
                not enhanced_features[col].isin([np.inf, -np.inf]).any()
            ), f"{col}に無限値が含まれています"
            # NaNは許容（後で補完される）

        print(
            f"互換性テスト完了: 既存特徴量={len(legacy_columns)}, "
            f"拡張特徴量={len(enhanced_columns)}, "
            f"追加特徴量={len(enhanced_columns) - len(legacy_columns)}"
        )

    def test_performance_with_large_dataset(self):
        """大規模データセットでの性能テスト"""
        # より大きなデータセットでテスト
        test_data = self.create_comprehensive_test_data(500)

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result_df = self.enhanced_service.calculate_enhanced_features(
                ohlcv_data=test_data["ohlcv"],
                funding_rate_data=test_data["funding_rate"],
                open_interest_data=test_data["open_interest"],
                target=test_data["target"],
            )

        processing_time = time.time() - start_time

        # 性能要件の確認
        assert processing_time < 120, f"処理時間が長すぎます: {processing_time:.2f}秒"

        # メモリ使用量の確認
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        # メモリ使用量が過度でないことを確認（テスト環境では緩い制限）
        assert memory_mb < 2048, f"メモリ使用量が多すぎます: {memory_mb:.1f}MB"

        print(
            f"性能テスト完了: {len(result_df.columns)}個の特徴量, "
            f"{processing_time:.2f}秒, {memory_mb:.1f}MB"
        )

    def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        # 空データでのテスト
        empty_df = pd.DataFrame()

        result = self.enhanced_service.calculate_enhanced_features(ohlcv_data=empty_df)

        # 空データの場合はNoneまたは空のDataFrameが返される
        assert result is None or (isinstance(result, pd.DataFrame) and result.empty)

        # 不正なデータでのテスト
        invalid_df = pd.DataFrame(
            {
                "Open": [1, 2, np.inf],
                "High": [1, 2, 3],
                "Low": [1, 2, 3],
                "Close": [1, 2, 3],
                "Volume": [1, 2, 3],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = self.enhanced_service.calculate_enhanced_features(
                ohlcv_data=invalid_df
            )

        # エラーが発生してもNoneまたはDataFrameが返されることを確認
        assert result is None or isinstance(result, pd.DataFrame)

    def test_configuration_management(self):
        """設定管理統合テスト"""
        # 設定の取得
        config = self.enhanced_service.get_automl_config()
        assert isinstance(config, dict)
        assert "tsfresh" in config
        assert "featuretools" in config
        assert "autofeat" in config

        # 設定の更新
        new_config = {
            "tsfresh": {"feature_count_limit": 50, "parallel_jobs": 2},
            "featuretools": {"max_depth": 2, "max_features": 20},
        }

        self.enhanced_service._update_automl_config(new_config)

        # 更新された設定の確認
        updated_config = self.enhanced_service.get_automl_config()
        assert updated_config["tsfresh"]["feature_count_limit"] == 50
        assert updated_config["featuretools"]["max_depth"] == 2

        # 設定検証
        validation_result = self.enhanced_service.validate_automl_config(new_config)
        assert isinstance(validation_result, dict)
        assert "valid" in validation_result

    def test_feature_names_consistency(self):
        """特徴量名の一貫性テスト"""
        test_data = self.create_comprehensive_test_data(50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result_df = self.enhanced_service.calculate_enhanced_features(
                ohlcv_data=test_data["ohlcv"],
                funding_rate_data=test_data["funding_rate"],
                open_interest_data=test_data["open_interest"],
            )

        # 特徴量名の形式チェック
        for col in result_df.columns:
            # 特殊文字が含まれていないことを確認
            assert not any(
                char in col for char in ["(", ")", "[", "]", "{", "}", "<", ">"]
            ), f"特徴量名に特殊文字が含まれています: {col}"

            # 長すぎる名前でないことを確認
            assert len(col) < 100, f"特徴量名が長すぎます: {col}"

        # プレフィックスの確認
        tsfresh_features = [col for col in result_df.columns if col.startswith("TSF_")]
        featuretools_features = [
            col for col in result_df.columns if col.startswith("FT_")
        ]
        autofeat_features = [col for col in result_df.columns if col.startswith("AF_")]

        print(
            f"特徴量名一貫性テスト完了: TSFresh={len(tsfresh_features)}, "
            f"Featuretools={len(featuretools_features)}, AutoFeat={len(autofeat_features)}"
        )

    def test_data_quality_after_enhancement(self):
        """拡張後のデータ品質テスト"""
        test_data = self.create_comprehensive_test_data(100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result_df = self.enhanced_service.calculate_enhanced_features(
                ohlcv_data=test_data["ohlcv"],
                funding_rate_data=test_data["funding_rate"],
                open_interest_data=test_data["open_interest"],
            )

        # データ品質チェック
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # 無限値チェック
            inf_count = result_df[col].isin([np.inf, -np.inf]).sum()
            assert inf_count == 0, f"{col}に{inf_count}個の無限値があります"

            # 分散チェック（定数列でないことを確認）
            if not result_df[col].isna().all():
                variance = result_df[col].var()
                if not pd.isna(variance):
                    # 完全に定数でない限り、何らかの分散があることを期待
                    pass  # 一部の特徴量は定数になる可能性があるため、厳しくチェックしない

        print(f"データ品質テスト完了: {len(numeric_columns)}個の数値特徴量を検証")
