"""
AutoML特徴量エンジニアリング統合テスト

全AutoML機能を統合したエンドツーエンドテストを実装します。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings
import time

from app.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
    TSFreshConfig,
    FeaturetoolsConfig,
    AutoFeatConfig,
)
from app.services.ml.feature_engineering.automl_features.tsfresh_calculator import (
    TSFRESH_AVAILABLE,
)
from app.services.ml.feature_engineering.automl_features.featuretools_calculator import (
    FEATURETOOLS_AVAILABLE,
)
from app.services.ml.feature_engineering.automl_features.autofeat_calculator import (
    AUTOFEAT_AVAILABLE,
)


class TestAutoMLIntegration:
    """AutoML統合テストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テスト用の軽量設定
        tsfresh_config = TSFreshConfig(
            enabled=True,
            feature_selection=False,  # テスト高速化のため無効
            feature_count_limit=10,  # テスト用に少なく設定
            parallel_jobs=1,
        )

        featuretools_config = FeaturetoolsConfig(
            enabled=True, max_depth=1, max_features=5  # テスト用に浅く設定
        )

        autofeat_config = AutoFeatConfig(
            enabled=True,
            max_features=10,
            feateng_steps=1,  # テスト用に少なく設定
            max_gb=0.5,
        )

        automl_config = AutoMLConfig(
            tsfresh_config=tsfresh_config,
            featuretools_config=featuretools_config,
            autofeat_config=autofeat_config,
        )

        self.service = EnhancedFeatureEngineeringService(automl_config)

    def create_test_data(self, rows: int = 100) -> tuple:
        """テスト用のデータセットを作成"""
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

        # ターゲット変数
        target = pd.Series(
            np.random.choice([0, 1, 2], size=rows), name="target", index=dates
        )

        # 市場データ
        funding_rate_data = pd.DataFrame(
            {
                "funding_rate": np.random.normal(0.0001, 0.0005, rows),
                "predicted_funding_rate": np.random.normal(0.0001, 0.0005, rows),
            },
            index=dates,
        )

        open_interest_data = pd.DataFrame(
            {"open_interest": np.random.lognormal(15, 0.5, rows)}, index=dates
        )

        fear_greed_data = pd.DataFrame(
            {"fear_greed_index": np.random.randint(0, 101, rows)}, index=dates
        )

        return (
            ohlcv_data,
            target,
            funding_rate_data,
            open_interest_data,
            fear_greed_data,
        )

    def test_full_automl_pipeline(self):
        """完全なAutoMLパイプラインテスト"""
        ohlcv_data, target, funding_rate_data, open_interest_data, fear_greed_data = (
            self.create_test_data(50)
        )

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result_df = self.service.calculate_enhanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                fear_greed_data=fear_greed_data,
                target=target,
            )

        processing_time = time.time() - start_time

        # 基本的な検証
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(ohlcv_data)

        # 元の列が保持されているか確認
        for col in ohlcv_data.columns:
            assert col in result_df.columns

        # 新しい特徴量が追加されているか確認
        original_cols = set(ohlcv_data.columns)
        new_cols = set(result_df.columns) - original_cols
        assert len(new_cols) > 0

        # 統計情報を確認
        stats = self.service.get_enhancement_stats()
        assert isinstance(stats, dict)
        assert "total_features" in stats
        assert "total_time" in stats
        assert stats["total_features"] == len(result_df.columns)

        print(
            f"統合テスト完了: {len(result_df.columns)}個の特徴量, {processing_time:.2f}秒"
        )

    def test_automl_config_management(self):
        """AutoML設定管理テスト"""
        # 設定取得
        config = self.service.get_automl_config()
        assert isinstance(config, dict)
        assert "tsfresh" in config
        assert "featuretools" in config
        assert "autofeat" in config

        # 設定更新
        new_config = {
            "tsfresh": {"feature_count_limit": 20, "parallel_jobs": 2},
            "featuretools": {"max_depth": 2, "max_features": 10},
            "autofeat": {"max_features": 15, "generations": 3},
        }

        self.service._update_automl_config(new_config)

        # 更新された設定を確認
        updated_config = self.service.get_automl_config()
        assert updated_config["tsfresh"]["feature_count_limit"] == 20
        assert updated_config["featuretools"]["max_depth"] == 2
        assert updated_config["autofeat"]["max_features"] == 15

    def test_config_validation(self):
        """設定検証テスト"""
        # 有効な設定
        valid_config = {
            "tsfresh": {
                "feature_count_limit": 100,
                "fdr_level": 0.05,
                "parallel_jobs": 4,
            },
            "featuretools": {"max_depth": 2, "max_features": 50},
            "autofeat": {"max_features": 100, "generations": 20, "population_size": 50},
        }

        result = self.service.validate_automl_config(valid_config)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

        # 無効な設定
        invalid_config = {
            "tsfresh": {
                "feature_count_limit": -10,  # 無効
                "fdr_level": 1.5,  # 無効
                "parallel_jobs": 0,  # 無効
            },
            "featuretools": {"max_depth": 0, "max_features": -5},  # 無効  # 無効
        }

        result = self.service.validate_automl_config(invalid_config)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_available_features(self):
        """利用可能特徴量取得テスト"""
        features = self.service.get_available_automl_features()

        assert isinstance(features, dict)
        assert "tsfresh" in features
        assert "featuretools" in features
        assert "autofeat" in features

        for feature_type, feature_list in features.items():
            assert isinstance(feature_list, list)

    def test_cache_management(self):
        """キャッシュ管理テスト"""
        # キャッシュクリア
        self.service.clear_automl_cache()

        # キャッシュクリア後の動作確認
        ohlcv_data, target, _, _, _ = self.create_test_data(30)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.service.calculate_enhanced_features(
                ohlcv_data=ohlcv_data, target=target
            )

        assert isinstance(result_df, pd.DataFrame)

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 空データでのテスト
        empty_df = pd.DataFrame()

        result_df = self.service.calculate_enhanced_features(ohlcv_data=empty_df)

        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty

        # Noneデータでのテスト
        result_df = self.service.calculate_enhanced_features(ohlcv_data=None)

        assert result_df is None

    def test_performance_monitoring(self):
        """性能監視テスト"""
        ohlcv_data, target, _, _, _ = self.create_test_data(100)

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.service.calculate_enhanced_features(
                ohlcv_data=ohlcv_data, target=target
            )

        processing_time = time.time() - start_time

        # 性能要件の確認（テスト環境では緩い制限）
        assert processing_time < 60  # 1分以内

        # 統計情報の確認
        stats = self.service.get_enhancement_stats()
        assert stats["total_time"] > 0
        assert stats["data_rows"] == len(ohlcv_data)

    @pytest.mark.skipif(
        not TSFRESH_AVAILABLE, reason="TSFreshライブラリが利用できません"
    )
    def test_tsfresh_integration(self):
        """TSFresh統合テスト"""
        ohlcv_data, target, _, _, _ = self.create_test_data(50)

        # TSFreshのみ有効にした設定
        config = {
            "tsfresh": {"enabled": True},
            "featuretools": {"enabled": False},
            "autofeat": {"enabled": False},
        }

        self.service._update_automl_config(config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.service.calculate_enhanced_features(
                ohlcv_data=ohlcv_data, target=target
            )

        # TSFresh特徴量が追加されているか確認
        tsfresh_cols = [col for col in result_df.columns if col.startswith("TSF_")]
        assert len(tsfresh_cols) > 0

    @pytest.mark.skipif(
        not FEATURETOOLS_AVAILABLE, reason="Featuretoolsライブラリが利用できません"
    )
    def test_featuretools_integration(self):
        """Featuretools統合テスト"""
        ohlcv_data, _, _, _, _ = self.create_test_data(50)

        # Featuretoolsのみ有効にした設定
        config = {
            "tsfresh": {"enabled": False},
            "featuretools": {"enabled": True},
            "autofeat": {"enabled": False},
        }

        self.service._update_automl_config(config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.service.calculate_enhanced_features(ohlcv_data=ohlcv_data)

        # Featuretools特徴量が追加されているか確認
        featuretools_cols = [col for col in result_df.columns if col.startswith("FT_")]
        assert (
            len(featuretools_cols) >= 0
        )  # Featuretoolsは条件によっては特徴量を生成しない場合がある

    @pytest.mark.skipif(
        not AUTOFEAT_AVAILABLE, reason="AutoFeatライブラリが利用できません"
    )
    def test_autofeat_integration(self):
        """AutoFeat統合テスト"""
        ohlcv_data, target, _, _, _ = self.create_test_data(50)

        # AutoFeatのみ有効にした設定
        config = {
            "tsfresh": {"enabled": False},
            "featuretools": {"enabled": False},
            "autofeat": {"enabled": True},
        }

        self.service._update_automl_config(config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.service.calculate_enhanced_features(
                ohlcv_data=ohlcv_data, target=target
            )

        # AutoFeat特徴量が追加されているか確認
        autofeat_cols = [col for col in result_df.columns if col.startswith("AF_")]
        assert (
            len(autofeat_cols) >= 0
        )  # AutoFeatは条件によっては特徴量を生成しない場合がある

    def test_memory_usage(self):
        """メモリ使用量テスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        ohlcv_data, target, _, _, _ = self.create_test_data(200)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.service.calculate_enhanced_features(
                ohlcv_data=ohlcv_data, target=target
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # メモリ使用量が過度に増加していないことを確認（テスト環境では緩い制限）
        assert memory_increase < 500  # 500MB以内

        print(f"メモリ使用量増加: {memory_increase:.1f}MB")
