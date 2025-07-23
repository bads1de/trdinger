"""
EnhancedFeatureEngineeringServiceの統合テスト

拡張特徴量エンジニアリングサービスの統合テストを実装します。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from app.core.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.core.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
    TSFreshConfig,
)
from app.core.services.ml.feature_engineering.automl_features.tsfresh_calculator import (
    TSFRESH_AVAILABLE,
)


class TestEnhancedFeatureEngineeringService:
    """EnhancedFeatureEngineeringServiceのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テスト用の軽量設定
        tsfresh_config = TSFreshConfig(
            enabled=True,
            feature_selection=False,  # テスト高速化のため無効
            feature_count_limit=20,  # テスト用に少なく設定
            parallel_jobs=1,
        )
        automl_config = AutoMLConfig(tsfresh_config=tsfresh_config)

        self.service = EnhancedFeatureEngineeringService(automl_config)

    def create_test_ohlcv_data(self, rows: int = 100) -> pd.DataFrame:
        """テスト用のOHLCVデータを作成"""
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

        data = {
            "Open": prices * (1 + np.random.normal(0, 0.001, rows)),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.005, rows))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, rows))),
            "Close": prices,
            "Volume": np.random.lognormal(10, 1, rows),
        }

        df = pd.DataFrame(data, index=dates)

        # High >= Close >= Low の制約を満たす
        df["High"] = np.maximum(df["High"], df[["Open", "Close"]].max(axis=1))
        df["Low"] = np.minimum(df["Low"], df[["Open", "Close"]].min(axis=1))

        return df

    def create_test_market_data(self, rows: int = 100) -> tuple:
        """テスト用の市場データを作成"""
        dates = pd.date_range(start="2023-01-01", periods=rows, freq="1h")

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

        # Fear & Greed データ
        fear_greed_data = pd.DataFrame(
            {"fear_greed_index": np.random.randint(0, 101, rows)}, index=dates
        )

        return funding_rate_data, open_interest_data, fear_greed_data

    def create_test_target(self, rows: int = 100) -> pd.Series:
        """テスト用のターゲット変数を作成"""
        np.random.seed(42)
        return pd.Series(np.random.choice([0, 1, 2], size=rows), name="target")

    def test_initialization(self):
        """初期化テスト"""
        # デフォルト設定での初期化
        service = EnhancedFeatureEngineeringService()
        assert service.automl_config is not None
        assert service.tsfresh_calculator is not None
        assert service.last_enhancement_stats == {}

        # カスタム設定での初期化
        custom_config = AutoMLConfig.get_financial_optimized_config()
        service_custom = EnhancedFeatureEngineeringService(custom_config)
        assert service_custom.automl_config == custom_config

    def test_calculate_enhanced_features_basic(self):
        """基本的な拡張特徴量計算テスト"""
        test_data = self.create_test_ohlcv_data(50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.service.calculate_enhanced_features(test_data)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(test_data)

        # 元の列が保持されているか確認
        for col in test_data.columns:
            assert col in result_df.columns

        # 手動特徴量が追加されているか確認
        original_cols = set(test_data.columns)
        new_cols = set(result_df.columns) - original_cols
        assert len(new_cols) > 0  # 何らかの特徴量が追加されている

    def test_calculate_enhanced_features_with_market_data(self):
        """市場データありでの拡張特徴量計算テスト"""
        test_data = self.create_test_ohlcv_data(50)
        funding_rate_data, open_interest_data, fear_greed_data = (
            self.create_test_market_data(50)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.service.calculate_enhanced_features(
                ohlcv_data=test_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                fear_greed_data=fear_greed_data,
            )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(test_data)

        # 市場データ特徴量が追加されているか確認
        market_feature_cols = [
            col for col in result_df.columns if "FR_" in col or "OI_" in col
        ]
        assert len(market_feature_cols) > 0

    @pytest.mark.skipif(
        not TSFRESH_AVAILABLE, reason="TSFreshライブラリが利用できません"
    )
    def test_calculate_enhanced_features_with_tsfresh(self):
        """TSFresh特徴量ありでの拡張特徴量計算テスト"""
        test_data = self.create_test_ohlcv_data(100)
        target = self.create_test_target(100)

        # TSFreshを有効にした設定
        tsfresh_config = TSFreshConfig(
            enabled=True,
            feature_selection=True,
            feature_count_limit=10,
            parallel_jobs=1,
        )
        automl_config = AutoMLConfig(tsfresh_config=tsfresh_config)
        service = EnhancedFeatureEngineeringService(automl_config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = service.calculate_enhanced_features(
                ohlcv_data=test_data, target=target
            )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(test_data)

        # TSFresh特徴量が追加されているか確認
        tsfresh_cols = [col for col in result_df.columns if col.startswith("TSF_")]
        assert len(tsfresh_cols) > 0

    def test_calculate_enhanced_features_tsfresh_disabled(self):
        """TSFresh無効での拡張特徴量計算テスト"""
        test_data = self.create_test_ohlcv_data(50)

        # TSFreshを無効にした設定
        tsfresh_config = TSFreshConfig(enabled=False)
        automl_config = AutoMLConfig(tsfresh_config=tsfresh_config)
        service = EnhancedFeatureEngineeringService(automl_config)

        result_df = service.calculate_enhanced_features(test_data)

        assert isinstance(result_df, pd.DataFrame)

        # TSFresh特徴量が追加されていないことを確認
        tsfresh_cols = [col for col in result_df.columns if col.startswith("TSF_")]
        assert len(tsfresh_cols) == 0

    def test_calculate_enhanced_features_empty_input(self):
        """空データでの拡張特徴量計算テスト"""
        empty_df = pd.DataFrame()

        result_df = self.service.calculate_enhanced_features(empty_df)

        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty

    def test_calculate_enhanced_features_none_input(self):
        """Noneデータでの拡張特徴量計算テスト"""
        result_df = self.service.calculate_enhanced_features(None)

        assert result_df is None

    def test_update_automl_config(self):
        """AutoML設定更新テスト"""
        original_limit = self.service.automl_config.tsfresh.feature_count_limit

        new_config = {"tsfresh": {"feature_count_limit": 100, "fdr_level": 0.01}}

        self.service._update_automl_config(new_config)

        assert self.service.automl_config.tsfresh.feature_count_limit == 100
        assert self.service.automl_config.tsfresh.fdr_level == 0.01
        assert self.service.tsfresh_calculator.config.feature_count_limit == 100

    def test_get_enhancement_stats(self):
        """拡張処理統計情報取得テスト"""
        test_data = self.create_test_ohlcv_data(50)

        # 拡張特徴量計算を実行
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.service.calculate_enhanced_features(test_data)

        stats = self.service.get_enhancement_stats()

        assert isinstance(stats, dict)
        expected_keys = [
            "manual_features",
            "tsfresh_features",
            "total_features",
            "manual_time",
            "tsfresh_time",
            "total_time",
            "data_rows",
        ]

        for key in expected_keys:
            assert key in stats

    def test_get_automl_config(self):
        """AutoML設定取得テスト"""
        config_dict = self.service.get_automl_config()

        assert isinstance(config_dict, dict)
        assert "tsfresh" in config_dict
        assert "featuretools" in config_dict
        assert "autofeat" in config_dict

    def test_set_automl_config(self):
        """AutoML設定設定テスト"""
        new_config = AutoMLConfig.get_financial_optimized_config()

        self.service.set_automl_config(new_config)

        assert self.service.automl_config == new_config
        assert self.service.tsfresh_calculator.config == new_config.tsfresh

    def test_get_available_automl_features(self):
        """利用可能なAutoML特徴量取得テスト"""
        features = self.service.get_available_automl_features()

        assert isinstance(features, dict)
        assert "tsfresh" in features
        assert "featuretools" in features
        assert "autofeat" in features
        assert isinstance(features["tsfresh"], list)

    def test_clear_automl_cache(self):
        """AutoMLキャッシュクリアテスト"""
        # キャッシュにデータを追加
        self.service.tsfresh_calculator.feature_cache["test"] = "data"

        self.service.clear_automl_cache()

        assert len(self.service.tsfresh_calculator.feature_cache) == 0

    def test_validate_automl_config_valid(self):
        """有効なAutoML設定の検証テスト"""
        valid_config = {
            "tsfresh": {
                "feature_count_limit": 100,
                "fdr_level": 0.05,
                "parallel_jobs": 4,
            }
        }

        result = self.service.validate_automl_config(valid_config)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_automl_config_invalid(self):
        """無効なAutoML設定の検証テスト"""
        invalid_config = {
            "tsfresh": {
                "feature_count_limit": -10,  # 無効な値
                "fdr_level": 1.5,  # 無効な値
                "parallel_jobs": 0,  # 無効な値
            }
        }

        result = self.service.validate_automl_config(invalid_config)

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_automl_config_warnings(self):
        """警告が出るAutoML設定の検証テスト"""
        warning_config = {
            "tsfresh": {
                "feature_count_limit": 600,  # 警告が出る値
                "parallel_jobs": 16,  # 警告が出る値
            }
        }

        result = self.service.validate_automl_config(warning_config)

        assert result["valid"] is True  # エラーではないが警告あり
        assert len(result["warnings"]) > 0
