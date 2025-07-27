"""
TSFreshFeatureCalculatorの単体テスト

TSFresh特徴量計算クラスの基本的な動作確認テストを実装します。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from app.services.ml.feature_engineering.automl_features.tsfresh_calculator import (
    TSFreshFeatureCalculator,
    TSFRESH_AVAILABLE,
)
from app.services.ml.feature_engineering.automl_features.automl_config import (
    TSFreshConfig,
)


class TestTSFreshCalculator:
    """TSFreshFeatureCalculatorのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.config = TSFreshConfig(
            enabled=True,
            feature_selection=True,
            fdr_level=0.05,
            feature_count_limit=50,
            parallel_jobs=1,  # テスト用に1に設定
        )
        self.calculator = TSFreshFeatureCalculator(self.config)

    def create_test_ohlcv_data(self, rows: int = 100) -> pd.DataFrame:
        """テスト用のOHLCVデータを作成"""
        np.random.seed(42)  # 再現性のため

        dates = pd.date_range(start="2023-01-01", periods=rows, freq="1h")

        # 現実的な価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, rows)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # 最低価格を設定

        prices = np.array(prices)

        # OHLCV データを生成
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

    def create_test_target(self, rows: int = 100) -> pd.Series:
        """テスト用のターゲット変数を作成"""
        np.random.seed(42)
        return pd.Series(np.random.choice([0, 1, 2], size=rows), name="target")

    def test_initialization(self):
        """初期化テスト"""
        # デフォルト設定での初期化
        calculator = TSFreshFeatureCalculator()
        assert calculator.config is not None
        assert calculator.feature_cache == {}
        assert calculator.selected_features is None

        # カスタム設定での初期化
        custom_config = TSFreshConfig(feature_count_limit=200)
        calculator_custom = TSFreshFeatureCalculator(custom_config)
        assert calculator_custom.config.feature_count_limit == 200

    def test_prepare_timeseries_data(self):
        """時系列データ変換テスト"""
        test_data = self.create_test_ohlcv_data(50)

        ts_data = self.calculator._prepare_timeseries_data(test_data)

        assert isinstance(ts_data, pd.DataFrame)
        assert "id" in ts_data.columns
        assert "time" in ts_data.columns
        assert "value" in ts_data.columns

        # 各価格系列が含まれているか確認
        unique_ids = ts_data["id"].unique()
        expected_ids = ["Open", "High", "Low", "Close", "Volume"]
        for expected_id in expected_ids:
            assert expected_id in unique_ids

    def test_prepare_timeseries_data_empty_input(self):
        """空データでの時系列変換テスト"""
        empty_df = pd.DataFrame()

        ts_data = self.calculator._prepare_timeseries_data(empty_df)

        assert isinstance(ts_data, pd.DataFrame)
        assert ts_data.empty

    def test_get_financial_feature_settings(self):
        """金融特徴量設定テスト"""
        settings = self.calculator._get_financial_feature_settings()

        assert isinstance(settings, dict)

        # 重要な金融特徴量が含まれているか確認
        expected_features = [
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "autocorrelation",
            "linear_trend",
            "number_peaks",
        ]

        for feature in expected_features:
            assert feature in settings

    def test_align_target_with_features(self):
        """ターゲット変数調整テスト"""
        test_data = self.create_test_ohlcv_data(50)
        target = self.create_test_target(50)

        # 同じ長さの場合
        features = pd.DataFrame(np.random.randn(50, 10))
        aligned_target = self.calculator._align_target_with_features(target, features)

        assert aligned_target is not None
        assert len(aligned_target) <= len(target)
        assert len(aligned_target) <= len(features)

    def test_align_target_with_features_different_lengths(self):
        """異なる長さでのターゲット変数調整テスト"""
        target = self.create_test_target(100)
        features = pd.DataFrame(np.random.randn(80, 10))

        aligned_target = self.calculator._align_target_with_features(target, features)

        assert aligned_target is not None
        assert len(aligned_target) <= min(len(target), len(features))

    def test_merge_features_with_original(self):
        """特徴量結合テスト"""
        original_df = self.create_test_ohlcv_data(50)
        features = pd.DataFrame(
            np.random.randn(50, 5),
            columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
        )

        result_df = self.calculator._merge_features_with_original(original_df, features)

        assert len(result_df) == len(original_df)

        # TSFreshプレフィックスが追加されているか確認
        for col in features.columns:
            expected_col = f"TSF_{col}"
            assert expected_col in result_df.columns

    def test_get_feature_names(self):
        """特徴量名取得テスト"""
        # 選択された特徴量がない場合
        feature_names = self.calculator.get_feature_names()
        assert isinstance(feature_names, list)
        assert all(name.startswith("TSF_") for name in feature_names)

        # 選択された特徴量がある場合
        self.calculator.selected_features = ["mean", "std", "skewness"]
        feature_names = self.calculator.get_feature_names()
        expected_names = ["TSF_mean", "TSF_std", "TSF_skewness"]
        assert feature_names == expected_names

    def test_clear_cache(self):
        """キャッシュクリアテスト"""
        # キャッシュにデータを追加
        self.calculator.feature_cache["test"] = "data"
        assert len(self.calculator.feature_cache) > 0

        # キャッシュをクリア
        self.calculator.clear_cache()
        assert len(self.calculator.feature_cache) == 0

    @pytest.mark.skipif(
        not TSFRESH_AVAILABLE, reason="TSFreshライブラリが利用できません"
    )
    def test_calculate_tsfresh_features_basic(self):
        """基本的なTSFresh特徴量計算テスト"""
        test_data = self.create_test_ohlcv_data(100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.calculator.calculate_tsfresh_features(test_data)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(test_data)

        # 元の列が保持されているか確認
        for col in test_data.columns:
            assert col in result_df.columns

        # TSFresh特徴量が追加されているか確認
        tsfresh_cols = [col for col in result_df.columns if col.startswith("TSF_")]
        if TSFRESH_AVAILABLE:
            assert len(tsfresh_cols) > 0

    @pytest.mark.skipif(
        not TSFRESH_AVAILABLE, reason="TSFreshライブラリが利用できません"
    )
    def test_calculate_tsfresh_features_with_target(self):
        """ターゲット変数ありでのTSFresh特徴量計算テスト"""
        test_data = self.create_test_ohlcv_data(100)
        target = self.create_test_target(100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = self.calculator.calculate_tsfresh_features(
                test_data, target=target, feature_selection=True
            )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(test_data)

    def test_calculate_tsfresh_features_empty_input(self):
        """空データでのTSFresh特徴量計算テスト"""
        empty_df = pd.DataFrame()

        result_df = self.calculator.calculate_tsfresh_features(empty_df)

        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty

    def test_calculate_tsfresh_features_none_input(self):
        """Noneデータでの特徴量計算テスト"""
        result_df = self.calculator.calculate_tsfresh_features(None)

        assert result_df is None

    @patch(
        "app.services.ml.feature_engineering.automl_features.tsfresh_calculator.TSFRESH_AVAILABLE",
        False,
    )
    def test_calculate_tsfresh_features_library_unavailable(self):
        """TSFreshライブラリが利用できない場合のテスト"""
        test_data = self.create_test_ohlcv_data(50)

        result_df = self.calculator.calculate_tsfresh_features(test_data)

        # 元のDataFrameがそのまま返されることを確認
        pd.testing.assert_frame_equal(result_df, test_data)
