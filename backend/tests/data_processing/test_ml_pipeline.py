"""
MLパイプラインの包括的テスト

このモジュールは、機械学習に特化したデータ処理パイプラインの
各機能を徹底的にテストします。
"""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.services.ml.preprocessing.pipeline import (
    create_ml_pipeline,
    create_classification_pipeline,
    create_regression_pipeline,
    get_ml_pipeline_info,
    optimize_ml_pipeline,
)


class TestMLPipeline:
    """MLパイプラインの基本テスト"""

    @pytest.fixture
    def sample_ohlcv_data(self) -> pd.DataFrame:
        """テスト用OHLCVデータ.

        Returns:
            1000行のOHLCVデータを含むDataFrame
        """
        np.random.seed(42)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1h"),
                "open": np.random.uniform(40000, 50000, 1000),
                "high": np.random.uniform(40000, 50000, 1000),
                "low": np.random.uniform(40000, 50000, 1000),
                "close": np.random.uniform(40000, 50000, 1000),
                "volume": np.random.uniform(100, 1000, 1000),
            }
        )

    @pytest.fixture
    def sample_numeric_data(self) -> pd.DataFrame:
        """数値データのみのサンプル.

        Returns:
            数値特徴量を含むDataFrame
        """
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "feature4": np.random.randn(100),
                "feature5": np.random.randn(100),
            }
        )

    @pytest.fixture
    def sample_target(self) -> pd.Series:
        """ターゲット変数.

        Returns:
            ターゲット値を含むSeries
        """
        np.random.seed(42)
        return pd.Series(np.random.randn(100))

    def test_pipeline_initialization(self):
        """正常系: パイプラインの初期化."""
        pipeline = create_ml_pipeline()

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0

    def test_pipeline_with_feature_selection(self):
        """正常系: 特徴量選択付きパイプライン."""
        pipeline = create_ml_pipeline(
            feature_selection=True, n_features=3, selection_method="f_regression"
        )

        assert pipeline is not None
        step_names = [step[0] for step in pipeline.steps]
        assert "feature_selection" in step_names

    def test_pipeline_with_scaling(self):
        """正常系: スケーリング付きパイプライン."""
        pipeline = create_ml_pipeline(scaling=True, scaling_method="standard")

        assert pipeline is not None
        step_names = [step[0] for step in pipeline.steps]
        assert "scaler" in step_names

    def test_pipeline_fit_transform(self, sample_numeric_data: pd.DataFrame):
        """正常系: fit_transformの実行."""
        pipeline = create_ml_pipeline(scaling=True)
        transformed_data = pipeline.fit_transform(sample_numeric_data)

        assert transformed_data is not None
        assert len(transformed_data) > 0
        # スケーリング後は配列になる
        assert isinstance(transformed_data, np.ndarray)

    def test_pipeline_transform_only(self, sample_numeric_data: pd.DataFrame):
        """正常系: fitなしでのtransform."""
        pipeline = create_ml_pipeline(scaling=True)
        pipeline.fit(sample_numeric_data)

        new_data = sample_numeric_data.iloc[-50:]
        transformed = pipeline.transform(new_data)

        assert len(transformed) > 0

    def test_pipeline_with_missing_data(self):
        """異常系: 欠損値を含むデータ."""
        data_with_nan = pd.DataFrame(
            {
                "close": [100, np.nan, 102, 103, 104],
                "volume": [1000, 1100, np.nan, 1200, 1300],
            }
        )

        pipeline = create_ml_pipeline()
        # 欠損値は前処理で処理される
        transformed = pipeline.fit_transform(data_with_nan)
        assert transformed is not None

    def test_pipeline_empty_data(self):
        """異常系: 空のデータフレーム."""
        empty_df = pd.DataFrame()
        pipeline = create_ml_pipeline()

        with pytest.raises((ValueError, IndexError)):
            pipeline.fit_transform(empty_df)

    def test_pipeline_with_custom_config(self):
        """正常系: カスタム設定でのパイプライン."""
        pipeline = create_ml_pipeline(
            feature_selection=True,
            n_features=5,
            selection_method="mutual_info",
            scaling=True,
            scaling_method="robust",
        )

        assert pipeline is not None
        step_names = [step[0] for step in pipeline.steps]
        assert "feature_selection" in step_names
        assert "scaler" in step_names

    def test_pipeline_transformer_order(self):
        """正常系: トランスフォーマーの実行順序."""
        pipeline = create_ml_pipeline(
            feature_selection=True, n_features=3, scaling=True
        )

        step_names = [step[0] for step in pipeline.steps]

        # 前処理 -> 特徴量選択 -> スケーリングの順序
        assert "preprocessing" in step_names
        preprocessing_idx = step_names.index("preprocessing")

        if "feature_selection" in step_names:
            feature_selection_idx = step_names.index("feature_selection")
            assert preprocessing_idx < feature_selection_idx

        if "scaler" in step_names:
            scaler_idx = step_names.index("scaler")
            if "feature_selection" in step_names:
                assert feature_selection_idx < scaler_idx


class TestMLPipelineScaling:
    """スケーリング機能のテスト"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """スケーリング用サンプルデータ."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 100, 50),
                "feature2": np.random.uniform(-50, 50, 50),
                "feature3": np.random.uniform(1000, 2000, 50),
            }
        )

    @pytest.mark.parametrize("scaler_method", ["standard", "minmax", "robust"])
    def test_different_scalers(self, scaler_method: str, sample_data: pd.DataFrame):
        """正常系: 複数のスケーリング手法をテスト.

        Args:
            scaler_method: スケーリング手法
            sample_data: テストデータ
        """
        pipeline = create_ml_pipeline(scaling=True, scaling_method=scaler_method)
        transformed = pipeline.fit_transform(sample_data)

        assert transformed is not None
        assert len(transformed) == len(sample_data)

    def test_standard_scaler_properties(self, sample_data: pd.DataFrame):
        """正常系: 標準スケーラーの特性確認."""
        pipeline = create_ml_pipeline(scaling=True, scaling_method="standard")
        transformed = pipeline.fit_transform(sample_data)

        # 標準化後は平均≈0、標準偏差≈1
        assert np.abs(transformed.mean()) < 0.1
        assert np.abs(transformed.std() - 1.0) < 0.1

    def test_minmax_scaler_properties(self, sample_data: pd.DataFrame):
        """正常系: MinMaxスケーラーの特性確認."""
        pipeline = create_ml_pipeline(scaling=True, scaling_method="minmax")
        transformed = pipeline.fit_transform(sample_data)

        # MinMax後は[0, 1]の範囲
        assert transformed.min() >= -0.01  # 浮動小数点誤差を考慮
        assert transformed.max() <= 1.01


class TestMLPipelineFeatureSelection:
    """特徴量選択機能のテスト"""

    @pytest.fixture
    def sample_features(self) -> pd.DataFrame:
        """特徴量選択用データ."""
        np.random.seed(42)
        n_samples = 200
        return pd.DataFrame(
            {f"feature_{i}": np.random.randn(n_samples) for i in range(20)}
        )

    @pytest.fixture
    def sample_target(self) -> pd.Series:
        """ターゲット変数."""
        np.random.seed(42)
        return pd.Series(np.random.randn(200))

    def test_feature_selection_reduces_features(
        self, sample_features: pd.DataFrame, sample_target: pd.Series
    ):
        """正常系: 特徴量選択で特徴量数が減少."""
        n_features_to_select = 10
        pipeline = create_ml_pipeline(
            feature_selection=True, n_features=n_features_to_select, scaling=False
        )

        pipeline.fit(sample_features, sample_target)
        transformed = pipeline.transform(sample_features)

        assert transformed.shape[1] == n_features_to_select

    @pytest.mark.parametrize("selection_method", ["f_regression", "mutual_info"])
    def test_different_selection_methods(
        self,
        selection_method: str,
        sample_features: pd.DataFrame,
        sample_target: pd.Series,
    ):
        """正常系: 異なる特徴量選択手法.

        Args:
            selection_method: 選択手法
            sample_features: 特徴量データ
            sample_target: ターゲット変数
        """
        pipeline = create_ml_pipeline(
            feature_selection=True,
            n_features=5,
            selection_method=selection_method,
        )

        pipeline.fit(sample_features, sample_target)
        transformed = pipeline.transform(sample_features)

        assert transformed.shape[1] == 5


class TestRegressionPipeline:
    """回帰パイプラインのテスト"""

    @pytest.fixture
    def regression_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """回帰用データ."""
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(10)})
        y = pd.Series(np.random.randn(100))
        return X, y

    def test_regression_pipeline_creation(self):
        """正常系: 回帰パイプラインの作成."""
        pipeline = create_regression_pipeline()

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_regression_pipeline_with_robust_scaling(
        self, regression_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: ロバストスケーリングを使用した回帰パイプライン."""
        X, y = regression_data
        pipeline = create_regression_pipeline(scaling=True, scaling_method="robust")

        pipeline.fit(X, y)
        transformed = pipeline.transform(X)

        assert transformed is not None
        assert len(transformed) == len(X)


class TestClassificationPipeline:
    """分類パイプラインのテスト"""

    @pytest.fixture
    def classification_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """分類用データ."""
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(10)})
        y = pd.Series(np.random.choice([0, 1], 100))
        return X, y

    def test_classification_pipeline_creation(self):
        """正常系: 分類パイプラインの作成."""
        pipeline = create_classification_pipeline()

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_classification_with_feature_selection(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 特徴量選択付き分類パイプライン."""
        X, y = classification_data
        pipeline = create_classification_pipeline(
            feature_selection=True,
            n_features=5,
            selection_method="f_classif",
        )

        pipeline.fit(X, y)
        transformed = pipeline.transform(X)

        assert transformed.shape[1] == 5


class TestPipelineInfo:
    """パイプライン情報取得のテスト"""

    @pytest.fixture
    def fitted_pipeline(self) -> tuple[Pipeline, pd.DataFrame]:
        """適合済みパイプライン."""
        np.random.seed(42)
        data = pd.DataFrame({f"feature_{i}": np.random.randn(50) for i in range(5)})
        target = pd.Series(np.random.randn(50))

        pipeline = create_ml_pipeline(
            feature_selection=True, n_features=3, scaling=True
        )
        pipeline.fit(data, target)

        return pipeline, data

    def test_get_ml_pipeline_info(self, fitted_pipeline: tuple[Pipeline, pd.DataFrame]):
        """正常系: パイプライン情報の取得."""
        pipeline, _ = fitted_pipeline
        info = get_ml_pipeline_info(pipeline)

        assert isinstance(info, dict)
        assert "pipeline_type" in info
        assert info["pipeline_type"] == "ml"
        assert "n_steps" in info
        assert "step_names" in info
        assert "has_preprocessing" in info
        assert "has_feature_selection" in info
        assert "has_scaling" in info

    def test_pipeline_info_step_names(
        self, fitted_pipeline: tuple[Pipeline, pd.DataFrame]
    ):
        """正常系: ステップ名の確認."""
        pipeline, _ = fitted_pipeline
        info = get_ml_pipeline_info(pipeline)

        assert isinstance(info["step_names"], list)
        assert len(info["step_names"]) > 0


class TestOptimizePipeline:
    """パイプライン最適化のテスト"""

    @pytest.fixture
    def optimization_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """最適化用データ."""
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(500) for i in range(50)})
        y = pd.Series(np.random.randn(500))
        return X, y

    def test_optimize_ml_pipeline_regression(
        self, optimization_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 回帰タスク用の最適化."""
        X, y = optimization_data
        pipeline = optimize_ml_pipeline(X, y, task_type="regression")

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_optimize_ml_pipeline_classification(self):
        """正常系: 分類タスク用の最適化."""
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(500) for i in range(50)})
        y = pd.Series(np.random.choice([0, 1], 500))

        pipeline = optimize_ml_pipeline(X, y, task_type="classification")

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_optimize_with_max_features(
        self, optimization_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 最大特徴量数指定での最適化."""
        X, y = optimization_data
        max_features = 20

        pipeline = optimize_ml_pipeline(
            X, y, task_type="regression", max_features=max_features
        )

        assert pipeline is not None


class TestPipelineErrorHandling:
    """パイプラインのエラーハンドリング"""

    def test_invalid_data_type(self):
        """異常系: 無効なデータ型."""
        pipeline = create_ml_pipeline()

        with pytest.raises((AttributeError, ValueError)):
            pipeline.fit_transform([1, 2, 3])  # リストは不可

    def test_invalid_scaling_method(self):
        """異常系: 無効なスケーリング方法."""
        with pytest.raises(ValueError) as exc_info:
            create_ml_pipeline(scaling=True, scaling_method="invalid_method")

        assert "サポートされていない" in str(exc_info.value)

    def test_invalid_selection_method(self):
        """異常系: 無効な特徴量選択方法."""
        with pytest.raises(ValueError) as exc_info:
            create_ml_pipeline(
                feature_selection=True,
                n_features=5,
                selection_method="invalid_method",
            )

        assert "サポートされていない" in str(exc_info.value)

    def test_insufficient_data_for_feature_selection(self):
        """異常系: 特徴量選択に不十分なデータ."""
        small_data = pd.DataFrame({"value": [1, 2, 3]})
        pipeline = create_ml_pipeline(feature_selection=True, n_features=10)

        # データ不足の場合は適切に処理される
        with pytest.raises((ValueError, IndexError)):
            pipeline.fit_transform(small_data)


class TestPipelinePerformance:
    """パイプラインのパフォーマンステスト"""

    def test_large_dataset_processing(self):
        """正常系: 大規模データセットの処理."""
        np.random.seed(42)
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(10000) for i in range(20)}
        )

        pipeline = create_ml_pipeline(scaling=True)

        start_time = time.time()
        result = pipeline.fit_transform(large_data)
        duration = time.time() - start_time

        assert result is not None
        # 合理的な時間内に完了
        assert duration < 30, f"処理時間が長すぎます: {duration}秒"

    def test_pipeline_caching_behavior(self):
        """正常系: パイプラインのキャッシング動作."""
        np.random.seed(42)
        data = pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(10)})

        pipeline = create_ml_pipeline(scaling=True)

        # 1回目の実行
        start_time = time.time()
        pipeline.fit(data)
        first_duration = time.time() - start_time

        # 2回目の実行（既にfitされている）
        start_time = time.time()
        result = pipeline.transform(data)
        second_duration = time.time() - start_time

        assert result is not None
        # transformはfitより高速
        assert second_duration <= first_duration


class TestPipelineIntegration:
    """パイプライン統合テスト"""

    @pytest.fixture
    def realistic_market_data(self) -> pd.DataFrame:
        """現実的な市場データ.

        トレンド、ボラティリティ、異常値を含むデータ
        """
        np.random.seed(42)
        n_samples = 500

        # トレンド成分
        trend = np.linspace(100, 150, n_samples)

        # ランダムウォーク
        random_walk = np.cumsum(np.random.randn(n_samples) * 2)

        # 組み合わせ
        close = trend + random_walk

        # 異常値を追加
        close[100:105] = close[100:105] * 1.5

        return pd.DataFrame(
            {
                "open": close + np.random.randn(n_samples),
                "high": close + np.abs(np.random.randn(n_samples)),
                "low": close - np.abs(np.random.randn(n_samples)),
                "close": close,
                "volume": np.random.uniform(1000, 10000, n_samples),
            }
        )

    def test_end_to_end_pipeline(self, realistic_market_data: pd.DataFrame):
        """正常系: エンドツーエンドパイプライン."""
        target = pd.Series(np.random.randn(len(realistic_market_data)))

        pipeline = create_ml_pipeline(
            feature_selection=True,
            n_features=3,
            selection_method="f_regression",
            scaling=True,
            scaling_method="robust",
        )

        # パイプライン実行
        pipeline.fit(realistic_market_data, target)
        result = pipeline.transform(realistic_market_data)

        # 結果の検証
        assert result is not None
        assert len(result) == len(realistic_market_data)
        assert result.shape[1] == 3  # 3つの特徴量が選択されている

    def test_pipeline_with_all_options(self, realistic_market_data: pd.DataFrame):
        """正常系: すべてのオプションを有効にしたパイプライン."""
        target = pd.Series(np.random.randn(len(realistic_market_data)))

        pipeline = create_ml_pipeline(
            feature_selection=True,
            n_features=4,
            selection_method="mutual_info",
            scaling=True,
            scaling_method="standard",
        )

        pipeline.fit(realistic_market_data, target)
        result = pipeline.transform(realistic_market_data)

        # すべての処理が正常に実行される
        assert result is not None
        assert not np.isnan(result).any()




