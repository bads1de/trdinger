"""
包括的パイプラインの包括的テスト

このモジュールは、前処理、特徴量選択、スケーリングを組み合わせた
包括的なデータ処理パイプラインの機能をテストします。
"""

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.utils.data_processing.pipelines.comprehensive_pipeline import (
    create_comprehensive_pipeline,
    create_eda_pipeline,
    create_production_pipeline,
    get_comprehensive_pipeline_info,
    optimize_comprehensive_pipeline,
    validate_comprehensive_pipeline,
)


class TestComprehensivePipeline:
    """包括的パイプラインの基本テスト"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """テスト用サンプルデータ.

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
    def sample_data_with_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """ターゲット付きサンプルデータ.

        Returns:
            特徴量DataFrameとターゲットSeriesのタプル
        """
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(200) for i in range(10)})
        y = pd.Series(np.random.randn(200))
        return X, y

    def test_pipeline_creation(self):
        """正常系: パイプラインの作成."""
        pipeline = create_comprehensive_pipeline()

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0

    def test_full_pipeline_execution(self, sample_data: pd.DataFrame):
        """正常系: フルパイプラインの実行."""
        pipeline = create_comprehensive_pipeline(
            outlier_removal=True,
            feature_selection=False,
            scaling=False,
        )
        result = pipeline.fit_transform(sample_data)

        assert result is not None
        assert isinstance(result, (pd.DataFrame, np.ndarray))

    def test_pipeline_with_feature_selection(
        self, sample_data_with_target: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 特徴量選択機能のテスト."""
        X, y = sample_data_with_target
        pipeline = create_comprehensive_pipeline(feature_selection=True, n_features=5)
        pipeline.fit(X, y)
        result = pipeline.transform(X)

        assert result is not None
        # 特徴量選択により特徴量数が減少
        assert result.shape[1] == 5

    def test_pipeline_with_all_features(self):
        """正常系: すべての機能を有効にしたパイプライン."""
        pipeline = create_comprehensive_pipeline(
            outlier_removal=True,
            feature_selection=True,
            n_features=5,
            scaling=True,
            scaling_method="standard",
            polynomial_features=False,
        )

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_with_polynomial_features(self, sample_data: pd.DataFrame):
        """正常系: 多項式特徴量の追加."""
        pipeline = create_comprehensive_pipeline(
            polynomial_features=True, polynomial_degree=2, interaction_only=False
        )

        result = pipeline.fit_transform(sample_data)

        assert result is not None
        # 多項式特徴量により特徴量数が増加
        assert result.shape[1] > sample_data.shape[1]


class TestPipelineConfiguration:
    """パイプライン設定のテスト"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """テスト用データ."""
        np.random.seed(42)
        return pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(10)})

    def test_outlier_removal_configuration(self, sample_data: pd.DataFrame):
        """正常系: 外れ値除去の設定."""
        pipeline = create_comprehensive_pipeline(
            outlier_removal=True,
            outlier_method="isolation_forest",
            outlier_contamination=0.1,
        )

        result = pipeline.fit_transform(sample_data)
        assert result is not None

    def test_imputation_strategies(self, sample_data: pd.DataFrame):
        """正常系: 補間戦略のテスト."""
        # 欠損値を追加
        data_with_nan = sample_data.copy()
        data_with_nan.iloc[10:20, 0] = np.nan

        for strategy in ["mean", "median"]:
            pipeline = create_comprehensive_pipeline(numeric_strategy=strategy)
            result = pipeline.fit_transform(data_with_nan)
            assert result is not None

    def test_scaling_methods(self, sample_data: pd.DataFrame):
        """正常系: スケーリング手法のテスト."""
        for method in ["standard", "robust", "minmax"]:
            pipeline = create_comprehensive_pipeline(
                scaling=True, scaling_method=method
            )
            result = pipeline.fit_transform(sample_data)
            assert result is not None

    def test_feature_selection_methods(self, sample_data: pd.DataFrame):
        """正常系: 特徴量選択手法のテスト."""
        target = pd.Series(np.random.randn(len(sample_data)))

        for method in ["f_regression", "mutual_info"]:
            pipeline = create_comprehensive_pipeline(
                feature_selection=True,
                n_features=5,
                selection_method=method,
            )
            pipeline.fit(sample_data, target)
            result = pipeline.transform(sample_data)
            assert result is not None


class TestProductionPipeline:
    """本番環境パイプラインのテスト"""

    @pytest.fixture
    def production_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """本番環境用データ."""
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(500) for i in range(30)})
        y = pd.Series(np.random.randn(500), name="target")
        return X, y

    def test_production_pipeline_creation(self):
        """正常系: 本番パイプラインの作成."""
        pipeline = create_production_pipeline(target_column="target")

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_production_pipeline_with_feature_selection(
        self, production_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 特徴量選択付き本番パイプライン."""
        X, y = production_data
        pipeline = create_production_pipeline(
            target_column="target", feature_selection=True, n_features=15
        )

        pipeline.fit(X, y)
        result = pipeline.transform(X)

        assert result is not None
        assert result.shape[1] == 15

    def test_production_pipeline_with_polynomial(
        self, production_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 多項式特徴量付き本番パイプライン."""
        X, y = production_data
        pipeline = create_production_pipeline(
            target_column="target",
            include_polynomial=True,
        )

        result = pipeline.fit_transform(X)
        assert result is not None

    def test_production_pipeline_scaling(
        self, production_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 本番パイプラインのスケーリング."""
        X, y = production_data

        for scaling_method in ["robust", "standard"]:
            pipeline = create_production_pipeline(
                target_column="target", scaling_method=scaling_method
            )
            result = pipeline.fit_transform(X)
            assert result is not None


class TestEDAPipeline:
    """EDA（探索的データ分析）パイプラインのテスト"""

    @pytest.fixture
    def eda_data(self) -> pd.DataFrame:
        """EDA用データ."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "price": np.random.uniform(100, 200, 200),
                "volume": np.random.uniform(1000, 5000, 200),
                "volatility": np.random.uniform(0.01, 0.05, 200),
            }
        )

    def test_eda_pipeline_creation(self):
        """正常系: EDAパイプラインの作成."""
        pipeline = create_eda_pipeline()

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_eda_pipeline_minimal_preprocessing(self, eda_data: pd.DataFrame):
        """正常系: 最小限の前処理でのEDAパイプライン."""
        pipeline = create_eda_pipeline(
            include_detailed_preprocessing=False,
            include_feature_engineering=False,
        )

        result = pipeline.fit_transform(eda_data)
        assert result is not None

    def test_eda_pipeline_with_feature_engineering(self, eda_data: pd.DataFrame):
        """正常系: 特徴量エンジニアリング付きEDAパイプライン."""
        pipeline = create_eda_pipeline(
            include_detailed_preprocessing=True,
            include_feature_engineering=True,
        )

        result = pipeline.fit_transform(eda_data)
        assert result is not None


class TestPipelineCaching:
    """パイプラインキャッシング機能のテスト"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """キャッシングテスト用データ."""
        np.random.seed(42)
        return pd.DataFrame({f"feature_{i}": np.random.randn(1000) for i in range(20)})

    def test_pipeline_performance(self, sample_data: pd.DataFrame):
        """正常系: パイプラインのパフォーマンス."""
        pipeline = create_comprehensive_pipeline(outlier_removal=True, scaling=True)

        # 1回目の実行
        start_time = time.time()
        pipeline.fit(sample_data)
        first_duration = time.time() - start_time

        # 2回目の実行（transformのみ）
        start_time = time.time()
        result = pipeline.transform(sample_data)
        second_duration = time.time() - start_time

        assert result is not None
        # transformはfitより高速であるべき
        assert second_duration <= first_duration

    def test_repeated_transform_consistency(self, sample_data: pd.DataFrame):
        """正常系: 繰り返しのtransformの一貫性."""
        pipeline = create_comprehensive_pipeline(scaling=True)
        pipeline.fit(sample_data)

        result1 = pipeline.transform(sample_data)
        result2 = pipeline.transform(sample_data)

        # 同じデータに対するtransformは同じ結果を返す
        if isinstance(result1, np.ndarray) and isinstance(result2, np.ndarray):
            np.testing.assert_array_almost_equal(result1, result2)
        else:
            pd.testing.assert_frame_equal(pd.DataFrame(result1), pd.DataFrame(result2))


class TestPipelineInfo:
    """パイプライン情報取得のテスト"""

    @pytest.fixture
    def fitted_pipeline(self) -> tuple[Pipeline, pd.DataFrame]:
        """適合済みパイプライン."""
        np.random.seed(42)
        data = pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(10)})
        target = pd.Series(np.random.randn(100))

        pipeline = create_comprehensive_pipeline(
            feature_selection=True, n_features=5, scaling=True
        )
        pipeline.fit(data, target)

        return pipeline, data

    def test_get_comprehensive_pipeline_info(
        self, fitted_pipeline: tuple[Pipeline, pd.DataFrame]
    ):
        """正常系: パイプライン情報の取得."""
        pipeline, _ = fitted_pipeline
        info = get_comprehensive_pipeline_info(pipeline)

        assert isinstance(info, dict)
        assert "pipeline_type" in info
        assert info["pipeline_type"] == "comprehensive"
        assert "n_steps" in info
        assert "step_names" in info

    def test_pipeline_info_components(
        self, fitted_pipeline: tuple[Pipeline, pd.DataFrame]
    ):
        """正常系: パイプライン構成要素の確認."""
        pipeline, _ = fitted_pipeline
        info = get_comprehensive_pipeline_info(pipeline)

        assert "has_preprocessing" in info
        assert "has_feature_selection" in info
        assert "has_scaling" in info
        assert "has_polynomial_features" in info


class TestPipelineValidation:
    """パイプライン検証のテスト"""

    @pytest.fixture
    def validation_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """検証用データ."""
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(150) for i in range(8)})
        y = pd.Series(np.random.randn(150))
        return X, y

    def test_validate_comprehensive_pipeline(
        self, validation_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: パイプラインの検証."""
        X, y = validation_data
        pipeline = create_comprehensive_pipeline(scaling=True)

        validation_results = validate_comprehensive_pipeline(pipeline, X, y)

        assert isinstance(validation_results, dict)
        assert "pipeline_creation" in validation_results
        assert "fit_success" in validation_results
        assert "transform_success" in validation_results
        assert "output_shape" in validation_results
        assert "processing_time" in validation_results

    def test_validation_with_errors(self):
        """異常系: エラー発生時の検証."""
        # 極小データでエラーを発生させる
        invalid_data = pd.DataFrame({"feature1": [1]})
        target = pd.Series([1])

        pipeline = create_comprehensive_pipeline(
            feature_selection=True, n_features=10  # データより多い特徴量を要求
        )

        validation_results = validate_comprehensive_pipeline(
            pipeline, invalid_data, target
        )

        # 検証結果を確認
        assert isinstance(validation_results, dict)
        assert "errors" in validation_results


class TestPipelineOptimization:
    """パイプライン最適化のテスト"""

    @pytest.fixture
    def optimization_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """最適化用データ."""
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(800) for i in range(40)})
        y = pd.Series(np.random.randn(800))
        return X, y

    def test_optimize_comprehensive_pipeline_regression(
        self, optimization_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 回帰タスク用の最適化."""
        X, y = optimization_data
        pipeline = optimize_comprehensive_pipeline(X, y, task_type="regression")

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_optimize_comprehensive_pipeline_classification(self):
        """正常系: 分類タスク用の最適化."""
        np.random.seed(42)
        X = pd.DataFrame({f"feature_{i}": np.random.randn(800) for i in range(40)})
        y = pd.Series(np.random.choice([0, 1], 800))

        pipeline = optimize_comprehensive_pipeline(X, y, task_type="classification")

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_optimization_with_time_budget(
        self, optimization_data: tuple[pd.DataFrame, pd.Series]
    ):
        """正常系: 時間予算付き最適化."""
        X, y = optimization_data
        time_budget = 30.0  # 30秒

        start_time = time.time()
        pipeline = optimize_comprehensive_pipeline(
            X, y, task_type="regression", time_budget=time_budget
        )
        duration = time.time() - start_time

        assert pipeline is not None
        # 時間予算内に完了（最適化自体は高速なので実際にはすぐ完了）
        assert duration < time_budget + 5  # バッファを含む

    @pytest.mark.parametrize("data_size", [500, 5000, 15000])
    def test_optimization_data_size_adaptation(self, data_size: int):
        """正常系: データサイズに応じた最適化.

        Args:
            data_size: データサイズ
        """
        np.random.seed(42)
        X = pd.DataFrame(
            {f"feature_{i}": np.random.randn(data_size) for i in range(30)}
        )
        y = pd.Series(np.random.randn(data_size))

        pipeline = optimize_comprehensive_pipeline(X, y, task_type="regression")

        assert pipeline is not None
        # データサイズに応じて適切な設定がなされる
        info = get_comprehensive_pipeline_info(pipeline)
        assert "n_steps" in info


class TestPipelineErrorHandling:
    """パイプラインのエラーハンドリング"""

    def test_empty_dataframe(self):
        """異常系: 空のDataFrame."""
        empty_df = pd.DataFrame()
        pipeline = create_comprehensive_pipeline()

        # 空のDataFrameはエラーを発生させるか、空の結果を返す
        try:
            result = pipeline.fit_transform(empty_df)
            # エラーが発生しない場合、結果が空であることを確認
            assert len(result) == 0 or result.shape[1] == 0
        except (ValueError, IndexError, KeyError):
            # エラーが発生するのも正常
            pass

    def test_invalid_configuration(self):
        """異常系: 無効な設定."""
        with pytest.raises(ValueError):
            create_comprehensive_pipeline(scaling=True, scaling_method="invalid_method")

    def test_mismatched_dimensions(self):
        """異常系: 次元の不一致."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {f"feature_{i}": np.random.randn(100) for i in range(10)}
        )
        X_test = pd.DataFrame({f"feature_{i}": np.random.randn(50) for i in range(5)})

        pipeline = create_comprehensive_pipeline(scaling=True)
        pipeline.fit(X_train)

        # 異なる特徴量数でのtransformはエラー
        with pytest.raises((ValueError, KeyError)):
            pipeline.transform(X_test)


class TestPipelinePerformance:
    """パイプラインのパフォーマンステスト"""

    def test_large_dataset_processing(self):
        """正常系: 大規模データセットの処理."""
        np.random.seed(42)
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(20000) for i in range(30)}
        )

        pipeline = create_comprehensive_pipeline(outlier_removal=True, scaling=True)

        start_time = time.time()
        result = pipeline.fit_transform(large_data)
        duration = time.time() - start_time

        assert result is not None
        # 合理的な時間内に完了
        assert duration < 60, f"処理時間が長すぎます: {duration}秒"

    def test_memory_efficiency(self):
        """正常系: メモリ効率のテスト."""
        np.random.seed(42)
        data = pd.DataFrame({f"feature_{i}": np.random.randn(5000) for i in range(50)})

        pipeline = create_comprehensive_pipeline(optimize_dtypes=True)
        result = pipeline.fit_transform(data)

        assert result is not None


class TestPipelineIntegration:
    """パイプライン統合テスト"""

    @pytest.fixture
    def realistic_trading_data(self) -> pd.DataFrame:
        """現実的な取引データ.

        トレンド、ボラティリティ、異常値を含むデータ
        """
        np.random.seed(42)
        n_samples = 1000

        # 価格データ
        trend = np.linspace(40000, 50000, n_samples)
        noise = np.random.randn(n_samples) * 500
        close = trend + noise

        # ボリュームデータ
        volume = np.random.lognormal(10, 1, n_samples)

        # 異常値を追加
        close[100:103] = close[100:103] * 1.3
        volume[200:202] = volume[200:202] * 3

        return pd.DataFrame(
            {
                "open": close + np.random.randn(n_samples) * 100,
                "high": close + np.abs(np.random.randn(n_samples) * 100),
                "low": close - np.abs(np.random.randn(n_samples) * 100),
                "close": close,
                "volume": volume,
            }
        )

    def test_end_to_end_comprehensive_pipeline(
        self, realistic_trading_data: pd.DataFrame
    ):
        """正常系: エンドツーエンド包括的パイプライン."""
        target = pd.Series(np.random.randn(len(realistic_trading_data)))

        pipeline = create_comprehensive_pipeline(
            outlier_removal=True,
            outlier_method="isolation_forest",
            feature_selection=True,
            n_features=3,
            scaling=True,
            scaling_method="robust",
        )

        # パイプライン実行
        pipeline.fit(realistic_trading_data, target)
        result = pipeline.transform(realistic_trading_data)

        # 結果の検証
        assert result is not None
        assert len(result) > 0
        # numpy配列の場合、NaNチェックを適切に行う
        if isinstance(result, np.ndarray):
            assert not np.isnan(result).any()
        else:
            assert not result.isnull().any().any()

    def test_comprehensive_pipeline_all_features(
        self, realistic_trading_data: pd.DataFrame
    ):
        """正常系: 全機能を使用した包括的パイプライン."""
        target = pd.Series(np.random.randn(len(realistic_trading_data)))

        pipeline = create_comprehensive_pipeline(
            outlier_removal=True,
            numeric_strategy="median",
            feature_selection=True,
            n_features=4,
            selection_method="f_regression",
            scaling=True,
            scaling_method="robust",
            polynomial_features=False,
            optimize_dtypes=True,
        )

        pipeline.fit(realistic_trading_data, target)
        result = pipeline.transform(realistic_trading_data)

        # すべての処理が正常に実行される
        assert result is not None
        assert result.shape[1] == 4  # 特徴量選択により4つの特徴量

    def test_production_ready_pipeline(self, realistic_trading_data: pd.DataFrame):
        """正常系: 本番環境対応パイプライン."""
        target = pd.Series(np.random.randn(len(realistic_trading_data)), name="returns")

        pipeline = create_production_pipeline(
            target_column="returns",
            feature_selection=True,
            n_features=3,
            scaling_method="robust",
        )

        # トレーニングデータでfitてテストデータでtransform
        train_size = int(len(realistic_trading_data) * 0.8)
        train_data = realistic_trading_data[:train_size]
        test_data = realistic_trading_data[train_size:]
        train_target = target[:train_size]

        pipeline.fit(train_data, train_target)
        test_result = pipeline.transform(test_data)

        # テストデータが正常に処理される
        assert test_result is not None
        assert len(test_result) == len(test_data)
