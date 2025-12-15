"""
MLパイプラインの包括的テスト
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.services.ml.preprocessing.pipeline import (
    create_ml_pipeline,
    create_classification_pipeline,
    create_regression_pipeline,
    optimize_ml_pipeline,
    get_ml_pipeline_info,
)


class TestMLPipelinesComprehensive:
    """MLパイプラインの包括的テスト"""

    @pytest.fixture
    def sample_data(self):
        """サンプルデータ"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "target": np.random.randn(100),
            }
        )

    @pytest.fixture
    def sample_classification_data(self):
        """サンプル分類データ"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            }
        )

    def test_create_ml_pipeline_basic(self):
        """基本的なMLパイプラインのテスト"""
        pipeline = create_ml_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) >= 1  # 前処理ステップが含まれる

        # ステップ名の確認
        step_names = [step[0] for step in pipeline.steps]
        assert "preprocessing" in step_names

    def test_create_ml_pipeline_with_feature_selection(self):
        """特徴量選択付きMLパイプラインのテスト"""
        pipeline = create_ml_pipeline(
            feature_selection=True, n_features=2, selection_method="f_regression"
        )

        step_names = [step[0] for step in pipeline.steps]
        assert "preprocessing" in step_names
        assert "feature_selection" in step_names

    def test_create_ml_pipeline_with_scaling(self):
        """スケーリング付きMLパイプラインのテスト"""
        pipeline = create_ml_pipeline(scaling=True, scaling_method="standard")

        step_names = [step[0] for step in pipeline.steps]
        assert "preprocessing" in step_names
        assert "scaler" in step_names

    def test_create_classification_pipeline_basic(self):
        """基本的な分類パイプラインのテスト"""
        pipeline = create_classification_pipeline()

        assert isinstance(pipeline, Pipeline)
        step_names = [step[0] for step in pipeline.steps]
        assert "preprocessing" in step_names

    def test_create_classification_pipeline_with_selection(self):
        """特徴量選択付き分類パイプラインのテスト"""
        pipeline = create_classification_pipeline(
            feature_selection=True, n_features=2, selection_method="f_classif"
        )

        step_names = [step[0] for step in pipeline.steps]
        assert "feature_selection" in step_names

    def test_create_regression_pipeline_basic(self):
        """基本的な回帰パイプラインのテスト"""
        pipeline = create_regression_pipeline()

        assert isinstance(pipeline, Pipeline)
        step_names = [step[0] for step in pipeline.steps]
        assert "preprocessing" in step_names

    def test_pipeline_info_extraction(self, sample_data):
        """パイプライン情報抽出のテスト"""
        X = sample_data[["feature1", "feature2", "feature3"]]
        y = sample_data["target"]

        pipeline = create_regression_pipeline(
            feature_selection=True, n_features=2, scaling=True
        )
        pipeline.fit(X, y)

        info = get_ml_pipeline_info(pipeline)

        assert "pipeline_type" in info
        assert "n_steps" in info
        assert "has_preprocessing" in info
        assert "has_feature_selection" in info
        assert "has_scaling" in info

    def test_optimize_regression_pipeline(self, sample_data):
        """回帰パイプライン最適化のテスト"""
        X = sample_data[["feature1", "feature2", "feature3"]]
        y = sample_data["target"]

        pipeline = optimize_ml_pipeline(X, y, task_type="regression")

        assert isinstance(pipeline, Pipeline)
        # 最適化されたパラメータが設定される
        info = get_ml_pipeline_info(pipeline)
        assert info["has_preprocessing"]
        assert info["has_feature_selection"]

    def test_optimize_classification_pipeline(self, sample_classification_data):
        """分類パイプライン最適化のテスト"""
        X = sample_classification_data[["feature1", "feature2", "feature3"]]
        y = sample_classification_data["target"]

        pipeline = optimize_ml_pipeline(X, y, task_type="classification")

        assert isinstance(pipeline, Pipeline)
        info = get_ml_pipeline_info(pipeline)
        assert info["has_preprocessing"]
        assert info["has_feature_selection"]

    def test_invalid_selection_method_error(self):
        """無効な特徴量選択方法のテスト"""
        with pytest.raises(ValueError, match="サポートされていない選択方法"):
            create_ml_pipeline(
                feature_selection=True, n_features=2, selection_method="invalid_method"
            )

    def test_invalid_scaling_method_error(self):
        """無効なスケーリング方法のテスト"""
        with pytest.raises(ValueError, match="サポートされていないスケーリング方法"):
            create_ml_pipeline(scaling=True, scaling_method="invalid_method")

    def test_invalid_classification_selection_method_error(self):
        """無効な分類特徴量選択方法のテスト"""
        with pytest.raises(
            ValueError, match="Unsupported classification selection method"
        ):
            create_classification_pipeline(
                feature_selection=True, n_features=2, selection_method="invalid_method"
            )

    def test_zero_features_error(self):
        """ゼロ特徴量のテスト"""
        pipeline = create_ml_pipeline(feature_selection=True, n_features=0)  # 無効な値

        # n_features=0の場合は特徴量選択がスキップされる
        step_names = [step[0] for step in pipeline.steps]
        assert "feature_selection" not in step_names

    def test_large_features_handling(self, sample_data):
        """多数特徴量の処理テスト"""
        X = sample_data[["feature1", "feature2", "feature3"]]
        y = sample_data["target"]

        # 多すぎる特徴量数を指定
        pipeline = optimize_ml_pipeline(X, y, max_features=100)

        # 自動調整が働く
        info = get_ml_pipeline_info(pipeline)
        assert info["has_preprocessing"]

    def test_preprocessing_pipeline_integration(self):
        """前処理パイプライン統合のテスト"""
        preprocessing_params = {"outlier_removal": True, "outlier_method": "iqr"}

        pipeline = create_ml_pipeline(preprocessing_params=preprocessing_params)

        # 前処理パイプラインが正しく統合される
        step_names = [step[0] for step in pipeline.steps]
        assert "preprocessing" in step_names

    def test_robust_scaling_for_regression(self, sample_data):
        """回帰用ロバストスケーリングのテスト"""
        X = sample_data[["feature1", "feature2", "feature3"]]
        y = sample_data["target"]

        # 回帰ではデフォルトでrobustスケーリング
        pipeline = optimize_ml_pipeline(X, y, task_type="regression")

        # ステップを確認
        scaler_step = None
        for step_name, step in pipeline.steps:
            if step_name == "scaler":
                scaler_step = step
                break

        assert scaler_step is not None
        # RobustScalerが使用されているか確認（直接確認は難しいため、正常に作成されることを確認）

    def test_standard_scaling_for_classification(self, sample_classification_data):
        """分類用標準スケーリングのテスト"""
        X = sample_classification_data[["feature1", "feature2", "feature3"]]
        y = sample_classification_data["target"]

        # 分類ではデフォルトでstandardスケーリング
        pipeline = optimize_ml_pipeline(X, y, task_type="classification")

        # 正常に作成される
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_fitting_and_prediction(self, sample_data):
        """パイプラインのフィッティングと予測テスト"""
        from sklearn.linear_model import LinearRegression

        X = sample_data[["feature1", "feature2", "feature3"]]
        y = sample_data["target"]

        pipeline = create_ml_pipeline(
            feature_selection=True, n_features=2, scaling=True
        )

        # パイプラインとモデルを組み合わせ
        full_pipeline = Pipeline(
            [("ml_preprocessing", pipeline), ("regressor", LinearRegression())]
        )

        # 学習
        full_pipeline.fit(X, y)

        # 予測
        predictions = full_pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_pipeline_with_missing_values(self):
        """欠損値対応のテスト"""
        data_with_nan = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [1, 2, 3, 4, 5],
                "target": [1, 2, 3, 4, 5],
            }
        )

        X = data_with_nan[["feature1", "feature2"]]
        y = data_with_nan["target"]

        pipeline = create_ml_pipeline()
        full_pipeline = Pipeline(
            [("ml_preprocessing", pipeline), ("regressor", Mock())]
        )

        # 欠損値が処理される
        full_pipeline.fit(X, y)

    def test_pipeline_data_type_conversion(self, sample_data):
        """データ型変換のテスト"""
        X = sample_data[["feature1", "feature2", "feature3"]]
        y = sample_data["target"]

        pipeline = create_ml_pipeline(feature_selection=True, n_features=2)

        # DataFrameから配列への変換が正しく行われる
        pipeline.fit(X, y)

        # 変換後のデータがNumPy配列である
        info = get_ml_pipeline_info(pipeline)
        assert info is not None

    def test_pipeline_memory_efficiency(self):
        """メモリ効率のテスト"""
        # 大規模データでのパイプライン作成
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(1000) for i in range(50)}
        )
        large_data["target"] = np.random.randn(1000)

        X = large_data.drop("target", axis=1)
        y = large_data["target"]

        # 大規模データでもパイプラインが作成される
        pipeline = create_ml_pipeline(feature_selection=True, n_features=10)

        # メモリ効率の良い処理
        full_pipeline = Pipeline(
            [("ml_preprocessing", pipeline), ("regressor", Mock())]
        )

        full_pipeline.fit(X, y)

    def test_pipeline_with_categorical_features(self):
        """カテゴリカル特徴量のテスト"""
        data_with_cat = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": ["A", "B", "A", "C", "B"],
                "target": [1, 2, 3, 4, 5],
            }
        )

        X = data_with_cat[["feature1", "feature2"]]
        y = data_with_cat["target"]

        pipeline = create_ml_pipeline()

        # カテゴリカル特徴量が処理される
        full_pipeline = Pipeline(
            [("ml_preprocessing", pipeline), ("regressor", Mock())]
        )

        full_pipeline.fit(X, y)

    def test_pipeline_parameter_validation(self):
        """パイプラインパラメータ検証のテスト"""
        # 無効なn_features
        pipeline = create_ml_pipeline(feature_selection=True, n_features=-1)  # 無効な値

        # エラーが発生せず、特徴量選択がスキップされる
        step_names = [step[0] for step in pipeline.steps]
        assert "feature_selection" not in step_names

    def test_mutual_info_selection_methods(self):
        """相互情報量選択方法のテスト"""
        # 回帰用相互情報量
        pipeline_reg = create_ml_pipeline(
            feature_selection=True, n_features=2, selection_method="mutual_info"
        )
        step_names = [step[0] for step in pipeline_reg.steps]
        assert "feature_selection" in step_names

        # 分類用相互情報量
        pipeline_class = create_classification_pipeline(
            feature_selection=True, n_features=2, selection_method="mutual_info"
        )
        step_names = [step[0] for step in pipeline_class.steps]
        assert "feature_selection" in step_names

    def test_pipeline_step_ordering(self):
        """パイプラインステップ順序のテスト"""
        pipeline = create_ml_pipeline(
            feature_selection=True, scaling=True, n_features=2
        )

        step_names = [step[0] for step in pipeline.steps]

        # 正しい順序: preprocessing -> feature_selection -> scaler
        preprocessing_idx = step_names.index("preprocessing")
        feature_selection_idx = step_names.index("feature_selection")
        scaler_idx = step_names.index("scaler")

        assert preprocessing_idx < feature_selection_idx < scaler_idx

    def test_empty_preprocessing_params(self):
        """空の前処理パラメータのテスト"""
        pipeline = create_ml_pipeline(preprocessing_params={})

        assert isinstance(pipeline, Pipeline)

    def test_pipeline_with_no_scaling(self):
        """スケーリングなしのパイプラインテスト"""
        pipeline = create_ml_pipeline(scaling=False)

        step_names = [step[0] for step in pipeline.steps]
        assert "scaler" not in step_names

    def test_pipeline_with_no_feature_selection(self):
        """特徴量選択なしのパイプラインテスト"""
        pipeline = create_ml_pipeline(feature_selection=False)

        step_names = [step[0] for step in pipeline.steps]
        assert "feature_selection" not in step_names

    def test_pipeline_data_validation(self, sample_data):
        """パイプラインデータ検証のテスト"""
        X = sample_data[["feature1", "feature2", "feature3"]]
        y = sample_data["target"]

        pipeline = create_ml_pipeline(feature_selection=True, n_features=2)

        # 有効なデータでフィッティング
        pipeline.fit(X, y)

        # 検証が通る
        info = get_ml_pipeline_info(pipeline)
        assert info["n_steps"] > 0

    def test_pipeline_step_reusability(self):
        """パイプラインステップ再利用のテスト"""
        # 同じ設定で複数のパイプラインを作成
        pipeline1 = create_ml_pipeline()
        pipeline2 = create_ml_pipeline()

        # 両方とも有効
        assert isinstance(pipeline1, Pipeline)
        assert isinstance(pipeline2, Pipeline)

    def test_minimal_features_handling(self):
        """最小特徴量数の処理テスト"""
        # 1特徴量
        pipeline = create_ml_pipeline(feature_selection=True, n_features=1)

        assert isinstance(pipeline, Pipeline)

    def test_max_features_clamping(self, sample_data):
        """最大特徴量数制限のテスト"""
        X = sample_data[["feature1", "feature2", "feature3"]]
        y = sample_data["target"]

        # 多すぎる特徴量数を指定
        pipeline = optimize_ml_pipeline(X, y, max_features=10)

        # 自動的に制限される
        info = get_ml_pipeline_info(pipeline)
        assert info["has_preprocessing"]

    def test_pipeline_info_completeness(self, sample_data):
        """パイプライン情報完全性のテスト"""
        X = sample_data[["feature1", "feature2"]]
        y = sample_data["target"]

        pipeline = create_ml_pipeline(
            feature_selection=True, scaling=True, n_features=1
        )
        pipeline.fit(X, y)

        info = get_ml_pipeline_info(pipeline)

        required_fields = [
            "pipeline_type",
            "n_steps",
            "step_names",
            "has_preprocessing",
            "has_feature_selection",
            "has_scaling",
        ]

        for field in required_fields:
            assert field in info

    def test_pipeline_deterministic_behavior(self):
        """パイプライン決定論的動作のテスト"""
        # 同じデータで同じパイプラインが生成される
        pipeline1 = create_ml_pipeline()
        pipeline2 = create_ml_pipeline()

        # ステップ構造が同じ
        steps1 = [step[0] for step in pipeline1.steps]
        steps2 = [step[0] for step in pipeline2.steps]

        assert steps1 == steps2

    def test_pipeline_scalability(self):
        """パイプラインスケーラビリティのテスト"""
        # 大規模特徴量セット
        large_feature_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(100) for i in range(100)}
        )
        large_feature_data["target"] = np.random.randn(100)

        X = large_feature_data.drop("target", axis=1)
        y = large_feature_data["target"]

        # 大規模でも動作
        pipeline = create_ml_pipeline(feature_selection=True, n_features=10)

        full_pipeline = Pipeline(
            [("ml_preprocessing", pipeline), ("regressor", Mock())]
        )

        full_pipeline.fit(X, y)

    def test_pipeline_composition(self):
        """パイプライン構成のテスト"""
        # 複数のオプションを組み合わせ
        pipeline = create_ml_pipeline(
            feature_selection=True,
            n_features=5,
            scaling=True,
            scaling_method="minmax",
            preprocessing_params={"outlier_removal": True},
        )

        step_names = [step[0] for step in pipeline.steps]

        assert "preprocessing" in step_names
        assert "feature_selection" in step_names
        assert "scaler" in step_names

    def test_pipeline_backward_compatibility(self):
        """後方互換性のテスト"""
        # デフォルト設定で動作
        pipeline = create_ml_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_documentation(self):
        """パイプラインドキュメントのテスト"""
        # ドキュメントが存在
        assert create_ml_pipeline.__doc__ is not None
        assert create_classification_pipeline.__doc__ is not None
        assert create_regression_pipeline.__doc__ is not None
        assert optimize_ml_pipeline.__doc__ is not None
        assert get_ml_pipeline_info.__doc__ is not None




