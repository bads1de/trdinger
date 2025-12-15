"""
トランスフォーマーの単体テスト

このモジュールは、データ処理パイプラインで使用される
各トランスフォーマーの個別動作を徹底的にテストします。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from app.utils.data_processing.pipelines.preprocessing_pipeline import (
    CategoricalEncoderTransformer,
    CategoricalPipelineTransformer,
    DtypeOptimizerTransformer,
    MixedTypeTransformer,
    OutlierRemovalTransformer,
    create_preprocessing_pipeline,
)


class TestOutlierRemovalTransformer:
    """外れ値除去トランスフォーマーのテスト"""

    @pytest.fixture
    def normal_data(self) -> pd.DataFrame:
        """正常なデータ."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )

    @pytest.fixture
    def data_with_outliers(self) -> pd.DataFrame:
        """外れ値を含むデータ."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        # 外れ値を追加
        data.loc[0, "feature1"] = 100  # 極端に大きい値
        data.loc[1, "feature2"] = -100  # 極端に小さい値
        return data

    def test_transformer_initialization(self):
        """正常系: トランスフォーマーの初期化."""
        transformer = OutlierRemovalTransformer(
            method="isolation_forest", contamination=0.1
        )

        assert transformer is not None
        assert transformer.method == "isolation_forest"
        assert transformer.contamination == 0.1

    def test_fit_method(self, normal_data: pd.DataFrame):
        """正常系: fitメソッド."""
        transformer = OutlierRemovalTransformer(method="isolation_forest")
        transformer.fit(normal_data)

        assert transformer.detector_ is not None
        assert isinstance(transformer.detector_, IsolationForest)

    def test_transform_method(self, data_with_outliers: pd.DataFrame):
        """正常系: transformメソッド."""
        transformer = OutlierRemovalTransformer(
            method="isolation_forest", contamination=0.1
        )
        transformer.fit(data_with_outliers)
        result = transformer.transform(data_with_outliers)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data_with_outliers)

    def test_fit_transform(self, data_with_outliers: pd.DataFrame):
        """正常系: fit_transformメソッド."""
        transformer = OutlierRemovalTransformer(method="isolation_forest")
        result = transformer.fit_transform(data_with_outliers)

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_outlier_detection(self, data_with_outliers: pd.DataFrame):
        """正常系: 外れ値が適切に検出される."""
        transformer = OutlierRemovalTransformer(
            method="isolation_forest", contamination=0.05
        )
        transformer.fit(data_with_outliers)
        result = transformer.transform(data_with_outliers)

        # 外れ値が中央値で置き換えられている
        assert not np.any(np.abs(result["feature1"]) > 50)
        assert not np.any(np.abs(result["feature2"]) > 50)

    def test_numpy_array_input(self):
        """正常系: numpy配列の入力."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        data[0, 0] = 100  # 外れ値

        transformer = OutlierRemovalTransformer(method="isolation_forest")
        result = transformer.fit_transform(data)

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_get_feature_names_out(self):
        """正常系: 特徴名の取得."""
        transformer = OutlierRemovalTransformer()
        feature_names = ["feat1", "feat2", "feat3"]
        output_names = transformer.get_feature_names_out(feature_names)

        assert output_names == feature_names


class TestCategoricalEncoderTransformer:
    """カテゴリエンコーダートランスフォーマーのテスト"""

    @pytest.fixture
    def categorical_data(self) -> pd.DataFrame:
        """カテゴリカルデータ."""
        return pd.DataFrame(
            {
                "category1": ["A", "B", "C", "A", "B"] * 20,
                "category2": ["X", "Y", "Z", "X", "Y"] * 20,
                "numeric": np.random.randn(100),
            }
        )

    @pytest.fixture
    def data_with_missing(self) -> pd.DataFrame:
        """欠損値を含むカテゴリカルデータ."""
        data = pd.DataFrame(
            {
                "category1": ["A", "B", None, "A", "B"] * 20,
                "category2": ["X", None, "Z", "X", "Y"] * 20,
            }
        )
        return data

    def test_transformer_initialization(self):
        """正常系: トランスフォーマーの初期化."""
        transformer = CategoricalEncoderTransformer(encoding_type="label")

        assert transformer is not None
        assert transformer.encoding_type == "label"

    def test_fit_method(self, categorical_data: pd.DataFrame):
        """正常系: fitメソッド."""
        transformer = CategoricalEncoderTransformer(encoding_type="label")
        transformer.fit(categorical_data)

        assert len(transformer.encoders_) > 0
        assert "category1" in transformer.encoders_
        assert "category2" in transformer.encoders_

    def test_transform_method(self, categorical_data: pd.DataFrame):
        """正常系: transformメソッド."""
        transformer = CategoricalEncoderTransformer(encoding_type="label")
        transformer.fit(categorical_data)
        result = transformer.transform(categorical_data)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # カテゴリ列が数値にエンコードされている
        assert pd.api.types.is_numeric_dtype(result["category1"])
        assert pd.api.types.is_numeric_dtype(result["category2"])

    def test_encoding_with_missing_values(self, data_with_missing: pd.DataFrame):
        """正常系: 欠損値を含むデータのエンコーディング."""
        transformer = CategoricalEncoderTransformer(encoding_type="label")
        transformer.fit(data_with_missing)
        result = transformer.transform(data_with_missing)

        assert result is not None
        # 欠損値が適切に処理されている
        assert not result.isnull().any().any()

    def test_fit_transform(self, categorical_data: pd.DataFrame):
        """正常系: fit_transformメソッド."""
        transformer = CategoricalEncoderTransformer()
        result = transformer.fit_transform(categorical_data)

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_get_feature_names_out(self):
        """正常系: 特徴名の取得."""
        transformer = CategoricalEncoderTransformer()
        feature_names = ["cat1", "cat2"]
        output_names = transformer.get_feature_names_out(feature_names)

        assert output_names == feature_names


class TestDtypeOptimizerTransformer:
    """データ型最適化トランスフォーマーのテスト"""

    @pytest.fixture
    def unoptimized_data(self) -> pd.DataFrame:
        """最適化前のデータ."""
        return pd.DataFrame(
            {
                "small_int": np.array([1, 2, 3, 4, 5] * 20, dtype=np.int64),
                "medium_int": np.array([100, 200, 300, 400, 500] * 20, dtype=np.int64),
                "float_col": np.array([1.1, 2.2, 3.3, 4.4, 5.5] * 20, dtype=np.float64),
            }
        )

    def test_transformer_initialization(self):
        """正常系: トランスフォーマーの初期化."""
        transformer = DtypeOptimizerTransformer()

        assert transformer is not None

    def test_fit_method(self, unoptimized_data: pd.DataFrame):
        """正常系: fitメソッド."""
        transformer = DtypeOptimizerTransformer()
        result = transformer.fit(unoptimized_data)

        assert result is transformer

    def test_transform_method(self, unoptimized_data: pd.DataFrame):
        """正常系: transformメソッド."""
        transformer = DtypeOptimizerTransformer()
        result = transformer.transform(unoptimized_data)

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_dtype_optimization(self, unoptimized_data: pd.DataFrame):
        """正常系: データ型の最適化."""
        original_memory = unoptimized_data.memory_usage(deep=True).sum()

        transformer = DtypeOptimizerTransformer()
        optimized = transformer.fit_transform(unoptimized_data)

        optimized_memory = optimized.memory_usage(deep=True).sum()

        # メモリ使用量が減少または同等
        assert optimized_memory <= original_memory

    def test_integer_optimization(self):
        """正常系: 整数型の最適化."""
        data = pd.DataFrame(
            {
                "uint8_col": [1, 2, 3, 4, 5],
                "uint16_col": [100, 200, 300, 400, 500],
                "int8_col": [-5, -4, -3, -2, -1],
            }
        )

        transformer = DtypeOptimizerTransformer()
        optimized = transformer.transform(data)

        # 適切な整数型に最適化されている
        assert optimized["uint8_col"].dtype in [
            np.uint8,
            np.uint16,
            np.uint32,
        ]
        assert optimized["int8_col"].dtype in [np.int8, np.int16, np.int32]

    def test_get_feature_names_out(self):
        """正常系: 特徴名の取得."""
        transformer = DtypeOptimizerTransformer()
        feature_names = ["feat1", "feat2"]
        output_names = transformer.get_feature_names_out(feature_names)

        assert output_names == feature_names


class TestCategoricalPipelineTransformer:
    """カテゴリカルパイプライントランスフォーマーのテスト"""

    @pytest.fixture
    def categorical_data(self) -> pd.DataFrame:
        """カテゴリカルデータ."""
        return pd.DataFrame(
            {
                "category1": ["A", "B", "C", None, "A"] * 20,
                "category2": ["X", "Y", None, "X", "Y"] * 20,
            }
        )

    def test_transformer_initialization(self):
        """正常系: トランスフォーマーの初期化."""
        transformer = CategoricalPipelineTransformer(
            strategy="most_frequent",
            fill_value="Unknown",
            encoding=True,
            categorical_encoding="label",
        )

        assert transformer is not None
        assert transformer.strategy == "most_frequent"
        assert transformer.fill_value == "Unknown"

    def test_fit_method(self, categorical_data: pd.DataFrame):
        """正常系: fitメソッド."""
        transformer = CategoricalPipelineTransformer()
        transformer.fit(categorical_data)

        assert transformer.imputer_ is not None
        assert transformer.encoder_ is not None

    def test_transform_method(self, categorical_data: pd.DataFrame):
        """正常系: transformメソッド."""
        transformer = CategoricalPipelineTransformer()
        transformer.fit(categorical_data)
        result = transformer.transform(categorical_data)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # 欠損値が埋められている
        assert not result.isnull().any().any()

    def test_imputation_only(self, categorical_data: pd.DataFrame):
        """正常系: 補間のみ（エンコーディングなし）."""
        # カテゴリカルデータから欠損値を除いた簡易版でテスト
        clean_categorical = pd.DataFrame(
            {
                "category1": ["A", "B", "C", "A", "B"] * 20,
                "category2": ["X", "Y", "Z", "X", "Y"] * 20,
            }
        )

        transformer = CategoricalPipelineTransformer(encoding=False)
        transformer.fit(clean_categorical)
        result = transformer.transform(clean_categorical)

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_get_feature_names_out(self):
        """正常系: 特徴名の取得."""
        transformer = CategoricalPipelineTransformer()
        feature_names = ["cat1", "cat2"]
        output_names = transformer.get_feature_names_out(feature_names)

        assert output_names == feature_names


class TestMixedTypeTransformer:
    """混合型トランスフォーマーのテスト"""

    @pytest.fixture
    def mixed_data(self) -> pd.DataFrame:
        """混合型データ."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "numeric1": np.random.randn(100),
                "numeric2": np.random.randn(100),
                "category1": ["A", "B", "C"] * 33 + ["A"],
                "category2": ["X", "Y", "Z"] * 33 + ["X"],
            }
        )

    @pytest.fixture
    def numeric_pipeline(self):
        """数値パイプライン."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        return Pipeline([("imputer", SimpleImputer(strategy="median"))])

    @pytest.fixture
    def categorical_pipeline(self):
        """カテゴリカルパイプライン."""
        return CategoricalPipelineTransformer(strategy="most_frequent", encoding=True)

    def test_transformer_initialization(self, numeric_pipeline, categorical_pipeline):
        """正常系: トランスフォーマーの初期化."""
        transformer = MixedTypeTransformer(numeric_pipeline, categorical_pipeline)

        assert transformer is not None
        assert transformer.numeric_pipeline is not None
        assert transformer.categorical_pipeline is not None

    def test_fit_method(
        self, mixed_data: pd.DataFrame, numeric_pipeline, categorical_pipeline
    ):
        """正常系: fitメソッド."""
        transformer = MixedTypeTransformer(numeric_pipeline, categorical_pipeline)
        transformer.fit(mixed_data)

        assert transformer.numeric_columns_ is not None
        assert transformer.categorical_columns_ is not None
        assert len(transformer.numeric_columns_) == 2
        assert len(transformer.categorical_columns_) == 2

    def test_transform_method(
        self, mixed_data: pd.DataFrame, numeric_pipeline, categorical_pipeline
    ):
        """正常系: transformメソッド."""
        transformer = MixedTypeTransformer(numeric_pipeline, categorical_pipeline)
        transformer.fit(mixed_data)
        result = transformer.transform(mixed_data)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data)

    def test_separate_processing(
        self, mixed_data: pd.DataFrame, numeric_pipeline, categorical_pipeline
    ):
        """正常系: 数値とカテゴリが別々に処理される."""
        transformer = MixedTypeTransformer(numeric_pipeline, categorical_pipeline)
        transformer.fit(mixed_data)
        result = transformer.transform(mixed_data)

        # 数値列とカテゴリ列が適切に処理されている
        assert len(result.columns) == len(mixed_data.columns)

    def test_numpy_array_input(self, numeric_pipeline, categorical_pipeline):
        """正常系: numpy配列の入力."""
        np.random.seed(42)
        data = np.random.randn(100, 4)

        transformer = MixedTypeTransformer(numeric_pipeline, categorical_pipeline)
        result = transformer.fit_transform(data)

        assert result is not None

    def test_get_feature_names_out(
        self, mixed_data: pd.DataFrame, numeric_pipeline, categorical_pipeline
    ):
        """正常系: 特徴名の取得."""
        transformer = MixedTypeTransformer(numeric_pipeline, categorical_pipeline)
        transformer.fit(mixed_data)

        output_names = transformer.get_feature_names_out()

        assert output_names is not None
        assert len(output_names) > 0


class TestPreprocessingPipelineIntegration:
    """前処理パイプラインの統合テスト"""

    @pytest.fixture
    def complex_data(self) -> pd.DataFrame:
        """複雑なデータ."""
        np.random.seed(42)

        # カテゴリカルデータを作成（欠損値なし）
        cat1_values = ["A", "B", "C", "D"] * 50
        cat2_values = ["X", "Y", "W", "Z"] * 50

        data = pd.DataFrame(
            {
                "numeric1": np.random.randn(200),
                "numeric2": np.random.randn(200),
                "category1": cat1_values,
                "category2": cat2_values,
            }
        )
        # 外れ値を追加
        data.loc[0, "numeric1"] = 100
        # 数値列に欠損値を追加
        data.loc[10:15, "numeric2"] = np.nan
        # カテゴリ列に欠損値を追加（np.nanを使用）
        data.loc[20:25, "category1"] = np.nan
        data.loc[30:35, "category2"] = np.nan

        return data

    def test_full_pipeline_execution(self, complex_data: pd.DataFrame):
        """正常系: フルパイプラインの実行."""
        pipeline = create_preprocessing_pipeline(
            outlier_method="isolation_forest",
            numeric_strategy="median",
            categorical_strategy="most_frequent",
            categorical_encoding="label",
            optimize_dtypes=True,
        )

        result = pipeline.fit_transform(complex_data)

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_pipeline_without_outlier_removal(self, complex_data: pd.DataFrame):
        """正常系: 外れ値除去なしのパイプライン."""
        pipeline = create_preprocessing_pipeline(
            outlier_method=None, categorical_encoding="label"
        )

        result = pipeline.fit_transform(complex_data)
        assert result is not None

    def test_pipeline_components(self, complex_data: pd.DataFrame):
        """正常系: パイプライン構成要素の確認."""
        pipeline = create_preprocessing_pipeline()

        assert len(pipeline.steps) > 0
        step_names = [step[0] for step in pipeline.steps]
        assert "preprocessor" in step_names

    def test_pipeline_preserves_data_integrity(self, complex_data: pd.DataFrame):
        """正常系: データの整合性が保たれる."""
        pipeline = create_preprocessing_pipeline()
        result = pipeline.fit_transform(complex_data)

        # 行数が保たれる
        assert len(result) == len(complex_data)


class TestTransformerEdgeCases:
    """トランスフォーマーのエッジケーステスト"""

    def test_empty_dataframe(self):
        """異常系: 空のDataFrame."""
        empty_df = pd.DataFrame()
        transformer = OutlierRemovalTransformer()

        with pytest.raises((ValueError, IndexError)):
            transformer.fit_transform(empty_df)

    def test_single_row_dataframe(self):
        """エッジケース: 1行のDataFrame."""
        single_row = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        transformer = OutlierRemovalTransformer()

        # 1行でも処理できる
        result = transformer.fit_transform(single_row)
        assert result is not None

    def test_all_nan_column(self):
        """エッジケース: 全てNaNの列."""
        data = pd.DataFrame(
            {"feature1": [np.nan] * 10, "feature2": np.random.randn(10)}
        )

        pipeline = create_preprocessing_pipeline()
        # NaN列が適切に処理される
        result = pipeline.fit_transform(data)
        assert result is not None

    def test_single_category(self):
        """エッジケース: 単一カテゴリ."""
        data = pd.DataFrame({"category": ["A"] * 100})

        encoder = CategoricalEncoderTransformer()
        result = encoder.fit_transform(data)

        assert result is not None
        # 単一カテゴリでもエンコード可能
        assert len(result["category"].unique()) == 1

    def test_high_cardinality_category(self):
        """エッジケース: 高いカーディナリティのカテゴリ."""
        np.random.seed(42)
        data = pd.DataFrame({"category": [f"cat_{i}" for i in range(1000)]})

        encoder = CategoricalEncoderTransformer()
        result = encoder.fit_transform(data)

        assert result is not None
        # 高カーディナリティでも処理可能
        assert pd.api.types.is_numeric_dtype(result["category"])


class TestTransformerConsistency:
    """トランスフォーマーの一貫性テスト"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """サンプルデータ."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )

    def test_repeated_transform_consistency(self, sample_data: pd.DataFrame):
        """正常系: 繰り返しのtransformの一貫性."""
        transformer = OutlierRemovalTransformer()
        transformer.fit(sample_data)

        result1 = transformer.transform(sample_data)
        result2 = transformer.transform(sample_data)

        # 同じ入力に対して同じ出力
        pd.testing.assert_frame_equal(result1, result2)

    def test_fit_idempotency(self, sample_data: pd.DataFrame):
        """正常系: fitの冪等性."""
        transformer = OutlierRemovalTransformer()

        transformer.fit(sample_data)
        result1 = transformer.transform(sample_data)

        transformer.fit(sample_data)
        result2 = transformer.transform(sample_data)

        # 複数回fitしても同じ結果
        pd.testing.assert_frame_equal(result1, result2)


class TestTransformerPerformance:
    """トランスフォーマーのパフォーマンステスト"""

    def test_large_dataset_performance(self):
        """正常系: 大規模データセットのパフォーマンス."""
        np.random.seed(42)
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(10000) for i in range(20)}
        )

        import time

        transformer = OutlierRemovalTransformer()

        start_time = time.time()
        result = transformer.fit_transform(large_data)
        duration = time.time() - start_time

        assert result is not None
        # 合理的な時間内に完了
        assert duration < 10, f"処理時間が長すぎます: {duration}秒"

    def test_categorical_encoding_performance(self):
        """正常系: カテゴリカルエンコーディングのパフォーマンス."""
        np.random.seed(42)
        large_categorical = pd.DataFrame(
            {
                f"cat_{i}": np.random.choice(["A", "B", "C", "D", "E"], 5000)
                for i in range(10)
            }
        )

        import time

        encoder = CategoricalEncoderTransformer()

        start_time = time.time()
        result = encoder.fit_transform(large_categorical)
        duration = time.time() - start_time

        assert result is not None
        assert duration < 5, f"処理時間が長すぎます: {duration}秒"


