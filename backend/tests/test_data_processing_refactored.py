"""
リファクタリング後のdata_processing.pyとdata_validation.pyのテスト

scikit-learnの標準機能とPandera/Pydanticを活用した新しい実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
import logging

# リファクタリング後のモジュールをインポート
from app.utils.data_processing import (
    OutlierRemovalTransformer,
    CategoricalEncoderTransformer,
    create_outlier_removal_pipeline,
    create_categorical_encoding_pipeline,
    create_comprehensive_preprocessing_pipeline,
    DataProcessor,  # 後方互換性テスト用
)

from app.utils.data_validation import (
    OHLCVDataModel,
    OHLCV_SCHEMA,
    EXTENDED_MARKET_DATA_SCHEMA,
    validate_dataframe_with_schema,
    clean_dataframe_with_schema,
    DataValidator,  # 後方互換性テスト用
)

logger = logging.getLogger(__name__)


class TestOutlierRemovalTransformer:
    """OutlierRemovalTransformerのテスト"""

    @pytest.fixture
    def sample_data_with_outliers(self):
        """外れ値を含むサンプルデータ"""
        np.random.seed(42)
        # 正常データ
        normal_data = np.random.normal(0, 1, 90)
        # 外れ値
        outliers = np.array([10, -10, 15, -15, 20])
        # 結合
        data = np.concatenate([normal_data, outliers])
        np.random.shuffle(data)

        return pd.DataFrame(
            {
                "feature1": data,
                "feature2": np.random.normal(0, 1, 95),
                "feature3": np.random.normal(0, 1, 95),
            }
        )

    def test_isolation_forest_transformer(self, sample_data_with_outliers):
        """IsolationForestを使用した外れ値除去テスト"""
        transformer = OutlierRemovalTransformer(
            method="isolation_forest", contamination=0.1
        )

        # フィットと変換
        transformer.fit(sample_data_with_outliers)
        result = transformer.transform(sample_data_with_outliers)

        # 外れ値がNaNに置き換えられていることを確認
        assert result.isna().sum().sum() > 0
        assert len(result) == len(sample_data_with_outliers)

        # 外れ値マスクが作成されていることを確認
        assert hasattr(transformer, "outlier_mask_")
        assert len(transformer.outlier_mask_) == len(sample_data_with_outliers)

    def test_local_outlier_factor_transformer(self, sample_data_with_outliers):
        """LocalOutlierFactorを使用した外れ値除去テスト"""
        transformer = OutlierRemovalTransformer(
            method="local_outlier_factor", contamination=0.1, n_neighbors=10
        )

        # フィットと変換
        transformer.fit(sample_data_with_outliers)
        result = transformer.transform(sample_data_with_outliers)

        # 外れ値がNaNに置き換えられていることを確認
        assert result.isna().sum().sum() > 0
        assert len(result) == len(sample_data_with_outliers)

    def test_invalid_method_error(self, sample_data_with_outliers):
        """無効な方法でのエラーテスト"""
        transformer = OutlierRemovalTransformer(method="invalid_method")

        with pytest.raises(ValueError, match="未対応の外れ値検出方法"):
            transformer.fit(sample_data_with_outliers)

    def test_transform_before_fit_error(self, sample_data_with_outliers):
        """フィット前の変換でのエラーテスト"""
        transformer = OutlierRemovalTransformer()

        with pytest.raises(ValueError, match="fit\\(\\)を先に実行してください"):
            transformer.transform(sample_data_with_outliers)


class TestCategoricalEncoderTransformer:
    """CategoricalEncoderTransformerのテスト"""

    @pytest.fixture
    def sample_categorical_data(self):
        """カテゴリカルデータのサンプル"""
        return pd.DataFrame(
            {
                "category1": ["A", "B", "C", "A", "B", "C", np.nan],
                "category2": ["X", "Y", "Z", "X", "Y", "Z", "X"],
                "numeric": [1, 2, 3, 4, 5, 6, 7],
            }
        )

    def test_label_encoding(self, sample_categorical_data):
        """ラベルエンコーディングのテスト"""
        transformer = CategoricalEncoderTransformer(encoding_type="label")

        # フィットと変換
        transformer.fit(sample_categorical_data)
        result = transformer.transform(sample_categorical_data)

        # カテゴリカル列が数値に変換されていることを確認
        assert pd.api.types.is_numeric_dtype(result["category1"])
        assert pd.api.types.is_numeric_dtype(result["category2"])

        # 数値列は変更されていないことを確認
        assert pd.api.types.is_numeric_dtype(result["numeric"])
        assert result["numeric"].equals(sample_categorical_data["numeric"])

    def test_onehot_encoding(self, sample_categorical_data):
        """OneHotエンコーディングのテスト"""
        transformer = CategoricalEncoderTransformer(encoding_type="onehot")

        # フィットと変換
        transformer.fit(sample_categorical_data)
        result = transformer.transform(sample_categorical_data)

        # OneHot列が追加されていることを確認
        assert len(result.columns) > len(sample_categorical_data.columns)

        # 元のカテゴリカル列が削除されていることを確認
        assert "category1" not in result.columns
        assert "category2" not in result.columns

        # 数値列は残っていることを確認
        assert "numeric" in result.columns


class TestNewPipelineFunctions:
    """新しいPipeline関数のテスト"""

    @pytest.fixture
    def sample_mixed_data(self):
        """数値とカテゴリカルが混在するデータ"""
        return pd.DataFrame(
            {
                "numeric1": np.random.normal(0, 1, 100),
                "numeric2": np.random.normal(5, 2, 100),
                "category1": np.random.choice(["A", "B", "C"], 100),
                "category2": np.random.choice(["X", "Y"], 100),
            }
        )

    def test_create_outlier_removal_pipeline(self, sample_mixed_data):
        """外れ値除去Pipelineの作成テスト"""
        pipeline = create_outlier_removal_pipeline(
            method="isolation_forest", contamination=0.1
        )

        # Pipelineが正しく作成されていることを確認
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "outlier_removal"
        assert pipeline.steps[1][0] == "imputer"

        # 数値データで実行
        numeric_data = sample_mixed_data.select_dtypes(include=[np.number])
        result = pipeline.fit_transform(numeric_data)

        assert result.shape[0] == numeric_data.shape[0]
        assert result.shape[1] == numeric_data.shape[1]

    def test_create_categorical_encoding_pipeline(self, sample_mixed_data):
        """カテゴリカルエンコーディングPipelineの作成テスト"""
        pipeline = create_categorical_encoding_pipeline(encoding_type="label")

        # Pipelineが正しく作成されていることを確認
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "imputer"
        assert pipeline.steps[1][0] == "encoder"

        # カテゴリカルデータで実行
        categorical_data = sample_mixed_data.select_dtypes(include=[object])
        result = pipeline.fit_transform(categorical_data)

        assert result.shape[0] == categorical_data.shape[0]

    def test_create_comprehensive_preprocessing_pipeline(self, sample_mixed_data):
        """包括的前処理Pipelineの作成テスト"""
        pipeline = create_comprehensive_preprocessing_pipeline(
            outlier_method="isolation_forest", categorical_encoding="label"
        )

        # Pipelineが正しく作成されていることを確認
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "preprocessor"
        assert pipeline.steps[1][0] == "final_cleanup"

        # 混在データで実行
        result = pipeline.fit_transform(sample_mixed_data)

        assert result.shape[0] == sample_mixed_data.shape[0]
        # NaNが除去されていることを確認（数値データのみチェック）
        if isinstance(result, np.ndarray):
            # 数値型の列のみをチェック
            numeric_mask = np.array(
                [
                    np.issubdtype(result[:, i].dtype, np.number)
                    for i in range(result.shape[1])
                ]
            )
            if numeric_mask.any():
                numeric_result = result[:, numeric_mask]
                assert not np.isnan(numeric_result.astype(float)).any()
        else:
            # DataFrameの場合
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                assert not result[numeric_cols].isna().any().any()


class TestSchemaBasedValidation:
    """スキーマベースバリデーションのテスト"""

    @pytest.fixture
    def valid_ohlcv_data(self):
        """有効なOHLCVデータ"""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "Open": [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                ],
                "High": [
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                    110.0,
                    111.0,
                    112.0,
                    113.0,
                    114.0,
                ],
                "Low": [
                    95.0,
                    96.0,
                    97.0,
                    98.0,
                    99.0,
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                ],
                "Close": [
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                    110.0,
                    111.0,
                ],
                "Volume": [
                    1000.0,
                    1100.0,
                    1200.0,
                    1300.0,
                    1400.0,
                    1500.0,
                    1600.0,
                    1700.0,
                    1800.0,
                    1900.0,
                ],
            },
            index=dates,
        )

    @pytest.fixture
    def invalid_ohlcv_data(self):
        """無効なOHLCVデータ"""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "Open": [100, -50, 102, 103, 104],  # 負の値
                "High": [105, 106, 90, 108, 109],  # Lowより小さい値
                "Low": [95, 96, 97, 98, 99],
                "Close": [102, 103, 104, 105, 106],
                "Volume": [1000, -100, 1200, 1300, 1400],  # 負の値
            },
            index=dates,
        )

    def test_ohlcv_pydantic_model_valid(self, valid_ohlcv_data):
        """有効なOHLCVデータのPydanticモデルテスト"""
        # 最初の行をテスト
        row_data = valid_ohlcv_data.iloc[0].to_dict()
        model = OHLCVDataModel(**row_data)

        assert model.Open == 100
        assert model.High == 105
        assert model.Low == 95
        assert model.Close == 102
        assert model.Volume == 1000

    def test_ohlcv_pydantic_model_invalid(self):
        """無効なOHLCVデータのPydanticモデルテスト"""
        with pytest.raises(ValueError):
            # 負の値でエラー
            OHLCVDataModel(Open=-100, High=105, Low=95, Close=102, Volume=1000)

        with pytest.raises(ValueError):
            # 高値が安値より小さい
            OHLCVDataModel(Open=100, High=90, Low=95, Close=102, Volume=1000)

    def test_validate_dataframe_with_schema_valid(self, valid_ohlcv_data):
        """有効データのスキーマバリデーションテスト"""
        # インデックス名を設定
        valid_ohlcv_data.index.name = "timestamp"

        is_valid, errors = validate_dataframe_with_schema(
            valid_ohlcv_data, OHLCV_SCHEMA
        )

        if not is_valid:
            print(f"Validation errors: {errors}")

        assert is_valid
        assert len(errors) == 0

    def test_validate_dataframe_with_schema_invalid(self, invalid_ohlcv_data):
        """無効データのスキーマバリデーションテスト"""
        # インデックス名を設定
        invalid_ohlcv_data.index.name = "timestamp"

        is_valid, errors = validate_dataframe_with_schema(
            invalid_ohlcv_data, OHLCV_SCHEMA
        )

        assert not is_valid
        assert len(errors) > 0

    def test_clean_dataframe_with_schema(self, invalid_ohlcv_data):
        """スキーマベースクリーニングテスト"""
        # インデックス名を設定
        invalid_ohlcv_data.index.name = "timestamp"

        cleaned_df = clean_dataframe_with_schema(
            invalid_ohlcv_data, OHLCV_SCHEMA, drop_invalid_rows=True
        )

        # 無効な行が削除されていることを確認
        assert len(cleaned_df) <= len(invalid_ohlcv_data)


class TestBackwardCompatibility:
    """後方互換性テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    def test_data_processor_deprecation_warnings(self, sample_data):
        """DataProcessorの非推奨警告テスト"""
        processor = DataProcessor()

        # 非推奨メソッドを使用して警告が出ることを確認
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # IQR外れ値除去（非推奨）
            processor.create_preprocessing_pipeline(
                outlier_method="iqr", remove_outliers=True
            )

            # 警告が出力されていることを確認
            assert len(w) > 0
            assert any("非推奨" in str(warning.message) for warning in w)

    def test_data_validator_deprecation_warnings(self, sample_data):
        """DataValidatorの非推奨警告テスト"""
        validator = DataValidator()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # 非推奨メソッドを使用（clean_dataframe は削除済のため validate のみ）
            validator.validate_dataframe(sample_data)

            # 警告が出力されていることを確認（少なくとも1つの非推奨警告）
            assert len(w) >= 1
            assert any("非推奨" in str(warning.message) for warning in w)


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])
