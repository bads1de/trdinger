"""
前処理パイプライン モジュール

このモジュールは、データクリーニング、外れ値除去、補間、カテゴリエンコーディング、
dtype最適化のための複数のトランスフォーマーを組み合わせた包括的な前処理パイプラインを提供します。

パイプラインはscikit-learnの慣例に従い、MLワークフローで使用できます。
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# パイプラインのための簡略化されたトランスフォーマー
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest


class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    """IsolationForestを使用した外れ値除去トランスフォーマー。"""

    def __init__(self, method="isolation_forest", contamination=0.1, **kwargs):
        self.method = method
        self.contamination = contamination
        self.detector = None
        # Ignore extra kwargs to maintain compatibility

    def fit(self, X, y=None):
        if self.method == "isolation_forest":
            self.detector = IsolationForest(
                contamination=self.contamination, random_state=42
            )
            self.detector.fit(X)
        return self

    def transform(self, X):
        if self.detector is None:
            return X
        predictions = self.detector.predict(X)
        outlier_mask = predictions == -1  # -1 indicates outliers
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            # Replace outliers with median values instead of NaN
            for col in X_transformed.columns:
                if outlier_mask.any():
                    col_median = X_transformed[col].median()
                    X_transformed.loc[outlier_mask, col] = col_median
            return X_transformed
        else:
            X_transformed = X.copy()
            # For numpy arrays, replace with column medians
            for i in range(X_transformed.shape[1]):
                if outlier_mask.any():
                    col_median = np.nanmedian(X_transformed[:, i])
                    X_transformed[outlier_mask, i] = col_median
            return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        return input_features


class CategoricalEncoderTransformer(BaseEstimator, TransformerMixin):
    """カテゴリエンコーダートランスフォーマー。"""

    def __init__(self, encoding_type="label", **kwargs):
        self.encoding_type = encoding_type
        self.encoders_ = {}
        # 互換性を維持するために余分なkwargsを無視

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            categorical_columns = X.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_columns:
                encoder = LabelEncoder()
                encoder.fit(X[col].fillna("Unknown"))
                self.encoders_[col] = encoder
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_encoded = X.copy()
            for col, encoder in self.encoders_.items():
                if col in X_encoded.columns:
                    # NaNを"Unknown"で埋め、変換
                    filled_col = X_encoded[col].fillna("Unknown")
                    X_encoded[col] = encoder.transform(filled_col)
            return X_encoded
        return X

    def get_feature_names_out(self, input_features=None):
        """変換の出力特徴名を取得。"""
        return input_features


class DtypeOptimizerTransformer(BaseEstimator, TransformerMixin):
    """dtype最適化のためのトランスフォーマー。"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return self._optimize_dtypes(X)
        return X

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """メモリ効率のためにデータ型を最適化。"""
        result_df = df.copy()
        for col in result_df.columns:
            if result_df[col].dtype == "object":
                continue
            if pd.api.types.is_numeric_dtype(result_df[col]):
                if pd.api.types.is_integer_dtype(result_df[col]):
                    col_min = result_df[col].min()
                    col_max = result_df[col].max()
                    if col_min >= 0:
                        if col_max < 255:
                            result_df[col] = result_df[col].astype("uint8")
                        elif col_max < 65535:
                            result_df[col] = result_df[col].astype("uint16")
                        elif col_max < 4294967295:
                            result_df[col] = result_df[col].astype("uint32")
                    else:
                        if col_min > -128 and col_max < 127:
                            result_df[col] = result_df[col].astype("int8")
                        elif col_min > -32768 and col_max < 32767:
                            result_df[col] = result_df[col].astype("int16")
                        elif col_min > -2147483648 and col_max < 2147483647:
                            result_df[col] = result_df[col].astype("int32")
                elif pd.api.types.is_float_dtype(result_df[col]):
                    result_df[col] = pd.to_numeric(result_df[col], downcast="float")
        return result_df

    def get_feature_names_out(self, input_features=None):
        """変換の出力特徴名を取得。"""
        return input_features


logger = logging.getLogger(__name__)


class CategoricalPipelineTransformer(BaseEstimator, TransformerMixin):
    """DataFrameを維持しながらカテゴリ前処理を扱うカスタムトランスフォーマー。"""

    def __init__(
        self,
        strategy="most_frequent",
        fill_value="Unknown",
        encoding=True,
        categorical_encoding="label",
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        self.encoding = encoding
        self.categorical_encoding = categorical_encoding
        self.imputer = None
        self.encoder = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            # Fit imputer
            self.imputer = SimpleImputer(
                strategy=self.strategy, fill_value=self.fill_value
            )
            self.imputer.fit(X)

            # Fit encoder if enabled
            if self.encoding:
                self.encoder = CategoricalEncoderTransformer(
                    encoding_type=self.categorical_encoding
                )
                self.encoder.fit(X)

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = X.copy()

            # Apply imputation
            if self.imputer is not None:
                # Impute column by column to maintain DataFrame
                for col in result.columns:
                    if result[col].isnull().any():
                        col_data = result[col].values.reshape(-1, 1)
                        imputed = self.imputer.fit_transform(col_data)
                        result[col] = imputed.flatten()

            # Apply encoding
            if self.encoder is not None:
                result = self.encoder.transform(result)

            return result
        return X

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return input_features


def create_preprocessing_pipeline(
    outlier_method: str = "isolation_forest",
    outlier_contamination: float = 0.1,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    categorical_fill_value: str = "Unknown",
    categorical_encoding: str = "label",
    optimize_dtypes: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """
    包括的な前処理パイプラインを作成。

    このパイプラインには以下が含まれます：
    - 数値列の外れ値除去
    - 欠損値補間
    - カテゴリ変数エンコーディング
    - データ型最適化

    Args:
        outlier_method: 外れ値検出の方法 ('isolation_forest', 'local_outlier_factor')
        outlier_contamination: 外れ値の予想割合
        numeric_strategy: 数値列の補間戦略
        categorical_strategy: カテゴリ列の補間戦略
        categorical_fill_value: カテゴリ欠損値の埋め値
        categorical_encoding: カテゴリ変数のエンコーディング方法 ('label', 'onehot')
        optimize_dtypes: データタイプを最適化するかどうか
        **kwargs: トランスフォーマーの追加パラメータ

    Returns:
        設定されたsklearnパイプライン
    """
    logger.info("前処理パイプラインを作成中...")

    # Numerical preprocessing pipeline
    numeric_steps = []

    # 1. Imputation for numerical columns (first, before outlier removal)
    numeric_steps.append(("numeric_imputer", SimpleImputer(strategy=numeric_strategy)))

    # 2. Outlier removal (after imputation)
    if outlier_method:
        numeric_steps.append(
            (
                "outlier_removal",
                OutlierRemovalTransformer(
                    method=outlier_method, contamination=outlier_contamination, **kwargs
                ),
            )
        )

    numeric_pipeline = Pipeline(numeric_steps)

    # Categorical preprocessing pipeline
    categorical_pipeline = CategoricalPipelineTransformer(
        strategy=categorical_strategy,
        fill_value=categorical_fill_value,
        encoding=True,
        categorical_encoding=categorical_encoding,
    )

    # Apply transformations directly to handle mixed data types properly
    class MixedTypeTransformer(BaseEstimator, TransformerMixin):
        """Transformer that handles mixed data types properly."""

        def __init__(self, numeric_pipeline, categorical_pipeline):
            self.numeric_pipeline = numeric_pipeline
            self.categorical_pipeline = categorical_pipeline

        def fit(self, X, y=None):
            if isinstance(X, pd.DataFrame):
                # Separate numeric and categorical columns
                self.numeric_columns_ = X.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                self.categorical_columns_ = X.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                # Fit pipelines on respective columns
                if self.numeric_columns_:
                    self.numeric_pipeline.fit(X[self.numeric_columns_], y)
                if self.categorical_columns_:
                    self.categorical_pipeline.fit(X[self.categorical_columns_], y)
            else:
                # For numpy arrays, assume all columns are numeric
                self.numeric_columns_ = None
                self.categorical_columns_ = None
                self.numeric_pipeline.fit(X, y)

            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                result_parts = []

                # Transform numeric columns
                if self.numeric_columns_:
                    numeric_transformed = self.numeric_pipeline.transform(
                        X[self.numeric_columns_]
                    )
                    if isinstance(numeric_transformed, np.ndarray):
                        # Handle NaN values before creating DataFrame
                        if np.isnan(numeric_transformed).all():
                            # If all values are NaN, fill with zeros
                            numeric_transformed = np.zeros_like(numeric_transformed)
                        
                        # Ensure columns match and handle empty columns
                        available_columns = [col for col in self.numeric_columns_ if col in X.columns]
                        if available_columns:
                            numeric_df = pd.DataFrame(
                                numeric_transformed,
                                columns=available_columns,
                                index=X.index,
                            )
                        else:
                            numeric_df = pd.DataFrame(index=X.index)
                        
                        if numeric_df.empty and len(result_parts) == 0:
                            # If no numeric columns, return empty DataFrame with proper index
                            numeric_df = pd.DataFrame(index=X.index)
                            numeric_df = numeric_df.add_prefix('numeric_')
                        result_parts.append(numeric_df)
                    else:
                        numeric_df = numeric_transformed
                        result_parts.append(numeric_df)

                # Transform categorical columns
                if self.categorical_columns_:
                    categorical_transformed = self.categorical_pipeline.transform(
                        X[self.categorical_columns_]
                    )
                    if isinstance(categorical_transformed, np.ndarray):
                        # Handle NaN values in categorical data
                        if np.isnan(categorical_transformed).any():
                            # Fill NaN with 0 for categorical data
                            categorical_transformed = np.nan_to_num(categorical_transformed, nan=0.0)
                        categorical_df = pd.DataFrame(
                            categorical_transformed,
                            columns=self.categorical_columns_,
                            index=X.index,
                        )
                    else:
                        categorical_df = categorical_transformed
                    result_parts.append(categorical_df)

                # Combine results
                if len(result_parts) == 0:
                    # If no columns were transformed, return original DataFrame
                    return X
                elif len(result_parts) == 1:
                    return result_parts[0]
                else:
                    return pd.concat(result_parts, axis=1)
            else:
                # For numpy arrays
                return self.numeric_pipeline.transform(X)

        def get_feature_names_out(self, input_features=None):
            """Get output feature names."""
            feature_names = []
            if self.numeric_columns_:
                feature_names.extend(self.numeric_columns_)
            if self.categorical_columns_:
                feature_names.extend(self.categorical_columns_)
            return feature_names

    # Create mixed type transformer
    preprocessor = MixedTypeTransformer(numeric_pipeline, categorical_pipeline)

    # Main pipeline
    steps = [("preprocessor", preprocessor)]

    # Optional dtype optimization
    if optimize_dtypes:
        steps.append(("dtype_optimizer", DtypeOptimizerTransformer()))

    pipeline = Pipeline(steps)

    # Add DataFrame converter after pipeline creation
    final_steps = list(pipeline.steps)

    def to_dataframe_func(X):
        """Convert to DataFrame with proper column names."""
        try:
            columns = pipeline.get_feature_names_out()
            return pd.DataFrame(X, columns=columns)
        except Exception:
            # Fallback to generic column names
            if hasattr(X, "shape") and len(X.shape) > 1:
                columns = [f"feature_{i}" for i in range(X.shape[1])]
                return pd.DataFrame(X, columns=columns)
            return pd.DataFrame(X)

    def get_feature_names_out_func(self, input_features=None):
        """Get feature names for the output."""
        try:
            return pipeline.get_feature_names_out()
        except Exception:
            if input_features is not None:
                return input_features
            return ["feature"]

    final_steps.append(
        (
            "to_dataframe",
            FunctionTransformer(
                func=to_dataframe_func,
                validate=False,
                feature_names_out=get_feature_names_out_func,
            ),
        )
    )

    pipeline = Pipeline(final_steps)

    logger.info("Preprocessing pipeline created successfully")
    return pipeline


def create_preprocessing_pipeline(
    outlier_method: str = "isolation_forest",
    outlier_contamination: float = 0.1,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    categorical_fill_value: str = "Unknown",
    categorical_encoding: str = "label",
    optimize_dtypes: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """
    包括的な前処理パイプラインを作成。

    このパイプラインには以下が含まれます：
    - 数値列の外れ値除去
    - 欠損値補間
    - カテゴリ変数エンコーディング
    - データ型最適化

    Args:
        outlier_method: 外れ値検出の方法 ('isolation_forest', 'local_outlier_factor')
        outlier_contamination: 外れ値の予想割合
        numeric_strategy: 数値列の補間戦略
        categorical_strategy: カテゴリ列の補間戦略
        categorical_fill_value: カテゴリ欠損値の埋め値
        categorical_encoding: カテゴリ変数のエンコーディング方法 ('label', 'onehot')
        optimize_dtypes: データタイプを最適化するかどうか
        **kwargs: トランスフォーマーの追加パラメータ

    Returns:
        設定されたsklearnパイプライン
    """
    logger.info("前処理パイプラインを作成中...")

    # Numerical preprocessing pipeline
    numeric_steps = []

    # 1. Imputation for numerical columns (first, before outlier removal)
    numeric_steps.append(("numeric_imputer", SimpleImputer(strategy=numeric_strategy)))

    # 2. Outlier removal (after imputation)
    if outlier_method:
        numeric_steps.append(
            (
                "outlier_removal",
                OutlierRemovalTransformer(
                    method=outlier_method, contamination=outlier_contamination, **kwargs
                ),
            )
        )

    numeric_pipeline = Pipeline(numeric_steps)

    # Categorical preprocessing pipeline
    class CategoricalPipelineTransformer(BaseEstimator, TransformerMixin):
        """Custom transformer that handles categorical preprocessing while maintaining DataFrame."""

        def __init__(
            self,
            strategy="most_frequent",
            fill_value="Unknown",
            encoding=True,
            categorical_encoding="label",
        ):
            self.strategy = strategy
            self.fill_value = fill_value
            self.encoding = encoding
            self.categorical_encoding = categorical_encoding
            self.imputer = None
            self.encoder = None

        def fit(self, X, y=None):
            if isinstance(X, pd.DataFrame):
                # Fit imputer
                self.imputer = SimpleImputer(
                    strategy=self.strategy, fill_value=self.fill_value
                )
                self.imputer.fit(X)

                # Fit encoder if enabled
                if self.encoding:
                    self.encoder = CategoricalEncoderTransformer(
                        encoding_type=self.categorical_encoding
                    )
                    self.encoder.fit(X)

            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                result = X.copy()

                # Apply imputation
                if self.imputer is not None:
                    # Impute column by column to maintain DataFrame
                    for col in result.columns:
                        if result[col].isnull().any():
                            # Check for all NaN values in the column
                            if result[col].isnull().all():
                                # Handle all-NaN column by filling with constant value
                                constant_value = 0.0 if self.strategy in ['mean', 'median'] else 'Unknown'
                                result[col] = result[col].fillna(constant_value)
                            else:
                                # Normal imputation for columns with some valid values
                                col_data = result[col].values.reshape(-1, 1)
                                imputed = self.imputer.fit_transform(col_data)
                                result[col] = imputed.flatten()

                # Apply encoding
                if self.encoder is not None:
                    result = self.encoder.transform(result)

                return result
            return X

        def get_feature_names_out(self, input_features=None):
            """Get output feature names."""
            return input_features

    categorical_pipeline = CategoricalPipelineTransformer(
        strategy=categorical_strategy,
        fill_value=categorical_fill_value,
        encoding=categorical_encoding,
    )

    # Apply transformations directly to handle mixed data types properly
    class MixedTypeTransformer(BaseEstimator, TransformerMixin):
        """Transformer that handles mixed data types properly."""

        def __init__(self, numeric_pipeline, categorical_pipeline):
            self.numeric_pipeline = numeric_pipeline
            self.categorical_pipeline = categorical_pipeline

        def fit(self, X, y=None):
            if isinstance(X, pd.DataFrame):
                # Separate numeric and categorical columns
                self.numeric_columns_ = X.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                self.categorical_columns_ = X.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                # Fit pipelines on respective columns
                if self.numeric_columns_:
                    self.numeric_pipeline.fit(X[self.numeric_columns_], y)
                if self.categorical_columns_:
                    self.categorical_pipeline.fit(X[self.categorical_columns_], y)
            else:
                # For numpy arrays, assume all columns are numeric
                self.numeric_columns_ = None
                self.categorical_columns_ = None
                self.numeric_pipeline.fit(X, y)

            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                result_parts = []

                # Transform numeric columns
                if self.numeric_columns_:
                    numeric_transformed = self.numeric_pipeline.transform(
                        X[self.numeric_columns_]
                    )
                    if isinstance(numeric_transformed, np.ndarray):
                        # Handle NaN values before creating DataFrame
                        if np.isnan(numeric_transformed).all():
                            # If all values are NaN, fill with zeros
                            numeric_transformed = np.zeros_like(numeric_transformed)
                        
                        # Ensure columns match and handle empty columns
                        available_columns = [col for col in self.numeric_columns_ if col in X.columns]
                        if available_columns:
                            numeric_df = pd.DataFrame(
                                numeric_transformed,
                                columns=available_columns,
                                index=X.index,
                            )
                        else:
                            numeric_df = pd.DataFrame(index=X.index)
                        
                        if numeric_df.empty and len(result_parts) == 0:
                            # If no numeric columns, return empty DataFrame with proper index
                            numeric_df = pd.DataFrame(index=X.index)
                            numeric_df = numeric_df.add_prefix('numeric_')
                        result_parts.append(numeric_df)
                    else:
                        numeric_df = numeric_transformed
                        result_parts.append(numeric_df)

                # Transform categorical columns
                if self.categorical_columns_:
                    categorical_transformed = self.categorical_pipeline.transform(
                        X[self.categorical_columns_]
                    )
                    if isinstance(categorical_transformed, np.ndarray):
                        # Handle NaN values in categorical data
                        if np.isnan(categorical_transformed).any():
                            # Fill NaN with 0 for categorical data
                            categorical_transformed = np.nan_to_num(categorical_transformed, nan=0.0)
                        categorical_df = pd.DataFrame(
                            categorical_transformed,
                            columns=self.categorical_columns_,
                            index=X.index,
                        )
                    else:
                        categorical_df = categorical_transformed
                    result_parts.append(categorical_df)

                # Combine results
                if len(result_parts) == 0:
                    # If no columns were transformed, return original DataFrame
                    return X
                elif len(result_parts) == 1:
                    return result_parts[0]
                else:
                    return pd.concat(result_parts, axis=1)
            else:
                # For numpy arrays
                return self.numeric_pipeline.transform(X)

        def get_feature_names_out(self, input_features=None):
            """Get output feature names."""
            feature_names = []
            if self.numeric_columns_:
                feature_names.extend(self.numeric_columns_)
            if self.categorical_columns_:
                feature_names.extend(self.categorical_columns_)
            return feature_names

    # Create mixed type transformer
    preprocessor = MixedTypeTransformer(numeric_pipeline, categorical_pipeline)

    # Main pipeline
    steps = [("preprocessor", preprocessor)]

    # Optional dtype optimization
    if optimize_dtypes:
        steps.append(("dtype_optimizer", DtypeOptimizerTransformer()))

    pipeline = Pipeline(steps)

    # Add DataFrame converter after pipeline creation
    final_steps = list(pipeline.steps)

    def to_dataframe_func(X):
        """Convert to DataFrame with proper column names."""
        try:
            columns = pipeline.get_feature_names_out()
            return pd.DataFrame(X, columns=columns)
        except Exception:
            # Fallback to generic column names
            if hasattr(X, "shape") and len(X.shape) > 1:
                columns = [f"feature_{i}" for i in range(X.shape[1])]
                return pd.DataFrame(X, columns=columns)
            return pd.DataFrame(X)

    def get_feature_names_out_func(self, input_features=None):
        """Get feature names for the output."""
        try:
            return pipeline.get_feature_names_out()
        except Exception:
            if input_features is not None:
                return input_features
            return ["feature"]

    final_steps.append(
        (
            "to_dataframe",
            FunctionTransformer(
                func=to_dataframe_func,
                validate=False,
                feature_names_out=get_feature_names_out_func,
            ),
        )
    )

    pipeline = Pipeline(final_steps)

    logger.info("Preprocessing pipeline created successfully")
    return pipeline


def create_basic_preprocessing_pipeline(
    impute_strategy: str = "median", encode_categorical: bool = True
) -> Pipeline:
    """
    外れ値除去なしの基本的な前処理パイプラインを作成。

    Args:
        impute_strategy: 欠損値の補間戦略
        encode_categorical: カテゴリ変数をエンコードするかどうか

    Returns:
        基本的な前処理パイプライン
    """
    logger.info("基本的な前処理パイプラインを作成中...")

    # Simple imputation pipeline
    imputer = SimpleImputer(strategy=impute_strategy)

    steps = [("imputer", imputer)]

    if encode_categorical:
        steps.append(("encoder", CategoricalEncoderTransformer(encoding_type="label")))

    pipeline = Pipeline(steps)
    logger.info("Basic preprocessing pipeline created")
    return pipeline


def get_pipeline_info(pipeline: Pipeline) -> Dict[str, Any]:
    """
    適合済みのパイプラインの情報を取得。

    Args:
        pipeline: 適合済みのsklearnパイプライン

    Returns:
        パイプライン情報を含む辞書
    """
    info = {
        "n_steps": len(pipeline.steps),
        "step_names": [step[0] for step in pipeline.steps],
        "is_fitted": hasattr(pipeline, "feature_names_in_"),
    }

    if hasattr(pipeline, "feature_names_in_"):
        info["n_features_in"] = len(pipeline.feature_names_in_)
        info["feature_names_in"] = pipeline.feature_names_in_.tolist()

    try:
        if hasattr(pipeline, "get_feature_names_out"):
            feature_names_out = pipeline.get_feature_names_out()
            info["n_features_out"] = len(feature_names_out)
            info["feature_names_out"] = feature_names_out.tolist()
    except Exception:
        info["feature_names_out"] = None

    return info
